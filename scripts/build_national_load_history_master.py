#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_national_load_history_master.py

Build a clean Dutch national hourly load master history from two source files:

1) time_series_2006_2018.csv
2) time_series_profiles_2015_2020.csv

Agreed source-priority rule
---------------------------
For 2015-2020:
    prefer NL_load_actual_entsoe_transparency
    else   NL_load_forecast_entsoe_transparency

For 2006-2014:
    prefer NL_load_entsoe_transparency
    else   NL_load_entsoe_power_statistics
    else   interpolated_values

For overlap years 2015-2018:
    use the profile file as primary
    keep the older file for overlap diagnostics / validation
    allow the older file only as emergency overlap fallback before interpolation

Output
------
Primary output:
    derived/national_hourly_load_history.csv

Additional outputs:
    derived/national_hourly_load_overlap_diagnostics_2015_2018.csv
    derived/national_hourly_load_history_qc_by_year.csv
    derived/national_hourly_load_history_build_meta.json

Required master columns:
    timestamp_utc
    year
    hour_of_year
    nat_load_MW

Additional audit columns are preserved for publication-quality traceability.

Important design choices
------------------------
- Work strictly in UTC.
- Deduplicate timestamps before merging.
- Build complete UTC hourly year-panels.
- Interpolate only small internal gaps after source-priority selection.
- Drop years with excessive unresolved missingness.
- Fix the historical window explicitly to 2006-2020.

Why fix 2006-2020 explicitly?
-----------------------------
Both source files contain edge timestamps such as:
    2005-12-31T23:00:00Z  (for local Dutch 2006-01-01 00:00)
    2014-12-31T23:00:00Z  (for local Dutch 2015-01-01 00:00)
If the year window is inferred mechanically from UTC timestamps, the build may
accidentally include a useless partial year 2005. That is structurally wrong for
this master artifact, so we pin the intended Dutch historical window directly.
"""

from __future__ import annotations
from pathlib import Path
import json
import sys
from typing import Iterable, List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pipeline_config as cfg  # type: ignore
except Exception as exc:
    raise RuntimeError(
        f"Could not import pipeline_config.py from {REPO_ROOT}. "
        "Run this script from the final repo structure."
    ) from exc

# =============================================================================
# USER CONFIG
# =============================================================================

# =============================================================================
# USER CONFIG
# =============================================================================

OLD_FILE = getattr(
    cfg,
    "NATIONAL_LOAD_OLD_FILE_PATH",
    cfg.EXTERNAL_DIR / "time_series_2006_2018.csv",
)
PROFILE_FILE = getattr(
    cfg,
    "NATIONAL_LOAD_PROFILE_FILE_PATH",
    cfg.EXTERNAL_DIR / "time_series_profiles_2015_2020.csv",
)

OUT_MASTER = getattr(
    cfg,
    "NATIONAL_LOAD_HISTORY_PATH",
    cfg.DERIVED_DIR / "national_hourly_load_history.csv",
)
OUT_OVERLAP = getattr(
    cfg,
    "NATIONAL_LOAD_OVERLAP_DIAG_PATH",
    cfg.DERIVED_DIR / "national_hourly_load_overlap_diagnostics_2015_2018.csv",
)
OUT_QC = getattr(
    cfg,
    "NATIONAL_LOAD_HISTORY_QC_PATH",
    cfg.DERIVED_DIR / "national_hourly_load_history_qc_by_year.csv",
)
OUT_META = getattr(
    cfg,
    "NATIONAL_LOAD_HISTORY_META_PATH",
    cfg.DERIVED_DIR / "national_hourly_load_history_build_meta.json",
)
# Historical window agreed for the national master artifact.
MIN_YEAR = 2006
MAX_YEAR = 2020

# Only interpolate short internal gaps after source selection.
MAX_INTERP_GAP_HOURS = 6

# Year-level quality thresholds.
MAX_PRE_INTERP_MISSING_SHARE = 0.02   # >2% missing before small-gap interpolation => drop year
MAX_POST_INTERP_MISSING_SHARE = 0.001 # >0.1% unresolved after interpolation => drop year


# =============================================================================
# HELPERS
# =============================================================================

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    # sep=None with python engine is safer when users open/save in Excel and separators vary.
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def first_valid(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if not s.empty else np.nan


def parse_utc_timestamp(df: pd.DataFrame, label: str) -> pd.Series:
    if "utc_timestamp" not in df.columns:
        raise KeyError(f"{label}: required column 'utc_timestamp' not found")
    ts = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError(f"{label}: failed to parse all utc_timestamp values")
    return ts.dt.floor("h")


def normalize_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def deduplicate_hourly(df: pd.DataFrame, value_cols: List[str], label: str) -> pd.DataFrame:
    """
    Collapse duplicate UTC timestamps by taking the first non-null per value column.
    """
    if "timestamp_utc" not in df.columns:
        raise KeyError(f"{label}: missing timestamp_utc before deduplication")

    n_before = len(df)
    dup_rows = int(df["timestamp_utc"].duplicated(keep=False).sum())

    agg = {c: first_valid for c in value_cols if c in df.columns}
    out = (
        df.groupby("timestamp_utc", as_index=False)
          .agg(agg)
          .sort_values("timestamp_utc")
          .reset_index(drop=True)
    )

    print(
        f"[INFO] {label}: rows_before={n_before:,} | "
        f"duplicate_timestamp_rows={dup_rows:,} | rows_after_dedup={len(out):,}"
    )
    return out


def build_hourly_year_panel(year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz="UTC")
    ts = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    out = pd.DataFrame({"timestamp_utc": ts})
    out["year"] = year
    out["hour_of_year"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def choose_old_source(row: pd.Series):
    if pd.notna(row.get("load_old_transparency_MW")):
        return row["load_old_transparency_MW"], "old_transparency"
    if pd.notna(row.get("load_old_power_statistics_MW")):
        return row["load_old_power_statistics_MW"], "old_power_statistics"
    if pd.notna(row.get("load_old_interpolated_MW")):
        return row["load_old_interpolated_MW"], "old_interpolated"
    return np.nan, "missing"


def choose_profile_source(row: pd.Series):
    if pd.notna(row.get("load_profile_actual_MW")):
        return row["load_profile_actual_MW"], "profile_actual"
    if pd.notna(row.get("load_profile_forecast_MW")):
        return row["load_profile_forecast_MW"], "profile_forecast"
    return np.nan, "missing"


def choose_final_source(row: pd.Series):
    year = int(row["year"])

    if 2015 <= year <= 2020:
        if pd.notna(row.get("load_profile_primary_MW")):
            return row["load_profile_primary_MW"], row["load_profile_primary_source"]
        if year <= 2018 and pd.notna(row.get("load_old_primary_MW")):
            return row["load_old_primary_MW"], f"{row['load_old_primary_source']}_fallback_in_overlap"
        return np.nan, "missing"

    if 2006 <= year <= 2014:
        if pd.notna(row.get("load_old_primary_MW")):
            return row["load_old_primary_MW"], row["load_old_primary_source"]
        return np.nan, "missing"

    return np.nan, "outside_target_window"


def interpolate_small_gaps_with_flags(s: pd.Series, limit: int):
    """
    Linear interpolation for short internal gaps only.
    Returns:
        filled_series,
        was_interpolated_flag (0/1)
    """
    original_missing = s.isna()
    filled = s.interpolate(method="linear", limit=limit, limit_area="inside")
    was_interpolated = (original_missing & filled.notna()).astype(int)
    return filled, was_interpolated


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return np.nan
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return np.nan
    return a.corr(b)


def compute_overlap_diagnostics(df: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        g = df.loc[df["year"] == year, ["load_profile_primary_MW", "load_old_primary_MW"]].copy()
        g = g.dropna()

        if g.empty:
            rows.append({
                "year": int(year),
                "n_overlap_hours": 0,
                "mae_MW": np.nan,
                "rmse_MW": np.nan,
                "mbe_MW": np.nan,
                "mape_pct": np.nan,
                "corr": np.nan,
                "max_abs_diff_MW": np.nan,
            })
            continue

        diff = g["load_profile_primary_MW"] - g["load_old_primary_MW"]
        denom = g["load_old_primary_MW"].replace(0, np.nan)

        rows.append({
            "year": int(year),
            "n_overlap_hours": int(len(g)),
            "mae_MW": float(np.mean(np.abs(diff))),
            "rmse_MW": float(np.sqrt(np.mean(diff ** 2))),
            "mbe_MW": float(np.mean(diff)),
            "mape_pct": float(np.nanmean(np.abs(diff) / denom) * 100.0),
            "corr": float(safe_corr(g["load_profile_primary_MW"], g["load_old_primary_MW"])),
            "max_abs_diff_MW": float(np.max(np.abs(diff))),
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    for p in [OUT_MASTER, OUT_OVERLAP, OUT_QC, OUT_META]:
        ensure_parent(p)

    print("[INFO] Reading source files...")
    old_raw = read_csv_auto(OLD_FILE)
    prof_raw = read_csv_auto(PROFILE_FILE)

    old_keep = [
        "utc_timestamp",
        "interpolated_values",
        "NL_load_entsoe_power_statistics",
        "NL_load_entsoe_transparency",
    ]
    prof_keep = [
        "utc_timestamp",
        "NL_load_actual_entsoe_transparency",
        "NL_load_forecast_entsoe_transparency",
    ]

    missing_old = [c for c in old_keep if c not in old_raw.columns]
    missing_prof = [c for c in prof_keep if c not in prof_raw.columns]
    if missing_old:
        raise KeyError(f"OLD_FILE missing required columns: {missing_old}")
    if missing_prof:
        raise KeyError(f"PROFILE_FILE missing required columns: {missing_prof}")

    old = old_raw[old_keep].copy()
    prof = prof_raw[prof_keep].copy()

    old["timestamp_utc"] = parse_utc_timestamp(old, "OLD_FILE")
    prof["timestamp_utc"] = parse_utc_timestamp(prof, "PROFILE_FILE")

    old = normalize_numeric(old, old_keep[1:])
    prof = normalize_numeric(prof, prof_keep[1:])

    old = deduplicate_hourly(old, old_keep[1:], "OLD_FILE")
    prof = deduplicate_hourly(prof, prof_keep[1:], "PROFILE_FILE")

    old = old.rename(columns={
        "NL_load_entsoe_transparency": "load_old_transparency_MW",
        "NL_load_entsoe_power_statistics": "load_old_power_statistics_MW",
        "interpolated_values": "load_old_interpolated_MW",
    })
    prof = prof.rename(columns={
        "NL_load_actual_entsoe_transparency": "load_profile_actual_MW",
        "NL_load_forecast_entsoe_transparency": "load_profile_forecast_MW",
    })

    print(f"[INFO] Building full UTC master panel for years {MIN_YEAR}-{MAX_YEAR}...")
    years = list(range(MIN_YEAR, MAX_YEAR + 1))
    master = pd.concat([build_hourly_year_panel(y) for y in years], ignore_index=True)

    master = master.merge(old, on="timestamp_utc", how="left")
    master = master.merge(prof, on="timestamp_utc", how="left")

    old_choice = master.apply(choose_old_source, axis=1, result_type="expand")
    old_choice.columns = ["load_old_primary_MW", "load_old_primary_source"]
    master = pd.concat([master, old_choice], axis=1)

    prof_choice = master.apply(choose_profile_source, axis=1, result_type="expand")
    prof_choice.columns = ["load_profile_primary_MW", "load_profile_primary_source"]
    master = pd.concat([master, prof_choice], axis=1)

    final_choice = master.apply(choose_final_source, axis=1, result_type="expand")
    final_choice.columns = ["nat_load_pre_interp_MW", "nat_load_source_pre_interp"]
    master = pd.concat([master, final_choice], axis=1)

    overlap_years = [2015, 2016, 2017, 2018]
    overlap_diag = compute_overlap_diagnostics(master, overlap_years)
    overlap_diag.to_csv(OUT_OVERLAP, index=False)
    print(f"[OK] Saved overlap diagnostics: {OUT_OVERLAP}")

    kept_parts = []
    qc_rows = []

    for year, g in master.groupby("year", sort=True):
        g = g.sort_values("timestamp_utc").copy()

        pre_missing_hours = int(g["nat_load_pre_interp_MW"].isna().sum())
        pre_missing_share = pre_missing_hours / len(g)

        g["nat_load_MW"], g["nat_load_was_interpolated"] = interpolate_small_gaps_with_flags(
            g["nat_load_pre_interp_MW"],
            limit=MAX_INTERP_GAP_HOURS,
        )

        g["nat_load_source"] = g["nat_load_source_pre_interp"]
        interp_mask = g["nat_load_was_interpolated"].eq(1)
        g.loc[interp_mask, "nat_load_source"] = (
            g.loc[interp_mask, "nat_load_source_pre_interp"].fillna("missing") + "_interp"
        )

        post_missing_hours = int(g["nat_load_MW"].isna().sum())
        post_missing_share = post_missing_hours / len(g)

        keep_year = (
            pre_missing_share <= MAX_PRE_INTERP_MISSING_SHARE
            and post_missing_share <= MAX_POST_INTERP_MISSING_SHARE
        )

        qc_rows.append({
            "year": int(year),
            "n_hours": int(len(g)),
            "pre_interp_missing_hours": pre_missing_hours,
            "pre_interp_missing_share": pre_missing_share,
            "post_interp_missing_hours": post_missing_hours,
            "post_interp_missing_share": post_missing_share,
            "kept": int(keep_year),
        })

        if keep_year:
            kept_parts.append(g)
        else:
            print(
                f"[WARN] Dropping year {year}: "
                f"pre_interp_missing_share={pre_missing_share:.4%}, "
                f"post_interp_missing_share={post_missing_share:.4%}"
            )

    if not kept_parts:
        raise RuntimeError("All years were dropped by QC thresholds. Inspect the raw files or relax thresholds carefully.")

    out = pd.concat(kept_parts, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)

    desired_cols = [
        "timestamp_utc",
        "year",
        "hour_of_year",
        "nat_load_MW",
        "nat_load_source",
        "nat_load_was_interpolated",
        "nat_load_pre_interp_MW",
        "nat_load_source_pre_interp",
        "load_profile_actual_MW",
        "load_profile_forecast_MW",
        "load_profile_primary_MW",
        "load_profile_primary_source",
        "load_old_transparency_MW",
        "load_old_power_statistics_MW",
        "load_old_interpolated_MW",
        "load_old_primary_MW",
        "load_old_primary_source",
    ]
    out = out[[c for c in desired_cols if c in out.columns]].copy()
    out.to_csv(OUT_MASTER, index=False)
    print(f"[OK] Saved master national load history: {OUT_MASTER}")

    qc = pd.DataFrame(qc_rows).sort_values("year").reset_index(drop=True)
    qc.to_csv(OUT_QC, index=False)
    print(f"[OK] Saved QC summary: {OUT_QC}")

    meta = {
        "old_file": str(OLD_FILE),
        "profile_file": str(PROFILE_FILE),
        "out_master": str(OUT_MASTER),
        "out_overlap": str(OUT_OVERLAP),
        "out_qc": str(OUT_QC),
        "historical_window": [MIN_YEAR, MAX_YEAR],
        "max_interp_gap_hours": MAX_INTERP_GAP_HOURS,
        "max_pre_interp_missing_share": MAX_PRE_INTERP_MISSING_SHARE,
        "max_post_interp_missing_share": MAX_POST_INTERP_MISSING_SHARE,
        "years_retained": qc.loc[qc["kept"] == 1, "year"].astype(int).tolist(),
        "years_dropped": qc.loc[qc["kept"] == 0, "year"].astype(int).tolist(),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved build metadata: {OUT_META}")

    print("\n=== BUILD SUMMARY ===")
    print(f"Years retained: {meta['years_retained']}")
    print(f"Years dropped : {meta['years_dropped']}")
    print(f"Rows written  : {len(out):,}")


if __name__ == "__main__":
    main()