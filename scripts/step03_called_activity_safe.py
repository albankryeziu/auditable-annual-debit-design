#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step03_called_activity_safe.py

Purpose
-------
Build a strict, exact-path GOPACS called-activity panel for diagnostics.

This script supports two exact modes:

A) PUBLIC_ACTIVITY mode
   Uses the exact configured public EUR and MWh files.

B) STEP02_LOCALIZED mode
   If one or both public files are missing, uses the exact configured
   step02 localized province-hour GOPACS panel instead.

Important design rule
---------------------
This script uses exact paths only, in this order:
    pipeline_config.py -> environment variable -> hard default

No recursive search is allowed.

Outputs
-------
Primary output:
    data/derived/gopacs_called_activity_hour.csv

QC / metadata outputs:
    qc/step03_called_activity_summary.json
    data/derived/gopacs_called_activity_hour_metadata.json

Main output columns
-------------------
timestamp_utc
operator
gopacs_eur
gopacs_mwh
gopacs_activation
source_granularity

Important scope note
--------------------
This is a diagnostic stage. It is NOT the main historical-overlap builder
for the paper. If the configured source-support panel extends beyond the
observed public GOPACS activity window, that extension is retained only as
diagnostic support padding and is explicitly recorded in metadata.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pipeline_config as cfg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        f"Could not import pipeline_config.py from {REPO_ROOT}. "
        "Run this script from the final repo structure."
    ) from exc


# =============================================================================
# PATH / COLUMN HELPERS
# =============================================================================
def _get_path(*names: str, env_name: Optional[str] = None, default: Optional[Path] = None) -> Path:
    for name in names:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if value:
                return Path(value)
    if env_name:
        env_value = os.environ.get(env_name)
        if env_value:
            return Path(env_value)
    if default is not None:
        return Path(default)
    raise AttributeError(f"Could not resolve any of path names {names!r}.")


def _require_existing_path(label: str, path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n"
            f"  {path}\n\n"
            f"Fix pipeline_config.py so this path points to the correct file."
        )
    print(f"{label}: using -> {path}")
    return path


def _pick_column(columns: Iterable[str], candidates: Iterable[str], *, required: bool = True) -> Optional[str]:
    low = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    if required:
        raise KeyError(f"Could not find any of columns {list(candidates)!r}. Found: {list(columns)!r}")
    return None


def _normalize_operator_value(x: object) -> str:
    s = "" if x is None else str(x).strip().lower()
    if not s:
        return "national"
    mapping = {
        "nl": "national",
        "nederland": "national",
        "netherlands": "national",
        "landelijk": "national",
        "national": "national",
        "total": "national",
        "all": "national",
    }
    return mapping.get(s, s)


def _detect_source_granularity(ts: pd.Series) -> str:
    ts = pd.to_datetime(ts, utc=True, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(ts) <= 1:
        return "unknown"

    delta_sec = ts.diff().dropna().dt.total_seconds()
    if delta_sec.empty:
        return "unknown"

    min_sec = float(delta_sec.min())
    if min_sec < 3600.0:
        return "subhourly"
    if abs(min_sec - 3600.0) < 1e-9:
        return "hourly"
    if min_sec < 86400.0:
        return "intraday_nonhourly"
    if abs(min_sec - 86400.0) < 1e-9:
        return "daily"
    if min_sec <= 31.0 * 86400.0:
        return "monthly_or_lower_frequency"
    return "low_frequency"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), newline="") as tmp:
        df.to_csv(tmp.name, index=False, encoding="utf-8")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


# =============================================================================
# DATA CONTAINERS
# =============================================================================
@dataclass(frozen=True)
class SourcePanelSummary:
    hourly_timestamps: pd.DataFrame
    source_path: Path
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    n_hours: int


# =============================================================================
# SOURCE PANEL SUPPORT WINDOW
# =============================================================================
def _load_source_hour_support(source_path: Path) -> SourcePanelSummary:
    df = pd.read_csv(source_path)
    df.columns = [str(c).strip() for c in df.columns]

    ts_col = _pick_column(
        df.columns,
        ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc", "timestamp"],
        required=False,
    )

    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if ts.isna().all():
            raise ValueError(f"[SOURCE_PANEL] Failed to parse any timestamps from {source_path}.")
        out = pd.DataFrame({"timestamp_utc": ts.dropna().dt.floor("h").drop_duplicates().sort_values()})
    else:
        year_col = _pick_column(df.columns, ["year"])
        hour_col = _pick_column(df.columns, ["hour", "hour_of_year"], required=False)
        if hour_col is None:
            raise ValueError(
                f"[SOURCE_PANEL] Need either a timestamp column or year + hour/hour_of_year in:\n"
                f"  {source_path}\n"
                f"Columns were: {list(df.columns)}"
            )

        tmp = df[[year_col, hour_col]].copy()
        tmp.columns = ["year", "hour_like"]
        tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
        tmp["hour_like"] = pd.to_numeric(tmp["hour_like"], errors="coerce")
        tmp = tmp.dropna().copy()
        tmp["year"] = tmp["year"].astype(int)
        tmp["hour_like"] = tmp["hour_like"].astype(int)

        pieces = []
        for y, g in tmp.groupby("year", sort=True):
            H = int(
                (
                    pd.Timestamp(f"{y+1}-01-01T00:00:00Z")
                    - pd.Timestamp(f"{y}-01-01T00:00:00Z")
                ) / pd.Timedelta(hours=1)
            )
            mn = int(g["hour_like"].min())
            mx = int(g["hour_like"].max())

            gg = g.copy()
            if mn == 0 and mx <= H - 1:
                gg["hour_1based"] = gg["hour_like"] + 1
            elif mn >= 1 and mx <= H:
                gg["hour_1based"] = gg["hour_like"]
            else:
                raise ValueError(
                    f"[SOURCE_PANEL] year={y}: invalid hour/hour_of_year range. "
                    f"min={mn}, max={mx}, expected 0..{H-1} or 1..{H}."
                )

            gg["timestamp_utc"] = (
                pd.Timestamp(f"{y}-01-01T00:00:00Z")
                + pd.to_timedelta(gg["hour_1based"] - 1, unit="h")
            )
            pieces.append(gg[["timestamp_utc"]])

        out = (
            pd.concat(pieces, ignore_index=True)
            .drop_duplicates()
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )

    if out.empty:
        raise ValueError(f"[SOURCE_PANEL] No usable hourly support extracted from {source_path}.")

    return SourcePanelSummary(
        hourly_timestamps=out,
        source_path=source_path,
        start_utc=pd.Timestamp(out["timestamp_utc"].min()),
        end_utc=pd.Timestamp(out["timestamp_utc"].max()),
        n_hours=int(len(out)),
    )


# =============================================================================
# PUBLIC EUR / MWH MODE
# =============================================================================
def _read_public_activity_file(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} file:\n  {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError(f"[{label}] File is empty: {path}")
    return df


def _build_timestamp_column(df: pd.DataFrame, label: str) -> pd.Series:
    ts_col = _pick_column(
        df.columns,
        ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc", "timestamp", "datetime", "datehour"],
        required=False,
    )
    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if ts.notna().any():
            return ts

    date_col = _pick_column(df.columns, ["date", "datum", "day"], required=False)
    hour_col = _pick_column(df.columns, ["hour", "uur"], required=False)
    if date_col is not None and hour_col is not None:
        d = pd.to_datetime(df[date_col], errors="coerce")
        h = pd.to_numeric(df[hour_col], errors="coerce")
        ts = pd.to_datetime(d.dt.strftime("%Y-%m-%d"), utc=True, errors="coerce") + pd.to_timedelta(h.fillna(0), unit="h")
        return ts

    year_col = _pick_column(df.columns, ["year", "jaar"], required=False)
    month_col = _pick_column(df.columns, ["month", "maand"], required=False)
    day_col = _pick_column(df.columns, ["day", "dag"], required=False)
    hour_col2 = _pick_column(df.columns, ["hour", "uur"], required=False)
    if year_col and month_col and day_col and hour_col2:
        y = pd.to_numeric(df[year_col], errors="coerce")
        m = pd.to_numeric(df[month_col], errors="coerce")
        d = pd.to_numeric(df[day_col], errors="coerce")
        h = pd.to_numeric(df[hour_col2], errors="coerce")
        ts = pd.to_datetime(
            {"year": y, "month": m, "day": d},
            utc=True,
            errors="coerce",
        ) + pd.to_timedelta(h.fillna(0), unit="h")
        return ts

    raise ValueError(
        f"[{label}] Could not build a UTC timestamp column. "
        f"Columns were: {list(df.columns)}"
    )


def _pick_value_column(df: pd.DataFrame, label: str, preferred_candidates: list[str]) -> str:
    col = _pick_column(df.columns, preferred_candidates, required=False)
    if col is not None:
        return col

    blacklist = {
        "year", "month", "day", "hour", "uur",
        "timestamp_utc", "utc_timestamp", "utc_time", "time_utc", "timestamp", "datetime",
    }

    numeric_candidates = []
    for c in df.columns:
        if str(c).strip().lower() in blacklist:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            numeric_candidates.append(c)

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    raise ValueError(
        f"[{label}] Could not identify a unique value column. "
        f"Tried preferred={preferred_candidates}, numeric_candidates={numeric_candidates}"
    )


def _load_public_metric(path: Path, label: str, value_candidates: list[str]) -> pd.DataFrame:
    raw = _read_public_activity_file(path, label=label)
    ts = _build_timestamp_column(raw, label=label)
    val_col = _pick_value_column(raw, label=label, preferred_candidates=value_candidates)

    operator_col = _pick_column(
        raw.columns,
        ["operator", "area", "scope", "zone", "country", "portfolio"],
        required=False,
    )

    out = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(ts, utc=True, errors="coerce"),
        "value": pd.to_numeric(raw[val_col], errors="coerce"),
    })

    if operator_col is not None:
        out["operator"] = raw[operator_col].map(_normalize_operator_value)
    else:
        out["operator"] = "national"

    out = out.dropna(subset=["timestamp_utc"]).copy()
    out["timestamp_utc"] = out["timestamp_utc"].dt.floor("h")
    out["value"] = out["value"].fillna(0.0).astype(float)

    out = (
        out.groupby(["timestamp_utc", "operator"], as_index=False)["value"]
        .sum()
        .sort_values(["timestamp_utc", "operator"])
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError(f"[{label}] No usable rows remained after timestamp/value parsing.")

    return out


def _metric_national_series(metric_df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, str]:
    """
    Build a clean national series for one metric.

    Rule:
      - if any non-national operator rows exist, national = sum(non-national)
      - else if direct national rows exist, national = direct national
      - else national is absent
    """
    non_national = metric_df[metric_df["operator"] != "national"][["timestamp_utc", value_col]].copy()
    direct_national = metric_df[metric_df["operator"] == "national"][["timestamp_utc", value_col]].copy()

    if not non_national.empty:
        out = (
            non_national.groupby("timestamp_utc", as_index=False)[value_col]
            .sum()
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )
        source_mode = "sum_non_national"
    elif not direct_national.empty:
        out = (
            direct_national.groupby("timestamp_utc", as_index=False)[value_col]
            .sum()
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )
        source_mode = "direct_national"
    else:
        out = pd.DataFrame(columns=["timestamp_utc", value_col])
        source_mode = "absent"

    out["operator"] = "national"
    return out, source_mode


def _window_meta_from_observed(
    support: SourcePanelSummary,
    observed_ts: pd.Series,
) -> dict[str, Any]:
    observed = pd.to_datetime(observed_ts, utc=True, errors="coerce").dropna()
    if observed.empty:
        return {
            "observed_start_utc": None,
            "observed_end_utc": None,
            "support_extends_before_observed": None,
            "support_extends_after_observed": None,
        }

    observed_start = pd.Timestamp(observed.min())
    observed_end = pd.Timestamp(observed.max())
    return {
        "observed_start_utc": str(observed_start),
        "observed_end_utc": str(observed_end),
        "support_extends_before_observed": bool(support.start_utc < observed_start),
        "support_extends_after_observed": bool(support.end_utc > observed_end),
    }


def _build_called_activity_from_public(
    source_support: SourcePanelSummary,
    eur_path: Path,
    mwh_path: Path,
) -> tuple[pd.DataFrame, dict]:
    eur = _load_public_metric(
        eur_path,
        label="EUR",
        value_candidates=["gopacs_eur", "called_eur_total", "called_eur", "eur", "amount_eur", "cost_eur", "value_eur"],
    ).rename(columns={"value": "gopacs_eur"})

    mwh = _load_public_metric(
        mwh_path,
        label="MWH",
        value_candidates=["gopacs_mwh", "called_mwh_total", "called_mwh", "mwh", "volume_mwh", "energy_mwh"],
    ).rename(columns={"value": "gopacs_mwh"})

    eur_gran = _detect_source_granularity(eur["timestamp_utc"])
    mwh_gran = _detect_source_granularity(mwh["timestamp_utc"])

    eur_non_national = eur[eur["operator"] != "national"][["timestamp_utc", "operator", "gopacs_eur"]].copy()
    mwh_non_national = mwh[mwh["operator"] != "national"][["timestamp_utc", "operator", "gopacs_mwh"]].copy()

    non_national_ops = sorted(
        set(eur_non_national["operator"].astype(str).unique().tolist())
        .union(set(mwh_non_national["operator"].astype(str).unique().tolist()))
    )

    pieces = []

    # Non-national panel
    if non_national_ops:
        non_nat_index = (
            source_support.hourly_timestamps.assign(_key=1)
            .merge(pd.DataFrame({"operator": non_national_ops, "_key": 1}), on="_key", how="inner")
            .drop(columns="_key")
            .sort_values(["timestamp_utc", "operator"])
            .reset_index(drop=True)
        )

        non_nat = non_nat_index.merge(
            eur_non_national,
            on=["timestamp_utc", "operator"],
            how="left",
            validate="one_to_one",
        ).merge(
            mwh_non_national,
            on=["timestamp_utc", "operator"],
            how="left",
            validate="one_to_one",
        )

        non_nat["gopacs_eur"] = pd.to_numeric(non_nat["gopacs_eur"], errors="coerce").fillna(0.0)
        non_nat["gopacs_mwh"] = pd.to_numeric(non_nat["gopacs_mwh"], errors="coerce").fillna(0.0)
        pieces.append(non_nat)

    # National panel, metric by metric, without double counting
    nat_eur, eur_national_mode = _metric_national_series(eur, "gopacs_eur")
    nat_mwh, mwh_national_mode = _metric_national_series(mwh, "gopacs_mwh")

    nat_index = source_support.hourly_timestamps.copy()
    nat_index["operator"] = "national"

    national = nat_index.merge(
        nat_eur[["timestamp_utc", "operator", "gopacs_eur"]],
        on=["timestamp_utc", "operator"],
        how="left",
        validate="one_to_one",
    ).merge(
        nat_mwh[["timestamp_utc", "operator", "gopacs_mwh"]],
        on=["timestamp_utc", "operator"],
        how="left",
        validate="one_to_one",
    )

    national["gopacs_eur"] = pd.to_numeric(national["gopacs_eur"], errors="coerce").fillna(0.0)
    national["gopacs_mwh"] = pd.to_numeric(national["gopacs_mwh"], errors="coerce").fillna(0.0)

    pieces.insert(0, national)

    out = pd.concat(pieces, ignore_index=True) if pieces else nat_index.assign(gopacs_eur=0.0, gopacs_mwh=0.0)

    out = (
        out.groupby(["timestamp_utc", "operator"], as_index=False)[["gopacs_eur", "gopacs_mwh"]]
        .sum()
        .sort_values(["timestamp_utc", "operator"])
        .reset_index(drop=True)
    )

    out["gopacs_activation"] = ((out["gopacs_eur"] > 0) | (out["gopacs_mwh"] > 0)).astype(int)

    if eur_gran in {"hourly", "subhourly"} and mwh_gran in {"hourly", "subhourly"}:
        source_granularity = "actual_hourly_or_subhourly"
        is_hourly_observed = True
    else:
        source_granularity = f"eur={eur_gran};mwh={mwh_gran}"
        is_hourly_observed = False

    out["source_granularity"] = source_granularity

    observed_ts = pd.concat([eur["timestamp_utc"], mwh["timestamp_utc"]], ignore_index=True)
    observed_window_meta = _window_meta_from_observed(source_support, observed_ts)

    meta = {
        "mode": "PUBLIC_ACTIVITY",
        "diagnostic_only": True,
        "source_support_path": str(source_support.source_path),
        "eur_path": str(eur_path),
        "mwh_path": str(mwh_path),
        "support_start_utc": str(source_support.start_utc),
        "support_end_utc": str(source_support.end_utc),
        "support_n_hours": int(source_support.n_hours),
        "eur_detected_granularity": eur_gran,
        "mwh_detected_granularity": mwh_gran,
        "source_granularity": source_granularity,
        "is_hourly_observed": bool(is_hourly_observed),
        "hourly_mode": "actual_hourly_or_subhourly" if is_hourly_observed else "aligned_to_support_from_public_files",
        "eur_national_mode": eur_national_mode,
        "mwh_national_mode": mwh_national_mode,
        "operators_present": sorted(out["operator"].astype(str).unique().tolist()),
        **observed_window_meta,
    }

    return out, meta


# =============================================================================
# STEP02 LOCALIZED FALLBACK MODE
# =============================================================================
def _load_step02_localized_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    ts_col = _pick_column(
        df.columns,
        ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc", "timestamp"],
        required=False,
    )
    if ts_col is None:
        raise ValueError(f"[STEP02] Missing timestamp column in {path}. Columns: {list(df.columns)}")

    province_col = _pick_column(df.columns, ["province"])

    eur_col = _pick_column(
        df.columns,
        ["gopacs_eur", "announcement_eur", "eur", "eur_total", "cost_eur", "gopacs_cost_eur"],
        required=False,
    )
    mwh_col = _pick_column(
        df.columns,
        [
            "gopacs_mwh",
            "announcement_mwh",
            "announcement_mwh_equiv",
            "mwh",
            "mwh_equiv",
            "mwh_total",
            "gopacs_mwh_equiv",
        ],
        required=False,
    )
    activation_col = _pick_column(
        df.columns,
        [
            "gopacs_activation",
            "activation",
            "announcement_activation",
            "is_active",
            "has_announcement",
        ],
        required=False,
    )
    count_col = _pick_column(
        df.columns,
        ["n_unique_records", "n_record_hours"],
        required=False,
    )

    if eur_col is None and mwh_col is None and activation_col is None and count_col is None:
        raise ValueError(
            f"[STEP02] Could not find usable GOPACS value columns in {path}. "
            f"Columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.floor("h"),
        "province": df[province_col].astype(str).str.strip(),
    })
    out = out.dropna(subset=["timestamp_utc"]).copy()

    out["gopacs_eur_local"] = pd.to_numeric(df[eur_col], errors="coerce").fillna(0.0) if eur_col is not None else 0.0
    out["gopacs_mwh_local"] = pd.to_numeric(df[mwh_col], errors="coerce").fillna(0.0) if mwh_col is not None else 0.0

    if activation_col is not None:
        out["gopacs_activation_local"] = pd.to_numeric(df[activation_col], errors="coerce").fillna(0.0)
        out["gopacs_activation_local"] = (out["gopacs_activation_local"] > 0).astype(int)
    elif count_col is not None:
        out["gopacs_activation_local"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0)
        out["gopacs_activation_local"] = (out["gopacs_activation_local"] > 0).astype(int)
    else:
        out["gopacs_activation_local"] = (
            (out["gopacs_eur_local"] > 0) | (out["gopacs_mwh_local"] > 0)
        ).astype(int)

    out = (
        out.groupby(["timestamp_utc", "province"], as_index=False)[
            ["gopacs_eur_local", "gopacs_mwh_local", "gopacs_activation_local"]
        ]
        .sum()
        .sort_values(["timestamp_utc", "province"])
        .reset_index(drop=True)
    )
    out["gopacs_activation_local"] = (out["gopacs_activation_local"] > 0).astype(int)

    return out


def _build_called_activity_from_step02(
    source_support: SourcePanelSummary,
    step02_path: Path,
) -> tuple[pd.DataFrame, dict]:
    loc = _load_step02_localized_panel(step02_path)

    panel_index = source_support.hourly_timestamps.copy()
    panel_index["operator"] = "national"

    national = (
        loc.groupby("timestamp_utc", as_index=False)[
            ["gopacs_eur_local", "gopacs_mwh_local", "gopacs_activation_local"]
        ]
        .sum()
        .rename(
            columns={
                "gopacs_eur_local": "gopacs_eur",
                "gopacs_mwh_local": "gopacs_mwh",
                "gopacs_activation_local": "gopacs_activation",
            }
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    national["gopacs_activation"] = (national["gopacs_activation"] > 0).astype(int)
    national["operator"] = "national"

    out = panel_index.merge(
        national[["timestamp_utc", "operator", "gopacs_eur", "gopacs_mwh", "gopacs_activation"]],
        on=["timestamp_utc", "operator"],
        how="left",
        validate="one_to_one",
    )

    out["gopacs_eur"] = pd.to_numeric(out["gopacs_eur"], errors="coerce").fillna(0.0)
    out["gopacs_mwh"] = pd.to_numeric(out["gopacs_mwh"], errors="coerce").fillna(0.0)
    out["gopacs_activation"] = pd.to_numeric(out["gopacs_activation"], errors="coerce").fillna(0).astype(int)
    out["gopacs_activation"] = (out["gopacs_activation"] > 0).astype(int)

    source_granularity = "from_localized_step02_panel_DIAGNOSTIC_ONLY"
    out["source_granularity"] = source_granularity

    observed_window_meta = _window_meta_from_observed(source_support, loc["timestamp_utc"])

    meta = {
        "mode": "STEP02_LOCALIZED",
        "diagnostic_only": True,
        "source_support_path": str(source_support.source_path),
        "step02_panel_path": str(step02_path),
        "support_start_utc": str(source_support.start_utc),
        "support_end_utc": str(source_support.end_utc),
        "support_n_hours": int(source_support.n_hours),
        "source_granularity": source_granularity,
        "is_hourly_observed": True,
        "hourly_mode": "from_localized_step02_panel",
        "operators_present": ["national"],
        **observed_window_meta,
    }

    return out, meta


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    configured_residual_path = _get_path(
        "HISTORICAL_PROVINCE_STRESS_PATH",
        "HISTORICAL_BACKBONE_PATH",
        "PAPER_BACKBONE_PATH",
        "PROVINCIAL_RESIDUAL_LOAD_SOURCE_PATH",
        "PROVINCE_STRESS_PATH",
        "PROVINCE_STRESS_PANEL_PATH",
        "RESIDUAL_LOAD_PANEL_PATH",
        "PUBLIC_STRESS_PATH",
        "STATE_BACKBONE_PATH",
        "PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_PATH",
        env_name="PROVINCIAL_RESIDUAL_LOAD_SOURCE_PATH",
        default=REPO_ROOT / "data" / "derived" / "provincial_residual_load_2025_2035_learned_optionB.csv",
    )

    configured_eur_path = _get_path(
        "EUR_PATH",
        env_name="GOPACS_EUR_PATH",
        default=REPO_ROOT / "data" / "external" / "gopacs_called_orders_eur_public.csv",
    )

    configured_mwh_path = _get_path(
        "MWH_PATH",
        env_name="GOPACS_MWH_PATH",
        default=REPO_ROOT / "data" / "external" / "gopacs_called_volumes_mwh_public.csv",
    )

    configured_step02_path = _get_path(
        "GOPACS_ANNOUNCEMENT_PANEL_PATH",
        "ANNOUNCEMENT_PANEL_PATH",
        "STEP02_PANEL_PATH",
        env_name="GOPACS_ANNOUNCEMENT_PANEL_PATH",
        default=REPO_ROOT / "data" / "derived" / "gopacs_announcement_province_hour.csv",
    )

    output_path = _get_path(
        "ACTIVITY_PATH",
        "ACTIVITY_DIAGNOSTIC_PATH",
        "GOPACS_CALLED_ACTIVITY_PATH",
        env_name="ACTIVITY_PATH",
        default=REPO_ROOT / "data" / "derived" / "gopacs_called_activity_hour.csv",
    )

    qc_dir = _get_path(
        "QC_DIR",
        env_name="QC_DIR",
        default=REPO_ROOT / "qc",
    )

    residual_path = _require_existing_path("residual-load source panel", configured_residual_path)
    source_support = _load_source_hour_support(residual_path)

    eur_exists = configured_eur_path.exists()
    mwh_exists = configured_mwh_path.exists()

    print("=" * 72)
    print("STEP 03: BUILD CALLED ACTIVITY PANEL (STRICT / EXACT PATHS)")
    print("=" * 72)
    print(f"Source panel : {residual_path}")
    print(f"EUR file     : {configured_eur_path}")
    print(f"MWH file     : {configured_mwh_path}")
    print(f"Step02 panel : {configured_step02_path}")
    print(f"Output       : {output_path}")

    if eur_exists and mwh_exists:
        print("\nMode: PUBLIC_ACTIVITY")
        eur_path = _require_existing_path("public EUR activity file", configured_eur_path)
        mwh_path = _require_existing_path("public MWh activity file", configured_mwh_path)

        out, meta = _build_called_activity_from_public(
            source_support=source_support,
            eur_path=eur_path,
            mwh_path=mwh_path,
        )
    else:
        print("\nMode: STEP02_LOCALIZED fallback")
        if not eur_exists:
            print(f"  public EUR file missing -> {configured_eur_path}")
        if not mwh_exists:
            print(f"  public MWh file missing -> {configured_mwh_path}")

        step02_path = _require_existing_path("step02 localized GOPACS panel", configured_step02_path)
        out, meta = _build_called_activity_from_step02(
            source_support=source_support,
            step02_path=step02_path,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(output_path, out)

    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_path = qc_dir / "step03_called_activity_summary.json"

    summary = {
        "rows": int(len(out)),
        "operators": sorted(out["operator"].astype(str).unique().tolist()),
        "mode": meta["mode"],
        "diagnostic_only": True,
        "source_panel_path": str(residual_path),
        "output_path": str(output_path),
        "support_start_utc": str(source_support.start_utc),
        "support_end_utc": str(source_support.end_utc),
        "support_n_hours": int(source_support.n_hours),
        "eur_nonzero_share": float((out["gopacs_eur"] > 0).mean()) if len(out) else 0.0,
        "mwh_nonzero_share": float((out["gopacs_mwh"] > 0).mean()) if len(out) else 0.0,
        "activation_nonzero_share": float((out["gopacs_activation"] > 0).mean()) if len(out) else 0.0,
        "source_granularity": meta["source_granularity"],
        "is_hourly_observed": meta["is_hourly_observed"],
        "observed_start_utc": meta.get("observed_start_utc"),
        "observed_end_utc": meta.get("observed_end_utc"),
        "support_extends_before_observed": meta.get("support_extends_before_observed"),
        "support_extends_after_observed": meta.get("support_extends_after_observed"),
    }

    _atomic_write_json(qc_path, summary)

    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    _atomic_write_json(metadata_path, meta)

    print(f"\nOutput rows : {len(out):,}")
    print(f"Operators   : {', '.join(summary['operators'])}")
    print(f"Mode        : {summary['mode']}")
    print(f"Support     : {summary['support_start_utc']} .. {summary['support_end_utc']}")
    print(f"Observed    : {summary['observed_start_utc']} .. {summary['observed_end_utc']}")
    print(f"Granularity : {summary['source_granularity']}")
    print(f"\n✓ Written: {output_path}")
    print(f"✓ Written: {metadata_path}")
    print(f"✓ Written: {qc_path}")


if __name__ == "__main__":
    main()