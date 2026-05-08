#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_provincial_hourly_load_from_learned_targets.py

Purpose
-------
Allocate learned national hourly HH/CO load to Dutch provinces while preserving
sector structure.

This script is downstream of:
    - build_province_sector_targets_and_factors.py
    - learn_national_load_and_sector_shapes.py

and upstream of:
    - build_provincial_residual_load_from_learned_load_optionB.py

Core allocation logic
---------------------
For each province p, year y, and hour t:

    prov_hh_load_MW(p,y,t) = hh_share_hat(p,y) * nat_hh_load_MW(y,t)
    prov_co_load_MW(p,y,t) = co_share_hat(p,y) * nat_co_load_MW(y,t)
    prov_load_MW(p,y,t)    = prov_hh_load_MW(p,y,t) + prov_co_load_MW(p,y,t)

Important convention update
---------------------------
The newer upstream pipeline uses 1-based hour_of_year:
    1, 2, ..., H_year
where H_year is 8760 or 8784.

Older versions of this script expected 0-based hour_of_year:
    0, 1, ..., H_year-1

This updated version accepts BOTH conventions on input, standardizes internally
to 1-based hour_of_year, and writes the output in 1-based convention so it is
consistent with the new national master and learned national load scripts.

Inputs
------
Preferred national hourly input:
    derived/national_hourly_load_2025_2035_learned.csv
Required columns:
    timestamp_utc, year, hour_of_year,
    nat_hh_load_MW, nat_co_load_MW, nat_load_MW

Provincial share input:
    derived/provincial_sector_shares_2012_2035.csv
Required columns:
    province, year, hh_share_hat, co_share_hat

Outputs
-------
    derived/provincial_hourly_load_2025_2035_learned.csv
    derived/provincial_hourly_load_2025_2035_qc_by_year.csv
    derived/provincial_hourly_load_2025_2035_diagnostics.json

Output columns
--------------
    timestamp_utc
    year
    hour_of_year
    province
    hh_share_hat
    co_share_hat
    prov_hh_load_MW
    prov_co_load_MW
    prov_load_MW
"""

from pathlib import Path
import json
import sys
from typing import List, Optional

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
# CONFIG
# =============================================================================
# =============================================================================
# CONFIG
# =============================================================================
IN_NAT_HOURLY = getattr(
    cfg,
    "NATIONAL_HOURLY_LOAD_LEARNED_PATH",
    cfg.DERIVED_DIR / "national_hourly_load_2025_2035_learned.csv",
)
IN_PROV_SHARES = getattr(
    cfg,
    "PROVINCIAL_SECTOR_SHARES_PATH",
    cfg.DERIVED_DIR / "provincial_sector_shares_2012_2035.csv",
)
OUT_PROV_HOURLY = getattr(
    cfg,
    "PROVINCIAL_HOURLY_LOAD_LEARNED_PATH",
    cfg.DERIVED_DIR / "provincial_hourly_load_2025_2035_learned.csv",
)
OUT_QC = getattr(
    cfg,
    "PROVINCIAL_HOURLY_LOAD_LEARNED_QC_PATH",
    cfg.DERIVED_DIR / "provincial_hourly_load_2025_2035_qc_by_year.csv",
)
OUT_DIAG = getattr(
    cfg,
    "PROVINCIAL_HOURLY_LOAD_LEARNED_DIAG_PATH",
    cfg.DERIVED_DIR / "provincial_hourly_load_2025_2035_diagnostics.json",
)
FORECAST_START = 2025
FORECAST_END = 2035
FORECAST_YEARS = list(range(FORECAST_START, FORECAST_END + 1))

PROVINCES = [
    "Drenthe",
    "Flevoland",
    "Friesland",
    "Gelderland",
    "Groningen",
    "Limburg",
    "Noord-Brabant",
    "Noord-Holland",
    "Overijssel",
    "Utrecht",
    "Zeeland",
    "Zuid-Holland",
]

TOL = 1e-9


# =============================================================================
# HELPERS
# =============================================================================
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def hours_in_year_utc(y: int) -> int:
    t0 = pd.Timestamp(f"{y}-01-01T00:00:00Z")
    t1 = pd.Timestamp(f"{y+1}-01-01T00:00:00Z")
    return int((t1 - t0) / pd.Timedelta(hours=1))


# =============================================================================
# INPUT LOADERS
# =============================================================================
def load_national_hourly() -> pd.DataFrame:
    if not IN_NAT_HOURLY.exists():
        raise FileNotFoundError(f"Missing national hourly file:\n  {IN_NAT_HOURLY}")

    df = pd.read_csv(IN_NAT_HOURLY)
    df.columns = [c.strip() for c in df.columns]

    ts_col = pick_first_existing(df.columns.tolist(), ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc"])
    req = ["year", "hour_of_year", "nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"]
    miss = [c for c in req if c not in df.columns]
    if ts_col is None:
        raise ValueError(f"[NAT_HOURLY] Missing UTC timestamp column. Has: {list(df.columns)}")
    if miss:
        raise ValueError(f"[NAT_HOURLY] Missing required columns: {miss}")

    df["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df["year"] = to_num(df["year"]).astype("Int64")
    df["hour_of_year"] = to_num(df["hour_of_year"]).astype("Int64")
    for c in ["nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"]:
        df[c] = to_num(df[c]).astype(float)

    df = df.dropna(subset=["timestamp_utc", "year", "hour_of_year", "nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"]).copy()
    df["year"] = df["year"].astype(int)
    df["hour_of_year"] = df["hour_of_year"].astype(int)

    df = df[df["year"].between(FORECAST_START, FORECAST_END, inclusive="both")].copy()
    df = df.sort_values(["year", "timestamp_utc"]).reset_index(drop=True)

    # Validate year-by-year and standardize hour_of_year to 1-based.
    parts = []
    qc_rows = []
    for y, g in df.groupby("year", sort=True):
        g = g.sort_values("timestamp_utc").copy()
        H = hours_in_year_utc(int(y))
        if len(g) != H:
            raise ValueError(f"[NAT_HOURLY] year={y}: row count {len(g)} != expected {H}.")

        hov = g["hour_of_year"].to_numpy(dtype=int)
        hov_min = int(hov.min())
        hov_max = int(hov.max())
        hov_nunique = int(pd.Series(hov).nunique())

        accepted_input = None
        if hov_min == 0 and hov_max == H - 1 and hov_nunique == H:
            accepted_input = "0_based"
            g["hour_of_year_input"] = g["hour_of_year"]
            g["hour_of_year"] = g["hour_of_year"] + 1
        elif hov_min == 1 and hov_max == H and hov_nunique == H:
            accepted_input = "1_based"
            g["hour_of_year_input"] = g["hour_of_year"]
        else:
            raise ValueError(
                f"[NAT_HOURLY] year={y}: hour_of_year coverage invalid. "
                f"min={hov_min}, max={hov_max}, nunique={hov_nunique}, "
                f"expected either 0..{H-1} or 1..{H}."
            )

        # Enforce consistency with timestamp ordering.
        g["hour_of_year_expected_1based"] = np.arange(1, H + 1, dtype=int)
        if not np.array_equal(g["hour_of_year"].to_numpy(dtype=int), g["hour_of_year_expected_1based"].to_numpy(dtype=int)):
            raise ValueError(
                f"[NAT_HOURLY] year={y}: hour_of_year does not match chronological timestamp order after standardization."
            )

        # Sector consistency.
        sector_gap = np.abs(g["nat_load_MW"] - (g["nat_hh_load_MW"] + g["nat_co_load_MW"]))
        max_sector_gap = float(sector_gap.max())
        if max_sector_gap > 1e-6:
            raise ValueError(f"[NAT_HOURLY] year={y}: nat_load_MW != nat_hh_load_MW + nat_co_load_MW (max gap {max_sector_gap}).")

        qc_rows.append({
            "year": int(y),
            "n_hours": int(H),
            "hour_of_year_input_convention": accepted_input,
            "nat_hh_sum_MWh": float(g["nat_hh_load_MW"].sum()),
            "nat_co_sum_MWh": float(g["nat_co_load_MW"].sum()),
            "nat_total_sum_MWh": float(g["nat_load_MW"].sum()),
        })
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    qc = pd.DataFrame(qc_rows).sort_values("year").reset_index(drop=True)
    return out[[
        "timestamp_utc", "year", "hour_of_year",
        "nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"
    ]].copy(), qc


def load_provincial_shares() -> pd.DataFrame:
    if not IN_PROV_SHARES.exists():
        raise FileNotFoundError(f"Missing provincial sector shares file:\n  {IN_PROV_SHARES}")

    df = pd.read_csv(IN_PROV_SHARES)
    df.columns = [c.strip() for c in df.columns]

    req = ["province", "year", "hh_share_hat", "co_share_hat"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"[PROV_SHARES] Missing required columns: {miss}")

    df["province"] = df["province"].astype(str).str.strip()
    df["year"] = to_num(df["year"]).astype("Int64")
    df["hh_share_hat"] = to_num(df["hh_share_hat"]).astype(float)
    df["co_share_hat"] = to_num(df["co_share_hat"]).astype(float)

    df = df.dropna(subset=["province", "year", "hh_share_hat", "co_share_hat"]).copy()
    df["year"] = df["year"].astype(int)

    df = df[df["year"].between(FORECAST_START, FORECAST_END, inclusive="both")].copy()
    df = df[df["province"].isin(PROVINCES)].copy()

    # Enforce complete province-year grid.
    full_idx = pd.MultiIndex.from_product([FORECAST_YEARS, PROVINCES], names=["year", "province"])
    df = df.set_index(["year", "province"]).sort_index()
    missing_idx = full_idx.difference(df.index)
    if len(missing_idx) > 0:
        sample = list(missing_idx[:10])
        raise ValueError(f"[PROV_SHARES] Missing province-year share rows, e.g. {sample}")

    df = df.loc[full_idx].reset_index()

    # Yearly simplex checks.
    rows = []
    for y, g in df.groupby("year", sort=True):
        hh_sum = float(g["hh_share_hat"].sum())
        co_sum = float(g["co_share_hat"].sum())
        if abs(hh_sum - 1.0) > 1e-8:
            raise ValueError(f"[PROV_SHARES] year={y}: hh_share_hat sums to {hh_sum}, not 1.")
        if abs(co_sum - 1.0) > 1e-8:
            raise ValueError(f"[PROV_SHARES] year={y}: co_share_hat sums to {co_sum}, not 1.")
        if (g["hh_share_hat"] <= 0).any():
            raise ValueError(f"[PROV_SHARES] year={y}: hh_share_hat contains nonpositive entries.")
        if (g["co_share_hat"] <= 0).any():
            raise ValueError(f"[PROV_SHARES] year={y}: co_share_hat contains nonpositive entries.")
        rows.append({
            "year": int(y),
            "hh_share_hat_sum": hh_sum,
            "co_share_hat_sum": co_sum,
        })

    return df[["province", "year", "hh_share_hat", "co_share_hat"]].copy(), pd.DataFrame(rows)


# =============================================================================
# ALLOCATION
# =============================================================================
def build_provincial_hourly(nat: pd.DataFrame, prov: pd.DataFrame) -> pd.DataFrame:
    out = nat.merge(prov, on="year", how="inner", validate="many_to_many")

    # Because each year should have exactly 12 provincial rows, validate counts.
    counts = out.groupby(["year", "timestamp_utc"], as_index=False)["province"].count().rename(columns={"province": "n_provinces"})
    bad = counts.loc[counts["n_provinces"] != len(PROVINCES)]
    if not bad.empty:
        raise ValueError("[ALLOC] Some hourly timestamps do not expand to all 12 provinces.")

    out["prov_hh_load_MW"] = out["hh_share_hat"] * out["nat_hh_load_MW"]
    out["prov_co_load_MW"] = out["co_share_hat"] * out["nat_co_load_MW"]
    out["prov_load_MW"] = out["prov_hh_load_MW"] + out["prov_co_load_MW"]

    # Hourly exactness checks against national load.
    agg = (
        out.groupby(["year", "timestamp_utc", "hour_of_year"], as_index=False)[[
            "prov_hh_load_MW", "prov_co_load_MW", "prov_load_MW",
            "nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"
        ]]
        .sum()
    )
    # nat_* were repeated 12 times by the merge, so divide those repeated sums back.
    agg["nat_hh_load_MW"] /= len(PROVINCES)
    agg["nat_co_load_MW"] /= len(PROVINCES)
    agg["nat_load_MW"] /= len(PROVINCES)

    max_gap_hh = float(np.abs(agg["prov_hh_load_MW"] - agg["nat_hh_load_MW"]).max())
    max_gap_co = float(np.abs(agg["prov_co_load_MW"] - agg["nat_co_load_MW"]).max())
    max_gap_tot = float(np.abs(agg["prov_load_MW"] - agg["nat_load_MW"]).max())

    if max(max_gap_hh, max_gap_co, max_gap_tot) > 1e-6:
        raise ValueError(
            f"[ALLOC] Hourly provincial sums do not match national totals. "
            f"max gaps: HH={max_gap_hh}, CO={max_gap_co}, TOT={max_gap_tot}"
        )

    return out[[
        "timestamp_utc", "year", "hour_of_year", "province",
        "hh_share_hat", "co_share_hat",
        "prov_hh_load_MW", "prov_co_load_MW", "prov_load_MW"
    ]].copy()


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("=== Build provincial hourly load from learned HH/CO targets and shapes ===")
    for p in [OUT_PROV_HOURLY, OUT_QC, OUT_DIAG]:
        ensure_parent(p)

    nat, nat_qc = load_national_hourly()
    prov, prov_qc = load_provincial_shares()
    out = build_provincial_hourly(nat, prov)

    # Annual QC summary.
    qc_rows = []
    for y, g in out.groupby("year", sort=True):
        nat_y = nat.loc[nat["year"] == y]
        qc_rows.append({
            "year": int(y),
            "n_hours": int(len(nat_y)),
            "n_rows_out": int(len(g)),
            "n_expected_rows": int(len(nat_y) * len(PROVINCES)),
            "prov_hh_sum_MWh": float(g["prov_hh_load_MW"].sum()),
            "prov_co_sum_MWh": float(g["prov_co_load_MW"].sum()),
            "prov_total_sum_MWh": float(g["prov_load_MW"].sum()),
            "nat_hh_sum_MWh": float(nat_y["nat_hh_load_MW"].sum()),
            "nat_co_sum_MWh": float(nat_y["nat_co_load_MW"].sum()),
            "nat_total_sum_MWh": float(nat_y["nat_load_MW"].sum()),
            "hour_of_year_min": int(g["hour_of_year"].min()),
            "hour_of_year_max": int(g["hour_of_year"].max()),
        })

    qc = pd.DataFrame(qc_rows).sort_values("year").reset_index(drop=True)
    qc.to_csv(OUT_QC, index=False)

    out = out.sort_values(["year", "timestamp_utc", "province"]).reset_index(drop=True)
    out.to_csv(OUT_PROV_HOURLY, index=False)

    diag = {
        "input_files": {
            "national_hourly": str(IN_NAT_HOURLY),
            "provincial_shares": str(IN_PROV_SHARES),
        },
        "forecast_years": FORECAST_YEARS,
        "province_count": len(PROVINCES),
        "hour_of_year_output_convention": "1_based",
        "notes": [
            "This script accepts either 0-based or 1-based hour_of_year from the national hourly input.",
            "It standardizes internally and outputs 1-based hour_of_year to match the updated upstream pipeline.",
            "Provincial hourly loads preserve sector structure by allocating nat_hh_load_MW and nat_co_load_MW separately.",
        ],
        "nat_hourly_qc": nat_qc.to_dict(orient="records"),
        "prov_share_qc": prov_qc.to_dict(orient="records"),
    }
    with open(OUT_DIAG, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print(f"[OK] Wrote provincial hourly load: {OUT_PROV_HOURLY}")
    print(f"[OK] Wrote QC by year: {OUT_QC}")
    print(f"[OK] Wrote diagnostics: {OUT_DIAG}")
    print("\n=== SUMMARY ===")
    print(f"Years covered               : {FORECAST_START}-{FORECAST_END}")
    print(f"Output hour_of_year         : 1-based")
    print(f"Rows written                : {len(out):,}")
    print(f"Provinces per hour          : {len(PROVINCES)}")


if __name__ == "__main__":
    main()
