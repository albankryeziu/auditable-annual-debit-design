#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A_build_debit_optimization_inputs.py
====================================

Build annual optimization inputs for the annual debit-rule paper.

UPDATED VERSION WITH TDTR CANDIDATE-HOUR SUPPORT AND DEPTH-SCARCITY CALIBRATION
------------------------------------------------------------------------------

This version treats the TDTR candidate-hour support produced by
G_build_tdtr_candidate_hour_support.py as a first-class modelling object and
separates two different annual contract quantities:

    1. Candidate restriction-hour support:
           |S_y| approximately 15% of annual hours under the Dutch TDTR-like
           availability interpretation.

    2. Weighted-depth annual allowance:
           H_i^{depth} = rho_depth * |S_y|,

       where rho_depth is the average allowed full-power-equivalent restriction
       depth inside the selected TDTR candidate support.

This is the key correction for meaningful M2/M3 comparisons. The 15% rule
selects the hours in which restrictions may occur. It should not automatically
mean that each participant may be restricted at full contracted capacity in
every selected hour.

Core design logic
-----------------
1. The primitive physical relief layer is fixed across M1/M2/M3.
2. Candidate TDTR hours are selected upstream by G, using public/auditable
   stress scores. A does not re-rank hours.
3. A enforces the candidate-hour support as an availability restriction:

       Pbar_{p,t} = 0 outside selected TDTR support,
       Pbar_{p,t} = P_ctr,p inside selected TDTR support,

   unless a stricter upstream availability field already exists, in which case
   the support is multiplied into that field.

4. Requirements and coefficients are also filtered to selected support as a
   defensive consistency step. If H already exported support-filtered
   requirements, this is idempotent.

5. M1 touched-time allowance is calibrated to the number of selected TDTR
   support hours:

       H_i^{M1} = |S_y|.

   For the medium 15% scenario this matches the Dutch-practice statement that
   restrictions can occur in at most approximately 15% of annual hours.

6. M2/M3 weighted-depth allowance is calibrated separately:

       H_i^{depth} = rho_depth * |S_y|.

   The parameter rho_depth controls the average full-power-equivalent depth
   allowed inside candidate hours. For example, rho_depth=0.02 means an average
   2% restriction depth across the TDTR support.

Inputs
------
1) data/derived/auditable_state_province_hour.csv

2) PTDF_OUTPUT_PUBLICATION/debit_paper_inputs/
       active_branches_{year}.csv
       branch_hour_requirements_{year}.csv.gz
       branch_hour_province_coefficients_{year}.csv.gz

3) data/derived/tdtr_candidate_hour_support.csv, or scenario-specific file:
       data/derived/tdtr_candidate_hour_support__{scenario}.csv

4) Optional user-supplied participant capacity table.

5) Optional user-supplied tariff table.

Outputs
-------
data/derived/debit_optimization_inputs/
    debit_opt_state_{year}.csv/.parquet
    debit_opt_participants_{year}.csv
    debit_opt_requirements_{year}.csv/.parquet
    debit_opt_coefficients_{year}.csv/.parquet
    debit_opt_active_branches_{year}.csv
    debit_opt_tariff_inputs_{year}.csv
    debit_opt_year_manifest_{year}.json
    debit_optimization_input_manifest.json

Paper interpretation
--------------------
The support identifies candidate TDTR scarcity hours. The annual debit-rule
comparison then asks how restrictions assigned inside this same candidate-hour
support are charged against annual allowances under M1/M2/M3. This prevents M1
from becoming a full-year arbitrary binary activation problem and aligns the
operating layer with a TDTR-like access-window design.

The weighted-depth allowance rho_depth is a scarcity-calibration parameter, not
a learned performance knob. It should be reported as part of the experimental
design and varied in sensitivity runs.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pipeline_config as cfg  # type: ignore
except Exception:  # pragma: no cover
    cfg = None


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def cfg_get(name: str, default: Any) -> Any:
    if cfg is not None and hasattr(cfg, name):
        value = getattr(cfg, name)
        if value is not None and value != "":
            return value
    return os.environ.get(name, default)


def cfg_bool(name: str, default: bool) -> bool:
    val = cfg_get(name, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)

    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def cfg_float(name: str, default: float) -> float:
    val = cfg_get(name, default)
    try:
        return float(val)
    except Exception:
        return float(default)


def cfg_int(name: str, default: int) -> int:
    val = cfg_get(name, default)
    try:
        return int(val)
    except Exception:
        return int(default)


# =============================================================================
# GLOBAL CONFIG
# =============================================================================

DEFAULT_PROVINCES = [
    "Groningen",
    "Friesland",
    "Drenthe",
    "Overijssel",
    "Flevoland",
    "Gelderland",
    "Utrecht",
    "Noord-Holland",
    "Zuid-Holland",
    "Zeeland",
    "Noord-Brabant",
    "Limburg",
]

DERIVED_DIR = Path(cfg_get("DERIVED_DIR", REPO_ROOT / "data" / "derived"))
OUTPUTS_DIR = Path(cfg_get("OUTPUTS_DIR", REPO_ROOT / "outputs"))
QC_DIR = Path(cfg_get("QC_DIR", REPO_ROOT / "qc"))

YEARS = sorted(dict.fromkeys(map(int, cfg_get("ALL_YEARS", [2021, 2022, 2023, 2024, 2025]))))
PROVINCES = list(cfg_get("NL_PROVINCES", DEFAULT_PROVINCES))

STATE_PANEL_CSV = Path(
    cfg_get(
        "AUDITABLE_STATE_PANEL_PATH",
        cfg_get("STATE_PANEL_PATH", DERIVED_DIR / "auditable_state_province_hour.csv"),
    )
)

PTDF_DEBIT_INPUT_DIR = Path(
    cfg_get(
        "PTDF_DEBIT_INPUT_DIR",
        REPO_ROOT / "PTDF_OUTPUT_PUBLICATION" / "debit_paper_inputs",
    )
)

OPT_INPUT_DIR = Path(
    cfg_get(
        "DEBIT_OPT_INPUT_DIR",
        DERIVED_DIR / "debit_optimization_inputs",
    )
)

# =============================================================================
# TDTR CANDIDATE-HOUR SUPPORT CONFIG
# =============================================================================

TDTR_SUPPORT_SCENARIO = str(cfg_get("TDTR_SUPPORT_SCENARIO", "medium")).strip()
TDTR_SUPPORT_REQUIRED = cfg_bool("TDTR_SUPPORT_REQUIRED", True)
TDTR_SUPPORT_APPLY_TO_REQUIREMENTS = cfg_bool("TDTR_SUPPORT_APPLY_TO_REQUIREMENTS", True)
TDTR_SUPPORT_APPLY_TO_PBAR = cfg_bool("TDTR_SUPPORT_APPLY_TO_PBAR", True)

TDTR_SUPPORT_MASK_COLUMN_CANDIDATES = [
    "selected",
    "is_selected",
    "is_tdtr_candidate",
    "is_tdtr_candidate_hour",
    "tdtr_candidate",
    "support",
    "y_support",
    "candidate_hour",
]

TDTR_SUPPORT_PATH = Path(cfg_get("TDTR_SUPPORT_PATH", DERIVED_DIR / "tdtr_candidate_hour_support.csv"))

TDTR_SUPPORT_SCENARIO_PATH = Path(
    cfg_get(
        "TDTR_SUPPORT_SCENARIO_PATH",
        DERIVED_DIR / f"tdtr_candidate_hour_support__{TDTR_SUPPORT_SCENARIO}.csv",
    )
)

TDTR_SUPPORT_ALL_SCENARIOS_PATH = Path(
    cfg_get(
        "TDTR_SUPPORT_ALL_SCENARIOS_PATH",
        DERIVED_DIR / "tdtr_candidate_hour_support_all_scenarios.csv",
    )
)

# =============================================================================
# PARTICIPANT / TARIFF INPUTS
# =============================================================================

PARTICIPANT_CAPACITY_PATH_RAW = cfg_get("DEBIT_PARTICIPANT_CAPACITY_PATH", None)
PARTICIPANT_CAPACITY_PATH = (
    None
    if PARTICIPANT_CAPACITY_PATH_RAW in [None, "", "none", "None"]
    else Path(PARTICIPANT_CAPACITY_PATH_RAW)
)

TARIFF_TABLE_PATH_RAW = cfg_get("DEBIT_TARIFF_TABLE_PATH", None)
TARIFF_TABLE_PATH = (
    None
    if TARIFF_TABLE_PATH_RAW in [None, "", "none", "None"]
    else Path(TARIFF_TABLE_PATH_RAW)
)

# =============================================================================
# DUTCH TDTR AND ALLOWANCE SETTINGS
# =============================================================================

USE_TDTR85_BUDGETS = cfg_bool("USE_TDTR85_BUDGETS", True)

TDTR_AVAILABILITY_SHARE = cfg_float("TDTR_AVAILABILITY_SHARE", 0.85)
if not (0.0 < TDTR_AVAILABILITY_SHARE < 1.0):
    raise ValueError("TDTR_AVAILABILITY_SHARE must lie strictly between 0 and 1.")

TDTR_RESTRICTION_SHARE = 1.0 - TDTR_AVAILABILITY_SHARE

# Main correction:
# candidate hours are around 15% of the year, but weighted-depth budget is
# rho_depth times those selected hours.
DEPTH_BUDGET_MODE = str(cfg_get("DEBIT_DEPTH_BUDGET_MODE", "rho_support")).strip().lower()

DEPTH_BUDGET_RHO = cfg_float(
    "DEBIT_DEPTH_BUDGET_RHO",
    cfg_float("TDTR_DEPTH_RHO", 0.02),
)

if DEPTH_BUDGET_MODE in {"rho_support", "share_of_support", "rho_times_support"}:
    if not (0.0 < DEPTH_BUDGET_RHO <= 1.0):
        raise ValueError(
            "DEBIT_DEPTH_BUDGET_RHO / TDTR_DEPTH_RHO must lie in (0,1] "
            "when DEBIT_DEPTH_BUDGET_MODE='rho_support'."
        )

DEPTH_BUDGET_HOURS_FALLBACK = cfg_float("DEBIT_DEPTH_BUDGET_HOURS", 300.0)
TOUCH_BUDGET_HOURS_FALLBACK = cfg_int("DEBIT_TOUCH_BUDGET_HOURS", 300)

# For the TDTR-support paper, this should remain True:
# M1 touched-token allowance equals the selected TDTR support hours.
TOUCH_BUDGET_FROM_SUPPORT = cfg_bool("DEBIT_TOUCH_BUDGET_FROM_SUPPORT", True)

CONTRACT_CAPACITY_MODE = str(
    cfg_get("DEBIT_CONTRACT_CAPACITY_MODE", "fraction_of_p99_pressure")
).lower().strip()

CONTRACT_CAPACITY_FRACTION = cfg_float("DEBIT_CONTRACT_CAPACITY_FRACTION", 0.30)
CONTRACT_CAPACITY_MIN_MW = cfg_float("DEBIT_CONTRACT_CAPACITY_MIN_MW", 25.0)

CONTRACT_CAPACITY_MAX_MW_RAW = cfg_get("DEBIT_CONTRACT_CAPACITY_MAX_MW", None)
CONTRACT_CAPACITY_MAX_MW = (
    None
    if CONTRACT_CAPACITY_MAX_MW_RAW in [None, "", "none", "None"]
    else float(CONTRACT_CAPACITY_MAX_MW_RAW)
)

# =============================================================================
# DUTCH DSO TARIFF / COST SETTINGS
# =============================================================================

FIRM_CAPACITY_MULTIPLIER = cfg_float("FIRM_CAPACITY_MULTIPLIER", 1.0)
TDTR_CAPACITY_MULTIPLIER = cfg_float("TDTR_CAPACITY_MULTIPLIER", 0.0)
TARIFF_REFERENCE_YEAR = cfg_int("DEBIT_TARIFF_REFERENCE_YEAR", 2026)

DEFAULT_PROVINCE_DSO = {
    "Groningen": "Enexis",
    "Drenthe": "Enexis",
    "Overijssel": "Enexis",
    "Noord-Brabant": "Enexis",
    "Limburg": "Enexis",
    "Friesland": "Liander",
    "Flevoland": "Liander",
    "Gelderland": "Liander",
    "Noord-Holland": "Liander",
    "Zuid-Holland": "Stedin",
    "Utrecht": "Stedin",
    "Zeeland": "Stedin",
}

PROVINCE_DSO = dict(cfg_get("DEBIT_PROVINCE_DSO_MAP", DEFAULT_PROVINCE_DSO))

DEFAULT_DSO_TARIFFS_2026 = {
    "Enexis": {
        "gv_class": "MS-D (125,1500] kW",
        "fixed_eur_year": 441.0,
        "capacity_eur_mw_year": 28_910.0,
        "peak_eur_mw_month": 3_660.00,
        "energy_high_eur_mwh": 24.7,
        "energy_low_eur_mwh": 24.7,
    },
    "Liander": {
        "gv_class": "MS (136,2000] kW",
        "fixed_eur_year": 441.0,
        "capacity_eur_mw_year": 27_480.0,
        "peak_eur_mw_month": 3_570.00,
        "energy_high_eur_mwh": 22.6,
        "energy_low_eur_mwh": 22.6,
    },
    "Stedin": {
        "gv_class": "MS (151,1500] kW",
        "fixed_eur_year": 441.0,
        "capacity_eur_mw_year": 24_273.6,
        "peak_eur_mw_month": 3_096.68,
        "energy_high_eur_mwh": 19.8,
        "energy_low_eur_mwh": 19.8,
    },
}

SLACK_PENALTY = cfg_float("DEBIT_SLACK_PENALTY", cfg_float("M_PENALTY", 5000.0))

MIN_REQUIREMENT_MW = cfg_float("DEBIT_MIN_REQUIREMENT_MW", 1e-9)
MIN_COEFFICIENT = cfg_float("DEBIT_MIN_COEFFICIENT", 1e-12)
RESTRICTION_COST_MIN = cfg_float("DEBIT_RESTRICTION_COST_MIN", 1e-6)
PRICE_WINSOR_Q = cfg_float("DEBIT_PRICE_COST_WINSOR_Q", 0.995)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def hours_in_year_utc(year: int) -> int:
    return int(
        (
            pd.Timestamp(f"{year + 1}-01-01T00:00:00Z")
            - pd.Timestamp(f"{year}-01-01T00:00:00Z")
        )
        / pd.Timedelta(hours=1)
    )


def pick_column(
    columns: Iterable[str],
    candidates: Iterable[str],
    required: bool = True,
) -> Optional[str]:
    cols = [str(c).strip() for c in columns]
    low = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {list(candidates)!r}. "
            f"Available columns: {cols!r}"
        )

    return None


def minmax01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    lo = float(x.min())
    hi = float(x.max())

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=s.index)

    return (x - lo) / (hi - lo)


def safe_numeric(df: pd.DataFrame, col: str, default: float | pd.Series = 0.0) -> pd.Series:
    if col not in df.columns:
        if isinstance(default, pd.Series):
            return default.astype(float)
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def safe_to_parquet_and_csv(df: pd.DataFrame, stem: Path) -> dict:
    stem.parent.mkdir(parents=True, exist_ok=True)

    csv_path = stem.with_suffix(".csv")
    parquet_path = stem.with_suffix(".parquet")

    out = {"csv": str(csv_path), "parquet": None}

    df.to_csv(csv_path, index=False)

    try:
        df.to_parquet(parquet_path, index=False)
        out["parquet"] = str(parquet_path)
    except Exception as exc:
        print(f"[WARN] Could not write parquet {parquet_path.name}: {exc}")

    print(f"[SAVE] {csv_path}" + (f" and {parquet_path}" if out["parquet"] else ""))
    return out


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    print(f"[SAVE] {path}")


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}:\n  {path}")
    return path


# =============================================================================
# TDTR CANDIDATE-HOUR SUPPORT
# =============================================================================

def resolve_support_path() -> Optional[Path]:
    candidates: list[Path] = []

    if TDTR_SUPPORT_SCENARIO and TDTR_SUPPORT_SCENARIO.lower() not in {"", "none", "main"}:
        candidates.append(TDTR_SUPPORT_SCENARIO_PATH)

    candidates.append(TDTR_SUPPORT_PATH)
    candidates.append(TDTR_SUPPORT_ALL_SCENARIOS_PATH)

    for p in candidates:
        if p.exists():
            return p

    return None


def load_tdtr_support() -> tuple[pd.DataFrame, dict]:
    path = resolve_support_path()

    meta = {
        "support_required": TDTR_SUPPORT_REQUIRED,
        "support_scenario": TDTR_SUPPORT_SCENARIO,
        "support_path": str(path) if path else None,
        "apply_to_requirements": TDTR_SUPPORT_APPLY_TO_REQUIREMENTS,
        "apply_to_pbar": TDTR_SUPPORT_APPLY_TO_PBAR,
    }

    if path is None:
        if TDTR_SUPPORT_REQUIRED:
            raise FileNotFoundError(
                "TDTR candidate-hour support is required but no support file was found. Tried:\n"
                f"  {TDTR_SUPPORT_SCENARIO_PATH}\n"
                f"  {TDTR_SUPPORT_PATH}\n"
                f"  {TDTR_SUPPORT_ALL_SCENARIOS_PATH}\n"
                "Run G_build_tdtr_candidate_hour_support.py first, or set TDTR_SUPPORT_REQUIRED=0."
            )

        print("[WARN] No TDTR support file found. Continuing with full-year support.")
        return pd.DataFrame(), meta

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "scenario" in df.columns and TDTR_SUPPORT_SCENARIO and path.name.endswith("all_scenarios.csv"):
        before = len(df)
        df = df[df["scenario"].astype(str).str.lower() == TDTR_SUPPORT_SCENARIO.lower()].copy()
        print(
            f"[TDTR SUPPORT] filtered all-scenarios file to scenario={TDTR_SUPPORT_SCENARIO}: "
            f"{before:,} -> {len(df):,} rows"
        )

    y_col = pick_column(df.columns, ["year"])
    h_col = pick_column(df.columns, ["hour_of_year", "hour", "hoy"])
    ts_col = pick_column(
        df.columns,
        ["timestamp_utc", "timestamp", "utc_time", "time_utc"],
        required=False,
    )
    mask_col = pick_column(df.columns, TDTR_SUPPORT_MASK_COLUMN_CANDIDATES, required=False)

    if mask_col is None:
        raise ValueError(
            f"TDTR support file {path} has no support mask column. "
            f"Tried {TDTR_SUPPORT_MASK_COLUMN_CANDIDATES}. "
            f"Columns={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["year"] = pd.to_numeric(df[y_col], errors="coerce").astype("Int64")
    out["hour_of_year"] = pd.to_numeric(df[h_col], errors="coerce").astype("Int64")
    out["tdtr_support"] = pd.to_numeric(df[mask_col], errors="coerce").fillna(0.0).astype(float)

    out = out.dropna(subset=["year", "hour_of_year"]).copy()
    out["year"] = out["year"].astype(int)
    out["hour_of_year"] = out["hour_of_year"].astype(int)
    out["tdtr_support"] = (out["tdtr_support"] > 0.5).astype(int)

    if ts_col is not None:
        out["timestamp_utc"] = (
            pd.to_datetime(df.loc[out.index, ts_col], utc=True, errors="coerce")
            .dt.floor("h")
        )
    else:
        out["timestamp_utc"] = (
            pd.to_datetime(out["year"].astype(str) + "-01-01T00:00:00Z", utc=True)
            + pd.to_timedelta(out["hour_of_year"] - 1, unit="h")
        )

    out = (
        out[["year", "hour_of_year", "timestamp_utc", "tdtr_support"]]
        .groupby(["year", "hour_of_year", "timestamp_utc"], as_index=False)["tdtr_support"]
        .max()
        .sort_values(["year", "hour_of_year"])
        .reset_index(drop=True)
    )

    summary = (
        out.groupby("year", as_index=False)
        .agg(hours=("hour_of_year", "nunique"), selected=("tdtr_support", "sum"))
    )
    summary["selected_share"] = summary["selected"] / summary["hours"].replace(0, np.nan)

    print(f"[TDTR SUPPORT] using {path}")
    print(summary.to_string(index=False))

    meta.update(
        {
            "support_path": str(path),
            "support_rows": int(len(out)),
            "support_selected_by_year": {
                str(int(r["year"])): {
                    "hours": int(r["hours"]),
                    "selected": int(r["selected"]),
                    "selected_share": float(r["selected_share"]),
                }
                for _, r in summary.iterrows()
            },
        }
    )

    return out, meta


def validate_support_year(support: pd.DataFrame, year: int) -> pd.DataFrame:
    H = hours_in_year_utc(year)

    if support.empty:
        ts = pd.date_range(pd.Timestamp(f"{year}-01-01T00:00:00Z"), periods=H, freq="h")
        return pd.DataFrame(
            {
                "year": year,
                "hour_of_year": np.arange(1, H + 1),
                "timestamp_utc": ts,
                "tdtr_support": 1,
            }
        )

    sub = support.loc[support["year"] == int(year)].copy()

    if sub.empty:
        if TDTR_SUPPORT_REQUIRED:
            raise ValueError(f"TDTR support has no rows for year={year}.")

        ts = pd.date_range(pd.Timestamp(f"{year}-01-01T00:00:00Z"), periods=H, freq="h")
        return pd.DataFrame(
            {
                "year": year,
                "hour_of_year": np.arange(1, H + 1),
                "timestamp_utc": ts,
                "tdtr_support": 1,
            }
        )

    expected = pd.DataFrame(
        {
            "year": year,
            "hour_of_year": np.arange(1, H + 1, dtype=int),
            "timestamp_utc_expected": pd.date_range(
                pd.Timestamp(f"{year}-01-01T00:00:00Z"),
                periods=H,
                freq="h",
            ),
        }
    )

    sub = expected.merge(
        sub[["year", "hour_of_year", "tdtr_support"]],
        on=["year", "hour_of_year"],
        how="left",
    )

    missing = int(sub["tdtr_support"].isna().sum())

    if missing > 0 and TDTR_SUPPORT_REQUIRED:
        raise ValueError(
            f"TDTR support year={year} is incomplete: missing {missing} hours out of {H}."
        )

    sub["tdtr_support"] = sub["tdtr_support"].fillna(0).astype(int)
    sub = sub.rename(columns={"timestamp_utc_expected": "timestamp_utc"})

    return sub[["year", "hour_of_year", "timestamp_utc", "tdtr_support"]]


# =============================================================================
# TARIFF AND PARTICIPANT INPUTS
# =============================================================================

def load_tariff_table_for_years(years: list[int]) -> pd.DataFrame:
    if TARIFF_TABLE_PATH is not None:
        require_file(TARIFF_TABLE_PATH, "debit tariff table")

        tariff = pd.read_csv(TARIFF_TABLE_PATH, low_memory=False)
        tariff.columns = [str(c).strip() for c in tariff.columns]

        province_col = pick_column(tariff.columns, ["province", "provincie", "node"], required=False)
        dso_col = pick_column(tariff.columns, ["dso", "DSO", "netbeheerder"], required=False)
        year_col = pick_column(tariff.columns, ["year", "tariff_year", "reference_year"], required=False)

        if province_col is None and dso_col is None:
            raise ValueError("Tariff table must contain either a province column or a DSO column.")

        if province_col is None:
            pieces = []
            for _, row in tariff.iterrows():
                dso = str(row[dso_col]).strip()
                provs = [
                    p
                    for p, d in PROVINCE_DSO.items()
                    if str(d).strip().lower() == dso.lower()
                ]
                for p in provs:
                    rr = row.copy()
                    rr["province"] = p
                    pieces.append(rr)
            tariff = pd.DataFrame(pieces)
        elif province_col != "province":
            tariff = tariff.rename(columns={province_col: "province"})

        if dso_col is not None and dso_col != "dso":
            tariff = tariff.rename(columns={dso_col: "dso"})

        if "dso" not in tariff.columns:
            tariff["dso"] = tariff["province"].map(PROVINCE_DSO)

        if year_col is not None and year_col != "year":
            tariff = tariff.rename(columns={year_col: "year"})

        if "year" not in tariff.columns:
            tariff["year"] = TARIFF_REFERENCE_YEAR

        rename_candidates = {
            "tariff_fixed_eur_year": [
                "fixed_eur_year",
                "tariff_fixed_eur_year",
                "F_eur_year",
                "F",
            ],
            "tariff_capacity_eur_mw_year": [
                "capacity_eur_mw_year",
                "tariff_capacity_eur_mw_year",
                "tau_MWc_eur_mw_year",
                "tau_mwc",
                "tau_MWc",
                "tau_mwc_eur_per_mw_year",
            ],
            "tariff_peak_eur_mw_month": [
                "peak_eur_mw_month",
                "tariff_peak_eur_mw_month",
                "tau_MWmax_eur_mw_month",
                "tau_mwmax",
                "tau_MWmax",
                "tau_mwmax_eur_per_mw_month",
            ],
            "tariff_energy_high_eur_mwh": [
                "energy_high_eur_mwh",
                "tariff_energy_high_eur_mwh",
                "tau_kwh_high_eur_mwh",
                "tau_energy_high",
                "tau_kwh_h_eur_per_mwh",
            ],
            "tariff_energy_low_eur_mwh": [
                "energy_low_eur_mwh",
                "tariff_energy_low_eur_mwh",
                "tau_kwh_low_eur_mwh",
                "tau_energy_low",
                "tau_kwh_l_eur_per_mwh",
            ],
        }

        for target, cands in rename_candidates.items():
            col = pick_column(tariff.columns, cands, required=False)
            if col is None:
                raise ValueError(f"Tariff table missing required numeric column for {target}.")
            if col != target:
                tariff = tariff.rename(columns={col: target})

        for c in rename_candidates:
            tariff[c] = pd.to_numeric(tariff[c], errors="coerce")
            if tariff[c].isna().any():
                raise ValueError(f"Tariff table has NaN values in {c}.")

        tariff["province"] = tariff["province"].astype(str).str.strip()
        tariff["dso"] = tariff["dso"].astype(str).str.strip()
        tariff["year"] = pd.to_numeric(tariff["year"], errors="coerce").astype(int)

        if "gv_class" not in tariff.columns:
            tariff["gv_class"] = "user-supplied"

        tariff["tariff_source"] = "user_supplied_tariff_table"

        rows = []

        for y in years:
            for p in PROVINCES:
                sub = tariff[tariff["province"] == p].copy()

                if sub.empty:
                    raise ValueError(f"Tariff table missing province={p}.")

                exact = sub[sub["year"] == int(y)]

                if exact.empty:
                    sub["_gap"] = (sub["year"] - int(y)).abs()
                    row = sub.sort_values(["_gap", "year"]).iloc[0].copy()
                    row["tariff_reference_year"] = int(row["year"])
                    row["year"] = int(y)
                    row["tariff_source"] = "user_supplied_nearest_year"
                else:
                    row = exact.iloc[0].copy()
                    row["tariff_reference_year"] = int(row["year"])
                    row["year"] = int(y)

                rows.append(row)

        out = pd.DataFrame(rows)

    else:
        rows = []

        for y in years:
            for p in PROVINCES:
                dso = PROVINCE_DSO.get(p)

                if dso not in DEFAULT_DSO_TARIFFS_2026:
                    raise ValueError(f"No embedded DSO tariff proxy for province={p}, dso={dso!r}.")

                t = DEFAULT_DSO_TARIFFS_2026[dso]

                rows.append(
                    {
                        "year": int(y),
                        "province": p,
                        "dso": dso,
                        "gv_class": t["gv_class"],
                        "tariff_fixed_eur_year": float(t["fixed_eur_year"]),
                        "tariff_capacity_eur_mw_year": float(t["capacity_eur_mw_year"]),
                        "tariff_peak_eur_mw_month": float(t["peak_eur_mw_month"]),
                        "tariff_energy_high_eur_mwh": float(t["energy_high_eur_mwh"]),
                        "tariff_energy_low_eur_mwh": float(t["energy_low_eur_mwh"]),
                        "tariff_reference_year": int(TARIFF_REFERENCE_YEAR),
                        "tariff_source": "embedded_representative_2026_dso_transport_proxy",
                    }
                )

        out = pd.DataFrame(rows)

    out = out[
        [
            "year",
            "province",
            "dso",
            "gv_class",
            "tariff_reference_year",
            "tariff_source",
            "tariff_fixed_eur_year",
            "tariff_capacity_eur_mw_year",
            "tariff_peak_eur_mw_month",
            "tariff_energy_high_eur_mwh",
            "tariff_energy_low_eur_mwh",
        ]
    ].copy()

    out["firm_capacity_multiplier"] = float(FIRM_CAPACITY_MULTIPLIER)
    out["tdtr_capacity_multiplier"] = float(TDTR_CAPACITY_MULTIPLIER)

    out["firm_capacity_charge_eur_mw_year"] = (
        out["tariff_capacity_eur_mw_year"] * out["firm_capacity_multiplier"]
    )

    out["tdtr_capacity_charge_eur_mw_year"] = (
        out["tariff_capacity_eur_mw_year"] * out["tdtr_capacity_multiplier"]
    )

    out["tdtr_capacity_discount_eur_mw_year"] = (
        out["firm_capacity_charge_eur_mw_year"]
        - out["tdtr_capacity_charge_eur_mw_year"]
    )

    out["tau_mwc_eur_per_mw_year"] = out["tariff_capacity_eur_mw_year"]
    out["tau_mwmax_eur_per_mw_month"] = out["tariff_peak_eur_mw_month"]
    out["tau_kwh_h_eur_per_mwh"] = out["tariff_energy_high_eur_mwh"]
    out["tau_kwh_l_eur_per_mwh"] = out["tariff_energy_low_eur_mwh"]

    return out.sort_values(["year", "province"]).reset_index(drop=True)


def load_participant_capacity_overrides() -> Optional[pd.DataFrame]:
    if PARTICIPANT_CAPACITY_PATH is None:
        return None

    require_file(PARTICIPANT_CAPACITY_PATH, "participant capacity override table")

    df = pd.read_csv(PARTICIPANT_CAPACITY_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    province_col = pick_column(df.columns, ["province", "provincie", "node"])
    year_col = pick_column(df.columns, ["year"], required=False)
    p_col = pick_column(df.columns, ["P_ctr_MW", "Pctr_MW", "contracted_capacity_MW", "P_i", "P"])

    if province_col != "province":
        df = df.rename(columns={province_col: "province"})

    if p_col != "P_ctr_MW":
        df = df.rename(columns={p_col: "P_ctr_MW"})

    if year_col is not None and year_col != "year":
        df = df.rename(columns={year_col: "year"})

    df["province"] = df["province"].astype(str).str.strip()
    df["P_ctr_MW"] = pd.to_numeric(df["P_ctr_MW"], errors="coerce")

    if df["P_ctr_MW"].isna().any() or (df["P_ctr_MW"] <= 0).any():
        raise ValueError("Participant capacity override contains invalid P_ctr_MW values.")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)

    return df


# =============================================================================
# LOADERS
# =============================================================================

def load_state_panel() -> pd.DataFrame:
    require_file(STATE_PANEL_CSV, "auditable state panel")

    df = pd.read_csv(STATE_PANEL_CSV, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    province_col = pick_column(df.columns, ["province", "provincie"])
    ts_col = pick_column(
        df.columns,
        ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc", "timestamp"],
        required=False,
    )

    df["province"] = df[province_col].astype(str).str.strip()

    if ts_col is not None:
        df["timestamp_utc"] = (
            pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            .dt.floor("h")
        )
    else:
        y_col = pick_column(df.columns, ["year"])
        h_col = pick_column(df.columns, ["hour_of_year", "hour", "hoy"])

        y = pd.to_numeric(df[y_col], errors="coerce").astype("Int64")
        h = pd.to_numeric(df[h_col], errors="coerce").astype("Int64")

        df["timestamp_utc"] = (
            pd.to_datetime(y.astype(str) + "-01-01T00:00:00Z", utc=True)
            + pd.to_timedelta(h.astype(float) - 1.0, unit="h")
        )

    df = df.dropna(subset=["timestamp_utc"]).copy()
    df["year"] = df["timestamp_utc"].dt.year.astype(int)

    h_col = pick_column(df.columns, ["hour_of_year", "hour", "hoy"], required=False)

    if h_col is not None:
        df["hour_of_year"] = pd.to_numeric(df[h_col], errors="coerce").astype("Int64")
    else:
        jan1 = pd.to_datetime(df["year"].astype(str) + "-01-01T00:00:00Z", utc=True)
        df["hour_of_year"] = ((df["timestamp_utc"] - jan1).dt.total_seconds() // 3600).astype(int) + 1

    df["hour_of_year"] = df["hour_of_year"].astype(int)

    load_col = pick_column(
        df.columns,
        [
            "ptdf_load_pressure_MW",
            "prov_load_MW",
            "RL_MW",
            "residual_load_MW",
            "rl_mw",
            "nat_load_MW",
        ],
        required=False,
    )

    if load_col is None:
        raise ValueError(
            "State panel needs ptdf_load_pressure_MW, prov_load_MW, RL_MW, "
            "or similar load-pressure column."
        )

    df["ptdf_load_pressure_MW"] = (
        pd.to_numeric(df[load_col], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    price_col = pick_column(
        df.columns,
        [
            "price_da_eur_per_mwh",
            "price_da_eur_mwh",
            "day_ahead_price_eur_per_mwh",
            "da_price_eur_per_mwh",
            "price_eur_per_mwh",
            "electricity_price_eur_mwh",
            "price",
        ],
        required=True,
    )

    df["price_da_eur_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
    df["price_da_eur_per_mwh"] = df["price_da_eur_mwh"]

    if df["price_da_eur_mwh"].isna().any():
        raise ValueError(
            "The day-ahead price column contains missing/non-numeric values. "
            "Fix the state panel before running A."
        )

    if "prov_load_MW" not in df.columns:
        df["prov_load_MW"] = df["ptdf_load_pressure_MW"]
    else:
        df["prov_load_MW"] = (
            pd.to_numeric(df["prov_load_MW"], errors="coerce")
            .fillna(df["ptdf_load_pressure_MW"])
            .astype(float)
        )

    defaults = {
        "price_da_norm": np.nan,
        "rl_plus_norm": 0.0,
        "rl_minus_norm": 0.0,
        "s_plus_norm": 0.0,
        "s_minus_norm": 0.0,
        "structural_exposure": 1.0 / max(len(PROVINCES), 1),
        "is_holiday": 0,
        "hour_of_week": np.nan,
    }

    for c, default in defaults.items():
        if c not in df.columns:
            df[c] = default

        if c != "is_holiday":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default).astype(float)

    if df["price_da_norm"].isna().all():
        df["price_da_norm"] = (
            df.groupby("year", sort=False)["price_da_eur_mwh"]
            .transform(minmax01)
            .astype(float)
        )
    else:
        df["price_da_norm"] = (
            pd.to_numeric(df["price_da_norm"], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

    df["month"] = df["timestamp_utc"].dt.month.astype(int)
    df["hour_of_day_utc"] = df["timestamp_utc"].dt.hour.astype(int)
    df["weekday_utc"] = df["timestamp_utc"].dt.weekday.astype(int)
    df["is_weekend"] = df["weekday_utc"].isin([5, 6]).astype(int)
    df["hour_of_week"] = df["weekday_utc"] * 24 + df["hour_of_day_utc"]

    return df


def validate_state_year(state: pd.DataFrame, support_y: pd.DataFrame, year: int) -> pd.DataFrame:
    H = hours_in_year_utc(year)

    sub = state.loc[(state["year"] == year) & (state["province"].isin(PROVINCES))].copy()
    expected_rows = H * len(PROVINCES)

    if len(sub) != expected_rows:
        raise ValueError(
            f"State panel year {year}: rows={len(sub):,}, expected={expected_rows:,}."
        )

    pieces = []
    expected = pd.date_range(pd.Timestamp(f"{year}-01-01T00:00:00Z"), periods=H, freq="h")

    for p, g in sub.groupby("province", sort=False):
        gg = g.sort_values("timestamp_utc").copy()

        if len(gg) != H or not pd.DatetimeIndex(gg["timestamp_utc"]).equals(expected):
            raise ValueError(
                f"State panel year={year}, province={p} is not a complete UTC hourly year."
            )

        gg["hour_of_year"] = np.arange(1, H + 1, dtype=int)
        pieces.append(gg)

    out = pd.concat(pieces, ignore_index=True)

    out = out.merge(
        support_y[["year", "hour_of_year", "tdtr_support"]],
        on=["year", "hour_of_year"],
        how="left",
        validate="many_to_one",
    )

    out["tdtr_support"] = out["tdtr_support"].fillna(0).astype(int)

    return out.sort_values(["province", "hour_of_year"]).reset_index(drop=True)


def load_h_outputs(year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    req_path = require_file(
        PTDF_DEBIT_INPUT_DIR / f"branch_hour_requirements_{year}.csv.gz",
        f"H requirements for {year}",
    )
    coef_path = require_file(
        PTDF_DEBIT_INPUT_DIR / f"branch_hour_province_coefficients_{year}.csv.gz",
        f"H coefficients for {year}",
    )
    active_path = require_file(
        PTDF_DEBIT_INPUT_DIR / f"active_branches_{year}.csv",
        f"H active branch table for {year}",
    )

    req = pd.read_csv(req_path, compression="gzip", low_memory=False)
    coef = pd.read_csv(coef_path, compression="gzip", low_memory=False)
    active = pd.read_csv(active_path, low_memory=False)

    req.columns = [str(c).strip() for c in req.columns]
    coef.columns = [str(c).strip() for c in coef.columns]
    active.columns = [str(c).strip() for c in active.columns]

    req_branch = pick_column(req.columns, ["branch_id", "branch", "line_id"])
    req_ts = pick_column(req.columns, ["timestamp_utc", "timestamp", "time", "utc_time"])
    req_h = pick_column(req.columns, ["hour_of_year", "hour", "hoy"])
    req_R = pick_column(
        req.columns,
        ["R_MW", "R_value", "requirement_MW", "relief_requirement_MW", "requirement", "R"],
    )

    coef_branch = pick_column(coef.columns, ["branch_id", "branch", "line_id"])
    coef_ts = pick_column(coef.columns, ["timestamp_utc", "timestamp", "time", "utc_time"])
    coef_h = pick_column(coef.columns, ["hour_of_year", "hour", "hoy"])
    coef_prov = pick_column(coef.columns, ["province", "provincie", "node"])
    coef_a = pick_column(
        coef.columns,
        [
            "coefficient",
            "a_value",
            "coefficient_value",
            "ptdf_coefficient",
            "relief_coefficient",
            "A_value",
            "a",
        ],
    )

    req = req.rename(
        columns={
            req_branch: "branch_id",
            req_ts: "timestamp_utc",
            req_h: "hour_of_year",
            req_R: "R_MW",
        }
    )

    coef = coef.rename(
        columns={
            coef_branch: "branch_id",
            coef_ts: "timestamp_utc",
            coef_h: "hour_of_year",
            coef_prov: "province",
            coef_a: "coefficient",
        }
    )

    req["branch_id"] = req["branch_id"].astype(str)
    coef["branch_id"] = coef["branch_id"].astype(str)

    req["timestamp_utc"] = (
        pd.to_datetime(req["timestamp_utc"], utc=True, errors="coerce")
        .dt.floor("h")
    )
    coef["timestamp_utc"] = (
        pd.to_datetime(coef["timestamp_utc"], utc=True, errors="coerce")
        .dt.floor("h")
    )

    req["hour_of_year"] = pd.to_numeric(req["hour_of_year"], errors="coerce").astype(int)
    coef["hour_of_year"] = pd.to_numeric(coef["hour_of_year"], errors="coerce").astype(int)

    req["R_MW"] = pd.to_numeric(req["R_MW"], errors="coerce").fillna(0.0).astype(float)
    coef["coefficient"] = pd.to_numeric(coef["coefficient"], errors="coerce").fillna(0.0).astype(float)
    coef["province"] = coef["province"].astype(str).str.strip()

    req = req.loc[req["R_MW"] > MIN_REQUIREMENT_MW].copy()
    coef = coef.loc[
        (coef["coefficient"].abs() > MIN_COEFFICIENT)
        & (coef["province"].isin(PROVINCES))
    ].copy()

    return req, coef, active


# =============================================================================
# TRANSFORMERS
# =============================================================================

def apply_support_to_requirements_and_coefficients(
    req: pd.DataFrame,
    coef: pd.DataFrame,
    support_y: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Defensively restrict branch-hour requirements and coefficients to the
    selected TDTR candidate-hour support.

    This function is intentionally idempotent:
      - if H already wrote support-filtered requirements, applying this again
        should not change them materially;
      - if H also wrote a tdtr_support column, we remove/ignore it before
        merging the authoritative support_y mask from G/A.
    """
    if req.empty or coef.empty or not TDTR_SUPPORT_APPLY_TO_REQUIREMENTS:
        return req, coef, {
            "support_applied_to_requirements": False,
            "requirements_before_support": int(len(req)),
            "coefficients_before_support": int(len(coef)),
            "requirements_after_support": int(len(req)),
            "coefficients_after_support": int(len(coef)),
            "dropped_requirement_rows_outside_support": 0,
            "total_R_before_support_MWh_proxy": float(req["R_MW"].sum()) if len(req) and "R_MW" in req.columns else 0.0,
            "total_R_after_support_MWh_proxy": float(req["R_MW"].sum()) if len(req) and "R_MW" in req.columns else 0.0,
        }

    before_req = int(len(req))
    before_coef = int(len(coef))
    before_R = float(req["R_MW"].sum()) if len(req) and "R_MW" in req.columns else 0.0

    support_key = (
        support_y[["hour_of_year", "tdtr_support"]]
        .drop_duplicates("hour_of_year")
        .copy()
    )

    support_key["hour_of_year"] = pd.to_numeric(
        support_key["hour_of_year"],
        errors="coerce",
    ).astype(int)

    support_key["tdtr_support"] = (
        pd.to_numeric(support_key["tdtr_support"], errors="coerce")
        .fillna(0)
        .astype(float)
        .gt(0.5)
        .astype(int)
    )

    req_work = req.copy()
    req_work = req_work.drop(
        columns=[
            c
            for c in ["tdtr_support", "tdtr_support_x", "tdtr_support_y"]
            if c in req_work.columns
        ],
        errors="ignore",
    )

    req_work["hour_of_year"] = pd.to_numeric(
        req_work["hour_of_year"],
        errors="coerce",
    ).astype(int)

    req2 = req_work.merge(
        support_key,
        on="hour_of_year",
        how="left",
        validate="many_to_one",
    )

    req2["tdtr_support"] = (
        pd.to_numeric(req2["tdtr_support"], errors="coerce")
        .fillna(0)
        .astype(float)
        .gt(0.5)
        .astype(int)
    )

    req2 = (
        req2.loc[(req2["tdtr_support"] == 1) & (req2["R_MW"] > MIN_REQUIREMENT_MW)]
        .drop(columns=["tdtr_support"])
        .copy()
    )

    key_cols = ["branch_id", "timestamp_utc", "hour_of_year"]

    for c in key_cols:
        if c not in req2.columns:
            raise KeyError(f"Requirement table is missing key column {c!r}.")
        if c not in coef.columns:
            raise KeyError(f"Coefficient table is missing key column {c!r}.")

    keep_keys = req2[key_cols].drop_duplicates()

    coef_work = coef.copy()
    coef_work["hour_of_year"] = pd.to_numeric(
        coef_work["hour_of_year"],
        errors="coerce",
    ).astype(int)

    coef2 = coef_work.merge(
        keep_keys,
        on=key_cols,
        how="inner",
        validate="many_to_one",
    )

    after_R = float(req2["R_MW"].sum()) if len(req2) else 0.0

    meta = {
        "support_applied_to_requirements": True,
        "requirements_before_support": before_req,
        "coefficients_before_support": before_coef,
        "requirements_after_support": int(len(req2)),
        "coefficients_after_support": int(len(coef2)),
        "total_R_before_support_MWh_proxy": before_R,
        "total_R_after_support_MWh_proxy": after_R,
        "dropped_requirement_rows_outside_support": int(before_req - len(req2)),
        "support_collision_handled": True,
    }

    return req2, coef2, meta


def build_requirements_and_coefficients(
    req: pd.DataFrame,
    coef: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if req.empty or coef.empty:
        return (
            pd.DataFrame(
                columns=[
                    "constraint_id",
                    "branch_id",
                    "timestamp_utc",
                    "hour_of_year",
                    "R_MW",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "constraint_id",
                    "branch_id",
                    "timestamp_utc",
                    "hour_of_year",
                    "province",
                    "coefficient",
                ]
            ),
        )

    req = req.copy().sort_values(["timestamp_utc", "branch_id", "hour_of_year"]).reset_index(drop=True)

    agg_spec = {"R_MW": ("R_MW", "sum")}

    for c in ["raw_excess_MW", "raw_flow_MW", "loading_frac", "requirement_scale_factor"]:
        if c in req.columns:
            agg_spec[c] = (c, "mean" if c != "raw_excess_MW" else "sum")

    req = req.groupby(
        ["timestamp_utc", "branch_id", "hour_of_year"],
        as_index=False,
    ).agg(**agg_spec)

    req = req.loc[req["R_MW"] > MIN_REQUIREMENT_MW].copy()
    req["constraint_id"] = np.arange(len(req), dtype=int)

    key_cols = ["branch_id", "timestamp_utc", "hour_of_year"]

    coeff = coef.merge(
        req[key_cols + ["constraint_id"]],
        on=key_cols,
        how="inner",
        validate="many_to_one",
    )

    coeff = (
        coeff[
            [
                "constraint_id",
                "branch_id",
                "timestamp_utc",
                "hour_of_year",
                "province",
                "coefficient",
            ]
        ]
        .groupby(
            ["constraint_id", "branch_id", "timestamp_utc", "hour_of_year", "province"],
            as_index=False,
        )["coefficient"]
        .sum()
    )

    coeff = coeff.loc[coeff["coefficient"].abs() > MIN_COEFFICIENT].copy()

    has_coef = set(coeff["constraint_id"].unique().tolist())
    req = req.loc[req["constraint_id"].isin(has_coef)].copy()

    remap = {old: new for new, old in enumerate(req["constraint_id"].tolist())}

    req["constraint_id_old"] = req["constraint_id"].astype(int)
    req["constraint_id"] = req["constraint_id_old"].map(remap).astype(int)

    coeff = coeff.loc[coeff["constraint_id"].isin(remap)].copy()
    coeff["constraint_id_old"] = coeff["constraint_id"].astype(int)
    coeff["constraint_id"] = coeff["constraint_id_old"].map(remap).astype(int)

    keep_req = [
        "constraint_id",
        "constraint_id_old",
        "branch_id",
        "timestamp_utc",
        "hour_of_year",
        "R_MW",
        "raw_excess_MW",
        "raw_flow_MW",
        "loading_frac",
        "requirement_scale_factor",
    ]

    req = req[[c for c in keep_req if c in req.columns]].sort_values("constraint_id").reset_index(drop=True)

    keep_coeff = [
        "constraint_id",
        "constraint_id_old",
        "branch_id",
        "timestamp_utc",
        "hour_of_year",
        "province",
        "coefficient",
    ]

    coeff = coeff[keep_coeff].sort_values(["constraint_id", "province"]).reset_index(drop=True)

    return req, coeff


def build_effectivity_features(
    state_y: pd.DataFrame,
    req: pd.DataFrame,
    coeff: pd.DataFrame,
) -> pd.DataFrame:
    base = state_y[["province", "hour_of_year"]].drop_duplicates().copy()

    if req.empty or coeff.empty:
        base["ptdf_effectivity_raw"] = 0.0
        base["ptdf_effectivity_norm"] = 0.0
        base["ptdf_requirement_exposure_MW"] = 0.0
        base["ptdf_active_constraints"] = 0
        return base

    tmp = coeff.copy()

    tmp = tmp.merge(
        req[["constraint_id", "R_MW"]],
        on="constraint_id",
        how="left",
        validate="many_to_one",
    )

    tmp["hour_of_year"] = pd.to_numeric(tmp["hour_of_year"], errors="coerce").astype(int)
    tmp["R_MW"] = pd.to_numeric(tmp["R_MW"], errors="coerce").fillna(0.0).astype(float)
    tmp["coefficient"] = pd.to_numeric(tmp["coefficient"], errors="coerce").fillna(0.0).astype(float)

    tmp["weighted_requirement"] = tmp["coefficient"] * tmp["R_MW"]

    feat = tmp.groupby(["province", "hour_of_year"], as_index=False).agg(
        ptdf_effectivity_raw=("coefficient", "sum"),
        ptdf_requirement_exposure_MW=("weighted_requirement", "sum"),
        ptdf_active_constraints=("constraint_id", "nunique"),
    )

    full = base.merge(feat, on=["province", "hour_of_year"], how="left")

    for c in ["ptdf_effectivity_raw", "ptdf_requirement_exposure_MW", "ptdf_active_constraints"]:
        full[c] = pd.to_numeric(full[c], errors="coerce").fillna(0.0)

    full["ptdf_effectivity_norm"] = (
        full.groupby("province", sort=False)["ptdf_requirement_exposure_MW"]
        .transform(minmax01)
        .astype(float)
    )

    return full


def tdtr_budgets_for_year(year: int, selected_support_hours: int) -> tuple[float, int, int, dict]:
    annual_hours = hours_in_year_utc(year)
    selected_support_hours = int(selected_support_hours)

    if selected_support_hours <= 0:
        raise ValueError(
            f"Year {year}: selected_support_hours={selected_support_hours}. "
            "Cannot build TDTR budgets without a positive candidate-hour support."
        )

    if USE_TDTR85_BUDGETS:
        if TOUCH_BUDGET_FROM_SUPPORT:
            touch_budget = int(selected_support_hours)
            touch_budget_mode = "support_hours"
        else:
            touch_budget = int(math.floor(TDTR_RESTRICTION_SHARE * annual_hours))
            touch_budget_mode = "floor_restriction_share_times_annual_hours"
    else:
        touch_budget = int(TOUCH_BUDGET_HOURS_FALLBACK)
        touch_budget_mode = "fixed_fallback"

    mode = DEPTH_BUDGET_MODE

    if mode in {"rho_support", "share_of_support", "rho_times_support"}:
        depth_budget = float(DEPTH_BUDGET_RHO * selected_support_hours)
        depth_budget_mode = "rho_times_support_hours"

    elif mode in {"support_hours", "legacy_support", "full_support"}:
        depth_budget = float(selected_support_hours)
        depth_budget_mode = "support_hours_legacy"

    elif mode in {"rho_annual", "share_of_annual", "rho_times_annual"}:
        depth_budget = float(DEPTH_BUDGET_RHO * annual_hours)
        depth_budget_mode = "rho_times_annual_hours"

    elif mode in {"fixed", "fixed_hours"}:
        depth_budget = float(DEPTH_BUDGET_HOURS_FALLBACK)
        depth_budget_mode = "fixed_hours"

    else:
        raise ValueError(
            f"Unknown DEBIT_DEPTH_BUDGET_MODE={DEPTH_BUDGET_MODE!r}. "
            "Use one of: rho_support, support_hours, rho_annual, fixed."
        )

    if depth_budget < 0:
        raise ValueError(f"Depth budget is negative for year={year}: {depth_budget}")

    meta = {
        "annual_hours": int(annual_hours),
        "selected_support_hours": int(selected_support_hours),
        "tdtr_availability_share": float(TDTR_AVAILABILITY_SHARE),
        "tdtr_restriction_share": float(TDTR_RESTRICTION_SHARE),
        "touch_budget_mode": touch_budget_mode,
        "touch_budget_tokens": int(touch_budget),
        "depth_budget_mode": depth_budget_mode,
        "depth_budget_rho": float(DEPTH_BUDGET_RHO),
        "depth_budget_hours": float(depth_budget),
        "depth_budget_as_share_of_support": float(depth_budget / max(selected_support_hours, 1)),
        "depth_budget_as_share_of_annual": float(depth_budget / max(annual_hours, 1)),
    }

    return depth_budget, touch_budget, annual_hours, meta


def build_participants(
    state_y: pd.DataFrame,
    year: int,
    capacity_overrides: Optional[pd.DataFrame],
    selected_support_hours: int,
) -> pd.DataFrame:
    depth_budget, touch_budget, annual_hours, budget_meta = tdtr_budgets_for_year(
        year,
        selected_support_hours,
    )

    rows = []

    for p, g in state_y.groupby("province", sort=False):
        P_source = "constructed_representative_public_pressure"

        pressure = (
            pd.to_numeric(g["ptdf_load_pressure_MW"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )

        q99 = float(pressure.quantile(0.99))
        q95 = float(pressure.quantile(0.95))
        mean = float(pressure.mean())

        P_override = None

        if capacity_overrides is not None:
            sub = capacity_overrides[capacity_overrides["province"] == p].copy()

            if "year" in sub.columns:
                exact = sub[sub["year"].astype(int) == int(year)]
                if not exact.empty:
                    sub = exact

            if not sub.empty:
                P_override = float(sub.iloc[0]["P_ctr_MW"])

        if P_override is not None:
            P = P_override
            P_source = "user_supplied_capacity"
        else:
            if CONTRACT_CAPACITY_MODE == "fraction_of_p99_pressure":
                P = CONTRACT_CAPACITY_FRACTION * q99

            elif CONTRACT_CAPACITY_MODE == "fraction_of_p95_pressure":
                P = CONTRACT_CAPACITY_FRACTION * q95

            elif CONTRACT_CAPACITY_MODE == "fraction_of_mean_pressure":
                P = CONTRACT_CAPACITY_FRACTION * mean

            elif CONTRACT_CAPACITY_MODE == "fixed":
                P = float(cfg_get("DEBIT_CONTRACT_CAPACITY_FIXED_MW", 250.0))
                P_source = "fixed_config_capacity"

            else:
                raise ValueError(
                    f"Unknown DEBIT_CONTRACT_CAPACITY_MODE={CONTRACT_CAPACITY_MODE!r}"
                )

            P = max(CONTRACT_CAPACITY_MIN_MW, float(P))

            if CONTRACT_CAPACITY_MAX_MW is not None:
                P = min(CONTRACT_CAPACITY_MAX_MW, P)

        rows.append(
            {
                "year": int(year),
                "province": p,
                "P_ctr_MW": float(P),
                "P_i": float(P),
                "Pctr_MW": float(P),
                "Pctr_mw": float(P),

                # Weighted-depth allowance used by M2/M3:
                "H_depth_hours": float(depth_budget),
                "depth_budget_hours": float(depth_budget),
                "H_i": float(depth_budget),
                "H": float(depth_budget),

                # Touched-time allowance used by M1:
                "H_touch_tokens": int(touch_budget),
                "touched_budget_tokens": int(touch_budget),
                "Hbar_i": int(touch_budget),
                "Hbar": int(touch_budget),

                # Slack penalty written explicitly so B/B1 do not fall back silently.
                "slack_penalty": float(SLACK_PENALTY),
                "M": float(SLACK_PENALTY),
                "emergency_slack_penalty": float(SLACK_PENALTY),

                "tdtr_availability_share": float(TDTR_AVAILABILITY_SHARE),
                "tdtr_restriction_share": float(TDTR_RESTRICTION_SHARE),
                "annual_hours": int(annual_hours),
                "tdtr_support_hours": int(selected_support_hours),
                "depth_budget_rho": float(DEPTH_BUDGET_RHO),
                "depth_budget_mode": budget_meta["depth_budget_mode"],
                "depth_budget_as_share_of_support": budget_meta["depth_budget_as_share_of_support"],
                "depth_budget_as_share_of_annual": budget_meta["depth_budget_as_share_of_annual"],
                "touch_budget_mode": budget_meta["touch_budget_mode"],

                "pressure_q99_MW": q99,
                "pressure_q95_MW": q95,
                "pressure_mean_MW": mean,
                "contract_capacity_mode": CONTRACT_CAPACITY_MODE,
                "contract_capacity_fraction": CONTRACT_CAPACITY_FRACTION,
                "contract_capacity_source": P_source,
            }
        )

    return pd.DataFrame(rows).sort_values("province").reset_index(drop=True)


def add_public_economic_and_cost_signals(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()

    out["price_da_eur_mwh"] = safe_numeric(out, "price_da_eur_mwh", 0.0)
    out["day_ahead_price_eur_per_mwh"] = out["price_da_eur_mwh"]

    out["tariff_capacity_eur_mw_year"] = safe_numeric(out, "tariff_capacity_eur_mw_year", 0.0)
    out["tariff_peak_eur_mw_month"] = safe_numeric(out, "tariff_peak_eur_mw_month", 0.0)
    out["tariff_energy_high_eur_mwh"] = safe_numeric(out, "tariff_energy_high_eur_mwh", 0.0)
    out["tariff_energy_low_eur_mwh"] = safe_numeric(
        out,
        "tariff_energy_low_eur_mwh",
        out["tariff_energy_high_eur_mwh"],
    )

    out["tau_mwc_eur_per_mw_year"] = out["tariff_capacity_eur_mw_year"]
    out["tau_mwmax_eur_per_mw_month"] = out["tariff_peak_eur_mw_month"]
    out["tau_kwh_h_eur_per_mwh"] = out["tariff_energy_high_eur_mwh"]
    out["tau_kwh_l_eur_per_mwh"] = out["tariff_energy_low_eur_mwh"]

    out["tariff_energy_mean_eur_mwh"] = (
        0.5 * (out["tariff_energy_high_eur_mwh"] + out["tariff_energy_low_eur_mwh"])
    )

    annual_hours = out.groupby(["province", "year"])["hour_of_year"].transform("nunique").astype(float)
    median_price = out.groupby(["province", "year"], sort=False)["price_da_eur_mwh"].transform("median").astype(float)

    price_extreme = (out["price_da_eur_mwh"] - median_price).abs()

    if 0.0 < PRICE_WINSOR_Q < 1.0:
        cap = float(price_extreme.quantile(PRICE_WINSOR_Q))
        price_extreme = price_extreme.clip(upper=cap)

    out["cost_component_price_timing_eur_per_mwh"] = price_extreme
    out["cost_component_energy_tariff_eur_per_mwh"] = out["tariff_energy_mean_eur_mwh"]
    out["cost_component_capacity_tariff_eur_per_mwh"] = (
        out["tariff_capacity_eur_mw_year"] / annual_hours
    )

    out["restriction_cost"] = (
        out["cost_component_price_timing_eur_per_mwh"]
        + out["cost_component_energy_tariff_eur_per_mwh"]
        + out["cost_component_capacity_tariff_eur_per_mwh"]
    ).clip(lower=RESTRICTION_COST_MIN)

    out["cost_eur_per_mwh"] = out["restriction_cost"]
    out["c_value"] = out["restriction_cost"]

    out["restriction_cost_source"] = (
        "dutch_public_proxy_abs_DA_minus_province_year_median_plus_"
        "DSO_energy_tariff_plus_DSO_capacity_tariff_per_hour"
    )

    for c in [
        "rl_plus_norm",
        "rl_minus_norm",
        "s_plus_norm",
        "s_minus_norm",
        "structural_exposure",
        "ptdf_effectivity_norm",
        "ptdf_requirement_exposure_MW",
        "ptdf_active_constraints",
    ]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)

    out["price_high_signal_norm"] = (
        out.groupby("year", sort=False)["price_da_eur_mwh"]
        .transform(minmax01)
        .astype(float)
    )

    out["price_low_signal_norm"] = (
        out.groupby("year", sort=False)["price_da_eur_mwh"]
        .transform(lambda s: minmax01(-pd.to_numeric(s, errors="coerce").fillna(0.0)))
        .astype(float)
    )

    out["tariff_capacity_signal_norm"] = (
        out.groupby("year", sort=False)["tariff_capacity_eur_mw_year"]
        .transform(minmax01)
        .astype(float)
    )

    out["tariff_energy_signal_norm"] = (
        out.groupby("year", sort=False)["tariff_energy_mean_eur_mwh"]
        .transform(minmax01)
        .astype(float)
    )

    out["tariff_peak_signal_norm"] = (
        out.groupby("year", sort=False)["tariff_peak_eur_mw_month"]
        .transform(minmax01)
        .astype(float)
    )

    out["tdtr_discount_signal_norm"] = (
        out.groupby("year", sort=False)["tdtr_capacity_discount_eur_mw_year"]
        .transform(minmax01)
        .astype(float)
    )

    out["import_burden_signal_norm"] = (
        0.55 * out["price_low_signal_norm"]
        + 0.25 * out["tariff_energy_signal_norm"]
        + 0.20 * out["tariff_peak_signal_norm"]
    )

    out["export_burden_signal_norm"] = (
        0.60 * out["price_high_signal_norm"]
        + 0.25 * out["tariff_energy_signal_norm"]
        + 0.15 * out["tariff_capacity_signal_norm"]
    )

    out["burden_proxy"] = (
        0.30 * out["import_burden_signal_norm"]
        + 0.30 * out["export_burden_signal_norm"]
        + 0.20 * out["structural_exposure"]
        + 0.10 * out["s_minus_norm"]
        + 0.10 * (1.0 - out["ptdf_effectivity_norm"])
    )

    active_constraints_norm = (
        out.groupby(["province", "year"], sort=False)["ptdf_active_constraints"]
        .transform(minmax01)
        .astype(float)
    )

    out["relief_value_proxy"] = (
        0.50 * out["ptdf_effectivity_norm"]
        + 0.25 * out["rl_plus_norm"]
        + 0.15 * out["s_plus_norm"]
        + 0.10 * active_constraints_norm
    )

    out["burden_proxy_norm"] = (
        out.groupby(["province", "year"], sort=False)["burden_proxy"]
        .transform(minmax01)
        .astype(float)
    )

    out["relief_value_proxy_norm"] = (
        out.groupby(["province", "year"], sort=False)["relief_value_proxy"]
        .transform(minmax01)
        .astype(float)
    )

    out["restriction_cost_norm"] = (
        out.groupby(["province", "year"], sort=False)["restriction_cost"]
        .transform(minmax01)
        .astype(float)
    )

    out["ptdf_exposure_norm"] = (
        out.groupby(["province", "year"], sort=False)["ptdf_requirement_exposure_MW"]
        .transform(minmax01)
        .astype(float)
    )

    out["B_proxy_baseline"] = out["burden_proxy_norm"]
    out["B_proxy"] = out["burden_proxy_norm"]
    out["relief_proxy_baseline"] = out["relief_value_proxy_norm"]

    return out


def build_state_for_optimization(
    state_y: pd.DataFrame,
    participants: pd.DataFrame,
    effectivity: pd.DataFrame,
    tariff_table: pd.DataFrame,
) -> pd.DataFrame:
    keep = [
        "province",
        "timestamp_utc",
        "year",
        "hour_of_year",
        "month",
        "hour_of_day_utc",
        "weekday_utc",
        "hour_of_week",
        "is_weekend",
        "is_holiday",
        "price_da_eur_mwh",
        "price_da_norm",
        "rl_plus_norm",
        "rl_minus_norm",
        "s_plus_norm",
        "s_minus_norm",
        "structural_exposure",
        "prov_load_MW",
        "ptdf_load_pressure_MW",
        "tdtr_support",
    ]

    keep = [c for c in keep if c in state_y.columns]
    out = state_y[keep].copy()

    out = out.merge(
        participants[
            [
                "province",
                "P_ctr_MW",
                "P_i",
                "Pctr_MW",
                "Pctr_mw",
                "H_depth_hours",
                "depth_budget_hours",
                "H_i",
                "H",
                "H_touch_tokens",
                "touched_budget_tokens",
                "Hbar_i",
                "Hbar",
                "slack_penalty",
                "M",
                "emergency_slack_penalty",
                "tdtr_availability_share",
                "tdtr_restriction_share",
                "annual_hours",
                "tdtr_support_hours",
                "depth_budget_rho",
                "depth_budget_mode",
                "touch_budget_mode",
            ]
        ],
        on="province",
        how="left",
        validate="many_to_one",
    )

    out = out.merge(
        effectivity,
        on=["province", "hour_of_year"],
        how="left",
        validate="one_to_one",
    )

    tariff_y = tariff_table[tariff_table["year"].astype(int) == int(out["year"].iloc[0])].copy()

    out = out.merge(
        tariff_y,
        on=["year", "province"],
        how="left",
        validate="many_to_one",
    )

    pbar_col = pick_column(
        out.columns,
        ["Pbar_value", "Pbar", "available_capacity_MW", "availability_MW", "P_available"],
        required=False,
    )

    if pbar_col is None:
        out["Pbar_value_raw_before_tdtr_support"] = out["P_ctr_MW"]
    else:
        out["Pbar_value_raw_before_tdtr_support"] = (
            pd.to_numeric(out[pbar_col], errors="coerce")
            .fillna(out["P_ctr_MW"])
            .astype(float)
        )

    out["Pbar_value_raw_before_tdtr_support"] = np.minimum(
        out["Pbar_value_raw_before_tdtr_support"].clip(lower=0.0),
        out["P_ctr_MW"],
    )

    out["tdtr_support"] = (
        pd.to_numeric(out.get("tdtr_support", 1), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if TDTR_SUPPORT_APPLY_TO_PBAR:
        out["Pbar_value"] = out["Pbar_value_raw_before_tdtr_support"] * out["tdtr_support"]
        out["pbar_support_policy"] = "Pbar_raw_times_tdtr_candidate_hour_support"
    else:
        out["Pbar_value"] = out["Pbar_value_raw_before_tdtr_support"]
        out["pbar_support_policy"] = "support_not_applied_to_Pbar"

    out["pbar_mw"] = out["Pbar_value"]
    out["Pctr_mw"] = out["P_ctr_MW"]

    for c in [
        "ptdf_effectivity_raw",
        "ptdf_effectivity_norm",
        "ptdf_requirement_exposure_MW",
        "ptdf_active_constraints",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    if out["P_ctr_MW"].isna().any() or (out["P_ctr_MW"] <= 0).any():
        raise ValueError("Invalid P_ctr_MW after participant merge.")

    if out["tariff_capacity_eur_mw_year"].isna().any():
        raise ValueError("Missing tariff rows after tariff merge.")

    out = add_public_economic_and_cost_signals(out)

    return out.sort_values(["province", "hour_of_year"]).reset_index(drop=True)


# =============================================================================
# MAIN YEAR BUILDER
# =============================================================================

def build_year(
    year: int,
    state: pd.DataFrame,
    support: pd.DataFrame,
    support_meta: dict,
    tariff_table: pd.DataFrame,
    capacity_overrides: Optional[pd.DataFrame],
) -> dict:
    print("\n" + "-" * 78)
    print(f"[YEAR] {year}")
    print("-" * 78)

    support_y = validate_support_year(support, year)

    selected_hours = int(support_y["tdtr_support"].sum())
    H = hours_in_year_utc(year)

    print(
        f"[TDTR SUPPORT] year={year} selected={selected_hours:,}/{H:,} "
        f"({selected_hours / H:.2%}) scenario={TDTR_SUPPORT_SCENARIO}"
    )

    depth_budget, touch_budget, H_year, budget_meta = tdtr_budgets_for_year(
        year,
        selected_hours,
    )

    print(
        f"[ALLOWANCE] M1 touched Hbar={touch_budget} hours | "
        f"M2/M3 depth H={depth_budget:.3f} full-power hours | "
        f"rho={DEPTH_BUDGET_RHO:.4f} | mode={budget_meta['depth_budget_mode']}"
    )

    state_y = validate_state_year(state, support_y, year)
    req_raw, coef_raw, active = load_h_outputs(year)

    req_raw, coef_raw, support_filter_meta = apply_support_to_requirements_and_coefficients(
        req_raw,
        coef_raw,
        support_y,
    )

    req, coeff = build_requirements_and_coefficients(req_raw, coef_raw)
    effectivity = build_effectivity_features(state_y, req, coeff)

    participants = build_participants(
        state_y,
        year,
        capacity_overrides,
        selected_support_hours=selected_hours,
    )

    opt_state = build_state_for_optimization(
        state_y,
        participants,
        effectivity,
        tariff_table,
    )

    if len(opt_state) != H * len(PROVINCES):
        raise RuntimeError(f"Internal state row mismatch for {year}.")

    if not req.empty:
        if req["constraint_id"].min() != 0 or req["constraint_id"].max() != len(req) - 1:
            raise RuntimeError("constraint_id is not dense.")

        missing_req = sorted(set(coeff["constraint_id"].unique()) - set(req["constraint_id"].unique()))
        if missing_req:
            raise RuntimeError(f"Coefficient rows reference missing constraints: {missing_req[:10]}")

    outside_pbar_positive = int(
        ((opt_state["tdtr_support"] == 0) & (opt_state["Pbar_value"] > 1e-9)).sum()
    )

    if TDTR_SUPPORT_APPLY_TO_PBAR and outside_pbar_positive > 0:
        raise RuntimeError(
            f"Support/Pbar inconsistency: {outside_pbar_positive} rows outside support still have Pbar > 0."
        )

    if TDTR_SUPPORT_APPLY_TO_REQUIREMENTS and not req.empty:
        bad_req = req.merge(
            support_y[["hour_of_year", "tdtr_support"]],
            on="hour_of_year",
            how="left",
        )

        bad_count = int((bad_req["tdtr_support"].fillna(0).astype(int) == 0).sum())

        if bad_count > 0:
            raise RuntimeError(
                f"Support/requirements inconsistency: {bad_count} requirements outside support."
            )

    year_dir = OPT_INPUT_DIR

    state_paths = safe_to_parquet_and_csv(opt_state, year_dir / f"debit_opt_state_{year}")
    req_paths = safe_to_parquet_and_csv(req, year_dir / f"debit_opt_requirements_{year}")
    coeff_paths = safe_to_parquet_and_csv(coeff, year_dir / f"debit_opt_coefficients_{year}")

    participants_path = year_dir / f"debit_opt_participants_{year}.csv"
    participants.to_csv(participants_path, index=False)
    print(f"[SAVE] {participants_path}")

    active_path_out = year_dir / f"debit_opt_active_branches_{year}.csv"
    active.to_csv(active_path_out, index=False)
    print(f"[SAVE] {active_path_out}")

    tariff_y = tariff_table[tariff_table["year"].astype(int) == int(year)].copy()
    tariff_path = year_dir / f"debit_opt_tariff_inputs_{year}.csv"
    tariff_y.to_csv(tariff_path, index=False)
    print(f"[SAVE] {tariff_path}")

    cost_desc = opt_state["restriction_cost"].describe(percentiles=[0.01, 0.5, 0.99]).to_dict()

    pbar_sum_in = float(opt_state.loc[opt_state["tdtr_support"] == 1, "Pbar_value"].sum())
    pbar_sum_out = float(opt_state.loc[opt_state["tdtr_support"] == 0, "Pbar_value"].sum())

    support_hours_by_province = (
        opt_state.groupby("province")["tdtr_support"].sum().astype(int).to_dict()
    )

    summary = {
        "year": int(year),
        "hours": int(H),
        "provinces": PROVINCES,
        "tdtr_support_scenario": TDTR_SUPPORT_SCENARIO,
        "tdtr_support_selected_hours": selected_hours,
        "tdtr_support_selected_share": float(selected_hours / H),
        "tdtr_support_path": support_meta.get("support_path"),
        "tdtr_support_apply_to_requirements": bool(TDTR_SUPPORT_APPLY_TO_REQUIREMENTS),
        "tdtr_support_apply_to_pbar": bool(TDTR_SUPPORT_APPLY_TO_PBAR),
        "support_filter_meta": support_filter_meta,
        "support_hours_by_province": {str(k): int(v) for k, v in support_hours_by_province.items()},

        "n_state_rows": int(len(opt_state)),
        "n_participants": int(len(participants)),
        "n_requirements": int(len(req)),
        "n_coefficients": int(len(coeff)),
        "total_requirement_MW_h": float(req["R_MW"].sum()) if len(req) else 0.0,
        "max_requirement_MW": float(req["R_MW"].max()) if len(req) else 0.0,
        "mean_contract_capacity_MW": float(participants["P_ctr_MW"].mean()),

        "Pbar_sum_inside_support_MW_h_proxy": pbar_sum_in,
        "Pbar_sum_outside_support_MW_h_proxy": pbar_sum_out,

        "tdtr_availability_share": float(TDTR_AVAILABILITY_SHARE),
        "tdtr_restriction_share": float(TDTR_RESTRICTION_SHARE),
        "annual_hours": int(H_year),

        "depth_budget_hours": float(depth_budget),
        "depth_budget_rho": float(DEPTH_BUDGET_RHO),
        "depth_budget_mode": budget_meta["depth_budget_mode"],
        "depth_budget_as_share_of_support": budget_meta["depth_budget_as_share_of_support"],
        "depth_budget_as_share_of_annual": budget_meta["depth_budget_as_share_of_annual"],

        "touch_budget_tokens": int(touch_budget),
        "touch_budget_mode": budget_meta["touch_budget_mode"],

        "slack_penalty": float(SLACK_PENALTY),

        "restriction_cost_formula": (
            "abs(DA price - province-year median DA price) "
            "+ DSO energy transport tariff "
            "+ DSO contracted-capacity tariff / annual hours"
        ),
        "restriction_cost_min": float(cost_desc.get("min", np.nan)),
        "restriction_cost_p50": float(cost_desc.get("50%", np.nan)),
        "restriction_cost_p99": float(cost_desc.get("99%", np.nan)),
        "restriction_cost_max": float(cost_desc.get("max", np.nan)),

        "tariff_sources": sorted(tariff_y["tariff_source"].dropna().unique().tolist()),
        "tariff_reference_years": sorted(map(int, tariff_y["tariff_reference_year"].dropna().unique().tolist())),

        "paths": {
            "state": state_paths,
            "requirements": req_paths,
            "coefficients": coeff_paths,
            "participants_csv": str(participants_path),
            "active_branches_csv": str(active_path_out),
            "tariff_inputs_csv": str(tariff_path),
        },

        "interpretation_note": (
            "TDTR candidate-hour support is imposed as an eligibility/availability mask. "
            "M1 touched allowance equals selected support hours. "
            "M2/M3 weighted-depth allowance equals rho_depth times selected support hours. "
            "The debit-rule comparison is conditional on this common candidate-hour support."
        ),
    }

    save_json(summary, year_dir / f"debit_opt_year_manifest_{year}.json")

    print(
        f"[INFO] year={year} | support={selected_hours:,}/{H:,} | "
        f"requirements={len(req):,} | coeffs={len(coeff):,} | "
        f"total R={summary['total_requirement_MW_h']:.3f} MW-h | "
        f"mean Pctr={summary['mean_contract_capacity_MW']:.2f} MW | "
        f"H_depth={depth_budget:.3f} | H_touch={touch_budget} | "
        f"rho={DEPTH_BUDGET_RHO:.4f} | "
        f"Pbar out-of-support sum={pbar_sum_out:.6g} | "
        f"cost p50={summary['restriction_cost_p50']:.3f}, "
        f"p99={summary['restriction_cost_p99']:.3f} €/MW-h"
    )

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dir(OPT_INPUT_DIR)
    ensure_dir(OUTPUTS_DIR)
    ensure_dir(QC_DIR)

    print("=" * 78)
    print("A: BUILD DEBIT OPTIMIZATION INPUTS")
    print("=" * 78)
    print(f"Repo root       : {REPO_ROOT}")
    print(f"State panel     : {STATE_PANEL_CSV}")
    print(f"H input dir     : {PTDF_DEBIT_INPUT_DIR}")
    print(f"Output dir      : {OPT_INPUT_DIR}")
    print(f"Years           : {YEARS}")
    print(f"Provinces       : {PROVINCES}")
    print(f"TDTR scenario   : {TDTR_SUPPORT_SCENARIO}")
    print(f"Support required: {TDTR_SUPPORT_REQUIRED}")
    print(f"Support -> R    : {TDTR_SUPPORT_APPLY_TO_REQUIREMENTS}")
    print(f"Support -> Pbar : {TDTR_SUPPORT_APPLY_TO_PBAR}")
    print(f"Use TDTR85      : {USE_TDTR85_BUDGETS}")
    print(
        f"TDTR availability/restriction share: "
        f"{TDTR_AVAILABILITY_SHARE:.3f} / {TDTR_RESTRICTION_SHARE:.3f}"
    )
    print(f"M1 touch budget : {'support hours' if TOUCH_BUDGET_FROM_SUPPORT else 'floor(0.15*T)'}")
    print(f"M2/M3 depth mode: {DEPTH_BUDGET_MODE}")
    print(f"M2/M3 rho depth : {DEPTH_BUDGET_RHO:.6f}")
    print(f"Pctr mode       : {CONTRACT_CAPACITY_MODE}")
    print(f"Pctr fraction   : {CONTRACT_CAPACITY_FRACTION}")
    print(f"Pctr min/max    : {CONTRACT_CAPACITY_MIN_MW} / {CONTRACT_CAPACITY_MAX_MW}")
    print(
        f"Participant cap.: "
        f"{PARTICIPANT_CAPACITY_PATH if PARTICIPANT_CAPACITY_PATH else 'constructed representative'}"
    )
    print(
        f"Tariff input    : "
        f"{TARIFF_TABLE_PATH if TARIFF_TABLE_PATH else 'embedded representative 2026 DSO proxy'}"
    )
    print(f"m_FA/m_TDTR     : {FIRM_CAPACITY_MULTIPLIER} / {TDTR_CAPACITY_MULTIPLIER}")
    print("Cost construction : Dutch public price/tariff proxy")
    print(
        "Cost formula      : |DA price - province-year median DA price| "
        "+ DSO energy tariff + DSO capacity tariff / annual hours"
    )
    print(f"Slack penalty     : {SLACK_PENALTY:g}")
    print("=" * 78)

    support, support_meta = load_tdtr_support()
    state = load_state_panel()
    tariff_table = load_tariff_table_for_years(YEARS)
    capacity_overrides = load_participant_capacity_overrides()

    tariff_global_path = OPT_INPUT_DIR / "debit_opt_tariff_inputs_all_years.csv"
    tariff_table.to_csv(tariff_global_path, index=False)
    print(f"[SAVE] {tariff_global_path}")

    summaries = {}

    for year in YEARS:
        summaries[str(year)] = build_year(
            int(year),
            state,
            support,
            support_meta,
            tariff_table,
            capacity_overrides,
        )

    manifest = {
        "script": Path(__file__).name,
        "repo_root": str(REPO_ROOT),
        "state_panel_csv": str(STATE_PANEL_CSV),
        "ptdf_debit_input_dir": str(PTDF_DEBIT_INPUT_DIR),
        "output_dir": str(OPT_INPUT_DIR),
        "years": YEARS,
        "provinces": PROVINCES,

        "tdtr_support": support_meta,
        "use_tdtr85_budgets": USE_TDTR85_BUDGETS,
        "tdtr_availability_share": TDTR_AVAILABILITY_SHARE,
        "tdtr_restriction_share": TDTR_RESTRICTION_SHARE,

        "touch_budget_from_support": TOUCH_BUDGET_FROM_SUPPORT,
        "depth_budget_mode": DEPTH_BUDGET_MODE,
        "depth_budget_rho": DEPTH_BUDGET_RHO,
        "depth_budget_hours_fallback": DEPTH_BUDGET_HOURS_FALLBACK,
        "touch_budget_hours_fallback": TOUCH_BUDGET_HOURS_FALLBACK,

        "contract_capacity_mode": CONTRACT_CAPACITY_MODE,
        "contract_capacity_fraction": CONTRACT_CAPACITY_FRACTION,
        "contract_capacity_min_MW": CONTRACT_CAPACITY_MIN_MW,
        "contract_capacity_max_MW": CONTRACT_CAPACITY_MAX_MW,
        "participant_capacity_path": str(PARTICIPANT_CAPACITY_PATH) if PARTICIPANT_CAPACITY_PATH else None,

        "tariff_table_path": str(TARIFF_TABLE_PATH) if TARIFF_TABLE_PATH else None,
        "tariff_reference_year": TARIFF_REFERENCE_YEAR,
        "province_dso_map": PROVINCE_DSO,
        "firm_capacity_multiplier": FIRM_CAPACITY_MULTIPLIER,
        "tdtr_capacity_multiplier": TDTR_CAPACITY_MULTIPLIER,

        "restriction_cost_formula": (
            "abs(DA price - province-year median DA price) "
            "+ DSO energy transport tariff "
            "+ DSO contracted-capacity tariff / annual hours"
        ),
        "price_cost_winsor_q": PRICE_WINSOR_Q,
        "slack_penalty": SLACK_PENALTY,
        "tariff_global_csv": str(tariff_global_path),

        "year_summaries": summaries,

        "interpretation_note": (
            "A imposes the G-built TDTR candidate-hour support on Pbar and, defensively, on relief requirements. "
            "M1 touched-time allowance is set equal to the selected candidate-hour support. "
            "M2/M3 weighted-depth allowance is separately calibrated as rho_depth times selected support hours. "
            "This makes the debit-rule comparison conditional on a common TDTR support while preserving scarcity in the weighted-depth budget."
        ),
    }

    save_json(manifest, OPT_INPUT_DIR / "debit_optimization_input_manifest.json")

    print("\n[DONE]")


if __name__ == "__main__":
    main()