#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1_run_m1_touched_only.py
=========================

Support-aware solver for the M1 touched-time debit rule.

UPDATED FOR SUPPORT-AWARE A OUTPUTS
-----------------------------------
This version assumes A_build_debit_optimization_inputs.py is the source of truth
for:
    - TDTR candidate-hour support;
    - Pbar after support masking;
    - operating costs;
    - H_touched;
    - H_depth audit column if present;
    - slack penalty;
    - relief requirements and coefficients.

The M1 rule is

    d_{i,t}^{M1} = 1{x_{i,t} > 0},
    sum_t d_{i,t}^{M1} <= H_i^{touch}.

Key mathematical update
-----------------------
If the selected TDTR support has at most H_i^{touch} positive-Pbar eligible
hours for every province i, then the M1 binary variables are redundant. In that
case, M1 is solved exactly as a continuous LP:

    0 <= x_{i,t} <= Pbar_{i,t},
    A x + s >= R,
    s >= 0.

This is not a relaxation under the detected condition. Every feasible LP
solution can touch at most the number of eligible positive-Pbar hours, and that
number is already no larger than the touched-token budget.

If the support is wider than the M1 touched budget, B1 solves the compact
support-restricted MILP:

    x_{i,t} <= Pbar_{i,t} y_{i,t},
    sum_t y_{i,t} <= H_i^{touch},
    y_{i,t} binary,

with Pbar_{i,t}=0 outside TDTR support.

Main robustness fix
-------------------
A may export several synonymous audit columns, for example both P_ctr_MW and
P_i, or both H_touch_tokens and Hbar. This script never renames such columns
blindly. Instead, it coalesces all aliases into one canonical column and removes
the duplicates. This prevents pandas from returning a DataFrame when the code
expects a Series.

Expected inputs
---------------
data/derived/debit_optimization_inputs/
    debit_opt_state_{year}.parquet/.csv
    debit_opt_requirements_{year}.parquet/.csv
    debit_opt_coefficients_{year}.parquet/.csv
    debit_opt_participants_{year}.csv
    debit_opt_year_manifest_{year}.json

Outputs
-------
outputs/results/m1_only/
    m1_only_summary.csv
    m1_only_manifest.json

outputs/results/m1_only/solutions/
    M1_province_hour_{year}.csv.gz
    M1_positive_slack_{year}.csv.gz
    M1_budget_summary_{year}.csv
    M1_support_budget_diagnostic_{year}.csv
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp
except Exception as exc:
    raise RuntimeError("scipy.sparse is required for sparse relief matrices.") from exc

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:
    raise RuntimeError("gurobipy is required. Install/configure Gurobi first.") from exc


# =============================================================================
# PATH SETUP / CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else Path.cwd()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pipeline_config as cfg  # type: ignore
except Exception:
    cfg = None


def cfg_get(name: str, default):
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


INPUT_DIR = Path(
    cfg_get(
        "DEBIT_OPT_INPUT_DIR",
        REPO_ROOT / "data" / "derived" / "debit_optimization_inputs",
    )
)

RESULTS_DIR = Path(
    os.getenv(
        "M1_RESULTS_DIR",
        str(REPO_ROOT / "outputs" / "results" / "m1_only"),
    )
)

SOLUTION_DIR = Path(
    os.getenv(
        "M1_SOLUTION_DIR",
        str(RESULTS_DIR / "solutions"),
    )
)


def parse_years() -> List[int]:
    raw = os.getenv(
        "DEBIT_YEARS",
        ",".join(str(y) for y in cfg_get("ALL_YEARS", [2021, 2022, 2023, 2024, 2025])),
    )
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


YEARS = parse_years()

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
PROVINCES = list(cfg_get("NL_PROVINCES", DEFAULT_PROVINCES))

# General Gurobi settings.
OUTPUT_FLAG = int(os.getenv("GUROBI_OUTPUT_FLAG", "1"))
THREADS = int(os.getenv("GUROBI_THREADS", "0"))
LP_METHOD = int(os.getenv("GUROBI_LP_METHOD", "-1"))

# Support-aware M1 mode.
M1_SOLVE_MODE = str(cfg_get("M1_SOLVE_MODE", "auto")).strip().lower()
M1_TIGHTEN_USEFUL_PBAR = cfg_bool("M1_TIGHTEN_USEFUL_PBAR", False)

M1_COMPACT_MIP_TIME_LIMIT = float(cfg_get("M1_COMPACT_MIP_TIME_LIMIT", 1800.0))
M1_COMPACT_MIP_GAP = float(cfg_get("M1_COMPACT_MIP_GAP", 0.01))
M1_COMPACT_MIP_OUTPUT_FLAG = int(cfg_get("M1_COMPACT_MIP_OUTPUT_FLAG", OUTPUT_FLAG))
M1_COMPACT_MIP_FOCUS = int(cfg_get("M1_COMPACT_MIP_FOCUS", 3))
M1_COMPACT_MIP_HEURISTICS = float(cfg_get("M1_COMPACT_MIP_HEURISTICS", 0.05))
M1_COMPACT_MIP_CUTS = int(cfg_get("M1_COMPACT_MIP_CUTS", 3))
M1_COMPACT_MIP_PRESOLVE = int(cfg_get("M1_COMPACT_MIP_PRESOLVE", 2))
M1_COMPACT_NODEFILE_START = float(cfg_get("M1_COMPACT_NODEFILE_START", 0.5))
M1_REQUIRE_SUPPORT_REDUCTION = cfg_bool("M1_REQUIRE_SUPPORT_REDUCTION", True)

# Numerical tolerances.
TOUCH_TOL = float(os.getenv("TOUCH_TOL", "1e-7"))
POSITIVE_SLACK_TOL = float(os.getenv("POSITIVE_SLACK_TOL", "1e-7"))
BUDGET_BINDING_TOL = float(os.getenv("BUDGET_BINDING_TOL", "1e-5"))

REQUIRE_A_COSTS = os.getenv("REQUIRE_A_COSTS", "1").strip().lower() not in {"0", "false", "no", "n"}
REQUIRE_A_BUDGETS = cfg_bool("REQUIRE_A_BUDGETS", True)
REQUIRE_POSITIVE_SLACK_PENALTY = cfg_bool("REQUIRE_POSITIVE_SLACK_PENALTY", True)

DEFAULT_SLACK_PENALTY = float(
    os.getenv(
        "DEBIT_SLACK_PENALTY",
        os.getenv("M_PENALTY", str(cfg_get("M_PENALTY", 5000.0))),
    )
)


# =============================================================================
# ALIASES
# =============================================================================

ALIASES: Dict[str, List[str]] = {
    "province": ["province", "provincie", "node", "deployment_unit", "n", "area"],
    "timestamp": ["timestamp_utc", "timestamp", "datetime_utc", "time", "datetime", "utc_timestamp"],
    "hour": ["hour_of_year", "hour", "h", "t", "hour_index", "time_index", "hoy"],
    "branch": ["branch_id", "constraint_id", "constraint", "k", "line_id", "monitored_constraint", "element_id"],
    "requirement": ["R_MW", "R_mw", "relief_requirement_mw", "requirement_mw", "R", "relief_mw", "required_relief_mw"],
    "coefficient": ["coefficient", "a", "a_kit", "effectivity", "ptdf_effectivity", "relief_coeff", "signed_effectivity"],
    "pbar": ["Pbar_value", "pbar_mw", "Pbar_mw", "Pbar", "available_restrictable_mw", "availability_mw", "available_mw", "max_restriction_mw"],
    "pctr": ["P_ctr_MW", "Pctr_mw", "pctr_mw", "P_i_ctr_mw", "P_i", "contract_capacity_mw", "contracted_capacity_mw", "Pctr", "p_ctr_mw"],
    "touch_budget": ["H_touch_tokens", "touched_budget_tokens", "H_touched", "Hbar_i", "Hbar", "Hbar_eff", "tdtr_touched_budget", "annual_touch_budget", "touch_budget"],
    "depth_budget": ["H_depth_hours", "depth_budget_hours", "H_depth", "H_i", "H", "tdtr_depth_budget", "annual_depth_budget", "depth_budget"],
    "cost": ["cost_eur_per_mwh", "restriction_cost", "c_value", "restriction_cost_eur_per_mwh", "restriction_cost_eur_per_mw", "cost", "c", "c_eur_per_mwh"],
    "slack_penalty": ["slack_penalty", "M", "M_k", "emergency_slack_penalty"],
    "support": ["tdtr_support", "is_tdtr_candidate", "candidate_hour", "support", "selected", "is_selected", "y_support"],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class YearInput:
    year: int
    state: pd.DataFrame
    requirements: pd.DataFrame
    coefficients: pd.DataFrame
    participants: pd.DataFrame
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def total_requirement(self) -> float:
        return float(self.requirements["R_mw"].sum())


@dataclass
class M1Result:
    year: int
    status: str
    status_code: int
    quality_flag: str
    objective: float
    runtime_sec: float
    x: pd.DataFrame
    slack: pd.DataFrame
    budget: pd.DataFrame
    total_relief_requirement: float
    total_slack: float
    operating_cost_x: float
    slack_penalty_cost: float
    relief_served_share: float
    total_touched_tokens: float
    allowance_productivity: float
    best_lower_bound: Optional[float]
    certified_gap: Optional[float]
    mip_gap: Optional[float]
    best_bound: Optional[float]
    node_count: Optional[float]
    solution_count: Optional[int]
    model_rows: Optional[int]
    model_cols: Optional[int]
    model_nnz: Optional[int]
    n_binary_rows_active: int
    n_binary_rows_total: int
    notes: List[str] = field(default_factory=list)


# =============================================================================
# GENERIC UTILITIES
# =============================================================================

def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTION_DIR.mkdir(parents=True, exist_ok=True)


def read_table(base: Path, required: bool = True) -> Optional[pd.DataFrame]:
    if base.suffix:
        candidates = [base]
    else:
        candidates = [
            base.with_suffix(".parquet"),
            base.with_suffix(".csv"),
            Path(str(base) + ".csv.gz"),
        ]

    for path in candidates:
        if path.exists():
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            if str(path).endswith(".csv.gz"):
                return pd.read_csv(path, compression="gzip", low_memory=False)
            return pd.read_csv(path, low_memory=False)

    if required:
        raise FileNotFoundError(f"Missing required table. Tried: {candidates}")
    return None


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".csv.gz"):
        df.to_csv(path, index=index, compression="gzip")
    else:
        df.to_csv(path, index=index)
    print(f"[SAVE] {path}")


def alias_positions(df: pd.DataFrame, candidates: Sequence[str]) -> List[int]:
    candidate_lowers = {str(c).strip().lower() for c in candidates}
    return [i for i, col in enumerate(df.columns) if str(col).strip().lower() in candidate_lowers]


def has_any_alias(df: pd.DataFrame, candidates: Sequence[str]) -> bool:
    return len(alias_positions(df, candidates)) > 0


def find_col(df: pd.DataFrame, key: str, required: bool = True, context: str = "") -> Optional[str]:
    aliases = ALIASES.get(key, [key])
    lower_map: Dict[str, str] = {}
    for c in df.columns:
        lower_map.setdefault(str(c).strip().lower(), str(c))

    for cand in aliases:
        if cand in df.columns:
            return cand
        lc = cand.strip().lower()
        if lc in lower_map:
            return lower_map[lc]

    if required:
        raise KeyError(
            f"Could not find column for {key!r} in {context or 'dataframe'}. "
            f"Tried aliases={aliases}. Available columns={list(df.columns)}"
        )
    return None


def coalesce_numeric_columns(
    df: pd.DataFrame,
    candidates: Iterable[str],
    out_col: str,
    *,
    required: bool = False,
    context: str = "dataframe",
    default: Optional[float] = None,
) -> pd.DataFrame:
    """
    Collapse all numeric aliases into one canonical column.

    This intentionally removes all alias columns first, including an existing
    canonical column, then appends exactly one canonical output column. This is
    the crucial guard against duplicate labels such as P_ctr_MW and P_i both
    being renamed to Pctr_mw.
    """
    df = df.copy()
    positions = alias_positions(df, list(candidates))

    if not positions:
        if required:
            raise KeyError(f"Could not find aliases for {out_col!r} in {context}.")
        df[out_col] = default if default is not None else np.nan
        return df

    pieces: List[pd.Series] = []
    for pos in positions:
        pieces.append(pd.to_numeric(df.iloc[:, pos], errors="coerce"))

    value = pieces[0]
    for s in pieces[1:]:
        value = value.combine_first(s)

    keep_positions = [i for i in range(df.shape[1]) if i not in set(positions)]
    out = df.iloc[:, keep_positions].copy()
    out[out_col] = value.to_numpy()
    return out


def coalesce_text_columns(
    df: pd.DataFrame,
    candidates: Iterable[str],
    out_col: str,
    *,
    required: bool = False,
    context: str = "dataframe",
    default: Optional[str] = None,
) -> pd.DataFrame:
    """
    Collapse all text aliases into one canonical column and remove duplicates.
    Empty strings are treated as missing.
    """
    df = df.copy()
    positions = alias_positions(df, list(candidates))

    if not positions:
        if required:
            raise KeyError(f"Could not find aliases for {out_col!r} in {context}.")
        df[out_col] = default
        return df

    pieces: List[pd.Series] = []
    for pos in positions:
        s = df.iloc[:, pos].astype("string")
        s = s.str.strip()
        s = s.mask(s == "", pd.NA)
        pieces.append(s)

    value = pieces[0]
    for s in pieces[1:]:
        value = value.combine_first(s)

    keep_positions = [i for i in range(df.shape[1]) if i not in set(positions)]
    out = df.iloc[:, keep_positions].copy()
    out[out_col] = value.astype(object).to_numpy()
    return out


def to_numeric_required(s: pd.Series, name: str) -> pd.Series:
    if isinstance(s, pd.DataFrame):
        raise TypeError(
            f"Column {name} resolved to a DataFrame, which means duplicate canonical columns remain: "
            f"{list(s.columns)}. This should not happen after alias coalescing."
        )
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().any():
        bad = int(out.isna().sum())
        raise ValueError(f"Column {name} has {bad} missing/non-numeric values.")
    return out.astype(float)


def add_timestamp_and_hour(df: pd.DataFrame, year: int, context: str) -> pd.DataFrame:
    df = df.copy()

    hcol = find_col(df, "hour", required=False, context=context)
    tcol = find_col(df, "timestamp", required=False, context=context)

    if hcol is not None and hcol != "hour_of_year":
        df = df.rename(columns={hcol: "hour_of_year"})

    if tcol is not None and tcol != "timestamp_utc":
        df = df.rename(columns={tcol: "timestamp_utc"})

    if "timestamp_utc" in df.columns:
        ts_raw = df["timestamp_utc"]
        if isinstance(ts_raw, pd.DataFrame):
            ts_raw = ts_raw.iloc[:, 0]
        ts = pd.to_datetime(ts_raw, errors="coerce", utc=True).dt.floor("h")
        if ts.isna().any():
            raise ValueError(f"{context}: timestamp_utc has unparseable values.")
        df["timestamp_utc"] = ts

        if "hour_of_year" not in df.columns:
            start = pd.Timestamp(f"{year}-01-01T00:00:00Z")
            df["hour_of_year"] = ((ts - start) / pd.Timedelta(hours=1)).round().astype(int) + 1

    if "hour_of_year" not in df.columns:
        raise KeyError(f"{context} must contain timestamp_utc or hour_of_year.")

    h = df["hour_of_year"]
    if isinstance(h, pd.DataFrame):
        h = h.iloc[:, 0]
    df["hour_of_year"] = pd.to_numeric(h, errors="raise").astype(int)

    return df


def align_hour_basis(reference_hours: Iterable[int], df: pd.DataFrame, context: str) -> pd.DataFrame:
    df = df.copy()
    ref = set(int(x) for x in reference_hours)
    hrs = set(int(x) for x in df["hour_of_year"].dropna().unique())

    if hrs.issubset(ref):
        return df

    hrs_minus = set(h - 1 for h in hrs)
    if hrs_minus.issubset(ref):
        df["hour_of_year"] = df["hour_of_year"].astype(int) - 1
        print(f"[INFO] {context}: shifted hour_of_year down by 1 to match state.")
        return df

    hrs_plus = set(h + 1 for h in hrs)
    if hrs_plus.issubset(ref):
        df["hour_of_year"] = df["hour_of_year"].astype(int) + 1
        print(f"[INFO] {context}: shifted hour_of_year up by 1 to match state.")
        return df

    if len(hrs & ref) == 0 and hrs:
        raise ValueError(
            f"{context}: hour_of_year has no overlap with state. "
            f"Example source hours={sorted(list(hrs))[:5]}, state hours={sorted(list(ref))[:5]}"
        )
    return df


def load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def status_name_from_code(code: int) -> str:
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return mapping.get(int(code), f"STATUS_{int(code)}")


def safe_attr(model: gp.Model, attr: str, default=None):
    try:
        return getattr(model, attr)
    except Exception:
        return default


def certified_gap(ub: float, lb: Optional[float]) -> Optional[float]:
    if lb is None or not np.isfinite(lb) or not np.isfinite(ub):
        return None
    return max(0.0, float((ub - lb) / max(1.0, abs(ub))))


def set_common_params(model: gp.Model, output_flag: int) -> None:
    model.Params.OutputFlag = int(output_flag)
    if THREADS > 0:
        model.Params.Threads = THREADS
    if LP_METHOD >= 0:
        model.Params.Method = LP_METHOD


def infer_tdtr_support_hours_from_state(state: pd.DataFrame) -> int:
    if "tdtr_support" in state.columns:
        tmp = state[["hour_of_year", "tdtr_support"]].copy()
        tmp["tdtr_support"] = (
            pd.to_numeric(tmp["tdtr_support"], errors="coerce")
            .fillna(0.0)
            .astype(float)
            .gt(0.5)
            .astype(int)
        )
        return int(tmp.groupby("hour_of_year")["tdtr_support"].max().sum())

    tmp = state[["hour_of_year", "pbar_mw"]].copy()
    tmp["pbar_mw"] = pd.to_numeric(tmp["pbar_mw"], errors="coerce").fillna(0.0)
    return int((tmp.groupby("hour_of_year")["pbar_mw"].max() > TOUCH_TOL).sum())


# =============================================================================
# INPUT NORMALIZATION
# =============================================================================

def normalize_state(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = add_timestamp_and_hour(df, year, "state")
    df = coalesce_text_columns(df, ALIASES["province"], "province", required=True, context="state")
    df["province"] = df["province"].astype(str).str.strip()

    df = coalesce_numeric_columns(df, ALIASES["pbar"], "pbar_mw", required=True, context="state")

    if not has_any_alias(df, ALIASES["cost"]):
        if REQUIRE_A_COSTS:
            raise ValueError("State file lacks A's cost_eur_per_mwh / restriction_cost / c_value column.")
        df["cost_eur_per_mwh"] = 1.0
        df["cost_source"] = "uniform_debug_not_publication"
    else:
        df = coalesce_numeric_columns(df, ALIASES["cost"], "cost_eur_per_mwh", required=True, context="state")
        if "cost_source" not in df.columns:
            df["cost_source"] = "A_output_cost_eur_per_mwh"

    df["cost_eur_per_mwh"] = to_numeric_required(df["cost_eur_per_mwh"], "cost_eur_per_mwh")
    if (df["cost_eur_per_mwh"] < 0).any():
        raise ValueError("Negative restriction costs found.")

    support_was_present = has_any_alias(df, ALIASES["support"])
    df = coalesce_numeric_columns(df, ALIASES["support"], "tdtr_support", required=False, context="state")

    if support_was_present and df["tdtr_support"].notna().any():
        df["tdtr_support"] = (
            pd.to_numeric(df["tdtr_support"], errors="coerce")
            .fillna(0.0)
            .astype(float)
            .gt(0.5)
            .astype(int)
        )
    else:
        df["tdtr_support"] = (
            pd.to_numeric(df["pbar_mw"], errors="coerce")
            .fillna(0.0)
            .gt(TOUCH_TOL)
            .astype(int)
        )
        warnings.warn("State file has no tdtr_support column. Inferring support from positive Pbar.")

    keep = [
        c
        for c in [
            "province",
            "timestamp_utc",
            "year",
            "hour_of_year",
            "pbar_mw",
            "tdtr_support",
            "cost_eur_per_mwh",
            "cost_source",
            "restriction_cost_source",
        ]
        if c in df.columns
    ]

    out = (
        df[keep]
        .sort_values(["province", "hour_of_year"])
        .drop_duplicates(["province", "hour_of_year"], keep="first")
        .reset_index(drop=True)
    )

    out["pbar_mw"] = pd.to_numeric(out["pbar_mw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["tdtr_support"] = pd.to_numeric(out["tdtr_support"], errors="coerce").fillna(0).astype(int)
    out.loc[out["tdtr_support"] == 0, "pbar_mw"] = 0.0

    return out


def normalize_participants(df: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize participant table without duplicate-column bugs.

    This is the main fix for the pandas error:
        TypeError: arg must be a list, tuple, 1-d array, or Series

    That error occurred because multiple capacity aliases were renamed to
    Pctr_mw, so df["Pctr_mw"] returned a DataFrame. Here aliases are coalesced
    and removed before numeric conversion.
    """
    df = df.copy()

    touch_budget_present = has_any_alias(df, ALIASES["touch_budget"])

    df = coalesce_text_columns(df, ALIASES["province"], "province", required=True, context="participants")
    df = coalesce_numeric_columns(df, ALIASES["pctr"], "Pctr_mw", required=True, context="participants")
    df = coalesce_numeric_columns(df, ALIASES["touch_budget"], "H_touched", required=False, context="participants")
    df = coalesce_numeric_columns(df, ALIASES["depth_budget"], "H_depth", required=False, context="participants")
    df = coalesce_numeric_columns(df, ALIASES["slack_penalty"], "slack_penalty", required=False, context="participants")

    df["province"] = df["province"].astype(str).str.strip()
    df["Pctr_mw"] = to_numeric_required(df["Pctr_mw"], "Pctr_mw")

    support_hours = infer_tdtr_support_hours_from_state(state)

    if REQUIRE_A_BUDGETS and not touch_budget_present:
        raise KeyError(
            "Participants file has no H_touch_tokens/H_touched/Hbar column. "
            "A must write the touched-time budget for B1. "
            "Set REQUIRE_A_BUDGETS=0 only for debugging fallback."
        )

    if touch_budget_present:
        df["H_touched"] = np.floor(to_numeric_required(df["H_touched"], "H_touched")).astype(int)
        df["H_touched_source"] = "A_participant_file"
    else:
        df["H_touched"] = int(support_hours)
        df["H_touched_source"] = "B1_fallback_support_hours"
        warnings.warn(
            f"H_touched missing. Using fallback H_touched=support_hours={support_hours}. "
            "For final runs, A should write H_touched."
        )

    if df["H_depth"].notna().any():
        df["H_depth"] = pd.to_numeric(df["H_depth"], errors="coerce")
        df["H_depth_source"] = "A_participant_file"
    else:
        df["H_depth"] = np.nan
        df["H_depth_source"] = "not_used_by_B1"

    if df["slack_penalty"].notna().any():
        df["slack_penalty"] = to_numeric_required(df["slack_penalty"], "slack_penalty")
        df["slack_penalty_source"] = "A_participant_file"
    else:
        df["slack_penalty"] = DEFAULT_SLACK_PENALTY
        df["slack_penalty_source"] = "B1_fallback"
        warnings.warn(
            "Participants file has no slack_penalty/M column. Using "
            f"DEFAULT_SLACK_PENALTY={DEFAULT_SLACK_PENALTY:g}. For final runs, write it from A."
        )

    if (df["Pctr_mw"] <= 0).any():
        bad = df.loc[df["Pctr_mw"] <= 0, ["province", "Pctr_mw"]]
        raise ValueError("Nonpositive contracted capacity found.\n" + bad.to_string(index=False))

    if (df["H_touched"] < 0).any():
        bad = df.loc[df["H_touched"] < 0, ["province", "H_touched"]]
        raise ValueError("Negative touched-token allowance found.\n" + bad.to_string(index=False))

    if REQUIRE_POSITIVE_SLACK_PENALTY and (df["slack_penalty"] <= 0).any():
        bad = df.loc[df["slack_penalty"] <= 0, ["province", "slack_penalty"]]
        raise ValueError(
            "Slack penalty must be strictly positive. Otherwise emergency slack is free.\n"
            + bad.to_string(index=False)
        )

    df["tdtr_support_hours"] = int(support_hours)
    df["touch_budget_to_support_ratio"] = df["H_touched"] / max(float(support_hours), 1.0)
    df["depth_budget_to_support_ratio"] = df["H_depth"] / max(float(support_hours), 1.0)

    pctr_map = df.set_index("province")["Pctr_mw"].astype(float).to_dict()
    pctr_series = state["province"].map(pctr_map)

    if pctr_series.isna().any():
        missing = sorted(state.loc[pctr_series.isna(), "province"].unique().tolist())
        raise ValueError(f"State contains provinces absent from participants: {missing}")

    pctr_series = pctr_series.astype(float)
    state["pbar_mw"] = pd.to_numeric(state["pbar_mw"], errors="coerce").fillna(pctr_series)
    state["pbar_mw"] = np.minimum(state["pbar_mw"].astype(float), pctr_series).clip(lower=0.0)
    state.loc[state["tdtr_support"].astype(int) == 0, "pbar_mw"] = 0.0

    return df.sort_values("province").reset_index(drop=True)


def normalize_requirements(df: pd.DataFrame, year: int, state_hours: Iterable[int]) -> pd.DataFrame:
    df = add_timestamp_and_hour(df, year, "requirements")
    df = coalesce_text_columns(df, ALIASES["branch"], "branch_id", required=True, context="requirements")
    df = coalesce_numeric_columns(df, ALIASES["requirement"], "R_mw", required=True, context="requirements")

    df = align_hour_basis(state_hours, df, "requirements")
    df["branch_id"] = df["branch_id"].astype(str)
    df["R_mw"] = to_numeric_required(df["R_mw"], "R_mw").clip(lower=0.0)

    out = df.groupby(["branch_id", "hour_of_year"], as_index=False)["R_mw"].sum()
    out = out[out["R_mw"] > 0].reset_index(drop=True)
    return out


def normalize_coefficients(df: pd.DataFrame, year: int, state_hours: Iterable[int]) -> pd.DataFrame:
    df = add_timestamp_and_hour(df, year, "coefficients")
    df = coalesce_text_columns(df, ALIASES["branch"], "branch_id", required=True, context="coefficients")
    df = coalesce_text_columns(df, ALIASES["province"], "province", required=True, context="coefficients")
    df = coalesce_numeric_columns(df, ALIASES["coefficient"], "a", required=True, context="coefficients")

    df = align_hour_basis(state_hours, df, "coefficients")
    df["branch_id"] = df["branch_id"].astype(str)
    df["province"] = df["province"].astype(str).str.strip()
    df["a"] = to_numeric_required(df["a"], "a")

    df = df[np.isfinite(df["a"]) & (np.abs(df["a"]) > 1e-12)].copy()
    out = df.groupby(["branch_id", "hour_of_year", "province"], as_index=False)["a"].sum()
    out = out[np.abs(out["a"]) > 1e-12].reset_index(drop=True)
    return out


def load_year_input(year: int) -> YearInput:
    state = read_table(INPUT_DIR / f"debit_opt_state_{year}", required=True)
    req = read_table(INPUT_DIR / f"debit_opt_requirements_{year}", required=True)
    coeff = read_table(INPUT_DIR / f"debit_opt_coefficients_{year}", required=True)
    part = read_table(INPUT_DIR / f"debit_opt_participants_{year}.csv", required=True)

    assert state is not None and req is not None and coeff is not None and part is not None

    state = normalize_state(state, year)
    part = normalize_participants(part, state)
    req = normalize_requirements(req, year, state["hour_of_year"].unique())
    coeff = normalize_coefficients(coeff, year, state["hour_of_year"].unique())

    metadata = load_json_if_exists(INPUT_DIR / f"debit_opt_year_manifest_{year}.json")

    yi = YearInput(
        year=year,
        state=state,
        requirements=req,
        coefficients=coeff,
        participants=part,
        metadata=metadata,
    )
    validate_year_input(yi)
    return yi


def validate_year_input(yi: YearInput) -> None:
    st, req, coeff, part = yi.state, yi.requirements, yi.coefficients, yi.participants

    for col in ["province", "hour_of_year", "pbar_mw", "tdtr_support", "cost_eur_per_mwh"]:
        if col not in st.columns:
            raise KeyError(f"State missing {col}")

    for col in ["branch_id", "hour_of_year", "R_mw"]:
        if col not in req.columns:
            raise KeyError(f"Requirements missing {col}")

    for col in ["branch_id", "hour_of_year", "province", "a"]:
        if col not in coeff.columns:
            raise KeyError(f"Coefficients missing {col}")

    for col in ["province", "Pctr_mw", "H_touched", "slack_penalty"]:
        if col not in part.columns:
            raise KeyError(f"Participants missing {col}")

    for name, frame in [("state", st), ("requirements", req), ("coefficients", coeff), ("participants", part)]:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        if duplicates:
            raise ValueError(f"{name} still has duplicate columns after normalization: {duplicates}")

    if st["pbar_mw"].isna().any() or st["cost_eur_per_mwh"].isna().any():
        raise ValueError("State contains missing pbar or cost.")

    if ((st["tdtr_support"].astype(int) == 0) & (st["pbar_mw"].astype(float) > TOUCH_TOL)).any():
        bad = st.loc[
            (st["tdtr_support"].astype(int) == 0) & (st["pbar_mw"].astype(float) > TOUCH_TOL)
        ].head(10)
        raise ValueError("State has positive Pbar outside TDTR support.\n" + bad.to_string(index=False))

    if REQUIRE_POSITIVE_SLACK_PENALTY and (part["slack_penalty"] <= 0).any():
        bad = part.loc[part["slack_penalty"] <= 0, ["province", "slack_penalty"]]
        raise ValueError("Nonpositive slack penalty found.\n" + bad.to_string(index=False))

    if (part["H_touched"] < 0).any():
        raise ValueError("Negative H_touched found.")

    state_keys = set(zip(st["province"].astype(str), st["hour_of_year"].astype(int)))
    coeff_keys = set(zip(coeff["province"].astype(str), coeff["hour_of_year"].astype(int)))
    missing = coeff_keys - state_keys
    if missing:
        raise ValueError(f"Coefficient province-hour keys missing from state. Sample={list(missing)[:5]}")

    req_keys = set(zip(req["branch_id"].astype(str), req["hour_of_year"].astype(int)))
    coeff_req_keys = set(zip(coeff["branch_id"].astype(str), coeff["hour_of_year"].astype(int)))
    orphan = coeff_req_keys - req_keys
    if orphan:
        warnings.warn(
            f"Year {yi.year}: {len(orphan)} coefficient branch-hour keys are absent from positive requirements "
            "and will be ignored in the sparse relief matrix."
        )


# =============================================================================
# SPARSE STRUCTURES
# =============================================================================

def build_indices(yi: YearInput) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participants = yi.participants.copy().sort_values("province").reset_index(drop=True)
    participants["part_idx"] = np.arange(len(participants), dtype=np.int64)

    state = yi.state.copy().merge(
        participants[["province", "part_idx"]],
        on="province",
        how="left",
        validate="many_to_one",
    )

    if state["part_idx"].isna().any():
        raise ValueError("Some state provinces are absent from participants.")

    state = state.sort_values(["hour_of_year", "part_idx"]).reset_index(drop=True)
    state["x_idx"] = np.arange(len(state), dtype=np.int64)

    req = yi.requirements.copy().sort_values(["hour_of_year", "branch_id"]).reset_index(drop=True)
    req["relief_idx"] = np.arange(len(req), dtype=np.int64)

    coeff = yi.coefficients.copy()
    coeff = coeff.merge(
        state[["province", "hour_of_year", "x_idx", "part_idx"]],
        on=["province", "hour_of_year"],
        how="inner",
        validate="many_to_one",
    )
    coeff = coeff.merge(
        req[["branch_id", "hour_of_year", "relief_idx"]],
        on=["branch_id", "hour_of_year"],
        how="inner",
        validate="many_to_one",
    )
    coeff = coeff[np.isfinite(coeff["a"]) & (np.abs(coeff["a"]) > 1e-12)].copy()

    return state, req, coeff, participants


def build_relief_x_matrix(req: pd.DataFrame, coeff: pd.DataFrame, n_x: int) -> sp.csr_matrix:
    if req.empty or coeff.empty:
        return sp.csr_matrix((len(req), n_x))

    rows = coeff["relief_idx"].to_numpy(dtype=np.int64)
    cols = coeff["x_idx"].to_numpy(dtype=np.int64)
    data = coeff["a"].to_numpy(dtype=float)

    return sp.coo_matrix((data, (rows, cols)), shape=(len(req), n_x)).tocsr()


def tighten_useful_pbar(state: pd.DataFrame, req: pd.DataFrame, coeff: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Optional conservative local Pbar tightening.

    Default is off because A already imposes TDTR support. The tightening is
    skipped automatically when signed coefficients exist.
    """
    state = state.copy()
    before = state["pbar_mw"].astype(float).copy()

    n_negative = int((pd.to_numeric(coeff["a"], errors="coerce") < -1e-12).sum())
    if n_negative > 0:
        state["pbar_mw_original"] = before
        state["useful_pbar_mw"] = before
        return state, {
            "skipped": True,
            "reason": "signed_coefficients_present",
            "n_negative_coefficients": n_negative,
            "before_positive_pbar": int((before > TOUCH_TOL).sum()),
            "after_positive_pbar": int((before > TOUCH_TOL).sum()),
            "mean_pbar_before": float(before.mean()),
            "mean_pbar_after": float(before.mean()),
            "max_pbar_before": float(before.max()),
            "max_pbar_after": float(before.max()),
        }

    tmp = coeff[["province", "hour_of_year", "relief_idx", "a"]].copy()
    tmp = tmp[tmp["a"] > 1e-12].copy()

    if tmp.empty:
        state["pbar_mw_original"] = before
        state["useful_pbar_mw"] = 0.0
        state["pbar_mw"] = 0.0
    else:
        tmp = tmp.merge(req[["relief_idx", "R_mw"]], on="relief_idx", how="left", validate="many_to_one")
        tmp["useful_ub_candidate"] = tmp["R_mw"].clip(lower=0.0) / tmp["a"].clip(lower=1e-12)
        useful = tmp.groupby(["province", "hour_of_year"], as_index=False).agg(
            useful_pbar_mw=("useful_ub_candidate", "max")
        )

        state["pbar_mw_original"] = before
        state = state.merge(useful, on=["province", "hour_of_year"], how="left", validate="one_to_one")
        state["useful_pbar_mw"] = pd.to_numeric(state["useful_pbar_mw"], errors="coerce").fillna(0.0)
        state["pbar_mw"] = np.minimum(before, state["useful_pbar_mw"].astype(float)).clip(lower=0.0)
        state.loc[state["tdtr_support"].astype(int) == 0, "pbar_mw"] = 0.0

    return state, {
        "skipped": False,
        "reason": "nonnegative_coefficients",
        "n_negative_coefficients": n_negative,
        "before_positive_pbar": int((before > TOUCH_TOL).sum()),
        "after_positive_pbar": int((state["pbar_mw"] > TOUCH_TOL).sum()),
        "mean_pbar_before": float(before.mean()),
        "mean_pbar_after": float(state["pbar_mw"].mean()),
        "max_pbar_before": float(before.max()),
        "max_pbar_after": float(state["pbar_mw"].max()),
    }


# =============================================================================
# SUPPORT DIAGNOSTICS
# =============================================================================

def support_budget_diagnostic(state: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["province", "H_touched"]
    for c in ["H_depth", "tdtr_support_hours"]:
        if c in participants.columns:
            merge_cols.append(c)

    diag = (
        state.assign(
            eligible=(state["pbar_mw"].to_numpy(dtype=float) > TOUCH_TOL).astype(int),
            support_flag=pd.to_numeric(state["tdtr_support"], errors="coerce").fillna(0).astype(int),
        )
        .groupby("province", as_index=False)
        .agg(
            eligible_hours=("eligible", "sum"),
            support_hours=("support_flag", "sum"),
            positive_pbar_sum=("pbar_mw", "sum"),
            max_pbar=("pbar_mw", "max"),
        )
        .merge(participants[merge_cols], on="province", how="left", validate="one_to_one")
    )

    diag["binary_redundant"] = diag["eligible_hours"] <= diag["H_touched"] + 1e-9
    diag["budget_slack_hours"] = diag["H_touched"] - diag["eligible_hours"]
    diag["budget_excess_hours"] = np.maximum(diag["eligible_hours"] - diag["H_touched"], 0)
    diag["touch_budget_to_eligible_ratio"] = diag["H_touched"] / diag["eligible_hours"].replace(0, np.nan)

    if "H_depth" in diag.columns:
        diag["depth_budget_to_support_ratio"] = diag["H_depth"] / diag["support_hours"].replace(0, np.nan)

    return diag.sort_values("province").reset_index(drop=True)


# =============================================================================
# RESULT CONSTRUCTION / SAVE
# =============================================================================

def build_result_from_values(
    yi: YearInput,
    state: pd.DataFrame,
    req: pd.DataFrame,
    participants: pd.DataFrame,
    x_val: np.ndarray,
    s_val: np.ndarray,
    y_val: np.ndarray,
    objective: float,
    runtime_sec: float,
    status: str,
    status_code: int,
    quality_flag: str,
    best_lower_bound: Optional[float],
    mip_gap: Optional[float],
    best_bound: Optional[float],
    node_count: Optional[float],
    solution_count: Optional[int],
    model_rows: Optional[int],
    model_cols: Optional[int],
    model_nnz: Optional[int],
    n_binary_rows_active: int,
    n_binary_rows_total: int,
    notes: List[str],
    save: bool = True,
) -> M1Result:
    c = state["cost_eur_per_mwh"].to_numpy(dtype=float)
    slack_penalty = float(participants["slack_penalty"].median())

    total_req = float(req["R_mw"].sum())
    total_slack = float(np.asarray(s_val, dtype=float).sum())
    served = 1.0 - total_slack / total_req if total_req > 0 else np.nan

    op_cost = float(np.dot(c, x_val))
    slack_cost = float(slack_penalty * total_slack)
    gap = certified_gap(objective, best_lower_bound)

    x_df = state[["province", "hour_of_year", "tdtr_support"]].copy()
    if "timestamp_utc" in state.columns:
        x_df["timestamp_utc"] = state["timestamp_utc"].to_numpy()

    pctr_map = participants.set_index("province")["Pctr_mw"].astype(float).to_dict()

    x_df["x_mw"] = np.asarray(x_val, dtype=float)
    x_df["touch_y_raw"] = np.asarray(y_val, dtype=float)
    x_df["touch_realized"] = (x_df["x_mw"].to_numpy(dtype=float) > TOUCH_TOL).astype(int)
    x_df["pbar_mw"] = state["pbar_mw"].to_numpy(dtype=float)
    x_df["Pctr_mw"] = x_df["province"].map(pctr_map).astype(float)
    x_df["cost_eur_per_mwh"] = c
    x_df["debit"] = x_df["touch_realized"].astype(float)
    x_df["rule"] = "M1"
    x_df["algorithm"] = quality_flag
    x_df["year"] = yi.year

    slack_df = req[["branch_id", "hour_of_year", "R_mw"]].copy()
    slack_df["slack_mw"] = np.asarray(s_val, dtype=float)
    slack_df["rule"] = "M1"
    slack_df["algorithm"] = quality_flag
    slack_df["year"] = yi.year

    budget = (
        x_df.groupby("province", as_index=False)
        .agg(
            total_x_mwh=("x_mw", "sum"),
            total_debit=("debit", "sum"),
            positive_hours=("touch_realized", "sum"),
            raw_y_sum=("touch_y_raw", "sum"),
            support_hours=("tdtr_support", "sum"),
            positive_pbar_hours=("pbar_mw", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > TOUCH_TOL).sum())),
        )
        .merge(
            participants[["province", "Pctr_mw", "H_touched", "tdtr_support_hours", "touch_budget_to_support_ratio"]],
            on="province",
            how="left",
        )
    )

    if "H_depth" in participants.columns:
        budget = budget.merge(
            participants[["province", "H_depth", "depth_budget_to_support_ratio"]],
            on="province",
            how="left",
        )

    budget["budget_utilization"] = budget["total_debit"] / budget["H_touched"].replace(0, np.nan)
    budget["raw_y_utilization"] = budget["raw_y_sum"] / budget["H_touched"].replace(0, np.nan)
    budget["is_budget_binding"] = (budget["H_touched"] - budget["total_debit"]).abs() <= BUDGET_BINDING_TOL
    budget["binary_redundant"] = budget["positive_pbar_hours"] <= budget["H_touched"] + 1e-9
    budget["rule"] = "M1"
    budget["algorithm"] = quality_flag
    budget["year"] = yi.year

    total_touched = float(budget["total_debit"].sum())
    productivity = (total_req - total_slack) / total_touched if total_touched > 0 else np.nan

    result = M1Result(
        year=yi.year,
        status=status,
        status_code=status_code,
        quality_flag=quality_flag,
        objective=float(objective),
        runtime_sec=float(runtime_sec),
        x=x_df,
        slack=slack_df,
        budget=budget,
        total_relief_requirement=total_req,
        total_slack=total_slack,
        operating_cost_x=op_cost,
        slack_penalty_cost=slack_cost,
        relief_served_share=float(served),
        total_touched_tokens=total_touched,
        allowance_productivity=float(productivity),
        best_lower_bound=best_lower_bound,
        certified_gap=gap,
        mip_gap=mip_gap,
        best_bound=best_bound,
        node_count=node_count,
        solution_count=solution_count,
        model_rows=model_rows,
        model_cols=model_cols,
        model_nnz=model_nnz,
        n_binary_rows_active=int(n_binary_rows_active),
        n_binary_rows_total=int(n_binary_rows_total),
        notes=notes,
    )

    if save:
        save_m1_result(result)

    return result


def save_m1_result(result: M1Result) -> None:
    save_table(result.x, SOLUTION_DIR / f"M1_province_hour_{result.year}.csv.gz")
    save_table(
        result.slack[result.slack["slack_mw"] > POSITIVE_SLACK_TOL].copy(),
        SOLUTION_DIR / f"M1_positive_slack_{result.year}.csv.gz",
    )
    save_table(result.budget, SOLUTION_DIR / f"M1_budget_summary_{result.year}.csv")


# =============================================================================
# EXACT LP SOLVER
# =============================================================================

def solve_m1_exact_lp_when_binary_redundant(
    yi: YearInput,
    state: pd.DataFrame,
    req: pd.DataFrame,
    coeff: pd.DataFrame,
    participants: pd.DataFrame,
    reason: str,
) -> M1Result:
    t0 = time.time()

    n_x = len(state)
    n_s = len(req)

    pbar = state["pbar_mw"].to_numpy(dtype=float)
    c = state["cost_eur_per_mwh"].to_numpy(dtype=float)
    rhs = req["R_mw"].to_numpy(dtype=float)
    slack_penalty = float(participants["slack_penalty"].median())

    if REQUIRE_POSITIVE_SLACK_PENALTY and slack_penalty <= 0:
        raise ValueError(f"M1 year={yi.year}: slack_penalty={slack_penalty:g}. Fix A output.")

    A_relief_x = build_relief_x_matrix(req, coeff, n_x=n_x)

    model = gp.Model(f"M1_exact_support_LP_{yi.year}")
    set_common_params(model, output_flag=OUTPUT_FLAG)

    x = model.addMVar(n_x, lb=np.zeros(n_x), ub=pbar, vtype=GRB.CONTINUOUS, name="x")
    s = model.addMVar(n_s, lb=np.zeros(n_s), vtype=GRB.CONTINUOUS, name="s")

    model.addConstr(A_relief_x @ x + s >= rhs, name="relief")
    model.setObjective(c @ x + slack_penalty * s.sum(), GRB.MINIMIZE)
    model.optimize()

    status = status_name_from_code(int(model.Status))
    if model.SolCount <= 0:
        raise RuntimeError(f"Exact support LP for year={yi.year} produced no solution. Status={status}")

    x_val = np.asarray(x.X, dtype=float)
    s_val = np.asarray(s.X, dtype=float)
    y_val = (x_val > TOUCH_TOL).astype(float)

    obj = float(model.ObjVal)
    runtime = time.time() - t0

    diag = support_budget_diagnostic(state, participants)
    save_table(diag, SOLUTION_DIR / f"M1_support_budget_diagnostic_{yi.year}.csv")

    notes = [
        reason,
        "M1 solved as an exact LP because eligible positive-Pbar hours do not exceed touched-token budgets.",
        "This is not a relaxation under the detected support/budget condition.",
    ]

    return build_result_from_values(
        yi=yi,
        state=state,
        req=req,
        participants=participants,
        x_val=x_val,
        s_val=s_val,
        y_val=y_val,
        objective=obj,
        runtime_sec=runtime,
        status=status,
        status_code=int(model.Status),
        quality_flag="exact_lp_binary_redundant_under_tdtr_support",
        best_lower_bound=obj,
        mip_gap=0.0,
        best_bound=obj,
        node_count=0.0,
        solution_count=int(model.SolCount),
        model_rows=int(safe_attr(model, "NumConstrs", 0)),
        model_cols=int(safe_attr(model, "NumVars", 0)),
        model_nnz=int(safe_attr(model, "NumNZs", 0)),
        n_binary_rows_active=0,
        n_binary_rows_total=len(state),
        notes=notes,
        save=True,
    )


# =============================================================================
# COMPACT SUPPORT-RESTRICTED MIP SOLVER
# =============================================================================

def solve_m1_compact_support_mip(
    yi: YearInput,
    state: pd.DataFrame,
    req: pd.DataFrame,
    coeff: pd.DataFrame,
    participants: pd.DataFrame,
) -> M1Result:
    t0 = time.time()

    n_x = len(state)
    n_s = len(req)

    pbar = state["pbar_mw"].to_numpy(dtype=float)
    c = state["cost_eur_per_mwh"].to_numpy(dtype=float)
    rhs = req["R_mw"].to_numpy(dtype=float)
    slack_penalty = float(participants["slack_penalty"].median())

    if REQUIRE_POSITIVE_SLACK_PENALTY and slack_penalty <= 0:
        raise ValueError(f"M1 year={yi.year}: slack_penalty={slack_penalty:g}. Fix A output.")

    A_relief_x = build_relief_x_matrix(req, coeff, n_x=n_x)

    part_order = participants.sort_values("province").reset_index(drop=True).copy()
    part_order["part_idx"] = np.arange(len(part_order), dtype=int)

    st = state[["province", "hour_of_year"]].copy()
    st = st.merge(part_order[["province", "part_idx"]], on="province", how="left", validate="many_to_one")
    if st["part_idx"].isna().any():
        raise RuntimeError("State has province rows absent from participant table.")

    part_idx = st["part_idx"].to_numpy(dtype=int)
    H_touch = part_order["H_touched"].to_numpy(dtype=float)

    eligible_binary_ub = (pbar > TOUCH_TOL).astype(float)
    n_active_binary = int(eligible_binary_ub.sum())
    n_full_binary = int(len(eligible_binary_ub))

    if M1_REQUIRE_SUPPORT_REDUCTION and n_active_binary > 0.50 * n_full_binary:
        raise RuntimeError(
            f"Compact MIP was asked to solve with {n_active_binary:,}/{n_full_binary:,} active binaries. "
            "This looks like the old full-year M1. Check that A imposed TDTR support and Pbar=0 outside support."
        )

    model = gp.Model(f"M1_compact_support_MIP_{yi.year}")
    model.Params.OutputFlag = M1_COMPACT_MIP_OUTPUT_FLAG
    if THREADS > 0:
        model.Params.Threads = THREADS
    model.Params.TimeLimit = M1_COMPACT_MIP_TIME_LIMIT
    model.Params.MIPGap = M1_COMPACT_MIP_GAP
    model.Params.MIPFocus = M1_COMPACT_MIP_FOCUS
    model.Params.Heuristics = M1_COMPACT_MIP_HEURISTICS
    model.Params.Cuts = M1_COMPACT_MIP_CUTS
    model.Params.Presolve = M1_COMPACT_MIP_PRESOLVE
    model.Params.NodefileStart = M1_COMPACT_NODEFILE_START

    x = model.addMVar(n_x, lb=np.zeros(n_x), ub=pbar, vtype=GRB.CONTINUOUS, name="x")
    s = model.addMVar(n_s, lb=np.zeros(n_s), vtype=GRB.CONTINUOUS, name="s")
    y = model.addMVar(n_x, lb=np.zeros(n_x), ub=eligible_binary_ub, vtype=GRB.BINARY, name="y")

    model.addConstr(A_relief_x @ x + s >= rhs, name="relief")
    model.addConstr(x <= pbar * y, name="touch_link")

    for i in range(len(part_order)):
        rows_i = np.where(part_idx == i)[0]
        model.addConstr(y[rows_i].sum() <= H_touch[i], name=f"touch_budget[{i}]")

    model.setObjective(c @ x + slack_penalty * s.sum(), GRB.MINIMIZE)
    model.optimize()

    status = status_name_from_code(int(model.Status))
    if model.SolCount <= 0:
        raise RuntimeError(f"Compact support MIP for year={yi.year} produced no incumbent. Status={status}")

    x_val = np.asarray(x.X, dtype=float)
    s_val = np.asarray(s.X, dtype=float)

    # Realized M1 debit is based on x > 0, not raw y, to avoid degenerate y=1, x=0.
    y_val = (x_val > TOUCH_TOL).astype(float)

    obj = float(model.ObjVal)
    runtime = time.time() - t0

    diag = support_budget_diagnostic(state, participants)
    save_table(diag, SOLUTION_DIR / f"M1_support_budget_diagnostic_{yi.year}.csv")

    mip_gap_raw = safe_attr(model, "MIPGap", np.nan)
    best_bound_raw = safe_attr(model, "ObjBound", np.nan)
    node_count_raw = safe_attr(model, "NodeCount", np.nan)

    mip_gap = float(mip_gap_raw) if mip_gap_raw is not None and np.isfinite(float(mip_gap_raw)) else None
    best_bound = float(best_bound_raw) if best_bound_raw is not None and np.isfinite(float(best_bound_raw)) else None
    node_count = float(node_count_raw) if node_count_raw is not None and np.isfinite(float(node_count_raw)) else None

    if mip_gap is not None and mip_gap <= M1_COMPACT_MIP_GAP + 1e-9:
        qflag = "compact_support_mip_gap_acceptable"
    elif status == "OPTIMAL":
        qflag = "compact_support_mip_optimal"
    else:
        qflag = "compact_support_mip_feasible_or_time_limited"

    notes = [
        f"Compact support-restricted MIP used with {n_active_binary:,} active binary rows out of {n_full_binary:,}.",
        "M1 debit reported from realized positive x, not raw y.",
    ]

    return build_result_from_values(
        yi=yi,
        state=state,
        req=req,
        participants=participants,
        x_val=x_val,
        s_val=s_val,
        y_val=y_val,
        objective=obj,
        runtime_sec=runtime,
        status=status,
        status_code=int(model.Status),
        quality_flag=qflag,
        best_lower_bound=best_bound,
        mip_gap=mip_gap,
        best_bound=best_bound,
        node_count=node_count,
        solution_count=int(safe_attr(model, "SolCount", 0)),
        model_rows=int(safe_attr(model, "NumConstrs", 0)),
        model_cols=int(safe_attr(model, "NumVars", 0)),
        model_nnz=int(safe_attr(model, "NumNZs", 0)),
        n_binary_rows_active=n_active_binary,
        n_binary_rows_total=n_full_binary,
        notes=notes,
        save=True,
    )


# =============================================================================
# DISPATCHER
# =============================================================================

def solve_m1(yi: YearInput) -> M1Result:
    state, req, coeff, participants = build_indices(yi)

    if M1_TIGHTEN_USEFUL_PBAR:
        state, tighten_summary = tighten_useful_pbar(state, req, coeff)
        print(f"[TIGHTEN] {tighten_summary}")
    else:
        tighten_summary = {"skipped": True, "reason": "disabled"}

    support_diag = support_budget_diagnostic(state, participants)
    save_table(support_diag, SOLUTION_DIR / f"M1_support_budget_diagnostic_{yi.year}.csv")

    eligible_total = int(support_diag["eligible_hours"].sum())
    support_total = int(support_diag["support_hours"].sum())
    redundant = bool(support_diag["binary_redundant"].all())

    print(
        f"[SUPPORT] positive-Pbar eligible province-hours={eligible_total:,} | "
        f"support province-hours={support_total:,} | binary_redundant={redundant}"
    )

    display_cols = ["province", "eligible_hours", "support_hours", "H_touched", "budget_slack_hours", "binary_redundant"]
    if "H_depth" in support_diag.columns:
        display_cols += ["H_depth", "depth_budget_to_support_ratio"]

    print("[SUPPORT] per-province eligible/Hbar:\n" + support_diag[display_cols].to_string(index=False))

    if M1_SOLVE_MODE == "cg":
        raise RuntimeError(
            "M1_SOLVE_MODE=cg was requested, but this updated B1 intentionally removes the old column-generation path. "
            "Use M1_SOLVE_MODE=auto. With support <= H_touch, M1 solves exactly as LP; otherwise compact MIP."
        )

    if M1_SOLVE_MODE in {"auto", "lp_if_free"} and redundant:
        return solve_m1_exact_lp_when_binary_redundant(
            yi=yi,
            state=state,
            req=req,
            coeff=coeff,
            participants=participants,
            reason=f"Detected eligible_hours <= H_touched for every province. Tightening summary: {tighten_summary}",
        )

    if M1_SOLVE_MODE == "lp_if_free" and not redundant:
        raise RuntimeError(
            "M1_SOLVE_MODE=lp_if_free but binaries are not redundant. Use M1_SOLVE_MODE=auto or compact_mip."
        )

    if M1_SOLVE_MODE in {"auto", "compact_mip"}:
        return solve_m1_compact_support_mip(
            yi=yi,
            state=state,
            req=req,
            coeff=coeff,
            participants=participants,
        )

    raise ValueError(f"Unknown M1_SOLVE_MODE={M1_SOLVE_MODE!r}. Use one of: auto, lp_if_free, compact_mip.")


# =============================================================================
# SUMMARY / MANIFEST
# =============================================================================

def result_summary_row(result: M1Result) -> Dict[str, object]:
    return {
        "year": result.year,
        "rule": "M1",
        "algorithm": result.quality_flag,
        "status": result.status,
        "status_code": result.status_code,
        "quality_flag": result.quality_flag,
        "objective": result.objective,
        "operating_cost_x": result.operating_cost_x,
        "slack_penalty_cost": result.slack_penalty_cost,
        "total_slack": result.total_slack,
        "total_relief_requirement": result.total_relief_requirement,
        "relief_served_share": result.relief_served_share,
        "total_touched_tokens": result.total_touched_tokens,
        "allowance_productivity": result.allowance_productivity,
        "mean_budget_utilization": float(result.budget["budget_utilization"].mean()) if not result.budget.empty else np.nan,
        "max_budget_utilization": float(result.budget["budget_utilization"].max()) if not result.budget.empty else np.nan,
        "n_budget_binding": int(result.budget["is_budget_binding"].sum()) if "is_budget_binding" in result.budget.columns else 0,
        "mean_touch_budget_to_support_ratio": float(result.budget["touch_budget_to_support_ratio"].mean()) if "touch_budget_to_support_ratio" in result.budget.columns else np.nan,
        "mean_depth_budget_to_support_ratio": float(result.budget["depth_budget_to_support_ratio"].mean()) if "depth_budget_to_support_ratio" in result.budget.columns else np.nan,
        "best_lower_bound": result.best_lower_bound,
        "certified_gap": result.certified_gap,
        "mip_gap": result.mip_gap,
        "best_bound": result.best_bound,
        "node_count": result.node_count,
        "solution_count": result.solution_count,
        "model_rows": result.model_rows,
        "model_cols": result.model_cols,
        "model_nnz": result.model_nnz,
        "n_binary_rows_active": result.n_binary_rows_active,
        "n_binary_rows_total": result.n_binary_rows_total,
        "runtime_sec": result.runtime_sec,
        "notes": " | ".join(result.notes),
    }


def write_manifest(summary_rows: List[Dict[str, object]], year_metadata: Dict[int, Dict[str, object]]) -> None:
    summary_df = pd.DataFrame(summary_rows)

    manifest = {
        "script": "B1_run_m1_touched_only.py",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "repo_root": str(REPO_ROOT),
        "input_dir": str(INPUT_DIR),
        "results_dir": str(RESULTS_DIR),
        "solution_dir": str(SOLUTION_DIR),
        "years": YEARS,
        "algorithm": "support-aware M1 touched-time solver",
        "budget_policy": {
            "A_is_source_of_truth": True,
            "require_A_budgets": REQUIRE_A_BUDGETS,
            "touch_budget_interpretation": "H_touched should be written by A; usually selected TDTR support hours",
            "depth_budget_audited_if_present": True,
            "fallback_touch_budget": "support_hours only if REQUIRE_A_BUDGETS=0",
        },
        "model": {
            "rule": "M1",
            "debit": "d_{i,t}=1{x_{i,t}>0}",
            "exact_lp_condition": "eligible positive-Pbar hours per participant <= touched-token allowance",
            "compact_mip_condition": "eligible support wider than touched-token allowance",
        },
        "schema_robustness": {
            "alias_coalescing": True,
            "duplicate_alias_columns_removed": True,
            "participant_capacity_aliases": ALIASES["pctr"],
            "touch_budget_aliases": ALIASES["touch_budget"],
        },
        "settings": {
            "m1_solve_mode": M1_SOLVE_MODE,
            "m1_tighten_useful_pbar": M1_TIGHTEN_USEFUL_PBAR,
            "compact_mip_time_limit": M1_COMPACT_MIP_TIME_LIMIT,
            "compact_mip_gap": M1_COMPACT_MIP_GAP,
            "compact_mip_focus": M1_COMPACT_MIP_FOCUS,
            "compact_mip_heuristics": M1_COMPACT_MIP_HEURISTICS,
            "compact_mip_cuts": M1_COMPACT_MIP_CUTS,
            "compact_mip_presolve": M1_COMPACT_MIP_PRESOLVE,
            "require_support_reduction": M1_REQUIRE_SUPPORT_REDUCTION,
            "touch_tol": TOUCH_TOL,
            "slack_penalty_fallback": DEFAULT_SLACK_PENALTY,
            "require_positive_slack_penalty": REQUIRE_POSITIVE_SLACK_PENALTY,
        },
        "gurobi": {
            "version": gp.gurobi.version(),
            "output_flag": OUTPUT_FLAG,
            "threads": THREADS,
            "lp_method": LP_METHOD,
        },
        "quality_counts": summary_df["quality_flag"].value_counts().to_dict() if not summary_df.empty else {},
        "year_metadata_from_A": year_metadata,
        "python": sys.version,
        "platform": platform.platform(),
        "interpretation_note": (
            "If H_touched equals the selected support size, M1 typically solves as exact LP because the touched budget "
            "is implied by support. If support is wider than H_touched, B1 solves the compact support-restricted MILP."
        ),
    }

    path = RESULTS_DIR / "m1_only_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[SAVE] {path}")


def print_header() -> None:
    print("=" * 78)
    print("B1: RUN SUPPORT-AWARE M1 TOUCHED-TIME SOLVER")
    print("=" * 78)
    print(f"Repo root       : {REPO_ROOT}")
    print(f"Input dir       : {INPUT_DIR}")
    print(f"Results dir     : {RESULTS_DIR}")
    print(f"Solution dir    : {SOLUTION_DIR}")
    print(f"Years           : {YEARS}")
    print(f"Provinces       : {PROVINCES}")
    print(f"M1 solve mode   : {M1_SOLVE_MODE}")
    print("Budget source   : A participant file; no old full-year budget rebuild")
    print(f"Require A budget: {REQUIRE_A_BUDGETS}")
    print("Exact LP test   : eligible positive-Pbar hours <= H_touched")
    print(f"Compact MIP TL/gap: {M1_COMPACT_MIP_TIME_LIMIT}s / {M1_COMPACT_MIP_GAP}")
    print(f"Require support reduction: {M1_REQUIRE_SUPPORT_REDUCTION}")
    print(f"Tighten Pbar    : {M1_TIGHTEN_USEFUL_PBAR}")
    print(f"A costs required: {REQUIRE_A_COSTS}")
    print(f"Slack positive  : {REQUIRE_POSITIVE_SLACK_PENALTY}")
    print(f"Threads         : {THREADS if THREADS > 0 else 'Gurobi default'}")
    print(f"Slack fallback  : {DEFAULT_SLACK_PENALTY:g}")
    print("=" * 78)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dirs()
    print_header()

    summary_rows: List[Dict[str, object]] = []
    year_metadata: Dict[int, Dict[str, object]] = {}

    for year in YEARS:
        print("\n" + "=" * 78)
        print(f"[YEAR] {year}")
        print("=" * 78)

        yi = load_year_input(year)
        year_metadata[year] = dict(yi.metadata)

        cost_desc = yi.state["cost_eur_per_mwh"].describe(percentiles=[0.01, 0.5, 0.99]).to_dict()
        slack_penalty = float(yi.participants["slack_penalty"].median())

        support_hours_by_year = (
            yi.state.groupby("hour_of_year", as_index=False)["tdtr_support"].max()["tdtr_support"].sum()
        )

        touch_ratio = yi.participants["H_touched"].mean() / max(float(support_hours_by_year), 1.0)

        if "H_depth" in yi.participants.columns:
            depth_ratio = yi.participants["H_depth"].mean() / max(float(support_hours_by_year), 1.0)
        else:
            depth_ratio = np.nan

        cost_source = yi.state["cost_source"].iloc[0] if "cost_source" in yi.state.columns else "A_output_cost_eur_per_mwh"

        print(
            f"[DATA] constraints={len(yi.requirements):,} | coeffs={len(yi.coefficients):,} | "
            f"state rows={len(yi.state):,} | total R={yi.total_requirement:,.3f} | slack penalty={slack_penalty:g}"
        )

        print(
            f"[TDTR-M1] support hours={int(support_hours_by_year):,} | "
            f"H_touch mean={yi.participants['H_touched'].mean():.3f} | "
            f"H_touch/support={touch_ratio:.3f} | "
            f"H_touch min={yi.participants['H_touched'].min()} | "
            f"H_touch max={yi.participants['H_touched'].max()} | "
            f"H_depth/support audit={depth_ratio:.3f}"
        )

        print(
            f"[COST] source={cost_source} | "
            f"min={cost_desc.get('min', np.nan):.3f}, "
            f"p50={cost_desc.get('50%', np.nan):.3f}, "
            f"p99={cost_desc.get('99%', np.nan):.3f}, "
            f"max={cost_desc.get('max', np.nan):.3f} €/MW-h"
        )

        result = solve_m1(yi)
        summary_rows.append(result_summary_row(result))

        print(
            f"[M1] objective={result.objective:,.3f} | bestBound={result.best_bound} | "
            f"certGap={result.certified_gap} | slack={result.total_slack:,.6f} | "
            f"served={result.relief_served_share:.6f} | runtime={result.runtime_sec:.1f}s | quality={result.quality_flag}"
        )

        for note in result.notes:
            print(f"[NOTE] {note}")

    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "m1_only_summary.csv"
    save_table(summary, summary_path)

    write_manifest(summary_rows, year_metadata)

    print("\n[DONE]")
    print(f"Summary : {summary_path}")
    print(f"Manifest: {RESULTS_DIR / 'm1_only_manifest.json'}")


if __name__ == "__main__":
    main()
