#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B_run_annual_lp.py
==================

Annual TDTR debit-rule LP/QP solver module for the annual debit-rule paper.

FIXED VERSION
-------------
This version fixes the duplicate-alias bug that caused

    TypeError: arg must be a list, tuple, 1-d array, or Series

when participant files contained multiple aliases for the same quantity, e.g.
P_ctr_MW and P_i, or H_depth and H_i. The script no longer blindly renames
alias columns. Instead, it canonicalizes them by coalescing all matching aliases
into one clean output column and removing the duplicate source columns.

A_build_debit_optimization_inputs.py is the source of truth for:
    - TDTR candidate-hour support;
    - Pbar after support masking;
    - operating costs;
    - annual touched-time budget H_touched;
    - annual depth budget H_depth;
    - slack penalty;
    - relief requirements R and coefficients a.

Core public API used by C_train_m3.py and D_run_full_benchmark.py:
    load_year_input
    solve_weighted_depth_lp
    solve_weighted_depth_qp
    kkt_equalization_diagnostic
    result_summary_row
    save_result
    load_q_file_for_year
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover
    raise RuntimeError("gurobipy is required for B_run_annual_lp.py.") from exc

try:
    import scipy.sparse as sp
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy.sparse is required for sparse matrix construction.") from exc


# =============================================================================
# Path setup and optional pipeline_config
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else Path.cwd()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import pipeline_config as cfg  # type: ignore
except Exception:  # pragma: no cover
    cfg = None


def cfg_get(name: str, default: Any) -> Any:
    if cfg is not None and hasattr(cfg, name):
        value = getattr(cfg, name)
        if value is not None and value != "":
            return value
    return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def env_int_list(name: str, default: Iterable[int]) -> List[int]:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return [int(x) for x in default]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PROVINCES = [
    "Groningen", "Friesland", "Drenthe", "Overijssel", "Flevoland", "Gelderland",
    "Utrecht", "Noord-Holland", "Zuid-Holland", "Zeeland", "Noord-Brabant", "Limburg",
]

INPUT_DIR = Path(os.getenv(
    "DEBIT_INPUT_DIR",
    str(cfg_get("DEBIT_OPT_INPUT_DIR", Path(cfg_get("DERIVED_DIR", REPO_ROOT / "data" / "derived")) / "debit_optimization_inputs")),
))

RESULTS_DIR = Path(os.getenv(
    "DEBIT_RESULTS_DIR",
    str(cfg_get("RESULTS_DIR", REPO_ROOT / "outputs" / "results")),
))

SOLUTION_DIR = Path(os.getenv(
    "DEBIT_SOLUTION_DIR",
    str(RESULTS_DIR / "solutions"),
))

M1_ONLY_RESULTS_DIR = Path(os.getenv(
    "M1_ONLY_RESULTS_DIR",
    str(RESULTS_DIR / "m1_only"),
))

YEARS = env_int_list("DEBIT_YEARS", cfg_get("ALL_YEARS", [2021, 2022, 2023, 2024, 2025]))
PROVINCES = list(cfg_get("NL_PROVINCES", DEFAULT_PROVINCES))

SOLVE_M2 = env_bool("SOLVE_M2", True)
SOLVE_M1 = env_bool("SOLVE_M1", False)
INCLUDE_B1_M1_RESULTS = env_bool("INCLUDE_B1_M1_RESULTS", True)
RUN_QP_SELF_TEST = env_bool("RUN_QP_SELF_TEST", False)

OUTPUT_FLAG = int(os.getenv("GUROBI_OUTPUT_FLAG", "1"))
THREADS = int(os.getenv("GUROBI_THREADS", "0"))
LP_METHOD = int(os.getenv("GUROBI_LP_METHOD", "-1"))

MIP_TIME_LIMIT = float(os.getenv("M1_TIME_LIMIT", "180"))
MIP_GAP = float(os.getenv("M1_MIP_GAP", "0.01"))
MIP_FOCUS = int(os.getenv("M1_MIP_FOCUS", "1"))
MIP_HEURISTICS = float(os.getenv("M1_HEURISTICS", "0.25"))
MIP_CUTS = int(os.getenv("M1_CUTS", "2"))
MIP_PRESOLVE = int(os.getenv("M1_PRESOLVE", "2"))
MIP_NODEFILE_START = float(os.getenv("M1_NODEFILE_START", "0.5"))
MIP_TINY_TOUCH_PENALTY = float(os.getenv("M1_TINY_TOUCH_PENALTY", "0.0"))
M1_WARMSTART_FROM_M2 = env_bool("M1_WARMSTART_FROM_M2", True)

QP_TAU_X = float(os.getenv("QP_TAU_X", "1e-6"))
QP_TAU_S = float(os.getenv("QP_TAU_S", "1e-8"))

TDTR_AVAILABILITY_SHARE = float(os.getenv("TDTR_AVAILABILITY_SHARE", str(cfg_get("TDTR_AVAILABILITY_SHARE", 0.85))))
TDTR_RESTRICTION_SHARE = float(os.getenv("TDTR_RESTRICTION_SHARE", str(1.0 - TDTR_AVAILABILITY_SHARE)))

TOUCH_TOL = float(os.getenv("TOUCH_TOL", "1e-7"))
POSITIVE_SLACK_TOL = float(os.getenv("POSITIVE_SLACK_TOL", "1e-7"))
BUDGET_BINDING_TOL = float(os.getenv("BUDGET_BINDING_TOL", "1e-5"))

REQUIRE_A_COSTS = env_bool("REQUIRE_A_COSTS", True)
REQUIRE_A_BUDGETS = env_bool("REQUIRE_A_BUDGETS", True)
TDTR_DEPTH_BUDGET_RHO = float(os.getenv(
    "TDTR_DEPTH_BUDGET_RHO",
    str(cfg_get("TDTR_DEPTH_BUDGET_RHO", cfg_get("DEBIT_DEPTH_BUDGET_RHO", 1.0))),
))
WARN_IF_DEPTH_BUDGET_RATIO_GE = float(os.getenv("WARN_IF_DEPTH_BUDGET_RATIO_GE", "0.95"))
REQUIRE_POSITIVE_SLACK_PENALTY = env_bool("REQUIRE_POSITIVE_SLACK_PENALTY", True)
WARN_IF_SLACK_PENALTY_LE_MAX_COST = env_bool("WARN_IF_SLACK_PENALTY_LE_MAX_COST", True)

M3_Q_FILE = os.getenv(
    "M3_Q_FILE",
    str(cfg_get("M3_WEIGHTS_PATH", REPO_ROOT / "data" / "derived" / "m3_debit_weights_by_year.parquet")),
).strip()
if M3_Q_FILE.lower() in {"", "none", "null"}:
    M3_Q_FILE = ""

M3_FEATURE_SET = os.getenv("M3_FEATURE_SET", str(cfg_get("MAIN_FEATURE_SET", "F2"))).strip()
M3_VARIANT = os.getenv("M3_VARIANT", str(cfg_get("MAIN_M3_VARIANT", "M3_LO"))).strip()
SOLVE_M3_FROM_Q_FILE = env_bool("SOLVE_M3_FROM_Q_FILE", env_bool("RUN_M3_IN_B", False))
M3_WRITE_Q_INVENTORY = env_bool("M3_WRITE_Q_INVENTORY", True)


# =============================================================================
# Aliases and data classes
# =============================================================================

ALIASES: Dict[str, List[str]] = {
    "province": ["province", "provincie", "node", "deployment_unit", "n", "area"],
    "timestamp": ["timestamp_utc", "timestamp", "datetime_utc", "time", "datetime", "utc_timestamp"],
    "hour": ["hour_of_year", "hour", "h", "t", "hour_index", "time_index", "hoy"],
    "branch": ["branch_id", "constraint_id", "constraint", "k", "line_id", "monitored_constraint", "element_id"],
    "requirement": ["R_MW", "relief_requirement_mw", "requirement_mw", "R_mw", "R", "relief_mw", "required_relief_mw"],
    "coefficient": ["coefficient", "a", "a_kit", "effectivity", "ptdf_effectivity", "relief_coeff", "signed_effectivity"],
    "pbar": ["pbar_mw", "Pbar_mw", "Pbar_value", "Pbar", "available_restrictable_mw", "availability_mw", "available_mw", "max_restriction_mw"],
    "support": ["tdtr_support", "is_tdtr_candidate", "candidate_hour", "support", "selected", "is_selected", "y_support"],
    "pctr": [
        "P_ctr_MW", "Pctr_mw", "pctr_mw", "P_i_ctr_mw", "P_i",
        "contract_capacity_mw", "contracted_capacity_mw", "Pctr", "p_ctr_mw",
    ],
    "depth_budget": [
        "H_depth_hours", "depth_budget_hours", "H_depth", "H_i", "H",
        "tdtr_depth_budget", "annual_depth_budget", "depth_budget", "H_depth_full_power_hours",
    ],
    "touch_budget": [
        "H_touch_tokens", "touched_budget_tokens", "H_touched", "Hbar_i", "Hbar",
        "Hbar_eff", "tdtr_touched_budget", "annual_touch_budget", "touch_budget",
    ],
    "cost": [
        "cost_eur_per_mwh", "restriction_cost", "c_value", "restriction_cost_eur_per_mwh",
        "restriction_cost_eur_per_mw", "cost", "c", "c_eur_per_mwh",
    ],
    "price": [
        "day_ahead_price_eur_per_mwh", "price_da_eur_mwh", "price_da_eur_per_mwh",
        "da_price_eur_per_mwh", "price_eur_per_mwh", "electricity_price_eur_mwh",
        "price", "spot_price",
    ],
    "debit_weight": ["debit_weight", "q", "weight", "q_weight", "m3_weight"],
    "feature_set": ["feature_set", "feature_level", "features", "level"],
    "variant": ["variant", "architecture", "m3_variant", "model_variant"],
    "slack_penalty": ["slack_penalty", "M", "M_k", "emergency_slack_penalty"],
}


@dataclass
class YearInput:
    year: int
    state: pd.DataFrame
    requirements: pd.DataFrame
    coefficients: pd.DataFrame
    participants: pd.DataFrame
    tariff: Optional[pd.DataFrame]
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def n_hours(self) -> int:
        return int(self.state["hour_of_year"].nunique())

    @property
    def provinces(self) -> List[str]:
        return list(self.participants["province"].astype(str))


@dataclass
class SolveResult:
    rule: str
    year: int
    status: str
    status_code: int
    objective: float
    runtime_sec: float
    x: pd.DataFrame
    slack: pd.DataFrame
    budget: pd.DataFrame
    kkt: pd.DataFrame
    total_relief_requirement: float
    total_slack: float
    operating_cost_x: float
    slack_penalty_cost: float
    relief_served_share: float
    total_debit: float
    allowance_productivity: float
    mip_gap: Optional[float] = None
    best_bound: Optional[float] = None
    node_count: Optional[float] = None
    solution_count: Optional[int] = None
    model_rows: Optional[int] = None
    model_cols: Optional[int] = None
    model_nnz: Optional[int] = None
    quality_flag: str = "unknown"
    notes: List[str] = field(default_factory=list)
    raw: Dict[str, object] = field(default_factory=dict)


# =============================================================================
# Robust column utilities
# =============================================================================

def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTION_DIR.mkdir(parents=True, exist_ok=True)


def read_table(base: Path, required: bool = True) -> Optional[pd.DataFrame]:
    if base.suffix:
        candidates = [base]
    else:
        candidates = [base.with_suffix(".parquet"), base.with_suffix(".csv"), Path(str(base) + ".csv.gz")]

    for path in candidates:
        if path.exists():
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            if str(path).endswith(".csv.gz"):
                return pd.read_csv(path, compression="gzip", low_memory=False)
            return pd.read_csv(path, low_memory=False)

    if required:
        raise FileNotFoundError(f"Missing required input table. Tried: {candidates}")
    return None


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".csv.gz"):
        df.to_csv(path, index=index, compression="gzip")
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index)
    print(f"[SAVE] {path}")


def _alias_positions(df: pd.DataFrame, key: str) -> List[int]:
    aliases = ALIASES.get(key, [key])
    lowers = {str(a).strip().lower() for a in aliases}
    return [i for i, col in enumerate(df.columns) if str(col).strip().lower() in lowers]


def find_col(df: pd.DataFrame, key: str, required: bool = True, context: str = "") -> Optional[str]:
    positions = _alias_positions(df, key)
    if positions:
        return str(df.columns[positions[0]])
    if required:
        aliases = ALIASES.get(key, [key])
        raise KeyError(
            f"Could not find column for {key!r} in {context or 'dataframe'}. "
            f"Tried aliases={aliases}. Available columns={list(df.columns)}"
        )
    return None


def coalesce_numeric_columns(
    df: pd.DataFrame,
    key: str,
    out_col: str,
    *,
    required: bool,
    context: str,
    allow_all_missing: bool = False,
) -> pd.DataFrame:
    """
    Coalesce all aliases of `key` into a single numeric `out_col` and remove
    the source alias columns. This prevents duplicate-column DataFrame returns
    from df[out_col].
    """
    out = df.copy()
    positions = _alias_positions(out, key)

    if not positions:
        if required:
            raise KeyError(f"Could not find any aliases for {out_col!r} in {context}. Available={list(out.columns)}")
        return out

    pieces: List[pd.Series] = []
    source_names: List[str] = []
    for pos in positions:
        source_names.append(str(out.columns[pos]))
        pieces.append(pd.to_numeric(out.iloc[:, pos], errors="coerce"))

    value = pieces[0]
    for s in pieces[1:]:
        value = value.combine_first(s)

    if value.isna().any() and not allow_all_missing:
        n_bad = int(value.isna().sum())
        raise ValueError(
            f"{context}: canonical column {out_col!r} has {n_bad} missing/non-numeric values after "
            f"coalescing aliases {source_names}."
        )

    keep_positions = [i for i in range(out.shape[1]) if i not in set(positions)]
    out2 = out.iloc[:, keep_positions].copy()
    out2[out_col] = value.to_numpy()
    return out2


def coalesce_text_columns(
    df: pd.DataFrame,
    key: str,
    out_col: str,
    *,
    required: bool,
    context: str,
) -> pd.DataFrame:
    out = df.copy()
    positions = _alias_positions(out, key)
    if not positions:
        if required:
            raise KeyError(f"Could not find any aliases for {out_col!r} in {context}. Available={list(out.columns)}")
        return out

    pieces = [out.iloc[:, pos].astype("string") for pos in positions]
    value = pieces[0]
    for s in pieces[1:]:
        value = value.combine_first(s)

    if value.isna().any():
        n_bad = int(value.isna().sum())
        raise ValueError(f"{context}: canonical column {out_col!r} has {n_bad} missing values after coalescing.")

    keep_positions = [i for i in range(out.shape[1]) if i not in set(positions)]
    out2 = out.iloc[:, keep_positions].copy()
    out2[out_col] = value.astype(str).to_numpy()
    return out2


def to_numeric_clean(s: Union[pd.Series, pd.DataFrame, np.ndarray, List[Any]], name: str) -> pd.Series:
    if isinstance(s, pd.DataFrame):
        if s.shape[1] != 1:
            raise TypeError(
                f"Column {name} resolved to a DataFrame with {s.shape[1]} columns. "
                "Use coalesce_numeric_columns before numeric conversion."
            )
        s = s.iloc[:, 0]
    out = pd.to_numeric(s, errors="coerce")
    if isinstance(out, np.ndarray):
        out = pd.Series(out)
    if out.isna().any():
        n_bad = int(out.isna().sum())
        raise ValueError(f"Column {name} contains {n_bad} nonnumeric/missing values after conversion.")
    return out.astype(float)


def add_timestamp_and_hour(df: pd.DataFrame, year: int, context: str) -> pd.DataFrame:
    out = df.copy()

    if _alias_positions(out, "timestamp"):
        out = coalesce_text_columns(out, "timestamp", "timestamp_utc", required=False, context=context)
    if _alias_positions(out, "hour"):
        out = coalesce_numeric_columns(out, "hour", "hour_of_year", required=False, context=context)

    if "timestamp_utc" in out.columns:
        ts = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True)
        if ts.isna().any():
            raise ValueError(f"{context}: timestamp_utc contains unparseable timestamps.")
        out["timestamp_utc"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S%z")
        if "hour_of_year" not in out.columns:
            start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
            out["hour_of_year"] = ((ts - start) / pd.Timedelta(hours=1)).round().astype(int) + 1

    if "hour_of_year" in out.columns:
        out["hour_of_year"] = pd.to_numeric(out["hour_of_year"], errors="raise").astype(int)

    return out


def infer_hour_from_timestamp_by_year(df: pd.DataFrame, context: str) -> pd.DataFrame:
    out = df.copy()

    if _alias_positions(out, "hour"):
        out = coalesce_numeric_columns(out, "hour", "hour_of_year", required=False, context=context)
        out["hour_of_year"] = pd.to_numeric(out["hour_of_year"], errors="raise").astype(int)
        return out

    if _alias_positions(out, "timestamp"):
        out = coalesce_text_columns(out, "timestamp", "timestamp_utc", required=True, context=context)
    else:
        raise KeyError(f"{context}: must contain hour_of_year or timestamp_utc.")

    if "year" not in out.columns:
        raise KeyError(f"{context}: timestamp-based hour inference requires a year column.")

    ts = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True)
    if ts.isna().any():
        raise ValueError(f"{context}: timestamp_utc contains unparseable timestamps.")

    yr = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    if yr.isna().any():
        raise ValueError(f"{context}: year contains missing/unparseable values.")

    hour = np.empty(len(out), dtype=int)
    for y in sorted(yr.dropna().unique()):
        mask = yr == y
        start = pd.Timestamp(f"{int(y)}-01-01 00:00:00", tz="UTC")
        hour[mask.to_numpy()] = ((ts[mask] - start) / pd.Timedelta(hours=1)).round().astype(int).to_numpy() + 1

    out["timestamp_utc"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    out["hour_of_year"] = hour
    return out


def align_hour_basis(reference_hours: Iterable[int], df: pd.DataFrame, context: str) -> pd.DataFrame:
    out = df.copy()
    if "hour_of_year" not in out.columns:
        return out

    ref = set(int(x) for x in reference_hours)
    hrs = set(int(x) for x in out["hour_of_year"].dropna().unique())

    if hrs and hrs.issubset(ref):
        return out

    hrs_minus = set(h - 1 for h in hrs)
    if hrs_minus and hrs_minus.issubset(ref):
        out["hour_of_year"] = out["hour_of_year"].astype(int) - 1
        print(f"[INFO] {context}: converted hour_of_year by -1 to match state panel.")
        return out

    hrs_plus = set(h + 1 for h in hrs)
    if hrs_plus and hrs_plus.issubset(ref):
        out["hour_of_year"] = out["hour_of_year"].astype(int) + 1
        print(f"[INFO] {context}: converted hour_of_year by +1 to match state panel.")
        return out

    overlap = len(hrs & ref)
    if overlap == 0 and hrs:
        raise ValueError(
            f"{context}: hour_of_year basis does not overlap state panel. "
            f"Example hours={sorted(list(hrs))[:5]}, state hours={sorted(list(ref))[:5]}"
        )

    return out


def status_name(model: gp.Model) -> str:
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
    return mapping.get(int(model.Status), f"STATUS_{int(model.Status)}")


def safe_attr(model: gp.Model, attr: str, default: object = None) -> object:
    try:
        return getattr(model, attr)
    except Exception:
        return default


def gurobi_params(model: gp.Model, output_flag: int = OUTPUT_FLAG, is_mip: bool = False) -> None:
    model.Params.OutputFlag = int(output_flag)
    if THREADS > 0:
        model.Params.Threads = THREADS
    if LP_METHOD >= 0:
        model.Params.Method = LP_METHOD
    if is_mip:
        model.Params.MIPGap = MIP_GAP
        model.Params.TimeLimit = MIP_TIME_LIMIT
        model.Params.MIPFocus = MIP_FOCUS
        model.Params.Heuristics = MIP_HEURISTICS
        model.Params.Cuts = MIP_CUTS
        model.Params.Presolve = MIP_PRESOLVE
        model.Params.NodefileStart = MIP_NODEFILE_START


def load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_tdtr_support_hours_from_state(state: pd.DataFrame) -> int:
    if "hour_of_year" not in state.columns:
        raise KeyError("State must contain hour_of_year to infer support hours.")

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
# Input normalization
# =============================================================================

def normalize_state(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = add_timestamp_and_hour(df, year, "state")
    out = coalesce_text_columns(out, "province", "province", required=True, context="state")

    if "hour_of_year" not in out.columns:
        raise KeyError("State panel must contain timestamp_utc or hour_of_year.")

    out = coalesce_numeric_columns(out, "pbar", "pbar_mw", required=True, context="state")
    out = coalesce_numeric_columns(out, "cost", "cost_eur_per_mwh", required=True, context="state")

    if _alias_positions(out, "price"):
        out = coalesce_numeric_columns(out, "price", "day_ahead_price_eur_per_mwh", required=False, context="state", allow_all_missing=True)

    out["province"] = out["province"].astype(str).str.strip()
    out["hour_of_year"] = pd.to_numeric(out["hour_of_year"], errors="raise").astype(int)
    out["pbar_mw"] = pd.to_numeric(out["pbar_mw"], errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    out["cost_eur_per_mwh"] = to_numeric_clean(out["cost_eur_per_mwh"], "cost_eur_per_mwh")

    if (out["cost_eur_per_mwh"] < 0).any():
        raise ValueError("cost_eur_per_mwh must be nonnegative. Fix A output.")

    if REQUIRE_A_COSTS and out["cost_eur_per_mwh"].nunique(dropna=True) <= 1:
        warnings.warn(
            "cost_eur_per_mwh has one unique value. Verify A output and price/tariff inputs."
        )

    if _alias_positions(out, "support"):
        out = coalesce_numeric_columns(out, "support", "tdtr_support", required=False, context="state", allow_all_missing=True)
        out["tdtr_support"] = pd.to_numeric(out["tdtr_support"], errors="coerce").fillna(0.0).astype(float).gt(0.5).astype(int)
    elif "tdtr_support" not in out.columns:
        out["tdtr_support"] = (out["pbar_mw"] > TOUCH_TOL).astype(int)
        warnings.warn("State file has no tdtr_support column. Inferring support from positive pbar_mw.")
    else:
        out["tdtr_support"] = pd.to_numeric(out["tdtr_support"], errors="coerce").fillna(0.0).astype(float).gt(0.5).astype(int)

    out.loc[out["tdtr_support"].astype(int) == 0, "pbar_mw"] = 0.0

    if "restriction_cost_source" not in out.columns:
        out["restriction_cost_source"] = "A_cost_eur_per_mwh"
    if "cost_source" not in out.columns:
        out["cost_source"] = "A_output_cost_eur_per_mwh"

    keep_preferred = [
        "province", "timestamp_utc", "year", "hour_of_year", "pbar_mw",
        "cost_eur_per_mwh", "restriction_cost_source", "cost_source", "tdtr_support",
        "Pbar_value_raw_before_tdtr_support", "pbar_support_policy", "price_da_norm",
        "rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm", "structural_exposure",
        "ptdf_effectivity_norm", "ptdf_requirement_exposure_MW", "ptdf_active_constraints",
        "burden_proxy_norm", "relief_value_proxy_norm", "restriction_cost",
    ]
    keep = [c for c in keep_preferred if c in out.columns]

    out = (
        out[keep]
        .sort_values(["province", "hour_of_year"])
        .drop_duplicates(["province", "hour_of_year"], keep="first")
        .reset_index(drop=True)
    )

    if ((out["tdtr_support"].astype(int) == 0) & (out["pbar_mw"].astype(float) > TOUCH_TOL)).any():
        bad = out.loc[
            (out["tdtr_support"].astype(int) == 0) & (out["pbar_mw"].astype(float) > TOUCH_TOL)
        ].head(10)
        raise ValueError("State has positive Pbar outside TDTR support.\n" + bad.to_string(index=False))

    return out


def normalize_requirements(df: pd.DataFrame, year: int, state_hours: Iterable[int]) -> pd.DataFrame:
    out = add_timestamp_and_hour(df, year, "requirements")
    out = coalesce_text_columns(out, "branch", "branch_id", required=True, context="requirements")
    out = coalesce_numeric_columns(out, "requirement", "R_mw", required=True, context="requirements")

    if "hour_of_year" not in out.columns:
        raise KeyError("Requirements must contain timestamp_utc or hour_of_year.")

    out = align_hour_basis(state_hours, out, "requirements")
    out["branch_id"] = out["branch_id"].astype(str)
    out["hour_of_year"] = pd.to_numeric(out["hour_of_year"], errors="raise").astype(int)
    out["R_mw"] = to_numeric_clean(out["R_mw"], "R_mw").clip(lower=0.0)

    out = out.groupby(["branch_id", "hour_of_year"], as_index=False)["R_mw"].sum()
    out = out[out["R_mw"] > 0.0].reset_index(drop=True)
    return out


def normalize_coefficients(df: pd.DataFrame, year: int, state_hours: Iterable[int]) -> pd.DataFrame:
    out = add_timestamp_and_hour(df, year, "coefficients")
    out = coalesce_text_columns(out, "branch", "branch_id", required=True, context="coefficients")
    out = coalesce_text_columns(out, "province", "province", required=True, context="coefficients")
    out = coalesce_numeric_columns(out, "coefficient", "a", required=True, context="coefficients")

    if "hour_of_year" not in out.columns:
        raise KeyError("Coefficients must contain timestamp_utc or hour_of_year.")

    out = align_hour_basis(state_hours, out, "coefficients")
    out["branch_id"] = out["branch_id"].astype(str)
    out["province"] = out["province"].astype(str).str.strip()
    out["hour_of_year"] = pd.to_numeric(out["hour_of_year"], errors="raise").astype(int)
    out["a"] = to_numeric_clean(out["a"], "a")

    out = out[np.isfinite(out["a"]) & (np.abs(out["a"]) > 1e-12)].copy()
    out = out.groupby(["branch_id", "hour_of_year", "province"], as_index=False)["a"].sum()
    out = out[np.abs(out["a"]) > 1e-12].reset_index(drop=True)
    return out


def normalize_participants(df: pd.DataFrame, year: int, n_hours: int, state: pd.DataFrame) -> pd.DataFrame:
    """
    Robust participant canonicalization. This is the main fix for your crash:
    it coalesces all capacity/budget/slack aliases before using df["Pctr_mw"].
    """
    out = df.copy()
    out = coalesce_text_columns(out, "province", "province", required=True, context="participants")
    out = coalesce_numeric_columns(out, "pctr", "Pctr_mw", required=True, context="participants")

    has_depth = bool(_alias_positions(out, "depth_budget"))
    has_touch = bool(_alias_positions(out, "touch_budget"))
    has_slack = bool(_alias_positions(out, "slack_penalty"))

    if REQUIRE_A_BUDGETS and not has_depth:
        raise KeyError(
            "Participants file has no H_depth/H_depth_hours/depth_budget column. "
            "A must write H_depth = rho * support_hours for B/M2/M3."
        )
    if REQUIRE_A_BUDGETS and not has_touch:
        raise KeyError(
            "Participants file has no H_touch_tokens/H_touched/Hbar column. "
            "A must write the touched-time budget for B/B1."
        )

    if has_depth:
        out = coalesce_numeric_columns(out, "depth_budget", "H_depth", required=True, context="participants")
    if has_touch:
        out = coalesce_numeric_columns(out, "touch_budget", "H_touched", required=True, context="participants")
    if has_slack:
        out = coalesce_numeric_columns(out, "slack_penalty", "slack_penalty", required=True, context="participants")

    out["province"] = out["province"].astype(str).str.strip()
    out["Pctr_mw"] = to_numeric_clean(out["Pctr_mw"], "Pctr_mw")

    if (out["Pctr_mw"] <= 0).any():
        bad = out.loc[out["Pctr_mw"] <= 0, "province"].tolist()
        raise ValueError(f"Nonpositive Pctr_mw for provinces: {bad}")

    support_hours = infer_tdtr_support_hours_from_state(state)

    if "H_depth" not in out.columns:
        out["H_depth"] = TDTR_DEPTH_BUDGET_RHO * float(support_hours)
        out["H_depth_source"] = f"B_fallback_rho_times_support_hours_rho={TDTR_DEPTH_BUDGET_RHO:g}"
        warnings.warn(
            f"H_depth missing. Using fallback H_depth = rho * support_hours = "
            f"{TDTR_DEPTH_BUDGET_RHO:g} * {support_hours}."
        )
    else:
        out["H_depth"] = to_numeric_clean(out["H_depth"], "H_depth")
        out["H_depth_source"] = "A_participant_file"

    if "H_touched" not in out.columns:
        out["H_touched"] = int(support_hours)
        out["H_touched_source"] = "B_fallback_support_hours"
        warnings.warn(f"H_touched missing. Using fallback H_touched = support_hours = {support_hours}.")
    else:
        out["H_touched"] = np.floor(to_numeric_clean(out["H_touched"], "H_touched")).astype(int)
        out["H_touched_source"] = "A_participant_file"

    if "slack_penalty" not in out.columns:
        out["slack_penalty"] = float(os.getenv("DEBIT_SLACK_PENALTY", os.getenv("M_PENALTY", "5000.0")))
        out["slack_penalty_source"] = "B_fallback"
        warnings.warn(
            "Participants file has no slack_penalty/M column. Using "
            f"{float(out['slack_penalty'].iloc[0]):g}. For final runs, write it from A."
        )
    else:
        out["slack_penalty"] = to_numeric_clean(out["slack_penalty"], "slack_penalty")
        out["slack_penalty_source"] = "A_participant_file"

    if (out["H_depth"] < -1e-9).any():
        raise ValueError("Negative H_depth found. Fix A budget calibration.")
    if (out["H_touched"] < 0).any():
        raise ValueError("Negative H_touched found. Fix A budget calibration.")
    if REQUIRE_POSITIVE_SLACK_PENALTY and (out["slack_penalty"] <= 0).any():
        bad = out.loc[out["slack_penalty"] <= 0, ["province", "slack_penalty"]]
        raise ValueError("Slack penalty must be strictly positive. Otherwise emergency slack can be free.\n" + bad.to_string(index=False))

    out["tdtr_support_hours"] = int(support_hours)
    out["depth_budget_to_support_ratio"] = out["H_depth"] / max(float(support_hours), 1.0)
    out["touch_budget_to_support_ratio"] = out["H_touched"] / max(float(support_hours), 1.0)

    if (out["depth_budget_to_support_ratio"] >= WARN_IF_DEPTH_BUDGET_RATIO_GE).any():
        warnings.warn(
            f"Some H_depth/support_hours ratios are >= {WARN_IF_DEPTH_BUDGET_RATIO_GE:g}. "
            "M2/M3 depth budgets may be weakly scarce or non-scarce. If M2 and M3 are identical, reduce rho in A."
        )

    pctr_map = out.set_index("province")["Pctr_mw"].astype(float)
    pctr_series = state["province"].map(pctr_map).astype(float)
    if pctr_series.isna().any():
        missing = sorted(state.loc[pctr_series.isna(), "province"].unique().tolist())
        raise ValueError(f"State contains provinces absent from participants: {missing}")

    state["pbar_mw"] = pd.to_numeric(state["pbar_mw"], errors="coerce").fillna(pctr_series)
    state["pbar_mw"] = np.minimum(state["pbar_mw"].astype(float), pctr_series).clip(lower=0.0)
    if "tdtr_support" in state.columns:
        state.loc[state["tdtr_support"].astype(int) == 0, "pbar_mw"] = 0.0

    return out.sort_values("province").reset_index(drop=True)


def load_year_manifest(year: int) -> Dict[str, object]:
    return load_json_if_exists(INPUT_DIR / f"debit_opt_year_manifest_{year}.json")


def load_year_input(year: int, input_dir: Optional[Union[str, Path]] = None) -> YearInput:
    global INPUT_DIR

    old_input_dir = INPUT_DIR
    if input_dir is not None:
        INPUT_DIR = Path(input_dir)

    try:
        state = read_table(INPUT_DIR / f"debit_opt_state_{year}", required=True)
        req = read_table(INPUT_DIR / f"debit_opt_requirements_{year}", required=True)
        coeff = read_table(INPUT_DIR / f"debit_opt_coefficients_{year}", required=True)
        part = read_table(INPUT_DIR / f"debit_opt_participants_{year}.csv", required=True)
        tariff = read_table(INPUT_DIR / f"debit_opt_tariff_inputs_{year}.csv", required=False)

        assert state is not None and req is not None and coeff is not None and part is not None

        state = normalize_state(state, year)
        req = normalize_requirements(req, year, state["hour_of_year"].unique())
        coeff = normalize_coefficients(coeff, year, state["hour_of_year"].unique())
        part = normalize_participants(part, year, state["hour_of_year"].nunique(), state)
        metadata = load_year_manifest(year)

        yi = YearInput(
            year=year,
            state=state,
            requirements=req,
            coefficients=coeff,
            participants=part,
            tariff=tariff,
            metadata=metadata,
        )
        validate_year_input(yi)
        return yi
    finally:
        INPUT_DIR = old_input_dir


def validate_year_input(yi: YearInput) -> None:
    st = yi.state
    req = yi.requirements
    coeff = yi.coefficients
    part = yi.participants

    for col in ["province", "hour_of_year", "pbar_mw", "cost_eur_per_mwh", "tdtr_support"]:
        if col not in st.columns:
            raise KeyError(f"State missing required column after normalization: {col}")
    for col in ["branch_id", "hour_of_year", "R_mw"]:
        if col not in req.columns:
            raise KeyError(f"Requirements missing required column after normalization: {col}")
    for col in ["branch_id", "hour_of_year", "province", "a"]:
        if col not in coeff.columns:
            raise KeyError(f"Coefficients missing required column after normalization: {col}")
    for col in ["province", "Pctr_mw", "H_depth", "H_touched", "slack_penalty"]:
        if col not in part.columns:
            raise KeyError(f"Participants missing required column after normalization: {col}")

    if st["pbar_mw"].isna().any():
        raise ValueError("State contains missing pbar_mw after participant merge.")
    if (st["pbar_mw"] < -1e-9).any():
        raise ValueError("Negative pbar_mw found.")
    if (st["cost_eur_per_mwh"] < -1e-9).any():
        raise ValueError("cost_eur_per_mwh must be nonnegative.")
    if (req["R_mw"] < -1e-9).any():
        raise ValueError("Negative relief requirement found.")

    if ((st["tdtr_support"].astype(int) == 0) & (st["pbar_mw"].astype(float) > TOUCH_TOL)).any():
        bad = st.loc[(st["tdtr_support"].astype(int) == 0) & (st["pbar_mw"].astype(float) > TOUCH_TOL)].head(10)
        raise ValueError("State has positive Pbar outside TDTR support.\n" + bad.to_string(index=False))

    if (part["H_depth"] < -1e-9).any():
        raise ValueError("Negative H_depth found.")
    if (part["H_touched"] < 0).any():
        raise ValueError("Negative H_touched found.")
    if REQUIRE_POSITIVE_SLACK_PENALTY and (part["slack_penalty"] <= 0).any():
        bad = part.loc[part["slack_penalty"] <= 0, ["province", "slack_penalty"]]
        raise ValueError("Nonpositive slack penalty found.\n" + bad.to_string(index=False))

    state_keys = set(zip(st["province"].astype(str), st["hour_of_year"].astype(int)))
    coeff_keys = set(zip(coeff["province"].astype(str), coeff["hour_of_year"].astype(int)))
    missing_state = coeff_keys - state_keys
    if missing_state:
        sample = list(missing_state)[:5]
        raise ValueError(f"Coefficient rows reference province-hour keys absent from state panel. Sample={sample}")

    req_keys = set(zip(req["branch_id"].astype(str), req["hour_of_year"].astype(int)))
    coeff_req_keys = set(zip(coeff["branch_id"].astype(str), coeff["hour_of_year"].astype(int)))
    orphan_coeff = coeff_req_keys - req_keys
    if orphan_coeff:
        warnings.warn(
            f"Year {yi.year}: {len(orphan_coeff)} coefficient constraint-hour keys are absent from requirements and will be ignored."
        )

    prov_state = set(st["province"].astype(str).unique())
    prov_part = set(part["province"].astype(str).unique())
    if prov_state - prov_part:
        raise ValueError(f"State has provinces absent from participants: {sorted(prov_state - prov_part)}")

    support_hours = infer_tdtr_support_hours_from_state(st)
    if support_hours <= 0 and len(req) > 0:
        raise ValueError(f"Year {yi.year}: TDTR support has zero selected hours but requirements are nonempty.")

    ratio = part["H_depth"].astype(float) / max(float(support_hours), 1.0)
    if (ratio >= WARN_IF_DEPTH_BUDGET_RATIO_GE).any():
        warnings.warn(
            f"Year {yi.year}: H_depth/support_hours is high. "
            f"min={ratio.min():.3f}, mean={ratio.mean():.3f}, max={ratio.max():.3f}. "
            "If M2/M3 budgets are nonbinding, reduce rho in A."
        )


# =============================================================================
# Matrix construction
# =============================================================================

def build_indices(yi: YearInput) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    state = yi.state.copy().sort_values(["province", "hour_of_year"]).reset_index(drop=True)
    req = yi.requirements.copy().sort_values(["branch_id", "hour_of_year"]).reset_index(drop=True)
    coeff = yi.coefficients.copy()

    state["x_idx"] = np.arange(len(state), dtype=np.int64)
    req["relief_idx"] = np.arange(len(req), dtype=np.int64)

    coeff = coeff.merge(
        state[["province", "hour_of_year", "x_idx"]],
        on=["province", "hour_of_year"], how="inner", validate="many_to_one",
    )
    coeff = coeff.merge(
        req[["branch_id", "hour_of_year", "relief_idx"]],
        on=["branch_id", "hour_of_year"], how="inner", validate="many_to_one",
    )
    coeff = coeff[np.isfinite(coeff["a"]) & (np.abs(coeff["a"]) > 1e-12)].copy()

    dims = {"n_x": int(len(state)), "n_s": int(len(req)), "n_relief": int(len(req)), "n_coeff": int(len(coeff))}
    return state, req, coeff, dims


def build_relief_matrix(req: pd.DataFrame, coeff: pd.DataFrame, n_x: int) -> sp.csr_matrix:
    n_relief = len(req)
    n_s = n_relief
    rows = np.concatenate([coeff["relief_idx"].to_numpy(dtype=np.int64), np.arange(n_s, dtype=np.int64)])
    cols = np.concatenate([coeff["x_idx"].to_numpy(dtype=np.int64), n_x + np.arange(n_s, dtype=np.int64)])
    data = np.concatenate([coeff["a"].to_numpy(dtype=float), np.ones(n_s, dtype=float)])
    return sp.coo_matrix((data, (rows, cols)), shape=(n_relief, n_x + n_s)).tocsr()


def prepare_q_vector(
    state: pd.DataFrame,
    q: Optional[Union[float, pd.DataFrame, pd.Series, Mapping[Tuple[str, int], float]]],
    rule_name: str,
) -> np.ndarray:
    if q is None:
        if rule_name.upper().startswith("M3"):
            raise ValueError(f"{rule_name}: q weights are required. No silent q=1 replacement for M3.")
        return np.ones(len(state), dtype=float)

    if isinstance(q, (float, int)):
        return np.full(len(state), float(q), dtype=float)

    if isinstance(q, pd.Series):
        arr = q.to_numpy(dtype=float)
        if len(arr) != len(state):
            raise ValueError(f"q Series length {len(arr)} does not match state length {len(state)}")
        return arr

    if isinstance(q, Mapping):
        vals = []
        for p, h in zip(state["province"], state["hour_of_year"]):
            key = (str(p), int(h))
            if key not in q:
                raise ValueError(f"Missing q weight for key={key}")
            vals.append(float(q[key]))
        return np.asarray(vals, dtype=float)

    if isinstance(q, pd.DataFrame):
        qdf = q.copy()
        qdf = coalesce_text_columns(qdf, "province", "province", required=True, context="q weights")
        if "hour_of_year" not in qdf.columns:
            qdf = infer_hour_from_timestamp_by_year(qdf, "q weights")
        else:
            qdf = coalesce_numeric_columns(qdf, "hour", "hour_of_year", required=False, context="q weights")
        qdf = coalesce_numeric_columns(qdf, "debit_weight", "debit_weight", required=True, context="q weights")

        qdf["province"] = qdf["province"].astype(str).str.strip()
        qdf["hour_of_year"] = pd.to_numeric(qdf["hour_of_year"], errors="raise").astype(int)
        qdf["debit_weight"] = pd.to_numeric(qdf["debit_weight"], errors="raise").astype(float)

        duplicate_count = int(qdf.duplicated(["province", "hour_of_year"]).sum())
        if duplicate_count > 0:
            warnings.warn(
                f"{rule_name}: q dataframe has {duplicate_count} duplicate province-hour rows. Averaging duplicate debit_weight values."
            )
            qdf = qdf.groupby(["province", "hour_of_year"], as_index=False).agg(debit_weight=("debit_weight", "mean"))

        merged = state[["province", "hour_of_year"]].merge(
            qdf[["province", "hour_of_year", "debit_weight"]],
            on=["province", "hour_of_year"], how="left", validate="one_to_one",
        )
        if merged["debit_weight"].isna().any():
            missing = merged.loc[merged["debit_weight"].isna(), ["province", "hour_of_year"]].head(10).to_dict("records")
            raise ValueError(
                f"{rule_name}: missing q weights for {int(merged['debit_weight'].isna().sum())} province-hours. "
                f"No silent q=1 replacement for M3. Sample={missing}"
            )
        return merged["debit_weight"].to_numpy(dtype=float)

    raise TypeError(f"Unsupported q type: {type(q)}")


def build_budget_matrix(
    state: pd.DataFrame,
    participants: pd.DataFrame,
    q_vec: np.ndarray,
    n_s: int,
) -> Tuple[sp.csr_matrix, np.ndarray, pd.DataFrame]:
    part = participants.copy().sort_values("province").reset_index(drop=True)
    prov_to_row = {p: i for i, p in enumerate(part["province"].astype(str))}
    pctr_map = part.set_index("province")["Pctr_mw"].astype(float).to_dict()

    rows = state["province"].astype(str).map(prov_to_row).to_numpy(dtype=np.int64)
    x_cols = state["x_idx"].to_numpy(dtype=np.int64)
    pctr = state["province"].astype(str).map(pctr_map).to_numpy(dtype=float)
    data = q_vec / pctr

    A = sp.coo_matrix((data, (rows, x_cols)), shape=(len(part), len(state) + n_s)).tocsr()
    rhs = part["H_depth"].to_numpy(dtype=float)
    return A, rhs, part


# =============================================================================
# M2/M3 weighted-depth solver API
# =============================================================================

def solve_weighted_depth_lp(
    yi: YearInput,
    q: Optional[Union[float, pd.DataFrame, pd.Series, Mapping[Tuple[str, int], float]]] = None,
    rule_name: str = "M2",
    regularized: bool = False,
    tau_x: float = QP_TAU_X,
    tau_s: float = QP_TAU_S,
    output_flag: int = OUTPUT_FLAG,
    save: bool = True,
    **kwargs: Any,
) -> SolveResult:
    if "rule" in kwargs and kwargs["rule"] is not None:
        rule_name = str(kwargs["rule"])

    state, req, coeff, dims = build_indices(yi)
    n_x, n_s = dims["n_x"], dims["n_s"]

    q_vec = prepare_q_vector(state, q, rule_name=rule_name)
    if not np.all(np.isfinite(q_vec)) or np.any(q_vec <= 0):
        raise ValueError(f"{rule_name}: all q weights must be positive and finite.")

    state["debit_weight"] = q_vec
    pbar = state["pbar_mw"].to_numpy(dtype=float)
    c = state["cost_eur_per_mwh"].to_numpy(dtype=float)
    slack_penalty = float(yi.participants["slack_penalty"].median())
    rhs_relief = req["R_mw"].to_numpy(dtype=float)

    if REQUIRE_POSITIVE_SLACK_PENALTY and slack_penalty <= 0:
        raise ValueError(f"{rule_name} year={yi.year}: slack_penalty={slack_penalty:g}. Fix A output.")
    if WARN_IF_SLACK_PENALTY_LE_MAX_COST and np.isfinite(c).any():
        max_c = float(np.nanmax(c))
        if slack_penalty <= max_c:
            warnings.warn(
                f"{rule_name} year={yi.year}: slack_penalty={slack_penalty:g} <= max restriction cost={max_c:g}. "
                "The solver may prefer emergency slack over costly restrictions."
            )
    if np.all(pbar <= TOUCH_TOL) and float(np.sum(rhs_relief)) > 0:
        warnings.warn(f"{rule_name} year={yi.year}: all Pbar are zero but requirements are positive.")

    A_relief = build_relief_matrix(req, coeff, n_x=n_x)
    A_budget, rhs_budget, part_budget = build_budget_matrix(state, yi.participants, q_vec, n_s=n_s)

    model = gp.Model(f"{rule_name}_{yi.year}")
    gurobi_params(model, output_flag=output_flag, is_mip=False)

    lb = np.zeros(n_x + n_s, dtype=float)
    ub = np.concatenate([pbar, np.full(n_s, GRB.INFINITY)])
    z = model.addMVar(n_x + n_s, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="z")
    x = z[:n_x]
    s = z[n_x:]

    relief_con = model.addMConstr(A_relief, z, ">", rhs_relief, name="relief")
    budget_con = model.addMConstr(A_budget, z, "<", rhs_budget, name="annual_depth_budget")

    obj = c @ x + slack_penalty * s.sum()
    if regularized:
        pctr_for_state = state["province"].map(yi.participants.set_index("province")["Pctr_mw"].astype(float)).to_numpy(dtype=float)
        obj = obj + 0.5 * tau_x * gp.quicksum((x[i] / float(pctr_for_state[i])) * (x[i] / float(pctr_for_state[i])) for i in range(n_x))
        obj = obj + 0.5 * tau_s * gp.quicksum(s[j] * s[j] for j in range(n_s))

    model.setObjective(obj, GRB.MINIMIZE)
    t0 = time.time()
    model.optimize()
    runtime = time.time() - t0

    st_name = status_name(model)
    if model.SolCount <= 0:
        raise RuntimeError(f"{rule_name} year {yi.year}: no solution. Status={st_name}")

    z_val = np.asarray(z.X, dtype=float)
    x_val = z_val[:n_x]
    s_val = z_val[n_x:]
    obj_val = float(model.ObjVal)

    total_slack = float(s_val.sum())
    total_req = float(req["R_mw"].sum())
    served = 1.0 - total_slack / total_req if total_req > 0 else np.nan
    op_cost_x = float(np.dot(c, x_val))
    slack_cost = float(slack_penalty * total_slack)

    try:
        relief_pi = np.asarray(relief_con.Pi, dtype=float)
    except Exception:
        relief_pi = np.full(n_s, np.nan)

    try:
        budget_pi = np.asarray(budget_con.Pi, dtype=float)
        mu = np.maximum(-budget_pi, 0.0)
    except Exception:
        budget_pi = np.full(len(part_budget), np.nan)
        mu = np.full(len(part_budget), np.nan)

    x_df = state[["province", "hour_of_year"]].copy()
    if "timestamp_utc" in state.columns:
        x_df["timestamp_utc"] = state["timestamp_utc"].to_numpy()

    x_df["x_mw"] = x_val
    x_df["pbar_mw"] = pbar
    x_df["cost_eur_per_mwh"] = c
    x_df["debit_weight"] = q_vec

    pctr_map = yi.participants.set_index("province")["Pctr_mw"].astype(float).to_dict()
    x_df["Pctr_mw"] = x_df["province"].map(pctr_map).astype(float)
    x_df["debit"] = x_df["debit_weight"] * x_df["x_mw"] / x_df["Pctr_mw"]
    x_df["rule"] = rule_name
    x_df["year"] = yi.year

    if "tdtr_support" in state.columns:
        x_df["tdtr_support"] = state["tdtr_support"].to_numpy()

    optional_state_cols = [
        "price_da_norm", "rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm",
        "structural_exposure", "ptdf_effectivity_norm", "ptdf_requirement_exposure_MW",
        "ptdf_active_constraints", "burden_proxy_norm", "relief_value_proxy_norm",
        "restriction_cost", "cost_source", "restriction_cost_source",
    ]
    for col in optional_state_cols:
        if col in state.columns and col not in x_df.columns:
            x_df[col] = state[col].to_numpy()

    slack_df = req[["branch_id", "hour_of_year", "R_mw"]].copy()
    slack_df["slack_mw"] = s_val
    slack_df["relief_dual_lambda"] = relief_pi
    slack_df["rule"] = rule_name
    slack_df["year"] = yi.year

    budget = x_df.groupby("province", as_index=False).agg(
        total_x_mwh=("x_mw", "sum"),
        total_debit=("debit", "sum"),
        positive_hours=("x_mw", lambda v: int((np.asarray(v) > TOUCH_TOL).sum())),
    )
    keep_part_cols = ["province", "Pctr_mw", "H_depth"]
    for ccol in ["tdtr_support_hours", "depth_budget_to_support_ratio"]:
        if ccol in yi.participants.columns:
            keep_part_cols.append(ccol)
    budget = budget.merge(yi.participants[keep_part_cols], on="province", how="left")
    budget = budget.merge(part_budget[["province"]].assign(mu_depth=mu, gurobi_budget_pi=budget_pi), on="province", how="left")
    budget["budget_utilization"] = budget["total_debit"] / budget["H_depth"].replace(0, np.nan)
    budget["is_budget_binding"] = (budget["H_depth"] - budget["total_debit"]).abs() <= BUDGET_BINDING_TOL
    budget["rule"] = rule_name
    budget["year"] = yi.year

    # Correct theorem/KKT diagnostic: alpha_{n,t} = mu_n * x_{n,t}/P_n.
    kkt = x_df.merge(budget[["province", "mu_depth"]], on="province", how="left")
    kkt["alpha_mu_x_over_P"] = kkt["mu_depth"] * kkt["x_mw"] / kkt["Pctr_mw"]
    kkt["alpha"] = kkt["alpha_mu_x_over_P"]
    kkt["scarcity_term_mu_q_over_P"] = kkt["mu_depth"] * kkt["debit_weight"] / kkt["Pctr_mw"]
    kkt = kkt[(kkt["x_mw"].abs() > TOUCH_TOL) | (kkt["alpha_mu_x_over_P"].abs() > 1e-12)].copy()

    total_debit = float(budget["total_debit"].sum())
    allowance_productivity = (total_req - total_slack) / total_debit if total_debit > 0 else np.nan

    quality = "optimal" if model.Status == GRB.OPTIMAL else "suboptimal_available"
    notes = [] if model.Status == GRB.OPTIMAL else [f"LP/QP ended with status {st_name}; solution available."]

    result = SolveResult(
        rule=rule_name,
        year=yi.year,
        status=st_name,
        status_code=int(model.Status),
        objective=obj_val,
        runtime_sec=runtime,
        x=x_df,
        slack=slack_df,
        budget=budget,
        kkt=kkt,
        total_relief_requirement=total_req,
        total_slack=total_slack,
        operating_cost_x=op_cost_x,
        slack_penalty_cost=slack_cost,
        relief_served_share=float(served),
        total_debit=total_debit,
        allowance_productivity=float(allowance_productivity),
        model_rows=int(model.NumConstrs),
        model_cols=int(model.NumVars),
        model_nnz=int(model.NumNZs),
        quality_flag=quality,
        notes=notes,
        raw={
            "regularized": regularized,
            "tau_x": tau_x,
            "tau_s": tau_s,
            "q_min": float(np.min(q_vec)),
            "q_max": float(np.max(q_vec)),
            "q_mean": float(np.mean(q_vec)),
            "q_std": float(np.std(q_vec)),
            "support_hours": int(infer_tdtr_support_hours_from_state(state)),
            "mean_depth_budget_to_support_ratio": float(yi.participants["depth_budget_to_support_ratio"].mean()) if "depth_budget_to_support_ratio" in yi.participants.columns else np.nan,
        },
    )

    if save:
        save_result(result)
    return result


def solve_weighted_depth_qp(
    yi: YearInput,
    q: Optional[Union[float, pd.DataFrame, pd.Series, Mapping[Tuple[str, int], float]]] = None,
    rule: str = "M2_QP",
    rule_name: Optional[str] = None,
    tau_x: float = QP_TAU_X,
    tau_s: float = QP_TAU_S,
    output_flag: int = OUTPUT_FLAG,
    save: bool = False,
) -> SolveResult:
    final_rule = rule_name if rule_name is not None else rule
    return solve_weighted_depth_lp(
        yi=yi,
        q=q,
        rule_name=final_rule,
        regularized=True,
        tau_x=tau_x,
        tau_s=tau_s,
        output_flag=output_flag,
        save=save,
    )


# =============================================================================
# Optional compact M1 diagnostic
# =============================================================================

def solve_fixed_touch_lp_for_warmstart(yi: YearInput, m2: SolveResult, output_flag: int = 0) -> Dict[str, np.ndarray]:
    state, req, coeff, dims = build_indices(yi)
    n_x, n_s = dims["n_x"], dims["n_s"]

    x_m2 = m2.x[["province", "hour_of_year", "x_mw"]].copy()
    state = state.merge(x_m2, on=["province", "hour_of_year"], how="left", validate="one_to_one")
    state["x_mw"] = state["x_mw"].fillna(0.0)

    hbar = yi.participants.set_index("province")["H_touched"].astype(int).to_dict()
    y_fixed = np.zeros(n_x, dtype=float)

    for prov, sub in state.groupby("province", sort=False):
        k = int(hbar.get(prov, 0))
        if k <= 0:
            continue
        cand = sub[sub["pbar_mw"] > TOUCH_TOL].copy()
        cand = cand.sort_values(["x_mw", "pbar_mw"], ascending=[False, False]).head(k)
        y_fixed[cand["x_idx"].to_numpy(dtype=int)] = 1.0

    pbar = state["pbar_mw"].to_numpy(dtype=float) * y_fixed
    c = state["cost_eur_per_mwh"].to_numpy(dtype=float)
    slack_penalty = float(yi.participants["slack_penalty"].median())
    rhs_relief = req["R_mw"].to_numpy(dtype=float)
    A_relief = build_relief_matrix(req, coeff, n_x=n_x)

    model = gp.Model(f"M1_warmstart_fixed_touch_{yi.year}")
    gurobi_params(model, output_flag=output_flag, is_mip=False)

    lb = np.zeros(n_x + n_s, dtype=float)
    ub = np.concatenate([pbar, np.full(n_s, GRB.INFINITY)])
    z = model.addMVar(n_x + n_s, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="z")
    model.addMConstr(A_relief, z, ">", rhs_relief, name="relief")
    model.setObjective(c @ z[:n_x] + slack_penalty * z[n_x:].sum(), GRB.MINIMIZE)
    model.optimize()

    if model.SolCount <= 0:
        return {"y": y_fixed, "x": np.zeros(n_x), "s": rhs_relief.copy()}

    z_val = np.asarray(z.X, dtype=float)
    return {"y": y_fixed, "x": z_val[:n_x], "s": z_val[n_x:]}


def solve_touched_time_milp(
    yi: YearInput,
    m2_for_warmstart: Optional[SolveResult] = None,
    rule_name: str = "M1_COMPACT_DIAGNOSTIC",
    output_flag: int = OUTPUT_FLAG,
    save: bool = True,
) -> SolveResult:
    state, req, coeff, dims = build_indices(yi)
    n_x, n_s = dims["n_x"], dims["n_s"]

    pbar = state["pbar_mw"].to_numpy(dtype=float)
    c = state["cost_eur_per_mwh"].to_numpy(dtype=float)
    slack_penalty = float(yi.participants["slack_penalty"].median())
    rhs_relief = req["R_mw"].to_numpy(dtype=float)
    A_relief = build_relief_matrix(req, coeff, n_x=n_x)

    part = yi.participants.copy().sort_values("province").reset_index(drop=True)
    prov_to_row = {p: i for i, p in enumerate(part["province"].astype(str))}
    budget_rows = state["province"].astype(str).map(prov_to_row).to_numpy(dtype=np.int64)
    A_touch = sp.coo_matrix((np.ones(n_x), (budget_rows, np.arange(n_x, dtype=np.int64))), shape=(len(part), n_x)).tocsr()
    rhs_touch = part["H_touched"].to_numpy(dtype=float)

    model = gp.Model(f"{rule_name}_{yi.year}")
    gurobi_params(model, output_flag=output_flag, is_mip=True)

    lb = np.zeros(n_x + n_s, dtype=float)
    ub = np.concatenate([pbar, np.full(n_s, GRB.INFINITY)])
    z = model.addMVar(n_x + n_s, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="z")
    x = z[:n_x]
    s = z[n_x:]
    y = model.addMVar(n_x, lb=np.zeros(n_x), ub=np.ones(n_x), vtype=GRB.BINARY, name="touch")

    model.addMConstr(A_relief, z, ">", rhs_relief, name="relief")
    model.addConstr(x <= pbar * y, name="touch_link")
    model.addMConstr(A_touch, y, "<", rhs_touch, name="annual_touch_budget")

    obj = c @ x + slack_penalty * s.sum()
    if MIP_TINY_TOUCH_PENALTY > 0:
        obj = obj + MIP_TINY_TOUCH_PENALTY * y.sum()
    model.setObjective(obj, GRB.MINIMIZE)

    notes: List[str] = ["Compact annual M1 MILP diagnostic. Publication M1 should come from B1_run_m1_touched_only.py."]

    if M1_WARMSTART_FROM_M2 and m2_for_warmstart is not None:
        try:
            warm = solve_fixed_touch_lp_for_warmstart(yi, m2_for_warmstart, output_flag=0)
            x.Start = warm["x"]
            s.Start = warm["s"]
            y.Start = warm["y"]
            notes.append("M1 warm start from fixed top-M2 touched hours LP was supplied.")
        except Exception as exc:
            notes.append(f"M1 warm start failed and was skipped: {exc}")

    t0 = time.time()
    model.optimize()
    runtime = time.time() - t0

    st_name = status_name(model)
    if model.SolCount <= 0:
        raise RuntimeError(f"{rule_name} year {yi.year}: no MIP incumbent. Status={st_name}")

    z_val = np.asarray(z.X, dtype=float)
    x_val = z_val[:n_x]
    s_val = z_val[n_x:]
    y_val = np.asarray(y.X, dtype=float)

    obj_val = float(model.ObjVal)
    best_bound = float(safe_attr(model, "ObjBound", np.nan))
    mip_gap = float(safe_attr(model, "MIPGap", np.nan))
    node_count = float(safe_attr(model, "NodeCount", np.nan))
    sol_count = int(safe_attr(model, "SolCount", 0))

    total_slack = float(s_val.sum())
    total_req = float(req["R_mw"].sum())
    served = 1.0 - total_slack / total_req if total_req > 0 else np.nan
    op_cost_x = float(np.dot(c, x_val))
    slack_cost = float(slack_penalty * total_slack)

    x_df = state[["province", "hour_of_year"]].copy()
    if "timestamp_utc" in state.columns:
        x_df["timestamp_utc"] = state["timestamp_utc"].to_numpy()
    x_df["x_mw"] = x_val
    x_df["touch_y_raw"] = y_val
    x_df["touch_realized"] = (x_val > TOUCH_TOL).astype(int)
    x_df["pbar_mw"] = pbar
    x_df["cost_eur_per_mwh"] = c
    pctr_map = yi.participants.set_index("province")["Pctr_mw"].astype(float).to_dict()
    x_df["Pctr_mw"] = x_df["province"].map(pctr_map).astype(float)
    x_df["debit_weight"] = np.nan
    x_df["debit"] = x_df["touch_realized"].astype(float)
    x_df["rule"] = rule_name
    x_df["year"] = yi.year
    if "tdtr_support" in state.columns:
        x_df["tdtr_support"] = state["tdtr_support"].to_numpy()

    slack_df = req[["branch_id", "hour_of_year", "R_mw"]].copy()
    slack_df["slack_mw"] = s_val
    slack_df["rule"] = rule_name
    slack_df["year"] = yi.year

    budget = x_df.groupby("province", as_index=False).agg(
        total_x_mwh=("x_mw", "sum"),
        total_debit=("debit", "sum"),
        positive_hours=("touch_realized", "sum"),
        raw_y_sum=("touch_y_raw", "sum"),
    )
    keep_cols = ["province", "Pctr_mw", "H_touched"]
    for ccol in ["tdtr_support_hours", "touch_budget_to_support_ratio"]:
        if ccol in yi.participants.columns:
            keep_cols.append(ccol)
    budget = budget.merge(yi.participants[keep_cols], on="province", how="left")
    budget["budget_utilization"] = budget["total_debit"] / budget["H_touched"].replace(0, np.nan)
    budget["raw_y_utilization"] = budget["raw_y_sum"] / budget["H_touched"].replace(0, np.nan)
    budget["is_budget_binding"] = (budget["H_touched"] - budget["total_debit"]).abs() <= BUDGET_BINDING_TOL
    budget["rule"] = rule_name
    budget["year"] = yi.year

    kkt = pd.DataFrame(columns=["rule", "year", "province", "hour_of_year", "x_mw"])
    total_debit = float(budget["total_debit"].sum())
    allowance_productivity = (total_req - total_slack) / total_debit if total_debit > 0 else np.nan

    if model.Status == GRB.OPTIMAL:
        quality = "optimal_compact_diagnostic"
    elif np.isfinite(mip_gap) and mip_gap <= MIP_GAP:
        quality = "mip_gap_acceptable_compact_diagnostic"
    else:
        quality = "incumbent_only_gap_too_large_compact_diagnostic"
        notes.append(f"Compact M1 not publication-grade: status={st_name}, gap={mip_gap}, bound={best_bound}.")

    result = SolveResult(
        rule=rule_name,
        year=yi.year,
        status=st_name,
        status_code=int(model.Status),
        objective=obj_val,
        runtime_sec=runtime,
        x=x_df,
        slack=slack_df,
        budget=budget,
        kkt=kkt,
        total_relief_requirement=total_req,
        total_slack=total_slack,
        operating_cost_x=op_cost_x,
        slack_penalty_cost=slack_cost,
        relief_served_share=float(served),
        total_debit=total_debit,
        allowance_productivity=float(allowance_productivity),
        mip_gap=mip_gap,
        best_bound=best_bound,
        node_count=node_count,
        solution_count=sol_count,
        model_rows=int(model.NumConstrs),
        model_cols=int(model.NumVars),
        model_nnz=int(model.NumNZs),
        quality_flag=quality,
        notes=notes,
        raw={"mip_tiny_touch_penalty": MIP_TINY_TOUCH_PENALTY},
    )
    if save:
        save_result(result)
    return result


# =============================================================================
# M3 q handling
# =============================================================================

def canonical_feature_set(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    if s == "":
        return ""
    u = s.upper().replace("FEATURE_", "").replace("FEATURE", "").replace("LEVEL_", "").replace("LEVEL", "").replace(" ", "").replace("-", "_")
    if u in {"0", "_0"}:
        return "F0"
    if u in {"1", "_1"}:
        return "F1"
    if u in {"2", "_2"}:
        return "F2"
    if u in {"3", "_3"}:
        return "F3"
    if "F0" in u:
        return "F0"
    if "F1" in u:
        return "F1"
    if "F2" in u:
        return "F2"
    if "F3" in u:
        return "F3"
    return u


def canonical_m3_variant(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    if s == "":
        return ""
    z = s.lower().replace("-", "_").replace(" ", "_").replace("__", "_").strip("_")
    mapping = {
        "plain": "M3_plain", "m3_plain": "M3_plain", "proxy": "M3_plain", "proxy_plain": "M3_plain",
        "oi": "M3_OI", "m3_oi": "M3_OI", "one_shot": "M3_OI", "oneshot": "M3_OI",
        "optimization_informed": "M3_OI", "opt_informed": "M3_OI",
        "lo": "M3_LO", "m3_lo": "M3_LO", "learn_optimize": "M3_LO", "learn_opt": "M3_LO",
        "alternating": "M3_LO", "alternating_lo": "M3_LO",
    }
    if z in mapping:
        return mapping[z]
    if z.endswith("_plain"):
        return "M3_plain"
    if z.endswith("_oi"):
        return "M3_OI"
    if z.endswith("_lo"):
        return "M3_LO"
    return s


def m3_rule_name(feature_set: str, variant: str) -> str:
    return f"M3_{canonical_feature_set(feature_set)}_{canonical_m3_variant(variant)}"


def read_m3_q_table(q_file: Union[str, Path]) -> pd.DataFrame:
    q_path = Path(q_file)
    if not q_path.exists():
        raise FileNotFoundError(f"M3_Q_FILE does not exist: {q_path}")

    qdf = pd.read_parquet(q_path) if q_path.suffix.lower() == ".parquet" else pd.read_csv(q_path, low_memory=False)
    if "year" not in qdf.columns:
        raise KeyError(f"M3 q file has no 'year' column: {q_path}")

    qdf = coalesce_text_columns(qdf, "province", "province", required=True, context="q file")
    qdf = coalesce_numeric_columns(qdf, "debit_weight", "debit_weight", required=True, context="q file")

    if _alias_positions(qdf, "feature_set"):
        qdf = coalesce_text_columns(qdf, "feature_set", "feature_set", required=True, context="q file")
    if _alias_positions(qdf, "variant"):
        qdf = coalesce_text_columns(qdf, "variant", "variant", required=True, context="q file")

    if "feature_set" not in qdf.columns:
        raise KeyError("M3 q file has no feature_set/feature_level column. C must write long weights with feature_set.")
    if "variant" not in qdf.columns:
        raise KeyError("M3 q file has no variant/architecture column. C must write long weights with variant.")

    qdf["year"] = pd.to_numeric(qdf["year"], errors="coerce").astype("Int64")
    if qdf["year"].isna().any():
        raise ValueError("M3 q file contains missing/unparseable year values.")

    qdf = infer_hour_from_timestamp_by_year(qdf, "q file")
    qdf["province"] = qdf["province"].astype(str).str.strip()
    qdf["hour_of_year"] = pd.to_numeric(qdf["hour_of_year"], errors="raise").astype(int)
    qdf["debit_weight"] = pd.to_numeric(qdf["debit_weight"], errors="raise").astype(float)
    qdf["feature_set_canon"] = qdf["feature_set"].map(canonical_feature_set)
    qdf["variant_canon"] = qdf["variant"].map(canonical_m3_variant)

    if qdf["debit_weight"].isna().any():
        raise ValueError("M3 q file contains missing debit_weight values.")
    if (qdf["debit_weight"] <= 0).any():
        raise ValueError("M3 q file contains nonpositive debit_weight values.")
    return qdf


def m3_q_inventory(qdf: pd.DataFrame) -> pd.DataFrame:
    return (
        qdf.groupby(["year", "feature_set_canon", "variant_canon"], dropna=False)
        .agg(
            n_rows=("debit_weight", "size"),
            n_provinces=("province", "nunique"),
            n_hours=("hour_of_year", "nunique"),
            q_min=("debit_weight", "min"),
            q_mean=("debit_weight", "mean"),
            q_max=("debit_weight", "max"),
        )
        .reset_index()
        .sort_values(["year", "feature_set_canon", "variant_canon"])
    )


def print_or_save_m3_q_inventory(q_file: Union[str, Path]) -> pd.DataFrame:
    qdf = read_m3_q_table(q_file)
    inv = m3_q_inventory(qdf)
    print("\n[M3-Q] Available q slices:")
    if inv.empty:
        print("[M3-Q] No q slices found.")
    else:
        print(inv.to_string(index=False))
    if M3_WRITE_Q_INVENTORY:
        save_table(inv, RESULTS_DIR / "m3_q_inventory_from_B.csv")
    return inv


def load_q_file_for_year(year: int, q_file: Union[str, Path], feature_set: str, variant: str) -> pd.DataFrame:
    q_path = Path(q_file)
    qdf = read_m3_q_table(q_path)
    inv = m3_q_inventory(qdf)
    if M3_WRITE_Q_INVENTORY:
        save_table(inv, RESULTS_DIR / "m3_q_inventory_from_B.csv")

    f = canonical_feature_set(feature_set)
    v = canonical_m3_variant(variant)
    sub = qdf[
        (qdf["year"].astype("Int64") == int(year))
        & (qdf["feature_set_canon"] == f)
        & (qdf["variant_canon"] == v)
    ].copy()

    if sub.empty:
        available_year = inv[inv["year"].astype("Int64") == int(year)].copy()
        msg = (
            f"No q weights found for year={year}, feature_set={f}, variant={v} in {q_path}.\n\n"
            f"Available q slices for year={year}:\n"
            f"{available_year.to_string(index=False) if not available_year.empty else '[none]'}\n\n"
            "This is a C output problem if you expected this slice. B will not replace missing M3 q weights by q=1."
        )
        raise ValueError(msg)

    duplicate_count = int(sub.duplicated(["province", "hour_of_year"]).sum())
    if duplicate_count > 0:
        warnings.warn(
            f"M3 q slice year={year}, feature_set={f}, variant={v} has {duplicate_count} duplicate province-hour rows. Averaging duplicates."
        )
        sub = sub.groupby(["province", "hour_of_year"], as_index=False).agg(debit_weight=("debit_weight", "mean"))

    return sub[["province", "hour_of_year", "debit_weight"]].copy()


# =============================================================================
# Diagnostics and saving
# =============================================================================

def kkt_equalization_diagnostic(result: SolveResult) -> pd.DataFrame:
    if result.kkt is None or result.kkt.empty:
        return pd.DataFrame()
    cols = [
        "rule", "year", "province", "hour_of_year", "x_mw", "Pctr_mw", "debit_weight",
        "mu_depth", "alpha", "alpha_mu_x_over_P", "scarcity_term_mu_q_over_P",
    ]
    return result.kkt[[c for c in cols if c in result.kkt.columns]].copy()


def result_summary_row(result: SolveResult) -> Dict[str, object]:
    mean_budget_util = float(result.budget["budget_utilization"].mean()) if not result.budget.empty else np.nan
    max_budget_util = float(result.budget["budget_utilization"].max()) if not result.budget.empty else np.nan
    n_budget_binding = int(result.budget["is_budget_binding"].sum()) if "is_budget_binding" in result.budget.columns else 0
    row = {
        "year": result.year,
        "rule": result.rule,
        "status": result.status,
        "status_code": result.status_code,
        "quality_flag": result.quality_flag,
        "objective": result.objective,
        "operating_cost_x": result.operating_cost_x,
        "slack_penalty_cost": result.slack_penalty_cost,
        "total_slack": result.total_slack,
        "total_relief_requirement": result.total_relief_requirement,
        "relief_served_share": result.relief_served_share,
        "total_debit": result.total_debit,
        "allowance_productivity": result.allowance_productivity,
        "mean_budget_utilization": mean_budget_util,
        "max_budget_utilization": max_budget_util,
        "n_budget_binding": n_budget_binding,
        "runtime_sec": result.runtime_sec,
        "mip_gap": result.mip_gap,
        "best_bound": result.best_bound,
        "node_count": result.node_count,
        "solution_count": result.solution_count,
        "model_rows": result.model_rows,
        "model_cols": result.model_cols,
        "model_nnz": result.model_nnz,
        "notes": " | ".join(result.notes),
    }
    for key, value in result.raw.items():
        row[f"raw_{key}"] = value
    return row


def save_result(result: SolveResult) -> None:
    prefix = f"{result.rule}_"
    save_table(result.x, SOLUTION_DIR / f"{prefix}province_hour_{result.year}.csv.gz")
    positive_slack = result.slack[result.slack["slack_mw"] > POSITIVE_SLACK_TOL].copy()
    save_table(positive_slack, SOLUTION_DIR / f"{prefix}positive_slack_{result.year}.csv.gz")
    save_table(result.budget, SOLUTION_DIR / f"{prefix}budget_summary_{result.year}.csv")


# =============================================================================
# Optional B1 import
# =============================================================================

def load_b1_m1_summary_rows(years: Iterable[int]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    years_set = set(int(y) for y in years)
    path = M1_ONLY_RESULTS_DIR / "m1_only_summary.csv"
    meta: Dict[str, object] = {
        "include_b1_m1_results": bool(INCLUDE_B1_M1_RESULTS),
        "m1_only_results_dir": str(M1_ONLY_RESULTS_DIR),
        "m1_only_summary_path": str(path),
        "loaded": False,
        "missing_years": [],
    }

    if not INCLUDE_B1_M1_RESULTS:
        return [], meta
    if not path.exists():
        meta["reason"] = "m1_only_summary.csv not found"
        print(f"[M1-B1] No B1 summary found at {path}. Skipping imported M1 rows.")
        return [], meta

    df = pd.read_csv(path, low_memory=False)
    if "year" not in df.columns:
        meta["reason"] = "B1 summary has no year column"
        print(f"[M1-B1] B1 summary at {path} has no year column. Skipping.")
        return [], meta

    if "rule" in df.columns:
        df = df[df["rule"].astype(str).str.upper() == "M1"].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["year"].isin(years_set)].copy()

    rows: List[Dict[str, object]] = []
    loaded_years = set()
    for _, r in df.iterrows():
        y = int(r["year"])
        loaded_years.add(y)
        row = r.to_dict()
        row["year"] = y
        row["rule"] = "M1"
        row["source"] = "B1_run_m1_touched_only.py"
        rows.append(row)

    missing = sorted(years_set - loaded_years)
    meta["loaded"] = len(rows) > 0
    meta["loaded_years"] = sorted(loaded_years)
    meta["missing_years"] = missing
    meta["n_rows"] = len(rows)

    if rows:
        print(f"[M1-B1] Imported {len(rows)} M1 summary rows from {path}.")
    if missing:
        print(f"[M1-B1] Missing B1 M1 rows for years: {missing}")
    return rows, meta


# =============================================================================
# Manifest and main
# =============================================================================

def write_manifest(
    summary_rows: List[Dict[str, object]],
    year_metadata: Dict[int, Dict[str, object]],
    cost_audit: Dict[int, Dict[str, object]],
    budget_audit: Dict[int, Dict[str, object]],
    b1_m1_meta: Dict[str, object],
) -> None:
    summary_df = pd.DataFrame(summary_rows)
    manifest = {
        "script": "B_run_annual_lp.py",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "repo_root": str(REPO_ROOT),
        "input_dir": str(INPUT_DIR),
        "results_dir": str(RESULTS_DIR),
        "solution_dir": str(SOLUTION_DIR),
        "years": YEARS,
        "provinces": PROVINCES,
        "bugfix": "participant/state/requirements/coefficients/q aliases are coalesced before canonical columns are used",
        "budget_policy": {
            "A_is_source_of_truth": True,
            "require_A_budgets": REQUIRE_A_BUDGETS,
            "depth_budget_interpretation": "H_depth should equal rho times selected TDTR support hours, written by A",
            "touch_budget_interpretation": "H_touched should equal selected TDTR support hours unless deliberately configured otherwise",
            "fallback_rho_if_A_budgets_not_required": TDTR_DEPTH_BUDGET_RHO,
            "warn_if_depth_budget_ratio_ge": WARN_IF_DEPTH_BUDGET_RATIO_GE,
        },
        "tdtr_availability_share_legacy_config_only": TDTR_AVAILABILITY_SHARE,
        "tdtr_restriction_share_legacy_config_only": TDTR_RESTRICTION_SHARE,
        "cost_source": "A_build_debit_optimization_inputs.py: cost_eur_per_mwh/restriction_cost",
        "cost_audit_by_year": cost_audit,
        "budget_audit_by_year": budget_audit,
        "solve_flags": {
            "solve_m2": SOLVE_M2,
            "solve_m1_compact_diagnostic": SOLVE_M1,
            "include_b1_m1_results": INCLUDE_B1_M1_RESULTS,
            "solve_m3_from_q_file": SOLVE_M3_FROM_Q_FILE,
            "run_qp_self_test": RUN_QP_SELF_TEST,
        },
        "m1_policy": {
            "publication_m1_source": "B1_run_m1_touched_only.py",
            "compact_m1_default": False,
            "b1_import": b1_m1_meta,
        },
        "m3": {
            "q_file": M3_Q_FILE if M3_Q_FILE else None,
            "feature_set": canonical_feature_set(M3_FEATURE_SET),
            "variant": canonical_m3_variant(M3_VARIANT),
            "rule": m3_rule_name(M3_FEATURE_SET, M3_VARIANT),
            "solve_in_B_executable_mode": SOLVE_M3_FROM_Q_FILE,
            "executable_mode_default": "off; set RUN_M3_IN_B=1 or SOLVE_M3_FROM_Q_FILE=1",
            "no_silent_q_one_for_m3": True,
            "q_inventory_written": bool(M3_WRITE_Q_INVENTORY),
        },
        "gurobi": {
            "version": gp.gurobi.version(),
            "lp_method": LP_METHOD,
            "threads": THREADS,
            "m1_time_limit": MIP_TIME_LIMIT,
            "m1_mip_gap": MIP_GAP,
            "m1_mip_focus": MIP_FOCUS,
            "m1_heuristics": MIP_HEURISTICS,
            "m1_cuts": MIP_CUTS,
            "m1_presolve": MIP_PRESOLVE,
        },
        "api_for_C_train_m3_and_D_run_full_benchmark": {
            "load_year_input": True,
            "solve_weighted_depth_lp": True,
            "solve_weighted_depth_qp": True,
            "kkt_equalization_diagnostic": True,
            "result_summary_row": True,
            "save_result": True,
            "load_q_file_for_year": True,
            "alpha_column": "alpha_mu_x_over_P and alias alpha",
        },
        "python": sys.version,
        "platform": platform.platform(),
        "summary_quality_counts": summary_df["quality_flag"].value_counts().to_dict() if not summary_df.empty and "quality_flag" in summary_df.columns else {},
        "year_metadata_from_A": year_metadata,
    }
    path = RESULTS_DIR / "benchmark_baseline_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[SAVE] {path}")


def print_header() -> None:
    print("=" * 78)
    print("B: RUN ANNUAL TDTR DEBIT LP BASELINES / C API")
    print("=" * 78)
    print(f"Repo root       : {REPO_ROOT}")
    print(f"Input dir       : {INPUT_DIR}")
    print(f"Results dir     : {RESULTS_DIR}")
    print(f"Solution dir    : {SOLUTION_DIR}")
    print(f"Years           : {YEARS}")
    print(f"Provinces       : {PROVINCES}")
    print(f"Solve M2        : {SOLVE_M2}")
    print(f"Solve compact M1: {SOLVE_M1}  (default off; publication M1 comes from B1)")
    print(f"Include B1 M1   : {INCLUDE_B1_M1_RESULTS}")
    print(f"B1 results dir  : {M1_ONLY_RESULTS_DIR}")
    print(f"Solve M3 file   : {SOLVE_M3_FROM_Q_FILE} {M3_Q_FILE if SOLVE_M3_FROM_Q_FILE else '(off by default in B)'}")
    print(f"M3 feature/var. : {M3_FEATURE_SET} / {M3_VARIANT}")
    print("Budget source   : A participant files; no old 0.15*T rebuild")
    print(f"Require A budget: {REQUIRE_A_BUDGETS}")
    print(f"Fallback rho    : {TDTR_DEPTH_BUDGET_RHO:g} (only if REQUIRE_A_BUDGETS=0 and H_depth missing)")
    print("Cost source     : A output cost_eur_per_mwh/restriction_cost only")
    print(f"A costs required: {REQUIRE_A_COSTS}")
    print(f"Slack positive  : {REQUIRE_POSITIVE_SLACK_PENALTY}")
    print("Alias handling  : coalesce duplicate aliases before canonical use")
    print(f"LP method       : {LP_METHOD}")
    print(f"Threads         : {THREADS if THREADS > 0 else 'Gurobi default'}")
    print(f"QP tau_x/s      : {QP_TAU_X} / {QP_TAU_S}")
    print("=" * 78)


def main() -> None:
    ensure_dirs()
    print_header()

    summary_rows: List[Dict[str, object]] = []
    kkt_rows: List[pd.DataFrame] = []
    year_metadata: Dict[int, Dict[str, object]] = {}
    cost_audit: Dict[int, Dict[str, object]] = {}
    budget_audit: Dict[int, Dict[str, object]] = {}

    b1_rows, b1_m1_meta = load_b1_m1_summary_rows(YEARS)
    summary_rows.extend(b1_rows)

    if M3_Q_FILE and Path(M3_Q_FILE).exists():
        try:
            print_or_save_m3_q_inventory(M3_Q_FILE)
        except Exception as exc:
            print(f"[M3-Q] Inventory check skipped/failed: {exc}")
    elif SOLVE_M3_FROM_Q_FILE:
        raise FileNotFoundError(f"SOLVE_M3_FROM_Q_FILE=1 but M3_Q_FILE does not exist: {M3_Q_FILE}")

    for year in YEARS:
        print("\n" + "=" * 78)
        print(f"[YEAR] {year}")
        print("=" * 78)

        yi = load_year_input(year)
        year_metadata[year] = dict(yi.metadata)

        total_R = float(yi.requirements["R_mw"].sum())
        slack_penalty = float(yi.participants["slack_penalty"].median())
        H_mean = float(yi.participants["H_depth"].mean())
        Hbar_mean = float(yi.participants["H_touched"].mean())
        pctr_mean = float(yi.participants["Pctr_mw"].mean())
        support_hours = float(infer_tdtr_support_hours_from_state(yi.state))
        rho_eff = H_mean / support_hours if support_hours > 0 else np.nan
        touch_ratio = Hbar_mean / support_hours if support_hours > 0 else np.nan

        cost_desc = yi.state["cost_eur_per_mwh"].describe(percentiles=[0.01, 0.5, 0.99]).to_dict()
        cost_source = yi.state["restriction_cost_source"].iloc[0] if "restriction_cost_source" in yi.state.columns else "A_cost_eur_per_mwh"

        cost_audit[year] = {
            "source": cost_source,
            "min": float(cost_desc.get("min", np.nan)),
            "p01": float(cost_desc.get("1%", np.nan)),
            "p50": float(cost_desc.get("50%", np.nan)),
            "p99": float(cost_desc.get("99%", np.nan)),
            "max": float(cost_desc.get("max", np.nan)),
            "mean": float(cost_desc.get("mean", np.nan)),
            "std": float(cost_desc.get("std", np.nan)),
        }
        budget_audit[year] = {
            "support_hours": support_hours,
            "H_depth_mean": H_mean,
            "H_depth_min": float(yi.participants["H_depth"].min()),
            "H_depth_max": float(yi.participants["H_depth"].max()),
            "H_depth_to_support_mean": rho_eff,
            "H_touch_mean": Hbar_mean,
            "H_touch_min": float(yi.participants["H_touched"].min()),
            "H_touch_max": float(yi.participants["H_touched"].max()),
            "H_touch_to_support_mean": touch_ratio,
        }

        print(
            f"[DATA] constraints={len(yi.requirements):,} | coeffs={len(yi.coefficients):,} | "
            f"state rows={len(yi.state):,} | total R={total_R:,.3f} | slack penalty={slack_penalty:g}"
        )
        print(
            f"[TDTR] support hours={support_hours:.0f} | "
            f"H_depth mean={H_mean:.3f} | effective rho=H_depth/support={rho_eff:.3f} | "
            f"H_touch mean={Hbar_mean:.3f} | H_touch/support={touch_ratio:.3f} | "
            f"Pctr mean={pctr_mean:.3f} MW"
        )
        print(
            f"[COST] source={cost_source} | min={cost_audit[year]['min']:.3f}, "
            f"p50={cost_audit[year]['p50']:.3f}, p99={cost_audit[year]['p99']:.3f}, "
            f"max={cost_audit[year]['max']:.3f} €/MW-h"
        )

        m2_result: Optional[SolveResult] = None
        if SOLVE_M2:
            print("\n[SOLVE] M2 uniform-depth LP")
            m2_result = solve_weighted_depth_lp(yi, q=1.0, rule_name="M2", regularized=False, save=True)
            summary_rows.append(result_summary_row(m2_result))
            kd = kkt_equalization_diagnostic(m2_result)
            if not kd.empty:
                kkt_rows.append(kd)
            print(
                f"[M2] objective={m2_result.objective:,.3f} | slack={m2_result.total_slack:.6f} | "
                f"served={m2_result.relief_served_share:.6f} | runtime={m2_result.runtime_sec:.1f}s | "
                f"quality={m2_result.quality_flag}"
            )
            n_bind = int(m2_result.budget["is_budget_binding"].sum()) if "is_budget_binding" in m2_result.budget.columns else 0
            if n_bind == 0:
                print(
                    "[WARN] M2 has zero binding depth budgets. Then M3 may not change the solution because "
                    "the annual debit constraint is non-scarce. Check rho in A."
                )

        if RUN_QP_SELF_TEST:
            print("\n[SOLVE] M2 regularized QP self-test")
            qp = solve_weighted_depth_qp(yi, q=1.0, rule="M2_QP_SELFTEST", tau_x=QP_TAU_X, tau_s=QP_TAU_S, save=False)
            print(
                f"[QP] objective={qp.objective:,.3f} | slack={qp.total_slack:.6f} | served={qp.relief_served_share:.6f} | runtime={qp.runtime_sec:.1f}s"
            )

        if SOLVE_M3_FROM_Q_FILE:
            print(f"\n[SOLVE] M3 weighted-depth LP from q file | feature={M3_FEATURE_SET} variant={M3_VARIANT}")
            qdf = load_q_file_for_year(year, M3_Q_FILE, M3_FEATURE_SET, M3_VARIANT)
            m3_rule = m3_rule_name(M3_FEATURE_SET, M3_VARIANT)
            m3 = solve_weighted_depth_lp(yi, q=qdf, rule_name=m3_rule, regularized=False, save=True)
            summary_rows.append(result_summary_row(m3))
            kd = kkt_equalization_diagnostic(m3)
            if not kd.empty:
                kkt_rows.append(kd)
            print(
                f"[{m3_rule}] objective={m3.objective:,.3f} | slack={m3.total_slack:.6f} | "
                f"served={m3.relief_served_share:.6f} | runtime={m3.runtime_sec:.1f}s | quality={m3.quality_flag}"
            )

        if SOLVE_M1:
            print("\n[SOLVE] compact M1 diagnostic MILP")
            m1 = solve_touched_time_milp(yi, m2_for_warmstart=m2_result, rule_name="M1_COMPACT_DIAGNOSTIC", save=True)
            summary_rows.append(result_summary_row(m1))
            print(
                f"[M1-COMPACT] objective={m1.objective:,.3f} | slack={m1.total_slack:.6f} | "
                f"served={m1.relief_served_share:.6f} | runtime={m1.runtime_sec:.1f}s | gap={m1.mip_gap} | quality={m1.quality_flag}"
            )
            for note in m1.notes:
                print(f"[NOTE] {note}")

    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "benchmark_baseline_results.csv"
    save_table(summary, summary_path)

    kkt = pd.concat(kkt_rows, ignore_index=True) if kkt_rows else pd.DataFrame()
    kkt_path = RESULTS_DIR / "kkt_equalization_diagnostic.csv"
    save_table(kkt, kkt_path)

    write_manifest(summary_rows, year_metadata, cost_audit, budget_audit, b1_m1_meta)

    print("\n[DONE]")
    print(f"Summary : {summary_path}")
    print(f"KKT     : {kkt_path}")
    print(f"Manifest: {RESULTS_DIR / 'benchmark_baseline_manifest.json'}")


if __name__ == "__main__":
    main()
