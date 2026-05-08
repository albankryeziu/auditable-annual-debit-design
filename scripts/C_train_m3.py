#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C_train_m3.py
==============

Publication-grade training script for auditable weighted-depth debit rates M3
for the annual TDTR-like debit-rule paper.

This script is designed to be the source of M3 debit weights consumed by
D_run_full_benchmark.py. It does not rebuild the physical relief layer. It reads
annual optimization inputs through B_run_annual_lp.py and trains bounded,
normalized, auditable accounting debit rates q_{n,t}.

Core discipline
---------------
1. The physical relief layer is fixed outside this script.
2. C reads annual inputs through B.load_year_input.
3. q is an annual debit accounting rate, not a price.
4. q is bounded: Q_MIN <= q <= Q_MAX.
5. q is normalized to mean one by province-year-feature-variant on active TDTR
   support, using an exact bounded mean-one projection.
6. q outside active TDTR support is set to one for audit neutrality.
7. F0 is calendar-only, not intercept-only.
8. OI/LO use alpha = mu*x/P, never mu*q*x/P.
9. Final paper benchmark values are produced by D, not by this script.

Feature ladder
--------------
F0: calendar-only public baseline.
F1: F0 + public load/price stress proxies.
F2: F1 + burden/relief/cost proxies.
F3: F2 + constructed PTDF-testbed effectivity/exposure proxies.

Architectures
-------------
M3_plain: proxy-grounded supervised fit.
M3_OI: one-shot optimization-informed fit initialized from M2 alpha diagnostics.
M3_LO: alternating learn-optimize fit initialized from M3_OI.

The OI/LO implementation uses two mechanisms:
    (i)  alpha-shaped targets: high-alpha hours are made cheaper in debit terms;
    (ii) true alpha-weighted regression: high-alpha training rows receive larger
         sample weights in the Ridge fit.

By default, alpha is generated from regularized QP solves through B's QP API.
Set C_ALPHA_SOLVE_MODE=lp if you deliberately want LP dual diagnostics instead.

Outputs
-------
data/derived/m3_debit_weights_by_year.parquet/.csv
    Long-format source-of-truth weights consumed by D.

data/derived/m3_debit_weights_wide_by_year.parquet/.csv
    Audit convenience table.

outputs/results/m3_training_q_audit.csv
outputs/results/m3_training_solution_summary.csv
outputs/logs/m3_training_log.csv
data/derived/m3_training_manifest.json

Run:
    python scripts/C_train_m3.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def env_str_list(name: str, default: Iterable[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return [str(x) for x in default]
    return [x.strip() for x in str(raw).split(",") if x.strip()]


# =============================================================================
# Import B API
# =============================================================================

try:
    import B_run_annual_lp as B  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import scripts/B_run_annual_lp.py. Run C from the repository "
        "root or ensure scripts/ is on sys.path."
    ) from exc


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_YEARS = [2021, 2022, 2023, 2024, 2025]
ALL_YEARS = env_int_list("DEBIT_YEARS", cfg_get("ALL_YEARS", DEFAULT_YEARS))
APPLY_YEARS = env_int_list("C_APPLY_YEARS", ALL_YEARS)
TRAIN_YEARS = env_int_list("C_TRAIN_YEARS", cfg_get("M3_TRAIN_YEARS", [2021, 2022, 2023]))
VALIDATION_YEAR = int(os.getenv("C_VALIDATION_YEAR", str(cfg_get("M3_VALIDATION_YEAR", 2024))))
TEST_YEAR = int(os.getenv("C_TEST_YEAR", str(cfg_get("M3_TEST_YEAR", 2025))))

FEATURE_SETS = [str(x).upper().strip() for x in env_str_list("C_FEATURE_SETS", ["F0", "F1", "F2", "F3"])]
VARIANTS = env_str_list("C_M3_VARIANTS", ["M3_plain", "M3_OI", "M3_LO"])

MAIN_FEATURE_SET = os.getenv("MAIN_FEATURE_SET", str(cfg_get("MAIN_FEATURE_SET", "F2"))).strip().upper()
MAIN_M3_VARIANT = os.getenv("MAIN_M3_VARIANT", str(cfg_get("MAIN_M3_VARIANT", "M3_LO"))).strip()

DERIVED_DIR = Path(cfg_get("DERIVED_DIR", REPO_ROOT / "data" / "derived"))
RESULTS_DIR = Path(cfg_get("RESULTS_DIR", REPO_ROOT / "outputs" / "results"))
LOG_DIR = Path(cfg_get("LOG_DIR", REPO_ROOT / "outputs" / "logs"))

WEIGHTS_LONG_PARQUET = Path(os.getenv(
    "M3_WEIGHTS_LONG_PARQUET",
    str(cfg_get("M3_WEIGHTS_PATH", DERIVED_DIR / "m3_debit_weights_by_year.parquet")),
))
WEIGHTS_LONG_CSV = WEIGHTS_LONG_PARQUET.with_suffix(".csv")

WEIGHTS_WIDE_PARQUET = Path(os.getenv(
    "M3_WEIGHTS_WIDE_PARQUET",
    str(DERIVED_DIR / "m3_debit_weights_wide_by_year.parquet"),
))
WEIGHTS_WIDE_CSV = WEIGHTS_WIDE_PARQUET.with_suffix(".csv")

TRAINING_LOG_CSV = LOG_DIR / "m3_training_log.csv"
TRAINING_SOLUTION_SUMMARY_CSV = RESULTS_DIR / "m3_training_solution_summary.csv"
TRAINING_Q_AUDIT_CSV = RESULTS_DIR / "m3_training_q_audit.csv"
TRAINING_MANIFEST_JSON = DERIVED_DIR / "m3_training_manifest.json"

Q_MIN = float(os.getenv("M3_Q_MIN", str(cfg_get("M3_Q_MIN", 0.25))))
Q_MAX = float(os.getenv("M3_Q_MAX", str(cfg_get("M3_Q_MAX", 4.0))))
if not (0.0 < Q_MIN < 1.0 < Q_MAX):
    raise ValueError("Require 0 < M3_Q_MIN < 1 < M3_Q_MAX for mean-one bounded projection.")

TARGET_GAIN = float(os.getenv("M3_TARGET_GAIN", str(cfg_get("M3_TARGET_GAIN", 1.20))))
RIDGE_ALPHA = float(os.getenv("M3_RIDGE_ALPHA", str(cfg_get("M3_RIDGE_ALPHA", 1.0))))
LO_ITERS = int(os.getenv("M3_LO_ITERS", str(cfg_get("M3_LO_ITERS", 2))))

# Alpha solve mode: qps are used for stable differentiable training diagnostics.
C_ALPHA_SOLVE_MODE = os.getenv("C_ALPHA_SOLVE_MODE", str(cfg_get("C_ALPHA_SOLVE_MODE", "qp"))).strip().lower()
if C_ALPHA_SOLVE_MODE not in {"qp", "lp"}:
    raise ValueError("C_ALPHA_SOLVE_MODE must be either 'qp' or 'lp'.")
C_ALPHA_FALLBACK_TO_LP = env_bool("C_ALPHA_FALLBACK_TO_LP", True)
C_ALPHA_SAMPLE_WEIGHT_GAIN = float(os.getenv("C_ALPHA_SAMPLE_WEIGHT_GAIN", str(cfg_get("C_ALPHA_SAMPLE_WEIGHT_GAIN", 4.0))))
C_ALPHA_SAMPLE_WEIGHT_CAP = float(os.getenv("C_ALPHA_SAMPLE_WEIGHT_CAP", str(cfg_get("C_ALPHA_SAMPLE_WEIGHT_CAP", 20.0))))
C_REQUIRE_ALPHA_FOR_MAIN = env_bool("C_REQUIRE_ALPHA_FOR_MAIN", True)

M3_NORMALIZE_ON_SUPPORT_ONLY = env_bool(
    "M3_NORMALIZE_ON_SUPPORT_ONLY",
    bool(cfg_get("M3_NORMALIZE_ON_SUPPORT_ONLY", True)),
)
M3_TRAIN_ON_SUPPORT_ONLY = env_bool(
    "M3_TRAIN_ON_SUPPORT_ONLY",
    bool(cfg_get("M3_TRAIN_ON_SUPPORT_ONLY", True)),
)
M3_SET_Q_OUTSIDE_SUPPORT_TO_ONE = env_bool(
    "M3_SET_Q_OUTSIDE_SUPPORT_TO_ONE",
    bool(cfg_get("M3_SET_Q_OUTSIDE_SUPPORT_TO_ONE", True)),
)

SOLVE_TRAINING_SUMMARY = env_bool("C_SOLVE_TRAINING_SUMMARY", True)
SOLVE_ALL_TRAINING_SUMMARIES = env_bool("C_SOLVE_ALL_TRAINING_SUMMARIES", True)
TRAINING_SUMMARY_OUTPUT_FLAG = int(os.getenv("C_TRAINING_SUMMARY_OUTPUT_FLAG", "0"))

TOUCH_TOL = float(os.getenv("TOUCH_TOL", "1e-7"))
EPS = 1e-12


# =============================================================================
# Optional sklearn
# =============================================================================

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class FitResult:
    feature_set: str
    variant: str
    feature_columns: List[str]
    model: Any
    scaler: Any
    fitted_with_sklearn: bool
    n_train_rows: int
    target_source: str
    sample_weight_source: str
    sample_weight_min: float
    sample_weight_mean: float
    sample_weight_max: float
    notes: List[str]


# =============================================================================
# Generic utilities
# =============================================================================

def ensure_dirs() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_LONG_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_WIDE_PARQUET.parent.mkdir(parents=True, exist_ok=True)


def canonical_feature_set(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip().upper().replace("FEATURE_", "").replace("FEATURE", "")
    s = s.replace("LEVEL_", "").replace("LEVEL", "").replace("-", "_").replace(" ", "")
    if s in {"0", "_0"}:
        return "F0"
    if s in {"1", "_1"}:
        return "F1"
    if s in {"2", "_2"}:
        return "F2"
    if s in {"3", "_3"}:
        return "F3"
    if "F0" in s:
        return "F0"
    if "F1" in s:
        return "F1"
    if "F2" in s:
        return "F2"
    if "F3" in s:
        return "F3"
    return s


def feature_level_int(feature_set: str) -> int:
    f = canonical_feature_set(feature_set)
    if f.startswith("F") and f[1:].isdigit():
        return int(f[1:])
    raise ValueError(f"Unknown feature set: {feature_set}")


def canonical_variant(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    z = s.lower().replace("-", "_").replace(" ", "_").replace("__", "_").strip("_")
    mapping = {
        "plain": "M3_plain",
        "m3_plain": "M3_plain",
        "proxy": "M3_plain",
        "proxy_plain": "M3_plain",
        "oi": "M3_OI",
        "m3_oi": "M3_OI",
        "one_shot": "M3_OI",
        "oneshot": "M3_OI",
        "optimization_informed": "M3_OI",
        "opt_informed": "M3_OI",
        "lo": "M3_LO",
        "m3_lo": "M3_LO",
        "learn_optimize": "M3_LO",
        "learn_opt": "M3_LO",
        "alternating": "M3_LO",
        "alternating_lo": "M3_LO",
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


def variant_short(variant: str) -> str:
    v = canonical_variant(variant)
    if v == "M3_plain":
        return "plain"
    if v == "M3_OI":
        return "OI"
    if v == "M3_LO":
        return "LO"
    return v.replace("M3_", "")


def split_role(year: int) -> str:
    y = int(year)
    if y in set(int(x) for x in TRAIN_YEARS):
        return "train"
    if y == int(VALIDATION_YEAR):
        return "validation"
    if y == int(TEST_YEAR):
        return "test"
    return "other"


def safe_numeric(s: Any, default: float = 0.0) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)
    return pd.Series(np.full(len(s), default, dtype=float))


def safe_minmax(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    lo = float(x.min())
    hi = float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=s.index)
    return (x - lo) / (hi - lo)


def save_table_both(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[SAVE] {parquet_path}")
    except Exception as exc:
        warnings.warn(f"Could not write parquet {parquet_path}: {exc}")
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] {csv_path}")


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print(f"[SAVE] {path}")


# =============================================================================
# Exact bounded mean-one projection
# =============================================================================

def project_bounded_mean_one(v: np.ndarray, lower: float = Q_MIN, upper: float = Q_MAX, target_mean: float = 1.0) -> np.ndarray:
    """Euclidean projection of v onto {u: lower<=u<=upper, mean(u)=target_mean}."""
    arr = np.asarray(v, dtype=float).copy()
    arr[~np.isfinite(arr)] = target_mean
    if arr.size == 0:
        return arr
    if not (lower <= target_mean <= upper):
        raise ValueError("Projection target mean must lie inside [lower, upper].")

    lam_lo = float(np.min(arr - upper))
    lam_hi = float(np.max(arr - lower))

    for _ in range(80):
        lam = 0.5 * (lam_lo + lam_hi)
        u = np.clip(arr - lam, lower, upper)
        if float(np.mean(u)) > target_mean:
            lam_lo = lam
        else:
            lam_hi = lam

    lam = 0.5 * (lam_lo + lam_hi)
    u = np.clip(arr - lam, lower, upper)

    # One final correction within free coordinates for numerical exactness.
    err = target_mean - float(np.mean(u))
    if abs(err) > 1e-11:
        free = (u > lower + 1e-10) & (u < upper - 1e-10)
        if np.any(free):
            u[free] += err * len(u) / float(np.sum(free))
            u = np.clip(u, lower, upper)
    return u


def normalize_q_by_province_year(df: pd.DataFrame, q_raw: np.ndarray, support_col: str = "active_support") -> np.ndarray:
    """Bound and normalize q by province-year, optionally only on active support."""
    q = np.asarray(q_raw, dtype=float).copy()
    q[~np.isfinite(q)] = 1.0
    q = np.clip(q, 1e-8, 1e8)

    keys = df[["year", "province", support_col]].copy()
    normalized = np.clip(q, Q_MIN, Q_MAX)

    for (_year, _province), idx in keys.groupby(["year", "province"], sort=False).groups.items():
        idx_arr = np.fromiter(idx, dtype=int)
        if M3_NORMALIZE_ON_SUPPORT_ONLY:
            support_mask = keys.loc[idx_arr, support_col].astype(bool).to_numpy()
            norm_idx = idx_arr[support_mask]
        else:
            norm_idx = idx_arr

        if len(norm_idx) == 0:
            continue
        normalized[norm_idx] = project_bounded_mean_one(q[norm_idx], lower=Q_MIN, upper=Q_MAX, target_mean=1.0)

    if M3_SET_Q_OUTSIDE_SUPPORT_TO_ONE:
        outside = ~keys[support_col].astype(bool).to_numpy()
        normalized[outside] = 1.0

    normalized[~np.isfinite(normalized)] = 1.0
    normalized = np.clip(normalized, Q_MIN, Q_MAX)
    return normalized


# =============================================================================
# Load annual inputs through B
# =============================================================================

def load_all_year_inputs(years: Iterable[int]) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for year in years:
        print(f"[LOAD] B.load_year_input({int(year)})")
        out[int(year)] = B.load_year_input(int(year))
    return out


# =============================================================================
# Feature engineering
# =============================================================================

def year_input_to_feature_frame(year: int, yi: Any) -> pd.DataFrame:
    state = yi.state.copy()
    state["year"] = int(year)
    state["split_role"] = split_role(int(year))
    state["province"] = state["province"].astype(str).str.strip()
    state["hour_of_year"] = pd.to_numeric(state["hour_of_year"], errors="raise").astype(int)

    if "timestamp_utc" in state.columns:
        ts = pd.to_datetime(state["timestamp_utc"], errors="coerce", utc=True)
    else:
        start = pd.Timestamp(f"{int(year)}-01-01 00:00:00", tz="UTC")
        ts = start + pd.to_timedelta(state["hour_of_year"] - 1, unit="h")
    if ts.isna().any():
        raise ValueError(f"Year {year}: unparseable timestamp_utc in state.")

    state["timestamp_utc"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    state["month"] = ts.dt.month.astype(int)
    state["weekday_utc"] = ts.dt.weekday.astype(int)
    state["hour_of_day_utc"] = ts.dt.hour.astype(int)
    state["hour_of_week"] = state["weekday_utc"] * 24 + state["hour_of_day_utc"]
    state["is_weekend"] = state["weekday_utc"].isin([5, 6]).astype(int)

    if "pbar_mw" in state.columns:
        pbar = pd.to_numeric(state["pbar_mw"], errors="coerce").fillna(0.0).astype(float)
    else:
        pbar = pd.Series(np.zeros(len(state), dtype=float), index=state.index)
        state["pbar_mw"] = pbar

    if "tdtr_support" in state.columns:
        support = pd.to_numeric(state["tdtr_support"], errors="coerce").fillna(0).astype(float) > 0.5
    else:
        support = pbar > TOUCH_TOL
        state["tdtr_support"] = support.astype(int)

    state["active_support"] = (support & (pbar > TOUCH_TOL)).astype(int)

    participants = yi.participants.copy()
    participants["province"] = participants["province"].astype(str).str.strip()
    if "Pctr_mw" not in participants.columns:
        raise KeyError("B.load_year_input must return participants with canonical Pctr_mw.")
    pctr_map = participants.set_index("province")["Pctr_mw"].astype(float).to_dict()
    state["Pctr_mw"] = state["province"].map(pctr_map).astype(float)
    if state["Pctr_mw"].isna().any():
        missing = sorted(state.loc[state["Pctr_mw"].isna(), "province"].unique().tolist())
        raise ValueError(f"Missing Pctr_mw for provinces: {missing}")

    # F0 calendar features.
    state["f0_intercept"] = 1.0
    state["f0_hour_sin"] = np.sin(2.0 * np.pi * state["hour_of_day_utc"] / 24.0)
    state["f0_hour_cos"] = np.cos(2.0 * np.pi * state["hour_of_day_utc"] / 24.0)
    state["f0_week_sin"] = np.sin(2.0 * np.pi * state["hour_of_week"] / 168.0)
    state["f0_week_cos"] = np.cos(2.0 * np.pi * state["hour_of_week"] / 168.0)
    state["f0_month_sin"] = np.sin(2.0 * np.pi * (state["month"] - 1) / 12.0)
    state["f0_month_cos"] = np.cos(2.0 * np.pi * (state["month"] - 1) / 12.0)
    state["f0_is_weekend"] = state["is_weekend"].astype(float)

    # F1 public load/price stress.
    for col in ["price_da_norm", "rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm"]:
        if col not in state.columns:
            state[col] = 0.0
        state[col] = pd.to_numeric(state[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    if "cost_eur_per_mwh" in state.columns:
        cost_source = "cost_eur_per_mwh"
    elif "restriction_cost" in state.columns:
        cost_source = "restriction_cost"
    else:
        cost_source = "_zero_cost_proxy"
        state[cost_source] = 0.0

    state["restriction_cost_norm"] = (
        state.groupby(["province", "year"], sort=False)[cost_source]
        .transform(safe_minmax)
        .astype(float)
    )

    state["f1_public_stress"] = (
        0.30 * state["price_da_norm"]
        + 0.25 * state["rl_plus_norm"]
        + 0.20 * state["rl_minus_norm"]
        + 0.15 * state["s_plus_norm"]
        + 0.10 * state["s_minus_norm"]
    )

    # F2 burden/relief/cost proxies.
    if "burden_proxy_norm" not in state.columns:
        state["burden_proxy_norm"] = (
            0.50 * state["restriction_cost_norm"]
            + 0.25 * state["s_minus_norm"]
            + 0.25 * state["rl_minus_norm"]
        )
    if "relief_value_proxy_norm" not in state.columns:
        relief_base = 0.40 * state["rl_plus_norm"] + 0.30 * state["s_plus_norm"] + 0.30 * state["price_da_norm"]
        state["relief_value_proxy_norm"] = (
            relief_base.groupby([state["province"], state["year"]]).transform(safe_minmax).astype(float)
        )

    state["burden_proxy_norm"] = pd.to_numeric(state["burden_proxy_norm"], errors="coerce").fillna(0.0).astype(float)
    state["relief_value_proxy_norm"] = pd.to_numeric(state["relief_value_proxy_norm"], errors="coerce").fillna(0.0).astype(float)

    # F3 constructed PTDF-testbed effectivity/exposure proxies.
    for col in ["ptdf_effectivity_norm", "ptdf_requirement_exposure_MW", "ptdf_active_constraints"]:
        if col not in state.columns:
            state[col] = 0.0
        state[col] = pd.to_numeric(state[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    if "ptdf_exposure_norm" not in state.columns:
        state["ptdf_exposure_norm"] = (
            state.groupby(["province", "year"], sort=False)["ptdf_requirement_exposure_MW"]
            .transform(safe_minmax)
            .astype(float)
        )
    else:
        state["ptdf_exposure_norm"] = pd.to_numeric(state["ptdf_exposure_norm"], errors="coerce").fillna(0.0).astype(float)

    state["ptdf_active_constraints_norm"] = (
        state.groupby(["province", "year"], sort=False)["ptdf_active_constraints"]
        .transform(safe_minmax)
        .astype(float)
    )

    # Common training proxy target. This is a training proxy projected onto the
    # different admissible feature sets; it is not the true burden process.
    state["proxy_burden_score"] = (
        0.45 * state["burden_proxy_norm"]
        + 0.35 * state["restriction_cost_norm"]
        + 0.20 * state["f1_public_stress"]
    )
    state["proxy_relief_score"] = (
        0.45 * state["relief_value_proxy_norm"]
        + 0.30 * state["ptdf_effectivity_norm"]
        + 0.25 * state["ptdf_exposure_norm"]
    )
    state["proxy_raw_score"] = state["proxy_burden_score"] - state["proxy_relief_score"]
    state["proxy_raw_score_centered"] = (
        state["proxy_raw_score"]
        - state.groupby(["province", "year"], sort=False)["proxy_raw_score"].transform("mean")
    )

    q_proxy_raw = np.exp(np.clip(TARGET_GAIN * state["proxy_raw_score_centered"].to_numpy(dtype=float), -20.0, 20.0))
    state["q_proxy_target"] = normalize_q_by_province_year(state, q_proxy_raw, support_col="active_support")

    keep_cols = [
        "year", "split_role", "province", "timestamp_utc", "hour_of_year",
        "active_support", "tdtr_support", "pbar_mw", "Pctr_mw",
        "cost_eur_per_mwh", "restriction_cost", "restriction_cost_norm",
        "f0_intercept", "f0_hour_sin", "f0_hour_cos", "f0_week_sin", "f0_week_cos",
        "f0_month_sin", "f0_month_cos", "f0_is_weekend",
        "price_da_norm", "rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm",
        "f1_public_stress",
        "burden_proxy_norm", "relief_value_proxy_norm", "proxy_burden_score",
        "proxy_relief_score", "proxy_raw_score", "proxy_raw_score_centered",
        "ptdf_effectivity_norm", "ptdf_exposure_norm",
        "ptdf_requirement_exposure_MW", "ptdf_active_constraints",
        "ptdf_active_constraints_norm", "q_proxy_target",
    ]
    keep_cols = [c for c in keep_cols if c in state.columns]
    return state[keep_cols].sort_values(["year", "province", "hour_of_year"]).reset_index(drop=True)


def build_master_feature_frame(year_inputs: Dict[int, Any]) -> pd.DataFrame:
    frames = [year_input_to_feature_frame(year, yi) for year, yi in year_inputs.items()]
    master = pd.concat(frames, ignore_index=True)

    for col in master.columns:
        if col not in {"province", "timestamp_utc", "split_role"}:
            master[col] = pd.to_numeric(master[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    master["year"] = master["year"].astype(int)
    master["hour_of_year"] = master["hour_of_year"].astype(int)
    master["active_support"] = master["active_support"].astype(int)
    master["split_role"] = master["year"].map(split_role)

    return master.sort_values(["year", "province", "hour_of_year"]).reset_index(drop=True)


def feature_columns_for_level(feature_set: str, available_cols: Sequence[str]) -> List[str]:
    f = canonical_feature_set(feature_set)
    f0 = [
        "f0_intercept", "f0_hour_sin", "f0_hour_cos", "f0_week_sin", "f0_week_cos",
        "f0_month_sin", "f0_month_cos", "f0_is_weekend",
    ]
    f1 = f0 + [
        "price_da_norm", "rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm", "f1_public_stress",
    ]
    f2 = f1 + [
        "burden_proxy_norm", "relief_value_proxy_norm", "restriction_cost_norm",
        "proxy_burden_score", "proxy_relief_score",
    ]
    f3 = f2 + ["ptdf_effectivity_norm", "ptdf_exposure_norm", "ptdf_active_constraints_norm"]
    mapping = {"F0": f0, "F1": f1, "F2": f2, "F3": f3}
    if f not in mapping:
        raise ValueError(f"Unknown feature set: {feature_set}")
    cols = [c for c in mapping[f] if c in available_cols]
    if not cols:
        raise RuntimeError(f"No feature columns available for {f}.")
    return cols


# =============================================================================
# Weighted ridge fallback
# =============================================================================

class NumpyStandardScaler:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "NumpyStandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale <= 1e-12] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted.")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class NumpyWeightedRidge:
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "NumpyWeightedRidge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        if sample_weight is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float)
            w[~np.isfinite(w)] = 1.0
            w = np.maximum(w, EPS)

        if self.fit_intercept:
            Xa = np.column_stack([np.ones(n, dtype=float), X])
            penalty = self.alpha * np.eye(p + 1)
            penalty[0, 0] = 0.0
        else:
            Xa = X
            penalty = self.alpha * np.eye(p)

        sw = np.sqrt(w)
        Xw = Xa * sw[:, None]
        yw = y * sw
        beta = np.linalg.solve(Xw.T @ Xw + penalty, Xw.T @ yw)

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("NumpyWeightedRidge model is not fitted.")
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_


def fit_regression(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_q_col: str,
    feature_set: str,
    variant: str,
    target_source: str,
    sample_weight_col: Optional[str] = None,
) -> FitResult:
    train_mask = df["year"].isin(TRAIN_YEARS)
    if M3_TRAIN_ON_SUPPORT_ONLY:
        train_mask = train_mask & (df["active_support"].astype(int) == 1)
    train = df.loc[train_mask].copy()
    if train.empty:
        raise RuntimeError(f"No training rows for {feature_set}/{variant}.")

    X = train[feature_cols].to_numpy(dtype=float)
    y_q = pd.to_numeric(train[target_q_col], errors="coerce").fillna(1.0).clip(Q_MIN, Q_MAX).to_numpy(dtype=float)
    y = np.log(np.clip(y_q, Q_MIN, Q_MAX))

    if sample_weight_col is not None and sample_weight_col in train.columns:
        weights = pd.to_numeric(train[sample_weight_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy(dtype=float)
        weights = np.clip(weights, EPS, C_ALPHA_SAMPLE_WEIGHT_CAP)
        weight_source = sample_weight_col
    else:
        weights = np.ones(len(train), dtype=float)
        weight_source = "uniform"

    notes: List[str] = []
    if HAS_SKLEARN:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, random_state=0)
        model.fit(Xs, y, sample_weight=weights)
        fitted_with_sklearn = True
    else:
        scaler = NumpyStandardScaler()
        Xs = scaler.fit_transform(X)
        model = NumpyWeightedRidge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(Xs, y, sample_weight=weights)
        fitted_with_sklearn = False
        notes.append("sklearn unavailable; used numpy weighted ridge fallback.")

    return FitResult(
        feature_set=canonical_feature_set(feature_set),
        variant=canonical_variant(variant),
        feature_columns=feature_cols,
        model=model,
        scaler=scaler,
        fitted_with_sklearn=fitted_with_sklearn,
        n_train_rows=int(len(train)),
        target_source=target_source,
        sample_weight_source=weight_source,
        sample_weight_min=float(np.min(weights)),
        sample_weight_mean=float(np.mean(weights)),
        sample_weight_max=float(np.max(weights)),
        notes=notes,
    )


def predict_q(df: pd.DataFrame, fit: FitResult) -> np.ndarray:
    X = df[fit.feature_columns].to_numpy(dtype=float)
    Xs = fit.scaler.transform(X)
    pred_log_q = np.asarray(fit.model.predict(Xs), dtype=float)
    pred_log_q = np.clip(pred_log_q, -20.0, 20.0)
    q_raw = np.exp(pred_log_q)
    return normalize_q_by_province_year(df, q_raw, support_col="active_support")


# =============================================================================
# Optimization-informed alpha diagnostics
# =============================================================================

def solve_one_for_alpha(yi: Any, q_arg: Any, rule_name: str) -> Any:
    if C_ALPHA_SOLVE_MODE == "qp":
        try:
            return B.solve_weighted_depth_qp(
                yi,
                q=q_arg,
                rule=rule_name,
                tau_x=getattr(B, "QP_TAU_X", 1e-6),
                tau_s=getattr(B, "QP_TAU_S", 1e-8),
                output_flag=TRAINING_SUMMARY_OUTPUT_FLAG,
                save=False,
            )
        except Exception as exc:
            if not C_ALPHA_FALLBACK_TO_LP:
                raise
            warnings.warn(f"QP alpha solve failed for {rule_name}, year={yi.year}; falling back to LP. Error: {exc}")
    return B.solve_weighted_depth_lp(
        yi,
        q=q_arg,
        rule_name=rule_name,
        regularized=False,
        output_flag=TRAINING_SUMMARY_OUTPUT_FLAG,
        save=False,
    )


def solve_rule_for_alpha(
    year_inputs: Dict[int, Any],
    q_by_year: Optional[Dict[int, pd.DataFrame]],
    rule_name: str,
    years: Iterable[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Solve weighted-depth problems and collect alpha = mu*x/P."""
    kkt_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for year in years:
        yi = year_inputs[int(year)]
        q_arg: Any = 1.0 if q_by_year is None else q_by_year[int(year)]
        t0 = time.time()
        try:
            res = solve_one_for_alpha(yi, q_arg=q_arg, rule_name=rule_name)
            runtime = time.time() - t0
            kd = B.kkt_equalization_diagnostic(res)
            if kd is not None and not kd.empty:
                kd = kd.copy()
                kd["source_rule"] = rule_name
                kd["alpha_solve_mode"] = C_ALPHA_SOLVE_MODE
                kkt_frames.append(kd)
            row = B.result_summary_row(res)
            row["source_rule"] = rule_name
            row["alpha_solve_mode"] = C_ALPHA_SOLVE_MODE
            row["training_solve_runtime_wall_sec"] = runtime
            row["split_role"] = split_role(int(year))
            summary_rows.append(row)
        except Exception as exc:
            summary_rows.append({
                "year": int(year),
                "rule": rule_name,
                "source_rule": rule_name,
                "alpha_solve_mode": C_ALPHA_SOLVE_MODE,
                "status": "FAILED",
                "quality_flag": "failed_training_alpha_solve",
                "notes": str(exc),
                "split_role": split_role(int(year)),
            })
            warnings.warn(f"Training alpha solve failed for {rule_name}, year={year}: {exc}")

    kkt = pd.concat(kkt_frames, ignore_index=True) if kkt_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return kkt, summary


def attach_alpha_target(
    df: pd.DataFrame,
    kkt: pd.DataFrame,
    target_col: str,
    sample_weight_col: str,
    blend_proxy: float,
    label: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Attach alpha-shaped target and alpha-based sample weights."""
    out = df.copy()
    out[sample_weight_col] = 1.0

    if kkt.empty or "alpha" not in kkt.columns:
        out[target_col] = out["q_proxy_target"].to_numpy(dtype=float)
        return out, {
            "label": label,
            "alpha_available": False,
            "reason": "empty_kkt_or_missing_alpha",
            "alpha_sum_train": 0.0,
            "alpha_max_train": 0.0,
            "blend_proxy": blend_proxy,
            "sample_weight_col": sample_weight_col,
        }

    kk = kkt[["year", "province", "hour_of_year", "alpha"]].copy()
    kk["year"] = pd.to_numeric(kk["year"], errors="coerce").astype("Int64")
    kk = kk.dropna(subset=["year"]).copy()
    kk["year"] = kk["year"].astype(int)
    kk["province"] = kk["province"].astype(str).str.strip()
    kk["hour_of_year"] = pd.to_numeric(kk["hour_of_year"], errors="coerce").astype(int)
    kk["alpha"] = pd.to_numeric(kk["alpha"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    kk = kk.groupby(["year", "province", "hour_of_year"], as_index=False)["alpha"].sum()

    out = out.merge(kk, on=["year", "province", "hour_of_year"], how="left", validate="one_to_one")
    out["alpha"] = pd.to_numeric(out["alpha"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)

    active_train = out["year"].isin(TRAIN_YEARS) & (out["active_support"].astype(int) == 1)
    alpha_train = out.loc[active_train, "alpha"].to_numpy(dtype=float)
    alpha_sum = float(np.sum(alpha_train)) if alpha_train.size else 0.0
    alpha_max = float(np.max(alpha_train)) if alpha_train.size else 0.0

    if alpha_sum <= 1e-10 or alpha_max <= 1e-12:
        out[target_col] = out["q_proxy_target"].to_numpy(dtype=float)
        out = out.drop(columns=["alpha"], errors="ignore")
        return out, {
            "label": label,
            "alpha_available": False,
            "reason": "alpha_zero_or_budget_nonbinding",
            "alpha_sum_train": alpha_sum,
            "alpha_max_train": alpha_max,
            "blend_proxy": blend_proxy,
            "sample_weight_col": sample_weight_col,
        }

    out["alpha_norm"] = (
        out.groupby(["year", "province"], sort=False)["alpha"]
        .transform(safe_minmax)
        .astype(float)
    )
    out["alpha_norm_centered"] = (
        out["alpha_norm"]
        - out.groupby(["year", "province"], sort=False)["alpha_norm"].transform("mean")
    )

    # Higher alpha should make the hour cheaper in annual-debit terms.
    q_alpha_raw = np.exp(np.clip(-TARGET_GAIN * out["alpha_norm_centered"].to_numpy(dtype=float), -20.0, 20.0))
    q_alpha = normalize_q_by_province_year(out, q_alpha_raw, support_col="active_support")

    q_proxy = out["q_proxy_target"].to_numpy(dtype=float)
    q_blend_raw = blend_proxy * q_proxy + (1.0 - blend_proxy) * q_alpha
    out[target_col] = normalize_q_by_province_year(out, q_blend_raw, support_col="active_support")

    # True alpha-weighted empirical risk. We normalize by the positive-alpha mean
    # on active training support and cap weights to avoid single-row domination.
    pos_alpha = alpha_train[alpha_train > 0]
    mean_pos_alpha = float(np.mean(pos_alpha)) if pos_alpha.size else 1.0
    if not np.isfinite(mean_pos_alpha) or mean_pos_alpha <= EPS:
        mean_pos_alpha = 1.0
    w = 1.0 + C_ALPHA_SAMPLE_WEIGHT_GAIN * out["alpha"].to_numpy(dtype=float) / mean_pos_alpha
    w = np.clip(w, 1.0, C_ALPHA_SAMPLE_WEIGHT_CAP)
    out[sample_weight_col] = w

    meta = {
        "label": label,
        "alpha_available": True,
        "reason": "alpha_blended_with_proxy_and_weighted_loss",
        "alpha_sum_train": alpha_sum,
        "alpha_max_train": alpha_max,
        "alpha_mean_positive_train": mean_pos_alpha,
        "blend_proxy": blend_proxy,
        "sample_weight_col": sample_weight_col,
        "sample_weight_gain": C_ALPHA_SAMPLE_WEIGHT_GAIN,
        "sample_weight_cap": C_ALPHA_SAMPLE_WEIGHT_CAP,
        "sample_weight_max": float(np.max(w)),
    }

    return out.drop(columns=["alpha", "alpha_norm", "alpha_norm_centered"], errors="ignore"), meta


# =============================================================================
# Training architectures
# =============================================================================

def build_q_dataframe_for_year(df: pd.DataFrame, q_col: str, year: int) -> pd.DataFrame:
    sub = df.loc[df["year"].astype(int) == int(year), ["province", "hour_of_year", q_col]].copy()
    sub = sub.rename(columns={q_col: "debit_weight"})
    sub["province"] = sub["province"].astype(str).str.strip()
    sub["hour_of_year"] = pd.to_numeric(sub["hour_of_year"], errors="raise").astype(int)
    sub["debit_weight"] = pd.to_numeric(sub["debit_weight"], errors="coerce").fillna(1.0).clip(Q_MIN, Q_MAX)
    return sub[["province", "hour_of_year", "debit_weight"]]


def train_plain(df: pd.DataFrame, feature_set: str) -> Tuple[pd.DataFrame, FitResult, Dict[str, Any]]:
    f = canonical_feature_set(feature_set)
    feature_cols = feature_columns_for_level(f, df.columns)
    work = df.copy()
    target_col = f"q_target_{f}_plain"
    work[target_col] = work["q_proxy_target"]

    fit = fit_regression(
        work,
        feature_cols=feature_cols,
        target_q_col=target_col,
        feature_set=f,
        variant="M3_plain",
        target_source="proxy_grounded_q_proxy_target",
        sample_weight_col=None,
    )
    q_col = f"q_{f}_M3_plain"
    work[q_col] = predict_q(work, fit)

    meta = {
        "feature_set": f,
        "variant": "M3_plain",
        "feature_columns": feature_cols,
        "n_train_rows": fit.n_train_rows,
        "target_source": fit.target_source,
        "sample_weight_source": fit.sample_weight_source,
        "sample_weight_min": fit.sample_weight_min,
        "sample_weight_mean": fit.sample_weight_mean,
        "sample_weight_max": fit.sample_weight_max,
        "notes": fit.notes,
    }
    return work, fit, meta


def train_oi(
    df: pd.DataFrame,
    feature_set: str,
    kkt_m2: pd.DataFrame,
) -> Tuple[pd.DataFrame, FitResult, Dict[str, Any]]:
    f = canonical_feature_set(feature_set)
    work, alpha_meta = attach_alpha_target(
        df,
        kkt=kkt_m2,
        target_col=f"q_target_{f}_oi",
        sample_weight_col=f"sample_weight_{f}_oi",
        blend_proxy=0.65,
        label=f"OI_{f}",
    )
    feature_cols = feature_columns_for_level(f, work.columns)
    fit = fit_regression(
        work,
        feature_cols=feature_cols,
        target_q_col=f"q_target_{f}_oi",
        feature_set=f,
        variant="M3_OI",
        target_source=f"optimization_informed_m2_alpha:{alpha_meta['reason']}",
        sample_weight_col=f"sample_weight_{f}_oi",
    )
    q_col = f"q_{f}_M3_OI"
    work[q_col] = predict_q(work, fit)

    meta = {
        "feature_set": f,
        "variant": "M3_OI",
        "feature_columns": feature_cols,
        "n_train_rows": fit.n_train_rows,
        "target_source": fit.target_source,
        "sample_weight_source": fit.sample_weight_source,
        "sample_weight_min": fit.sample_weight_min,
        "sample_weight_mean": fit.sample_weight_mean,
        "sample_weight_max": fit.sample_weight_max,
        "alpha_meta": alpha_meta,
        "notes": fit.notes,
    }
    return work, fit, meta


def train_lo(
    df: pd.DataFrame,
    year_inputs: Dict[int, Any],
    feature_set: str,
    oi_fit: FitResult,
) -> Tuple[pd.DataFrame, FitResult, Dict[str, Any], pd.DataFrame]:
    f = canonical_feature_set(feature_set)
    work = df.copy()
    current_q_col = f"q_{f}_M3_LO_iter0"
    work[current_q_col] = predict_q(work, oi_fit)

    all_summaries: List[pd.DataFrame] = []
    iter_meta: List[Dict[str, Any]] = []
    last_fit = oi_fit

    for it in range(1, LO_ITERS + 1):
        print(f"[LO] feature_set={f} iteration={it}/{LO_ITERS}")
        q_by_year = {int(year): build_q_dataframe_for_year(work, current_q_col, int(year)) for year in TRAIN_YEARS}
        kkt_lo, sol_summary = solve_rule_for_alpha(
            year_inputs=year_inputs,
            q_by_year=q_by_year,
            rule_name=f"M3_{f}_LO_iter{it}_alpha",
            years=TRAIN_YEARS,
        )
        if not sol_summary.empty:
            sol_summary["feature_set"] = f
            sol_summary["variant"] = f"M3_LO_iter{it}_alpha_source"
            all_summaries.append(sol_summary)

        target_col = f"q_target_{f}_lo_iter{it}"
        weight_col = f"sample_weight_{f}_lo_iter{it}"
        work, alpha_meta = attach_alpha_target(
            work,
            kkt=kkt_lo,
            target_col=target_col,
            sample_weight_col=weight_col,
            blend_proxy=0.55,
            label=f"LO_{f}_iter{it}",
        )
        feature_cols = feature_columns_for_level(f, work.columns)
        last_fit = fit_regression(
            work,
            feature_cols=feature_cols,
            target_q_col=target_col,
            feature_set=f,
            variant="M3_LO",
            target_source=f"learn_optimize_iter{it}:{alpha_meta['reason']}",
            sample_weight_col=weight_col,
        )
        current_q_col = f"q_{f}_M3_LO_iter{it}"
        work[current_q_col] = predict_q(work, last_fit)

        iter_meta.append({
            "iteration": it,
            "alpha_meta": alpha_meta,
            "n_train_rows": last_fit.n_train_rows,
            "target_source": last_fit.target_source,
            "sample_weight_source": last_fit.sample_weight_source,
            "sample_weight_min": last_fit.sample_weight_min,
            "sample_weight_mean": last_fit.sample_weight_mean,
            "sample_weight_max": last_fit.sample_weight_max,
        })

    final_q_col = f"q_{f}_M3_LO"
    work[final_q_col] = work[current_q_col].to_numpy(dtype=float)
    meta = {
        "feature_set": f,
        "variant": "M3_LO",
        "feature_columns": last_fit.feature_columns,
        "n_train_rows": last_fit.n_train_rows,
        "target_source": last_fit.target_source,
        "sample_weight_source": last_fit.sample_weight_source,
        "sample_weight_min": last_fit.sample_weight_min,
        "sample_weight_mean": last_fit.sample_weight_mean,
        "sample_weight_max": last_fit.sample_weight_max,
        "lo_iters": LO_ITERS,
        "iterations": iter_meta,
        "notes": last_fit.notes,
    }
    summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    return work, last_fit, meta, summary


# =============================================================================
# Output builders and audits
# =============================================================================

def make_long_weights(master: pd.DataFrame, q_columns: Dict[Tuple[str, str], str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    base_cols = ["year", "split_role", "province", "timestamp_utc", "hour_of_year", "active_support"]
    for (feature_set, variant), q_col in q_columns.items():
        f = canonical_feature_set(feature_set)
        v = canonical_variant(variant)
        sub = master[base_cols + [q_col]].copy()
        sub = sub.rename(columns={q_col: "debit_weight"})
        sub["feature_set"] = f
        sub["feature_level"] = feature_level_int(f)
        sub["variant"] = v
        sub["architecture"] = v
        sub["debit_weight"] = pd.to_numeric(sub["debit_weight"], errors="coerce").fillna(1.0).astype(float)
        sub["debit_weight"] = np.clip(sub["debit_weight"], Q_MIN, Q_MAX)
        if M3_SET_Q_OUTSIDE_SUPPORT_TO_ONE:
            sub.loc[sub["active_support"].astype(int) == 0, "debit_weight"] = 1.0
        rows.append(sub)

    long = pd.concat(rows, ignore_index=True)
    long = long[[
        "year", "split_role", "province", "timestamp_utc", "hour_of_year",
        "feature_set", "feature_level", "variant", "architecture", "debit_weight", "active_support",
    ]].copy()
    long["year"] = long["year"].astype(int)
    long["hour_of_year"] = long["hour_of_year"].astype(int)
    long["feature_level"] = long["feature_level"].astype(int)
    long["active_support"] = long["active_support"].astype(int)
    long["split_role"] = long["year"].map(split_role)
    long = long.sort_values(["year", "feature_set", "variant", "province", "hour_of_year"]).reset_index(drop=True)

    dup = int(long.duplicated(["year", "province", "hour_of_year", "feature_set", "variant"]).sum())
    if dup:
        raise RuntimeError(f"Long M3 weights contain {dup} duplicate keys.")
    return long


def make_wide_weights(long: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    idx_cols = ["year", "split_role", "province", "timestamp_utc", "hour_of_year", "feature_set", "feature_level", "active_support"]
    wide = long[idx_cols].drop_duplicates().copy()
    for variant in ["M3_plain", "M3_OI", "M3_LO"]:
        sub = long.loc[long["variant"] == variant, idx_cols + ["debit_weight"]].copy()
        if sub.empty:
            continue
        col = f"q_m3_{variant_short(variant)}"
        wide = wide.merge(sub.rename(columns={"debit_weight": col}), on=idx_cols, how="left", validate="one_to_one")

    if "q_m3_LO" in wide.columns:
        wide["q_m3"] = wide["q_m3_LO"]
    elif "q_m3_OI" in wide.columns:
        wide["q_m3"] = wide["q_m3_OI"]
    elif "q_m3_plain" in wide.columns:
        wide["q_m3"] = wide["q_m3_plain"]
    else:
        wide["q_m3"] = 1.0

    audit_cols = [
        "year", "province", "hour_of_year", "burden_proxy_norm", "relief_value_proxy_norm",
        "ptdf_effectivity_norm", "ptdf_exposure_norm", "restriction_cost_norm", "price_da_norm",
        "rl_plus_norm", "s_plus_norm", "s_minus_norm", "q_proxy_target",
    ]
    audit_cols = [c for c in audit_cols if c in master.columns]
    audit = master[audit_cols].drop_duplicates(["year", "province", "hour_of_year"]).copy()
    wide = wide.merge(audit, on=["year", "province", "hour_of_year"], how="left", validate="many_to_one")
    return wide.sort_values(["year", "feature_set", "province", "hour_of_year"]).reset_index(drop=True)


def _quantile(x: pd.Series, q: float) -> float:
    arr = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def q_audit_table(long: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["year", "split_role", "feature_set", "variant"]
    for keys, g in long.groupby(group_cols, sort=True):
        year, role, feature_set, variant = keys
        q_all = pd.to_numeric(g["debit_weight"], errors="coerce").astype(float)
        support = g[g["active_support"].astype(int) == 1].copy()
        off = g[g["active_support"].astype(int) == 0].copy()
        q_sup = pd.to_numeric(support["debit_weight"], errors="coerce").astype(float) if not support.empty else pd.Series(dtype=float)

        pv = support.groupby("province")["debit_weight"].mean() if not support.empty else pd.Series(dtype=float)
        off_dev = float(np.max(np.abs(pd.to_numeric(off["debit_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float) - 1.0))) if not off.empty else 0.0
        rows.append({
            "year": int(year),
            "split_role": role,
            "feature_set": feature_set,
            "variant": variant,
            "n_rows_all": int(len(g)),
            "n_rows_support": int(len(support)),
            "n_provinces": int(g["province"].nunique()),
            "n_hours_all": int(g["hour_of_year"].nunique()),
            "n_hours_support": int(support["hour_of_year"].nunique()) if not support.empty else 0,
            "q_all_min": float(q_all.min()),
            "q_all_mean": float(q_all.mean()),
            "q_all_std": float(q_all.std(ddof=0)),
            "q_all_max": float(q_all.max()),
            "q_support_min": float(q_sup.min()) if not q_sup.empty else np.nan,
            "q_support_p01": _quantile(q_sup, 0.01) if not q_sup.empty else np.nan,
            "q_support_mean": float(q_sup.mean()) if not q_sup.empty else np.nan,
            "q_support_std": float(q_sup.std(ddof=0)) if not q_sup.empty else np.nan,
            "q_support_p50": _quantile(q_sup, 0.50) if not q_sup.empty else np.nan,
            "q_support_p99": _quantile(q_sup, 0.99) if not q_sup.empty else np.nan,
            "q_support_max": float(q_sup.max()) if not q_sup.empty else np.nan,
            "q_support_nontrivial_share": float(np.mean(np.abs(q_sup.to_numpy(dtype=float) - 1.0) > 1e-6)) if not q_sup.empty else np.nan,
            "max_abs_province_support_mean_minus_one": float(np.max(np.abs(pv.to_numpy(dtype=float) - 1.0))) if not pv.empty else np.nan,
            "off_support_max_abs_q_minus_one": off_dev,
        })
    return pd.DataFrame(rows).sort_values(["year", "feature_set", "variant"]).reset_index(drop=True)


def assert_required_main_slice(long: pd.DataFrame, training_meta: List[Dict[str, Any]]) -> None:
    f_main = canonical_feature_set(MAIN_FEATURE_SET)
    v_main = canonical_variant(MAIN_M3_VARIANT)
    required = pd.MultiIndex.from_product(
        [APPLY_YEARS, [f_main], [v_main]],
        names=["year", "feature_set", "variant"],
    ).to_frame(index=False)
    have = long.groupby(["year", "feature_set", "variant"]).size().reset_index(name="n")
    missing = required.merge(have, on=["year", "feature_set", "variant"], how="left")
    missing = missing[missing["n"].isna() | (missing["n"] <= 0)]
    if not missing.empty:
        raise RuntimeError("C did not write required main M3 weights:\n" + missing.to_string(index=False))

    main = long[(long["feature_set"] == f_main) & (long["variant"] == v_main) & (long["year"].isin(APPLY_YEARS))].copy()
    bad = main[~np.isfinite(main["debit_weight"]) | (main["debit_weight"] <= 0)]
    if not bad.empty:
        raise RuntimeError("Main M3 slice contains invalid debit_weight values.")

    support = main[main["active_support"].astype(int) == 1]
    if not support.empty:
        means = support.groupby(["year", "province"])["debit_weight"].mean()
        max_err = float(np.max(np.abs(means.to_numpy(dtype=float) - 1.0)))
        if max_err > 1e-6:
            raise RuntimeError(f"Main M3 active-support q mean-one audit failed. max error={max_err:.3e}")

    off = main[main["active_support"].astype(int) == 0]
    if M3_SET_Q_OUTSIDE_SUPPORT_TO_ONE and not off.empty:
        off_err = float(np.max(np.abs(off["debit_weight"].to_numpy(dtype=float) - 1.0)))
        if off_err > 1e-12:
            raise RuntimeError(f"Main M3 off-support q audit failed. max |q-1|={off_err:.3e}")

    if C_REQUIRE_ALPHA_FOR_MAIN and v_main in {"M3_OI", "M3_LO"}:
        relevant = [m for m in training_meta if m.get("feature_set") == f_main and m.get("variant") == v_main]
        if not relevant:
            raise RuntimeError(f"No training metadata found for main {f_main}/{v_main}.")
        meta = relevant[-1]
        alpha_ok = False
        if v_main == "M3_OI":
            alpha_ok = bool(meta.get("alpha_meta", {}).get("alpha_available", False))
        elif v_main == "M3_LO":
            iterations = meta.get("iterations", [])
            alpha_ok = any(bool(it.get("alpha_meta", {}).get("alpha_available", False)) for it in iterations)
        if not alpha_ok:
            raise RuntimeError(
                f"Main variant {f_main}/{v_main} did not receive a usable alpha signal. "
                "For final M3_LO/OI paper experiments, rerun after fixing B/KKT or set C_REQUIRE_ALPHA_FOR_MAIN=0 for proxy-only diagnostics."
            )


# =============================================================================
# Optional training solution summaries
# =============================================================================

def solve_training_summaries(year_inputs: Dict[int, Any], long: pd.DataFrame) -> pd.DataFrame:
    if not SOLVE_TRAINING_SUMMARY:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    if SOLVE_ALL_TRAINING_SUMMARIES:
        combos = list(long[["feature_set", "variant"]].drop_duplicates().sort_values(["feature_set", "variant"]).itertuples(index=False, name=None))
    else:
        combos = [(canonical_feature_set(MAIN_FEATURE_SET), canonical_variant(MAIN_M3_VARIANT))]

    for feature_set, variant in combos:
        f = canonical_feature_set(feature_set)
        v = canonical_variant(variant)
        rule_name = f"M3_{f}_{v}"
        for year in APPLY_YEARS:
            print(f"[SUMMARY SOLVE] {rule_name} year={year}")
            qdf = long[(long["year"] == int(year)) & (long["feature_set"] == f) & (long["variant"] == v)][["province", "hour_of_year", "debit_weight"]].copy()
            try:
                res = B.solve_weighted_depth_lp(
                    year_inputs[int(year)],
                    q=qdf,
                    rule_name=rule_name,
                    regularized=False,
                    output_flag=TRAINING_SUMMARY_OUTPUT_FLAG,
                    save=False,
                )
                row = B.result_summary_row(res)
                row["feature_set"] = f
                row["variant"] = v
                row["architecture"] = v
                row["split_role"] = split_role(int(year))
                rows.append(row)
            except Exception as exc:
                rows.append({
                    "year": int(year),
                    "split_role": split_role(int(year)),
                    "rule": rule_name,
                    "feature_set": f,
                    "variant": v,
                    "architecture": v,
                    "status": "FAILED",
                    "quality_flag": "failed_training_summary_solve",
                    "notes": str(exc),
                })
                warnings.warn(f"Training summary solve failed for {rule_name}, year={year}: {exc}")
    return pd.DataFrame(rows)


# =============================================================================
# Manifest and logging
# =============================================================================

def print_header() -> None:
    print("=" * 78)
    print("C: TRAIN SUPPORT-AWARE AUDITABLE M3 DEBIT WEIGHTS -- SIGNED-OFF VERSION")
    print("=" * 78)
    print(f"Repo root              : {REPO_ROOT}")
    print(f"Derived dir            : {DERIVED_DIR}")
    print(f"Results dir            : {RESULTS_DIR}")
    print(f"Log dir                : {LOG_DIR}")
    print(f"Apply years            : {APPLY_YEARS}")
    print(f"Train years            : {TRAIN_YEARS}")
    print(f"Validation year        : {VALIDATION_YEAR}")
    print(f"Test year              : {TEST_YEAR}")
    print(f"Feature sets           : {FEATURE_SETS}")
    print(f"Variants               : {[canonical_variant(v) for v in VARIANTS]}")
    print(f"Main feature/variant   : {canonical_feature_set(MAIN_FEATURE_SET)} / {canonical_variant(MAIN_M3_VARIANT)}")
    print(f"q bounds               : [{Q_MIN}, {Q_MAX}]")
    print(f"Normalize on support   : {M3_NORMALIZE_ON_SUPPORT_ONLY}")
    print(f"Train on support only  : {M3_TRAIN_ON_SUPPORT_ONLY}")
    print(f"q outside support = 1  : {M3_SET_Q_OUTSIDE_SUPPORT_TO_ONE}")
    print(f"Alpha solve mode       : {C_ALPHA_SOLVE_MODE}")
    print(f"Alpha fallback to LP   : {C_ALPHA_FALLBACK_TO_LP}")
    print(f"Alpha sample weight    : gain={C_ALPHA_SAMPLE_WEIGHT_GAIN}, cap={C_ALPHA_SAMPLE_WEIGHT_CAP}")
    print(f"Require main alpha     : {C_REQUIRE_ALPHA_FOR_MAIN}")
    print(f"LO iterations          : {LO_ITERS}")
    print(f"Ridge alpha            : {RIDGE_ALPHA}")
    print(f"sklearn available      : {HAS_SKLEARN}")
    print(f"Solve summary          : {SOLVE_TRAINING_SUMMARY}")
    print(f"Solve all summaries    : {SOLVE_ALL_TRAINING_SUMMARIES}")
    print(f"Long output parquet    : {WEIGHTS_LONG_PARQUET}")
    print("=" * 78)


def build_manifest(
    long: pd.DataFrame,
    wide: pd.DataFrame,
    q_audit: pd.DataFrame,
    training_meta: List[Dict[str, Any]],
    training_log: pd.DataFrame,
    solution_summary: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "script": "C_train_m3.py",
        "version": "signed_off_experiment_ready",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "repo_root": str(REPO_ROOT),
        "derived_dir": str(DERIVED_DIR),
        "results_dir": str(RESULTS_DIR),
        "log_dir": str(LOG_DIR),
        "years": {
            "all_years": ALL_YEARS,
            "apply_years": APPLY_YEARS,
            "train_years": TRAIN_YEARS,
            "validation_year": VALIDATION_YEAR,
            "test_year": TEST_YEAR,
            "split_roles": {str(y): split_role(int(y)) for y in sorted(set(APPLY_YEARS) | set(TRAIN_YEARS) | {VALIDATION_YEAR, TEST_YEAR})},
        },
        "main_configuration": {
            "main_feature_set": canonical_feature_set(MAIN_FEATURE_SET),
            "main_variant": canonical_variant(MAIN_M3_VARIANT),
            "require_alpha_for_main": C_REQUIRE_ALPHA_FOR_MAIN,
        },
        "feature_ladder": {
            "F0": "calendar-only public baseline",
            "F1": "F0 + public load/price stress",
            "F2": "F1 + burden/relief/cost proxies",
            "F3": "F2 + constructed PTDF-testbed effectivity/exposure proxies",
        },
        "variants": {
            "M3_plain": "proxy-grounded auditable fit",
            "M3_OI": "one-shot optimization-informed alpha-shaped and alpha-weighted fit from M2 diagnostics",
            "M3_LO": "alternating learn-optimize alpha-shaped and alpha-weighted fit initialized from M3_OI",
        },
        "q_constraints": {
            "q_min": Q_MIN,
            "q_max": Q_MAX,
            "normalization_method": "exact bounded mean-one projection by province-year on active support",
            "normalize_on_support_only": M3_NORMALIZE_ON_SUPPORT_ONLY,
            "train_on_support_only": M3_TRAIN_ON_SUPPORT_ONLY,
            "set_q_outside_support_to_one": M3_SET_Q_OUTSIDE_SUPPORT_TO_ONE,
        },
        "alpha_training": {
            "alpha_definition": "alpha_{n,t}=mu_n*x_{n,t}/P_n",
            "incorrect_alpha_not_used": "mu_n*q_{n,t}*x_{n,t}/P_n",
            "alpha_solve_mode": C_ALPHA_SOLVE_MODE,
            "alpha_fallback_to_lp": C_ALPHA_FALLBACK_TO_LP,
            "sample_weight_gain": C_ALPHA_SAMPLE_WEIGHT_GAIN,
            "sample_weight_cap": C_ALPHA_SAMPLE_WEIGHT_CAP,
            "uses_alpha_shaped_targets": True,
            "uses_alpha_weighted_regression": True,
        },
        "modelling_discipline": {
            "physical_relief_layer_fixed": True,
            "q_is_accounting_debit_rate_not_price": True,
            "no_finite_sample_monotonicity_claim_from_richer_features": True,
            "final_benchmark_source": "D_run_full_benchmark.py, not C_train_m3.py",
        },
        "outputs": {
            "long_parquet": str(WEIGHTS_LONG_PARQUET),
            "long_csv": str(WEIGHTS_LONG_CSV),
            "wide_parquet": str(WEIGHTS_WIDE_PARQUET),
            "wide_csv": str(WEIGHTS_WIDE_CSV),
            "training_log_csv": str(TRAINING_LOG_CSV),
            "training_solution_summary_csv": str(TRAINING_SOLUTION_SUMMARY_CSV),
            "q_audit_csv": str(TRAINING_Q_AUDIT_CSV),
        },
        "dimensions": {
            "long_rows": int(len(long)),
            "wide_rows": int(len(wide)),
            "q_audit_rows": int(len(q_audit)),
            "training_log_rows": int(len(training_log)),
            "solution_summary_rows": int(len(solution_summary)),
        },
        "available_slices": long.groupby(["year", "split_role", "feature_set", "variant"]).size().reset_index(name="n_rows").to_dict("records"),
        "training_meta": training_meta,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()
    print_header()
    t_script = time.time()

    log_rows: List[Dict[str, Any]] = []
    training_meta: List[Dict[str, Any]] = []
    solution_summaries_from_alpha: List[pd.DataFrame] = []

    required_years = sorted(set(APPLY_YEARS) | set(TRAIN_YEARS) | {VALIDATION_YEAR, TEST_YEAR})
    year_inputs = load_all_year_inputs(required_years)
    master = build_master_feature_frame({y: year_inputs[y] for y in APPLY_YEARS})
    print(f"[FEATURES] master rows={len(master):,}, columns={len(master.columns):,}")

    variants_canon = [canonical_variant(v) for v in VARIANTS]
    need_oi_or_lo = ("M3_OI" in variants_canon) or ("M3_LO" in variants_canon)

    # M2 alpha is feature-independent. Compute once and reuse across F0-F3.
    kkt_m2 = pd.DataFrame()
    m2_alpha_summary = pd.DataFrame()
    if need_oi_or_lo:
        print("\n[ALPHA] Solving M2 diagnostics once for all OI initializations")
        kkt_m2, m2_alpha_summary = solve_rule_for_alpha(
            year_inputs=year_inputs,
            q_by_year=None,
            rule_name="M2_for_OI_alpha",
            years=TRAIN_YEARS,
        )
        if not m2_alpha_summary.empty:
            m2_alpha_summary["feature_set"] = "ALL"
            m2_alpha_summary["variant"] = "M3_OI_alpha_source"
            solution_summaries_from_alpha.append(m2_alpha_summary)
        alpha_sum = float(pd.to_numeric(kkt_m2.get("alpha", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not kkt_m2.empty else 0.0
        print(f"[ALPHA] M2 alpha rows={len(kkt_m2):,}, alpha_sum={alpha_sum:,.6g}")

    q_columns: Dict[Tuple[str, str], str] = {}

    for feature_set_raw in FEATURE_SETS:
        f = canonical_feature_set(feature_set_raw)
        print("\n" + "=" * 78)
        print(f"[FEATURE SET] {f}")
        print("=" * 78)

        fit_oi: Optional[FitResult] = None

        if "M3_plain" in variants_canon:
            t0 = time.time()
            work_plain, fit_plain, meta_plain = train_plain(master, f)
            q_col_plain = f"q_{f}_M3_plain"
            master[q_col_plain] = work_plain[q_col_plain].to_numpy(dtype=float)
            q_columns[(f, "M3_plain")] = q_col_plain
            training_meta.append(meta_plain)
            log_rows.append({
                "feature_set": f,
                "variant": "M3_plain",
                "status": "ok",
                "runtime_sec": time.time() - t0,
                "n_train_rows": fit_plain.n_train_rows,
                "target_source": fit_plain.target_source,
                "sample_weight_source": fit_plain.sample_weight_source,
                "sample_weight_min": fit_plain.sample_weight_min,
                "sample_weight_mean": fit_plain.sample_weight_mean,
                "sample_weight_max": fit_plain.sample_weight_max,
                "feature_columns": ",".join(fit_plain.feature_columns),
                "notes": " | ".join(fit_plain.notes),
            })
            print(f"[TRAIN] {f}/M3_plain rows={fit_plain.n_train_rows:,}")

        if need_oi_or_lo:
            t0 = time.time()
            work_oi, fit_oi, meta_oi = train_oi(master, f, kkt_m2=kkt_m2)
            q_col_oi = f"q_{f}_M3_OI"
            master[q_col_oi] = work_oi[q_col_oi].to_numpy(dtype=float)
            if "M3_OI" in variants_canon:
                q_columns[(f, "M3_OI")] = q_col_oi
            training_meta.append(meta_oi)
            alpha_meta = meta_oi.get("alpha_meta", {})
            log_rows.append({
                "feature_set": f,
                "variant": "M3_OI",
                "status": "ok",
                "runtime_sec": time.time() - t0,
                "n_train_rows": fit_oi.n_train_rows,
                "target_source": fit_oi.target_source,
                "sample_weight_source": fit_oi.sample_weight_source,
                "sample_weight_min": fit_oi.sample_weight_min,
                "sample_weight_mean": fit_oi.sample_weight_mean,
                "sample_weight_max": fit_oi.sample_weight_max,
                "alpha_available": alpha_meta.get("alpha_available", False),
                "alpha_reason": alpha_meta.get("reason", ""),
                "alpha_sum_train": alpha_meta.get("alpha_sum_train", np.nan),
                "alpha_max_train": alpha_meta.get("alpha_max_train", np.nan),
                "feature_columns": ",".join(fit_oi.feature_columns),
                "notes": " | ".join(fit_oi.notes),
            })
            print(f"[TRAIN] {f}/M3_OI rows={fit_oi.n_train_rows:,} alpha={alpha_meta.get('reason')}")

        if "M3_LO" in variants_canon:
            if fit_oi is None:
                raise RuntimeError("M3_LO requires M3_OI initialization.")
            t0 = time.time()
            work_lo, fit_lo, meta_lo, lo_summary = train_lo(master, year_inputs, f, fit_oi)
            q_col_lo = f"q_{f}_M3_LO"
            master[q_col_lo] = work_lo[q_col_lo].to_numpy(dtype=float)
            q_columns[(f, "M3_LO")] = q_col_lo
            training_meta.append(meta_lo)
            if not lo_summary.empty:
                solution_summaries_from_alpha.append(lo_summary)
            last_iter = meta_lo.get("iterations", [])[-1] if meta_lo.get("iterations") else {}
            alpha_meta = last_iter.get("alpha_meta", {}) if isinstance(last_iter, dict) else {}
            log_rows.append({
                "feature_set": f,
                "variant": "M3_LO",
                "status": "ok",
                "runtime_sec": time.time() - t0,
                "n_train_rows": fit_lo.n_train_rows,
                "target_source": fit_lo.target_source,
                "sample_weight_source": fit_lo.sample_weight_source,
                "sample_weight_min": fit_lo.sample_weight_min,
                "sample_weight_mean": fit_lo.sample_weight_mean,
                "sample_weight_max": fit_lo.sample_weight_max,
                "alpha_available": alpha_meta.get("alpha_available", False),
                "alpha_reason": alpha_meta.get("reason", ""),
                "alpha_sum_train": alpha_meta.get("alpha_sum_train", np.nan),
                "alpha_max_train": alpha_meta.get("alpha_max_train", np.nan),
                "feature_columns": ",".join(fit_lo.feature_columns),
                "notes": " | ".join(fit_lo.notes),
            })
            print(f"[TRAIN] {f}/M3_LO rows={fit_lo.n_train_rows:,} final_alpha={alpha_meta.get('reason')}")

    if not q_columns:
        raise RuntimeError("No M3 q columns were trained. Check C_FEATURE_SETS and C_M3_VARIANTS.")

    long = make_long_weights(master, q_columns)
    assert_required_main_slice(long, training_meta)
    wide = make_wide_weights(long, master)
    q_audit = q_audit_table(long)

    save_table_both(long, WEIGHTS_LONG_PARQUET, WEIGHTS_LONG_CSV)
    save_table_both(wide, WEIGHTS_WIDE_PARQUET, WEIGHTS_WIDE_CSV)

    q_audit.to_csv(TRAINING_Q_AUDIT_CSV, index=False)
    print(f"[SAVE] {TRAINING_Q_AUDIT_CSV}")

    training_log = pd.DataFrame(log_rows)
    training_log.to_csv(TRAINING_LOG_CSV, index=False)
    print(f"[SAVE] {TRAINING_LOG_CSV}")

    solution_summary_final = solve_training_summaries(year_inputs, long)
    if solution_summaries_from_alpha:
        solution_summary_alpha = pd.concat(solution_summaries_from_alpha, ignore_index=True)
        solution_summary = pd.concat([solution_summary_alpha, solution_summary_final], ignore_index=True) if not solution_summary_final.empty else solution_summary_alpha
    else:
        solution_summary = solution_summary_final
    solution_summary.to_csv(TRAINING_SOLUTION_SUMMARY_CSV, index=False)
    print(f"[SAVE] {TRAINING_SOLUTION_SUMMARY_CSV}")

    manifest = build_manifest(
        long=long,
        wide=wide,
        q_audit=q_audit,
        training_meta=training_meta,
        training_log=training_log,
        solution_summary=solution_summary,
    )
    manifest["runtime_sec"] = float(time.time() - t_script)
    write_json(manifest, TRAINING_MANIFEST_JSON)

    print("\n[DONE]")
    print(f"Long weights : {WEIGHTS_LONG_PARQUET}")
    print(f"Wide weights : {WEIGHTS_WIDE_PARQUET}")
    print(f"Q audit      : {TRAINING_Q_AUDIT_CSV}")
    print(f"Log          : {TRAINING_LOG_CSV}")
    print(f"Manifest     : {TRAINING_MANIFEST_JSON}")
    print("\nNext step: run D_run_full_benchmark.py. Final paper results should come from D, not this C summary.")


if __name__ == "__main__":
    main()
