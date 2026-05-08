#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
G_build_tdtr_candidate_hour_support.py
======================================

Build TDTR candidate-hour support sets for the annual debit-rule paper.

Purpose
-------
This script constructs the annual set of hours in which TDTR-like non-firm
restrictions are eligible to occur. The output is a support mask, not a
dispatch solution and not a curtailment allocation.

Conceptual separation
---------------------
1. Public stress / auditable state panel:
       data/derived/auditable_state_province_hour.csv

   This gives province-hour public stress signals.

2. TDTR candidate-hour support:
       select top alpha share of annual hours by public stress score.

   This determines where non-firm restrictions may be considered.

3. Debit-rule optimization:
       M1, M2, M3 are solved later on the same support-constrained testbed.

Main design
-----------
For each year y and scenario r:

    score_{p,t}^{r,y} = public directional province-hour stress

    score_t^{r,y} = aggregate_p score_{p,t}^{r,y}

    W_y^r = top K_y^r hours by score_t^{r,y},
            K_y^r = floor/ceil/round(alpha_r * |T_y|)

The support is common across M1/M2/M3 for a given scenario. It should later be
used to mask both:

    R_{k,t}       -> support_t * R_{k,t}
    Pbar_{p,t}   -> support_t * Pbar_{p,t}

Outputs
-------
Combined:
    data/derived/tdtr_candidate_hour_support_all_scenarios.csv
    data/derived/tdtr_candidate_hour_support_province_scores_all_scenarios.csv
    qc/tdtr_candidate_hour_support_summary_all_scenarios.json
    qc/tdtr_candidate_hour_support_overlap.csv

Backward-compatible main scenario:
    data/derived/tdtr_candidate_hour_support.csv
    data/derived/tdtr_candidate_hour_support_province_scores.csv
    qc/tdtr_candidate_hour_support_summary.json

Scenario-specific:
    data/derived/tdtr_candidate_hour_support__{scenario}.csv
    data/derived/tdtr_candidate_hour_support_province_scores__{scenario}.csv
    qc/tdtr_candidate_hour_support_summary__{scenario}.json

Optional figures:
    outputs/figures/tdtr_support/tdtr_ranked_score_curve__{scenario}__{year}.png
    outputs/figures/tdtr_support/tdtr_ranked_score_curve__{scenario}__{year}.pdf

Interpretation
--------------
This support is an auditable candidate-hour design. It is not a claim that
these are the exact hours in which a Dutch operator would restrict access.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =============================================================================
# PATH SETUP
# =============================================================================

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
# CONFIG HELPERS
# =============================================================================

def cfg_get(name: str, default: Any) -> Any:
    if hasattr(cfg, name):
        value = getattr(cfg, name)
        if value is not None and value != "":
            return value
    return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return str(default)
    return str(raw).strip()


# =============================================================================
# GLOBAL CONFIG
# =============================================================================

DERIVED_DIR = Path(cfg_get("DERIVED_DIR", REPO_ROOT / "data" / "derived"))
QC_DIR = Path(cfg_get("QC_DIR", REPO_ROOT / "qc"))
OUTPUTS_DIR = Path(cfg_get("OUTPUTS_DIR", REPO_ROOT / "outputs"))
FIGURES_DIR = Path(cfg_get("FIGURES_DIR", OUTPUTS_DIR / "figures"))

YEARS: List[int] = sorted(
    dict.fromkeys(
        int(y) for y in cfg_get("ALL_YEARS", [2021, 2022, 2023, 2024, 2025])
    )
)

PROVINCES: List[str] = list(
    cfg_get(
        "NL_PROVINCES",
        [
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
        ],
    )
)

STATE_PANEL_PATH = Path(
    cfg_get(
        "AUDITABLE_STATE_PATH",
        cfg_get(
            "STATE_PATH",
            cfg_get(
                "AUDITABLE_STATE_PANEL_PATH",
                DERIVED_DIR / "auditable_state_province_hour.csv",
            ),
        ),
    )
)

OUT_SUPPORT_ALL = DERIVED_DIR / "tdtr_candidate_hour_support_all_scenarios.csv"
OUT_PROVINCE_SCORES_ALL = DERIVED_DIR / "tdtr_candidate_hour_support_province_scores_all_scenarios.csv"
OUT_SUMMARY_ALL = QC_DIR / "tdtr_candidate_hour_support_summary_all_scenarios.json"
OUT_OVERLAP = QC_DIR / "tdtr_candidate_hour_support_overlap.csv"

# Backward-compatible default outputs for the main scenario.
OUT_SUPPORT_MAIN = DERIVED_DIR / "tdtr_candidate_hour_support.csv"
OUT_PROVINCE_SCORES_MAIN = DERIVED_DIR / "tdtr_candidate_hour_support_province_scores.csv"
OUT_SUMMARY_MAIN = QC_DIR / "tdtr_candidate_hour_support_summary.json"

FIG_DIR = FIGURES_DIR / "tdtr_support"

# Main scenario name copied to backward-compatible files.
MAIN_SCENARIO = env_str("TDTR_MAIN_SCENARIO", "medium")

# Optional plotting.
MAKE_FIGURES = env_bool("TDTR_MAKE_FIGURES", True)

# Rounding convention for K = alpha * annual hours.
# allowed: floor, ceil, round
ROUNDING = env_str("TDTR_ROUNDING", "floor").lower()

# Default score mode.
# allowed:
#   max_directional_public_stress
#   plus_only
#   minus_only
#   ptdf_effectivity
#   relief_value_proxy
#   burden_proxy
#   custom_linear
SCORE_MODE = env_str("TDTR_SCORE_MODE", "max_directional_public_stress").lower()

# Hour aggregation from province-level scores.
# allowed:
#   max
#   mean
#   blend
#   quantile
HOUR_AGGREGATION = env_str("TDTR_HOUR_AGGREGATION", "blend").lower()
BLEND_MAX_WEIGHT = env_float("TDTR_BLEND_MAX_WEIGHT", 0.70)
AGG_QUANTILE = env_float("TDTR_AGG_QUANTILE", 0.90)

# Directional weights.
W_PLUS = env_float("TDTR_W_PLUS", 0.50)
W_MINUS = env_float("TDTR_W_MINUS", 0.50)

# Whether to z-score / minmax selected raw score columns before combining.
NORMALIZE_COMPONENTS_BY_PROVINCE_YEAR = env_bool("TDTR_NORMALIZE_COMPONENTS", False)

# Tiny deterministic tie penalty: earlier timestamps win by default if scores tie.
TIE_BREAK_EARLIER_FIRST = env_bool("TDTR_TIE_BREAK_EARLIER_FIRST", True)

# Scenario specification.
# You can override by setting TDTR_SCENARIOS_JSON to a JSON object, for example:
# {
#   "restrictive": {"alpha": 0.10, "score_mode": "max_directional_public_stress", "hour_aggregation": "blend"},
#   "medium":      {"alpha": 0.15, "score_mode": "max_directional_public_stress", "hour_aggregation": "blend"},
#   "wide":        {"alpha": 0.20, "score_mode": "max_directional_public_stress", "hour_aggregation": "blend"}
# }
DEFAULT_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "restrictive": {
        "alpha": 0.10,
        "score_mode": SCORE_MODE,
        "hour_aggregation": HOUR_AGGREGATION,
        "w_plus": W_PLUS,
        "w_minus": W_MINUS,
        "blend_max_weight": BLEND_MAX_WEIGHT,
        "agg_quantile": AGG_QUANTILE,
    },
    "medium": {
        "alpha": 0.15,
        "score_mode": SCORE_MODE,
        "hour_aggregation": HOUR_AGGREGATION,
        "w_plus": W_PLUS,
        "w_minus": W_MINUS,
        "blend_max_weight": BLEND_MAX_WEIGHT,
        "agg_quantile": AGG_QUANTILE,
    },
    "wide": {
        "alpha": 0.20,
        "score_mode": SCORE_MODE,
        "hour_aggregation": HOUR_AGGREGATION,
        "w_plus": W_PLUS,
        "w_minus": W_MINUS,
        "blend_max_weight": BLEND_MAX_WEIGHT,
        "agg_quantile": AGG_QUANTILE,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    alpha: float
    score_mode: str
    hour_aggregation: str
    w_plus: float
    w_minus: float
    blend_max_weight: float
    agg_quantile: float


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[SAVE] {path}")


def hours_in_year_utc(year: int) -> int:
    t0 = pd.Timestamp(f"{year}-01-01T00:00:00Z")
    t1 = pd.Timestamp(f"{year + 1}-01-01T00:00:00Z")
    return int((t1 - t0) / pd.Timedelta(hours=1))


def pick_column(
    columns: Iterable[str],
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> Optional[str]:
    cols = [str(c).strip() for c in columns]
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    if required:
        raise KeyError(
            f"Could not find any of columns {list(candidates)!r}. "
            f"Available columns: {cols!r}"
        )
    return None


def minmax01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    lo = float(x.min())
    hi = float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=s.index)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0)


def safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def parse_scenarios() -> List[ScenarioSpec]:
    raw = os.environ.get("TDTR_SCENARIOS_JSON")
    if raw is None or raw.strip() == "":
        d = DEFAULT_SCENARIOS
    else:
        try:
            d = json.loads(raw)
        except Exception as exc:
            raise ValueError(
                "TDTR_SCENARIOS_JSON must be a valid JSON object."
            ) from exc

    specs: List[ScenarioSpec] = []
    for name, spec in d.items():
        alpha = float(spec.get("alpha", 0.15))
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Scenario {name!r} has invalid alpha={alpha}. Must lie in (0,1).")

        specs.append(
            ScenarioSpec(
                name=str(name),
                alpha=alpha,
                score_mode=str(spec.get("score_mode", SCORE_MODE)).lower(),
                hour_aggregation=str(spec.get("hour_aggregation", HOUR_AGGREGATION)).lower(),
                w_plus=float(spec.get("w_plus", W_PLUS)),
                w_minus=float(spec.get("w_minus", W_MINUS)),
                blend_max_weight=float(spec.get("blend_max_weight", BLEND_MAX_WEIGHT)),
                agg_quantile=float(spec.get("agg_quantile", AGG_QUANTILE)),
            )
        )

    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate scenario names: {names}")

    return specs


def k_for_year(year: int, alpha: float) -> int:
    H = hours_in_year_utc(year)
    raw = alpha * H
    if ROUNDING == "floor":
        k = int(math.floor(raw))
    elif ROUNDING == "ceil":
        k = int(math.ceil(raw))
    elif ROUNDING == "round":
        k = int(round(raw))
    else:
        raise ValueError("TDTR_ROUNDING must be one of {'floor', 'ceil', 'round'}.")

    k = max(1, min(H, k))
    return k


# =============================================================================
# LOAD AND VALIDATE STATE PANEL
# =============================================================================

def load_state_panel() -> pd.DataFrame:
    if not STATE_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing auditable state panel:\n  {STATE_PANEL_PATH}\n"
            "Run the public stress / auditable state build before this script."
        )

    df = pd.read_csv(STATE_PANEL_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    province_col = pick_column(df.columns, ["province", "provincie"])
    df["province"] = df[province_col].astype(str).str.strip()

    ts_col = pick_column(
        df.columns,
        ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc", "timestamp"],
        required=False,
    )

    if ts_col is not None:
        df["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.floor("h")
    else:
        y_col = pick_column(df.columns, ["year"])
        h_col = pick_column(df.columns, ["hour_of_year", "hour"])
        y = pd.to_numeric(df[y_col], errors="coerce").astype("Int64")
        h = pd.to_numeric(df[h_col], errors="coerce").astype("Int64")
        df["timestamp_utc"] = (
            pd.to_datetime(y.astype(str) + "-01-01T00:00:00Z", utc=True)
            + pd.to_timedelta(h.astype(float) - 1.0, unit="h")
        )

    df = df.dropna(subset=["timestamp_utc"]).copy()
    df["year"] = df["timestamp_utc"].dt.year.astype(int)

    # Rebuild hour_of_year consistently from UTC timestamp.
    jan1 = pd.to_datetime(df["year"].astype(str) + "-01-01T00:00:00Z", utc=True)
    df["hour_of_year"] = ((df["timestamp_utc"] - jan1).dt.total_seconds() // 3600).astype(int) + 1

    # Calendar fields.
    df["month"] = df["timestamp_utc"].dt.month.astype(int)
    df["weekday_utc"] = df["timestamp_utc"].dt.weekday.astype(int)
    df["hour_of_day_utc"] = df["timestamp_utc"].dt.hour.astype(int)
    df["hour_of_week"] = df["weekday_utc"] * 24 + df["hour_of_day_utc"]

    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["weekday_utc"].isin([5, 6]).astype(int)
    else:
        df["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0).astype(int)

    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0
    else:
        df["is_holiday"] = pd.to_numeric(df["is_holiday"], errors="coerce").fillna(0).astype(int)

    df = df[df["year"].isin(YEARS)].copy()
    df = df[df["province"].isin(PROVINCES)].copy()

    if df.empty:
        raise ValueError(
            "State panel is empty after filtering by YEARS and PROVINCES. "
            f"YEARS={YEARS}, PROVINCES={PROVINCES}"
        )

    # Standardize frequently used stress columns.
    for c in [
        "rl_plus_norm",
        "rl_minus_norm",
        "s_plus_norm",
        "s_minus_norm",
        "price_da_norm",
        "structural_exposure",
        "ptdf_effectivity_norm",
        "relief_value_proxy_norm",
        "burden_proxy_norm",
        "B_proxy_baseline",
        "B_proxy",
        "relief_proxy_baseline",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    validate_state_panel(df)
    return df


def validate_state_panel(df: pd.DataFrame) -> None:
    for year in YEARS:
        H = hours_in_year_utc(year)
        ydf = df[df["year"] == year].copy()

        if ydf.empty:
            raise ValueError(f"Missing year {year} in state panel.")

        expected_rows = H * len(PROVINCES)
        if len(ydf) != expected_rows:
            raise ValueError(
                f"Year {year}: state panel has {len(ydf):,} rows, "
                f"expected {expected_rows:,} = {H} hours x {len(PROVINCES)} provinces."
            )

        for p in PROVINCES:
            g = ydf[ydf["province"] == p].sort_values("timestamp_utc")
            if len(g) != H:
                raise ValueError(
                    f"Year {year}, province {p}: has {len(g):,} rows, expected {H}."
                )

            expected_ts = pd.date_range(
                pd.Timestamp(f"{year}-01-01T00:00:00Z"),
                periods=H,
                freq="h",
            )
            got_ts = pd.DatetimeIndex(g["timestamp_utc"])
            if not got_ts.equals(expected_ts):
                sample = pd.DataFrame(
                    {
                        "expected": expected_ts[:5].astype(str),
                        "got": got_ts[:5].astype(str),
                    }
                )
                raise ValueError(
                    f"Year {year}, province {p}: timestamps are not a complete UTC hourly year.\n"
                    f"{sample.to_string(index=False)}"
                )

        dup = ydf.duplicated(["province", "timestamp_utc"], keep=False)
        if dup.any():
            sample = ydf.loc[dup, ["province", "timestamp_utc"]].head(20)
            raise ValueError(
                f"Year {year}: duplicate province-hour rows.\n{sample.to_string(index=False)}"
            )


# =============================================================================
# SCORE CONSTRUCTION
# =============================================================================

def normalize_components_if_needed(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    if not NORMALIZE_COMPONENTS_BY_PROVINCE_YEAR:
        return out

    for c in cols:
        if c in out.columns:
            out[c] = (
                out.groupby(["province", "year"], sort=False)[c]
                .transform(minmax01)
                .astype(float)
            )
    return out


def compute_province_scores(df: pd.DataFrame, spec: ScenarioSpec) -> pd.DataFrame:
    """
    Return province-hour scores for one scenario.
    """
    out = df.copy()

    # Make missing optional columns explicit.
    for c in [
        "rl_plus_norm",
        "rl_minus_norm",
        "s_plus_norm",
        "s_minus_norm",
        "price_da_norm",
        "structural_exposure",
        "ptdf_effectivity_norm",
        "relief_value_proxy_norm",
        "burden_proxy_norm",
        "B_proxy_baseline",
        "B_proxy",
        "relief_proxy_baseline",
    ]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)

    component_cols = [
        "rl_plus_norm",
        "rl_minus_norm",
        "s_plus_norm",
        "s_minus_norm",
        "price_da_norm",
        "structural_exposure",
        "ptdf_effectivity_norm",
        "relief_value_proxy_norm",
        "burden_proxy_norm",
        "B_proxy_baseline",
        "B_proxy",
        "relief_proxy_baseline",
    ]
    out = normalize_components_if_needed(out, component_cols)

    mode = spec.score_mode

    if mode == "max_directional_public_stress":
        plus = np.maximum(out["s_plus_norm"].to_numpy(float), out["rl_plus_norm"].to_numpy(float))
        minus = np.maximum(out["s_minus_norm"].to_numpy(float), out["rl_minus_norm"].to_numpy(float))
        score = spec.w_plus * plus + spec.w_minus * minus

    elif mode == "plus_only":
        score = np.maximum(out["s_plus_norm"].to_numpy(float), out["rl_plus_norm"].to_numpy(float))

    elif mode == "minus_only":
        score = np.maximum(out["s_minus_norm"].to_numpy(float), out["rl_minus_norm"].to_numpy(float))

    elif mode == "ptdf_effectivity":
        score = out["ptdf_effectivity_norm"].to_numpy(float)

    elif mode == "relief_value_proxy":
        if "relief_value_proxy_norm" in out.columns:
            score = out["relief_value_proxy_norm"].to_numpy(float)
        else:
            score = out["relief_proxy_baseline"].to_numpy(float)

    elif mode == "burden_proxy":
        if "burden_proxy_norm" in out.columns:
            score = out["burden_proxy_norm"].to_numpy(float)
        else:
            score = np.maximum(out["B_proxy_baseline"].to_numpy(float), out["B_proxy"].to_numpy(float))

    elif mode == "custom_linear":
        # Conservative default custom linear rule:
        # high/low directional stress + price + effectivity + structural exposure.
        score = (
            0.25 * out["s_plus_norm"].to_numpy(float)
            + 0.25 * out["s_minus_norm"].to_numpy(float)
            + 0.15 * out["price_da_norm"].to_numpy(float)
            + 0.20 * out["ptdf_effectivity_norm"].to_numpy(float)
            + 0.15 * out["structural_exposure"].to_numpy(float)
        )

    else:
        raise ValueError(
            f"Unknown score_mode={mode!r}. "
            "Allowed: max_directional_public_stress, plus_only, minus_only, "
            "ptdf_effectivity, relief_value_proxy, burden_proxy, custom_linear."
        )

    out["province_score_raw"] = np.asarray(score, dtype=float)
    out["province_score_raw"] = out["province_score_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Normalize final province score to [0,1] within province-year for comparability.
    out["province_score"] = (
        out.groupby(["province", "year"], sort=False)["province_score_raw"]
        .transform(minmax01)
        .astype(float)
    )

    keep = [
        "province",
        "timestamp_utc",
        "year",
        "hour_of_year",
        "month",
        "weekday_utc",
        "hour_of_day_utc",
        "hour_of_week",
        "is_weekend",
        "is_holiday",
        "rl_plus_norm",
        "rl_minus_norm",
        "s_plus_norm",
        "s_minus_norm",
        "price_da_norm",
        "structural_exposure",
        "ptdf_effectivity_norm",
        "relief_value_proxy_norm",
        "burden_proxy_norm",
        "B_proxy_baseline",
        "province_score_raw",
        "province_score",
    ]
    keep = [c for c in keep if c in out.columns]

    result = out[keep].copy()
    result["scenario"] = spec.name
    result["alpha"] = float(spec.alpha)
    result["score_mode"] = spec.score_mode
    result["hour_aggregation"] = spec.hour_aggregation
    return result


def aggregate_hour_scores(prov_scores: pd.DataFrame, spec: ScenarioSpec) -> pd.DataFrame:
    """
    Aggregate province-hour scores to hour-level support scores.
    """
    mode = spec.hour_aggregation

    if mode == "max":
        agg = (
            prov_scores.groupby(["scenario", "year", "timestamp_utc", "hour_of_year"], as_index=False)
            .agg(
                hour_score=("province_score", "max"),
                hour_score_mean=("province_score", "mean"),
                hour_score_max=("province_score", "max"),
                hour_score_q=("province_score", lambda s: float(np.quantile(s.to_numpy(float), spec.agg_quantile))),
                active_province=("province", lambda s: str(s.iloc[0])),
            )
        )

    elif mode == "mean":
        agg = (
            prov_scores.groupby(["scenario", "year", "timestamp_utc", "hour_of_year"], as_index=False)
            .agg(
                hour_score=("province_score", "mean"),
                hour_score_mean=("province_score", "mean"),
                hour_score_max=("province_score", "max"),
                hour_score_q=("province_score", lambda s: float(np.quantile(s.to_numpy(float), spec.agg_quantile))),
                active_province=("province", lambda s: str(s.iloc[0])),
            )
        )

    elif mode == "quantile":
        agg = (
            prov_scores.groupby(["scenario", "year", "timestamp_utc", "hour_of_year"], as_index=False)
            .agg(
                hour_score=("province_score", lambda s: float(np.quantile(s.to_numpy(float), spec.agg_quantile))),
                hour_score_mean=("province_score", "mean"),
                hour_score_max=("province_score", "max"),
                hour_score_q=("province_score", lambda s: float(np.quantile(s.to_numpy(float), spec.agg_quantile))),
                active_province=("province", lambda s: str(s.iloc[0])),
            )
        )

    elif mode == "blend":
        base = (
            prov_scores.groupby(["scenario", "year", "timestamp_utc", "hour_of_year"], as_index=False)
            .agg(
                hour_score_mean=("province_score", "mean"),
                hour_score_max=("province_score", "max"),
                hour_score_q=("province_score", lambda s: float(np.quantile(s.to_numpy(float), spec.agg_quantile))),
            )
        )
        w = float(np.clip(spec.blend_max_weight, 0.0, 1.0))
        base["hour_score"] = w * base["hour_score_max"] + (1.0 - w) * base["hour_score_mean"]

        # Identify active province as the max-score province in that hour.
        idx = (
            prov_scores.sort_values(
                ["scenario", "year", "timestamp_utc", "province_score", "province"],
                ascending=[True, True, True, False, True],
            )
            .groupby(["scenario", "year", "timestamp_utc"], as_index=False)
            .first()[["scenario", "year", "timestamp_utc", "province"]]
            .rename(columns={"province": "active_province"})
        )
        agg = base.merge(idx, on=["scenario", "year", "timestamp_utc"], how="left", validate="one_to_one")

    else:
        raise ValueError("hour_aggregation must be one of {'max', 'mean', 'blend', 'quantile'}.")

    # Add calendar columns from province table.
    cal_cols = [
        "scenario",
        "year",
        "timestamp_utc",
        "hour_of_year",
        "month",
        "weekday_utc",
        "hour_of_day_utc",
        "hour_of_week",
        "is_weekend",
        "is_holiday",
    ]
    cal = prov_scores[cal_cols].drop_duplicates(["scenario", "year", "timestamp_utc"]).copy()
    agg = agg.merge(cal, on=["scenario", "year", "timestamp_utc", "hour_of_year"], how="left", validate="one_to_one")

    agg["hour_score"] = pd.to_numeric(agg["hour_score"], errors="coerce").fillna(0.0).astype(float)
    return agg.sort_values(["scenario", "year", "timestamp_utc"]).reset_index(drop=True)


# =============================================================================
# SUPPORT SELECTION
# =============================================================================

def select_support(hour_scores: pd.DataFrame, spec: ScenarioSpec) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Select top K hours per year for one scenario.
    """
    pieces = []
    summaries = []

    for year, g in hour_scores.groupby("year", sort=True):
        year = int(year)
        H = hours_in_year_utc(year)
        K = k_for_year(year, spec.alpha)

        if len(g) != H:
            raise ValueError(
                f"Scenario {spec.name}, year {year}: hour score rows={len(g)}, expected {H}."
            )

        gg = g.copy()
        gg["rank_tie"] = np.arange(len(gg), dtype=int)

        if TIE_BREAK_EARLIER_FIRST:
            gg = gg.sort_values(
                ["hour_score", "timestamp_utc"],
                ascending=[False, True],
                kind="mergesort",
            ).reset_index(drop=True)
        else:
            gg = gg.sort_values(
                ["hour_score", "timestamp_utc"],
                ascending=[False, False],
                kind="mergesort",
            ).reset_index(drop=True)

        gg["tdtr_rank"] = np.arange(1, len(gg) + 1, dtype=int)
        gg["is_tdtr_candidate_hour"] = (gg["tdtr_rank"] <= K).astype(int)
        gg["selected"] = gg["is_tdtr_candidate_hour"]
        gg["tdtr_alpha"] = float(spec.alpha)
        gg["tdtr_K"] = int(K)
        gg["tdtr_rounding"] = ROUNDING

        selected_scores = gg.loc[gg["is_tdtr_candidate_hour"].eq(1), "hour_score"]
        unselected_scores = gg.loc[gg["is_tdtr_candidate_hour"].eq(0), "hour_score"]

        min_selected = float(selected_scores.min()) if len(selected_scores) else np.nan
        max_unselected = float(unselected_scores.max()) if len(unselected_scores) else np.nan

        threshold_score = min_selected
        gg["tdtr_threshold_score"] = threshold_score
        gg["score_above_threshold"] = (gg["hour_score"] >= threshold_score).astype(int)

        selected_month_counts = (
            gg.loc[gg["is_tdtr_candidate_hour"].eq(1)]
            .groupby("month")["is_tdtr_candidate_hour"]
            .sum()
            .to_dict()
        )

        selected_hour_of_day_counts = (
            gg.loc[gg["is_tdtr_candidate_hour"].eq(1)]
            .groupby("hour_of_day_utc")["is_tdtr_candidate_hour"]
            .sum()
            .to_dict()
        )

        summaries.append(
            {
                "scenario": spec.name,
                "year": year,
                "annual_hours": H,
                "alpha": float(spec.alpha),
                "rounding": ROUNDING,
                "K": int(K),
                "selected_hours": int(gg["is_tdtr_candidate_hour"].sum()),
                "selected_share": float(gg["is_tdtr_candidate_hour"].mean()),
                "min_selected_score": min_selected,
                "max_unselected_score": max_unselected,
                "threshold_score": float(threshold_score),
                "mean_selected_score": float(selected_scores.mean()) if len(selected_scores) else np.nan,
                "mean_unselected_score": float(unselected_scores.mean()) if len(unselected_scores) else np.nan,
                "score_gap_selected_minus_unselected_mean": (
                    float(selected_scores.mean() - unselected_scores.mean())
                    if len(selected_scores) and len(unselected_scores)
                    else np.nan
                ),
                "selected_month_counts": {str(k): int(v) for k, v in selected_month_counts.items()},
                "selected_hour_of_day_utc_counts": {
                    str(k): int(v) for k, v in selected_hour_of_day_counts.items()
                },
            }
        )

        # Restore chronological order for output.
        gg = gg.sort_values("timestamp_utc").reset_index(drop=True)
        pieces.append(gg)

    out = pd.concat(pieces, ignore_index=True)
    return out, summaries


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def build_province_score_diagnostics(
    prov_scores: pd.DataFrame,
    support: pd.DataFrame,
) -> pd.DataFrame:
    key = ["scenario", "year", "timestamp_utc"]
    sup = support[key + ["is_tdtr_candidate_hour", "hour_score", "tdtr_rank"]].copy()

    df = prov_scores.merge(sup, on=key, how="left", validate="many_to_one")
    df["is_tdtr_candidate_hour"] = df["is_tdtr_candidate_hour"].fillna(0).astype(int)

    diag = (
        df.groupby(["scenario", "year", "province"], as_index=False)
        .agg(
            mean_score_all=("province_score", "mean"),
            mean_score_selected=("province_score", lambda s: float(np.nanmean(s[df.loc[s.index, "is_tdtr_candidate_hour"].eq(1)])) if df.loc[s.index, "is_tdtr_candidate_hour"].eq(1).any() else np.nan),
            mean_score_unselected=("province_score", lambda s: float(np.nanmean(s[df.loc[s.index, "is_tdtr_candidate_hour"].eq(0)])) if df.loc[s.index, "is_tdtr_candidate_hour"].eq(0).any() else np.nan),
            max_score_all=("province_score", "max"),
            selected_hours=("is_tdtr_candidate_hour", "sum"),
            total_hours=("is_tdtr_candidate_hour", "size"),
            mean_hour_score_selected=("hour_score", lambda s: float(np.nanmean(s[df.loc[s.index, "is_tdtr_candidate_hour"].eq(1)])) if df.loc[s.index, "is_tdtr_candidate_hour"].eq(1).any() else np.nan),
        )
    )

    diag["selected_share"] = diag["selected_hours"] / diag["total_hours"].replace(0, np.nan)
    diag["selected_minus_unselected_score"] = diag["mean_score_selected"] - diag["mean_score_unselected"]
    return diag.sort_values(["scenario", "year", "province"]).reset_index(drop=True)


def build_support_overlap(support_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenarios = sorted(support_all["scenario"].unique().tolist())

    for year in sorted(support_all["year"].unique().tolist()):
        sub = support_all[support_all["year"] == year].copy()

        sets: Dict[str, set] = {}
        for sc in scenarios:
            s = sub[
                (sub["scenario"] == sc)
                & (sub["is_tdtr_candidate_hour"] == 1)
            ]["timestamp_utc"]
            sets[sc] = set(pd.to_datetime(s, utc=True).astype(str).tolist())

        for i, a in enumerate(scenarios):
            for b in scenarios[i + 1:]:
                A = sets[a]
                B = sets[b]
                inter = len(A & B)
                union = len(A | B)
                rows.append(
                    {
                        "year": int(year),
                        "scenario_a": a,
                        "scenario_b": b,
                        "selected_a": int(len(A)),
                        "selected_b": int(len(B)),
                        "intersection": int(inter),
                        "union": int(union),
                        "jaccard_overlap": float(inter / union) if union > 0 else np.nan,
                        "share_of_a_in_b": float(inter / len(A)) if len(A) > 0 else np.nan,
                        "share_of_b_in_a": float(inter / len(B)) if len(B) > 0 else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def plot_ranked_score_curves(support_all: pd.DataFrame) -> None:
    if not MAKE_FIGURES:
        return
    if plt is None:
        print("[WARN] matplotlib not available; skipping TDTR support figures.")
        return

    ensure_dir(FIG_DIR)

    for (scenario, year), g in support_all.groupby(["scenario", "year"], sort=True):
        gg = g.sort_values("tdtr_rank").copy()

        x = gg["tdtr_rank"].to_numpy(dtype=int)
        y = gg["hour_score"].to_numpy(dtype=float)
        k = int(gg["tdtr_K"].iloc[0])
        threshold = float(gg["tdtr_threshold_score"].iloc[0])
        alpha = float(gg["tdtr_alpha"].iloc[0])

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(x, y, linewidth=1.2)
        ax.axvline(k, linestyle="--", linewidth=1.0)
        ax.axhline(threshold, linestyle=":", linewidth=1.0)
        ax.set_title(f"TDTR candidate-hour score curve: {scenario}, {year}")
        ax.set_xlabel("Annual hour rank by public stress score")
        ax.set_ylabel("Hour score")
        ax.grid(True, linewidth=0.3, alpha=0.4)

        ax.text(
            0.98,
            0.95,
            f"alpha={alpha:.2f}\nK={k:,}\nthreshold={threshold:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
        )

        fig.tight_layout()

        png = FIG_DIR / f"tdtr_ranked_score_curve__{scenario}__{year}.png"
        pdf = FIG_DIR / f"tdtr_ranked_score_curve__{scenario}__{year}.pdf"
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        plt.close(fig)

        print(f"[SAVE] {png}")
        print(f"[SAVE] {pdf}")


# =============================================================================
# OUTPUT WRITING
# =============================================================================

def write_outputs_for_scenario(
    spec: ScenarioSpec,
    support: pd.DataFrame,
    province_diag: pd.DataFrame,
    summary: dict,
) -> None:
    support_path = DERIVED_DIR / f"tdtr_candidate_hour_support__{spec.name}.csv"
    province_path = DERIVED_DIR / f"tdtr_candidate_hour_support_province_scores__{spec.name}.csv"
    summary_path = QC_DIR / f"tdtr_candidate_hour_support_summary__{spec.name}.json"

    ensure_dir(support_path.parent)
    ensure_dir(summary_path.parent)

    support.to_csv(support_path, index=False)
    province_diag.to_csv(province_path, index=False)
    save_json(summary, summary_path)

    print(f"[SAVE] {support_path}")
    print(f"[SAVE] {province_path}")


def write_backward_compatible_main(
    support_all: pd.DataFrame,
    province_scores_or_diag_all: pd.DataFrame,
    summary_all: dict,
) -> None:
    main_support = support_all[support_all["scenario"] == MAIN_SCENARIO].copy()
    main_province = province_scores_or_diag_all[
        province_scores_or_diag_all["scenario"] == MAIN_SCENARIO
    ].copy()

    if main_support.empty:
        raise ValueError(
            f"MAIN_SCENARIO={MAIN_SCENARIO!r} not found in support output. "
            f"Available scenarios: {sorted(support_all['scenario'].unique().tolist())}"
        )

    main_summary = {
        "main_scenario": MAIN_SCENARIO,
        "source": "extracted from tdtr_candidate_hour_support_summary_all_scenarios.json",
        "scenario_summary": summary_all["scenario_summaries"].get(MAIN_SCENARIO, {}),
        "year_summaries": [
            row
            for row in summary_all["year_summaries"]
            if row["scenario"] == MAIN_SCENARIO
        ],
        "notes": summary_all.get("notes", []),
    }

    main_support.to_csv(OUT_SUPPORT_MAIN, index=False)
    main_province.to_csv(OUT_PROVINCE_SCORES_MAIN, index=False)
    save_json(main_summary, OUT_SUMMARY_MAIN)

    print(f"[SAVE] {OUT_SUPPORT_MAIN}")
    print(f"[SAVE] {OUT_PROVINCE_SCORES_MAIN}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dir(DERIVED_DIR)
    ensure_dir(QC_DIR)
    ensure_dir(FIG_DIR)

    scenarios = parse_scenarios()

    print("=" * 78)
    print("G: BUILD TDTR CANDIDATE-HOUR SUPPORT")
    print("=" * 78)
    print(f"Repo root       : {REPO_ROOT}")
    print(f"State panel     : {STATE_PANEL_PATH}")
    print(f"Years           : {YEARS}")
    print(f"Provinces       : {PROVINCES}")
    print(f"Main scenario   : {MAIN_SCENARIO}")
    print(f"Rounding        : {ROUNDING}")
    print(f"Make figures    : {MAKE_FIGURES}")
    print("Scenarios:")
    for s in scenarios:
        print(
            f"  - {s.name:12s} alpha={s.alpha:.3f} "
            f"score={s.score_mode} aggregation={s.hour_aggregation}"
        )
    print("=" * 78)

    print(f"[INPUT] using auditable_state: {STATE_PANEL_PATH}")
    state = load_state_panel()

    all_support_parts: List[pd.DataFrame] = []
    all_province_scores_parts: List[pd.DataFrame] = []
    all_province_diag_parts: List[pd.DataFrame] = []
    all_year_summaries: List[dict] = []
    scenario_summaries: Dict[str, dict] = {}

    for spec in scenarios:
        print("\n" + "-" * 78)
        print(f"[SCENARIO] {spec.name}")
        print("-" * 78)

        province_scores = compute_province_scores(state, spec)
        hour_scores = aggregate_hour_scores(province_scores, spec)
        support, year_summaries = select_support(hour_scores, spec)
        province_diag = build_province_score_diagnostics(province_scores, support)

        all_support_parts.append(support)
        all_province_scores_parts.append(province_scores)
        all_province_diag_parts.append(province_diag)
        all_year_summaries.extend(year_summaries)

        scenario_summary = {
            "scenario": spec.name,
            "spec": asdict(spec),
            "years": YEARS,
            "rounding": ROUNDING,
            "selected_hours_by_year": {
                str(r["year"]): int(r["selected_hours"]) for r in year_summaries
            },
            "selected_share_by_year": {
                str(r["year"]): float(r["selected_share"]) for r in year_summaries
            },
            "threshold_score_by_year": {
                str(r["year"]): float(r["threshold_score"]) for r in year_summaries
            },
        }
        scenario_summaries[spec.name] = scenario_summary

        write_outputs_for_scenario(
            spec=spec,
            support=support,
            province_diag=province_diag,
            summary={
                "scenario_summary": scenario_summary,
                "year_summaries": year_summaries,
            },
        )

        print("[SUMMARY]")
        for r in year_summaries:
            print(
                f"year={r['year']} | selected={r['selected_hours']:,}/{r['annual_hours']:,} "
                f"({100.0 * r['selected_share']:.2f}%) | "
                f"min selected score={r['min_selected_score']:.4f} | "
                f"max unselected score={r['max_unselected_score']:.4f}"
            )

    support_all = pd.concat(all_support_parts, ignore_index=True)
    province_scores_all = pd.concat(all_province_scores_parts, ignore_index=True)
    province_diag_all = pd.concat(all_province_diag_parts, ignore_index=True)

    support_overlap = build_support_overlap(support_all)

    # Write combined outputs.
    support_all.to_csv(OUT_SUPPORT_ALL, index=False)
    province_scores_all.to_csv(OUT_PROVINCE_SCORES_ALL, index=False)
    support_overlap.to_csv(OUT_OVERLAP, index=False)

    print(f"\n[SAVE] {OUT_SUPPORT_ALL}")
    print(f"[SAVE] {OUT_PROVINCE_SCORES_ALL}")
    print(f"[SAVE] {OUT_OVERLAP}")

    summary_all = {
        "script": Path(__file__).name,
        "repo_root": str(REPO_ROOT),
        "state_panel_path": str(STATE_PANEL_PATH),
        "years": YEARS,
        "provinces": PROVINCES,
        "main_scenario": MAIN_SCENARIO,
        "rounding": ROUNDING,
        "normalize_components_by_province_year": NORMALIZE_COMPONENTS_BY_PROVINCE_YEAR,
        "tie_break_earlier_first": TIE_BREAK_EARLIER_FIRST,
        "scenario_summaries": scenario_summaries,
        "year_summaries": all_year_summaries,
        "combined_outputs": {
            "support_all_scenarios": str(OUT_SUPPORT_ALL),
            "province_scores_all_scenarios": str(OUT_PROVINCE_SCORES_ALL),
            "overlap_csv": str(OUT_OVERLAP),
        },
        "backward_compatible_main_outputs": {
            "support": str(OUT_SUPPORT_MAIN),
            "province_scores": str(OUT_PROVINCE_SCORES_MAIN),
            "summary": str(OUT_SUMMARY_MAIN),
        },
        "notes": [
            "The TDTR support is an annual candidate-hour eligibility mask, not a dispatch solution.",
            "The selected support is common to M1, M2, and M3 within each scenario.",
            "Downstream scripts should mask both R_{k,t} and Pbar_{i,t} using is_tdtr_candidate_hour.",
            "Scenario-specific support files allow sensitivity over restrictive, medium, and wide TDTR windows.",
            "The backward-compatible tdtr_candidate_hour_support.csv contains only the configured main scenario.",
        ],
    }
    save_json(summary_all, OUT_SUMMARY_ALL)

    # Backward-compatible output uses province diagnostics, not the full province-score panel,
    # because this file is smaller and more useful for quick inspection.
    write_backward_compatible_main(
        support_all=support_all,
        province_scores_or_diag_all=province_diag_all,
        summary_all=summary_all,
    )

    # Also save province diagnostics combined.
    province_diag_path = DERIVED_DIR / "tdtr_candidate_hour_support_province_diagnostics_all_scenarios.csv"
    province_diag_all.to_csv(province_diag_path, index=False)
    print(f"[SAVE] {province_diag_path}")

    plot_ranked_score_curves(support_all)

    print("\n[FINAL SUMMARY]")
    for r in all_year_summaries:
        print(
            f"scenario={r['scenario']:12s} | year={r['year']} | "
            f"selected={r['selected_hours']:,}/{r['annual_hours']:,} "
            f"({100.0 * r['selected_share']:.2f}%) | "
            f"threshold={r['threshold_score']:.4f}"
        )

    print("\n[DONE]")


if __name__ == "__main__":
    main()