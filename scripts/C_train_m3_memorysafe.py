"""
C_train_m3.py
=============

Train the learned auditable weighted-depth rule M3 for the public-only
implementation used in the debit-rule benchmark.

Implemented variants
--------------------
- M3_plain : plain proxy-grounded fit
- M3_OI    : one-shot optimization-informed fit initialized at q == 1
- M3_LO    : alternating learn-optimize fit initialized at q == 1

Implemented public feature ladders
----------------------------------
- F0 : calendar + province
- F1 : F0 + public market/system state
- F2 : F1 + public local stress features
- F3 : F2 + structural exposure and interactions

Important interpretation
------------------------
F0--F3 are public-feature ladders inside the deployable public-only M3 rule.
They are not the paper's modeled shared-information levels unless those columns
actually encode coarsened auditable TSO/DSO/developer disclosure signals.

Training discipline
-------------------
- proxy target: cfg.PRIMARY_BURDEN, usually B_proxy_baseline
- proxy normalization: training years only
- rule normalization: training years only, optionally followed by annual bounded
  mean-one projection
- optimization-informed weights: alpha[n,t] = sum_{i in I_n} mu_i x[i,t] / P_i;
  in the current one-representative-per-province data this is mu[n] x[n,t]/P[n]
- regularized QP training operator is used by default when tau_x or tau_s > 0;
  set M3_TAU_X = M3_TAU_S = 0 in pipeline_config.py to use the unregularized LP
"""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipeline_config as cfg  # type: ignore
from B_run_annual_lp import solve_weighted_depth_lp, solve_weighted_depth_qp  # type: ignore


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class FitConfig:
    ridge_alpha: float = float(getattr(cfg, "M3_RIDGE_ALPHA", 1.0))
    max_iter: int = int(getattr(cfg, "M3_MAX_ITER", 6))
    tol_rel: float = float(getattr(cfg, "M3_TOL_REL", 1e-4))
    damping: float = float(getattr(cfg, "M3_DAMPING", 0.7))
    eps_target: float = float(getattr(cfg, "M3_EPS_TARGET", 1e-8))

    q_min: float = float(getattr(cfg, "M3_Q_MIN", 0.25))
    q_max: float = float(getattr(cfg, "M3_Q_MAX", 4.0))
    use_annual_projection: bool = bool(getattr(cfg, "M3_USE_ANNUAL_PROJECTION", True))
    projection_tol: float = float(getattr(cfg, "M3_PROJECTION_TOL", 1e-10))

    tau_x: float = float(getattr(cfg, "M3_TAU_X", 1e-6))
    tau_s: float = float(getattr(cfg, "M3_TAU_S", 1e-6))
    train_time_limit: float = float(getattr(cfg, "M3_TRAIN_TIME_LIMIT", 120.0))

    alpha_floor_fraction: float = float(getattr(cfg, "M3_ALPHA_FLOOR_FRACTION", 1e-4))
    alpha_power: float = float(getattr(cfg, "M3_ALPHA_POWER", 1.0))
    min_positive_alpha_share: float = float(getattr(cfg, "M3_MIN_POSITIVE_ALPHA_SHARE", 1e-8))
    alpha_abs_tol: float = float(getattr(cfg, "M3_ALPHA_ABS_TOL", 1e-8))
    skip_lo_if_no_scarcity: bool = bool(getattr(cfg, "M3_SKIP_LO_IF_NO_SCARCITY", True))
    predict_chunk_size: int = int(getattr(cfg, "M3_PREDICT_CHUNK_SIZE", 50000))
    write_partial_outputs: bool = bool(getattr(cfg, "M3_WRITE_PARTIAL_OUTPUTS", True))


FIT_CONFIG = FitConfig()

if not (0.0 < FIT_CONFIG.q_min < 1.0 < FIT_CONFIG.q_max):
    raise ValueError("Require 0 < M3_Q_MIN < 1 < M3_Q_MAX.")
if not (0.0 < FIT_CONFIG.damping <= 1.0):
    raise ValueError("Require 0 < M3_DAMPING <= 1.")
if FIT_CONFIG.predict_chunk_size <= 0:
    raise ValueError("M3_PREDICT_CHUNK_SIZE must be positive.")

_UNIFORM_ALPHA_CACHE: Tuple[pd.DataFrame, float, dict] | None = None


# =============================================================================
# LOADERS
# =============================================================================

def _load_state() -> pd.DataFrame:
    state = pd.read_csv(cfg.STATE_PATH, parse_dates=["timestamp_utc"])
    state["timestamp_utc"] = pd.to_datetime(state["timestamp_utc"], utc=True)
    state["province"] = state["province"].astype(str).str.strip()
    if "year" not in state.columns:
        state["year"] = state["timestamp_utc"].dt.year.astype(int)
    return state[state["province"].isin(cfg.NL_PROVINCES)].copy()


def _load_lp_meta() -> dict:
    with (cfg.LP_DIR / "lp_meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_lp_year(year: int) -> pd.DataFrame:
    path = cfg.LP_DIR / f"lp_params_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing LP parameter file: {path}")
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["province"] = df["province"].astype(str).str.strip()
    return df[df["province"].isin(cfg.NL_PROVINCES)].copy()


# =============================================================================
# TARGET AND FEATURES
# =============================================================================

def _province_beta_map(train_df: pd.DataFrame, target_col: str) -> Dict[str, float]:
    yr_means = (
        train_df.groupby(["province", "year"], observed=True)[target_col]
        .mean()
        .reset_index()
        .rename(columns={target_col: "yr_mean"})
    )
    beta = yr_means.groupby("province", observed=True)["yr_mean"].mean().clip(lower=1e-9)
    return {str(k): float(v) for k, v in beta.items()}


def _build_target_column(state: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in state.columns:
        raise KeyError(f"Target column {target_col!r} not found in state panel.")
    out = state.copy()
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce").fillna(0.0)
    beta_map = _province_beta_map(out[out["year"].isin(cfg.TRAIN_YEARS)].copy(), target_col)
    out["beta_target"] = out["province"].map(beta_map).astype(float)
    out["b_target"] = out[target_col] / out["beta_target"].clip(lower=1e-9)
    return out


def _require_columns(df: pd.DataFrame, cols: List[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {context}: {missing}")


def _feature_frame(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set not in {"F0", "F1", "F2", "F3"}:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    _require_columns(df, ["hour_of_week", "month", "is_weekend", "is_holiday", "province"], f"features {feature_set}")
    if feature_set in {"F1", "F2", "F3"}:
        _require_columns(df, ["price_da_norm"], f"features {feature_set}")
    if feature_set in {"F2", "F3"}:
        _require_columns(df, ["rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm"], f"features {feature_set}")
    if feature_set == "F3":
        _require_columns(df, ["structural_exposure"], f"features {feature_set}")

    out = pd.DataFrame(index=df.index)

    # Categorical calendar and deployment-unit intercepts.
    out["hour_of_week"] = df["hour_of_week"].astype("Int64").astype(str)
    out["month"] = df["month"].astype("Int64").astype(str)
    out["is_weekend"] = df["is_weekend"].astype(int).astype(str)
    out["is_holiday"] = df["is_holiday"].astype(int).astype(str)

    if feature_set in {"F1", "F2", "F3"}:
        out["price_da_norm"] = pd.to_numeric(df["price_da_norm"], errors="coerce").fillna(0.0)

    if feature_set in {"F2", "F3"}:
        for col in ["rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm"]:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if feature_set == "F3":
        price = pd.to_numeric(df["price_da_norm"], errors="coerce").fillna(0.0)
        rl_plus = pd.to_numeric(df["rl_plus_norm"], errors="coerce").fillna(0.0)
        s_plus = pd.to_numeric(df["s_plus_norm"], errors="coerce").fillna(0.0)
        out["structural_exposure"] = pd.to_numeric(df["structural_exposure"], errors="coerce").fillna(0.0)
        out["price_x_rl_plus"] = price * rl_plus
        out["price_x_s_plus"] = price * s_plus

    out["province"] = df["province"].astype(str)
    out = pd.get_dummies(
        out,
        columns=["hour_of_week", "month", "is_weekend", "is_holiday", "province"],
        drop_first=False,
    )
    return out.astype(np.float32)


def _align_columns(train_X: pd.DataFrame, full_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Kept for compatibility. Full-panel prediction uses chunked alignment
    # to avoid allocating a dense 500k x 200 matrix at once.
    cols = list(train_X.columns)
    return train_X[cols], full_X.reindex(columns=cols, fill_value=0.0)


def _align_feature_chunk(chunk_X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Align one feature chunk to the training columns."""
    return chunk_X.reindex(columns=cols, fill_value=0.0).astype(np.float32)


# =============================================================================
# Q NORMALIZATION / PROJECTION
# =============================================================================

def _project_mean_one_box(v: np.ndarray, q_min: float, q_max: float, tol: float) -> np.ndarray:
    """Euclidean projection onto {q_min <= u <= q_max, mean(u)=1}."""
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("Projection input must be 1D.")
    if len(v) == 0:
        return v.copy()
    if not np.all(np.isfinite(v)):
        raise ValueError("Projection input contains non-finite values.")

    lam_l = float(np.min(v) - q_max)
    lam_u = float(np.max(v) - q_min)
    for _ in range(200):
        lam = 0.5 * (lam_l + lam_u)
        u = np.clip(v - lam, q_min, q_max)
        if abs(float(np.mean(u)) - 1.0) <= tol:
            return u
        if float(np.mean(u)) > 1.0:
            lam_l = lam
        else:
            lam_u = lam
    return np.clip(v - 0.5 * (lam_l + lam_u), q_min, q_max)


def _apply_q_bounds_and_optional_projection(out: pd.DataFrame) -> pd.DataFrame:
    """
    Apply contract-design bounds and, by default, annual mean-one projection.

    Projection uses the full annual feature path for each province/year. This is
    an annual-normalized offline specification. If strict online deployment is
    required, set M3_USE_ANNUAL_PROJECTION=False and report realized means.
    """
    out = out.copy()
    out["debit_weight_pre_project"] = pd.to_numeric(out["debit_weight"], errors="coerce").fillna(1.0)
    out["debit_weight"] = out["debit_weight_pre_project"].clip(FIT_CONFIG.q_min, FIT_CONFIG.q_max)

    if FIT_CONFIG.use_annual_projection:
        parts: List[pd.DataFrame] = []
        for (_, _), g in out.groupby(["province", "year"], observed=True, sort=False):
            g = g.sort_values("timestamp_utc").copy()
            projected = _project_mean_one_box(
                g["debit_weight_pre_project"].to_numpy(dtype=float),
                FIT_CONFIG.q_min,
                FIT_CONFIG.q_max,
                FIT_CONFIG.projection_tol,
            )
            g["debit_weight"] = projected
            parts.append(g)
        out = pd.concat(parts, ignore_index=True)

    out["q_realized_annual_mean"] = out.groupby(["province", "year"], observed=True)["debit_weight"].transform("mean")
    out["q_realized_annual_min"] = out.groupby(["province", "year"], observed=True)["debit_weight"].transform("min")
    out["q_realized_annual_max"] = out.groupby(["province", "year"], observed=True)["debit_weight"].transform("max")
    out["q_projection_used"] = int(FIT_CONFIG.use_annual_projection)
    out["q_min_contract"] = FIT_CONFIG.q_min
    out["q_max_contract"] = FIT_CONFIG.q_max
    return out


# =============================================================================
# RIDGE MODEL
# =============================================================================

def _normalize_sample_weight(sample_weight: np.ndarray) -> Tuple[np.ndarray, dict]:
    w_raw = np.asarray(sample_weight, dtype=float)
    if w_raw.ndim != 1:
        raise ValueError("sample_weight must be 1D.")
    w_raw = np.where(np.isfinite(w_raw), w_raw, 0.0)
    w_raw = np.maximum(w_raw, 0.0)

    positive_share = float(np.mean(w_raw > 0.0)) if len(w_raw) else 0.0
    positive = w_raw[w_raw > 0.0]

    if len(w_raw) == 0:
        raise ValueError("Empty sample_weight.")

    raw_mean = float(np.mean(w_raw))
    raw_max = float(np.max(w_raw)) if len(w_raw) else 0.0

    if (
        positive.size == 0
        or positive_share <= FIT_CONFIG.min_positive_alpha_share
        or raw_max <= FIT_CONFIG.alpha_abs_tol
    ):
        w = np.ones_like(w_raw)
        info = {
            "alpha_fallback_uniform": 1.0,
            "alpha_effective": 0.0,
            "alpha_raw_mean": raw_mean,
            "alpha_raw_max": raw_max,
            "alpha_positive_share": positive_share,
            "sample_weight_mean": 1.0,
            "sample_weight_max": 1.0,
            "alpha_mean": raw_mean,  # backward-compatible log column
        }
        return w, info

    w = w_raw.copy()
    if FIT_CONFIG.alpha_power != 1.0:
        w = np.power(w, FIT_CONFIG.alpha_power)

    floor = FIT_CONFIG.alpha_floor_fraction * float(np.mean(w[w > 0.0]))
    w = w + floor
    w = w / max(float(np.mean(w)), 1e-15)

    info = {
        "alpha_fallback_uniform": 0.0,
        "alpha_effective": 1.0,
        "alpha_raw_mean": raw_mean,
        "alpha_raw_max": raw_max,
        "alpha_positive_share": positive_share,
        "sample_weight_mean": float(np.mean(w)),
        "sample_weight_max": float(np.max(w)),
        "alpha_mean": raw_mean,  # backward-compatible log column
    }
    return w, info


def _fit_ridge_with_info(X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> Tuple[Ridge, dict]:
    w, info = _normalize_sample_weight(sample_weight)
    model = Ridge(alpha=FIT_CONFIG.ridge_alpha, fit_intercept=True)
    model.fit(X.to_numpy(dtype=np.float32), y, sample_weight=w)
    # Store exact training columns so full-panel prediction can be chunked.
    model._feature_columns = list(X.columns)  # type: ignore[attr-defined]
    return model, info


def _predict_raw_score(model: Ridge, X: pd.DataFrame) -> np.ndarray:
    pred = model.predict(X.to_numpy(dtype=np.float32))
    pred = np.clip(pred, -30.0, 30.0)
    return np.exp(pred)


def _damped_model(previous: Ridge, candidate: Ridge, damping: float) -> Ridge:
    out = copy.deepcopy(candidate)
    out.coef_ = (1.0 - damping) * previous.coef_ + damping * candidate.coef_
    out.intercept_ = (1.0 - damping) * float(previous.intercept_) + damping * float(candidate.intercept_)
    if hasattr(previous, "_feature_columns"):
        out._feature_columns = list(previous._feature_columns)  # type: ignore[attr-defined]
    return out


def _relative_model_change(previous: Ridge, current: Ridge) -> float:
    prev = np.r_[np.asarray(previous.coef_, dtype=float), float(previous.intercept_)]
    curr = np.r_[np.asarray(current.coef_, dtype=float), float(current.intercept_)]
    return float(np.linalg.norm(curr - prev) / max(np.linalg.norm(prev), 1e-9))


# =============================================================================
# WEIGHT DATAFRAME CONSTRUCTION
# =============================================================================

def _kappa_map(df: pd.DataFrame, raw_col: str = "raw_score") -> Dict[str, float]:
    yr_means = (
        df[df["year"].isin(cfg.TRAIN_YEARS)]
        .groupby(["province", "year"], observed=True)[raw_col]
        .mean()
        .reset_index()
        .rename(columns={raw_col: "yr_mean"})
    )
    kappas = yr_means.groupby("province", observed=True)["yr_mean"].mean().clip(lower=1e-9)
    return {str(k): float(v) for k, v in kappas.items()}


def _build_weight_df_uniform(state: pd.DataFrame, feature_set: str, variant: str) -> pd.DataFrame:
    out = state[["province", "timestamp_utc", "year"]].copy()
    out["raw_score"] = 1.0
    out["kappa"] = 1.0
    out["debit_weight"] = 1.0
    out["feature_set"] = feature_set
    out["variant"] = variant
    out = _apply_q_bounds_and_optional_projection(out)
    return out


def _build_weight_df_from_model(state: pd.DataFrame, model: Ridge, feature_set: str, variant: str) -> pd.DataFrame:
    """
    Build debit weights from a fitted Ridge model.

    Prediction over the full province-hour panel is done in chunks. A full dense
    design matrix for the current data can exceed 750 MB for F1/F2/F3.
    """
    full_state = state.copy()

    feature_cols = getattr(model, "_feature_columns", None)
    if feature_cols is None:
        train_state = state[state["year"].isin(cfg.TRAIN_YEARS)].copy()
        X_train = _feature_frame(train_state, feature_set)
        feature_cols = list(X_train.columns)
        model._feature_columns = feature_cols  # type: ignore[attr-defined]

    feature_cols = list(feature_cols)
    if len(feature_cols) != len(model.coef_):
        raise ValueError(
            f"Feature dimension mismatch: stored columns={len(feature_cols)}, "
            f"model coefficients={len(model.coef_)}."
        )

    raw_full = np.empty(len(full_state), dtype=np.float64)
    chunk_size = FIT_CONFIG.predict_chunk_size

    for start in range(0, len(full_state), chunk_size):
        end = min(start + chunk_size, len(full_state))
        chunk = full_state.iloc[start:end]
        X_chunk = _feature_frame(chunk, feature_set)
        X_chunk = _align_feature_chunk(X_chunk, feature_cols)
        raw_full[start:end] = _predict_raw_score(model, X_chunk)
        del X_chunk

    out = full_state[["province", "timestamp_utc", "year"]].copy()
    out["raw_score"] = raw_full
    kappa = _kappa_map(out, raw_col="raw_score")
    out["kappa"] = out["province"].map(kappa).astype(float)
    if out["kappa"].isna().any():
        missing = sorted(out.loc[out["kappa"].isna(), "province"].unique())
        raise ValueError(f"Missing training-year kappa for provinces: {missing}")
    out["debit_weight"] = out["raw_score"] / out["kappa"].clip(lower=1e-9)
    out["feature_set"] = feature_set
    out["variant"] = variant
    out = _apply_q_bounds_and_optional_projection(out)
    return out

def _q_matrix_for_year(weight_df: pd.DataFrame, year: int, provinces: List[str], timestamps: List[pd.Timestamp]) -> np.ndarray:
    T = len(timestamps)
    q = np.ones((len(provinces), T), dtype=float)
    ts_order = pd.Index(pd.to_datetime(timestamps, utc=True))
    sub_all = weight_df[weight_df["year"] == year].copy()
    sub_all["timestamp_utc"] = pd.to_datetime(sub_all["timestamp_utc"], utc=True)

    for n, province in enumerate(provinces):
        sub = sub_all[sub_all["province"] == province].copy()
        if sub.empty:
            raise ValueError(f"Missing q weights for province={province}, year={year}.")
        sub = sub.set_index("timestamp_utc").reindex(ts_order)
        if sub["debit_weight"].isna().any():
            raise ValueError(
                f"Missing {int(sub['debit_weight'].isna().sum())} q timestamps for province={province}, year={year}."
            )
        q[n, :] = pd.to_numeric(sub["debit_weight"], errors="coerce").to_numpy(dtype=float)
    return np.clip(q, FIT_CONFIG.q_min, FIT_CONFIG.q_max)


# =============================================================================
# ANNUAL ARRAYS AND OPTIMIZATION-INFORMED ALPHA
# =============================================================================

def _annual_arrays(
    lp_year: pd.DataFrame,
) -> Tuple[
    List[str],
    List[pd.Timestamp],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Convert one annual LP-parameter dataframe into dense arrays.

    Returns
    -------
    provinces  : list of deployment units
    timestamps : sorted annual time grid
    R          : (N,T) constructed relief requirement
    C          : (N,T) operating cost
    H          : (N,) weighted-depth annual allowance
    P          : (N,) representative/contract capacity
    Pbar       : (N,T) interval availability/capacity bound

    If the parameter file has no Pbar_value column, Pbar defaults to P.
    """
    provinces = [p for p in cfg.NL_PROVINCES if p in set(lp_year["province"])]
    timestamps = list(
        pd.to_datetime(lp_year["timestamp_utc"].drop_duplicates(), utc=True).sort_values()
    )

    N, T = len(provinces), len(timestamps)

    R = np.zeros((N, T), dtype=float)
    C = np.zeros((N, T), dtype=float)
    H = np.zeros(N, dtype=float)
    P = np.zeros(N, dtype=float)
    Pbar = np.zeros((N, T), dtype=float)

    ts_index = pd.DatetimeIndex(timestamps)

    for n, province in enumerate(provinces):
        sub = lp_year[lp_year["province"] == province].copy()
        sub["timestamp_utc"] = pd.to_datetime(sub["timestamp_utc"], utc=True)
        sub = sub.sort_values("timestamp_utc").set_index("timestamp_utc").reindex(ts_index)

        if len(sub) != T:
            raise ValueError(
                f"Unbalanced LP panel for province={province}: got {len(sub)}, expected {T}."
            )

        if sub["R_value"].isna().any():
            raise ValueError(f"Missing R_value entries for province={province}.")
        if sub["c_value"].isna().any():
            raise ValueError(f"Missing c_value entries for province={province}.")
        if sub["H_i"].isna().all():
            raise ValueError(f"Missing H_i for province={province}.")
        if sub["P_i"].isna().all():
            raise ValueError(f"Missing P_i for province={province}.")

        R[n, :] = pd.to_numeric(sub["R_value"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        C[n, :] = pd.to_numeric(sub["c_value"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        H[n] = float(pd.to_numeric(sub["H_i"], errors="coerce").dropna().iloc[0])
        P[n] = float(pd.to_numeric(sub["P_i"], errors="coerce").dropna().iloc[0])

        if "Pbar_value" in sub.columns:
            Pbar[n, :] = (
                pd.to_numeric(sub["Pbar_value"], errors="coerce")
                .fillna(P[n])
                .to_numpy(dtype=float)
            )
        else:
            Pbar[n, :] = P[n]

    R = np.maximum(np.nan_to_num(R, nan=0.0), 0.0)
    C = np.maximum(np.nan_to_num(C, nan=0.0), 0.0)
    H = np.maximum(np.nan_to_num(H, nan=0.0), 0.0)
    P = np.maximum(np.nan_to_num(P, nan=1.0), 1e-9)
    Pbar = np.maximum(np.nan_to_num(Pbar, nan=0.0), 0.0)

    return provinces, timestamps, R, C, H, P, Pbar

def _solve_training_operator(
    R: np.ndarray,
    C: np.ndarray,
    q: np.ndarray,
    H: np.ndarray,
    P: np.ndarray,
    Pbar: np.ndarray,
    M_penalty: float,
):
    """
    Solve the training operating problem.

    Uses the regularized QP if tau_x or tau_s is positive; otherwise uses the
    unregularized LP. Pbar is passed explicitly so the training operator uses
    the same interval availability bounds as the benchmark layer.
    """
    if FIT_CONFIG.tau_x > 0.0 or FIT_CONFIG.tau_s > 0.0:
        return solve_weighted_depth_qp(
            R=R,
            C=C,
            q=q,
            H=H,
            P=P,
            Pbar=Pbar,
            M_penalty=M_penalty,
            tau_x=FIT_CONFIG.tau_x,
            tau_s=FIT_CONFIG.tau_s,
            quiet=True,
            time_limit=FIT_CONFIG.train_time_limit,
        )

    return solve_weighted_depth_lp(
        R=R,
        C=C,
        q=q,
        H=H,
        P=P,
        Pbar=Pbar,
        M_penalty=M_penalty,
        quiet=True,
        time_limit=FIT_CONFIG.train_time_limit,
    )


def _build_training_alpha(weight_df: pd.DataFrame) -> Tuple[pd.DataFrame, float, dict]:
    alpha_rows: List[pd.DataFrame] = []
    train_costs: List[float] = []
    meta = _load_lp_meta()
    M_penalty = float(meta["M_penalty"])

    for year in cfg.TRAIN_YEARS:
        lp_year = _load_lp_year(year)
        provinces, timestamps, R, C, H, P, Pbar = _annual_arrays(lp_year)
        q = _q_matrix_for_year(weight_df, year, provinces, timestamps)
        res = _solve_training_operator(R, C, q, H, P, Pbar, M_penalty)
        train_costs.append(res.cost_total)

        z = res.x / P.reshape(-1, 1)
        for n, province in enumerate(provinces):
            sub = (
                lp_year[lp_year["province"] == province]
                .copy()
                .sort_values("timestamp_utc")[["province", "timestamp_utc"]]
            )
            sub["year"] = int(year)
            sub["alpha"] = float(res.mu[n]) * z[n, :]
            sub["mu"] = float(res.mu[n])
            sub["depth"] = z[n, :]
            alpha_rows.append(sub)

    if not alpha_rows:
        raise RuntimeError("No alpha rows constructed. Check TRAIN_YEARS and LP_DIR.")

    alpha_df = pd.concat(alpha_rows, ignore_index=True)
    alpha_vals = pd.to_numeric(alpha_df["alpha"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mu_vals = pd.to_numeric(alpha_df["mu"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    depth_vals = pd.to_numeric(alpha_df["depth"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    info = {
        "alpha_raw_mean": float(np.mean(alpha_vals)),
        "alpha_raw_max": float(np.max(alpha_vals)) if len(alpha_vals) else 0.0,
        "alpha_positive_share": float(np.mean(alpha_vals > 0.0)) if len(alpha_vals) else 0.0,
        "alpha_mean": float(np.mean(alpha_vals)),
        "mu_positive_share": float(np.mean(mu_vals > 0.0)) if len(mu_vals) else 0.0,
        "depth_positive_share": float(np.mean(depth_vals > 0.0)) if len(depth_vals) else 0.0,
    }
    avg_cost = float(np.mean(train_costs)) if train_costs else float("nan")
    return alpha_df, avg_cost, info


def _alpha_is_effective(alpha_info: dict) -> bool:
    return float(alpha_info.get("alpha_raw_max", 0.0)) > FIT_CONFIG.alpha_abs_tol


def _uniform_training_alpha_cached(state: pd.DataFrame) -> Tuple[pd.DataFrame, float, dict]:
    """Compute q == 1 training alpha once and reuse it across feature sets/variants."""
    global _UNIFORM_ALPHA_CACHE
    if _UNIFORM_ALPHA_CACHE is None:
        uniform = _build_weight_df_uniform(state, feature_set="F0", variant="M3_uniform_cache")
        alpha_df, avg_cost, alpha_info = _build_training_alpha(uniform)
        _UNIFORM_ALPHA_CACHE = (alpha_df.copy(), float(avg_cost), dict(alpha_info))
    alpha_df, avg_cost, alpha_info = _UNIFORM_ALPHA_CACHE
    return alpha_df.copy(), float(avg_cost), dict(alpha_info)


def _partial_paths() -> Tuple[Path, Path]:
    return (
        cfg.M3_WEIGHTS_PATH.with_suffix(".partial.parquet"),
        cfg.M3_LOG_PATH.with_suffix(".partial.csv"),
    )


def _write_partial_outputs(all_weights: List[pd.DataFrame], training_log: List[dict]) -> None:
    if not FIT_CONFIG.write_partial_outputs or not all_weights:
        return
    w_path, l_path = _partial_paths()
    w_path.parent.mkdir(parents=True, exist_ok=True)
    l_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_weights, ignore_index=True).to_parquet(w_path, index=False)
    pd.DataFrame(training_log).to_csv(l_path, index=False)


def _merge_alpha_to_training_state(state: pd.DataFrame, alpha_df: pd.DataFrame) -> pd.DataFrame:
    train_state = state[state["year"].isin(cfg.TRAIN_YEARS)].copy()
    merged = train_state.merge(
        alpha_df[["province", "timestamp_utc", "year", "alpha"]],
        on=["province", "timestamp_utc", "year"],
        how="left",
    )
    merged["alpha"] = pd.to_numeric(merged["alpha"], errors="coerce").fillna(0.0)
    return merged


# =============================================================================
# VARIANT FITTING
# =============================================================================

def _fit_variant(state_raw: pd.DataFrame, feature_set: str, variant: str) -> Tuple[pd.DataFrame, List[dict]]:
    state = _build_target_column(state_raw, cfg.PRIMARY_BURDEN)
    train_state = state[state["year"].isin(cfg.TRAIN_YEARS)].copy()

    X_train = _feature_frame(train_state, feature_set)
    y_train = np.log(np.clip(train_state["b_target"].to_numpy(dtype=float), FIT_CONFIG.eps_target, None))
    log_rows: List[dict] = []

    if variant == "M3_plain":
        model, weight_info = _fit_ridge_with_info(X_train, y_train, np.ones(len(train_state), dtype=float))
        weight_df = _build_weight_df_from_model(state, model, feature_set, variant)
        log_rows.append({
            "feature_set": feature_set,
            "variant": variant,
            "iter": 0,
            "delta_theta_rel": np.nan,
            "train_cost": np.nan,
            "regularized_training_operator": int(FIT_CONFIG.tau_x > 0 or FIT_CONFIG.tau_s > 0),
            "q_projection_used": int(FIT_CONFIG.use_annual_projection),
            **weight_info,
        })
        return weight_df, log_rows

    if variant not in {"M3_OI", "M3_LO"}:
        raise ValueError(f"Unknown M3 variant: {variant}")

    # Paper-consistent initialization: q == 1. This solve is independent of
    # the public feature set, so cache it.
    alpha_df, avg_cost, alpha_info = _uniform_training_alpha_cached(state)
    train_alpha = _merge_alpha_to_training_state(state, alpha_df)
    sample_weight_raw = train_alpha["alpha"].to_numpy(dtype=float)

    if variant == "M3_OI":
        model, weight_info = _fit_ridge_with_info(X_train, y_train, sample_weight_raw)
        weight_df = _build_weight_df_from_model(state, model, feature_set, variant)
        log_rows.append({
            "feature_set": feature_set,
            "variant": variant,
            "iter": 0,
            "delta_theta_rel": np.nan,
            "train_cost": avg_cost,
            "regularized_training_operator": int(FIT_CONFIG.tau_x > 0 or FIT_CONFIG.tau_s > 0),
            "q_projection_used": int(FIT_CONFIG.use_annual_projection),
            **alpha_info,
            **weight_info,
        })
        return weight_df, log_rows

    # M3_LO: first model is the OI refit from q == 1 scarcity weights.
    previous_model, first_weight_info = _fit_ridge_with_info(X_train, y_train, sample_weight_raw)

    # If the q == 1 scarcity signal is numerically zero, LO has no meaningful
    # optimization-informed direction. Do not spend hours resolving QPs that
    # only reproduce the same near-zero alpha.
    if FIT_CONFIG.skip_lo_if_no_scarcity and not _alpha_is_effective(alpha_info):
        weight_df = _build_weight_df_from_model(state, previous_model, feature_set, variant)
        log_rows.append({
            "feature_set": feature_set,
            "variant": variant,
            "iter": 0,
            "delta_theta_rel": np.nan,
            "train_cost": avg_cost,
            "regularized_training_operator": int(FIT_CONFIG.tau_x > 0 or FIT_CONFIG.tau_s > 0),
            "q_projection_used": int(FIT_CONFIG.use_annual_projection),
            "lo_skipped_no_scarcity": 1,
            **alpha_info,
            **first_weight_info,
        })
        return weight_df, log_rows

    current_weights = _build_weight_df_from_model(state, previous_model, feature_set, variant)
    best_model = copy.deepcopy(previous_model)
    best_cost = float("inf")

    # Evaluate initial learned rule and then iterate.
    for it in range(FIT_CONFIG.max_iter):
        alpha_df, avg_cost, alpha_info = _build_training_alpha(current_weights)
        if np.isfinite(avg_cost) and avg_cost < best_cost:
            best_cost = avg_cost
            best_model = copy.deepcopy(previous_model)

        train_alpha = _merge_alpha_to_training_state(state, alpha_df)
        sample_weight_raw = train_alpha["alpha"].to_numpy(dtype=float)

        if FIT_CONFIG.skip_lo_if_no_scarcity and not _alpha_is_effective(alpha_info):
            log_rows.append({
                "feature_set": feature_set,
                "variant": variant,
                "iter": it,
                "delta_theta_rel": 0.0,
                "train_cost": avg_cost,
                "regularized_training_operator": int(FIT_CONFIG.tau_x > 0 or FIT_CONFIG.tau_s > 0),
                "q_projection_used": int(FIT_CONFIG.use_annual_projection),
                "lo_skipped_no_scarcity": 1,
                **alpha_info,
                **first_weight_info,
            })
            break

        candidate_model, weight_info = _fit_ridge_with_info(X_train, y_train, sample_weight_raw)
        model_new = _damped_model(previous_model, candidate_model, FIT_CONFIG.damping)
        delta_rel = _relative_model_change(previous_model, model_new)

        log_rows.append({
            "feature_set": feature_set,
            "variant": variant,
            "iter": it,
            "delta_theta_rel": delta_rel,
            "train_cost": avg_cost,
            "regularized_training_operator": int(FIT_CONFIG.tau_x > 0 or FIT_CONFIG.tau_s > 0),
            "q_projection_used": int(FIT_CONFIG.use_annual_projection),
            **alpha_info,
            **weight_info,
        })

        previous_model = model_new
        current_weights = _build_weight_df_from_model(state, previous_model, feature_set, variant)

        if it > 0 and delta_rel < FIT_CONFIG.tol_rel:
            break

    weight_df = _build_weight_df_from_model(state, best_model, feature_set, variant)
    return weight_df, log_rows


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("C: TRAIN M3")
    print("=" * 72)
    print(f"State panel : {cfg.STATE_PATH}")
    print(f"LP meta     : {cfg.LP_DIR / 'lp_meta.json'}")
    print(f"Output      : {cfg.M3_WEIGHTS_PATH}")
    print(f"q bounds    : [{FIT_CONFIG.q_min}, {FIT_CONFIG.q_max}]")
    print(f"projection  : {FIT_CONFIG.use_annual_projection}")
    print(f"tau_x,tau_s : {FIT_CONFIG.tau_x}, {FIT_CONFIG.tau_s}")
    print(f"alpha tol   : {FIT_CONFIG.alpha_abs_tol}")
    print(f"chunk size  : {FIT_CONFIG.predict_chunk_size}")
    if FIT_CONFIG.write_partial_outputs:
        w_partial, l_partial = _partial_paths()
        print(f"partial out : {w_partial}")
        print(f"partial log : {l_partial}")

    if not cfg.STATE_PATH.exists():
        raise FileNotFoundError(f"Missing state panel: {cfg.STATE_PATH}")
    if not (cfg.LP_DIR / "lp_meta.json").exists():
        raise FileNotFoundError(f"Missing LP meta: {cfg.LP_DIR / 'lp_meta.json'}")

    cfg.M3_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.M3_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    state = _load_state()
    all_weights: List[pd.DataFrame] = []
    training_log: List[dict] = []

    for feature_set in ["F0", "F1", "F2", "F3"]:
        print("\n" + "─" * 72)
        print(f"Feature set: {feature_set}")
        print("─" * 72)
        for variant in cfg.M3_VARIANTS:
            t0 = time.time()
            print(f"Training {variant} / {feature_set} ...")
            weight_df, log_rows = _fit_variant(state, feature_set, variant)
            all_weights.append(weight_df)
            training_log.extend(log_rows)

            elapsed = time.time() - t0
            train_diag = weight_df[weight_df["year"].isin(cfg.TRAIN_YEARS)].groupby("province", observed=True)["debit_weight"].mean()
            holdout_diag = weight_df[weight_df["year"].isin(cfg.TEST_YEARS)].groupby("province", observed=True)["debit_weight"].mean()
            if len(train_diag):
                print(f"  train mean q:   min={train_diag.min():.3f} max={train_diag.max():.3f} mean={train_diag.mean():.3f}")
            if len(holdout_diag):
                print(f"  holdout mean q: min={holdout_diag.min():.3f} max={holdout_diag.max():.3f} mean={holdout_diag.mean():.3f}")
            if log_rows:
                last = log_rows[-1]
                if pd.notna(last.get("train_cost", np.nan)):
                    print(f"  last train cost: {last['train_cost']:.4f}")
                print(
                    f"  alpha positive share: {last.get('alpha_positive_share', np.nan):.4f} "
                    f"alpha max: {last.get('alpha_raw_max', np.nan):.4g} "
                    f"fallback: {last.get('alpha_fallback_uniform', np.nan)}"
                )
            print(f"  done in {elapsed:.1f}s")
            _write_partial_outputs(all_weights, training_log)

    out_weights = pd.concat(all_weights, ignore_index=True)
    out_log = pd.DataFrame(training_log)

    out_weights.to_parquet(cfg.M3_WEIGHTS_PATH, index=False)
    out_log.to_csv(cfg.M3_LOG_PATH, index=False)

    print(f"\n✓ {cfg.M3_WEIGHTS_PATH}")
    print(f"✓ {cfg.M3_LOG_PATH}")


if __name__ == "__main__":
    main()
