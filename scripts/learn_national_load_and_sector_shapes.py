#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
learn_national_load_and_sector_shapes.py

Purpose
-------
Build future national hourly Dutch electricity load for 2025-2035 with explicit
household (HH) and company/non-residential (CO) components.

This script sits between:
    1) build_national_load_history_master.py
    2) build_province_sector_targets_and_factors.py
and upstream of:
    3) build_provincial_hourly_load_from_learned_targets.py

What it does
------------
A. Read the QC-clean historical national hourly load master:
       derived/national_hourly_load_history.csv

B. Build historical national annual energy from the hourly master.

C. Read national HH/CO annual mix from:
       derived/national_sector_mix_2010_2035.csv
   (preferred)
   or, for backward compatibility, the legacy pair:
       derived/national_sector_mix_observed_2012_2024.csv
       derived/national_sector_mix_2025_2035_forecast.csv

D. Learn future national annual total load for 2025-2035.
   Default model is log-linear, not quadratic. Quadratic extrapolation is
   intentionally avoided as the default because it is fragile over 11 years.

E. Decompose historical national hourly load into two positive sectoral shapes:
       L_nat(y,t) ≈ E_HH(y) * phi_HH(y,t) + E_CO(y) * phi_CO(y,t)
   where each phi_s(y,·) is a positive hourly shape summing to one.

   Important identifiability note:
   --------------------------------
   Without observed hourly HH/CO metering, the decomposition is underidentified.
   So this script regularizes the two sector shapes toward economically
   interpretable calendar anchors:
       - HH more evening / morning / weekend / holiday weighted
       - CO more weekday daytime weighted
   while still fitting the observed national hourly total load.

F. Use forecast national annual totals and learned calendar-shape models to
   construct:
       derived/national_hourly_load_2025_2035_learned.csv

Main outputs
------------
- derived/national_annual_load_observed_hist.csv
- derived/national_annual_load_2025_2035_forecast.csv
- derived/national_sector_totals_observed_overlap.csv
- derived/national_sector_totals_2025_2035_forecast.csv
- derived/national_hourly_load_2025_2035_learned.csv
- derived/national_load_and_sector_shapes_diagnostics.json
- derived/national_shape_feature_coefficients.csv

Required future hourly output columns
-------------------------------------
- timestamp_utc
- year
- hour_of_year
- nat_hh_load_MW
- nat_co_load_MW
- nat_load_MW

Design improvements over the older version
------------------------------------------
1) Uses the new master file in derived/, not a raw root-level file.
2) Uses the new unified national_sector_mix_2010_2035.csv if available.
3) Does not re-interpolate historical years aggressively; the master file is
   already QCed and should be treated as the source of truth.
4) Uses 1-based hour_of_year to match the master builder.
5) Replaces unstable default quadratic annual extrapolation with log-linear.
6) Adds sector-anchor regularization so HH and CO do not collapse to nearly
   identical shapes.
7) Computes Dutch movable holidays algorithmically instead of hard-coding.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import sys

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
# Preferred new inputs
NAT_HOURLY_HISTORY_FILE = getattr(
    cfg,
    "NATIONAL_LOAD_HISTORY_PATH",
    cfg.DERIVED_DIR / "national_hourly_load_history.csv",
)
UNIFIED_MIX_FILE = getattr(
    cfg,
    "NATIONAL_SECTOR_MIX_UNIFIED_PATH",
    cfg.DERIVED_DIR / "national_sector_mix_2010_2035.csv",
)

# Legacy fallback inputs (supported only for backward compatibility)
LEGACY_OBS_MIX_FILE = getattr(
    cfg,
    "NATIONAL_SECTOR_MIX_OBS_LEGACY_PATH",
    cfg.DERIVED_DIR / "national_sector_mix_observed_2012_2024.csv",
)
LEGACY_FC_MIX_FILE = getattr(
    cfg,
    "NATIONAL_SECTOR_MIX_FORECAST_LEGACY_PATH",
    cfg.DERIVED_DIR / "national_sector_mix_2025_2035_forecast.csv",
)

# Outputs
OUT_OBS_ANN_LOAD = getattr(
    cfg,
    "NATIONAL_ANNUAL_LOAD_OBS_PATH",
    cfg.DERIVED_DIR / "national_annual_load_observed_hist.csv",
)
OUT_FC_ANN_LOAD = getattr(
    cfg,
    "NATIONAL_ANNUAL_LOAD_FORECAST_PATH",
    cfg.DERIVED_DIR / "national_annual_load_2025_2035_forecast.csv",
)
OUT_OBS_SECTOR_TOTALS = getattr(
    cfg,
    "NATIONAL_SECTOR_TOTALS_OBS_OVERLAP_PATH",
    cfg.DERIVED_DIR / "national_sector_totals_observed_overlap.csv",
)
OUT_FC_SECTOR_TOTALS = getattr(
    cfg,
    "NATIONAL_SECTOR_TOTALS_FORECAST_PATH",
    cfg.DERIVED_DIR / "national_sector_totals_2025_2035_forecast.csv",
)
OUT_FC_HOURLY = getattr(
    cfg,
    "NATIONAL_HOURLY_LOAD_LEARNED_PATH",
    cfg.DERIVED_DIR / "national_hourly_load_2025_2035_learned.csv",
)
OUT_DIAG = getattr(
    cfg,
    "NATIONAL_LOAD_SHAPES_DIAG_PATH",
    cfg.DERIVED_DIR / "national_load_and_sector_shapes_diagnostics.json",
)
OUT_COEFS = getattr(
    cfg,
    "NATIONAL_SHAPE_FEATURE_COEFS_PATH",
    cfg.DERIVED_DIR / "national_shape_feature_coefficients.csv",
)

FORECAST_START = 2025
FORECAST_END = 2035
FORECAST_YEARS = list(range(FORECAST_START, FORECAST_END + 1))

# Annual forecast model
ANNUAL_MODEL = "log_linear"   # options: log_linear, log_quadratic
ANNUAL_RIDGE = 1e-5

# Shape model features
USE_MONTH_DUMMIES = True
USE_HOUR_DUMMIES = True
USE_WEEKEND = True
USE_HOLIDAY = True
USE_INTERACTIONS_HOUR_WEEKEND = True

# Shape model optimization
RIDGE_DATA = 1e-4               # generic coefficient shrinkage
RIDGE_ANCHOR = 2e-2             # pull toward HH/CO calendar anchors
MAX_ITERS = 5000
LR = 0.03
SEED = 7
PRINT_EVERY = 500

# Small numerical floor
EPS = 1e-12


# =============================================================================
# GENERIC HELPERS
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


def softmax1d(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    ez = np.exp(z)
    s = ez.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(z) / len(z)
    return ez / s


def hours_in_year_utc(y: int) -> int:
    t0 = pd.Timestamp(f"{y}-01-01T00:00:00Z")
    t1 = pd.Timestamp(f"{y+1}-01-01T00:00:00Z")
    return int((t1 - t0) / pd.Timedelta(hours=1))


def annual_mwh_from_hourly_mw(x_mw: np.ndarray) -> float:
    # Hourly MW observed at hourly resolution integrates to MWh by summation.
    return float(np.sum(np.asarray(x_mw, dtype=float)))


# =============================================================================
# DUTCH HOLIDAYS (algorithmic)
# =============================================================================
def easter_sunday(year: int) -> pd.Timestamp:
    """Anonymous Gregorian computus."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def dutch_holidays(year: int) -> set[str]:
    easter = easter_sunday(year)
    good_friday = easter - pd.Timedelta(days=2)
    easter_monday = easter + pd.Timedelta(days=1)
    ascension = easter + pd.Timedelta(days=39)
    whit_sunday = easter + pd.Timedelta(days=49)
    whit_monday = easter + pd.Timedelta(days=50)

    kings_day = pd.Timestamp(year=year, month=4, day=27)
    # If King's Day falls on Sunday, celebration moves to Saturday 26 April.
    if kings_day.weekday() == 6:
        kings_day = pd.Timestamp(year=year, month=4, day=26)

    fixed = [
        pd.Timestamp(year=year, month=1, day=1),   # New Year
        good_friday,
        easter,
        easter_monday,
        kings_day,
        pd.Timestamp(year=year, month=5, day=5),   # Liberation Day (kept every year for consistency)
        ascension,
        whit_sunday,
        whit_monday,
        pd.Timestamp(year=year, month=12, day=25),
        pd.Timestamp(year=year, month=12, day=26),
    ]
    return {d.strftime("%Y-%m-%d") for d in fixed}


# =============================================================================
# INPUT LOADERS
# =============================================================================
def load_historical_national_hourly() -> pd.DataFrame:
    if not NAT_HOURLY_HISTORY_FILE.exists():
        raise FileNotFoundError(f"Missing historical national hourly master file:\n  {NAT_HOURLY_HISTORY_FILE}")

    df = pd.read_csv(NAT_HOURLY_HISTORY_FILE)
    df.columns = [c.strip() for c in df.columns]

    ts_col = pick_first_existing(df.columns.tolist(), ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc"])
    load_col = pick_first_existing(df.columns.tolist(), ["nat_load_MW", "load_MW", "NL_load_actual_entsoe_transparency"])

    if ts_col is None:
        raise ValueError(f"[NAT_HIST] Missing timestamp column. Has: {list(df.columns)}")
    if load_col is None:
        raise ValueError(f"[NAT_HIST] Missing load column. Has: {list(df.columns)}")

    df["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df["nat_load_MW"] = to_num(df[load_col]).astype(float)

    keep_cols = ["timestamp_utc", "nat_load_MW"]
    if "year" in df.columns:
        keep_cols.append("year")
    if "hour_of_year" in df.columns:
        keep_cols.append("hour_of_year")
    if "nat_load_source" in df.columns:
        keep_cols.append("nat_load_source")
    if "nat_load_was_interpolated" in df.columns:
        keep_cols.append("nat_load_was_interpolated")

    df = df[keep_cols].dropna(subset=["timestamp_utc", "nat_load_MW"]).copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if "year" not in df.columns:
        df["year"] = df["timestamp_utc"].dt.year.astype(int)
    else:
        df["year"] = to_num(df["year"]).astype(int)

    # Verify full-year integrity against the master output instead of rebuilding years from scratch.
    rows = []
    for y, g in df.groupby("year", sort=True):
        H = hours_in_year_utc(int(y))
        if len(g) != H:
            raise ValueError(
                f"[NAT_HIST] Year {y} has {len(g)} rows but expected {H}. "
                f"The master file should already be QC-clean."
            )
        gg = g.sort_values("timestamp_utc").copy()
        gg["hour_of_year"] = np.arange(1, H + 1, dtype=int)
        rows.append(gg)

    out = pd.concat(rows, ignore_index=True)
    return out[["year", "hour_of_year", "timestamp_utc", "nat_load_MW"]].copy()


def load_unified_mix_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = {"year", "omega_hh", "omega_co"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"[UNIFIED_MIX] Missing columns: {missing}")

    df["year"] = to_num(df["year"]).astype(int)
    df["omega_hh"] = to_num(df["omega_hh"]).astype(float)
    df["omega_co"] = to_num(df["omega_co"]).astype(float)

    # Optional observed annual totals from the province-sector stage.
    for c in [
        "nat_hh_electricity_MWh_year_obs",
        "nat_co_electricity_MWh_year_obs",
        "nat_total_electricity_MWh_year_obs",
        "omega_mode",
    ]:
        if c not in df.columns:
            df[c] = np.nan

    return df.sort_values("year").reset_index(drop=True)


def load_legacy_mix_files(obs_path: Path, fc_path: Path) -> pd.DataFrame:
    obs = pd.read_csv(obs_path)
    fc = pd.read_csv(fc_path)
    obs.columns = [c.strip() for c in obs.columns]
    fc.columns = [c.strip() for c in fc.columns]

    req_obs = {"year", "omega_hh", "omega_co"}
    req_fc = {"year"}
    miss_obs = sorted(req_obs - set(obs.columns))
    miss_fc = sorted(req_fc - set(fc.columns))
    if miss_obs:
        raise ValueError(f"[LEGACY_OBS_MIX] Missing columns: {miss_obs}")
    if miss_fc:
        raise ValueError(f"[LEGACY_FC_MIX] Missing columns: {miss_fc}")

    obs["year"] = to_num(obs["year"]).astype(int)
    obs["omega_hh"] = to_num(obs["omega_hh"]).astype(float)
    obs["omega_co"] = to_num(obs["omega_co"]).astype(float)

    fc["year"] = to_num(fc["year"]).astype(int)
    hh_col = pick_first_existing(fc.columns.tolist(), ["omega_hh_hat", "omega_hh"])
    co_col = pick_first_existing(fc.columns.tolist(), ["omega_co_hat", "omega_co"])
    if hh_col is None or co_col is None:
        raise ValueError("[LEGACY_FC_MIX] Need omega_hh_hat/omega_co_hat or omega_hh/omega_co")

    fc["omega_hh"] = to_num(fc[hh_col]).astype(float)
    fc["omega_co"] = to_num(fc[co_col]).astype(float)

    obs_keep = obs[["year", "omega_hh", "omega_co"]].copy()
    fc_keep = fc[["year", "omega_hh", "omega_co"]].copy()
    out = pd.concat([obs_keep, fc_keep], ignore_index=True)
    out = out.drop_duplicates(subset=["year"], keep="last").sort_values("year").reset_index(drop=True)

    out["nat_hh_electricity_MWh_year_obs"] = np.nan
    out["nat_co_electricity_MWh_year_obs"] = np.nan
    out["nat_total_electricity_MWh_year_obs"] = np.nan
    out["omega_mode"] = np.where(out["year"] <= obs["year"].max(), "legacy_observed_or_frozen", "legacy_forecast")
    return out


def load_sector_mix() -> pd.DataFrame:
    if UNIFIED_MIX_FILE.exists():
        print(f"[INFO] Using unified sector mix file: {UNIFIED_MIX_FILE}")
        return load_unified_mix_file(UNIFIED_MIX_FILE)

    if LEGACY_OBS_MIX_FILE.exists() and LEGACY_FC_MIX_FILE.exists():
        print("[WARN] Unified sector mix file not found; using legacy observed/forecast mix files.")
        return load_legacy_mix_files(LEGACY_OBS_MIX_FILE, LEGACY_FC_MIX_FILE)

    raise FileNotFoundError(
        "Missing national sector mix input. Expected either:\n"
        f"  {UNIFIED_MIX_FILE}\n"
        "or both legacy files:\n"
        f"  {LEGACY_OBS_MIX_FILE}\n"
        f"  {LEGACY_FC_MIX_FILE}"
    )


# =============================================================================
# ANNUAL NATIONAL LOAD FORECAST
# =============================================================================
@dataclass
class AnnualLoadTrend:
    years_train: np.ndarray
    beta: np.ndarray
    model_type: str


def fit_annual_load_trend(obs_annual: pd.DataFrame, model_type: str = ANNUAL_MODEL) -> AnnualLoadTrend:
    y = obs_annual["year"].to_numpy(dtype=int)
    e = obs_annual["nat_annual_mwh_obs"].to_numpy(dtype=float)

    if np.any(e <= 0):
        raise ValueError("[ANNUAL] Observed annual national load must be strictly positive.")

    t_raw = (y - y.min()).astype(float)
    mu = t_raw.mean()
    sd = t_raw.std() + 1e-12
    t = (t_raw - mu) / sd

    if model_type == "log_linear":
        X = np.column_stack([np.ones_like(t), t])
    elif model_type == "log_quadratic":
        X = np.column_stack([np.ones_like(t), t, t ** 2])
    else:
        raise ValueError(f"Unsupported ANNUAL_MODEL={model_type!r}")

    target = np.log(e)
    beta = np.linalg.solve(X.T @ X + ANNUAL_RIDGE * np.eye(X.shape[1]), X.T @ target)
    return AnnualLoadTrend(years_train=y, beta=beta, model_type=model_type)


def predict_annual_load_mwh(model: AnnualLoadTrend, years_pred: np.ndarray) -> np.ndarray:
    years_train = model.years_train.astype(int)
    years_pred = years_pred.astype(int)

    t_raw_tr = (years_train - years_train.min()).astype(float)
    mu = t_raw_tr.mean()
    sd = t_raw_tr.std() + 1e-12

    t_raw_pr = (years_pred - years_train.min()).astype(float)
    t_pr = (t_raw_pr - mu) / sd

    if model.model_type == "log_linear":
        Xp = np.column_stack([np.ones_like(t_pr), t_pr])
    else:
        Xp = np.column_stack([np.ones_like(t_pr), t_pr, t_pr ** 2])

    return np.exp(Xp @ model.beta)


# =============================================================================
# CALENDAR FEATURES
# =============================================================================
def build_calendar_features_for_year(y: int) -> pd.DataFrame:
    H = hours_in_year_utc(y)
    ts_utc = pd.date_range(pd.Timestamp(f"{y}-01-01T00:00:00Z"), periods=H, freq="h")
    ts_local = ts_utc.tz_convert("Europe/Amsterdam")

    df = pd.DataFrame({
        "timestamp_utc": ts_utc,
        "year": int(y),
        "hour_of_year": np.arange(1, H + 1, dtype=int),
        "hour_of_day": ts_local.hour.astype(int),
        "weekday": ts_local.weekday.astype(int),
        "month": ts_local.month.astype(int),
    })

    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    holidays = dutch_holidays(y)
    local_date_str = pd.Series(ts_local.date.astype(str), index=df.index)
    df["is_holiday"] = local_date_str.isin(holidays).astype(int)
    df["is_weekday_daytime"] = ((df["weekday"] <= 4) & (df["hour_of_day"].between(8, 17))).astype(int)
    df["is_evening"] = df["hour_of_day"].isin([17, 18, 19, 20, 21, 22]).astype(int)
    df["is_morning"] = df["hour_of_day"].isin([6, 7, 8]).astype(int)
    df["is_night"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    mats = []
    names = []

    mats.append(np.ones((len(df), 1), dtype=float))
    names.append("intercept")

    if USE_HOUR_DUMMIES:
        hod = pd.get_dummies(df["hour_of_day"], prefix="hod", drop_first=False).astype(float)
        mats.append(hod.to_numpy())
        names.extend(hod.columns.tolist())

    wd = pd.get_dummies(df["weekday"], prefix="wd", drop_first=False).astype(float)
    mats.append(wd.to_numpy())
    names.extend(wd.columns.tolist())

    if USE_MONTH_DUMMIES:
        mo = pd.get_dummies(df["month"], prefix="m", drop_first=False).astype(float)
        mats.append(mo.to_numpy())
        names.extend(mo.columns.tolist())

    if USE_WEEKEND:
        mats.append(df[["is_weekend"]].to_numpy(dtype=float))
        names.append("is_weekend")

    if USE_HOLIDAY:
        mats.append(df[["is_holiday"]].to_numpy(dtype=float))
        names.append("is_holiday")

    mats.append(df[["is_weekday_daytime", "is_evening", "is_morning", "is_night"]].to_numpy(dtype=float))
    names.extend(["is_weekday_daytime", "is_evening", "is_morning", "is_night"])

    if USE_INTERACTIONS_HOUR_WEEKEND and USE_HOUR_DUMMIES:
        hod = pd.get_dummies(df["hour_of_day"], prefix="hod", drop_first=False).astype(float)
        inter = hod.multiply(df["is_weekend"].to_numpy(dtype=float), axis=0)
        inter.columns = [f"{c}_x_weekend" for c in inter.columns]
        mats.append(inter.to_numpy())
        names.extend(inter.columns.tolist())

    X = np.column_stack(mats).astype(float)
    return X, names


# =============================================================================
# SHAPE IDENTIFICATION ANCHORS
# =============================================================================
def build_anchor_scores(cal: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build economically interpretable sector anchor scores at hourly resolution.
    These are not treated as truth, only as regularization targets to avoid the
    HH/CO shapes collapsing to near-identical profiles.
    """
    is_weekend = cal["is_weekend"].to_numpy(dtype=float)
    is_holiday = cal["is_holiday"].to_numpy(dtype=float)
    is_weekday_daytime = cal["is_weekday_daytime"].to_numpy(dtype=float)
    is_evening = cal["is_evening"].to_numpy(dtype=float)
    is_morning = cal["is_morning"].to_numpy(dtype=float)
    is_night = cal["is_night"].to_numpy(dtype=float)

    # Household anchor: stronger evening/morning, higher on weekends/holidays.
    hh_score = (
        0.35 * is_weekend
        + 0.45 * is_holiday
        + 0.85 * is_evening
        + 0.35 * is_morning
        - 0.45 * is_weekday_daytime
        - 0.35 * is_night
    )

    # Company/non-residential anchor: stronger weekday daytime, weaker on weekends/holidays/nights.
    co_score = (
        1.10 * is_weekday_daytime
        - 0.90 * is_weekend
        - 1.10 * is_holiday
        - 0.60 * is_night
        - 0.25 * is_evening
    )
    return hh_score, co_score


def project_anchor_scores_to_feature_betas(X: np.ndarray, score: np.ndarray, ridge: float) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ score)


# =============================================================================
# SHAPE LEARNING
# =============================================================================
@dataclass
class SectorShapeModel:
    beta_hh: np.ndarray
    beta_co: np.ndarray
    beta_hh_anchor: np.ndarray
    beta_co_anchor: np.ndarray
    feature_names: List[str]
    train_years: List[int]
    in_sample_rmse_mw: float


def prepare_overlap_training_data(
    nat_hourly_hist: pd.DataFrame,
    mix_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hist_years = sorted(nat_hourly_hist["year"].unique().tolist())
    mix_years = sorted(mix_df.loc[mix_df[["omega_hh", "omega_co"]].notna().all(axis=1), "year"].unique().tolist())
    overlap_years = sorted(set(hist_years).intersection(set(mix_years)))

    if not overlap_years:
        raise ValueError("[SHAPES] No overlap between historical hourly load years and available omega years.")

    nat = nat_hourly_hist[nat_hourly_hist["year"].isin(overlap_years)].copy()

    ann = (
        nat.groupby("year", as_index=False)["nat_load_MW"]
           .sum()
           .rename(columns={"nat_load_MW": "nat_annual_mwh_obs_overlap"})
           .sort_values("year")
           .reset_index(drop=True)
    )

    mix = mix_df[mix_df["year"].isin(overlap_years)].copy().sort_values("year").reset_index(drop=True)
    mix = mix.merge(ann, on="year", how="inner", validate="one_to_one")
    mix["hh_total_mwh_obs_overlap"] = mix["omega_hh"] * mix["nat_annual_mwh_obs_overlap"]
    mix["co_total_mwh_obs_overlap"] = mix["omega_co"] * mix["nat_annual_mwh_obs_overlap"]

    return nat, mix


def fit_sector_shape_model(
    nat_hourly_hist: pd.DataFrame,
    mix_overlap: pd.DataFrame,
) -> SectorShapeModel:
    rng = np.random.default_rng(SEED)

    by_year: Dict[int, Dict[str, np.ndarray | float]] = {}
    feature_names: Optional[List[str]] = None
    beta_hh_anchor_accum = None
    beta_co_anchor_accum = None

    for y in sorted(nat_hourly_hist["year"].unique()):
        cal = build_calendar_features_for_year(int(y))
        X, feature_names_y = build_feature_matrix(cal)
        if feature_names is None:
            feature_names = feature_names_y
        elif feature_names != feature_names_y:
            raise RuntimeError(f"[SHAPES] Feature mismatch in year {y}")

        load = nat_hourly_hist.loc[nat_hourly_hist["year"] == y].sort_values("hour_of_year")["nat_load_MW"].to_numpy(dtype=float)
        if len(load) != len(cal):
            raise ValueError(f"[SHAPES] Year {y}: hourly length mismatch")

        mix_row = mix_overlap.loc[mix_overlap["year"] == y]
        if mix_row.empty:
            raise ValueError(f"[SHAPES] Missing overlap annual totals for year {y}")

        E_hh = float(mix_row["hh_total_mwh_obs_overlap"].iloc[0])
        E_co = float(mix_row["co_total_mwh_obs_overlap"].iloc[0])

        hh_score, co_score = build_anchor_scores(cal)
        beta_hh_anchor_y = project_anchor_scores_to_feature_betas(X, hh_score, ridge=1e-4)
        beta_co_anchor_y = project_anchor_scores_to_feature_betas(X, co_score, ridge=1e-4)

        if beta_hh_anchor_accum is None:
            beta_hh_anchor_accum = beta_hh_anchor_y.copy()
            beta_co_anchor_accum = beta_co_anchor_y.copy()
        else:
            beta_hh_anchor_accum += beta_hh_anchor_y
            beta_co_anchor_accum += beta_co_anchor_y

        by_year[int(y)] = {
            "X": X,
            "load": load,
            "E_hh": E_hh,
            "E_co": E_co,
        }

    assert feature_names is not None
    p = len(feature_names)
    n_years = len(by_year)

    beta_hh_anchor = beta_hh_anchor_accum / n_years
    beta_co_anchor = beta_co_anchor_accum / n_years

    beta_hh = beta_hh_anchor + 0.01 * rng.standard_normal(p)
    beta_co = beta_co_anchor + 0.01 * rng.standard_normal(p)

    m_hh = np.zeros_like(beta_hh)
    v_hh = np.zeros_like(beta_hh)
    m_co = np.zeros_like(beta_co)
    v_co = np.zeros_like(beta_co)
    b1, b2 = 0.9, 0.999
    eps_adam = 1e-8

    best_rmse = np.inf
    best_beta_hh = beta_hh.copy()
    best_beta_co = beta_co.copy()

    for step in range(1, MAX_ITERS + 1):
        g_hh = np.zeros_like(beta_hh)
        g_co = np.zeros_like(beta_co)
        sse = 0.0
        nobs = 0

        for y, pack in by_year.items():
            X = pack["X"]
            load = pack["load"]
            E_hh = float(pack["E_hh"])
            E_co = float(pack["E_co"])

            z_hh = X @ beta_hh
            z_co = X @ beta_co
            phi_hh = softmax1d(z_hh)
            phi_co = softmax1d(z_co)

            pred = E_hh * phi_hh + E_co * phi_co
            resid = pred - load
            sse += float(np.sum(resid ** 2))
            nobs += len(load)

            s_hh = E_hh * resid
            inner_hh = float(np.dot(phi_hh, s_hh))
            g_hh += X.T @ (phi_hh * (s_hh - inner_hh))

            s_co = E_co * resid
            inner_co = float(np.dot(phi_co, s_co))
            g_co += X.T @ (phi_co * (s_co - inner_co))

        # Average gradients + regularization
        g_hh = g_hh / max(nobs, 1)
        g_co = g_co / max(nobs, 1)

        g_hh += 2.0 * RIDGE_DATA * beta_hh + 2.0 * RIDGE_ANCHOR * (beta_hh - beta_hh_anchor)
        g_co += 2.0 * RIDGE_DATA * beta_co + 2.0 * RIDGE_ANCHOR * (beta_co - beta_co_anchor)

        m_hh = b1 * m_hh + (1 - b1) * g_hh
        v_hh = b2 * v_hh + (1 - b2) * (g_hh * g_hh)
        mhat_hh = m_hh / (1 - b1 ** step)
        vhat_hh = v_hh / (1 - b2 ** step)

        m_co = b1 * m_co + (1 - b1) * g_co
        v_co = b2 * v_co + (1 - b2) * (g_co * g_co)
        mhat_co = m_co / (1 - b1 ** step)
        vhat_co = v_co / (1 - b2 ** step)

        beta_hh -= LR * mhat_hh / (np.sqrt(vhat_hh) + eps_adam)
        beta_co -= LR * mhat_co / (np.sqrt(vhat_co) + eps_adam)

        rmse = float(np.sqrt(sse / max(nobs, 1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_beta_hh = beta_hh.copy()
            best_beta_co = beta_co.copy()

        if step % PRINT_EVERY == 0:
            print(f"  step={step:4d} | in-sample RMSE={rmse:,.3f} MW")

    return SectorShapeModel(
        beta_hh=best_beta_hh,
        beta_co=best_beta_co,
        beta_hh_anchor=beta_hh_anchor,
        beta_co_anchor=beta_co_anchor,
        feature_names=feature_names,
        train_years=sorted(by_year.keys()),
        in_sample_rmse_mw=float(best_rmse),
    )


def predict_shapes_for_year(model: SectorShapeModel, y: int) -> pd.DataFrame:
    cal = build_calendar_features_for_year(y)
    X, feature_names = build_feature_matrix(cal)

    if feature_names != model.feature_names:
        raise RuntimeError("[SHAPES] Feature-name mismatch between training and prediction.")

    phi_hh = softmax1d(X @ model.beta_hh)
    phi_co = softmax1d(X @ model.beta_co)

    out = cal[["year", "hour_of_year", "timestamp_utc"]].copy()
    out["phi_hh"] = phi_hh
    out["phi_co"] = phi_co
    return out


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    for p in [
        OUT_OBS_ANN_LOAD,
        OUT_FC_ANN_LOAD,
        OUT_OBS_SECTOR_TOTALS,
        OUT_FC_SECTOR_TOTALS,
        OUT_FC_HOURLY,
        OUT_DIAG,
        OUT_COEFS,
    ]:
        ensure_parent(p)

    print("=== Load historical national hourly load master ===")
    nat_hourly_hist = load_historical_national_hourly()

    print("=== Load national HH/CO annual mix ===")
    mix_df = load_sector_mix()

    # Historical annual national load from the hourly master.
    obs_annual = (
        nat_hourly_hist.groupby("year", as_index=False)["nat_load_MW"]
            .sum()
            .rename(columns={"nat_load_MW": "nat_annual_mwh_obs"})
            .sort_values("year")
            .reset_index(drop=True)
    )
    obs_annual["nat_annual_twh_obs"] = obs_annual["nat_annual_mwh_obs"] / 1e6
    obs_annual.to_csv(OUT_OBS_ANN_LOAD, index=False)
    print(f"[OK] Wrote observed annual national load: {OUT_OBS_ANN_LOAD}")

    print(f"=== Fit annual national load forecast ({ANNUAL_MODEL}) ===")
    ann_model = fit_annual_load_trend(obs_annual, model_type=ANNUAL_MODEL)
    years_fc = np.array(FORECAST_YEARS, dtype=int)
    nat_annual_fc_mwh = predict_annual_load_mwh(ann_model, years_fc)

    fc_annual = pd.DataFrame({
        "year": years_fc,
        "nat_annual_mwh_hat": nat_annual_fc_mwh,
    })
    fc_annual["nat_annual_twh_hat"] = fc_annual["nat_annual_mwh_hat"] / 1e6
    fc_annual.to_csv(OUT_FC_ANN_LOAD, index=False)
    print(f"[OK] Wrote forecast annual national load: {OUT_FC_ANN_LOAD}")

    print("=== Build observed overlap annual sector totals from omega ===")
    nat_overlap, mix_overlap = prepare_overlap_training_data(nat_hourly_hist, mix_df)
    mix_overlap = mix_overlap.copy()
    mix_overlap["nat_annual_twh_obs_overlap"] = mix_overlap["nat_annual_mwh_obs_overlap"] / 1e6
    mix_overlap.to_csv(OUT_OBS_SECTOR_TOTALS, index=False)
    print(f"[OK] Wrote observed overlap sector totals: {OUT_OBS_SECTOR_TOTALS}")

    print("=== Learn sector hourly shapes from historical national load ===")
    shape_model = fit_sector_shape_model(nat_overlap, mix_overlap)

    print("=== Build future national HH/CO annual totals ===")
    mix_fc = mix_df[mix_df["year"].between(FORECAST_START, FORECAST_END, inclusive="both")].copy()
    if len(mix_fc) != len(FORECAST_YEARS):
        missing_fc = sorted(set(FORECAST_YEARS) - set(mix_fc["year"].tolist()))
        raise ValueError(f"[FUTURE] Missing omega years for forecast horizon: {missing_fc}")

    fc_sector_totals = mix_fc.merge(fc_annual, on="year", how="inner", validate="one_to_one")
    fc_sector_totals["hh_total_mwh_hat"] = fc_sector_totals["omega_hh"] * fc_sector_totals["nat_annual_mwh_hat"]
    fc_sector_totals["co_total_mwh_hat"] = fc_sector_totals["omega_co"] * fc_sector_totals["nat_annual_mwh_hat"]
    fc_sector_totals.to_csv(OUT_FC_SECTOR_TOTALS, index=False)
    print(f"[OK] Wrote forecast national sector totals: {OUT_FC_SECTOR_TOTALS}")

    print("=== Construct future hourly national HH/CO load ===")
    pieces = []
    for y in FORECAST_YEARS:
        sh = predict_shapes_for_year(shape_model, y)
        row = fc_sector_totals.loc[fc_sector_totals["year"] == y]
        if row.empty:
            raise ValueError(f"[FUTURE] Missing annual sector totals for year {y}")

        E_hh = float(row["hh_total_mwh_hat"].iloc[0])
        E_co = float(row["co_total_mwh_hat"].iloc[0])

        sh["nat_hh_load_MW"] = sh["phi_hh"] * E_hh
        sh["nat_co_load_MW"] = sh["phi_co"] * E_co
        sh["nat_load_MW"] = sh["nat_hh_load_MW"] + sh["nat_co_load_MW"]

        # Exact annual consistency
        err_hh = abs(sh["nat_hh_load_MW"].sum() - E_hh)
        err_co = abs(sh["nat_co_load_MW"].sum() - E_co)
        err_tot = abs(sh["nat_load_MW"].sum() - (E_hh + E_co))
        if max(err_hh, err_co, err_tot) > 1e-6:
            raise ValueError(f"[FUTURE] Annual exactness failed for year {y}")

        pieces.append(sh[[
            "timestamp_utc", "year", "hour_of_year", "nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"
        ]].copy())

    nat_hourly_fc = pd.concat(pieces, ignore_index=True)
    nat_hourly_fc.to_csv(OUT_FC_HOURLY, index=False)
    print(f"[OK] Wrote learned future hourly national load: {OUT_FC_HOURLY}")

    coef = pd.DataFrame({
        "feature_name": shape_model.feature_names,
        "beta_hh": shape_model.beta_hh,
        "beta_co": shape_model.beta_co,
        "beta_hh_anchor": shape_model.beta_hh_anchor,
        "beta_co_anchor": shape_model.beta_co_anchor,
    })
    coef.to_csv(OUT_COEFS, index=False)
    print(f"[OK] Wrote shape coefficients: {OUT_COEFS}")

    diag = {
        "input_files": {
            "nat_hourly_history": str(NAT_HOURLY_HISTORY_FILE),
            "sector_mix_used": str(UNIFIED_MIX_FILE if UNIFIED_MIX_FILE.exists() else "legacy_files"),
        },
        "historical_years_available": sorted(nat_hourly_hist["year"].unique().tolist()),
        "overlap_years_used_for_shape_learning": shape_model.train_years,
        "forecast_years": FORECAST_YEARS,
        "annual_model": ANNUAL_MODEL,
        "annual_model_beta": ann_model.beta.tolist(),
        "shape_model": {
            "ridge_data": RIDGE_DATA,
            "ridge_anchor": RIDGE_ANCHOR,
            "max_iters": MAX_ITERS,
            "learning_rate": LR,
            "seed": SEED,
            "in_sample_rmse_mw": shape_model.in_sample_rmse_mw,
        },
        "future_annual_totals_summary": {
            "nat_annual_twh_hat_min": float(fc_annual["nat_annual_twh_hat"].min()),
            "nat_annual_twh_hat_max": float(fc_annual["nat_annual_twh_hat"].max()),
            "omega_hh_min": float(fc_sector_totals["omega_hh"].min()),
            "omega_hh_max": float(fc_sector_totals["omega_hh"].max()),
        },
        "notes": [
            "Historical annual totals are computed directly from the QC-clean hourly master.",
            "Overlap HH/CO annual totals are built by applying omega_hh/omega_co to historical national annual totals.",
            "HH/CO hourly shapes are identified only up to sector-anchor regularization because sectoral hourly metering is not observed directly.",
            "Default annual extrapolation is log-linear rather than quadratic to reduce unstable long-horizon behavior.",
        ],
    }
    with open(OUT_DIAG, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    print(f"[OK] Wrote diagnostics: {OUT_DIAG}")

    print("\n=== SUMMARY ===")
    print(f"Historical years used        : {sorted(nat_hourly_hist['year'].unique().tolist())}")
    print(f"Overlap years for shapes     : {shape_model.train_years}")
    print(f"Annual forecast model        : {ANNUAL_MODEL}")
    print(f"In-sample shape RMSE (MW)    : {shape_model.in_sample_rmse_mw:,.3f}")
    print("Future annual national load (TWh):")
    print(fc_annual[["year", "nat_annual_twh_hat"]].to_string(index=False))


if __name__ == "__main__":
    main()