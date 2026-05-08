#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_provincial_residual_load_from_learned_load_optionB.py

Goal
----
Build Dutch provincial residual load RL_MW for 2025–2035 using:

  1) Learned provincial hourly load:
       derived/provincial_hourly_load_2025_2035_learned.csv
     produced by build_provincial_hourly_load_from_learned_targets.py

  2) Historical Dutch hourly RES templates:
       time_series_profiles_2015_2020.csv
     used only for HOURLY SHAPES (UTC aligned by month/day/hour)

  3) National annual RES targets:
       derived/nl_res_targets_twh_2025_2035_v1.csv
     with columns:
       year, solar_twh, wind_on_twh, wind_off_twh

  4) Provincial RES allocation:
     - solar + wind_on: provincial shares from CBS 85004NED
       (freeze or trend mode)
     - wind_off: year-specific offshore landing shares

Residual load definition
------------------------
    RL_MW = prov_load_MW - prov_solar_MW - prov_wind_on_MW - prov_wind_off_MW

Important convention
--------------------
This updated script accepts either 0-based or 1-based hour_of_year on input,
standardizes internally to 1-based, and writes output in 1-based convention.

Main outputs
------------
- derived/provincial_residual_load_2025_2035_learned_optionB.csv
- derived/province_res_shares_2025_2035_learned_optionB.csv
- derived/provincial_residual_load_2025_2035_learned_optionB_qc_by_year.csv
- derived/provincial_residual_load_2025_2035_learned_optionB_diagnostics.json

Main output columns
-------------------
timestamp_utc, year, hour_of_year, province,
prov_hh_load_MW, prov_co_load_MW, prov_load_MW,
nat_hh_load_MW, nat_co_load_MW, nat_load_MW,
prov_solar_MW, prov_wind_on_MW, prov_wind_off_MW, prov_wind_MW,
nat_solar_MW, nat_wind_on_MW, nat_wind_off_MW, nat_wind_MW,
RL_MW, nat_rl_MW, RL_short_MW, RL_surplus_MW,
share_solar, share_wind_on, share_offshore, template_year_res
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
import sys
import unicodedata

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
PROV_LOAD_FILE = getattr(
    cfg,
    "PROVINCIAL_HOURLY_LOAD_LEARNED_PATH",
    cfg.DERIVED_DIR / "provincial_hourly_load_2025_2035_learned.csv",
)
NL_PROFILE_FILE = getattr(
    cfg,
    "TIME_SERIES_PROFILES_2015_2020_PATH",
    cfg.EXTERNAL_DIR / "time_series_profiles_2015_2020.csv",
)
RES_TARGETS_FILE = getattr(
    cfg,
    "NL_RES_TARGETS_PATH",
    cfg.DERIVED_DIR / "nl_res_targets_twh_2025_2035_v1.csv",
)
OFFSHORE_LANDING_FILE = getattr(
    cfg,
    "OFFSHORE_LANDING_PANEL_PATH",
    cfg.DERIVED_DIR / "panel_offshore_landing_cap_mw_2025_2035_v1.csv",
)

CBS_PROV_LONG_CANDIDATES = [
    getattr(
        cfg,
        "CBS_85004_PROVINCE_LONG_PATH",
        cfg.DERIVED_DIR / "cbs_85004NED_province_long_2018_2024_v1.csv",
    )
]

OUT_FILE = getattr(
    cfg,
    "PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_PATH",
    cfg.DERIVED_DIR / "provincial_residual_load_2025_2035_learned_optionB.csv",
)
OUT_SHARES_FILE = getattr(
    cfg,
    "PROVINCE_RES_SHARES_LEARNED_OPTIONB_PATH",
    cfg.DERIVED_DIR / "province_res_shares_2025_2035_learned_optionB.csv",
)
OUT_QC_FILE = getattr(
    cfg,
    "PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_QC_PATH",
    cfg.DERIVED_DIR / "provincial_residual_load_2025_2035_learned_optionB_qc_by_year.csv",
)
OUT_DIAG_FILE = getattr(
    cfg,
    "PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_DIAG_PATH",
    cfg.DERIVED_DIR / "provincial_residual_load_2025_2035_learned_optionB_diagnostics.json",
)

FUTURE_START = 2025
FUTURE_END = 2035
FUTURE_YEARS = list(range(FUTURE_START, FUTURE_END + 1))

PROVINCES_12 = [
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

# Historical template QC
MAX_MISS_SOLAR = 0.02
MAX_MISS_WIND = 0.02

# Generation hygiene
CLIP_NEGATIVE_GEN_TO_ZERO = True

# Provincial onshore RES share forecast mode
# recommended conservative default: "freeze"
# alternative: "trend"
ONSHORE_SHARE_MODE = "freeze"
ONSHORE_SHARE_HISTORY_START = 2018
ONSHORE_SHARE_FREEZE_YEAR = 2024

# Deterministic additive-log-ratio share trend
SHARE_RIDGE_L2 = 1e-3
SHARE_EPS = 1e-9

# Tolerance
TOL = 1e-6


# =============================================================================
# HELPERS
# =============================================================================
PV_CODE_TO_NAME: Dict[str, str] = {
    "PV20": "Groningen",
    "PV21": "Friesland",
    "PV22": "Drenthe",
    "PV23": "Overijssel",
    "PV24": "Flevoland",
    "PV25": "Gelderland",
    "PV26": "Utrecht",
    "PV27": "Noord-Holland",
    "PV28": "Zuid-Holland",
    "PV29": "Zeeland",
    "PV30": "Noord-Brabant",
    "PV31": "Limburg",
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


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


def annual_twh_from_mw_hourly(x_mw: np.ndarray) -> float:
    return float(np.nansum(x_mw) / 1e6)


def max_abs_rel(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = np.abs(a - b)
    max_abs = float(np.nanmax(diff))
    denom = np.maximum(np.abs(b), 1e-9)
    max_rel = float(np.nanmax(diff / denom))
    return max_abs, max_rel


def fill_series_linear_ffill_bfill(x: pd.Series) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce").astype(float)
    y = y.interpolate(method="linear", limit_direction="both")
    return y.ffill().bfill()


def _norm_text(x: object) -> str:
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def normalize_province_name(x: object) -> Optional[str]:
    if pd.isna(x):
        return None
    s0 = str(x).strip().replace('"', "")
    s = _norm_text(s0).replace("(pv)", "").strip()

    mapping = {
        "groningen": "Groningen",
        "friesland": "Friesland",
        "fryslan": "Friesland",
        "drenthe": "Drenthe",
        "overijssel": "Overijssel",
        "flevoland": "Flevoland",
        "gelderland": "Gelderland",
        "utrecht": "Utrecht",
        "noord holland": "Noord-Holland",
        "noord-holland": "Noord-Holland",
        "zuid holland": "Zuid-Holland",
        "zuid-holland": "Zuid-Holland",
        "zeeland": "Zeeland",
        "noord brabant": "Noord-Brabant",
        "noord-brabant": "Noord-Brabant",
        "limburg": "Limburg",
    }

    if s0.strip().upper() in PV_CODE_TO_NAME:
        return PV_CODE_TO_NAME[s0.strip().upper()]
    if s in mapping:
        return mapping[s]

    for k, v in mapping.items():
        if k in s:
            return v
    return None


def normalize_hour_index_to_1_based(df: pd.DataFrame, year_col: str, hour_col: str, label: str) -> pd.DataFrame:
    df = df.copy()
    df[year_col] = to_num(df[year_col]).astype(int)
    df[hour_col] = to_num(df[hour_col]).astype(int)

    parts = []
    for y, g in df.groupby(year_col, sort=False):
        H = hours_in_year_utc(int(y))
        mn = int(g[hour_col].min())
        mx = int(g[hour_col].max())
        nunq = int(g[hour_col].nunique())
        gg = g.copy()

        if mn == 0 and mx == H - 1 and nunq == H:
            gg[hour_col] = gg[hour_col] + 1
        elif mn == 1 and mx == H and nunq == H:
            pass
        else:
            raise ValueError(
                f"[{label}] year={y}: hour index not recognized. "
                f"min={mn}, max={mx}, nunique={nunq}, expected 0..{H-1} or 1..{H}."
            )
        parts.append(gg)

    return pd.concat(parts, ignore_index=True)


def choose_template_years(available_years: List[int], future_years: List[int]) -> Dict[int, int]:
    avail = sorted(available_years)
    avail_leap = [y for y in avail if hours_in_year_utc(y) == 8784]
    avail_non = [y for y in avail if hours_in_year_utc(y) == 8760]

    out: Dict[int, int] = {}
    i_leap = 0
    i_non = 0

    for fy in future_years:
        H = hours_in_year_utc(fy)
        if H == 8784 and avail_leap:
            out[fy] = avail_leap[i_leap % len(avail_leap)]
            i_leap += 1
        elif H == 8760 and avail_non:
            out[fy] = avail_non[i_non % len(avail_non)]
            i_non += 1
        else:
            out[fy] = avail[(fy - future_years[0]) % len(avail)]
    return out


def align_by_utc_mdh(template_ts_utc: pd.DatetimeIndex, template_vals: np.ndarray, target_year: int) -> np.ndarray:
    H = hours_in_year_utc(target_year)
    t0 = pd.Timestamp(f"{target_year}-01-01T00:00:00Z")
    target_ts = pd.date_range(t0, periods=H, freq="h")

    tpl = pd.DataFrame({
        "m": template_ts_utc.month,
        "d": template_ts_utc.day,
        "h": template_ts_utc.hour,
        "val": template_vals,
    }).groupby(["m", "d", "h"], as_index=False)["val"].mean()

    tgt = pd.DataFrame({
        "m": target_ts.month,
        "d": target_ts.day,
        "h": target_ts.hour,
    })

    tpl_has_229 = ((tpl["m"] == 2) & (tpl["d"] == 29)).any()
    tgt_has_229 = ((tgt["m"] == 2) & (tgt["d"] == 29)).any()

    if tgt_has_229 and not tpl_has_229:
        tgt.loc[(tgt["m"] == 2) & (tgt["d"] == 29), "d"] = 28

    merged = tgt.merge(tpl, on=["m", "d", "h"], how="left")
    if merged["val"].isna().any():
        miss = merged.loc[merged["val"].isna(), ["m", "d", "h"]].head(20)
        raise ValueError("[ALIGN] Missing values after UTC (m,d,h) merge:\n" + miss.to_string(index=False))

    return merged["val"].to_numpy(dtype=float)


# =============================================================================
# DETERMINISTIC SOFTMAX / ALR SHARE TREND
# =============================================================================
def fit_alr_linear_share_model(
    years: np.ndarray,
    shares: np.ndarray,
    ridge_l2: float,
) -> Dict[str, object]:
    """
    Deterministic additive-log-ratio linear trend model:
        s_p(y) = softmax(alpha_p + beta_p * (y - y0))
    """
    years = years.astype(int)
    y0 = int(years.min())
    t = years.astype(float) - y0

    S = np.array(shares, dtype=float)
    S = np.clip(S, SHARE_EPS, 1.0)
    S = S / S.sum(axis=1, keepdims=True)

    K = S.shape[1]
    ref_idx = K - 1
    nonref_idx = [i for i in range(K) if i != ref_idx]

    Y = np.log(S[:, nonref_idx] / S[:, [ref_idx]])
    X = np.column_stack([np.ones_like(t), t])

    XtX = X.T @ X
    reg = ridge_l2 * np.eye(X.shape[1])
    B = np.linalg.solve(XtX + reg, X.T @ Y)

    alpha_nonref = B[0, :]
    beta_nonref = B[1, :]

    alpha = np.zeros(K, dtype=float)
    beta = np.zeros(K, dtype=float)
    alpha[nonref_idx] = alpha_nonref
    beta[nonref_idx] = beta_nonref

    return {
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "y0": y0,
    }


def predict_alr_linear_share_model(model: Dict[str, object], years_pred: np.ndarray) -> np.ndarray:
    years_pred = years_pred.astype(int)
    alpha = np.array(model["alpha"], dtype=float)
    beta = np.array(model["beta"], dtype=float)
    y0 = int(model["y0"])

    rows = []
    for y in years_pred:
        z = alpha + beta * (y - y0)
        z = z - np.max(z)
        ez = np.exp(z)
        s = ez / ez.sum()
        rows.append(s)
    return np.vstack(rows)


# =============================================================================
# 1) LEARNED PROVINCIAL HOURLY LOAD
# =============================================================================
def load_prov_load() -> pd.DataFrame:
    if not PROV_LOAD_FILE.exists():
        raise FileNotFoundError(f"Missing PROV_LOAD_FILE:\n  {PROV_LOAD_FILE}")

    prov = pd.read_csv(PROV_LOAD_FILE)
    prov.columns = [c.strip() for c in prov.columns]

    req = {
        "year",
        "hour_of_year",
        "province",
        "prov_hh_load_MW",
        "prov_co_load_MW",
        "prov_load_MW",
    }
    miss = sorted(req - set(prov.columns))
    if miss:
        raise ValueError(f"[PROV_LOAD] Missing columns: {miss}. Has: {list(prov.columns)}")

    ts_col = pick_first_existing(
        prov.columns.tolist(),
        ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc"],
    )

    prov["year"] = to_num(prov["year"]).astype(int)
    prov["province"] = prov["province"].astype(str).str.strip()
    prov["prov_hh_load_MW"] = to_num(prov["prov_hh_load_MW"]).astype(float)
    prov["prov_co_load_MW"] = to_num(prov["prov_co_load_MW"]).astype(float)
    prov["prov_load_MW"] = to_num(prov["prov_load_MW"]).astype(float)

    if ts_col is not None:
        prov["timestamp_utc_declared"] = pd.to_datetime(prov[ts_col], utc=True, errors="coerce")

    for c in ["nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"]:
        if c in prov.columns:
            prov[c] = to_num(prov[c]).astype(float)

    prov = prov[prov["year"].isin(FUTURE_YEARS)].copy()
    prov = prov[prov["province"].isin(PROVINCES_12)].copy()

    prov = normalize_hour_index_to_1_based(prov, "year", "hour_of_year", "PROV_LOAD")

    # Rebuild expected UTC timestamps from standardized 1-based hour_of_year
    prov["timestamp_utc"] = (
        pd.to_datetime(prov["year"].astype(str) + "-01-01T00:00:00Z", utc=True)
        + pd.to_timedelta(prov["hour_of_year"] - 1, unit="h")
    )

    if "timestamp_utc_declared" in prov.columns:
        bad_ts = prov["timestamp_utc_declared"].notna() & (prov["timestamp_utc_declared"] != prov["timestamp_utc"])
        if bad_ts.any():
            sample = prov.loc[bad_ts, ["year", "hour_of_year", "province", "timestamp_utc_declared", "timestamp_utc"]].head(20)
            raise ValueError("[PROV_LOAD] Declared timestamp_utc does not match year/hour_of_year.\n" + sample.to_string(index=False))
        prov = prov.drop(columns=["timestamp_utc_declared"])

    # Row-wise sector additivity
    row_gap = np.abs(prov["prov_load_MW"] - (prov["prov_hh_load_MW"] + prov["prov_co_load_MW"]))
    max_row_gap = float(row_gap.max())
    if max_row_gap > TOL:
        raise ValueError(f"[PROV_LOAD] Row-wise prov_load_MW != prov_hh_load_MW + prov_co_load_MW (max gap {max_row_gap}).")

    # Strict hourly province coverage
    for y in FUTURE_YEARS:
        H = hours_in_year_utc(y)
        sub = prov.loc[prov["year"] == y].copy()
        if sub.empty:
            raise ValueError(f"[PROV_LOAD] Missing year {y}.")

        counts = sub.groupby("hour_of_year")["province"].nunique()
        bad_hours = counts.loc[counts != len(PROVINCES_12)]
        if not bad_hours.empty:
            raise ValueError(f"[PROV_LOAD] year={y}: hours with !=12 provinces:\n{bad_hours.head(30).to_string()}")

        if len(sub) != H * len(PROVINCES_12):
            raise ValueError(f"[PROV_LOAD] year={y}: rows={len(sub)} expected={H * len(PROVINCES_12)}")

    # Reconstruct national hourly series exactly from provincial sums
    nat = (
        prov.groupby(["year", "hour_of_year", "timestamp_utc"], as_index=False)[
            ["prov_hh_load_MW", "prov_co_load_MW", "prov_load_MW"]
        ]
        .sum()
        .rename(columns={
            "prov_hh_load_MW": "nat_hh_load_MW_recon",
            "prov_co_load_MW": "nat_co_load_MW_recon",
            "prov_load_MW": "nat_load_MW_recon",
        })
        .sort_values(["year", "hour_of_year"])
        .reset_index(drop=True)
    )

    # If declared nat_* columns exist, verify them
    declared_cols = [c for c in ["nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"] if c in prov.columns]
    if declared_cols:
        nat_decl = (
            prov.groupby(["year", "hour_of_year"], as_index=False)[declared_cols]
            .first()
            .sort_values(["year", "hour_of_year"])
            .reset_index(drop=True)
        )
        chk = nat.merge(nat_decl, on=["year", "hour_of_year"], how="left")

        if "nat_hh_load_MW" in chk.columns:
            a, r = max_abs_rel(chk["nat_hh_load_MW_recon"].to_numpy(), chk["nat_hh_load_MW"].to_numpy())
            if a > TOL:
                raise ValueError(f"[PROV_LOAD] nat_hh_load_MW mismatch: max abs {a:.3e}, max rel {r:.3e}")
        if "nat_co_load_MW" in chk.columns:
            a, r = max_abs_rel(chk["nat_co_load_MW_recon"].to_numpy(), chk["nat_co_load_MW"].to_numpy())
            if a > TOL:
                raise ValueError(f"[PROV_LOAD] nat_co_load_MW mismatch: max abs {a:.3e}, max rel {r:.3e}")
        if "nat_load_MW" in chk.columns:
            a, r = max_abs_rel(chk["nat_load_MW_recon"].to_numpy(), chk["nat_load_MW"].to_numpy())
            if a > TOL:
                raise ValueError(f"[PROV_LOAD] nat_load_MW mismatch: max abs {a:.3e}, max rel {r:.3e}")

    nat = nat.rename(columns={
        "nat_hh_load_MW_recon": "nat_hh_load_MW",
        "nat_co_load_MW_recon": "nat_co_load_MW",
        "nat_load_MW_recon": "nat_load_MW",
    })

    prov = prov.drop(columns=[c for c in ["nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"] if c in prov.columns])
    prov = prov.merge(nat, on=["year", "hour_of_year", "timestamp_utc"], how="left", validate="many_to_one")

    return prov.sort_values(["year", "hour_of_year", "province"]).reset_index(drop=True)


# =============================================================================
# 2) HISTORICAL NL RES HOURLY TEMPLATES
# =============================================================================
def load_nl_templates() -> Tuple[Dict[int, Dict[str, np.ndarray]], str]:
    if not NL_PROFILE_FILE.exists():
        raise FileNotFoundError(f"Missing NL_PROFILE_FILE:\n  {NL_PROFILE_FILE}")

    df = pd.read_csv(NL_PROFILE_FILE)
    df.columns = [c.strip() for c in df.columns]

    ts_col = pick_first_existing(
        df.columns.tolist(),
        ["utc_timestamp", "timestamp_utc", "utc_time", "time_utc"],
    )
    if ts_col is None:
        raise ValueError(f"[NL_PROFILE] Need a UTC timestamp column. Has: {list(df.columns)}")

    solar_col = "NL_solar_generation_actual"
    if solar_col not in df.columns:
        raise ValueError("[NL_PROFILE] Missing NL_solar_generation_actual.")

    wind_tot_col = "NL_wind_generation_actual" if "NL_wind_generation_actual" in df.columns else None
    wind_on_col = "NL_wind_onshore_generation_actual" if "NL_wind_onshore_generation_actual" in df.columns else None
    wind_off_col = "NL_wind_offshore_generation_actual" if "NL_wind_offshore_generation_actual" in df.columns else None

    if wind_on_col is not None and wind_off_col is not None:
        wind_mode = "split"
    elif wind_tot_col is not None:
        wind_mode = "total"
    elif wind_on_col is not None:
        wind_mode = "on_only"
    else:
        raise ValueError("[NL_PROFILE] Missing wind columns (need split, total, or on_only).")

    df["ts_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"]).copy()
    df["year"] = df["ts_utc"].dt.year.astype(int)

    for c in [solar_col, wind_tot_col, wind_on_col, wind_off_col]:
        if c is not None and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if CLIP_NEGATIVE_GEN_TO_ZERO:
        df[solar_col] = df[solar_col].clip(lower=0.0)
        if wind_tot_col is not None:
            df[wind_tot_col] = df[wind_tot_col].clip(lower=0.0)
        if wind_on_col is not None:
            df[wind_on_col] = df[wind_on_col].clip(lower=0.0)
        if wind_off_col is not None:
            df[wind_off_col] = df[wind_off_col].clip(lower=0.0)

    templates: Dict[int, Dict[str, np.ndarray]] = {}

    for y in sorted(df["year"].unique().tolist()):
        idx = pd.date_range(
            pd.Timestamp(f"{y}-01-01T00:00:00Z"),
            pd.Timestamp(f"{y+1}-01-01T00:00:00Z"),
            freq="h",
            inclusive="left",
        )

        cols = [solar_col]
        if wind_mode == "split":
            cols += [wind_on_col, wind_off_col]
        elif wind_mode == "total":
            cols += [wind_tot_col]
        else:
            cols += [wind_on_col]

        sub = (
            df.loc[df["year"] == y, ["ts_utc"] + cols]
            .groupby("ts_utc", as_index=False)[cols]
            .mean()
            .sort_values("ts_utc")
            .reset_index(drop=True)
        )

        base = sub.set_index("ts_utc").reindex(idx)

        if float(base[solar_col].isna().mean()) > MAX_MISS_SOLAR:
            continue

        if wind_mode == "split":
            if float(base[wind_on_col].isna().mean()) > MAX_MISS_WIND:
                continue
            if float(base[wind_off_col].isna().mean()) > MAX_MISS_WIND:
                continue
        elif wind_mode == "total":
            if float(base[wind_tot_col].isna().mean()) > MAX_MISS_WIND:
                continue
        else:
            if float(base[wind_on_col].isna().mean()) > MAX_MISS_WIND:
                continue

        base[solar_col] = fill_series_linear_ffill_bfill(base[solar_col])

        wind_total = wind_on = wind_off = None
        if wind_mode == "split":
            base[wind_on_col] = fill_series_linear_ffill_bfill(base[wind_on_col])
            base[wind_off_col] = fill_series_linear_ffill_bfill(base[wind_off_col])
            wind_on = base[wind_on_col].to_numpy(dtype=float)
            wind_off = base[wind_off_col].to_numpy(dtype=float)
        elif wind_mode == "total":
            base[wind_tot_col] = fill_series_linear_ffill_bfill(base[wind_tot_col])
            wind_total = base[wind_tot_col].to_numpy(dtype=float)
        else:
            base[wind_on_col] = fill_series_linear_ffill_bfill(base[wind_on_col])
            wind_on = base[wind_on_col].to_numpy(dtype=float)

        templates[int(y)] = {
            "ts_utc": pd.DatetimeIndex(idx),
            "solar": base[solar_col].to_numpy(dtype=float),
            "wind_total": wind_total,
            "wind_on": wind_on,
            "wind_off": wind_off,
        }

    if not templates:
        raise ValueError("[NL_PROFILE] No usable template years survived QC.")

    print(f"[NL_PROFILE] template years used: {sorted(templates.keys())} (wind_mode={wind_mode})")
    return templates, wind_mode


# =============================================================================
# 3) NATIONAL ANNUAL RES TARGETS
# =============================================================================
def load_res_targets() -> pd.DataFrame:
    if not RES_TARGETS_FILE.exists():
        raise FileNotFoundError(f"Missing RES_TARGETS_FILE:\n  {RES_TARGETS_FILE}")

    t = pd.read_csv(RES_TARGETS_FILE)
    t.columns = [c.strip() for c in t.columns]

    req = {"year", "solar_twh", "wind_on_twh", "wind_off_twh"}
    miss = sorted(req - set(t.columns))
    if miss:
        raise ValueError(f"[RES_TARGETS] Missing columns: {miss}. Has: {list(t.columns)}")

    t["year"] = to_num(t["year"]).astype(int)
    for c in ["solar_twh", "wind_on_twh", "wind_off_twh"]:
        t[c] = to_num(t[c]).astype(float)

    t = t[t["year"].isin(FUTURE_YEARS)].copy()
    missing_years = sorted(set(FUTURE_YEARS) - set(t["year"].unique().tolist()))
    if missing_years:
        raise ValueError(f"[RES_TARGETS] Missing target rows for years: {missing_years}")

    return t.sort_values("year").reset_index(drop=True)


# =============================================================================
# 4) CBS ON/OFFSHORE SHARES
# =============================================================================
def load_cbs_onshore_capacity_history() -> pd.DataFrame:
    path = first_existing_path(CBS_PROV_LONG_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            "Could not find the CBS 85004NED provincial long file in any candidate path:\n"
            + "\n".join([f"  - {p}" for p in CBS_PROV_LONG_CANDIDATES])
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    tech_col = pick_first_existing(
        df.columns.tolist(),
        ["BronEnTechniek_label", "BronEnTechniek", "technology_label", "technology", "tech_label"],
    )
    if tech_col is None:
        raise ValueError(f"[CBS85004] Missing technology label column. Has: {list(df.columns)}")

    cap_col = pick_first_existing(
        df.columns.tolist(),
        [
            "OpgesteldElektrischVermogen_MW",
            "OpgesteldElektrischVermogen_1",
            "OpgesteldElektrischVermogen",
            "capacity_mw",
            "cap_mw",
        ],
    )
    if cap_col is None:
        raise ValueError(f"[CBS85004] Missing capacity column. Has: {list(df.columns)}")

    year_col = pick_first_existing(df.columns.tolist(), ["year", "Year", "jaar", "Jaar"])
    if year_col is None:
        raise ValueError(f"[CBS85004] Missing year column. Has: {list(df.columns)}")

    if "province" in df.columns:
        df["province"] = df["province"].map(normalize_province_name)
    elif "RegioS_label" in df.columns:
        df["province"] = df["RegioS_label"].map(normalize_province_name)
    elif "RegioS" in df.columns:
        df["province"] = df["RegioS"].astype(str).str.strip().str.upper().map(PV_CODE_TO_NAME)
    else:
        raise ValueError(f"[CBS85004] Missing province identifier. Has: {list(df.columns)}")

    df["year"] = to_num(df[year_col]).astype("Int64")
    df[cap_col] = to_num(df[cap_col]).astype(float)
    df["tech_label"] = df[tech_col].astype(str).map(_norm_text)

    df = df.dropna(subset=["year", "province", cap_col]).copy()
    df["year"] = df["year"].astype(int)
    df = df[df["province"].isin(PROVINCES_12)].copy()
    df = df[
        (df["year"] >= ONSHORE_SHARE_HISTORY_START)
        & (df["year"] <= ONSHORE_SHARE_FREEZE_YEAR)
    ].copy()

    if df.empty:
        raise ValueError("[CBS85004] No usable province/year rows after filtering.")

    is_solar = (
        df["tech_label"].str.contains("zon", na=False)
        | df["tech_label"].str.contains("zonnestroom", na=False)
        | df["tech_label"].str.contains(r"\bpv\b", na=False)
        | df["tech_label"].str.contains("solar", na=False)
    )
    is_wind = df["tech_label"].str.contains("wind", na=False)
    is_offshore_word = df["tech_label"].str.contains("zee|offshore", na=False)
    is_wind_on = is_wind & (~is_offshore_word)

    solar = (
        df.loc[is_solar]
        .groupby(["year", "province"], as_index=False)[cap_col]
        .sum()
        .rename(columns={cap_col: "solar_cap_mw"})
    )
    wind_on = (
        df.loc[is_wind_on]
        .groupby(["year", "province"], as_index=False)[cap_col]
        .sum()
        .rename(columns={cap_col: "wind_on_cap_mw"})
    )

    years = sorted(df["year"].unique().tolist())
    panel = pd.MultiIndex.from_product([years, PROVINCES_12], names=["year", "province"]).to_frame(index=False)
    panel = panel.merge(solar, on=["year", "province"], how="left")
    panel = panel.merge(wind_on, on=["year", "province"], how="left")
    panel["solar_cap_mw"] = panel["solar_cap_mw"].fillna(0.0)
    panel["wind_on_cap_mw"] = panel["wind_on_cap_mw"].fillna(0.0)

    solar_tot = panel.groupby("year")["solar_cap_mw"].transform("sum")
    wind_tot = panel.groupby("year")["wind_on_cap_mw"].transform("sum")

    panel["share_solar_obs"] = np.where(solar_tot > 0, panel["solar_cap_mw"] / solar_tot, 1.0 / len(PROVINCES_12))
    panel["share_wind_on_obs"] = np.where(wind_tot > 0, panel["wind_on_cap_mw"] / wind_tot, 1.0 / len(PROVINCES_12))

    return panel.sort_values(["year", "province"]).reset_index(drop=True)


def forecast_onshore_shares(obs_panel: pd.DataFrame) -> pd.DataFrame:
    years_fc = np.array(FUTURE_YEARS, dtype=int)
    years_tr = np.array(sorted(obs_panel["year"].unique().tolist()), dtype=int)

    mode = ONSHORE_SHARE_MODE.lower().strip()
    if mode not in {"freeze", "trend"}:
        raise ValueError("ONSHORE_SHARE_MODE must be one of {'freeze', 'trend'}.")

    if mode == "freeze":
        freeze_candidates = [y for y in years_tr.tolist() if y <= ONSHORE_SHARE_FREEZE_YEAR]
        if not freeze_candidates:
            freeze_year = int(years_tr.max())
        else:
            freeze_year = int(max(freeze_candidates))

        fr = obs_panel.loc[
            obs_panel["year"] == freeze_year,
            ["province", "share_solar_obs", "share_wind_on_obs"],
        ].copy()

        rows = []
        for y in FUTURE_YEARS:
            tmp = fr.copy()
            tmp["year"] = y
            tmp = tmp.rename(columns={
                "share_solar_obs": "share_solar",
                "share_wind_on_obs": "share_wind_on",
            })
            rows.append(tmp[["year", "province", "share_solar", "share_wind_on"]])

        out = pd.concat(rows, ignore_index=True)
        return out.sort_values(["year", "province"]).reset_index(drop=True)

    # trend mode
    mat_solar = (
        obs_panel.pivot(index="year", columns="province", values="share_solar_obs")
        .reindex(index=years_tr, columns=PROVINCES_12)
        .to_numpy(dtype=float)
    )
    mat_wind = (
        obs_panel.pivot(index="year", columns="province", values="share_wind_on_obs")
        .reindex(index=years_tr, columns=PROVINCES_12)
        .to_numpy(dtype=float)
    )

    mat_solar = np.where(np.isnan(mat_solar), 0.0, mat_solar)
    mat_wind = np.where(np.isnan(mat_wind), 0.0, mat_wind)

    mat_solar = np.where(mat_solar.sum(axis=1, keepdims=True) > 0, mat_solar / mat_solar.sum(axis=1, keepdims=True), 1.0 / len(PROVINCES_12))
    mat_wind = np.where(mat_wind.sum(axis=1, keepdims=True) > 0, mat_wind / mat_wind.sum(axis=1, keepdims=True), 1.0 / len(PROVINCES_12))

    print("[RES SHARES] fitting deterministic ALR trend for solar shares...")
    model_solar = fit_alr_linear_share_model(years_tr, mat_solar, SHARE_RIDGE_L2)

    print("[RES SHARES] fitting deterministic ALR trend for onshore wind shares...")
    model_wind = fit_alr_linear_share_model(years_tr, mat_wind, SHARE_RIDGE_L2)

    solar_fc = predict_alr_linear_share_model(model_solar, years_fc)
    wind_fc = predict_alr_linear_share_model(model_wind, years_fc)

    rows = []
    for i, y in enumerate(years_fc):
        for j, p in enumerate(PROVINCES_12):
            rows.append({
                "year": int(y),
                "province": p,
                "share_solar": float(solar_fc[i, j]),
                "share_wind_on": float(wind_fc[i, j]),
            })

    out = pd.DataFrame(rows).sort_values(["year", "province"]).reset_index(drop=True)

    # Re-normalize yearwise to remove tiny numerical drift
    for col in ["share_solar", "share_wind_on"]:
        gsum = out.groupby("year")[col].transform("sum")
        out[col] = np.where(gsum > 0, out[col] / gsum, 1.0 / len(PROVINCES_12))

    return out


def load_offshore_landing_shares() -> pd.DataFrame:
    if not OFFSHORE_LANDING_FILE.exists():
        raise FileNotFoundError(f"Missing OFFSHORE_LANDING_FILE:\n  {OFFSHORE_LANDING_FILE}")

    df = pd.read_csv(OFFSHORE_LANDING_FILE)
    df.columns = [c.strip() for c in df.columns]

    if "share_offshore" in df.columns:
        req = {"year", "province", "share_offshore"}
        miss = sorted(req - set(df.columns))
        if miss:
            raise ValueError(f"[OFFSHORE] Missing columns: {miss}. Has: {list(df.columns)}")

        df["year"] = to_num(df["year"]).astype(int)
        df["province"] = df["province"].map(normalize_province_name)
        df["share_offshore"] = to_num(df["share_offshore"]).astype(float)

        df = df[df["year"].isin(FUTURE_YEARS)].copy()
        df = df[df["province"].isin(PROVINCES_12)].copy()

        panel = pd.MultiIndex.from_product([FUTURE_YEARS, PROVINCES_12], names=["year", "province"]).to_frame(index=False)
        df = panel.merge(df[["year", "province", "share_offshore"]], on=["year", "province"], how="left")
        df["share_offshore"] = df["share_offshore"].fillna(0.0)

        tot = df.groupby("year")["share_offshore"].transform("sum")
        df["share_offshore"] = np.where(tot > 0, df["share_offshore"] / tot, 1.0 / len(PROVINCES_12))
        return df.sort_values(["year", "province"]).reset_index(drop=True)

    cap_col = pick_first_existing(
        df.columns.tolist(),
        ["landing_cap_mw", "landing_cap_MW", "offshore_landing_cap_mw", "cap_mw"],
    )
    if cap_col is None:
        raise ValueError(f"[OFFSHORE] Need landing capacity column. Has: {list(df.columns)}")

    req = {"year", "province"}
    miss = sorted(req - set(df.columns))
    if miss:
        raise ValueError(f"[OFFSHORE] Missing columns: {miss}. Has: {list(df.columns)}")

    df["year"] = to_num(df["year"]).astype(int)
    df["province"] = df["province"].map(normalize_province_name)
    df[cap_col] = to_num(df[cap_col]).astype(float)

    df = df[df["year"].isin(FUTURE_YEARS)].copy()
    df = df[df["province"].isin(PROVINCES_12)].copy()

    panel = pd.MultiIndex.from_product([FUTURE_YEARS, PROVINCES_12], names=["year", "province"]).to_frame(index=False)
    df = panel.merge(df[["year", "province", cap_col]], on=["year", "province"], how="left")
    df[cap_col] = df[cap_col].fillna(0.0)

    tot = df.groupby("year")[cap_col].transform("sum")
    df["share_offshore"] = np.where(tot > 0, df[cap_col] / tot, 1.0 / len(PROVINCES_12))

    return df[["year", "province", "share_offshore"]].sort_values(["year", "province"]).reset_index(drop=True)


def build_provincial_res_share_panel() -> pd.DataFrame:
    obs_on = load_cbs_onshore_capacity_history()
    fc_on = forecast_onshore_shares(obs_on)
    off = load_offshore_landing_shares()

    out = fc_on.merge(off, on=["year", "province"], how="inner", validate="one_to_one")

    chk = out.groupby("year")[["share_solar", "share_wind_on", "share_offshore"]].sum()
    bad = chk[
        ((chk["share_solar"] - 1.0).abs() > TOL)
        | ((chk["share_wind_on"] - 1.0).abs() > TOL)
        | ((chk["share_offshore"] - 1.0).abs() > TOL)
    ]
    if not bad.empty:
        raise ValueError(f"[RES SHARES] Sums not equal to 1:\n{bad.to_string()}")

    return out.sort_values(["year", "province"]).reset_index(drop=True)


# =============================================================================
# 5) BUILD NATIONAL HOURLY RES FROM TARGETS + TEMPLATES
# =============================================================================
def build_future_national_res(
    templates: Dict[int, Dict[str, np.ndarray]],
    wind_mode: str,
    prov_load: pd.DataFrame,
    targets: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    assign = choose_template_years(sorted(templates.keys()), FUTURE_YEARS)
    targets = targets.set_index("year")

    rows = []
    for y in FUTURE_YEARS:
        H = hours_in_year_utc(y)

        nat_load = (
            prov_load.loc[prov_load["year"] == y]
            .groupby(["year", "hour_of_year", "timestamp_utc"], as_index=False)[
                ["nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"]
            ]
            .first()
            .sort_values("hour_of_year")
            .reset_index(drop=True)
        )
        if len(nat_load) != H:
            raise ValueError(f"[NAT LOAD] year={y}: rows={len(nat_load)} expected={H}")

        t = assign[y]
        tpl = templates[t]

        solar_tpl = np.maximum(tpl["solar"], 0.0) if CLIP_NEGATIVE_GEN_TO_ZERO else tpl["solar"]
        solar_aligned = align_by_utc_mdh(tpl["ts_utc"], solar_tpl, target_year=y)
        ssum = float(np.sum(solar_aligned))
        solar_w = (solar_aligned / ssum) if ssum > 0 else (np.ones(H) / H)

        solar_twh = float(targets.loc[y, "solar_twh"])
        wind_on_twh = float(targets.loc[y, "wind_on_twh"])
        wind_off_twh = float(targets.loc[y, "wind_off_twh"])

        nat_solar_MW = solar_w * (solar_twh * 1e6)

        if wind_mode == "split":
            w_on_tpl = tpl["wind_on"]
            w_off_tpl = tpl["wind_off"]
            w_on_tpl = np.maximum(w_on_tpl, 0.0) if CLIP_NEGATIVE_GEN_TO_ZERO else w_on_tpl
            w_off_tpl = np.maximum(w_off_tpl, 0.0) if CLIP_NEGATIVE_GEN_TO_ZERO else w_off_tpl

            w_on_aligned = align_by_utc_mdh(tpl["ts_utc"], w_on_tpl, target_year=y)
            w_off_aligned = align_by_utc_mdh(tpl["ts_utc"], w_off_tpl, target_year=y)

            s_on = float(np.sum(w_on_aligned))
            s_off = float(np.sum(w_off_aligned))

            w_on = (w_on_aligned / s_on) if s_on > 0 else (np.ones(H) / H)
            w_off = (w_off_aligned / s_off) if s_off > 0 else (np.ones(H) / H)

            nat_wind_on_MW = w_on * (wind_on_twh * 1e6)
            nat_wind_off_MW = w_off * (wind_off_twh * 1e6)

        elif wind_mode == "total":
            w_tot_tpl = tpl["wind_total"]
            w_tot_tpl = np.maximum(w_tot_tpl, 0.0) if CLIP_NEGATIVE_GEN_TO_ZERO else w_tot_tpl
            w_tot_aligned = align_by_utc_mdh(tpl["ts_utc"], w_tot_tpl, target_year=y)
            s_tot = float(np.sum(w_tot_aligned))
            w_tot = (w_tot_aligned / s_tot) if s_tot > 0 else (np.ones(H) / H)

            nat_wind_on_MW = w_tot * (wind_on_twh * 1e6)
            nat_wind_off_MW = w_tot * (wind_off_twh * 1e6)

        else:  # on_only fallback
            w_on_tpl = tpl["wind_on"]
            w_on_tpl = np.maximum(w_on_tpl, 0.0) if CLIP_NEGATIVE_GEN_TO_ZERO else w_on_tpl
            w_on_aligned = align_by_utc_mdh(tpl["ts_utc"], w_on_tpl, target_year=y)
            s_on = float(np.sum(w_on_aligned))
            w_on = (w_on_aligned / s_on) if s_on > 0 else (np.ones(H) / H)

            nat_wind_on_MW = w_on * (wind_on_twh * 1e6)
            nat_wind_off_MW = w_on * (wind_off_twh * 1e6)

        nat_wind_MW = nat_wind_on_MW + nat_wind_off_MW
        nat_rl_MW = nat_load["nat_load_MW"].to_numpy(dtype=float) - nat_solar_MW - nat_wind_MW

        out = nat_load.copy()
        out["template_year_res"] = int(t)
        out["nat_solar_MW"] = nat_solar_MW
        out["nat_wind_on_MW"] = nat_wind_on_MW
        out["nat_wind_off_MW"] = nat_wind_off_MW
        out["nat_wind_MW"] = nat_wind_MW
        out["nat_rl_MW"] = nat_rl_MW

        # Annual target exactness
        sol_chk = annual_twh_from_mw_hourly(out["nat_solar_MW"].to_numpy())
        won_chk = annual_twh_from_mw_hourly(out["nat_wind_on_MW"].to_numpy())
        wof_chk = annual_twh_from_mw_hourly(out["nat_wind_off_MW"].to_numpy())

        if abs(sol_chk - solar_twh) > 1e-9:
            raise ValueError(f"[NAT RES] year={y}: solar annual mismatch {sol_chk} vs {solar_twh}")
        if abs(won_chk - wind_on_twh) > 1e-9:
            raise ValueError(f"[NAT RES] year={y}: wind_on annual mismatch {won_chk} vs {wind_on_twh}")
        if abs(wof_chk - wind_off_twh) > 1e-9:
            raise ValueError(f"[NAT RES] year={y}: wind_off annual mismatch {wof_chk} vs {wind_off_twh}")

        print(
            f"[NAT RES] {y}: template={t} | "
            f"load={annual_twh_from_mw_hourly(out['nat_load_MW'].to_numpy()):.2f} TWh | "
            f"solar={sol_chk:.2f} (target {solar_twh:.2f}) | "
            f"wind_on={won_chk:.2f} (target {wind_on_twh:.2f}) | "
            f"wind_off={wof_chk:.2f} (target {wind_off_twh:.2f})"
        )

        rows.append(out)

    nat_res = pd.concat(rows, ignore_index=True)
    nat_res = nat_res.sort_values(["year", "hour_of_year"]).reset_index(drop=True)
    return nat_res, assign


# =============================================================================
# 6) ALLOCATE NATIONAL RES TO PROVINCES AND BUILD RL
# =============================================================================
def allocate_and_build_rl(
    prov_load: pd.DataFrame,
    nat_res: pd.DataFrame,
    shares: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nat_cols = [
        "year",
        "hour_of_year",
        "timestamp_utc",
        "template_year_res",
        "nat_hh_load_MW",
        "nat_co_load_MW",
        "nat_load_MW",
        "nat_solar_MW",
        "nat_wind_on_MW",
        "nat_wind_off_MW",
        "nat_wind_MW",
        "nat_rl_MW",
    ]

    merged = prov_load.merge(
        nat_res[nat_cols],
        on=["year", "hour_of_year"],
        how="inner",
        validate="many_to_one",
        suffixes=("_prov", "_nat"),
    )

    # Timestamp consistency
    bad_ts = merged["timestamp_utc_prov"] != merged["timestamp_utc_nat"]
    if bad_ts.any():
        sample = merged.loc[bad_ts, ["year", "hour_of_year", "province", "timestamp_utc_prov", "timestamp_utc_nat"]].head(20)
        raise ValueError("[ALLOC] timestamp mismatch between prov_load and nat_res.\n" + sample.to_string(index=False))

    # Keep standardized timestamp and nat_* from nat_res
    merged["timestamp_utc"] = merged["timestamp_utc_nat"]
    merged = merged.drop(columns=["timestamp_utc_prov", "timestamp_utc_nat"])

    # Compare repeated nat_* columns coming from prov_load versus nat_res
    for c in ["nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"]:
        c_prov = f"{c}_prov"
        c_nat = f"{c}_nat"
        if c_prov in merged.columns and c_nat in merged.columns:
            diff = np.abs(merged[c_prov] - merged[c_nat])
            mx = float(diff.max())
            if mx > TOL:
                raise ValueError(f"[ALLOC] {c} mismatch between prov_load and nat_res (max gap {mx}).")
            merged = merged.drop(columns=[c_prov])
            merged = merged.rename(columns={c_nat: c})
        elif c_nat in merged.columns:
            merged = merged.rename(columns={c_nat: c})

    merged = merged.merge(shares, on=["year", "province"], how="inner", validate="many_to_one")

    merged["prov_solar_MW"] = merged["nat_solar_MW"] * merged["share_solar"]
    merged["prov_wind_on_MW"] = merged["nat_wind_on_MW"] * merged["share_wind_on"]
    merged["prov_wind_off_MW"] = merged["nat_wind_off_MW"] * merged["share_offshore"]
    merged["prov_wind_MW"] = merged["prov_wind_on_MW"] + merged["prov_wind_off_MW"]

    merged["RL_MW"] = merged["prov_load_MW"] - merged["prov_solar_MW"] - merged["prov_wind_MW"]
    merged["RL_short_MW"] = merged["RL_MW"].clip(lower=0.0)
    merged["RL_surplus_MW"] = (-merged["RL_MW"]).clip(lower=0.0)

    g = merged.groupby(["year", "hour_of_year"], as_index=False).agg(
        nat_load=("nat_load_MW", "first"),
        sum_prov_load=("prov_load_MW", "sum"),
        nat_solar=("nat_solar_MW", "first"),
        sum_prov_solar=("prov_solar_MW", "sum"),
        nat_wind_on=("nat_wind_on_MW", "first"),
        sum_prov_wind_on=("prov_wind_on_MW", "sum"),
        nat_wind_off=("nat_wind_off_MW", "first"),
        sum_prov_wind_off=("prov_wind_off_MW", "sum"),
        nat_wind=("nat_wind_MW", "first"),
        sum_prov_wind=("prov_wind_MW", "sum"),
        nat_rl=("nat_rl_MW", "first"),
        sum_prov_rl=("RL_MW", "sum"),
    )

    aL, rL = max_abs_rel(g["sum_prov_load"].to_numpy(), g["nat_load"].to_numpy())
    aS, rS = max_abs_rel(g["sum_prov_solar"].to_numpy(), g["nat_solar"].to_numpy())
    aWO, rWO = max_abs_rel(g["sum_prov_wind_on"].to_numpy(), g["nat_wind_on"].to_numpy())
    aWF, rWF = max_abs_rel(g["sum_prov_wind_off"].to_numpy(), g["nat_wind_off"].to_numpy())
    aW, rW = max_abs_rel(g["sum_prov_wind"].to_numpy(), g["nat_wind"].to_numpy())
    aR, rR = max_abs_rel(g["sum_prov_rl"].to_numpy(), g["nat_rl"].to_numpy())

    print("\n[CONSERVATION] max abs / max rel diffs")
    print(f"  LOAD     : {aL:.3e} / {rL:.3e}")
    print(f"  SOLAR    : {aS:.3e} / {rS:.3e}")
    print(f"  WIND_ON  : {aWO:.3e} / {rWO:.3e}")
    print(f"  WIND_OFF : {aWF:.3e} / {rWF:.3e}")
    print(f"  WIND_TOT : {aW:.3e} / {rW:.3e}")
    print(f"  RL       : {aR:.3e} / {rR:.3e}")

    if max(aL, aS, aWO, aWF, aW, aR) > TOL:
        raise ValueError("[CONSERVATION] Failed beyond tolerance.")

    keep = [
        "timestamp_utc",
        "year",
        "hour_of_year",
        "province",
        "prov_hh_load_MW",
        "prov_co_load_MW",
        "prov_load_MW",
        "nat_hh_load_MW",
        "nat_co_load_MW",
        "nat_load_MW",
        "prov_solar_MW",
        "prov_wind_on_MW",
        "prov_wind_off_MW",
        "prov_wind_MW",
        "nat_solar_MW",
        "nat_wind_on_MW",
        "nat_wind_off_MW",
        "nat_wind_MW",
        "RL_MW",
        "nat_rl_MW",
        "RL_short_MW",
        "RL_surplus_MW",
        "share_solar",
        "share_wind_on",
        "share_offshore",
        "template_year_res",
    ]
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep].sort_values(["year", "hour_of_year", "province"]).reset_index(drop=True)

    qc_rows = []
    for y, gy in out.groupby("year", sort=True):
        H = hours_in_year_utc(int(y))
        nat_y = nat_res.loc[nat_res["year"] == y]

        qc_rows.append({
            "year": int(y),
            "n_hours": int(H),
            "n_rows_out": int(len(gy)),
            "n_expected_rows": int(H * len(PROVINCES_12)),
            "template_year_res": int(nat_y["template_year_res"].iloc[0]),
            "nat_load_twh": annual_twh_from_mw_hourly(nat_y["nat_load_MW"].to_numpy()),
            "nat_solar_twh": annual_twh_from_mw_hourly(nat_y["nat_solar_MW"].to_numpy()),
            "nat_wind_on_twh": annual_twh_from_mw_hourly(nat_y["nat_wind_on_MW"].to_numpy()),
            "nat_wind_off_twh": annual_twh_from_mw_hourly(nat_y["nat_wind_off_MW"].to_numpy()),
            "nat_wind_twh": annual_twh_from_mw_hourly(nat_y["nat_wind_MW"].to_numpy()),
            "nat_rl_twh": annual_twh_from_mw_hourly(nat_y["nat_rl_MW"].to_numpy()),
            "sum_prov_load_twh": annual_twh_from_mw_hourly(gy["prov_load_MW"].to_numpy()),
            "sum_prov_solar_twh": annual_twh_from_mw_hourly(gy["prov_solar_MW"].to_numpy()),
            "sum_prov_wind_on_twh": annual_twh_from_mw_hourly(gy["prov_wind_on_MW"].to_numpy()),
            "sum_prov_wind_off_twh": annual_twh_from_mw_hourly(gy["prov_wind_off_MW"].to_numpy()),
            "sum_prov_wind_twh": annual_twh_from_mw_hourly(gy["prov_wind_MW"].to_numpy()),
            "sum_prov_rl_twh": annual_twh_from_mw_hourly(gy["RL_MW"].to_numpy()),
            "share_solar_sum": float(shares.loc[shares["year"] == y, "share_solar"].sum()),
            "share_wind_on_sum": float(shares.loc[shares["year"] == y, "share_wind_on"].sum()),
            "share_offshore_sum": float(shares.loc[shares["year"] == y, "share_offshore"].sum()),
        })

    qc = pd.DataFrame(qc_rows).sort_values("year").reset_index(drop=True)
    return out, qc


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("=== Build provincial residual load from learned hourly load + Option-B RES allocation ===")
    print(f"[REPO_ROOT] {REPO_ROOT}")

    for p in [OUT_FILE, OUT_SHARES_FILE, OUT_QC_FILE, OUT_DIAG_FILE]:
        ensure_parent(p)

    prov_load = load_prov_load()
    templates, wind_mode = load_nl_templates()
    targets = load_res_targets()
    shares = build_provincial_res_share_panel()

    shares.to_csv(OUT_SHARES_FILE, index=False)
    print(f"[OK] Wrote provincial RES share panel: {OUT_SHARES_FILE}")

    nat_res, template_assignment = build_future_national_res(templates, wind_mode, prov_load, targets)
    out, qc = allocate_and_build_rl(prov_load, nat_res, shares)

    out.to_csv(OUT_FILE, index=False)
    qc.to_csv(OUT_QC_FILE, index=False)

    diag = {
        "input_files": {
            "provincial_hourly_load": str(PROV_LOAD_FILE),
            "nl_profile_file": str(NL_PROFILE_FILE),
            "res_targets_file": str(RES_TARGETS_FILE),
            "offshore_landing_file": str(OFFSHORE_LANDING_FILE),
            "cbs_capacity_file_used": str(first_existing_path(CBS_PROV_LONG_CANDIDATES)),
        },
        "forecast_years": FUTURE_YEARS,
        "hour_of_year_output_convention": "1_based",
        "onshore_share_mode": ONSHORE_SHARE_MODE,
        "onshore_share_history_start": ONSHORE_SHARE_HISTORY_START,
        "onshore_share_freeze_year": ONSHORE_SHARE_FREEZE_YEAR,
        "wind_template_mode": wind_mode,
        "template_year_assignment": {str(k): int(v) for k, v in template_assignment.items()},
        "notes": [
            "This script accepts either 0-based or 1-based hour_of_year in the provincial load input and standardizes to 1-based.",
            "National hourly RES is built by scaling historical Dutch hourly templates to annual future TWh targets.",
            "Provincial solar and wind_on allocation uses province shares; offshore wind uses landing shares.",
            "Provincial sums are checked to recover national load, RES, and residual load exactly within tolerance.",
        ],
    }
    with open(OUT_DIAG_FILE, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print(f"\n[OK] Wrote main output: {OUT_FILE}")
    print(f"[OK] Wrote QC by year: {OUT_QC_FILE}")
    print(f"[OK] Wrote diagnostics: {OUT_DIAG_FILE}")

    print("\n=== SUMMARY ===")
    print(f"Years covered               : {FUTURE_START}-{FUTURE_END}")
    print(f"Output hour_of_year         : 1-based")
    print(f"Wind template mode          : {wind_mode}")
    print(f"Onshore share mode          : {ONSHORE_SHARE_MODE}")
    print(f"Rows written                : {len(out):,}")
    print(f"Provinces per hour          : {len(PROVINCES_12)}")
    print("\nSample:")
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()