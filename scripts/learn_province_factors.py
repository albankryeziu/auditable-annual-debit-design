#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
learn_province_factors.py

STEP 1: Learn provincial share factors from CBS and forecast to 2025–2035
and then allocate national hourly load (STEP 0) to provinces.

Inputs (in BASE):
  - 81528NED_CBS_HOUSEHOLDS.csv
  - 82538NED_CBS_COMPANIES.csv
  - 83704NED_CBS_DWELLINGS.csv
  - national_hourly_load_2025_2035.csv

Outputs:
  - province_share_factors_2012_2024_observed.csv
  - province_share_factors_2025_2035_forecast.csv
  - provincial_hourly_load_2025_2035.csv   (hour_of_year is UTC 0-based)

Notes:
  - OMEGA_MODE='freeze' and CO_UNIT='mwh' are preserved (your preference).
  - The STEP 2 allocation output includes nat_load_MW explicitly (and keeps load_MW alias).
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
HH_FILE = getattr(
    cfg,
    "CBS_HOUSEHOLDS_PATH",
    cfg.EXTERNAL_DIR / "81528NED_CBS_HOUSEHOLDS.csv",
)
CO_FILE = getattr(
    cfg,
    "CBS_COMPANIES_PATH",
    cfg.EXTERNAL_DIR / "82538NED_CBS_COMPANIES.csv",
)
DWELL_FILE = getattr(
    cfg,
    "CBS_DWELLINGS_PATH",
    cfg.EXTERNAL_DIR / "83704NED_CBS_DWELLINGS.csv",
)

NATIONAL_HOURLY_FILE = getattr(
    cfg,
    "NATIONAL_HOURLY_LOAD_LEGACY_PATH",
    cfg.DERIVED_DIR / "national_hourly_load_2025_2035.csv",
)

OUT_OBS = getattr(
    cfg,
    "PROVINCE_SHARE_FACTORS_OBS_PATH",
    cfg.DERIVED_DIR / "province_share_factors_2012_2024_observed.csv",
)
OUT_FC = getattr(
    cfg,
    "PROVINCE_SHARE_FACTORS_FORECAST_PATH",
    cfg.DERIVED_DIR / "province_share_factors_2025_2035_forecast.csv",
)
OUT_ALLOC = getattr(
    cfg,
    "PROVINCIAL_HOURLY_LOAD_LEGACY_PATH",
    cfg.DERIVED_DIR / "provincial_hourly_load_2025_2035.csv",
)

TRAIN_START   = 2012
TRAIN_END_OBS = 2024
FORECAST_YEARS = list(range(2025, 2036))

# Dwellings filters (CBS 83704NED “A” selection)
DWELL_WONINGTYPE_KEEP = "T001100"
DWELL_OPP_KEEP        = "T001116"

# Companies filter (total economy)
CO_SBI_TOTAL_ECONOMY = "T001081"

# Units for companies (affects omega only)
CO_UNIT: str = "mwh"  # {'mln_kwh','kwh','mwh'}

# Omega mode: 'freeze' recommended
OMEGA_MODE: str = "freeze"

OMEGA_MIN = 0.05
OMEGA_MAX = 0.85


# =============================================================================
# PROVINCE MAPPING
# =============================================================================
PROVINCE_NAMES = [
    "Groningen", "Friesland", "Drenthe", "Overijssel", "Flevoland", "Gelderland",
    "Utrecht", "Noord-Holland", "Zuid-Holland", "Zeeland", "Noord-Brabant", "Limburg"
]

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

LD_CODE_TO_NAME: Dict[str, str] = {
    "LD01": "Groningen",
    "LD02": "Friesland",
    "LD03": "Drenthe",
    "LD04": "Overijssel",
    "LD05": "Flevoland",
    "LD06": "Gelderland",
    "LD07": "Utrecht",
    "LD08": "Noord-Holland",
    "LD09": "Zuid-Holland",
    "LD10": "Zeeland",
    "LD11": "Noord-Brabant",
    "LD12": "Limburg",
}


# =============================================================================
# IO + CLEANING HELPERS
# =============================================================================
def _require_file(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required file:\n  {p}\n"
            f"Fix: ensure it exists in BASE or rename your export to this exact filename."
        )

def read_cbs_csv(path: Path) -> pd.DataFrame:
    _require_file(path)
    df = pd.read_csv(path, sep=";", engine="python", dtype=str, encoding_errors="replace")
    df.columns = [str(c).replace('"', "").strip() for c in df.columns]
    return df

def clean_str(x) -> str:
    if x is None:
        return ""
    return str(x).replace('"', "").strip()

def to_float_or_nan(x) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s == ".":
        return np.nan
    s = s.replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_year_from_perioden(s: str) -> Optional[int]:
    if not isinstance(s, str):
        return None
    t = s.strip()
    m = re.match(r"^(\d{4})JJ00$", t)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d{4})$", t)
    if m:
        return int(m.group(1))
    return None

def unit_multiplier_to_kwh(unit: str) -> float:
    u = unit.lower().strip()
    if u == "mln_kwh":
        return 1e6
    if u == "kwh":
        return 1.0
    if u == "mwh":
        return 1e3
    raise ValueError("CO_UNIT must be one of: 'mln_kwh', 'kwh', 'mwh'.")

def assert_unique_keys(df: pd.DataFrame, keys: List[str], label: str) -> None:
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        ex = df.loc[dup, keys].drop_duplicates().sort_values(keys).head(30)
        raise ValueError(f"[{label}] Duplicate rows for keys {keys}. Examples:\n{ex.to_string(index=False)}")

def assert_12_provinces_each_year(df: pd.DataFrame, label: str, year_min: int, year_max: int) -> None:
    sub = df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()
    cnt = sub.groupby("year")["province"].nunique()
    bad = cnt[cnt != 12]
    if not bad.empty:
        y0 = int(bad.index[0])
        have = set(sub[sub["year"] == y0]["province"].unique().tolist())
        missing = sorted(set(PROVINCE_NAMES) - have)
        raise ValueError(
            f"[{label}] Province coverage is incomplete in years {year_min}..{year_max}.\n"
            f"Counts:\n{bad.to_string()}\n"
            f"Example year {y0} missing: {missing}\n"
        )


# =============================================================================
# RegioS normalization (supports PV + LD)
# =============================================================================
_RX_PV = re.compile(r"^PV\d{2}$")
_RX_LD = re.compile(r"^LD\d{2}$")

def detect_regios_family(regios: pd.Series, label: str) -> str:
    codes = regios.astype(str).map(clean_str)
    pv = sorted({c for c in codes.unique() if _RX_PV.match(c)})
    ld = sorted({c for c in codes.unique() if _RX_LD.match(c)})

    pv_ok = len(pv) >= 10
    ld_ok = len(ld) >= 10

    if pv_ok and not ld_ok:
        return "PV"
    if ld_ok and not pv_ok:
        return "LD"
    if pv_ok and ld_ok:
        print(f"[INFO] {label}: detected both PV and LD codes; using PV mapping.")
        return "PV"

    sample = sorted(list(set(codes.unique())))[:40]
    raise ValueError(
        f"[{label}] Cannot detect province code family (PV/LD). "
        f"Found PV={len(pv)}, LD={len(ld)}. Sample RegioS={sample}"
    )

def normalize_regios(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["RegioS"] = df["RegioS"].map(clean_str)

    fam = detect_regios_family(df["RegioS"], label)
    if fam == "PV":
        keep = df["RegioS"].str.match(r"^(PV\d{2}|NL01)$")
        mapper = PV_CODE_TO_NAME
    else:
        keep = df["RegioS"].str.match(r"^(LD\d{2}|NL01)$")
        mapper = LD_CODE_TO_NAME

    df = df.loc[keep].copy()
    df["province"] = df["RegioS"].map(mapper)
    return df


# =============================================================================
# Forecast models (softmax time model + optional logit omega trend)
# =============================================================================
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def fit_softmax_time_model(
    years: np.ndarray,
    shares: np.ndarray,
    lam: float = 1e-2,
    iters: int = 8000,
    lr: float = 0.05,
    seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    years = years.astype(int)
    t_raw = (years - years.min()).astype(float)
    t = (t_raw - t_raw.mean()) / (t_raw.std() + 1e-12)

    X = np.vstack([np.ones_like(t), t, t**2]).T
    n = X.shape[0]
    K = shares.shape[1]

    W = 0.01 * rng.standard_normal((X.shape[1], K))

    # Adam
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    b1, b2 = 0.9, 0.999
    eps = 1e-8

    for step in range(1, iters + 1):
        P = softmax(X @ W)
        G = (X.T @ (P - shares)) / n + 2.0 * lam * W

        m = b1 * m + (1 - b1) * G
        v = b2 * v + (1 - b2) * (G * G)
        mhat = m / (1 - b1 ** step)
        vhat = v / (1 - b2 ** step)
        W -= lr * mhat / (np.sqrt(vhat) + eps)

        if step % 2000 == 0:
            ce = -np.mean(np.sum(shares * np.log(np.clip(P, 1e-12, 1.0)), axis=1))
            print(f"  step={step:5d}  CE={ce:.6f}")

    return W

def predict_softmax_time_model(years_train: np.ndarray, W: np.ndarray, years_pred: np.ndarray) -> np.ndarray:
    years_train = years_train.astype(int)
    years_pred = years_pred.astype(int)

    t_raw_tr = (years_train - years_train.min()).astype(float)
    mu = t_raw_tr.mean()
    sd = t_raw_tr.std() + 1e-12

    t_raw_pr = (years_pred - years_train.min()).astype(float)
    t_pr = (t_raw_pr - mu) / sd

    Xp = np.vstack([np.ones_like(t_pr), t_pr, t_pr**2]).T
    return softmax(Xp @ W)

def fit_logit_trend(years: np.ndarray, omega: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    years = years.astype(int)
    t_raw = (years - years.min()).astype(float)
    t = (t_raw - t_raw.mean()) / (t_raw.std() + 1e-12)
    X = np.vstack([np.ones_like(t), t, t**2]).T

    o = np.clip(omega, 1e-6, 1 - 1e-6)
    y = np.log(o / (1 - o))

    beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    return beta

def predict_logit_trend(years_train: np.ndarray, beta: np.ndarray, years_pred: np.ndarray) -> np.ndarray:
    years_train = years_train.astype(int)
    years_pred = years_pred.astype(int)

    t_raw_tr = (years_train - years_train.min()).astype(float)
    mu = t_raw_tr.mean()
    sd = t_raw_tr.std() + 1e-12

    t_raw_pr = (years_pred - years_train.min()).astype(float)
    t_pr = (t_raw_pr - mu) / sd

    Xp = np.vstack([np.ones_like(t_pr), t_pr, t_pr**2]).T
    yhat = Xp @ beta
    o = 1.0 / (1.0 + np.exp(-yhat))
    return np.clip(o, 0.0, 1.0)


# =============================================================================
# DWELLINGS (83704NED)
# =============================================================================
def load_dwellings_cbs() -> pd.DataFrame:
    df = read_cbs_csv(DWELL_FILE)

    req = {"Woningtype", "Oppervlakteklasse", "RegioS", "Perioden"}
    if not req.issubset(df.columns):
        raise ValueError(f"[DWELLINGS] Missing columns: {sorted(req - set(df.columns))}")

    val_cols = [c for c in df.columns if "BeginstandWoningvoorraad" in c]
    if not val_cols:
        raise ValueError("[DWELLINGS] Cannot find BeginstandWoningvoorraad column.")
    val_col = val_cols[0]

    df["Woningtype"] = df["Woningtype"].map(clean_str)
    df["Oppervlakteklasse"] = df["Oppervlakteklasse"].map(clean_str)
    df["year"] = df["Perioden"].map(parse_year_from_perioden)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    df = df[(df["Woningtype"] == DWELL_WONINGTYPE_KEEP) & (df["Oppervlakteklasse"] == DWELL_OPP_KEEP)].copy()

    df = normalize_regios(df, label="DWELLINGS")
    df = df.dropna(subset=["province"]).copy()

    df["dwellings"] = df[val_col].map(to_float_or_nan)
    df = df.dropna(subset=["dwellings"]).copy()

    out = df.groupby(["year", "province"], as_index=False)["dwellings"].mean()
    assert_unique_keys(out, ["year", "province"], "DWELLINGS_OUT")
    return out.sort_values(["year", "province"]).reset_index(drop=True)


# =============================================================================
# HOUSEHOLDS (81528NED) + DWELLINGS
# =============================================================================
def build_households_shares(dwell: pd.DataFrame) -> pd.DataFrame:
    df = read_cbs_csv(HH_FILE)

    req = {"Woningkenmerken", "RegioS", "Perioden"}
    if not req.issubset(df.columns):
        raise ValueError(f"[HOUSEHOLDS] Missing columns: {sorted(req - set(df.columns))}")

    df["Woningkenmerken"] = df["Woningkenmerken"].map(clean_str)
    df["year"] = df["Perioden"].map(parse_year_from_perioden)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    df = df[df["Woningkenmerken"] == "T001100"].copy()
    df = df[df["year"] >= TRAIN_START].copy()

    df = normalize_regios(df, label="HOUSEHOLDS")
    df = df.dropna(subset=["province"]).copy()

    elec_net = [c for c in df.columns if "NettoElektriciteitslevering" in c]
    elec_avg = [c for c in df.columns if "Elektriciteitslevering" in c and "Netto" not in c]
    if elec_net:
        elec_col = elec_net[0]
    elif elec_avg:
        elec_col = elec_avg[0]
    else:
        raise ValueError("[HOUSEHOLDS] Cannot find (Netto)Elektriciteitslevering column.")

    df["avg_kwh_per_dwelling"] = df[elec_col].map(to_float_or_nan)

    df = df.sort_values(["RegioS", "year"])
    df["avg_kwh_per_dwelling"] = df.groupby("RegioS")["avg_kwh_per_dwelling"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    df = df.dropna(subset=["avg_kwh_per_dwelling"]).copy()

    pv = df.groupby(["year", "province"], as_index=False).agg(
        avg_kwh_per_dwelling=("avg_kwh_per_dwelling", "mean")
    )
    assert_unique_keys(pv, ["year", "province"], "HOUSEHOLDS_RATE")

    pv = pv.merge(dwell, on=["year", "province"], how="left")
    if pv["dwellings"].isna().any():
        miss = pv[pv["dwellings"].isna()][["year", "province"]].drop_duplicates().sort_values(["year", "province"])
        raise ValueError(f"[HOUSEHOLDS] Missing dwellings after merge. Examples:\n{miss.head(30).to_string(index=False)}")

    pv["hh_total_kwh"] = pv["avg_kwh_per_dwelling"] * pv["dwellings"]
    pv["hh_total_mwh"] = pv["hh_total_kwh"] / 1000.0

    denom = pv.groupby("year")["hh_total_mwh"].transform("sum")
    pv["hh_share"] = np.where(denom > 0, pv["hh_total_mwh"] / denom, 1.0 / len(PROVINCE_NAMES))

    out = pv[["year", "province", "avg_kwh_per_dwelling", "dwellings", "hh_total_kwh", "hh_total_mwh", "hh_share"]].copy()
    assert_unique_keys(out, ["year", "province"], "HOUSEHOLDS_OUT")
    return out.sort_values(["year", "province"]).reset_index(drop=True)


# =============================================================================
# COMPANIES (82538NED)
# =============================================================================
def build_companies_shares() -> pd.DataFrame:
    df = read_cbs_csv(CO_FILE)

    req = {"BedrijfstakkenBranchesSBI2008", "RegioS", "Perioden"}
    if not req.issubset(df.columns):
        raise ValueError(f"[COMPANIES] Missing columns: {sorted(req - set(df.columns))}")

    elec_cols = [c for c in df.columns if "GeleverdeElektriciteit" in c]
    if not elec_cols:
        raise ValueError("[COMPANIES] Cannot find GeleverdeElektriciteit column.")
    elec_col = elec_cols[0]

    df["BedrijfstakkenBranchesSBI2008"] = df["BedrijfstakkenBranchesSBI2008"].map(clean_str)
    df["year"] = df["Perioden"].map(parse_year_from_perioden)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    df = df[df["BedrijfstakkenBranchesSBI2008"] == CO_SBI_TOTAL_ECONOMY].copy()
    df = normalize_regios(df, label="COMPANIES")

    df["co_elec_raw"] = df[elec_col].map(to_float_or_nan)
    df = df.dropna(subset=["co_elec_raw"]).copy()

    nl = (
        df[df["RegioS"] == "NL01"][["year", "co_elec_raw"]]
        .groupby("year", as_index=False)["co_elec_raw"].sum()
        .rename(columns={"co_elec_raw": "co_nl"})
    )

    pv = df.dropna(subset=["province"]).copy()
    pv = pv[pv["province"].isin(PROVINCE_NAMES)].copy()
    pv = pv.groupby(["year", "province"], as_index=False)["co_elec_raw"].sum()
    assert_unique_keys(pv, ["year", "province"], "COMPANIES_RAW_BY_PROV")

    pv_sum = pv.groupby("year", as_index=False)["co_elec_raw"].sum().rename(columns={"co_elec_raw": "co_sumprov"})
    tmp = pv_sum.merge(nl, on="year", how="left")

    tmp["scale"] = np.where(
        tmp["co_nl"].notna() & (tmp["co_sumprov"] > 0),
        tmp["co_nl"] / tmp["co_sumprov"],
        1.0
    )

    pv = pv.merge(tmp[["year", "scale", "co_nl"]], on="year", how="left")
    pv["co_elec_adj"] = pv["co_elec_raw"] * pv["scale"]

    denom = pv.groupby("year")["co_elec_adj"].transform("sum")
    pv["co_share"] = np.where(denom > 0, pv["co_elec_adj"] / denom, 1.0 / len(PROVINCE_NAMES))

    out = pv[["year", "province", "co_elec_raw", "co_elec_adj", "co_nl", "co_share"]].copy()
    assert_unique_keys(out, ["year", "province"], "COMPANIES_OUT")
    return out.sort_values(["year", "province"]).reset_index(drop=True)


# =============================================================================
# OBSERVED TABLE (2012–2024) + OMEGA
# =============================================================================
def build_observed_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dwell = load_dwellings_cbs()
    hh_all = build_households_shares(dwell)
    co_all = build_companies_shares()

    assert_12_provinces_each_year(dwell, "DWELLINGS", TRAIN_START, TRAIN_END_OBS)
    assert_12_provinces_each_year(hh_all, "HOUSEHOLDS", TRAIN_START, TRAIN_END_OBS)
    assert_12_provinces_each_year(co_all, "COMPANIES", TRAIN_START, TRAIN_END_OBS)

    hh_obs = hh_all[(hh_all["year"] >= TRAIN_START) & (hh_all["year"] <= TRAIN_END_OBS)].copy()
    co_obs = co_all[(co_all["year"] >= TRAIN_START) & (co_all["year"] <= TRAIN_END_OBS)].copy()

    obs = hh_obs.merge(co_obs, on=["year", "province"], how="inner")
    assert_unique_keys(obs, ["year", "province"], "OBS_MERGED")

    mult_kwh = unit_multiplier_to_kwh(CO_UNIT)
    obs["co_total_mwh"] = (obs["co_elec_adj"] * mult_kwh) / 1000.0

    hh_nat = obs.groupby("year", as_index=False)["hh_total_mwh"].sum().rename(columns={"hh_total_mwh": "hh_national_mwh"})
    co_nat = obs.groupby("year", as_index=False)["co_total_mwh"].sum().rename(columns={"co_total_mwh": "co_national_mwh"})
    obs = obs.merge(hh_nat, on="year", how="left").merge(co_nat, on="year", how="left")

    denom = obs["hh_national_mwh"] + obs["co_national_mwh"]
    obs["omega_hh"] = np.where(denom > 0, obs["hh_national_mwh"] / denom, np.nan)
    obs["omega_co"] = np.where(denom > 0, obs["co_national_mwh"] / denom, np.nan)

    om = obs.groupby("year")[["omega_hh", "omega_co"]].first()
    if (om["omega_hh"] < OMEGA_MIN).any() or (om["omega_hh"] > OMEGA_MAX).any():
        raise ValueError(
            f"Omega looks implausible (omega_hh outside [{OMEGA_MIN}, {OMEGA_MAX}]). "
            f"Likely wrong CO_UNIT='{CO_UNIT}'."
        )

    obs["total_share"] = obs["omega_hh"] * obs["hh_share"] + obs["omega_co"] * obs["co_share"]

    chk = obs.groupby("year")[["hh_share", "co_share", "total_share"]].sum()
    mask = (
        ((chk["hh_share"] - 1).abs() > 1e-6)
        | ((chk["co_share"] - 1).abs() > 1e-6)
        | ((chk["total_share"] - 1).abs() > 1e-6)
    )
    bad = chk[mask]
    if not bad.empty:
        raise ValueError(f"[OBSERVED] Share sums not equal to 1:\n{bad.to_string()}")

    return obs.sort_values(["year", "province"]).reset_index(drop=True), hh_all, co_all


# =============================================================================
# FORECAST 2025–2035
# =============================================================================
def _pivot_share(df: pd.DataFrame, col: str, years: np.ndarray) -> np.ndarray:
    mat = (
        df.pivot_table(index="year", columns="province", values=col, aggfunc="mean")
          .reindex(index=years, columns=PROVINCE_NAMES)
    )
    A = mat.to_numpy(dtype=float)
    A = np.where(np.isnan(A), 0.0, A)
    rs = A.sum(axis=1, keepdims=True)
    A = np.where(rs > 0, A / rs, 1.0 / len(PROVINCE_NAMES))
    return A

def forecast_shares_and_weights(obs: pd.DataFrame, years_fc: List[int]) -> pd.DataFrame:
    years_fc_arr = np.array(years_fc, dtype=int)
    train_years = np.array(sorted(obs["year"].unique().astype(int)), dtype=int)

    hh_tr = _pivot_share(obs, "hh_share", train_years)
    co_tr = _pivot_share(obs, "co_share", train_years)

    print("\nFitting softmax-time model for HOUSEHOLDS shares...")
    W_hh = fit_softmax_time_model(train_years, hh_tr, lam=1e-2, iters=8000, lr=0.05, seed=1)
    hh_fc = predict_softmax_time_model(train_years, W_hh, years_fc_arr)

    print("\nFitting softmax-time model for COMPANIES shares...")
    W_co = fit_softmax_time_model(train_years, co_tr, lam=1e-2, iters=8000, lr=0.05, seed=2)
    co_fc = predict_softmax_time_model(train_years, W_co, years_fc_arr)

    omega_by_year = obs.groupby("year", as_index=False)[["omega_hh", "omega_co"]].first().sort_values("year")
    omega_2024 = float(omega_by_year.loc[omega_by_year["year"] == TRAIN_END_OBS, "omega_hh"].iloc[0])

    if OMEGA_MODE.lower().strip() == "freeze":
        omega_hh_hat = np.full_like(years_fc_arr, omega_2024, dtype=float)
        omega_co_hat = 1.0 - omega_hh_hat
    elif OMEGA_MODE.lower().strip() == "trend":
        y_om = omega_by_year["year"].to_numpy(dtype=int)
        oco = omega_by_year["omega_co"].to_numpy(dtype=float)
        beta = fit_logit_trend(y_om, oco, lam=1e-3)
        omega_co_hat = predict_logit_trend(y_om, beta, years_fc_arr)
        omega_hh_hat = 1.0 - omega_co_hat
    else:
        raise ValueError("OMEGA_MODE must be one of: 'freeze', 'trend'.")

    rows = []
    for i, y in enumerate(years_fc_arr):
        for j, p in enumerate(PROVINCE_NAMES):
            total_hat = omega_hh_hat[i] * hh_fc[i, j] + omega_co_hat[i] * co_fc[i, j]
            rows.append({
                "year": int(y),
                "province": p,
                "hh_share_hat": float(hh_fc[i, j]),
                "co_share_hat": float(co_fc[i, j]),
                "omega_hh_hat": float(omega_hh_hat[i]),
                "omega_co_hat": float(omega_co_hat[i]),
                "total_share_hat": float(total_hat),
            })

    fc = pd.DataFrame(rows).sort_values(["year", "province"]).reset_index(drop=True)

    gsum = fc.groupby("year")["total_share_hat"].transform("sum")
    fc.loc[gsum > 0, "total_share_hat"] = fc.loc[gsum > 0, "total_share_hat"] / gsum[gsum > 0]

    chk = fc.groupby("year")[["hh_share_hat", "co_share_hat", "total_share_hat"]].sum()
    mask = (
        ((chk["hh_share_hat"] - 1).abs() > 1e-6)
        | ((chk["co_share_hat"] - 1).abs() > 1e-6)
        | ((chk["total_share_hat"] - 1).abs() > 1e-6)
    )
    bad = chk[mask]
    if not bad.empty:
        raise ValueError(f"[FORECAST] Share sums not equal to 1:\n{bad.to_string()}")

    return fc


# =============================================================================
# STEP 2: Allocate national hourly load to provinces + conservation check
# =============================================================================
def normalize_hour_of_year_to_0_based(nat: pd.DataFrame) -> pd.DataFrame:
    nat = nat.copy()
    nat["year"] = pd.to_numeric(nat["year"], errors="coerce").astype(int)
    nat["hour_of_year"] = pd.to_numeric(nat["hour_of_year"], errors="coerce").astype(int)

    def exp_hours(y: int) -> int:
        t0 = pd.Timestamp(f"{y}-01-01T00:00:00Z")
        t1 = pd.Timestamp(f"{y+1}-01-01T00:00:00Z")
        return int((t1 - t0) / pd.Timedelta(hours=1))

    out = []
    for y, g in nat.groupby("year", sort=False):
        H = exp_hours(int(y))
        mn, mx = int(g["hour_of_year"].min()), int(g["hour_of_year"].max())
        nunq = int(g["hour_of_year"].nunique())
        gg = g.copy()
        if mn == 0 and mx == H - 1 and nunq == H:
            pass
        elif mn == 1 and mx == H and nunq == H:
            gg["hour_of_year"] = gg["hour_of_year"] - 1
        else:
            raise ValueError(
                f"[NATIONAL] year={y}: hour_of_year not recognized. "
                f"min={mn}, max={mx}, nunique={nunq}, expected 0..{H-1} (or 1..{H})."
            )
        out.append(gg)
    return pd.concat(out, ignore_index=True)

def allocate_national_hourly_to_provinces(fc: pd.DataFrame) -> None:
    _require_file(NATIONAL_HOURLY_FILE)

    nat = pd.read_csv(NATIONAL_HOURLY_FILE)
    nat.columns = [c.strip() for c in nat.columns]

    if "year" not in nat.columns or "hour_of_year" not in nat.columns:
        raise ValueError("[NATIONAL] Missing 'year' or 'hour_of_year'.")

    # accept either load_MW or nat_load_MW
    if "nat_load_MW" in nat.columns:
        load_col = "nat_load_MW"
    elif "load_MW" in nat.columns:
        load_col = "load_MW"
    else:
        raise ValueError("[NATIONAL] Missing 'load_MW' (or 'nat_load_MW').")

    nat[load_col] = pd.to_numeric(nat[load_col], errors="coerce")
    nat = nat.dropna(subset=[load_col]).copy()
    nat = normalize_hour_of_year_to_0_based(nat)

    nat = nat[(nat["year"] >= 2025) & (nat["year"] <= 2035)].copy()

    # create explicit nat_load_MW + keep alias load_MW
    nat["nat_load_MW"] = nat[load_col].astype(float)
    nat["load_MW"] = nat["nat_load_MW"]

    merged = nat.merge(fc[["year", "province", "total_share_hat"]], on="year", how="inner")
    merged["prov_load_MW"] = merged["nat_load_MW"] * merged["total_share_hat"]

    # Conservation check (per hour)
    check = merged.groupby(["year", "hour_of_year"], as_index=False).agg(
        nat_load_MW=("nat_load_MW", "first"),
        sum_prov=("prov_load_MW", "sum"),
    )
    check["abs_diff"] = (check["sum_prov"] - check["nat_load_MW"]).abs()
    max_abs = float(check["abs_diff"].max())
    denom = np.maximum(check["nat_load_MW"].abs().to_numpy(), 1e-12)
    max_rel = float(np.max(check["abs_diff"].to_numpy() / denom))

    print("Join keys: ['year', 'hour_of_year'] (0-based)")
    print(f"Max abs diff (MW): {max_abs}")
    print(f"Max rel diff: {max_rel}")
    if max_abs > 1e-6:
        raise ValueError("[STEP2] Hourly conservation FAILED beyond tolerance.")

    print("[OK] Hourly conservation holds exactly (within float noise).")

    # Keep nat columns + add province columns (also preserves template_year/scale_factor if present)
    out_cols = list(nat.columns) + ["province", "total_share_hat", "prov_load_MW"]
    merged[out_cols].to_csv(OUT_ALLOC, index=False)
    print(f"[OK] Wrote STEP 2 allocation: {OUT_ALLOC} (prov col='prov_load_MW')")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=== STEP 1A: Build observed province factor table (2012–2024) using DWELLINGS “A” ===")
    obs, hh_all, co_all = build_observed_tables()
    OUT_OBS.parent.mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT_OBS, index=False)
    print(f"[OK] Wrote: {OUT_OBS}")

    print("\nObserved sum checks (must be 1.0 each year):")
    chk = obs.groupby("year")[["hh_share", "co_share", "total_share"]].sum().round(6)
    print(chk.to_string())

    print(f"\n=== STEP 1B: Forecast shares + omega for 2025–2035 (OMEGA_MODE={OMEGA_MODE}) ===")
    fc = forecast_shares_and_weights(obs, FORECAST_YEARS)
    fc.to_csv(OUT_FC, index=False)
    print(f"[OK] Wrote: {OUT_FC}")

    print("\nForecast sum checks (must be 1.0 each year):")
    chk2 = fc.groupby("year")[["hh_share_hat", "co_share_hat", "total_share_hat"]].sum().round(6)
    print(chk2.to_string())

    print("\nOmega checks:")
    chk3 = fc.groupby("year")[["omega_hh_hat", "omega_co_hat"]].first().round(6)
    print(chk3.to_string())

    print("\n=== STEP 2: Allocate national hourly load to provinces ===")
    allocate_national_hourly_to_provinces(fc)

if __name__ == "__main__":
    main()
