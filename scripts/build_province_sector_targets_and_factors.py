#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_province_sector_targets_and_factors.py

Purpose
-------
Build annual provincial household/company electricity totals and province-sector
allocation factors from CBS StatLine tables, then forecast provincial HH/CO
shares to 2025-2035 with simplex-consistent softmax time models.

Pipeline role
-------------
This script is the annual province-sector structure builder in the Dutch
residual-load pipeline. It should be run after the national historical load
master exists, and before the national HH/CO hourly learning stage.

Expected input files
--------------------
- 82538NED_CBS_COMPANIES.csv
- 81528NED_CBS_HOUSEHOLDS.csv
- 83704NED_CBS_DWELLINGS.csv

Key design choices
------------------
- Historical overlap starts at 2012 by default because the dwelling-stock
  input is typically the limiting series in the current pipeline.
- 2024 is kept in observed outputs but excluded from fitting by default
  because it is provisional.
- HH/CO provincial shares are forecast with an additive-log-ratio time model,
  then mapped back through softmax so shares are strictly positive and sum to 1.
- National HH/CO mix omega_hh / omega_co is forecast using freeze by default.

Outputs
-------
- derived/provincial_sector_observed_annual_2012_2024.csv
- derived/provincial_sector_shares_2012_2035.csv
- derived/national_sector_mix_2012_2035.csv
- derived/provincial_sector_share_factors_2012_2035.csv
- derived/provincial_sector_targets_and_factors_diagnostics.json
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
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
# USER CONFIG
# =============================================================================

# =============================================================================
# USER CONFIG
# =============================================================================

FILE_COMPANIES = getattr(
    cfg,
    "CBS_COMPANIES_PATH",
    cfg.EXTERNAL_DIR / "82538NED_CBS_COMPANIES.csv",
)
FILE_HOUSEHOLDS = getattr(
    cfg,
    "CBS_HOUSEHOLDS_PATH",
    cfg.EXTERNAL_DIR / "81528NED_CBS_HOUSEHOLDS.csv",
)
FILE_DWELLINGS = getattr(
    cfg,
    "CBS_DWELLINGS_PATH",
    cfg.EXTERNAL_DIR / "83704NED_CBS_DWELLINGS.csv",
)

# Historical/forecast horizon
HIST_START = 2012
HIST_END = 2024
FORECAST_START = 2025
FORECAST_END = 2035

# Fit settings
USE_2024_IN_TRAIN = False
TRAIN_END_YEAR = 2024 if USE_2024_IN_TRAIN else 2023
RIDGE_L2 = 1e-3
EPS_SHARE = 1e-9
OMEGA_MODE = "freeze"       # alternatives: "freeze", "trend"

# Units
HH_AVG_UNIT_RAW = "auto"    # auto, kwh_per_dwelling, mwh_per_dwelling
DWELLINGS_UNIT_RAW = "auto" # auto, count, thousand_count
CO_UNIT_RAW = "auto"        # auto, mwh, kwh, mln_kwh

# Share-factor diagnostics
SHARE_FACTOR_BASE_YEAR = 2023

# Debugging
DEBUG_EXTRACTORS = True

# Canonical province names used downstream
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

PROVINCE_MAP = {
    # Text names
    "drenthe": "Drenthe",
    "flevoland": "Flevoland",
    "fryslan": "Friesland",
    "friesland": "Friesland",
    "gelderland": "Gelderland",
    "groningen": "Groningen",
    "limburg": "Limburg",
    "noord brabant": "Noord-Brabant",
    "noord-brabant": "Noord-Brabant",
    "n brabant": "Noord-Brabant",
    "nbrabant": "Noord-Brabant",
    "noord holland": "Noord-Holland",
    "noord-holland": "Noord-Holland",
    "n holland": "Noord-Holland",
    "nholland": "Noord-Holland",
    "overijssel": "Overijssel",
    "utrecht": "Utrecht",
    "zeeland": "Zeeland",
    "zuid holland": "Zuid-Holland",
    "zuid-holland": "Zuid-Holland",
    "z holland": "Zuid-Holland",
    "zholland": "Zuid-Holland",

    # CBS region codes
    "pv20": "Groningen",
    "pv21": "Friesland",
    "pv22": "Drenthe",
    "pv23": "Overijssel",
    "pv24": "Flevoland",
    "pv25": "Gelderland",
    "pv26": "Utrecht",
    "pv27": "Noord-Holland",
    "pv28": "Zuid-Holland",
    "pv29": "Zeeland",
    "pv30": "Noord-Brabant",
    "pv31": "Limburg",
}

OUT_HIST = getattr(
    cfg,
    "PROVINCIAL_SECTOR_OBS_PATH",
    cfg.DERIVED_DIR / f"provincial_sector_observed_annual_{HIST_START}_{HIST_END}.csv",
)
OUT_SHARES = getattr(
    cfg,
    "PROVINCIAL_SECTOR_SHARES_PATH",
    cfg.DERIVED_DIR / f"provincial_sector_shares_{HIST_START}_{FORECAST_END}.csv",
)
OUT_OMEGA = getattr(
    cfg,
    "NATIONAL_SECTOR_MIX_PATH",
    cfg.DERIVED_DIR / f"national_sector_mix_{HIST_START}_{FORECAST_END}.csv",
)
OUT_FACTORS = getattr(
    cfg,
    "PROVINCIAL_SECTOR_SHARE_FACTORS_PATH",
    cfg.DERIVED_DIR / f"provincial_sector_share_factors_{HIST_START}_{FORECAST_END}.csv",
)
OUT_DIAG = getattr(
    cfg,
    "PROVINCIAL_SECTOR_DIAG_PATH",
    cfg.DERIVED_DIR / "provincial_sector_targets_and_factors_diagnostics.json",
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _norm_text(x: object) -> str:
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def _norm_col(x: object) -> str:
    s = _norm_text(x)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse_year(x: object) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x)
    m = re.search(r"(19\d{2}|20\d{2})", s)
    return int(m.group(1)) if m else None


def canonical_province(x: object) -> Optional[str]:
    if pd.isna(x):
        return None
    s = _norm_text(x)
    if s in PROVINCE_MAP:
        return PROVINCE_MAP[s]
    for k, v in PROVINCE_MAP.items():
        if k in s:
            return v
    return None


def strip_empty(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(axis=0, how="all").dropna(axis=1, how="all")
    keep_cols = [c for c in out.columns if _norm_col(c) not in {"unnamed", ""}]
    return out[keep_cols].copy()


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Robust reader for CBS StatLine-style CSV exports.
    Tries explicit separators first, then a header-row scan fallback.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    # First pass: explicit separators in a safe order for CBS exports.
    for sep in [";", ",", "\t", "|", None]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")
            if df.shape[1] > 1:
                df.columns = [str(c).strip().strip('"').strip() for c in df.columns]
                return strip_empty(df)
        except Exception:
            pass

    # Second pass: search for a likely header row in the first 80 lines.
    lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    header_idx = None
    for i, line in enumerate(lines[:80]):
        probe = _norm_text(line)
        if (
            ("period" in probe or "perioden" in probe or "jaar" in probe)
            and ("regio" in probe or "provincie" in probe or "region" in probe)
        ):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not detect a usable header row in {path}")

    for sep in [";", ",", "\t", "|", None]:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                encoding="utf-8-sig",
                skiprows=header_idx,
            )
            if df.shape[1] > 1:
                df.columns = [str(c).strip().strip('"').strip() for c in df.columns]
                return strip_empty(df)
        except Exception:
            pass

    raise ValueError(f"Could not parse CSV with a usable separator/header: {path}")


def coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Robust numeric coercion for CBS exports.
    Handles:
      - European decimals: 1.234,56
      - Plain decimals: 1234.56
      - Missing symbols like '.', '-', '?', 'x'
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()

    # Handle known missing markers
    s = s.replace({
        "": np.nan,
        ".": np.nan,
        "-": np.nan,
        "?": np.nan,
        "x": np.nan,
        "X": np.nan,
    })

    s = s.str.replace("\u00A0", " ", regex=False)
    s = s.str.replace(r"\s+", "", regex=True)

    # If a value ends in comma-decimals, treat dots as thousand separators
    has_decimal_comma = s.str.contains(r",\d{1,6}$", regex=True, na=False)
    s = s.where(~has_decimal_comma, s.str.replace(".", "", regex=False))

    # Convert decimal comma to decimal point
    s = s.str.replace(",", ".", regex=False)

    # Keep digits, minus, dot only
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    s = s.replace({"": np.nan, ".": np.nan, "-": np.nan, "-.": np.nan}).infer_objects(copy=False)

    return pd.to_numeric(s, errors="coerce")


def numeric_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if coerce_numeric(df[c]).notna().sum() > 0:
            cols.append(c)
    return cols


def contains_any(s: object, patterns: Iterable[str]) -> bool:
    st = _norm_text(s)
    return any(p in st for p in patterns)


def find_year_col(df: pd.DataFrame) -> str:
    candidates = []
    for c in df.columns:
        nc = _norm_col(c)
        score = 0.0
        if any(tok in nc for tok in ["period", "perioden", "jaar", "year"]):
            score += 10.0
        share_parsed = df[c].map(parse_year).notna().mean()
        score += 20.0 * share_parsed
        candidates.append((score, c))
    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] <= 0:
        raise KeyError("Could not identify a year column")
    return candidates[0][1]


def find_region_col(df: pd.DataFrame) -> str:
    candidates = []
    for c in df.columns:
        nc = _norm_col(c)
        score = 0.0
        if any(tok in nc for tok in ["regio", "provincie", "province", "region"]):
            score += 10.0
        prov_share = df[c].map(canonical_province).notna().mean()
        score += 30.0 * prov_share
        candidates.append((score, c))
    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] <= 0:
        raise KeyError("Could not identify a region/province column")
    return candidates[0][1]


def select_best_numeric_col(
    df: pd.DataFrame,
    positive_name_patterns: List[str],
    negative_name_patterns: Optional[List[str]] = None,
) -> str:
    negative_name_patterns = negative_name_patterns or []
    candidates = []

    for c in numeric_columns(df):
        nc = _norm_col(c)
        score = 0.0

        if any(p in nc for p in positive_name_patterns):
            score += 100.0
        if any(p in nc for p in negative_name_patterns):
            score -= 200.0

        nonnull_share = coerce_numeric(df[c]).notna().mean()
        score += 20.0 * nonnull_share
        candidates.append((score, c))

    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] <= 0:
        raise KeyError(
            f"Could not confidently identify numeric column. "
            f"positive={positive_name_patterns}, negative={negative_name_patterns}"
        )
    return candidates[0][1]


def find_optional_text_col(df: pd.DataFrame, pattern_groups: List[List[str]]) -> Optional[str]:
    best = None
    best_score = -1.0
    num_cols = set(numeric_columns(df))

    for c in df.columns:
        if c in num_cols:
            continue
        nc = _norm_col(c)
        score = 0.0
        for patterns in pattern_groups:
            if any(p in nc for p in patterns):
                score += 100.0
        if score > best_score:
            best = c
            best_score = score

    return best if best_score > 0 else None


def restrict_to_provinces_and_years(df: pd.DataFrame, year_col: str, region_col: str) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[year_col].map(parse_year)
    out["province"] = out[region_col].map(canonical_province)
    out = out[out["year"].between(HIST_START, HIST_END, inclusive="both")]
    out = out[out["province"].isin(PROVINCES)]
    return out.copy()


def filter_to_total_if_available(
    df: pd.DataFrame,
    col: str,
    total_patterns: List[str],
) -> pd.DataFrame:
    """
    Only filters if total-like rows actually exist.
    Otherwise returns the dataframe unchanged.
    """
    out = df.copy()
    key = out[col].astype(str).map(_norm_text)
    mask = key.map(lambda x: contains_any(x, total_patterns))
    if mask.any():
        out = out[mask].copy()
    return out


def maybe_filter_total_dimensions(
    df: pd.DataFrame,
    exclude_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Conservative fallback:
    for text columns with both total-like and non-total rows, keep total-like rows.
    """
    total_patterns = ["totaal", "total", "alle", "all"]
    exclude = set(exclude_cols)
    out = df.copy()

    num_cols = set(numeric_columns(out))
    for c in out.columns:
        if c in exclude or c in num_cols:
            continue
        vals = out[c].dropna().astype(str)
        if vals.empty:
            continue
        is_total = vals.map(lambda x: contains_any(x, total_patterns))
        if is_total.any() and (~is_total).any():
            total_vals = set(vals[is_total].unique())
            out = out[out[c].isin(total_vals)].copy()

    return out


def assert_complete_panel(df: pd.DataFrame, value_col: str, label: str) -> None:
    full_idx = pd.MultiIndex.from_product(
        [PROVINCES, range(HIST_START, HIST_END + 1)],
        names=["province", "year"],
    )
    chk = df.set_index(["province", "year"]).reindex(full_idx)
    miss = int(chk[value_col].isna().sum())
    if miss > 0:
        raise ValueError(
            f"[{label}] Incomplete province-year panel after extraction: {miss} missing cells. "
            f"This usually means the CSV parser selected the wrong dimension or value column."
        )


def debug_preview(label: str, df: pd.DataFrame, cols: List[str], n: int = 10) -> None:
    if not DEBUG_EXTRACTORS:
        return
    print(f"\n[DEBUG {label}] rows={len(df)} cols={len(df.columns)}")
    for c in cols:
        if c in df.columns:
            vals = df[c].dropna().astype(str).head(n).tolist()
            print(f"  {c}: {vals}")


# =============================================================================
# DATASET-SPECIFIC EXTRACTORS
# =============================================================================

def extract_household_avg_per_dwelling(path: Path) -> pd.DataFrame:
    """
    Extract household average electricity per dwelling from 81528NED.

    Robust logic:
    - identify year/region/electricity columns
    - optionally identify a housing-type dimension
    - filter to provinces/years first
    - only filter to total housing rows if such rows exist
    """
    raw = strip_empty(read_csv_flexible(path))
    debug_preview("81528NED RAW", raw, list(raw.columns[:8]), n=5)

    year_col = find_year_col(raw)
    region_col = find_region_col(raw)

    elec_col = select_best_numeric_col(
        raw,
        positive_name_patterns=[
            "gemiddelde_elektriciteitslevering",
            "elektriciteitslevering",
            "electriciteitslevering",
            "elektr",
            "stroom",
        ],
        negative_name_patterns=[
            "netto",
            "gas",
            "aardgas",
            "stadsverwarming",
            "warmte",
            "percentage",
            "aandeel",
        ],
    )

    housing_col = find_optional_text_col(
        raw,
        [
            ["woningkenmerken"],
            ["woningkenmerk"],
            ["woningtype"],
            ["woning"],
        ],
    )

    df = restrict_to_provinces_and_years(raw, year_col, region_col)
    debug_preview("81528NED AFTER YEAR/PROV", df, [region_col, year_col], n=10)

    if df.empty:
        raise ValueError(
            "[81528NED] No rows remained after year/province filtering. "
            "Most likely cause: missing province mapping (e.g. PV20-PV31) "
            "or the wrong region column was selected."
        )

    if housing_col is not None and housing_col in df.columns:
        if DEBUG_EXTRACTORS:
            sample_vals = df[housing_col].dropna().astype(str).head(20).tolist()
            print(f"[DEBUG 81528NED] housing_col={housing_col} sample={sample_vals}")

        df = filter_to_total_if_available(
            df,
            housing_col,
            total_patterns=[
                "totaal woningen",
                "totaal woningtypen",
                "totaal type",
                "alle woningen",
                "totaal",
                "total",
                "all",
            ],
        )

    if df.empty:
        raise ValueError(
            "[81528NED] No province-year rows remained after optional housing filter. "
            "Inspect the housing dimension labels in the raw CSV."
        )

    # Apply conservative total filtering across remaining text dimensions.
    df = maybe_filter_total_dimensions(
        df,
        exclude_cols=[year_col, region_col, "year", "province", elec_col] + ([housing_col] if housing_col else []),
    )

    df["hh_avg_electricity_raw"] = coerce_numeric(df[elec_col])

    out = (
        df.groupby(["province", "year"], as_index=False)["hh_avg_electricity_raw"]
        .mean()
        .sort_values(["year", "province"])
        .reset_index(drop=True)
    )

    if out["hh_avg_electricity_raw"].isna().all():
        raise ValueError(
            "[81528NED] Extracted household electricity column is entirely NaN. "
            "Likely the wrong electricity column was selected."
        )

    assert_complete_panel(out, "hh_avg_electricity_raw", "81528NED")
    return out


def extract_dwelling_counts(path: Path) -> pd.DataFrame:
    """
    Extract dwelling stock from 83704NED.
    """
    raw = strip_empty(read_csv_flexible(path))
    debug_preview("83704NED RAW", raw, list(raw.columns[:8]), n=5)

    year_col = find_year_col(raw)
    region_col = find_region_col(raw)

    stock_col = select_best_numeric_col(
        raw,
        positive_name_patterns=[
            "beginstand_woningvoorraad",
            "woningvoorraad",
            "beginstand",
            "voorraad",
            "woning",
            "dwelling",
            "house",
            "stock",
        ],
        negative_name_patterns=["mutatie", "groei", "percentage", "aandeel"],
    )

    type_col = find_optional_text_col(raw, [["woningtype"], ["woning"], ["type"]])
    area_col = find_optional_text_col(raw, [["oppervlakteklasse"], ["oppervlakte"], ["klasse"]])

    df = restrict_to_provinces_and_years(raw, year_col, region_col)
    if df.empty:
        raise ValueError(
            "[83704NED] No rows remained after year/province filtering. "
            "Check region mapping and detected columns."
        )

    if type_col is not None and type_col in df.columns:
        df = filter_to_total_if_available(
            df,
            type_col,
            total_patterns=["totaal", "total", "all"],
        )

    if area_col is not None and area_col in df.columns:
        df = filter_to_total_if_available(
            df,
            area_col,
            total_patterns=["totaal", "total", "all"],
        )

    df = maybe_filter_total_dimensions(
        df,
        exclude_cols=[year_col, region_col, "year", "province", stock_col],
    )

    df["dwellings_raw"] = coerce_numeric(df[stock_col])

    out = (
        df.groupby(["province", "year"], as_index=False)["dwellings_raw"]
        .max()
        .sort_values(["year", "province"])
        .reset_index(drop=True)
    )

    if out["dwellings_raw"].isna().all():
        raise ValueError(
            "[83704NED] Extracted dwelling-stock column is entirely NaN. "
            "Likely the wrong stock column was selected."
        )

    assert_complete_panel(out, "dwellings_raw", "83704NED")
    return out


def extract_company_electricity(path: Path) -> pd.DataFrame:
    """
    Extract business/non-residential electricity from 82538NED.
    """
    raw = strip_empty(read_csv_flexible(path))
    debug_preview("82538NED RAW", raw, list(raw.columns[:8]), n=5)

    year_col = find_year_col(raw)
    region_col = find_region_col(raw)

    elec_col = select_best_numeric_col(
        raw,
        positive_name_patterns=["elektr", "electric", "stroom"],
        negative_name_patterns=["gas", "aardgas", "warmte", "district", "percentage", "aandeel"],
    )

    df = restrict_to_provinces_and_years(raw, year_col, region_col)
    if df.empty:
        raise ValueError(
            "[82538NED] No rows remained after year/province filtering. "
            "Check region mapping and detected columns."
        )

    # Restrict to total sector row if a sector-like dimension exists and total rows are present.
    num_cols = set(numeric_columns(df))
    for c in df.columns:
        if c in num_cols or c in {year_col, region_col, "year", "province"}:
            continue
        nc = _norm_col(c)
        if any(tok in nc for tok in ["sbi", "bedrijf", "bedrijfstak", "branche", "activiteit", "sector"]):
            df = filter_to_total_if_available(
                df,
                c,
                total_patterns=["totaal", "total", "alle bedrijven", "bedrijven totaal", "all"],
            )

    df = maybe_filter_total_dimensions(
        df,
        exclude_cols=[year_col, region_col, "year", "province", elec_col],
    )

    df["co_electricity_raw"] = coerce_numeric(df[elec_col])

    out = (
        df.groupby(["province", "year"], as_index=False)["co_electricity_raw"]
        .sum()
        .sort_values(["year", "province"])
        .reset_index(drop=True)
    )

    if out["co_electricity_raw"].isna().all():
        raise ValueError(
            "[82538NED] Extracted company electricity column is entirely NaN. "
            "Likely the wrong electricity column was selected."
        )

    assert_complete_panel(out, "co_electricity_raw", "82538NED")
    return out


# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

def convert_hh_avg_to_kwh_per_dwelling(s: pd.Series, mode: str) -> Tuple[pd.Series, str]:
    s = s.astype(float)
    if mode == "kwh_per_dwelling":
        return s, mode
    if mode == "mwh_per_dwelling":
        return s * 1000.0, mode

    med = float(np.nanmedian(s))
    if med < 20:
        return s * 1000.0, "auto->mwh_per_dwelling"
    return s, "auto->kwh_per_dwelling"


def convert_dwellings_to_count(s: pd.Series, mode: str) -> Tuple[pd.Series, str]:
    s = s.astype(float)
    if mode == "count":
        return s, mode
    if mode == "thousand_count":
        return s * 1000.0, mode

    med = float(np.nanmedian(s))
    if med < 10_000:
        return s * 1000.0, "auto->thousand_count"
    return s, "auto->count"


def convert_company_to_mwh(s: pd.Series, mode: str) -> Tuple[pd.Series, str]:
    s = s.astype(float)
    if mode == "mwh":
        return s, mode
    if mode == "kwh":
        return s / 1000.0, mode
    if mode == "mln_kwh":
        return s * 1000.0, mode  # 1 million kWh = 1000 MWh

    med = float(np.nanmedian(s))
    if med < 200_000:
        return s * 1000.0, "auto->mln_kwh"
    return s, "auto->mwh"


# =============================================================================
# SHARE MODELS
# =============================================================================

def fit_softmax_time_model(
    share_df: pd.DataFrame,
    share_col: str,
    train_end_year: int,
    ridge_l2: float,
) -> Dict[str, object]:
    provinces = list(PROVINCES)
    ref = provinces[-1]
    y0 = HIST_START

    train = share_df.loc[share_df["year"] <= train_end_year, ["year", "province", share_col]].copy()
    train = train.pivot(index="year", columns="province", values=share_col).reindex(columns=provinces)

    if train.isna().any().any():
        missing = int(train.isna().sum().sum())
        raise ValueError(f"Missing shares in training data for {share_col}: {missing} cells")

    years = train.index.to_numpy(dtype=float)
    t = years - y0

    S = train.to_numpy(dtype=float)
    S = np.clip(S, EPS_SHARE, 1.0)
    S = S / S.sum(axis=1, keepdims=True)

    ref_idx = provinces.index(ref)
    nonref_idx = [i for i in range(len(provinces)) if i != ref_idx]

    Y = np.log(S[:, nonref_idx] / S[:, [ref_idx]])
    X = np.column_stack([np.ones_like(t), t])

    XtX = X.T @ X
    reg = ridge_l2 * np.eye(X.shape[1])
    B = np.linalg.solve(XtX + reg, X.T @ Y)

    alpha_nonref = B[0, :]
    beta_nonref = B[1, :]

    alpha = np.zeros(len(provinces), dtype=float)
    beta = np.zeros(len(provinces), dtype=float)
    alpha[nonref_idx] = alpha_nonref
    beta[nonref_idx] = beta_nonref

    return {
        "provinces": provinces,
        "ref_province": ref,
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "y0": int(y0),
        "train_years": years.astype(int).tolist(),
    }


def predict_softmax_shares(model: Dict[str, object], years: Iterable[int]) -> pd.DataFrame:
    provinces = model["provinces"]
    alpha = np.array(model["alpha"], dtype=float)
    beta = np.array(model["beta"], dtype=float)
    y0 = int(model["y0"])

    rows = []
    for year in years:
        z = alpha + beta * (year - y0)
        z = z - np.max(z)
        ez = np.exp(z)
        s = ez / ez.sum()
        for p, v in zip(provinces, s):
            rows.append({"year": int(year), "province": p, "share_hat": float(v)})
    return pd.DataFrame(rows)


def fit_omega_series(obs: pd.DataFrame, mode: str) -> pd.DataFrame:
    years_full = list(range(HIST_START, FORECAST_END + 1))
    out = obs.sort_values("year").reset_index(drop=True).copy()
    hist_map = out.set_index("year")[["omega_hh_obs", "omega_co_obs"]]

    rows = []
    if mode == "freeze":
        base_year = int(out.loc[out["year"] <= TRAIN_END_YEAR, "year"].max())
        omega_hh_base = float(out.loc[out["year"] == base_year, "omega_hh_obs"].iloc[0])

        for y in years_full:
            if y in hist_map.index:
                oh = float(hist_map.loc[y, "omega_hh_obs"])
                mode_used = "observed"
            else:
                oh = omega_hh_base
                mode_used = f"freeze_from_{base_year}"
            rows.append({"year": y, "omega_hh": oh, "omega_co": 1.0 - oh, "omega_mode": mode_used})
        return pd.DataFrame(rows)

    if mode == "trend":
        train = out.loc[out["year"] <= TRAIN_END_YEAR, ["year", "omega_hh_obs"]].copy()
        t = train["year"].to_numpy(dtype=float) - HIST_START
        y = np.clip(train["omega_hh_obs"].to_numpy(dtype=float), EPS_SHARE, 1.0 - EPS_SHARE)

        logit = np.log(y / (1.0 - y))
        X = np.column_stack([np.ones_like(t), t])
        reg = RIDGE_L2 * np.eye(2)
        coef = np.linalg.solve(X.T @ X + reg, X.T @ logit)
        a, b = coef

        for yv in years_full:
            if yv in hist_map.index:
                oh = float(hist_map.loc[yv, "omega_hh_obs"])
                mode_used = "observed"
            else:
                tt = yv - HIST_START
                lg = a + b * tt
                oh = 1.0 / (1.0 + np.exp(-lg))
                mode_used = "trend"
            rows.append({"year": yv, "omega_hh": oh, "omega_co": 1.0 - oh, "omega_mode": mode_used})
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported OMEGA_MODE={mode!r}; use 'freeze' or 'trend'")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    for p in [OUT_HIST, OUT_SHARES, OUT_OMEGA, OUT_FACTORS, OUT_DIAG]:
        ensure_parent(p)

    print("[INFO] Extracting household average electricity per dwelling ...")
    hh_avg = extract_household_avg_per_dwelling(FILE_HOUSEHOLDS)
    hh_avg["hh_avg_kwh_per_dwelling"], hh_avg_unit_used = convert_hh_avg_to_kwh_per_dwelling(
        hh_avg["hh_avg_electricity_raw"], HH_AVG_UNIT_RAW
    )

    print("[INFO] Extracting dwelling stock ...")
    dw = extract_dwelling_counts(FILE_DWELLINGS)
    dw["dwellings_count"], dwell_unit_used = convert_dwellings_to_count(
        dw["dwellings_raw"], DWELLINGS_UNIT_RAW
    )

    print("[INFO] Extracting company electricity ...")
    co = extract_company_electricity(FILE_COMPANIES)
    co["co_electricity_MWh_year"], co_unit_used = convert_company_to_mwh(
        co["co_electricity_raw"], CO_UNIT_RAW
    )

    # Historical observed annual totals
    hist = hh_avg.merge(dw, on=["province", "year"], how="inner", validate="one_to_one")
    hist["hh_electricity_MWh_year"] = hist["hh_avg_kwh_per_dwelling"] * hist["dwellings_count"] / 1000.0

    hist = hist.merge(
        co[["province", "year", "co_electricity_MWh_year"]],
        on=["province", "year"],
        how="inner",
        validate="one_to_one",
    )

    # Restrict to a full province-year grid
    full_idx = pd.MultiIndex.from_product(
        [PROVINCES, range(HIST_START, HIST_END + 1)],
        names=["province", "year"],
    )
    hist = hist.set_index(["province", "year"]).reindex(full_idx).reset_index()

    miss_hh = int(hist["hh_electricity_MWh_year"].isna().sum())
    miss_co = int(hist["co_electricity_MWh_year"].isna().sum())
    if miss_hh > 0 or miss_co > 0:
        raise ValueError(
            f"Incomplete province-year panel after merging: "
            f"missing HH cells={miss_hh}, missing CO cells={miss_co}. "
            f"This usually means one extractor still selected the wrong rows or value column."
        )

    # National annual observed totals and omega series
    nat = (
        hist.groupby("year", as_index=False)[["hh_electricity_MWh_year", "co_electricity_MWh_year"]]
        .sum()
        .rename(columns={
            "hh_electricity_MWh_year": "nat_hh_electricity_MWh_year_obs",
            "co_electricity_MWh_year": "nat_co_electricity_MWh_year_obs",
        })
        .sort_values("year")
        .reset_index(drop=True)
    )
    nat["nat_total_electricity_MWh_year_obs"] = (
        nat["nat_hh_electricity_MWh_year_obs"] + nat["nat_co_electricity_MWh_year_obs"]
    )
    nat["omega_hh_obs"] = nat["nat_hh_electricity_MWh_year_obs"] / nat["nat_total_electricity_MWh_year_obs"]
    nat["omega_co_obs"] = 1.0 - nat["omega_hh_obs"]

    # Observed provincial shares
    hist = hist.merge(
        nat[["year", "nat_hh_electricity_MWh_year_obs", "nat_co_electricity_MWh_year_obs"]],
        on="year",
        how="left",
        validate="many_to_one",
    )
    hist["hh_share_obs"] = hist["hh_electricity_MWh_year"] / hist["nat_hh_electricity_MWh_year_obs"]
    hist["co_share_obs"] = hist["co_electricity_MWh_year"] / hist["nat_co_electricity_MWh_year_obs"]

    hist_out = hist[
        [
            "province",
            "year",
            "hh_avg_kwh_per_dwelling",
            "dwellings_count",
            "hh_electricity_MWh_year",
            "co_electricity_MWh_year",
            "hh_share_obs",
            "co_share_obs",
        ]
    ].sort_values(["year", "province"]).reset_index(drop=True)
    hist_out.to_csv(OUT_HIST, index=False)
    print(f"[OK] Saved observed provincial annual sector totals: {OUT_HIST}")

    # Fit share models
    hh_model = fit_softmax_time_model(
        hist_out[["province", "year", "hh_share_obs"]].copy(),
        share_col="hh_share_obs",
        train_end_year=TRAIN_END_YEAR,
        ridge_l2=RIDGE_L2,
    )
    co_model = fit_softmax_time_model(
        hist_out[["province", "year", "co_share_obs"]].copy(),
        share_col="co_share_obs",
        train_end_year=TRAIN_END_YEAR,
        ridge_l2=RIDGE_L2,
    )

    years_full = list(range(HIST_START, FORECAST_END + 1))
    hh_hat = predict_softmax_shares(hh_model, years_full).rename(columns={"share_hat": "hh_share_hat"})
    co_hat = predict_softmax_shares(co_model, years_full).rename(columns={"share_hat": "co_share_hat"})

    shares = (
        hh_hat.merge(co_hat, on=["province", "year"], how="inner", validate="one_to_one")
        .merge(
            hist_out[["province", "year", "hh_share_obs", "co_share_obs"]],
            on=["province", "year"],
            how="left",
        )
        .sort_values(["year", "province"])
        .reset_index(drop=True)
    )

    # Diagnostics: sums must be 1 by year
    hh_sums = (
        shares.groupby("year", as_index=False)["hh_share_hat"]
        .sum()
        .rename(columns={"hh_share_hat": "hh_share_hat_sum"})
    )
    co_sums = (
        shares.groupby("year", as_index=False)["co_share_hat"]
        .sum()
        .rename(columns={"co_share_hat": "co_share_hat_sum"})
    )
    shares = shares.merge(hh_sums, on="year", how="left").merge(co_sums, on="year", how="left")

    shares.to_csv(OUT_SHARES, index=False)
    print(f"[OK] Saved provincial sector shares {HIST_START}-{FORECAST_END}: {OUT_SHARES}")

    # National omega forecast
    omega = fit_omega_series(nat[["year", "omega_hh_obs", "omega_co_obs"]].copy(), OMEGA_MODE)
    omega = omega.merge(
        nat[
            [
                "year",
                "nat_hh_electricity_MWh_year_obs",
                "nat_co_electricity_MWh_year_obs",
                "nat_total_electricity_MWh_year_obs",
            ]
        ],
        on="year",
        how="left",
    ).sort_values("year").reset_index(drop=True)
    omega.to_csv(OUT_OMEGA, index=False)
    print(f"[OK] Saved national sector mix {HIST_START}-{FORECAST_END}: {OUT_OMEGA}")

    # Relative share factors vs base year
    base = shares.loc[shares["year"] == SHARE_FACTOR_BASE_YEAR, ["province", "hh_share_hat", "co_share_hat"]].copy()
    if base.empty:
        raise ValueError(f"SHARE_FACTOR_BASE_YEAR={SHARE_FACTOR_BASE_YEAR} not found in share table")

    base = base.rename(columns={
        "hh_share_hat": "hh_share_hat_base",
        "co_share_hat": "co_share_hat_base",
    })

    factors = shares.merge(base, on="province", how="left", validate="many_to_one")
    factors["hh_share_factor_vs_base"] = factors["hh_share_hat"] / factors["hh_share_hat_base"]
    factors["co_share_factor_vs_base"] = factors["co_share_hat"] / factors["co_share_hat_base"]
    factors.to_csv(OUT_FACTORS, index=False)
    print(f"[OK] Saved provincial share factors: {OUT_FACTORS}")

    diag = {
        "input_files": {
            "companies": str(FILE_COMPANIES),
            "households": str(FILE_HOUSEHOLDS),
            "dwellings": str(FILE_DWELLINGS),
        },
        "historical_window": [HIST_START, HIST_END],
        "forecast_window": [FORECAST_START, FORECAST_END],
        "train_end_year": TRAIN_END_YEAR,
        "omega_mode": OMEGA_MODE,
        "ridge_l2": RIDGE_L2,
        "share_factor_base_year": SHARE_FACTOR_BASE_YEAR,
        "units_used": {
            "household_avg": hh_avg_unit_used,
            "dwellings": dwell_unit_used,
            "companies": co_unit_used,
        },
        "diagnostics": {
            "hh_share_hat_sum_min": float(shares["hh_share_hat_sum"].min()),
            "hh_share_hat_sum_max": float(shares["hh_share_hat_sum"].max()),
            "co_share_hat_sum_min": float(shares["co_share_hat_sum"].min()),
            "co_share_hat_sum_max": float(shares["co_share_hat_sum"].max()),
            "omega_hh_min": float(omega["omega_hh"].min()),
            "omega_hh_max": float(omega["omega_hh"].max()),
            "n_hist_rows": int(len(hist_out)),
            "n_share_rows": int(len(shares)),
        },
        "notes": [
            "Historical overlap starts at HIST_START, currently 2012.",
            "2024 is included in observed outputs but excluded from fitting by default.",
            "If extraction fails, inspect the debug prints for selected columns and sample values.",
            "The most likely one-time adjustment is still the exact numeric column selected in one CBS export.",
        ],
    }
    with open(OUT_DIAG, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    print(f"[OK] Saved diagnostics: {OUT_DIAG}")

    print("\n=== BUILD SUMMARY ===")
    print(f"Observed years      : {HIST_START}-{HIST_END}")
    print(f"Forecast years      : {FORECAST_START}-{FORECAST_END}")
    print(f"Train end year      : {TRAIN_END_YEAR}")
    print(f"Omega mode          : {OMEGA_MODE}")
    print(f"HH unit used        : {hh_avg_unit_used}")
    print(f"Dwellings unit used : {dwell_unit_used}")
    print(f"CO unit used        : {co_unit_used}")
    print(f"HH share sum range  : [{shares['hh_share_hat_sum'].min():.12f}, {shares['hh_share_hat_sum'].max():.12f}]")
    print(f"CO share sum range  : [{shares['co_share_hat_sum'].min():.12f}, {shares['co_share_hat_sum'].max():.12f}]")


if __name__ == "__main__":
    main()