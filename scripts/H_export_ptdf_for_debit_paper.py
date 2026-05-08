#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H_export_ptdf_for_debit_paper.py
================================

Export PTDF-based debit-paper inputs for the annual debit-rule pipeline.

UPDATED VERSION WITH TDTR CANDIDATE-HOUR SUPPORT
------------------------------------------------
This version reads the TDTR candidate-hour support produced by
G_build_tdtr_candidate_hour_support.py and uses it to restrict the constructed
relief requirement layer to candidate scarcity hours.

Interpretation
--------------
- The raw PTDF calculation is used to create a constructed public-data relief
  testbed.
- The candidate-hour support is a public/auditable TDTR window-selection layer.
- H builds branch-hour relief requirements only inside the candidate support.
- The same support is used later by A to zero out participant availability
  outside candidate hours. Therefore M1, M2, and M3 are compared on the same
  TDTR-eligible physical relief instance.

This is not an operator-grade Dutch congestion reconstruction. It is a controlled
public-data testbed for comparing annual allowance-consumption laws.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR
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


DEFAULT_PROVINCES = [
    "Groningen", "Friesland", "Drenthe", "Overijssel", "Flevoland", "Gelderland",
    "Utrecht", "Noord-Holland", "Zuid-Holland", "Zeeland", "Noord-Brabant", "Limburg",
]

DERIVED_DIR = Path(cfg_get("DERIVED_DIR", REPO_ROOT / "data" / "derived"))
PTDF_PUBLICATION_DIR = Path(cfg_get(
    "PTDF_PUBLICATION_DIR",
    os.environ.get(
        "PTDF_PUBLICATION_DIR",
        r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\Netherlands_Data\PTDF_OUTPUT_PUBLICATION",
    ),
))
STATE_PANEL_CSV = Path(cfg_get(
    "AUDITABLE_STATE_PANEL_PATH",
    cfg_get("STATE_PANEL_PATH", DERIVED_DIR / "auditable_state_province_hour.csv"),
))
PROVINCE_BUS_WEIGHTS_CSV = Path(cfg_get("PROVINCE_BUS_WEIGHTS_PATH", PTDF_PUBLICATION_DIR / "province_to_bus_weights.csv"))
OUTPUT_DIR = Path(cfg_get("PTDF_DEBIT_INPUT_DIR", REPO_ROOT / "PTDF_OUTPUT_PUBLICATION" / "debit_paper_inputs"))

YEARS = sorted(dict.fromkeys(map(int, cfg_get("ALL_YEARS", [2021, 2022, 2023, 2024, 2025]))))
PROVINCES = list(cfg_get("NL_PROVINCES", DEFAULT_PROVINCES))

# TDTR support.
TDTR_SUPPORT_SCENARIO = str(cfg_get("TDTR_SUPPORT_SCENARIO", "medium")).strip()
TDTR_SUPPORT_REQUIRED = cfg_bool("TDTR_SUPPORT_REQUIRED", True)
APPLY_TDTR_SUPPORT_TO_R = cfg_bool("PTDF_APPLY_TDTR_SUPPORT_TO_REQUIREMENTS", True)
TDTR_SUPPORT_PATH = Path(cfg_get("TDTR_SUPPORT_PATH", DERIVED_DIR / "tdtr_candidate_hour_support.csv"))
TDTR_SUPPORT_SCENARIO_PATH = Path(cfg_get(
    "TDTR_SUPPORT_SCENARIO_PATH",
    DERIVED_DIR / f"tdtr_candidate_hour_support__{TDTR_SUPPORT_SCENARIO}.csv",
))
TDTR_SUPPORT_ALL_SCENARIOS_PATH = Path(cfg_get(
    "TDTR_SUPPORT_ALL_SCENARIOS_PATH",
    DERIVED_DIR / "tdtr_candidate_hour_support_all_scenarios.csv",
))
TDTR_SUPPORT_MASK_COLUMN_CANDIDATES = [
    "selected", "is_selected", "is_tdtr_candidate", "tdtr_candidate", "support", "y_support", "candidate_hour",
]

RHO = float(cfg_get("PTDF_RHO", 0.70))
MAX_ACTIVE_BRANCHES = int(cfg_get("PTDF_MAX_ACTIVE_BRANCHES", 40))
SCREENING_SCORE = str(cfg_get("PTDF_SCREENING_SCORE", "max_loading_frac")).lower().strip()
DEFAULT_RATING_MVA = float(cfg_get("PTDF_DEFAULT_RATING_MVA", 1000.0))
USE_VOLTAGE_PROXY_RATINGS = cfg_bool("PTDF_USE_VOLTAGE_PROXY_RATINGS", True)

REQUIREMENT_SCALING_MODE = str(cfg_get("PTDF_REQUIREMENT_SCALING_MODE", "annual_public_load_fraction")).lower().strip()
ANNUAL_RELIEF_FRACTION = float(cfg_get("PTDF_ANNUAL_RELIEF_FRACTION", 0.02))
REQUIREMENT_CAP_QUANTILE_RAW = cfg_get("PTDF_REQUIREMENT_CAP_QUANTILE", 0.995)
REQUIREMENT_CAP_QUANTILE = None if REQUIREMENT_CAP_QUANTILE_RAW in [None, "", 0, "0", "none", "None"] else float(REQUIREMENT_CAP_QUANTILE_RAW)
MIN_REQUIREMENT_MW = float(cfg_get("PTDF_MIN_REQUIREMENT_MW", 1e-9))
MIN_COEFFICIENT = float(cfg_get("PTDF_MIN_COEFFICIENT", 1e-12))


def hours_in_year_utc(year: int) -> int:
    return int((pd.Timestamp(f"{year+1}-01-01T00:00:00Z") - pd.Timestamp(f"{year}-01-01T00:00:00Z")) / pd.Timedelta(hours=1))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_column(columns: Iterable[str], candidates: Iterable[str], required: bool = True) -> Optional[str]:
    cols = [str(c).strip() for c in columns]
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of {list(candidates)!r}. Available columns: {cols!r}")
    return None


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    print(f"[SAVE] {path}")


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[SAVE] {path}")


def is_meta_column(c: str) -> bool:
    cl = str(c).lower()
    return any(tok in cl for tok in ["branch", "line", "from", "to", "rating", "limit", "loading", "score", "active", "rank", "hours"])


# =============================================================================
# TDTR SUPPORT
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
        "apply_to_R": APPLY_TDTR_SUPPORT_TO_R,
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
        print("[WARN] No TDTR support file found. H will use full-year support.")
        return pd.DataFrame(), meta

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "scenario" in df.columns and path.name.endswith("all_scenarios.csv") and TDTR_SUPPORT_SCENARIO:
        before = len(df)
        df = df[df["scenario"].astype(str).str.lower() == TDTR_SUPPORT_SCENARIO.lower()].copy()
        print(f"[TDTR SUPPORT] filtered all-scenarios support to scenario={TDTR_SUPPORT_SCENARIO}: {before:,} -> {len(df):,}")

    y_col = pick_column(df.columns, ["year"])
    h_col = pick_column(df.columns, ["hour_of_year", "hour", "hoy"])
    ts_col = pick_column(df.columns, ["timestamp_utc", "timestamp", "utc_time", "time_utc"], required=False)
    mask_col = pick_column(df.columns, TDTR_SUPPORT_MASK_COLUMN_CANDIDATES, required=False)
    if mask_col is None:
        raise ValueError(
            f"TDTR support file {path} has no support mask column. Tried {TDTR_SUPPORT_MASK_COLUMN_CANDIDATES}. "
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
        out["timestamp_utc"] = pd.to_datetime(df.loc[out.index, ts_col], utc=True, errors="coerce").dt.floor("h")
    else:
        out["timestamp_utc"] = pd.to_datetime(out["year"].astype(str) + "-01-01T00:00:00Z", utc=True) + pd.to_timedelta(out["hour_of_year"] - 1, unit="h")

    out = (
        out[["year", "hour_of_year", "timestamp_utc", "tdtr_support"]]
        .groupby(["year", "hour_of_year", "timestamp_utc"], as_index=False)["tdtr_support"]
        .max()
        .sort_values(["year", "hour_of_year"])
        .reset_index(drop=True)
    )
    summary = out.groupby("year", as_index=False).agg(hours=("hour_of_year", "nunique"), selected=("tdtr_support", "sum"))
    summary["selected_share"] = summary["selected"] / summary["hours"].replace(0, np.nan)
    print(f"[TDTR SUPPORT] using {path}")
    print(summary.to_string(index=False))
    meta.update({
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
    })
    return out, meta


def validate_support_year(support: pd.DataFrame, year: int) -> pd.DataFrame:
    H = hours_in_year_utc(year)
    expected = pd.DataFrame({
        "year": year,
        "hour_of_year": np.arange(1, H + 1, dtype=int),
        "timestamp_utc": pd.date_range(pd.Timestamp(f"{year}-01-01T00:00:00Z"), periods=H, freq="h"),
    })
    if support.empty:
        expected["tdtr_support"] = 1
        return expected
    sub = support.loc[support["year"] == int(year), ["year", "hour_of_year", "tdtr_support"]].copy()
    if sub.empty and TDTR_SUPPORT_REQUIRED:
        raise ValueError(f"TDTR support has no rows for year={year}.")
    out = expected.merge(sub, on=["year", "hour_of_year"], how="left")
    missing = int(out["tdtr_support"].isna().sum())
    if missing > 0 and TDTR_SUPPORT_REQUIRED:
        raise ValueError(f"TDTR support year={year} is incomplete: missing {missing} hours out of {H}.")
    out["tdtr_support"] = out["tdtr_support"].fillna(0).astype(int)
    return out


# =============================================================================
# PTDF matrix
# =============================================================================
def find_ptdf_file() -> Path:
    names = [
        "ac_ptdf_wide.csv", "ptdf_matrix.csv", "PTDF_matrix.csv", "ptdf_public_matrix.csv",
        "PTDF_public_matrix.csv", "branch_bus_ptdf.csv", "Branch_Bus_PTDF.csv", "ptdf.csv", "PTDF.csv",
        "ac_ptdf_long.csv", "ptdf_long.csv", "PTDF_long.csv", "ac_ptdf_matrices.npz",
    ]
    for name in names:
        p = PTDF_PUBLICATION_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No PTDF matrix found in {PTDF_PUBLICATION_DIR}. Available: {[p.name for p in PTDF_PUBLICATION_DIR.glob('*')]}")


def load_ptdf_matrix() -> tuple[pd.DataFrame, Path]:
    p = find_ptdf_file()
    if p.suffix.lower() == ".npz":
        z = np.load(p, allow_pickle=True)
        key = next((k for k in ["ptdf", "PTDF", "matrix", "ac_ptdf"] if k in z.files), None)
        if key is None:
            raise ValueError(f"{p} has no PTDF matrix key. Keys={z.files}")
        mat = np.asarray(z[key], dtype=float)
        branch_ids = None
        for k in ["branch_ids", "branches", "branch_id"]:
            if k in z.files:
                branch_ids = [str(x) for x in z[k].tolist()]
                break
        if branch_ids is None:
            branch_ids = [str(i) for i in range(mat.shape[0])]
        bus_ids = None
        for k in ["bus_ids", "buses", "bus_names"]:
            if k in z.files:
                bus_ids = [str(x) for x in z[k].tolist()]
                break
        if bus_ids is None:
            bus_ids = [str(i) for i in range(mat.shape[1])]
        return pd.DataFrame(mat, index=branch_ids, columns=bus_ids), p

    if "long" in p.name.lower():
        df = pd.read_csv(p, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        branch_col = pick_column(df.columns, ["branch_id", "branch", "line_id", "line"])
        bus_col = pick_column(df.columns, ["bus_id", "bus", "bus_name"])
        val_col = pick_column(df.columns, ["ptdf", "PTDF", "value", "coefficient", "coef"])
        df[branch_col] = df[branch_col].astype(str).str.strip()
        df[bus_col] = df[bus_col].astype(str).str.strip()
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
        out = df.pivot_table(index=branch_col, columns=bus_col, values=val_col, aggfunc="mean", fill_value=0.0)
        out.index.name = "branch_id"
        return out.astype(float), p

    df = pd.read_csv(p, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    branch_col = pick_column(df.columns, ["branch_id", "branch", "line_id", "line", "id", "branch_i"], required=False)
    if branch_col is None:
        df.insert(0, "branch_id", [str(i) for i in range(len(df))])
        branch_col = "branch_id"
    bus_cols = []
    for c in df.columns:
        if c == branch_col or is_meta_column(c):
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
            bus_cols.append(c)
    if not bus_cols:
        raise ValueError(f"Could not infer bus columns from {p}.")
    out = df[[branch_col] + bus_cols].copy()
    out[branch_col] = out[branch_col].astype(str).str.strip()
    for c in bus_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out = out.drop_duplicates(branch_col).set_index(branch_col)
    out.index.name = "branch_id"
    out.columns = [str(c).strip() for c in out.columns]
    return out.astype(float), p


# =============================================================================
# Branch metadata / ratings
# =============================================================================
def find_branch_metadata() -> Optional[Path]:
    for name in ["ac_branches_main_component.csv", "ac_branches_all.csv", "branches_main_component.csv", "branches.csv", "branch_metadata.csv"]:
        p = PTDF_PUBLICATION_DIR / name
        if p.exists():
            return p
    return None


def voltage_from_row(row: pd.Series) -> Optional[float]:
    for c in row.index:
        cl = str(c).lower()
        if any(tok in cl for tok in ["v_nom", "voltage", "vn_kv", "kv", "nominal"]):
            val = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
            if pd.notna(val) and 20 <= float(val) <= 1000:
                return float(val)
    text = " ".join(str(v) for v in row.dropna().tolist())
    vals = []
    for m in re.findall(r"(\d{2,3}(?:\.\d+)?)\s*kV", text, flags=re.IGNORECASE):
        v = float(m)
        if 20 <= v <= 1000:
            vals.append(v)
    return max(vals) if vals else None


def proxy_rating(voltage_kv: Optional[float]) -> tuple[float, str]:
    if voltage_kv is None or not np.isfinite(voltage_kv):
        return DEFAULT_RATING_MVA, f"default:{DEFAULT_RATING_MVA}"
    v = float(voltage_kv)
    if v >= 380:
        return 2500.0, f"voltage_proxy:{v:g}kV"
    if v >= 220:
        return 1500.0, f"voltage_proxy:{v:g}kV"
    if v >= 150:
        return 1000.0, f"voltage_proxy:{v:g}kV"
    if v >= 110:
        return 700.0, f"voltage_proxy:{v:g}kV"
    if v >= 50:
        return 300.0, f"voltage_proxy:{v:g}kV"
    return DEFAULT_RATING_MVA, f"default:{DEFAULT_RATING_MVA}"


def load_branch_meta(branch_ids: list[str]) -> tuple[pd.DataFrame, Optional[Path]]:
    base = pd.DataFrame({"branch_id": [str(b) for b in branch_ids]})
    p = find_branch_metadata()
    if p is None:
        base["rating_mva"] = DEFAULT_RATING_MVA
        base["rating_source"] = f"default:{DEFAULT_RATING_MVA}"
        return base, None

    raw = pd.read_csv(p, low_memory=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    branch_col = pick_column(raw.columns, ["branch_id", "branch", "line_id", "line", "id", "branch_i"], required=False)
    if branch_col is None:
        raw.insert(0, "branch_id", [str(i) for i in range(len(raw))])
        branch_col = "branch_id"
    raw["branch_id"] = raw[branch_col].astype(str).str.strip()
    meta = base.merge(raw.drop_duplicates("branch_id"), on="branch_id", how="left")

    rating_col = pick_column(meta.columns, [
        "rating_mva", "rate_mva", "rateA", "rate_a", "rate_a_mva", "thermal_limit_mva",
        "thermal_rating_mva", "s_nom", "s_nom_mva", "limit_mva", "capacity_mva",
    ], required=False)

    ratings, sources, voltages = [], [], []
    for _, row in meta.iterrows():
        val = np.nan
        if rating_col is not None:
            val = pd.to_numeric(pd.Series([row.get(rating_col)]), errors="coerce").iloc[0]
        if pd.notna(val) and float(val) > 0:
            ratings.append(float(val)); sources.append(f"metadata:{rating_col}"); voltages.append(voltage_from_row(row))
        elif USE_VOLTAGE_PROXY_RATINGS:
            v = voltage_from_row(row)
            r, s = proxy_rating(v)
            ratings.append(r); sources.append(s); voltages.append(v)
        else:
            ratings.append(DEFAULT_RATING_MVA); sources.append(f"default:{DEFAULT_RATING_MVA}"); voltages.append(None)
    meta["rating_mva"] = ratings
    meta["rating_source"] = sources
    meta["voltage_kv_inferred"] = voltages
    return meta, p


# =============================================================================
# State panel and weights
# =============================================================================
def load_state_panel() -> tuple[pd.DataFrame, str, str]:
    if not STATE_PANEL_CSV.exists():
        raise FileNotFoundError(f"Missing state panel: {STATE_PANEL_CSV}. Run the auditable state pipeline first.")
    df = pd.read_csv(STATE_PANEL_CSV, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    province_col = pick_column(df.columns, ["province", "provincie"])
    ts_col = pick_column(df.columns, ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc"], required=False)
    df["province"] = df[province_col].astype(str).str.strip()
    if ts_col is not None:
        df["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.floor("h")
    else:
        y = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        h_col = pick_column(df.columns, ["hour_of_year", "hour", "hoy"])
        h = pd.to_numeric(df[h_col], errors="coerce").astype("Int64")
        df["timestamp_utc"] = pd.to_datetime(y.astype(str) + "-01-01T00:00:00Z", utc=True) + pd.to_timedelta(h.astype(float) - 1.0, unit="h")
    df = df.dropna(subset=["timestamp_utc"]).copy()
    df["year"] = df["timestamp_utc"].dt.year.astype(int)

    mw_col = pick_column(df.columns, ["ptdf_load_pressure_MW", "ptdf_load_pressure_mw"], required=False)
    if mw_col is None:
        mw_col = pick_column(df.columns, ["prov_load_MW", "prov_load_mw", "RL_MW", "residual_load_MW", "rl_mw", "RL0_MW"])
        source = f"fallback:{mw_col}"
    else:
        source = "state_panel:ptdf_load_pressure_MW"
    df[mw_col] = pd.to_numeric(df[mw_col], errors="coerce")
    if df[mw_col].isna().any():
        raise ValueError(f"{mw_col} contains NaNs.")
    if "ptdf_load_pressure_source" in df.columns:
        source = "; ".join(f"{k}={v}" for k, v in df["ptdf_load_pressure_source"].astype(str).value_counts().to_dict().items())
    return df, mw_col, source


def validate_year_state(panel: pd.DataFrame, year: int, mw_col: str) -> pd.DataFrame:
    H = hours_in_year_utc(year)
    ypanel = panel.loc[(panel["year"] == year) & (panel["province"].isin(PROVINCES))].copy()
    if len(ypanel) != H * len(PROVINCES):
        raise ValueError(f"State panel year {year}: rows={len(ypanel):,}, expected={H * len(PROVINCES):,}.")
    parts = []
    expected = pd.date_range(pd.Timestamp(f"{year}-01-01T00:00:00Z"), periods=H, freq="h")
    for p, g in ypanel.groupby("province", sort=False):
        gg = g.sort_values("timestamp_utc").copy()
        if len(gg) != H or not pd.DatetimeIndex(gg["timestamp_utc"]).equals(expected):
            raise ValueError(f"State panel year {year}, province {p}: not a complete UTC hourly year.")
        gg["hour_of_year"] = np.arange(1, H + 1, dtype=int)
        parts.append(gg)
    out = pd.concat(parts, ignore_index=True)
    out[mw_col] = pd.to_numeric(out[mw_col], errors="coerce").fillna(0.0).astype(float)
    return out


def load_weights(bus_ids: list[str]) -> pd.DataFrame:
    if not PROVINCE_BUS_WEIGHTS_CSV.exists():
        raise FileNotFoundError(f"Missing province-to-bus weights: {PROVINCE_BUS_WEIGHTS_CSV}")
    df = pd.read_csv(PROVINCE_BUS_WEIGHTS_CSV, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if {"province", "bus_id", "weight"}.issubset(df.columns):
        long = df[["province", "bus_id", "weight"]].copy()
    else:
        prov_col = pick_column(df.columns, ["province", "provincie"], required=False)
        if prov_col is None:
            raise ValueError("Weights file must be long province,bus_id,weight or wide with province column.")
        rows = []
        for _, row in df.iterrows():
            for b in bus_ids:
                if b in df.columns:
                    rows.append({"province": row[prov_col], "bus_id": b, "weight": row[b]})
        long = pd.DataFrame(rows)
    long["province"] = long["province"].astype(str).str.strip()
    long["bus_id"] = long["bus_id"].astype(str).str.strip()
    long["weight"] = pd.to_numeric(long["weight"], errors="coerce").fillna(0.0)

    panel = pd.MultiIndex.from_product([PROVINCES, bus_ids], names=["province", "bus_id"]).to_frame(index=False)
    out = panel.merge(long, on=["province", "bus_id"], how="left")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    parts = []
    missing = []
    for p, g in out.groupby("province", sort=False):
        gg = g.copy()
        s = float(gg["weight"].sum())
        if s <= 0:
            missing.append(p)
        else:
            gg["weight"] /= s
        parts.append(gg)
    if missing:
        raise ValueError(f"Province-to-bus mapping incomplete. Missing positive weights for {missing}")
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# Year export
# =============================================================================
def active_score(summary: pd.DataFrame) -> pd.Series:
    if SCREENING_SCORE in summary.columns:
        return summary[SCREENING_SCORE]
    if SCREENING_SCORE == "p99":
        return summary["p99_loading_frac"]
    return summary["max_loading_frac"]


def export_year(year: int, ptdf: pd.DataFrame, branch_meta: pd.DataFrame, state_year: pd.DataFrame, mw_col: str, weights: pd.DataFrame, support_y: pd.DataFrame) -> dict:
    H = hours_in_year_utc(year)
    branch_ids = list(ptdf.index.astype(str))
    bus_ids = list(ptdf.columns.astype(str))

    support_vec = support_y.sort_values("hour_of_year")["tdtr_support"].to_numpy(dtype=float)
    if support_vec.size != H:
        raise ValueError(f"Support vector length for year={year} is {support_vec.size}, expected {H}.")
    selected_hours = int(support_vec.sum())

    load_hp = (
        state_year.pivot_table(index="hour_of_year", columns="province", values=mw_col, aggfunc="mean")
        .reindex(index=np.arange(1, H + 1), columns=PROVINCES)
        .fillna(0.0)
        .astype(float)
    )
    W = (
        weights.pivot_table(index="province", columns="bus_id", values="weight", aggfunc="sum", fill_value=0.0)
        .reindex(index=PROVINCES, columns=bus_ids)
        .fillna(0.0)
        .astype(float)
    )

    bus_pressure_hb = load_hp.to_numpy(float) @ W.to_numpy(float)       # H x B
    PTDF_kb = ptdf.reindex(index=branch_ids, columns=bus_ids).fillna(0.0).to_numpy(float)
    flow_kh = PTDF_kb @ bus_pressure_hb.T                               # K x H

    ratings = branch_meta.set_index("branch_id")["rating_mva"].reindex(branch_ids).fillna(DEFAULT_RATING_MVA).astype(float).to_numpy()
    ratings = np.where(ratings > 0, ratings, DEFAULT_RATING_MVA)
    loading_kh = np.abs(flow_kh) / ratings[:, None]

    summary = pd.DataFrame({
        "branch_id": branch_ids,
        "rating_mva": ratings,
        "max_loading_frac": loading_kh.max(axis=1),
        "p99_loading_frac": np.quantile(loading_kh, 0.99, axis=1),
        "mean_loading_frac": loading_kh.mean(axis=1),
        "hours_over_rho": (loading_kh > RHO).sum(axis=1).astype(int),
        "hours_over_rho_in_tdtr_support": ((loading_kh > RHO) & (support_vec[None, :] > 0.5)).sum(axis=1).astype(int),
    })
    meta_cols = [c for c in ["branch_id", "rating_source", "voltage_kv_inferred", "from_bus", "to_bus", "name", "line_name"] if c in branch_meta.columns]
    summary = summary.merge(branch_meta[meta_cols].drop_duplicates("branch_id"), on="branch_id", how="left")
    summary["rating_source"] = summary["rating_source"].fillna(f"default:{DEFAULT_RATING_MVA}")
    summary["selection_score"] = active_score(summary)
    summary = summary.sort_values(["selection_score", "hours_over_rho", "max_loading_frac"], ascending=[False, False, False]).reset_index(drop=True)
    summary["active_rank"] = np.arange(1, len(summary) + 1, dtype=int)
    active_ids = summary.head(MAX_ACTIVE_BRANCHES)["branch_id"].astype(str).tolist()
    summary["is_active"] = summary["branch_id"].astype(str).isin(active_ids)

    active_idx = [branch_ids.index(b) for b in active_ids]
    flow_ah = flow_kh[active_idx, :]
    ratings_a = ratings[active_idx]
    loading_ah = np.abs(flow_ah) / ratings_a[:, None]
    raw_excess_full_ah = np.maximum(np.abs(flow_ah) - RHO * ratings_a[:, None], 0.0)

    if APPLY_TDTR_SUPPORT_TO_R:
        raw_excess_ah = raw_excess_full_ah * support_vec[None, :]
    else:
        raw_excess_ah = raw_excess_full_ah.copy()

    raw_total_R_full_year = float(raw_excess_full_ah.sum())
    raw_total_R_after_support = float(raw_excess_ah.sum())
    annual_pressure_volume_full_year = float(load_hp.to_numpy(float).sum())
    annual_pressure_volume_support = float((load_hp.to_numpy(float) * support_vec[:, None]).sum())

    if REQUIREMENT_SCALING_MODE in {"raw", "unscaled", "none"}:
        target_total_R = raw_total_R_after_support
        scale_factor = 1.0
        R_ah = raw_excess_ah.copy()
    elif REQUIREMENT_SCALING_MODE in {"annual_public_load_fraction", "controlled_annual_volume"}:
        # When TDTR support is imposed, scale to a fraction of the public load pressure
        # inside the candidate support. This keeps the constructed relief volume aligned
        # with the restricted operating window instead of the whole year.
        pressure_base = annual_pressure_volume_support if APPLY_TDTR_SUPPORT_TO_R else annual_pressure_volume_full_year
        target_total_R = ANNUAL_RELIEF_FRACTION * pressure_base
        scale_factor = target_total_R / raw_total_R_after_support if raw_total_R_after_support > 0 else 0.0
        R_ah = raw_excess_ah * scale_factor
    else:
        raise ValueError(f"Unknown PTDF_REQUIREMENT_SCALING_MODE={REQUIREMENT_SCALING_MODE}")

    cap_value = None
    cap_rebalance_factor = 1.0
    if REQUIREMENT_CAP_QUANTILE is not None and R_ah.max() > 0:
        positive = R_ah[R_ah > MIN_REQUIREMENT_MW]
        if positive.size > 0:
            cap_value = float(np.quantile(positive, REQUIREMENT_CAP_QUANTILE))
            capped = np.minimum(R_ah, cap_value)
            capped_total = float(capped.sum())
            if capped_total > 0 and target_total_R > 0:
                cap_rebalance_factor = target_total_R / capped_total
                R_ah = capped * cap_rebalance_factor

    # Strictly enforce support after cap/rebalance. Numerical rebalance cannot create
    # positive values outside support, but this guard makes the contract explicit.
    if APPLY_TDTR_SUPPORT_TO_R:
        R_ah = R_ah * support_vec[None, :]

    scaled_total_R = float(R_ah.sum())

    province_sens_kp = PTDF_kb @ W.to_numpy(float).T                    # K x P
    province_sens_ap = province_sens_kp[active_idx, :]
    timestamps = pd.date_range(pd.Timestamp(f"{year}-01-01T00:00:00Z"), periods=H, freq="h")

    req_rows = []
    coef_rows = []
    for a, branch_id in enumerate(active_ids):
        mask = R_ah[a, :] > MIN_REQUIREMENT_MW
        if not np.any(mask):
            continue
        for h0 in np.nonzero(mask)[0]:
            hour = int(h0 + 1)
            if APPLY_TDTR_SUPPORT_TO_R and support_vec[h0] <= 0.5:
                raise RuntimeError("Internal error: positive requirement outside TDTR support.")
            req_rows.append({
                "branch_id": branch_id,
                "timestamp_utc": timestamps[h0],
                "hour_of_year": hour,
                "tdtr_support": int(support_vec[h0] > 0.5),
                "R_MW": float(R_ah[a, h0]),
                "raw_excess_MW": float(raw_excess_full_ah[a, h0]),
                "raw_excess_after_tdtr_support_MW": float(raw_excess_ah[a, h0]),
                "raw_flow_MW": float(flow_ah[a, h0]),
                "loading_frac": float(loading_ah[a, h0]),
                "requirement_scale_factor": float(scale_factor),
            })
            sign = 1.0 if flow_ah[a, h0] >= 0 else -1.0
            coeffs = np.maximum(sign * province_sens_ap[a, :], 0.0)
            for province, coef in zip(PROVINCES, coeffs):
                if coef > MIN_COEFFICIENT:
                    coef_rows.append({
                        "branch_id": branch_id,
                        "timestamp_utc": timestamps[h0],
                        "hour_of_year": hour,
                        "province": province,
                        "coefficient": float(coef),
                    })

    requirements = pd.DataFrame(req_rows, columns=[
        "branch_id", "timestamp_utc", "hour_of_year", "tdtr_support", "R_MW", "raw_excess_MW",
        "raw_excess_after_tdtr_support_MW", "raw_flow_MW", "loading_frac", "requirement_scale_factor",
    ])
    coefficients = pd.DataFrame(coef_rows, columns=["branch_id", "timestamp_utc", "hour_of_year", "province", "coefficient"])

    active_table = summary.sort_values("active_rank").copy()
    if not requirements.empty:
        byb = requirements.groupby("branch_id", as_index=False).agg(
            total_R_MW_h=("R_MW", "sum"),
            max_R_MW=("R_MW", "max"),
            positive_requirement_hours=("R_MW", "size"),
        )
        active_table = active_table.merge(byb, on="branch_id", how="left")
    for c in ["total_R_MW_h", "max_R_MW", "positive_requirement_hours"]:
        if c not in active_table.columns:
            active_table[c] = 0
        active_table[c] = active_table[c].fillna(0)

    summary_obj = {
        "year": int(year),
        "n_hours": int(H),
        "tdtr_support_scenario": TDTR_SUPPORT_SCENARIO,
        "tdtr_support_selected_hours": int(selected_hours),
        "tdtr_support_selected_share": float(selected_hours / H),
        "apply_tdtr_support_to_R": bool(APPLY_TDTR_SUPPORT_TO_R),
        "n_branches_total": int(len(branch_ids)),
        "n_active_branches": int(len(active_ids)),
        "rho": float(RHO),
        "screening_score": SCREENING_SCORE,
        "requirement_scaling_mode": REQUIREMENT_SCALING_MODE,
        "annual_relief_fraction": float(ANNUAL_RELIEF_FRACTION),
        "annual_public_pressure_volume_full_year_MWh_proxy": annual_pressure_volume_full_year,
        "annual_public_pressure_volume_tdtr_support_MWh_proxy": annual_pressure_volume_support,
        "raw_total_excess_full_year_MW_h": raw_total_R_full_year,
        "raw_total_excess_after_tdtr_support_MW_h": raw_total_R_after_support,
        "target_total_requirement_MW_h": float(target_total_R),
        "scaled_total_requirement_MW_h": scaled_total_R,
        "initial_requirement_scale_factor": float(scale_factor),
        "requirement_cap_quantile": REQUIREMENT_CAP_QUANTILE,
        "requirement_cap_value_before_rebalance_MW": cap_value,
        "cap_rebalance_factor": float(cap_rebalance_factor),
        "positive_branch_hour_requirements": int(len(requirements)),
        "coefficient_rows": int(len(coefficients)),
        "max_raw_loading_frac": float(summary["max_loading_frac"].max()),
        "p99_raw_loading_frac_max_over_branches": float(summary["p99_loading_frac"].max()),
        "rating_source_counts": summary["rating_source"].astype(str).value_counts().to_dict(),
        "interpretation_note": "Raw PTDF loading screens branches. R_MW is a TDTR-support-filtered constructed relief requirement, not operator redispatch volume.",
    }
    return {
        "active_branches": active_table,
        "requirements": requirements,
        "coefficients": coefficients,
        "flow_summary": summary,
        "summary": summary_obj,
    }


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    print("=" * 78)
    print("H: EXPORT PTDF DEBIT-PAPER INPUTS")
    print("=" * 78)
    print(f"PTDF publication dir : {PTDF_PUBLICATION_DIR}")
    print(f"State panel          : {STATE_PANEL_CSV}")
    print(f"Province-bus weights : {PROVINCE_BUS_WEIGHTS_CSV}")
    print(f"Output dir           : {OUTPUT_DIR}")
    print(f"Years                : {YEARS}")
    print(f"TDTR scenario        : {TDTR_SUPPORT_SCENARIO}")
    print(f"Apply support to R   : {APPLY_TDTR_SUPPORT_TO_R}")
    print(f"rho                  : {RHO}")
    print(f"max active branches  : {MAX_ACTIVE_BRANCHES}")
    print(f"requirement scaling  : {REQUIREMENT_SCALING_MODE}")
    print(f"annual relief frac   : {ANNUAL_RELIEF_FRACTION}")
    print("=" * 78)

    support, support_meta = load_tdtr_support()
    ptdf, ptdf_path = load_ptdf_matrix()
    ptdf.index = ptdf.index.astype(str)
    ptdf.columns = ptdf.columns.astype(str)
    branch_ids = list(ptdf.index.astype(str))
    bus_ids = list(ptdf.columns.astype(str))

    branch_meta, branch_meta_path = load_branch_meta(branch_ids)
    branch_meta["branch_id"] = branch_meta["branch_id"].astype(str)
    panel, mw_col, mw_source = load_state_panel()
    weights = load_weights(bus_ids)

    slack_bus = None
    net_manifest = PTDF_PUBLICATION_DIR / "network_manifest.json"
    if net_manifest.exists():
        try:
            with net_manifest.open("r", encoding="utf-8") as f:
                d = json.load(f)
            for key in ["slack_bus", "slack_bus_id", "slack"]:
                if key in d:
                    slack_bus = d[key]
                    break
        except Exception:
            slack_bus = None

    print(f"[INFO] PTDF matrix file    : {ptdf_path}")
    print(f"[INFO] PTDF shape          : {ptdf.shape[0]} branches x {ptdf.shape[1]} buses")
    print(f"[INFO] Active buses        : {len(bus_ids)}")
    print(f"[INFO] Slack bus           : {slack_bus if slack_bus is not None else 'not found'}")
    print(f"[INFO] Branch metadata     : {branch_meta_path if branch_meta_path else 'not found'}")
    print(f"[INFO] Branch rating source: {branch_meta['rating_source'].astype(str).value_counts().to_dict()}")
    print(f"[INFO] State panel rows    : {len(panel):,}")
    print(f"[INFO] PTDF MW column      : {mw_col}")
    print(f"[INFO] PTDF MW source      : {mw_source}")
    print(f"[INFO] Province-bus W shape: {len(PROVINCES)} provinces x {len(bus_ids)} buses")

    manifest = {
        "script": Path(__file__).name,
        "repo_root": str(REPO_ROOT),
        "ptdf_publication_dir": str(PTDF_PUBLICATION_DIR),
        "ptdf_matrix_file": str(ptdf_path),
        "branch_metadata_file": str(branch_meta_path) if branch_meta_path else None,
        "state_panel_csv": str(STATE_PANEL_CSV),
        "province_bus_weights_csv": str(PROVINCE_BUS_WEIGHTS_CSV),
        "output_dir": str(OUTPUT_DIR),
        "years": YEARS,
        "tdtr_support": support_meta,
        "rho": RHO,
        "max_active_branches": MAX_ACTIVE_BRANCHES,
        "requirement_scaling_mode": REQUIREMENT_SCALING_MODE,
        "annual_relief_fraction": ANNUAL_RELIEF_FRACTION,
        "requirement_cap_quantile": REQUIREMENT_CAP_QUANTILE,
        "ptdf_mw_column": mw_col,
        "ptdf_mw_source": mw_source,
        "slack_bus": slack_bus,
        "year_summaries": {},
        "interpretation_note": "Constructed public-data PTDF relief testbed filtered to the G-built TDTR candidate-hour support; not an operator congestion reconstruction.",
    }

    for year in YEARS:
        print("\n" + "-" * 78)
        print(f"[YEAR] {year}")
        print("-" * 78)
        support_y = validate_support_year(support, year)
        print(
            f"[TDTR SUPPORT] year={year} selected={int(support_y['tdtr_support'].sum()):,}/"
            f"{hours_in_year_utc(year):,} ({support_y['tdtr_support'].mean():.2%})"
        )
        ypanel = validate_year_state(panel, year, mw_col)
        result = export_year(year, ptdf, branch_meta, ypanel, mw_col, weights, support_y)

        save_csv(result["active_branches"], OUTPUT_DIR / f"active_branches_{year}.csv")
        save_csv(result["requirements"], OUTPUT_DIR / f"branch_hour_requirements_{year}.csv.gz", compression="gzip")
        save_csv(result["coefficients"], OUTPUT_DIR / f"branch_hour_province_coefficients_{year}.csv.gz", compression="gzip")
        save_csv(result["flow_summary"], OUTPUT_DIR / f"branch_flow_summary_{year}.csv")
        save_json(result["summary"], OUTPUT_DIR / f"ptdf_export_summary_{year}.json")
        manifest["year_summaries"][str(year)] = result["summary"]

        s = result["summary"]
        print(
            f"[INFO] year={year} | support={s['tdtr_support_selected_hours']:,}/{s['n_hours']:,} | "
            f"active={s['n_active_branches']} | positive requirements={s['positive_branch_hour_requirements']:,} | "
            f"coefficient rows={s['coefficient_rows']:,} | "
            f"raw full-year R={s['raw_total_excess_full_year_MW_h']:.3f} | "
            f"raw support R={s['raw_total_excess_after_tdtr_support_MW_h']:.3f} | "
            f"scaled support R={s['scaled_total_requirement_MW_h']:.3f}"
        )

    print("\n" + "=" * 78)
    save_json(manifest, OUTPUT_DIR / "ptdf_debit_export_manifest.json")
    print("[DONE]")
    print("=" * 78)


if __name__ == "__main__":
    main()
