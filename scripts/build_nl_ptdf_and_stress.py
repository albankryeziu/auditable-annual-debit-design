#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_nl_ptdf_and_stress.py

Repair/build the main-paper auditable province-hour state panel and the
province-to-bus weight matrix used by H_export_ptdf_for_debit_paper.py.

This is the MAIN PAPER branch:
- uses province_hour_public_stress_panel.csv as the stress backbone;
- does not use the future residual-load scenario scaffold as the historical
  stress backbone;
- creates ptdf_load_pressure_MW from prov_load_MW for the PTDF stress screen;
- repairs province_to_bus_weights.csv if missing/incomplete;
- accepts ac_ptdf_wide.csv, ac_ptdf_long.csv, or ac_ptdf_matrices.npz.

Interpretation: this prepares a public-data PTDF stress testbed. It is not an
operator redispatch or physical congestion reconstruction.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
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
    return default


DEFAULT_PROVINCES = [
    "Groningen", "Friesland", "Drenthe", "Overijssel", "Flevoland", "Gelderland",
    "Utrecht", "Noord-Holland", "Zuid-Holland", "Zeeland", "Noord-Brabant", "Limburg",
]

PROVINCE_CENTROIDS = {
    "Groningen": (53.2194, 6.5665), "Friesland": (53.1642, 5.7818),
    "Drenthe": (52.9476, 6.6231), "Overijssel": (52.4388, 6.5016),
    "Flevoland": (52.5279, 5.5954), "Gelderland": (52.0452, 5.8718),
    "Utrecht": (52.0907, 5.1214), "Noord-Holland": (52.5200, 4.7885),
    "Zuid-Holland": (52.0208, 4.4938), "Zeeland": (51.4940, 3.8497),
    "Noord-Brabant": (51.4827, 5.2322), "Limburg": (51.2092, 5.9336),
}

PROVINCE_HINTS = {
    "Groningen": ["groningen", "eems", "eemshaven", "vierverlaten", "meeden"],
    "Friesland": ["friesland", "leeuwarden", "drachten", "heerenveen"],
    "Drenthe": ["drenthe", "assen", "emmen", "hoogeveen", "zeyerveen"],
    "Overijssel": ["overijssel", "zwolle", "hengelo", "almelo", "enschede", "deventer"],
    "Flevoland": ["flevoland", "lelystad", "almere", "zeewolde"],
    "Gelderland": ["gelderland", "arnhem", "nijmegen", "dodewaard", "doetinchem", "apeldoorn"],
    "Utrecht": ["utrecht", "breukelen", "lage weide", "soest"],
    "Noord-Holland": ["noord holland", "noord-holland", "amsterdam", "diemen", "beverwijk", "hemweg", "oostzaan", "velsen"],
    "Zuid-Holland": ["zuid holland", "zuid-holland", "rotterdam", "maasvlakte", "bleiswijk", "wateringen", "krimpen", "den haag"],
    "Zeeland": ["zeeland", "borssele", "terneuzen", "vlissingen", "goes"],
    "Noord-Brabant": ["noord brabant", "noord-brabant", "eindhoven", "tilburg", "breda", "geertruidenberg", "maasbracht"],
    "Limburg": ["limburg", "maastricht", "venlo", "roermond", "graetheide", "buggenum"],
}

DERIVED_DIR = Path(cfg_get("DERIVED_DIR", REPO_ROOT / "data" / "derived"))
ALL_YEARS = sorted(dict.fromkeys(map(int, cfg_get("ALL_YEARS", [2021, 2022, 2023, 2024, 2025]))))
PROVINCES = list(cfg_get("NL_PROVINCES", DEFAULT_PROVINCES))

PUBLIC_STRESS_PANEL = Path(cfg_get(
    "PROVINCE_STRESS_PATH",
    cfg_get("PROVINCE_STRESS_PANEL_PATH", DERIVED_DIR / "province_hour_public_stress_panel.csv"),
))
STATE_PANEL_OUT = Path(cfg_get(
    "AUDITABLE_STATE_PANEL_PATH",
    cfg_get("STATE_PANEL_PATH", DERIVED_DIR / "auditable_state_province_hour.csv"),
))
PTDF_PUBLICATION_DIR = Path(cfg_get(
    "PTDF_PUBLICATION_DIR",
    os.environ.get(
        "PTDF_PUBLICATION_DIR",
        r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\Netherlands_Data\PTDF_OUTPUT_PUBLICATION",
    ),
))
WEIGHTS_MAIN = PTDF_PUBLICATION_DIR / "province_to_bus_weights.csv"
WEIGHTS_INITIAL = PTDF_PUBLICATION_DIR / "province_to_bus_weights_initial.csv"
MANIFEST_OUT = DERIVED_DIR / "auditable_state_manifest.json"


def hours_in_year_utc(year: int) -> int:
    return int((pd.Timestamp(f"{year+1}-01-01T00:00:00Z") - pd.Timestamp(f"{year}-01-01T00:00:00Z")) / pd.Timedelta(hours=1))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def pick_column(columns: Iterable[str], candidates: Iterable[str], required: bool = True) -> Optional[str]:
    cols = [str(c).strip() for c in columns]
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of {list(candidates)!r}. Available columns: {cols!r}")
    return None


def backup_if_exists(path: Path, suffix: str) -> Optional[Path]:
    if not path.exists():
        return None
    bdir = path.parent / "_backups"
    bdir.mkdir(parents=True, exist_ok=True)
    bpath = bdir / f"{path.stem}.{suffix}{path.suffix}"
    shutil.copy2(path, bpath)
    print(f"[BACKUP] {bpath}")
    return bpath


def minmax01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    lo, hi = float(x.min()), float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return pd.Series(np.zeros(len(x)), index=s.index)
    return (x - lo) / (hi - lo)


def build_state_panel() -> pd.DataFrame:
    if not PUBLIC_STRESS_PANEL.exists():
        raise FileNotFoundError(
            f"Missing public stress panel:\n  {PUBLIC_STRESS_PANEL}\n"
            "Run build_province_hour_public_stress_panel.py first."
        )

    df = pd.read_csv(PUBLIC_STRESS_PANEL, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    province_col = pick_column(df.columns, ["province", "provincie"])
    ts_col = pick_column(df.columns, ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc"], required=False)
    year_col = pick_column(df.columns, ["year"], required=False)
    hour_col = pick_column(df.columns, ["hour_of_year", "hoy", "hour"], required=False)

    out = df.copy()
    out["province"] = out[province_col].astype(str).str.strip()

    if ts_col is not None:
        out["timestamp_utc"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce").dt.floor("h")
    elif year_col is not None and hour_col is not None:
        y = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
        h = pd.to_numeric(out[hour_col], errors="coerce").astype("Int64")
        out["timestamp_utc"] = pd.to_datetime(y.astype(str) + "-01-01T00:00:00Z", utc=True, errors="coerce") + pd.to_timedelta(h.astype(float) - 1.0, unit="h")
    else:
        raise ValueError("Public stress panel needs timestamp_utc or year + hour_of_year.")

    out = out.dropna(subset=["timestamp_utc"]).copy()
    out["year"] = out["timestamp_utc"].dt.year.astype(int)
    out = out[out["year"].isin(ALL_YEARS) & out["province"].isin(PROVINCES)].copy()

    parts = []
    for (province, year), g in out.groupby(["province", "year"], sort=False):
        H = hours_in_year_utc(int(year))
        gg = g.sort_values("timestamp_utc").copy()
        if len(gg) != H:
            raise ValueError(f"{province}, {year}: {len(gg)} rows, expected {H}.")
        expected = pd.date_range(pd.Timestamp(f"{year}-01-01T00:00:00Z"), periods=H, freq="h")
        if not pd.DatetimeIndex(gg["timestamp_utc"]).equals(expected):
            raise ValueError(f"{province}, {year}: timestamps are not a complete UTC hourly year.")
        gg["hour_of_year"] = np.arange(1, H + 1, dtype=int)
        parts.append(gg)
    out = pd.concat(parts, ignore_index=True)

    if "price_da_eur_mwh" in out.columns and "price_da_eur_per_mwh" not in out.columns:
        out = out.rename(columns={"price_da_eur_mwh": "price_da_eur_per_mwh"})

    for col in [
        "nat_load_MW", "prov_load_MW", "province_load_share", "price_da_eur_per_mwh",
        "price_da_norm", "rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm",
        "structural_exposure",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out["prov_load_MW"].isna().any():
        raise ValueError("The public stress panel must contain prov_load_MW.")

    if out["price_da_norm"].isna().any() and out["price_da_eur_per_mwh"].notna().any():
        out["price_da_norm"] = out.groupby("year", sort=False)["price_da_eur_per_mwh"].transform(minmax01)

    if out["province_load_share"].isna().any():
        out["province_load_share"] = np.where(out["nat_load_MW"].abs() > 1e-12, out["prov_load_MW"] / out["nat_load_MW"], np.nan)

    if out["structural_exposure"].isna().any():
        prov_ann = out.groupby(["province", "year"])["prov_load_MW"].transform("sum")
        nat_ann = out.groupby("year")["prov_load_MW"].transform("sum")
        out["structural_exposure"] = np.where(nat_ann > 0, prov_ann / nat_ann, 1.0 / len(PROVINCES))

    if out["rl_plus_norm"].isna().any():
        out["rl_plus_norm"] = out.groupby(["province", "year"], sort=False)["prov_load_MW"].transform(minmax01)
    if out["rl_minus_norm"].isna().any():
        out["rl_minus_norm"] = out.groupby(["province", "year"], sort=False)["prov_load_MW"].transform(lambda s: minmax01(-s))
    if out["s_plus_norm"].isna().any():
        out["s_plus_norm"] = 0.65 * out["rl_plus_norm"] + 0.25 * out["price_da_norm"].fillna(0.0) + 0.10 * out["structural_exposure"]
        out["s_plus_norm"] = out.groupby(["province", "year"], sort=False)["s_plus_norm"].transform(minmax01)
    if out["s_minus_norm"].isna().any():
        price_low = out.groupby("year", sort=False)["price_da_eur_per_mwh"].transform(lambda s: minmax01(-pd.to_numeric(s, errors="coerce").fillna(0.0)))
        out["s_minus_norm"] = 0.65 * out["rl_minus_norm"] + 0.25 * price_low.fillna(0.0) + 0.10 * (1.0 - out["structural_exposure"])
        out["s_minus_norm"] = out.groupby(["province", "year"], sort=False)["s_minus_norm"].transform(minmax01)

    local = out["timestamp_utc"].dt.tz_convert("Europe/Amsterdam")
    out["month"] = local.dt.month.astype(int)
    out["hour_of_week"] = (local.dt.weekday * 24 + local.dt.hour).astype(int)
    out["is_weekend"] = local.dt.weekday.isin([5, 6]).astype(int)
    if "is_holiday" not in out.columns:
        out["is_holiday"] = 0
    out["is_holiday"] = pd.to_numeric(out["is_holiday"], errors="coerce").fillna(0).astype(int)

    # Critical PTDF export column. Historical main-paper branch uses public load,
    # not the 2025-2035 residual-load scenario scaffold.
    out["ptdf_load_pressure_MW"] = out["prov_load_MW"].astype(float).clip(lower=0.0)
    out["ptdf_load_pressure_source"] = "public_load_pressure:prov_load_MW"

    for col in ["gopacs_eur", "gopacs_mwh", "gopacs_activation", "B_agg", "E_exposure", "B_proxy_baseline"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if "split" not in out.columns:
        out["split"] = out["year"].map(lambda y: "train" if y <= 2023 else ("val" if y == 2024 else "test"))

    keep = [
        "province", "timestamp_utc", "year", "hour_of_year", "hour_of_week", "month",
        "is_weekend", "is_holiday", "nat_load_MW", "prov_load_MW", "province_load_share",
        "price_da_eur_per_mwh", "price_da_norm", "rl_plus_norm", "rl_minus_norm",
        "s_plus_norm", "s_minus_norm", "structural_exposure", "ptdf_load_pressure_MW",
        "ptdf_load_pressure_source", "gopacs_eur", "gopacs_mwh", "gopacs_activation",
        "B_agg", "E_exposure", "B_proxy_baseline", "split",
    ]
    out = out[keep].sort_values(["province", "timestamp_utc"]).reset_index(drop=True)

    expected_rows = sum(hours_in_year_utc(y) for y in ALL_YEARS) * len(PROVINCES)
    if len(out) != expected_rows:
        raise ValueError(f"State panel rows={len(out):,}, expected={expected_rows:,}.")
    if out.duplicated(["province", "timestamp_utc"]).any():
        raise ValueError("Duplicate province-hour rows in state panel.")
    return out


def ptdf_matrix_file() -> Path:
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


def is_meta_column(c: str) -> bool:
    cl = str(c).lower()
    return any(tok in cl for tok in ["branch", "line", "from", "to", "rating", "limit", "loading", "score", "active", "rank", "hours"])


def load_bus_ids() -> list[str]:
    p = ptdf_matrix_file()
    if p.suffix.lower() == ".npz":
        z = np.load(p, allow_pickle=True)
        for key in ["bus_ids", "buses", "bus_names"]:
            if key in z.files:
                return [str(x) for x in z[key].tolist()]
        raise ValueError(f"{p} has no bus_ids/buses/bus_names array.")
    if "long" in p.name.lower():
        df = pd.read_csv(p, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        bus_col = pick_column(df.columns, ["bus_id", "bus", "bus_name"])
        return sorted(df[bus_col].dropna().astype(str).str.strip().unique().tolist())
    sample = pd.read_csv(p, nrows=10)
    sample.columns = [str(c).strip() for c in sample.columns]
    bus_cols = []
    for c in sample.columns:
        if is_meta_column(c):
            continue
        if pd.to_numeric(sample[c], errors="coerce").notna().sum() > 0:
            bus_cols.append(str(c))
    if not bus_cols:
        raise ValueError(f"Could not identify bus columns in {p}.")
    return bus_cols


def load_bus_metadata(bus_ids: list[str]) -> pd.DataFrame:
    for name in ["ac_buses_main_component.csv", "ac_buses_all.csv", "buses_main_component.csv", "buses.csv", "bus_metadata.csv"]:
        p = PTDF_PUBLICATION_DIR / name
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            bus_col = pick_column(df.columns, ["bus_id", "bus", "bus_name", "name", "Bus", "Bus ID"], required=False)
            if bus_col is None:
                # choose column with largest overlap with PTDF bus names
                best, best_overlap = None, -1
                bus_set = set(map(str, bus_ids))
                for c in df.columns:
                    vals = set(df[c].dropna().astype(str).str.strip())
                    overlap = len(vals & bus_set)
                    if overlap > best_overlap:
                        best, best_overlap = c, overlap
                bus_col = best if best_overlap > 0 else None
            if bus_col is not None:
                df["bus_id"] = df[bus_col].astype(str).str.strip()
                return pd.DataFrame({"bus_id": bus_ids}).merge(df.drop_duplicates("bus_id"), on="bus_id", how="left")
    return pd.DataFrame({"bus_id": bus_ids})


def norm_province(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    for p in PROVINCES:
        if s == p.lower().replace("-", " "):
            return p
    for p in PROVINCES:
        if p.lower().replace("-", " ") in s:
            return p
    return None


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(max(1 - a, 0)))


def complete_weights(rows: pd.DataFrame, bus_ids: list[str], zero_source: str) -> pd.DataFrame:
    panel = pd.MultiIndex.from_product([PROVINCES, bus_ids], names=["province", "bus_id"]).to_frame(index=False)
    out = panel.merge(rows, on=["province", "bus_id"], how="left")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out["weight_source"] = out.get("weight_source", zero_source)
    out["weight_source"] = out["weight_source"].fillna(zero_source)
    parts = []
    for p, g in out.groupby("province", sort=False):
        gg = g.copy()
        s = float(gg["weight"].sum())
        if s <= 0:
            gg["weight"] = 1.0 / len(bus_ids)
            gg["weight_source"] = "uniform_province_fallback"
        else:
            gg["weight"] /= s
        parts.append(gg)
    return pd.concat(parts, ignore_index=True)


def make_initial_weights(bus_ids: list[str]) -> pd.DataFrame:
    meta = load_bus_metadata(bus_ids)
    meta["bus_id"] = meta["bus_id"].astype(str)

    prov_col = pick_column(meta.columns, ["province", "provincie", "province_name", "region", "RegioS_label"], required=False)
    if prov_col is not None:
        meta["province_guess"] = meta[prov_col].map(norm_province)
        if meta["province_guess"].notna().sum() > 0:
            rows = []
            for p in PROVINCES:
                buses = meta.loc[meta["province_guess"] == p, "bus_id"].tolist()
                for b in buses:
                    rows.append({"province": p, "bus_id": b, "weight": 1 / len(buses), "weight_source": f"bus_metadata:{prov_col}"})
            if rows:
                return complete_weights(pd.DataFrame(rows), bus_ids, "zero_not_in_province")

    lat_col = pick_column(meta.columns, ["lat", "latitude", "Lat", "Latitude"], required=False)
    lon_col = pick_column(meta.columns, ["lon", "lng", "longitude", "Lon", "Longitude"], required=False)
    if lat_col and lon_col:
        lat = pd.to_numeric(meta[lat_col], errors="coerce")
        lon = pd.to_numeric(meta[lon_col], errors="coerce")
        if lat.between(50, 54.5).mean() > 0.5 and lon.between(2, 8).mean() > 0.5:
            geo = meta.assign(_lat=lat, _lon=lon).dropna(subset=["_lat", "_lon"])
            rows = []
            for p in PROVINCES:
                plat, plon = PROVINCE_CENTROIDS[p]
                ds = []
                for _, r in geo.iterrows():
                    ds.append((str(r["bus_id"]), max(haversine_km(plat, plon, r["_lat"], r["_lon"]), 1.0)))
                ds = sorted(ds, key=lambda x: x[1])[: min(8, len(ds))]
                inv = np.array([1 / d**2 for _, d in ds], dtype=float)
                inv /= inv.sum()
                for (b, _), w in zip(ds, inv):
                    rows.append({"province": p, "bus_id": b, "weight": float(w), "weight_source": "inverse_distance_centroid"})
            return complete_weights(pd.DataFrame(rows), bus_ids, "zero_far_bus")

    guessed = []
    for _, r in meta.iterrows():
        text = " ".join(str(v) for v in r.dropna().tolist()).lower().replace("-", " ")
        for p, hints in PROVINCE_HINTS.items():
            if any(h.replace("-", " ") in text for h in hints):
                guessed.append({"province": p, "bus_id": str(r["bus_id"])})
                break
    if guessed:
        gd = pd.DataFrame(guessed).drop_duplicates()
        rows = []
        for p in PROVINCES:
            buses = gd.loc[gd["province"] == p, "bus_id"].tolist()
            if buses:
                for b in buses:
                    rows.append({"province": p, "bus_id": b, "weight": 1 / len(buses), "weight_source": "station_name_hint"})
        return complete_weights(pd.DataFrame(rows), bus_ids, "zero_text_hint")

    rows = [{"province": p, "bus_id": b, "weight": 1 / len(bus_ids), "weight_source": "uniform_all_buses_fallback"} for p in PROVINCES for b in bus_ids]
    return pd.DataFrame(rows)


def read_weights(path: Path, bus_ids: list[str]) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if {"province", "bus_id", "weight"}.issubset(df.columns):
        out = df[["province", "bus_id", "weight"]].copy()
        out["province"] = out["province"].astype(str).str.strip()
        out["bus_id"] = out["bus_id"].astype(str).str.strip()
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
        out["weight_source"] = df["weight_source"] if "weight_source" in df.columns else "existing_long"
        return complete_weights(out, bus_ids, "existing_missing_zero")
    prov_col = pick_column(df.columns, ["province", "provincie"], required=False)
    if prov_col is not None:
        rows = []
        for _, row in df.iterrows():
            for b in bus_ids:
                if b in df.columns:
                    rows.append({"province": str(row[prov_col]).strip(), "bus_id": b, "weight": row[b], "weight_source": "existing_wide"})
        if rows:
            return complete_weights(pd.DataFrame(rows), bus_ids, "existing_wide_missing_zero")
    return None


def validate_weights(w: Optional[pd.DataFrame], bus_ids: list[str]) -> tuple[bool, dict]:
    if w is None or w.empty:
        return False, {"reason": "missing_or_empty"}
    missing_prov = sorted(set(PROVINCES) - set(w.loc[w["weight"] > 0, "province"].unique()))
    missing_bus = sorted(set(bus_ids) - set(w["bus_id"].astype(str).unique()))
    sums = w.groupby("province")["weight"].sum().to_dict()
    bad_sums = {p: float(sums.get(p, 0)) for p in PROVINCES if abs(float(sums.get(p, 0)) - 1.0) > 1e-6}
    ok = not missing_prov and not missing_bus and not bad_sums
    return ok, {
        "n_rows": int(len(w)), "missing_provinces": missing_prov,
        "missing_buses_count": len(missing_bus), "bad_weight_sums": bad_sums,
        "weight_source_counts": w.get("weight_source", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
    }


def repair_weights(bus_ids: list[str]) -> tuple[pd.DataFrame, dict]:
    initial = make_initial_weights(bus_ids)
    ensure_parent(WEIGHTS_INITIAL)
    initial.to_csv(WEIGHTS_INITIAL, index=False)
    print(f"[SAVE] {WEIGHTS_INITIAL}")

    existing = read_weights(WEIGHTS_MAIN, bus_ids)
    ok, summary = validate_weights(existing, bus_ids)
    if ok:
        print(f"[KEEP] Existing complete mapping kept: {WEIGHTS_MAIN}")
        return existing, {"action": "kept_existing", "existing_summary": summary}

    if WEIGHTS_MAIN.exists():
        backup_if_exists(WEIGHTS_MAIN, "backup_before_repair")
    initial_ok, initial_summary = validate_weights(initial, bus_ids)
    if not initial_ok:
        raise ValueError("Generated initial province-to-bus mapping is incomplete:\n" + json.dumps(initial_summary, indent=2))
    initial.to_csv(WEIGHTS_MAIN, index=False)
    print(f"[SAVE] {WEIGHTS_MAIN}")
    return initial, {"action": "wrote_initial_to_main", "previous_summary": summary, "initial_summary": initial_summary}


def main() -> None:
    print("=" * 78)
    print("BUILD / REPAIR MAIN-PAPER NL PTDF/STRESS INPUTS")
    print("=" * 78)
    print(f"Repo root           : {REPO_ROOT}")
    print(f"Public stress panel : {PUBLIC_STRESS_PANEL}")
    print(f"State panel out     : {STATE_PANEL_OUT}")
    print(f"PTDF publication dir: {PTDF_PUBLICATION_DIR}")
    print(f"Target years        : {ALL_YEARS}")
    print(f"Provinces           : {PROVINCES}")
    print("=" * 78)

    print("\n[1/2] Rebuilding auditable state panel from public stress backbone...")
    state = build_state_panel()
    ensure_parent(STATE_PANEL_OUT)
    backup_if_exists(STATE_PANEL_OUT, "backup_before_public_stress_rebuild")
    state.to_csv(STATE_PANEL_OUT, index=False)
    print(f"[SAVE] {STATE_PANEL_OUT}")

    print("\n[2/2] Building / repairing province-to-bus weights...")
    bus_ids = load_bus_ids()
    weights, weight_action = repair_weights(bus_ids)

    validation = {
        "n_rows": int(len(state)),
        "years": sorted(map(int, state["year"].unique().tolist())),
        "provinces": sorted(state["province"].unique().tolist()),
        "rows_expected": int(sum(hours_in_year_utc(y) for y in ALL_YEARS) * len(PROVINCES)),
        "ptdf_load_pressure_min": float(state["ptdf_load_pressure_MW"].min()),
        "ptdf_load_pressure_max": float(state["ptdf_load_pressure_MW"].max()),
        "ptdf_load_pressure_mean": float(state["ptdf_load_pressure_MW"].mean()),
    }
    manifest = {
        "script": Path(__file__).name,
        "repo_root": str(REPO_ROOT),
        "public_stress_panel": str(PUBLIC_STRESS_PANEL),
        "state_panel_out": str(STATE_PANEL_OUT),
        "ptdf_publication_dir": str(PTDF_PUBLICATION_DIR),
        "province_bus_weights": str(WEIGHTS_MAIN),
        "province_bus_weights_initial": str(WEIGHTS_INITIAL),
        "target_years": ALL_YEARS,
        "provinces": PROVINCES,
        "state_panel_validation": validation,
        "province_bus_weight_action": weight_action,
        "province_bus_weight_source_counts": weights["weight_source"].astype(str).value_counts().to_dict(),
        "interpretation_note": "Public-data PTDF stress screen; not an operator redispatch reconstruction.",
    }
    ensure_parent(MANIFEST_OUT)
    with MANIFEST_OUT.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SAVE] {MANIFEST_OUT}")
    print("\n[VALIDATION SUMMARY]")
    print(json.dumps(validation, indent=2))
    print("\n[DONE]")


if __name__ == "__main__":
    main()
