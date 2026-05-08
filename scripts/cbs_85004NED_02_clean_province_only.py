#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cbs_85004NED_02_clean_province_only.py  (LOUD + STRICT)

Reads raw CBS 85004NED export (WIDE) and writes the province-only LONG file needed by
build_provincial_residual_load_optionB.py (CBS_PROV_LONG).

Input (from downloader):
  CBS/RESREGIO/raw/cbs_85004NED_resregio_solar_wind_raw_2018-2024_v1.csv

Output (required by RL builder):
  CBS/RESREGIO/processed/cbs_85004NED_province_long_2018-2024_v1.csv

Key columns in output:
  RegioS, province, year, BronEnTechniek, BronEnTechniek_label, OpgesteldElektrischVermogen_MW
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# CONFIG
# =============================================================================
BASE = Path(r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Project III\data_processing")

RAW_DIR  = BASE / "CBS" / "RESREGIO" / "raw"
PROC_DIR = BASE / "CBS" / "RESREGIO" / "processed"

IN_RAW = RAW_DIR / "cbs_85004NED_resregio_solar_wind_raw_2018-2024_v1.csv"
OUT_LONG = PROC_DIR / "cbs_85004NED_province_long_2018-2024_v1.csv"

MIN_YEAR = 2018
MAX_YEAR = 2024

PV_TO_PROV = {
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
PROVINCES_12 = list(PV_TO_PROV.values())


# =============================================================================
# Helpers
# =============================================================================
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _require_cols(df: pd.DataFrame, req: set[str], label: str) -> None:
    miss = sorted(req - set(df.columns))
    if miss:
        raise ValueError(f"[{label}] Missing columns: {miss}. Has: {list(df.columns)}")

def _pick_capacity_col(cols: list[str]) -> str:
    candidates = [
        "OpgesteldElektrischVermogen_MW",
        "OpgesteldElektrischVermogen_1",
        "OpgesteldElektrischVermogen",
    ]
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(
        "Cannot find installed-capacity column. Expected one of:\n"
        f"  {candidates}\nFound columns:\n  {cols}"
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("=== CBS 85004NED: clean -> province_long (2018–2024) ===")
    print(f"[BASE] {BASE}")
    print(f"[IN ] {IN_RAW}")
    print(f"[OUT] {OUT_LONG}")

    if not IN_RAW.exists():
        existing = [p.name for p in RAW_DIR.glob("*.csv")]
        raise FileNotFoundError(
            f"Missing raw CBS file:\n  {IN_RAW}\n"
            f"CSV files in raw dir:\n  {existing}\n"
            "Run the downloader first."
        )

    df = pd.read_csv(IN_RAW)
    df.columns = [c.strip() for c in df.columns]
    print(f"[RAW] rows={len(df)} cols={len(df.columns)}")

    _require_cols(df, {"RegioS", "BronEnTechniek", "Perioden"}, "RAW")

    # year column exists in your downloader output, but keep robust:
    if "year" not in df.columns:
        df["year"] = df["Perioden"].astype(str).str.extract(r"(\d{4})", expand=False)
    df["year"] = _to_num(df["year"]).astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    df = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)].copy()
    print(f"[FILTER] years {MIN_YEAR}..{MAX_YEAR} -> rows={len(df)}")
    if df.empty:
        raise ValueError("[CLEAN] No rows after year filter.")

    # PV provinces only
    df["RegioS"] = df["RegioS"].astype(str).str.strip()
    df = df[df["RegioS"].isin(PV_TO_PROV.keys())].copy()
    print(f"[FILTER] PV20..PV31 -> rows={len(df)}")
    if df.empty:
        # This means your raw download does not include PV rows (but it should).
        # In that case you'd need to re-download without any hidden RegioS restriction.
        raise ValueError(
            "[CLEAN] No PV20..PV31 rows found in raw file.\n"
            "Check that the raw CSV actually contains provincial RegioS codes."
        )

    df["province"] = df["RegioS"].map(PV_TO_PROV)

    # capacity column
    cap_col = _pick_capacity_col(list(df.columns))
    df["OpgesteldElektrischVermogen_MW"] = _to_num(df[cap_col]).astype(float)

    # label column (should exist due to downloader join)
    if "BronEnTechniek_label" not in df.columns:
        print("[WARN] BronEnTechniek_label missing; falling back to BronEnTechniek codes.")
        df["BronEnTechniek_label"] = df["BronEnTechniek"].astype(str)

    # Keep minimal columns for RL builder
    keep = [
        "RegioS", "province", "year",
        "BronEnTechniek", "BronEnTechniek_label",
        "OpgesteldElektrischVermogen_MW",
    ]
    df = df[keep].copy()

    # Aggregate duplicates (shouldn't be many, but safe)
    out = (
        df.groupby(["RegioS", "province", "year", "BronEnTechniek", "BronEnTechniek_label"], as_index=False)
          ["OpgesteldElektrischVermogen_MW"].sum()
    )

    # QC province coverage
    prov_cov = out.groupby("year")["province"].nunique().sort_index()
    print("\n[QC] province coverage per year:")
    print(prov_cov.to_string())
    bad = prov_cov[prov_cov != 12]
    if not bad.empty:
        raise ValueError(f"[QC] Some years do not have 12 provinces:\n{bad.to_string()}")

    # Diagnostic for RL classifier (zon/wind keywords)
    labs = out["BronEnTechniek_label"].astype(str).str.lower()
    print("\n[QC] keyword hits in labels:")
    print(f"  contains('zon')  = {int(labs.str.contains('zon', na=False).sum())}")
    print(f"  contains('wind') = {int(labs.str.contains('wind', na=False).sum())}")
    print("\n[QC] top labels (lowercased):")
    print(labs.value_counts().head(25).to_string())

    # Write
    _ensure_dir(OUT_LONG)
    out.to_csv(OUT_LONG, index=False)
    print(f"\n[OK] wrote:\n  {OUT_LONG}")
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
