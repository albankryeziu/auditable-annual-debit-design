#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_targets_and_offshore_panel_from_table.py

Build the two missing inputs for Option-B RL construction from your LaTeX table:

(1) panel_offshore_landing_cap_mw_2025_2035_v1.csv
    - Year x Province panel (12 provinces each year)
    - Landing provinces get linearly interpolated GW between 2024 and 2035 endpoints
    - Non-landing provinces get 0 MW
    - Output columns:
        year, province, landing_cap_mw, nl_offshore_cap_mw

(2) nl_res_targets_twh_2025_2035_v1.csv
    - National annual targets in TWh:
        year, solar_twh, wind_on_twh, wind_off_twh
    - Offshore wind targets derived from offshore capacity + capacity factor:
        wind_off_twh(y) = cap_offshore_GW(y) * 8.76 * CF_OFF
    - Solar and onshore wind targets set as baseline 2024 values grown by optional CAGR
      (default 0.0 => flat). Replace later with KEV/PBL scenario totals.

Notes / contracts:
- Provinces must match your RL builder's PROVINCES_12 naming.
- Offshore panel must include all 12 provinces per year, or RL builder will fail.
"""

from pathlib import Path
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
# PATHS (match your RL builder)
# =============================================================================
# =============================================================================
# PATHS (match your RL builder)
# =============================================================================
OUT_OFFSHORE = getattr(
    cfg,
    "OFFSHORE_LANDING_PANEL_PATH",
    cfg.DERIVED_DIR / "panel_offshore_landing_cap_mw_2025_2035_v1.csv",
)
OUT_TARGETS = getattr(
    cfg,
    "NL_RES_TARGETS_PATH",
    cfg.DERIVED_DIR / "nl_res_targets_twh_2025_2035_v1.csv",
)

FUTURE_YEARS = list(range(2025, 2036))

# =============================================================================
# CANONICAL NL PROVINCES (must match RL builder exactly)
# =============================================================================
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

# =============================================================================
# YOUR LaTeX TABLE (GW) — endpoint capacities
# =============================================================================
# Landing provinces only (others are 0 by construction)
OFFSHORE_GW_2024 = {
    "Groningen": 0.6,
    "Noord-Holland": 1.1,
    "Zuid-Holland": 1.5,
    "Zeeland": 1.5,
}
OFFSHORE_GW_2035 = {
    "Groningen": 6.0,
    "Noord-Holland": 2.5,
    "Zuid-Holland": 5.5,
    "Zeeland": 7.5,
}
LANDING_PROVS = sorted(OFFSHORE_GW_2024.keys())

# =============================================================================
# ASSUMPTIONS / CALIBRATION (edit later when you have proper scenario totals)
# =============================================================================
# Offshore capacity factor used to translate capacity -> annual energy target.
CF_OFF = 0.40

# Baseline national generation in 2024 (TWh) as placeholders:
# (Replace with your scenario targets when available.)
SOLAR_TWH_2024   = 22.0
WIND_ON_TWH_2024 = 18.0

# Optional growth rates (CAGR) for 2024->future targets; default 0 => flat.
GROWTH_SOLAR   = 0.00
GROWTH_WIND_ON = 0.00

# =============================================================================
# Helpers
# =============================================================================
def interp_linear(y: int, y0: int, y1: int, v0: float, v1: float) -> float:
    """Linear interpolation for scalar v between years y0..y1 inclusive."""
    if y1 == y0:
        return float(v1)
    w = (y - y0) / (y1 - y0)
    return float((1.0 - w) * v0 + w * v1)

def cap_gw_to_twh(cap_gw: np.ndarray, cf: float) -> np.ndarray:
    """
    Convert capacity in GW to annual energy in TWh using:
        TWh = GW * 8.76 * CF
    (8760 hours/year, 1 GW = 1000 MW, 1 TWh = 1e6 MWh => factor 8.76)
    """
    return cap_gw * 8.76 * float(cf)

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1) Build offshore landing capacity panel (MW) for 2025–2035, 12 provinces/year
# =============================================================================
def build_offshore_panel() -> pd.DataFrame:
    # Basic validation: landing provinces must be subset of PROVINCES_12
    missing = sorted(set(LANDING_PROVS) - set(PROVINCES_12))
    if missing:
        raise ValueError(f"Landing provinces not in PROVINCES_12: {missing}")

    # Validate endpoints are complete and consistent
    for p in LANDING_PROVS:
        if p not in OFFSHORE_GW_2035:
            raise ValueError(f"Province '{p}' missing in OFFSHORE_GW_2035.")

    rows = []
    for y in FUTURE_YEARS:
        for p in PROVINCES_12:
            if p in LANDING_PROVS:
                gw = interp_linear(y, 2024, 2035, OFFSHORE_GW_2024[p], OFFSHORE_GW_2035[p])
            else:
                gw = 0.0
            rows.append({"year": y, "province": p, "landing_cap_mw": 1000.0 * gw})

    df = pd.DataFrame(rows)

    # Add national total per year (MW)
    tot = df.groupby("year")["landing_cap_mw"].sum().reset_index(name="nl_offshore_cap_mw")
    df = df.merge(tot, on="year", how="left")

    # Strict check: 12 provinces each year
    cnt = df.groupby("year")["province"].nunique()
    bad = cnt[cnt != 12]
    if not bad.empty:
        raise RuntimeError(f"Internal error: province coverage not 12:\n{bad.to_string()}")

    _ensure_dir(OUT_OFFSHORE)
    df.to_csv(OUT_OFFSHORE, index=False)
    print(f"[OK] wrote offshore landing panel:\n  {OUT_OFFSHORE}")
    print(df.head(12).to_string(index=False))

    return df

# =============================================================================
# 2) Build national RES targets table (TWh) for 2025–2035
# =============================================================================
def build_targets(offshore_panel: pd.DataFrame) -> pd.DataFrame:
    # National offshore capacity in GW (sum over provinces)
    cap_gw = (
        offshore_panel.groupby("year")["landing_cap_mw"].sum()
        .reindex(FUTURE_YEARS)
        .to_numpy(dtype=float) / 1000.0
    )

    wind_off_twh = cap_gw_to_twh(cap_gw, CF_OFF)

    # Baseline growth for solar and onshore wind (placeholders until scenario totals exist)
    years_from_2024 = np.array([y - 2024 for y in FUTURE_YEARS], dtype=float)
    solar_twh   = float(SOLAR_TWH_2024)   * np.power(1.0 + float(GROWTH_SOLAR),   years_from_2024)
    wind_on_twh = float(WIND_ON_TWH_2024) * np.power(1.0 + float(GROWTH_WIND_ON), years_from_2024)

    out = pd.DataFrame({
        "year": FUTURE_YEARS,
        "solar_twh": solar_twh,
        "wind_on_twh": wind_on_twh,
        "wind_off_twh": wind_off_twh,
    })

    _ensure_dir(OUT_TARGETS)
    out.to_csv(OUT_TARGETS, index=False)
    print(f"\n[OK] wrote RES targets:\n  {OUT_TARGETS}")
    print(out.head(12).to_string(index=False))

    # Quick implied offshore diagnostics
    cap_gw_series = pd.Series(cap_gw, index=FUTURE_YEARS)
    print("\n[INFO] implied offshore targets (capacity -> TWh):")
    for y in [FUTURE_YEARS[0], FUTURE_YEARS[1], FUTURE_YEARS[2], FUTURE_YEARS[-3], FUTURE_YEARS[-2], FUTURE_YEARS[-1]]:
        i = FUTURE_YEARS.index(y)
        print(f"  {y}: cap={cap_gw_series.loc[y]:.2f} GW -> wind_off={wind_off_twh[i]:.2f} TWh (CF_OFF={CF_OFF:.2f})")

    return out

# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    offshore = build_offshore_panel()
    _ = build_targets(offshore)

if __name__ == "__main__":
    main()
