"""
A_construct_lp_params.py
=========================

Construct the annual LP/MILP/QP parameter files consumed by C_train_m3.py and
D_run_full_benchmark.py.

For each year in cfg.ALL_YEARS produces

    data/derived/lp_params/lp_params_{year}.parquet

with columns

    province
    timestamp_utc
    R_value
    c_value
    H_i
    Hbar_i
    P_i
    Pbar_value

Also produces

    data/derived/lp_params/lp_meta.json

with keys

    M_penalty
    rho
    ETA_R
    ETA_C
    P_MW
    hour_step
    per_year

Operating-layer mapping
-----------------------

The constructed operating layer is

    R_{n,t} = ETA_R * max(0, stress_{n,t}),

where stress is selected by

    s_plus_norm  if available,
    rl_plus_norm otherwise,
    zero otherwise.

The operating cost is

    c_{n,t} = ETA_C * P_n * max(0.05, 1 + price_da_norm_{n,t}).

The representative capacity is

    P_n = REPRESENTATIVE_CAPACITY_MW.

The depth budget is calibrated as

    D0_n = sum_t min(R_{n,t}, P_n) / P_n,
    H_n  = rho * D0_n.

This is deliberately calibrated from the natural unbudgeted depth. If rho < 1,
the annual weighted-depth budget is tight relative to the unbudgeted solution,
so annual scarcity multipliers mu can become positive. This is essential for
optimization-informed M3 training, because the training weight is

    alpha_{n,t} = mu_n * x_{n,t} / P_n.

If H is too loose, mu = 0 and M3_OI / M3_LO collapse to essentially the same
plain proxy fit as M3_plain.

Touched-time budget
-------------------

The touched-time budget is

    Hbar_n = floor(rho * T_active_n),

where

    T_active_n = #{t : R_{n,t} > 0}.

HOUR_STEP subsampling
---------------------

The state panel is typically hourly. To keep annual optimization instances
manageable, rows are deterministically subsampled by province-year:

    keep every HOUR_STEP-th timestamp.

Default HOUR_STEP is read from pipeline_config.py, falling back to 3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# REPOSITORY IMPORTS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipeline_config as cfg  # type: ignore


# =============================================================================
# CONFIG
# =============================================================================

HOUR_STEP: int = int(getattr(cfg, "HOUR_STEP", 3))


# =============================================================================
# HELPERS
# =============================================================================

def _require(path: Path, label: str) -> Path:
    """Check that an input path exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n"
            f"  {path}\n"
            f"Fix pipeline_config.py or run the upstream step first."
        )
    print(f"  {label}: {path}")
    return path


def _load_state() -> pd.DataFrame:
    """Load the auditable state panel and keep Dutch provinces only."""
    _require(cfg.STATE_PATH, "auditable state panel")

    state = pd.read_csv(cfg.STATE_PATH, parse_dates=["timestamp_utc"])
    state["timestamp_utc"] = pd.to_datetime(state["timestamp_utc"], utc=True)
    state["province"] = state["province"].astype(str).str.strip()

    if "year" not in state.columns:
        state["year"] = state["timestamp_utc"].dt.year.astype(int)

    state = state[state["province"].isin(cfg.NL_PROVINCES)].copy()

    numeric_cols = [
        "year",
        "price_da_norm",
        "rl_plus_norm",
        "rl_minus_norm",
        "s_plus_norm",
        "s_minus_norm",
        "structural_exposure",
    ]

    for col in numeric_cols:
        if col in state.columns:
            state[col] = pd.to_numeric(state[col], errors="coerce")

    return state.sort_values(["province", "timestamp_utc"]).reset_index(drop=True)


def _stress_signal(df: pd.DataFrame) -> pd.Series:
    """
    Select the best available forward-looking positive stress signal.

    Priority:
        1. s_plus_norm
        2. rl_plus_norm
        3. zero

    Both s_plus_norm and rl_plus_norm are interpreted as normalized positive
    local/network stress signals. This function avoids hard failure if one of
    them is missing.
    """
    for col in ["s_plus_norm", "rl_plus_norm"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    return pd.Series(0.0, index=df.index, dtype=float)


def _price_signal(df: pd.DataFrame) -> pd.Series:
    """Return normalized day-ahead price signal, falling back to zero."""
    if "price_da_norm" not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)

    return pd.to_numeric(df["price_da_norm"], errors="coerce").fillna(0.0)


def _calibrate_H(R_yr: np.ndarray, P_n: float, rho: float) -> float:
    """
    Calibrate the weighted-depth allowance.

        D0 = sum_t min(R_t, P_n) / P_n,
        H  = rho * D0.

    Important:
    Do not force H >= 1. If D0 is small, forcing H to 1 can make the annual
    budget nonbinding and kill the scarcity multiplier mu. For provinces with
    essentially no constructed relief need, H=0 is fine because x=0 and slack
    feasibility are still available.
    """
    R_yr = np.asarray(R_yr, dtype=float)
    D0 = float(np.sum(np.minimum(np.maximum(R_yr, 0.0), P_n) / P_n))

    if D0 <= 1e-10:
        return 0.0

    return max(float(rho) * D0, 1e-8)


def _calibrate_Hbar(R_yr: np.ndarray, rho: float) -> float:
    """
    Calibrate the touched-time allowance.

        Hbar = floor(rho * T_active),
        T_active = #{t : R_t > 0}.

    If there is no active interval, Hbar=0. Do not force it to one, otherwise
    touched-time accounting receives artificial budget in inactive provinces.
    """
    R_yr = np.asarray(R_yr, dtype=float)
    T_active = float((R_yr > 1e-9).sum())

    if T_active <= 0:
        return 0.0

    return float(np.floor(float(rho) * T_active))


def _subsample_province_year(grp: pd.DataFrame) -> pd.DataFrame:
    """Deterministically keep every HOUR_STEP-th row of a province-year panel."""
    grp = grp.sort_values("timestamp_utc").reset_index(drop=True)
    return grp.iloc[::HOUR_STEP].copy()


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _safe_min(values: List[float]) -> float:
    return float(np.min(values)) if values else float("nan")


def _safe_max(values: List[float]) -> float:
    return float(np.max(values)) if values else float("nan")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("A: CONSTRUCT LP PARAMETERS")
    print("=" * 70)
    print(f"State panel : {cfg.STATE_PATH}")
    print(f"Output dir  : {cfg.LP_DIR}")
    print(f"Years       : {cfg.ALL_YEARS}")
    print(f"ETA_R       : {cfg.ETA_R}")
    print(f"ETA_C       : {cfg.ETA_C}")
    print(f"M_penalty   : {cfg.M_PENALTY}")
    print(f"Rho         : {cfg.BASELINE_RHO}")
    print(f"P_MW        : {cfg.REPRESENTATIVE_CAPACITY_MW}")
    print(f"HOUR_STEP   : {HOUR_STEP}")

    if HOUR_STEP <= 0:
        raise ValueError("HOUR_STEP must be positive.")

    cfg.LP_DIR.mkdir(parents=True, exist_ok=True)

    state = _load_state()
    years_in_state = sorted(state["year"].dropna().astype(int).unique().tolist())

    print(
        f"\nState panel: {len(state):,} rows  "
        f"years={years_in_state}  "
        f"provinces={state['province'].nunique()}"
    )

    P_val = float(cfg.REPRESENTATIVE_CAPACITY_MW)
    rho = float(cfg.BASELINE_RHO)

    if P_val <= 0:
        raise ValueError("REPRESENTATIVE_CAPACITY_MW must be positive.")
    if not (0.0 <= rho <= 1.0):
        print(f"[WARN] BASELINE_RHO={rho} is outside [0,1]. This is allowed but unusual.")

    meta_per_year: Dict[str, dict] = {}

    for year in cfg.ALL_YEARS:
        yr_state = state[state["year"] == int(year)].copy()

        if yr_state.empty:
            print(f"\n  [WARN] Year {year}: no rows in state panel, skipping")
            continue

        print(
            f"\n  Year {year}: {len(yr_state):,} province-hours "
            f"({yr_state['province'].nunique()} provinces)"
        )

        yr_sub = (
            yr_state
            .groupby("province", group_keys=False, observed=True)
            .apply(_subsample_province_year)
            .reset_index(drop=True)
        )

        if yr_sub.empty:
            print(f"  [WARN] Year {year}: empty after subsampling, skipping")
            continue

        yr_sub["stress_for_R"] = _stress_signal(yr_sub)
        yr_sub["price_for_c"] = _price_signal(yr_sub)

        timestamps = sorted(pd.to_datetime(yr_sub["timestamp_utc"].unique(), utc=True))
        T = len(timestamps)

        print(f"  Subsampled (step={HOUR_STEP}): {T} intervals")

        rows: List[pd.DataFrame] = []
        H_vals: Dict[str, float] = {}
        Hbar_vals: Dict[str, float] = {}
        D0_vals: Dict[str, float] = {}
        active_vals: Dict[str, float] = {}
        Rmax_vals: Dict[str, float] = {}
        Rmean_vals: Dict[str, float] = {}

        ts_index = pd.DatetimeIndex(timestamps)

        for prov in cfg.NL_PROVINCES:
            sub_p = (
                yr_sub[yr_sub["province"] == prov]
                .copy()
                .sort_values("timestamp_utc")
                .set_index("timestamp_utc")
                .reindex(ts_index)
            )

            if sub_p.empty:
                print(f"    [WARN] {prov}: no rows for year {year}")
                continue

            stress = pd.to_numeric(sub_p["stress_for_R"], errors="coerce").fillna(0.0).clip(lower=0.0)
            R_yr = (float(cfg.ETA_R) * stress).to_numpy(dtype=float)
            R_yr = np.nan_to_num(R_yr, nan=0.0, posinf=0.0, neginf=0.0)
            R_yr = np.maximum(R_yr, 0.0)

            price_norm = pd.to_numeric(sub_p["price_for_c"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            c_yr = float(cfg.ETA_C) * P_val * np.maximum(0.05, 1.0 + price_norm)
            c_yr = np.nan_to_num(
                c_yr,
                nan=float(cfg.ETA_C) * P_val * 0.05,
                posinf=float(cfg.ETA_C) * P_val * 0.05,
                neginf=float(cfg.ETA_C) * P_val * 0.05,
            )
            c_yr = np.maximum(c_yr, float(cfg.ETA_C) * P_val * 0.05)

            D0_prov = float(np.sum(np.minimum(R_yr, P_val) / P_val))
            T_active = float((R_yr > 1e-9).sum())

            H_prov = _calibrate_H(R_yr, P_val, rho)
            Hbar_prov = _calibrate_Hbar(R_yr, rho)

            H_vals[prov] = H_prov
            Hbar_vals[prov] = Hbar_prov
            D0_vals[prov] = D0_prov
            active_vals[prov] = T_active
            Rmax_vals[prov] = float(np.max(R_yr)) if len(R_yr) else 0.0
            Rmean_vals[prov] = float(np.mean(R_yr)) if len(R_yr) else 0.0

            prov_df = pd.DataFrame(
                {
                    "province": prov,
                    "timestamp_utc": timestamps,
                    "R_value": R_yr,
                    "c_value": c_yr,
                    "H_i": H_prov,
                    "Hbar_i": Hbar_prov,
                    "P_i": P_val,
                    "Pbar_value": np.full(T, P_val, dtype=float),
                }
            )
            rows.append(prov_df)

        if not rows:
            print(f"  [WARN] Year {year}: no province rows, skipping parquet")
            continue

        yr_df = pd.concat(rows, ignore_index=True)

        R_flat = pd.to_numeric(yr_df["R_value"], errors="coerce").fillna(0.0)
        C_flat = pd.to_numeric(yr_df["c_value"], errors="coerce").fillna(0.0)

        D0_arr = np.array(list(D0_vals.values()), dtype=float)
        H_arr = np.array(list(H_vals.values()), dtype=float)
        Hbar_arr = np.array(list(Hbar_vals.values()), dtype=float)
        active_arr = np.array(list(active_vals.values()), dtype=float)

        positive_D0 = D0_arr > 1e-10
        positive_active = active_arr > 1e-10

        print(
            f"  R_value: mean={R_flat.mean():.3f}  "
            f"max={R_flat.max():.3f}  "
            f"nonzero={100.0 * float((R_flat > 1e-9).mean()):.1f}%"
        )
        print(
            f"  c_value: mean={C_flat.mean():.3f}  "
            f"min={C_flat.min():.3f}  max={C_flat.max():.3f}"
        )
        print(
            f"  D0_depth: mean={_safe_mean(list(D0_vals.values())):.3f}  "
            f"min={_safe_min(list(D0_vals.values())):.3f}  "
            f"max={_safe_max(list(D0_vals.values())):.3f}"
        )
        print(
            f"  H_depth: mean={_safe_mean(list(H_vals.values())):.3f}  "
            f"min={_safe_min(list(H_vals.values())):.3f}  "
            f"max={_safe_max(list(H_vals.values())):.3f}"
        )
        print(
            f"  H_touch: mean={_safe_mean(list(Hbar_vals.values())):.1f}  "
            f"min={_safe_min(list(Hbar_vals.values())):.1f}  "
            f"max={_safe_max(list(Hbar_vals.values())):.1f}"
        )

        if positive_D0.any():
            ratio = H_arr[positive_D0] / D0_arr[positive_D0]
            print(
                f"  H/D0: mean={ratio.mean():.3f}  "
                f"min={ratio.min():.3f}  "
                f"max={ratio.max():.3f}  "
                f"target rho={rho:.3f}"
            )
        else:
            print("  H/D0: undefined because all D0 values are zero")

        if positive_active.any():
            ratio_touch = Hbar_arr[positive_active] / active_arr[positive_active]
            print(
                f"  Hbar/T_active: mean={ratio_touch.mean():.3f}  "
                f"min={ratio_touch.min():.3f}  "
                f"max={ratio_touch.max():.3f}"
            )
        else:
            print("  Hbar/T_active: undefined because all T_active values are zero")

        if (H_arr > D0_arr + 1e-8).any():
            bad = [
                prov for prov in H_vals
                if H_vals[prov] > D0_vals[prov] + 1e-8
            ]
            print(f"  [WARN] H_i > D0_i for provinces: {bad}")

        if R_flat.max() > P_val + 1e-8:
            print(
                f"  [WARN] Some R_value exceeds P={P_val:.2f}. "
                f"Slack or multiple resources may be needed."
            )

        out_path = cfg.LP_DIR / f"lp_params_{year}.parquet"
        yr_df.to_parquet(out_path, index=False)
        print(f"  ✓ {out_path}  ({len(yr_df):,} rows)")

        meta_per_year[str(year)] = {
            "T": int(T),
            "n_provinces": int(len(rows)),
            "H_depth": round(float(np.mean(H_arr)), 6) if len(H_arr) else None,
            "H_touch": round(float(np.mean(Hbar_arr)), 6) if len(Hbar_arr) else None,
            "D0_depth": round(float(np.mean(D0_arr)), 6) if len(D0_arr) else None,
            "T_active": round(float(np.mean(active_arr)), 6) if len(active_arr) else None,
            "R_mean": round(float(R_flat.mean()), 6),
            "R_max": round(float(R_flat.max()), 6),
            "R_nonzero_share": round(float((R_flat > 1e-9).mean()), 6),
            "H_by_province_depth": {p: round(float(v), 6) for p, v in H_vals.items()},
            "H_by_province_touch": {p: round(float(v), 6) for p, v in Hbar_vals.items()},
            "D0_by_province": {p: round(float(v), 6) for p, v in D0_vals.items()},
            "T_active_by_province": {p: round(float(v), 6) for p, v in active_vals.items()},
            "Rmax_by_province": {p: round(float(v), 6) for p, v in Rmax_vals.items()},
            "Rmean_by_province": {p: round(float(v), 6) for p, v in Rmean_vals.items()},
        }

    meta = {
        "M_penalty": float(cfg.M_PENALTY),
        "rho": float(rho),
        "ETA_R": float(cfg.ETA_R),
        "ETA_C": float(cfg.ETA_C),
        "P_MW": float(P_val),
        "hour_step": int(HOUR_STEP),
        "train_years": list(map(int, cfg.TRAIN_YEARS)),
        "val_years": list(map(int, cfg.VAL_YEARS)),
        "test_years": list(map(int, cfg.TEST_YEARS)),
        "all_years": list(map(int, cfg.ALL_YEARS)),
        "per_year": meta_per_year,
        "columns": [
            "province",
            "timestamp_utc",
            "R_value",
            "c_value",
            "H_i",
            "Hbar_i",
            "P_i",
            "Pbar_value",
        ],
        "calibration_note": (
            "H_i = rho * sum_t min(R_value, P_i)/P_i. "
            "H_i is not forced to be at least one, because that can make "
            "low-scarcity instances nonbinding and eliminate the M3 "
            "optimization-informed training signal."
        ),
    }

    meta_path = cfg.LP_DIR / "lp_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ {meta_path}")

    print("\nSummary:")
    for yr, m in meta_per_year.items():
        print(
            f"  {yr}: T={m['T']}  "
            f"D0={m['D0_depth']:.3f}  "
            f"H_depth={m['H_depth']:.3f}  "
            f"H_touch={m['H_touch']:.1f}  "
            f"R_nonzero={100.0 * m['R_nonzero_share']:.1f}%  "
            f"N={m['n_provinces']}"
        )

    if not meta_per_year:
        raise RuntimeError("No LP parameter files were written. Check cfg.ALL_YEARS and state panel years.")


if __name__ == "__main__":
    main()