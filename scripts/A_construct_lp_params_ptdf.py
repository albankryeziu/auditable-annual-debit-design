"""
A_construct_lp_params_ptdf.py
=============================

Construct network-aware annual LP/MILP/QP parameter arrays from the PTDF debit
artifacts produced by H_export_ptdf_for_debit_paper.py.

This script writes efficient dense annual arrays for the downstream benchmark:

    data/derived/lp_params/lp_network_arrays_{year}.npz

with arrays

    A       : oriented PTDF relief coefficients, shape (K,N,T)
    R       : branch relief requirements, shape (K,T)
    C       : restriction cost, shape (N,T)
    H       : weighted-depth annual allowance, shape (N,)
    Hbar    : touched-time annual allowance, shape (N,)
    P       : representative/contract capacity, shape (N,)
    Pbar    : available restrictable capacity, shape (N,T)

It also writes audit tables:

    lp_network_dispatch_params_{year}.parquet       province-time C/P/Pbar/H/Hbar
    lp_network_relief_params_{year}.parquet         branch-time R/flow/rating
    lp_meta.json                                    mode='network_ptdf'

Allowance calibration
---------------------
The default calibration is reference-use calibration from the unbudgeted network
operating problem:

    min sum c*x + M*sum s
    s.t. sum_n A[k,n,t] x[n,t] + s[k,t] >= R[k,t]
         0 <= x[n,t] <= Pbar[n,t], s[k,t] >= 0.

Then

    D0_n  = sum_t x_ref[n,t] / P_n
    H_n   = rho * D0_n
    Hbar_n = floor(rho * #{t: x_ref[n,t] > tol}).

This is more defensible than calibrating H directly from branch-level R, because
branch requirements are coupled through A.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gurobipy is required for reference-use calibration. Activate the Gurobi environment."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pipeline_config as cfg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Could not import pipeline_config.py. Run from the repository/scripts directory.") from exc


# =============================================================================
# CONFIG
# =============================================================================

PTDF_DIR = Path(getattr(cfg, "PTDF_OUTPUT_PUBLICATION", REPO_ROOT / "PTDF_OUTPUT_PUBLICATION"))
PTDF_DEBIT_DIR = Path(getattr(cfg, "PTDF_DEBIT_DIR", PTDF_DIR / "debit_paper_inputs"))
LP_DIR = Path(getattr(cfg, "LP_DIR", REPO_ROOT / "data" / "derived" / "lp_params"))
STATE_PATH = Path(getattr(cfg, "STATE_PATH"))

NL_PROVINCES = list(getattr(cfg, "NL_PROVINCES"))
ALL_YEARS = [int(y) for y in getattr(cfg, "ALL_YEARS")]
TRAIN_YEARS = [int(y) for y in getattr(cfg, "TRAIN_YEARS", [])]
VAL_YEARS = [int(y) for y in getattr(cfg, "VAL_YEARS", [])]
TEST_YEARS = [int(y) for y in getattr(cfg, "TEST_YEARS", [])]

RHO = float(getattr(cfg, "BASELINE_RHO", 0.7))
M_PENALTY = float(getattr(cfg, "M_PENALTY", 5000.0))
REPRESENTATIVE_CAPACITY_MW = float(getattr(cfg, "REPRESENTATIVE_CAPACITY_MW", 50.0))
REFERENCE_TIME_LIMIT = float(getattr(cfg, "PTDF_REFERENCE_TIME_LIMIT", 300.0))
REFERENCE_SOLVER_QUIET = bool(getattr(cfg, "PTDF_REFERENCE_SOLVER_QUIET", True))
TOUCH_TOL = float(getattr(cfg, "PTDF_REFERENCE_TOUCH_TOL", 1e-7))
H_MIN_POSITIVE = float(getattr(cfg, "PTDF_H_MIN_POSITIVE", 1e-8))

USE_EUR_COSTS = bool(getattr(cfg, "USE_EUR_COSTS", True))
HOUR_STEP = int(getattr(cfg, "HOUR_STEP", 3))
PRICE_FLOOR_EUR_MWH = float(getattr(cfg, "PRICE_FLOOR_EUR_MWH", 5.0))
PRICE_CAP_EUR_MWH = float(getattr(cfg, "PRICE_CAP_EUR_MWH", 1000.0))
ETA_C = float(getattr(cfg, "ETA_C", 1.0))

PRICE_COL_CANDIDATES = list(getattr(cfg, "PRICE_EUR_MWH_COLUMNS", [
    "price_da_eur_mwh",
    "price_da_EUR_MWh",
    "day_ahead_price_eur_mwh",
    "day_ahead_price_EUR_MWh",
    "da_price_eur_mwh",
    "price_da",
    "price_EUR_MWh",
]))

AVAILABILITY_COL_CANDIDATES = list(getattr(cfg, "PBARE_MW_COLUMNS", [
    "Pbar_value",
    "available_restrictable_mw",
    "available_capacity_mw",
    "Pbar_MW",
]))


# =============================================================================
# BASIC HELPERS
# =============================================================================

def _require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}:\n  {path}")
    print(f"  {label}: {path}")
    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _load_manifest() -> dict:
    path = _require(PTDF_DEBIT_DIR / "ptdf_debit_manifest.json", "PTDF debit manifest")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_state() -> pd.DataFrame:
    _require(STATE_PATH, "state panel")
    state = pd.read_csv(STATE_PATH, parse_dates=["timestamp_utc"])
    state["timestamp_utc"] = pd.to_datetime(state["timestamp_utc"], utc=True, errors="coerce")
    state["province"] = state["province"].astype(str).str.strip()
    if "year" not in state.columns:
        state["year"] = state["timestamp_utc"].dt.year.astype(int)
    state["year"] = pd.to_numeric(state["year"], errors="coerce").astype("Int64")
    state = state[state["province"].isin(NL_PROVINCES)].copy()
    if state.empty:
        raise RuntimeError("State panel is empty after filtering to NL_PROVINCES.")
    return state.sort_values(["province", "timestamp_utc"]).reset_index(drop=True)


def _select_price_col(df: pd.DataFrame) -> str | None:
    for c in PRICE_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _select_availability_col(df: pd.DataFrame) -> str | None:
    for c in AVAILABILITY_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _province_capacity_vector() -> np.ndarray:
    cap = getattr(cfg, "PROVINCE_CAPACITY_MW", None)
    if isinstance(cap, dict):
        vals = [float(cap.get(p, REPRESENTATIVE_CAPACITY_MW)) for p in NL_PROVINCES]
        return np.maximum(np.asarray(vals, dtype=float), 1e-9)
    return np.full(len(NL_PROVINCES), REPRESENTATIVE_CAPACITY_MW, dtype=float)


def _timestamps_from_npz(npz) -> pd.DatetimeIndex:
    if "timestamp_ns" not in npz.files:
        raise KeyError("relief effectiveness npz is missing timestamp_ns")
    return pd.to_datetime(np.asarray(npz["timestamp_ns"], dtype=np.int64), utc=True)


# =============================================================================
# COST AND AVAILABILITY ARRAYS
# =============================================================================

def _cost_and_availability_for_year(
    state: pd.DataFrame,
    year: int,
    timestamps: pd.DatetimeIndex,
    P: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    yr = state[state["year"].astype(int) == int(year)].copy()
    if yr.empty:
        raise RuntimeError(f"No state rows for year {year}.")

    price_col = _select_price_col(yr)
    availability_col = _select_availability_col(yr)
    ts_index = pd.DatetimeIndex(timestamps)
    N, T = len(NL_PROVINCES), len(ts_index)

    C = np.zeros((N, T), dtype=float)
    Pbar = np.zeros((N, T), dtype=float)
    audit_parts: List[pd.DataFrame] = []

    for n, province in enumerate(NL_PROVINCES):
        sub = yr[yr["province"] == province].copy()
        if sub.empty:
            raise RuntimeError(f"Year {year}: state panel missing province {province}.")
        sub = sub.sort_values("timestamp_utc").set_index("timestamp_utc").reindex(ts_index)

        if USE_EUR_COSTS and price_col is not None:
            price = pd.to_numeric(sub[price_col], errors="coerce").ffill().bfill().fillna(PRICE_FLOOR_EUR_MWH)
            price = price.clip(lower=PRICE_FLOOR_EUR_MWH, upper=PRICE_CAP_EUR_MWH)
            # Cost unit: EUR per MW restricted during the modeled interval.
            # If HOUR_STEP=3 and source data is hourly, one modeled interval represents 3 hours.
            C[n, :] = price.to_numpy(dtype=float) * float(HOUR_STEP)
        else:
            if USE_EUR_COSTS and price_col is None:
                print("  [WARN] no EUR/MWh price column found; falling back to normalized price_da_norm cost.")
            if "price_da_norm" in sub.columns:
                pn = pd.to_numeric(sub["price_da_norm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            else:
                pn = np.zeros(T, dtype=float)
            C[n, :] = ETA_C * P[n] * np.maximum(0.05, 1.0 + pn)

        if availability_col is not None:
            av = pd.to_numeric(sub[availability_col], errors="coerce").fillna(P[n]).clip(lower=0.0, upper=P[n])
            Pbar[n, :] = av.to_numpy(dtype=float)
        else:
            Pbar[n, :] = P[n]

        audit_parts.append(pd.DataFrame({
            "year": int(year),
            "province": province,
            "timestamp_utc": ts_index,
            "c_value": C[n, :],
            "P_i": P[n],
            "Pbar_value": Pbar[n, :],
            "price_column_used": price_col if price_col is not None else "normalized_fallback",
            "availability_column_used": availability_col if availability_col is not None else "P_i_default",
        }))

    return C, Pbar, pd.concat(audit_parts, ignore_index=True)


# =============================================================================
# REFERENCE SOLVE FOR ALLOWANCE CALIBRATION
# =============================================================================

def _gurobi_env() -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0 if REFERENCE_SOLVER_QUIET else 1)
    env.setParam("LogToConsole", 0 if REFERENCE_SOLVER_QUIET else 1)
    env.start()
    return env


def _solve_reference_unbudgeted(
    A: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    Pbar: np.ndarray,
    M_penalty: float,
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    """Solve the unbudgeted network LP used only for allowance calibration."""
    K, N, T = A.shape
    if R.shape != (K, T):
        raise ValueError(f"R shape {R.shape} incompatible with A shape {A.shape}.")
    if C.shape != (N, T) or Pbar.shape != (N, T):
        raise ValueError("C and Pbar must have shape (N,T).")

    env = _gurobi_env()
    model = gp.Model("ptdf_reference_unbudgeted", env=env)
    model.Params.TimeLimit = REFERENCE_TIME_LIMIT
    model.Params.Method = 2

    x = model.addMVar((N, T), lb=0.0, ub=Pbar, name="x")
    s = model.addMVar((K, T), lb=0.0, name="s")
    model.setObjective((C * x).sum() + float(M_penalty) * s.sum(), GRB.MINIMIZE)

    # Add one vectorized-ish constraint per branch/time. This is explicit but robust.
    for k in range(K):
        for t in range(T):
            if R[k, t] > 1e-9:
                model.addConstr(
                    gp.quicksum(float(A[k, n, t]) * x[n, t] for n in range(N)) + s[k, t] >= float(R[k, t]),
                    name=f"relief_{k}_{t}",
                )

    model.optimize()
    status_map = {
        GRB.OPTIMAL: "Optimal",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.INFEASIBLE: "Infeasible",
        GRB.INF_OR_UNBD: "InfOrUnbd",
        GRB.UNBOUNDED: "Unbounded",
        GRB.SUBOPTIMAL: "SubOptimal",
        GRB.NUMERIC: "Numeric",
    }
    status = status_map.get(model.Status, f"GRB_{model.Status}")
    runtime = float(model.Runtime)
    if model.SolCount <= 0:
        model.dispose(); env.dispose()
        raise RuntimeError(f"Reference unbudgeted solve failed: status={status}")

    x_sol = x.X.reshape(N, T)
    s_sol = s.X.reshape(K, T)
    model.dispose(); env.dispose()
    return x_sol, s_sol, status, runtime


def _calibrate_allowances(x_ref: np.ndarray, P: np.ndarray, rho: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth_ref = x_ref / P.reshape(-1, 1)
    D0 = depth_ref.sum(axis=1)
    touched_ref = (x_ref > TOUCH_TOL * P.reshape(-1, 1)).sum(axis=1).astype(float)

    H = rho * D0
    H = np.where(D0 > 1e-10, np.maximum(H, H_MIN_POSITIVE), 0.0)
    Hbar = np.floor(rho * touched_ref)
    Hbar = np.where(touched_ref > 0, np.maximum(Hbar, 0.0), 0.0)
    return H.astype(float), Hbar.astype(float), D0.astype(float), touched_ref.astype(float)


# =============================================================================
# YEAR LOADING AND WRITING
# =============================================================================

def _load_ptdf_year(manifest: dict, year: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    yf = manifest["year_files"].get(str(year))
    if yf is None:
        raise KeyError(f"PTDF manifest has no files for year {year}.")

    eff_path = Path(yf["relief_effectiveness_npz"])
    relief_path = Path(yf["relief_requirements"])
    _require(eff_path, f"PTDF relief effectiveness npz for {year}")
    _require(relief_path, f"PTDF relief requirements for {year}")

    npz = np.load(eff_path, allow_pickle=True)
    A = np.asarray(npz["A_relief"], dtype=float)
    branch_ids = np.asarray(npz["branch_id"]).astype(str)
    timestamps = _timestamps_from_npz(npz)

    rel = pd.read_parquet(relief_path)
    rel["timestamp_utc"] = pd.to_datetime(rel["timestamp_utc"], utc=True)
    rel["branch_id"] = rel["branch_id"].astype(str)

    K, N, T = A.shape
    if N != len(NL_PROVINCES):
        raise ValueError(f"A has N={N} provinces, expected {len(NL_PROVINCES)}.")
    if len(timestamps) != T:
        raise ValueError("timestamp length in npz does not match A.shape[2].")

    R = np.zeros((K, T), dtype=float)
    flow = np.zeros((K, T), dtype=float)
    rating = np.zeros(K, dtype=float)
    ts_index = pd.DatetimeIndex(timestamps)

    for k, bid in enumerate(branch_ids):
        sub = rel[rel["branch_id"] == bid].copy()
        if sub.empty:
            raise RuntimeError(f"Year {year}: relief requirements missing branch_id={bid}.")
        sub = sub.sort_values("timestamp_utc").set_index("timestamp_utc").reindex(ts_index)
        if sub["R_value"].isna().any():
            raise RuntimeError(f"Year {year}, branch {bid}: missing R_value after timestamp alignment.")
        R[k, :] = pd.to_numeric(sub["R_value"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "flow_MW" in sub.columns:
            flow[k, :] = pd.to_numeric(sub["flow_MW"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "rating_mva" in sub.columns:
            rating[k] = float(pd.to_numeric(sub["rating_mva"], errors="coerce").dropna().iloc[0])

    R = np.maximum(np.nan_to_num(R, nan=0.0), 0.0)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return A, R, rel, timestamps, branch_ids, rating


def _write_year_outputs(
    year: int,
    A: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    H: np.ndarray,
    Hbar: np.ndarray,
    P: np.ndarray,
    Pbar: np.ndarray,
    timestamps: pd.DatetimeIndex,
    branch_ids: np.ndarray,
    rating: np.ndarray,
    dispatch_audit: pd.DataFrame,
    relief_audit: pd.DataFrame,
    x_ref: np.ndarray,
    s_ref: np.ndarray,
    D0: np.ndarray,
    touched_ref: np.ndarray,
) -> Dict[str, str]:
    LP_DIR.mkdir(parents=True, exist_ok=True)
    array_path = LP_DIR / f"lp_network_arrays_{year}.npz"
    dispatch_path = LP_DIR / f"lp_network_dispatch_params_{year}.parquet"
    relief_path = LP_DIR / f"lp_network_relief_params_{year}.parquet"
    ref_path = LP_DIR / f"lp_network_reference_solution_{year}.parquet"

    np.savez_compressed(
        array_path,
        A=A,
        R=R,
        C=C,
        H=H,
        Hbar=Hbar,
        P=P,
        Pbar=Pbar,
        province=np.array(NL_PROVINCES, dtype=object),
        branch_id=branch_ids.astype(str),
        rating_mva=rating.astype(float),
        timestamp_ns=timestamps.asi8,
        x_ref=x_ref,
        s_ref=s_ref,
        D0=D0,
        touched_ref=touched_ref,
    )

    # Add H/Hbar/D0 to dispatch audit.
    disp = dispatch_audit.copy()
    h_map = {p: H[i] for i, p in enumerate(NL_PROVINCES)}
    hb_map = {p: Hbar[i] for i, p in enumerate(NL_PROVINCES)}
    d0_map = {p: D0[i] for i, p in enumerate(NL_PROVINCES)}
    tr_map = {p: touched_ref[i] for i, p in enumerate(NL_PROVINCES)}
    disp["H_i"] = disp["province"].map(h_map).astype(float)
    disp["Hbar_i"] = disp["province"].map(hb_map).astype(float)
    disp["D0_ref_depth"] = disp["province"].map(d0_map).astype(float)
    disp["T_ref_touched"] = disp["province"].map(tr_map).astype(float)
    disp.to_parquet(dispatch_path, index=False)

    relief_audit.to_parquet(relief_path, index=False)

    # Reference solution long audit.
    parts = []
    for n, prov in enumerate(NL_PROVINCES):
        parts.append(pd.DataFrame({
            "year": int(year),
            "province": prov,
            "timestamp_utc": timestamps,
            "x_ref": x_ref[n, :],
            "depth_ref": x_ref[n, :] / max(P[n], 1e-9),
            "P_i": P[n],
            "H_i": H[n],
            "Hbar_i": Hbar[n],
        }))
    pd.concat(parts, ignore_index=True).to_parquet(ref_path, index=False)

    return {
        "arrays_npz": str(array_path),
        "dispatch_params": str(dispatch_path),
        "relief_params": str(relief_path),
        "reference_solution": str(ref_path),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("A: CONSTRUCT NETWORK-AWARE LP PARAMETERS FROM PTDF ARTIFACTS")
    print("=" * 78)
    print(f"PTDF debit dir : {PTDF_DEBIT_DIR}")
    print(f"LP output dir  : {LP_DIR}")
    print(f"State panel    : {STATE_PATH}")
    print(f"Years          : {ALL_YEARS}")
    print(f"rho            : {RHO}")
    print(f"M penalty      : {M_PENALTY}")
    print(f"P representative: {REPRESENTATIVE_CAPACITY_MW}")
    print(f"EUR costs      : {USE_EUR_COSTS}")

    if not (RHO >= 0.0):
        raise ValueError("BASELINE_RHO must be nonnegative.")
    if M_PENALTY <= 0:
        raise ValueError("M_PENALTY must be positive.")

    manifest = _load_manifest()
    state = _load_state()
    P = _province_capacity_vector()

    meta_per_year: Dict[str, dict] = {}
    files_per_year: Dict[str, Dict[str, str]] = {}

    for year in ALL_YEARS:
        print("\n" + "-" * 78)
        print(f"Year {year}")
        print("-" * 78)

        A, R, relief_audit, timestamps, branch_ids, rating = _load_ptdf_year(manifest, year)
        K, N, T = A.shape
        if N != len(NL_PROVINCES):
            raise ValueError(f"Expected N={len(NL_PROVINCES)}, got N={N} in A.")

        C, Pbar, dispatch_audit = _cost_and_availability_for_year(state, year, timestamps, P)

        print(f"  dimensions: K={K}, N={N}, T={T}")
        print(f"  R: total={R.sum():.3f}, mean={R.mean():.6f}, max={R.max():.3f}, nonzero={100*np.mean(R>1e-9):.2f}%")
        print(f"  A_relief: min={A.min():.5f}, max={A.max():.5f}, negative_share={np.mean(A < -1e-12):.3f}")
        print(f"  C: mean={C.mean():.3f}, min={C.min():.3f}, max={C.max():.3f}")

        x_ref, s_ref, ref_status, ref_runtime = _solve_reference_unbudgeted(A, R, C, Pbar, M_PENALTY)
        H, Hbar, D0, touched_ref = _calibrate_allowances(x_ref, P, RHO)

        print(
            f"  reference solve: status={ref_status}, runtime={ref_runtime:.1f}s, "
            f"x_total={x_ref.sum():.3f}, slack_total={s_ref.sum():.3f}"
        )
        print(
            f"  D0 depth: mean={D0.mean():.3f}, min={D0.min():.3f}, max={D0.max():.3f}\n"
            f"  H depth : mean={H.mean():.3f}, min={H.min():.3f}, max={H.max():.3f}\n"
            f"  Hbar    : mean={Hbar.mean():.1f}, min={Hbar.min():.1f}, max={Hbar.max():.1f}"
        )

        year_files = _write_year_outputs(
            year=year,
            A=A,
            R=R,
            C=C,
            H=H,
            Hbar=Hbar,
            P=P,
            Pbar=Pbar,
            timestamps=timestamps,
            branch_ids=branch_ids,
            rating=rating,
            dispatch_audit=dispatch_audit,
            relief_audit=relief_audit,
            x_ref=x_ref,
            s_ref=s_ref,
            D0=D0,
            touched_ref=touched_ref,
        )
        files_per_year[str(year)] = year_files

        meta_per_year[str(year)] = {
            "K_active": int(K),
            "N_provinces": int(N),
            "T": int(T),
            "R_total": float(R.sum()),
            "R_mean": float(R.mean()),
            "R_max": float(R.max()),
            "R_nonzero_share": float(np.mean(R > 1e-9)),
            "A_min": float(A.min()),
            "A_max": float(A.max()),
            "A_negative_share": float(np.mean(A < -1e-12)),
            "C_mean": float(C.mean()),
            "C_min": float(C.min()),
            "C_max": float(C.max()),
            "reference_status": ref_status,
            "reference_runtime": float(ref_runtime),
            "reference_x_total": float(x_ref.sum()),
            "reference_slack_total": float(s_ref.sum()),
            "D0_mean": float(D0.mean()),
            "D0_min": float(D0.min()),
            "D0_max": float(D0.max()),
            "H_mean": float(H.mean()),
            "H_min": float(H.min()),
            "H_max": float(H.max()),
            "Hbar_mean": float(Hbar.mean()),
            "Hbar_min": float(Hbar.min()),
            "Hbar_max": float(Hbar.max()),
            "H_by_province": {p: float(H[i]) for i, p in enumerate(NL_PROVINCES)},
            "Hbar_by_province": {p: float(Hbar[i]) for i, p in enumerate(NL_PROVINCES)},
            "D0_by_province": {p: float(D0[i]) for i, p in enumerate(NL_PROVINCES)},
        }

    meta = {
        "mode": "network_ptdf",
        "schema_version": "ptdf_network_arrays_v1",
        "ptdf_debit_manifest": str(PTDF_DEBIT_DIR / "ptdf_debit_manifest.json"),
        "lp_dir": str(LP_DIR),
        "years": ALL_YEARS,
        "train_years": TRAIN_YEARS,
        "val_years": VAL_YEARS,
        "test_years": TEST_YEARS,
        "all_years": ALL_YEARS,
        "provinces": NL_PROVINCES,
        "rho": RHO,
        "M_penalty": M_PENALTY,
        "P_MW_default": REPRESENTATIVE_CAPACITY_MW,
        "P_by_province": {p: float(P[i]) for i, p in enumerate(NL_PROVINCES)},
        "use_eur_costs": USE_EUR_COSTS,
        "hour_step": HOUR_STEP,
        "price_floor_eur_mwh": PRICE_FLOOR_EUR_MWH,
        "price_cap_eur_mwh": PRICE_CAP_EUR_MWH,
        "allowance_calibration": "reference_unbudgeted_network_use",
        "arrays": {
            "A": "shape (K,N,T), oriented relief coefficient",
            "R": "shape (K,T), branch relief requirement",
            "C": "shape (N,T), restriction cost",
            "H": "shape (N,), weighted-depth allowance",
            "Hbar": "shape (N,), touched-time allowance",
            "P": "shape (N,), capacity",
            "Pbar": "shape (N,T), availability bound",
        },
        "files_per_year": files_per_year,
        "per_year": meta_per_year,
    }

    LP_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = LP_DIR / "lp_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    print("\n" + "=" * 78)
    print("NETWORK LP PARAMETER CONSTRUCTION COMPLETE")
    print("=" * 78)
    print(f"✓ {meta_path}")
    print("\nNext required code updates:")
    print("  1. B_run_annual_lp.py: add network-aware LP/QP/MILP solvers accepting A with shape (K,N,T).")
    print("  2. C_train_m3.py and D_run_full_benchmark.py: load lp_network_arrays_{year}.npz when lp_meta['mode']=='network_ptdf'.")


if __name__ == "__main__":
    main()
