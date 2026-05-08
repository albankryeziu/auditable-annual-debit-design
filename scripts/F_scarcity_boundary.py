#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F_scarcity_boundary.py
======================

Scarcity-boundary and rho-sensitivity experiment for the annual debit-rule paper.

Purpose
-------
If previous results were weak, the first thing to check is not a new figure; it
is whether the annual allowance is actually scarce. This script varies the
annual allowance generosity parameter rho and re-solves the operating models.

It answers three questions:

1) At which rho do annual-budget multipliers become positive?
2) Does M3 differ from M2 only in the scarce regime?
3) Are weak M3 results caused by slack budgets, nearly flat q weights, or proxy
   / LP misalignment?

The script works with the current province LP schema and with an optional
PTDF/network LP schema.

Current province schema
-----------------------
    province, timestamp_utc, R_value, c_value, H_i, Hbar_i, P_i, Pbar_value

Optional network schema
-----------------------
    branch_id, province, timestamp_utc, A_value or a_relief,
    R_value, c_value, H_i, Hbar_i, P_i, Pbar_value

For the network schema, the script includes internal network-aware LP/MILP
solvers. This does NOT mean your existing B_run_annual_lp.py is network-aware;
it only means this robustness script can evaluate network-shaped LP parameter
files if they already exist.

Run
---
    python scripts/F_scarcity_boundary.py

Useful options
--------------
    --rho-grid 0.40,0.50,0.60,0.70,0.80,0.90,1.00
    --include-m1 0
    --years 2025
    --allowance-mode scale-current
    --feature-set F2
    --variant M3_LO
    --network-negative-policy clip

Allowance modes
---------------
scale-current:
    H(rho) = H_base * rho / rho_base.
    Hbar(rho) = Hbar_base * rho / rho_base.
    This works for province and network schemas and preserves the exact
    calibration used by A_construct_lp_params.py.

reconstruct-province:
    Reconstruct H from R via sum_t min(R_nt,P_n)/P_n. Province schema only.
    This is useful to verify that A_construct_lp_params.py did not inject an
    accidental floor that killed scarcity.

Outputs
-------
    outputs/results/scarcity_boundary_results.csv
    outputs/results/scarcity_boundary_pairwise.csv
    outputs/tables/T8_scarcity_boundary_summary.csv
    outputs/tables/T8_scarcity_boundary_summary.tex
    outputs/figures/F10_scarcity_boundary_cost.pdf/png
    outputs/figures/F10_scarcity_boundary_mu.pdf/png
    outputs/figures/F10_scarcity_boundary_saving.pdf/png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# REPO SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipeline_config as cfg  # type: ignore

from B_run_annual_lp import (  # type: ignore
    solve_touched_time_milp,
    solve_weighted_depth_lp,
)

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None


# =============================================================================
# PATHS / DEFAULTS
# =============================================================================

RESULTS_DIR = Path(getattr(cfg, "RESULTS_DIR", REPO_ROOT / "outputs" / "results"))
TABLES_DIR = Path(getattr(cfg, "TABLES_DIR", REPO_ROOT / "outputs" / "tables"))
FIGURES_DIR = Path(getattr(cfg, "FIGURES_DIR", REPO_ROOT / "outputs" / "figures"))
LP_DIR = Path(getattr(cfg, "LP_DIR", REPO_ROOT / "data" / "derived" / "lp_params"))
M3_WEIGHTS_PATH = Path(getattr(cfg, "M3_WEIGHTS_PATH", REPO_ROOT / "data" / "derived" / "m3_debit_weights_by_year.parquet"))

NL_PROVINCES = list(getattr(cfg, "NL_PROVINCES", []))
if not NL_PROVINCES:
    raise ValueError("cfg.NL_PROVINCES is empty.")

MAIN_FEATURE_SET = str(getattr(cfg, "MAIN_FEATURE_SET", "F2"))
MAIN_M3_VARIANT = str(getattr(cfg, "MAIN_M3_VARIANT", "M3_LO"))
M_PENALTY = float(getattr(cfg, "M_PENALTY", 5000.0))
BASELINE_RHO = float(getattr(cfg, "BASELINE_RHO", 0.70))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# INTERNAL RESULT OBJECT
# =============================================================================

@dataclass
class NetLPResult:
    x: np.ndarray
    s: np.ndarray
    mu: np.ndarray
    lambda_relief: np.ndarray
    cost_total: float
    cost_curtail: float
    cost_slack: float
    slack_total: float
    budget_util: float
    status: str
    runtime: float = 0.0
    mip_gap: float = 0.0
    obj_bound: float = math.nan


# =============================================================================
# BASIC HELPERS
# =============================================================================

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _lp_meta() -> Dict[str, Any]:
    return _load_json(LP_DIR / "lp_meta.json")


def _available_lp_years() -> List[int]:
    years: List[int] = []
    for p in sorted(LP_DIR.glob("lp_params_*.parquet")):
        try:
            years.append(int(p.stem.replace("lp_params_", "")))
        except ValueError:
            continue
    return sorted(set(years))


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _load_lp_year(year: int) -> pd.DataFrame:
    path = LP_DIR / f"lp_params_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing LP params for year={year}: {path}")
    df = pd.read_parquet(path)
    if "timestamp_utc" not in df.columns:
        raise KeyError(f"{path} has no timestamp_utc column.")
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    if "province" in df.columns:
        df["province"] = df["province"].astype(str).str.strip()
    return df


def _is_network_schema(df: pd.DataFrame) -> bool:
    return "branch_id" in df.columns and ("A_value" in df.columns or "a_relief" in df.columns)


def _status(model: gp.Model) -> str:  # type: ignore[name-defined]
    mapping = {
        GRB.OPTIMAL: "Optimal",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.INFEASIBLE: "Infeasible",
        GRB.INF_OR_UNBD: "InfOrUnbd",
        GRB.UNBOUNDED: "Unbounded",
        GRB.SUBOPTIMAL: "SubOptimal",
        GRB.NUMERIC: "Numeric",
        GRB.INTERRUPTED: "Interrupted",
    }
    return mapping.get(model.Status, f"GRB_{model.Status}")


def _safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 1e-12:
        return math.nan
    return float(num / den)


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0 or float(x.sum()) <= 1e-12:
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def _hhi(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    s = float(x.sum())
    if x.size == 0 or s <= 1e-12:
        return 0.0
    w = x / s
    return float(np.sum(w * w))


# =============================================================================
# M3 WEIGHTS
# =============================================================================

def _load_m3_weights() -> pd.DataFrame:
    if not M3_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing M3 weights: {M3_WEIGHTS_PATH}")
    df = pd.read_parquet(M3_WEIGHTS_PATH)
    required = {"province", "timestamp_utc", "year", "feature_set", "variant", "debit_weight"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"M3 weights missing required columns: {missing}")
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["province"] = df["province"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["feature_set"] = df["feature_set"].astype(str)
    df["variant"] = df["variant"].astype(str)
    df["debit_weight"] = pd.to_numeric(df["debit_weight"], errors="coerce")
    if df["debit_weight"].isna().any() or (df["debit_weight"] <= 0).any():
        raise ValueError("M3 weights contain NaN or nonpositive values.")
    return df


def _q_matrix(
    weights: pd.DataFrame,
    year: int,
    feature_set: str,
    variant: str,
    timestamps: Sequence[pd.Timestamp],
    provinces: Sequence[str],
    *,
    strict: bool,
) -> np.ndarray:
    q = np.ones((len(provinces), len(timestamps)), dtype=float)
    ts_order = pd.Index(pd.to_datetime(list(timestamps), utc=True))
    subset = weights[
        (weights["year"].astype(int) == int(year))
        & (weights["feature_set"] == str(feature_set))
        & (weights["variant"] == str(variant))
    ].copy()
    if subset.empty:
        msg = f"No M3 weights for year={year}, feature_set={feature_set}, variant={variant}."
        if strict:
            raise ValueError(msg)
        print(f"  [WARN] {msg} Using q == 1.")
        return q
    subset["timestamp_utc"] = pd.to_datetime(subset["timestamp_utc"], utc=True)
    subset["province"] = subset["province"].astype(str).str.strip()
    for n, province in enumerate(provinces):
        sub = subset[subset["province"] == province].sort_values("timestamp_utc").set_index("timestamp_utc").reindex(ts_order)
        if sub.empty or sub["debit_weight"].isna().all():
            msg = f"Missing M3 weights for province={province}, year={year}."
            if strict:
                raise ValueError(msg)
            print(f"  [WARN] {msg} Using q == 1 for this province.")
            continue
        vals = pd.to_numeric(sub["debit_weight"], errors="coerce").fillna(1.0).to_numpy(float)
        if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
            raise ValueError(f"Invalid M3 weights for province={province}, year={year}.")
        q[n, :] = vals
    return q


# =============================================================================
# ARRAY BUILDERS
# =============================================================================

def _province_arrays(lp_year: pd.DataFrame) -> Dict[str, Any]:
    provinces = NL_PROVINCES[:]
    timestamps = pd.DatetimeIndex(pd.to_datetime(lp_year["timestamp_utc"].drop_duplicates(), utc=True).sort_values())
    N, T = len(provinces), len(timestamps)
    required = {"province", "timestamp_utc", "R_value", "c_value", "H_i", "Hbar_i", "P_i"}
    missing = sorted(required.difference(lp_year.columns))
    if missing:
        raise KeyError(f"Province schema missing columns: {missing}")

    R = np.zeros((N, T), dtype=float)
    C = np.zeros((N, T), dtype=float)
    H = np.zeros(N, dtype=float)
    Hbar = np.zeros(N, dtype=float)
    P = np.zeros(N, dtype=float)
    Pbar = np.zeros((N, T), dtype=float)

    df = lp_year.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["province"] = df["province"].astype(str).str.strip()
    for n, province in enumerate(provinces):
        sub = df[df["province"] == province].sort_values("timestamp_utc").set_index("timestamp_utc").reindex(timestamps)
        if sub.empty or sub["R_value"].isna().all():
            raise ValueError(f"LP data missing province={province}.")
        if sub["R_value"].isna().any() or sub["c_value"].isna().any():
            raise ValueError(f"LP panel is unbalanced for province={province}.")
        R[n, :] = pd.to_numeric(sub["R_value"], errors="coerce").fillna(0.0).to_numpy(float)
        C[n, :] = pd.to_numeric(sub["c_value"], errors="coerce").fillna(0.0).to_numpy(float)
        H[n] = float(pd.to_numeric(sub["H_i"], errors="coerce").dropna().iloc[0])
        Hbar[n] = float(pd.to_numeric(sub["Hbar_i"], errors="coerce").dropna().iloc[0])
        P[n] = float(pd.to_numeric(sub["P_i"], errors="coerce").dropna().iloc[0])
        if "Pbar_value" in sub.columns:
            Pbar[n, :] = pd.to_numeric(sub["Pbar_value"], errors="coerce").fillna(P[n]).to_numpy(float)
        else:
            Pbar[n, :] = P[n]

    return {
        "mode": "province",
        "provinces": provinces,
        "timestamps": timestamps,
        "R": np.maximum(np.nan_to_num(R), 0.0),
        "C": np.maximum(np.nan_to_num(C), 0.0),
        "H_base": np.maximum(np.nan_to_num(H), 0.0),
        "Hbar_base": np.maximum(np.nan_to_num(Hbar), 0.0),
        "P": np.maximum(np.nan_to_num(P, nan=1.0), 1e-9),
        "Pbar": np.maximum(np.nan_to_num(Pbar), 0.0),
    }


def _network_arrays(lp_year: pd.DataFrame, negative_policy: str) -> Dict[str, Any]:
    if negative_policy not in {"clip", "allow", "error"}:
        raise ValueError("negative_policy must be clip, allow, or error.")
    provinces = NL_PROVINCES[:]
    timestamps = pd.DatetimeIndex(pd.to_datetime(lp_year["timestamp_utc"].drop_duplicates(), utc=True).sort_values())
    branches = sorted(pd.to_numeric(lp_year["branch_id"], errors="coerce").dropna().astype(int).unique().tolist())
    a_col = "a_relief" if "a_relief" in lp_year.columns else "A_value"
    required = {"branch_id", "province", "timestamp_utc", a_col, "R_value", "c_value", "H_i", "Hbar_i", "P_i"}
    missing = sorted(required.difference(lp_year.columns))
    if missing:
        raise KeyError(f"Network schema missing columns: {missing}")

    K, N, T = len(branches), len(provinces), len(timestamps)
    b_to_k = {b: k for k, b in enumerate(branches)}
    p_to_n = {p: n for n, p in enumerate(provinces)}
    t_to_j = {ts: j for j, ts in enumerate(timestamps)}

    df = lp_year.copy()
    df["branch_id"] = pd.to_numeric(df["branch_id"], errors="coerce").astype("Int64")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["province"] = df["province"].astype(str).str.strip()
    df[a_col] = pd.to_numeric(df[a_col], errors="coerce").fillna(0.0)
    df["R_value"] = pd.to_numeric(df["R_value"], errors="coerce").fillna(0.0).clip(lower=0.0)

    A = np.zeros((K, N, T), dtype=float)
    R = np.zeros((K, T), dtype=float)
    C = np.zeros((N, T), dtype=float)
    H = np.zeros(N, dtype=float)
    Hbar = np.zeros(N, dtype=float)
    P = np.zeros(N, dtype=float)
    Pbar = np.zeros((N, T), dtype=float)

    rel = df.groupby(["branch_id", "timestamp_utc"], observed=True)["R_value"].max().reset_index()
    for r in rel.itertuples(index=False):
        bid = int(getattr(r, "branch_id"))
        ts = getattr(r, "timestamp_utc")
        if bid in b_to_k and ts in t_to_j:
            R[b_to_k[bid], t_to_j[ts]] = float(getattr(r, "R_value"))

    coeff = df[["branch_id", "province", "timestamp_utc", a_col]].dropna(subset=["branch_id", "province", "timestamp_utc"])
    neg_count = int((coeff[a_col] < -1e-12).sum())
    if neg_count and negative_policy == "error":
        raise ValueError(f"Found {neg_count} negative {a_col} coefficients. Use directional a_relief or clip/allow.")
    for r in coeff.itertuples(index=False):
        bid = int(getattr(r, "branch_id"))
        prov = str(getattr(r, "province"))
        ts = getattr(r, "timestamp_utc")
        val = float(getattr(r, a_col))
        if negative_policy == "clip":
            val = max(val, 0.0)
        if bid in b_to_k and prov in p_to_n and ts in t_to_j:
            A[b_to_k[bid], p_to_n[prov], t_to_j[ts]] = val

    prov_static = df.groupby("province", observed=True).agg(H_i=("H_i", "first"), Hbar_i=("Hbar_i", "first"), P_i=("P_i", "first")).reset_index()
    for r in prov_static.itertuples(index=False):
        prov = str(getattr(r, "province"))
        if prov in p_to_n:
            n = p_to_n[prov]
            H[n] = float(pd.to_numeric(getattr(r, "H_i"), errors="coerce"))
            Hbar[n] = float(pd.to_numeric(getattr(r, "Hbar_i"), errors="coerce"))
            P[n] = float(pd.to_numeric(getattr(r, "P_i"), errors="coerce"))

    agg_dict = {"c_value": ("c_value", "first")}
    if "Pbar_value" in df.columns:
        agg_dict["Pbar_value"] = ("Pbar_value", "first")
    else:
        agg_dict["Pbar_value"] = ("P_i", "first")
    prov_time = df.groupby(["province", "timestamp_utc"], observed=True).agg(**agg_dict).reset_index()
    for r in prov_time.itertuples(index=False):
        prov = str(getattr(r, "province"))
        ts = getattr(r, "timestamp_utc")
        if prov in p_to_n and ts in t_to_j:
            n, t = p_to_n[prov], t_to_j[ts]
            C[n, t] = float(pd.to_numeric(getattr(r, "c_value"), errors="coerce"))
            Pbar[n, t] = float(pd.to_numeric(getattr(r, "Pbar_value"), errors="coerce"))

    for n in range(N):
        if not np.isfinite(P[n]) or P[n] <= 0:
            P[n] = float(getattr(cfg, "REPRESENTATIVE_CAPACITY_MW", 50.0))
        missing = Pbar[n, :] <= 0
        Pbar[n, missing] = P[n]

    return {
        "mode": "network",
        "provinces": provinces,
        "timestamps": timestamps,
        "branches": branches,
        "A": A,
        "R": np.maximum(np.nan_to_num(R), 0.0),
        "C": np.maximum(np.nan_to_num(C), 0.0),
        "H_base": np.maximum(np.nan_to_num(H), 0.0),
        "Hbar_base": np.maximum(np.nan_to_num(Hbar), 0.0),
        "P": np.maximum(np.nan_to_num(P, nan=1.0), 1e-9),
        "Pbar": np.maximum(np.nan_to_num(Pbar), 0.0),
        "a_column": a_col,
        "negative_coefficients_seen": neg_count,
        "negative_policy": negative_policy,
    }


def _annual_arrays(lp_year: pd.DataFrame, negative_policy: str) -> Dict[str, Any]:
    if _is_network_schema(lp_year):
        return _network_arrays(lp_year, negative_policy=negative_policy)
    return _province_arrays(lp_year)


# =============================================================================
# ALLOWANCE RECALIBRATION
# =============================================================================

def _base_rho(meta: Dict[str, Any]) -> float:
    for key in ["rho", "BASELINE_RHO", "baseline_rho"]:
        if key in meta:
            try:
                val = float(meta[key])
                if val > 0:
                    return val
            except Exception:
                pass
    return BASELINE_RHO if BASELINE_RHO > 0 else 1.0


def _apply_rho(arrays: Dict[str, Any], rho: float, rho_base: float, allowance_mode: str) -> Dict[str, Any]:
    out = dict(arrays)
    mode = arrays["mode"]
    P = np.asarray(arrays["P"], dtype=float)
    H_base = np.asarray(arrays["H_base"], dtype=float)
    Hbar_base = np.asarray(arrays["Hbar_base"], dtype=float)

    if allowance_mode == "scale-current" or mode == "network":
        scale = float(rho) / max(float(rho_base), 1e-12)
        H = np.maximum(H_base * scale, 0.0)
        Hbar = np.maximum(np.floor(Hbar_base * scale + 1e-9), 0.0)
    elif allowance_mode == "reconstruct-province":
        if mode != "province":
            raise ValueError("reconstruct-province allowance mode is only valid for province schema.")
        R = np.asarray(arrays["R"], dtype=float)
        D0 = np.sum(np.minimum(np.maximum(R, 0.0), P[:, None]) / P[:, None], axis=1)
        active = np.sum(R > 1e-9, axis=1)
        H = np.maximum(float(rho) * D0, 0.0)
        H = np.where(D0 <= 1e-10, 0.0, np.maximum(H, 1e-8))
        Hbar = np.floor(float(rho) * active + 1e-9)
    else:
        raise ValueError("allowance_mode must be scale-current or reconstruct-province")

    out["H"] = H
    out["Hbar"] = Hbar
    out["rho"] = float(rho)
    out["rho_base"] = float(rho_base)
    out["allowance_mode"] = allowance_mode
    return out


# =============================================================================
# NETWORK SOLVERS
# =============================================================================

def _solve_network_weighted_lp(
    A: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    q: np.ndarray,
    H: np.ndarray,
    P: np.ndarray,
    Pbar: np.ndarray,
    M_penalty: float,
    *,
    quiet: bool = True,
    time_limit: float = 300.0,
) -> NetLPResult:
    if gp is None or GRB is None:
        raise RuntimeError("gurobipy is required for network-schema solves.")
    K, N, T = A.shape
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0 if quiet else 1)
    env.setParam("LogToConsole", 0 if quiet else 1)
    env.start()
    model = gp.Model("scarcity_boundary_network_lp", env=env)
    model.Params.TimeLimit = float(time_limit)
    model.Params.Method = 1

    x = model.addMVar((N, T), lb=0.0, ub=Pbar, name="x")
    s = model.addMVar((K, T), lb=0.0, name="s")

    relief_cons: Dict[Tuple[int, int], Any] = {}
    for t in range(T):
        for k in range(K):
            relief_cons[(k, t)] = model.addConstr(A[k, :, t] @ x[:, t] + s[k, t] >= R[k, t], name=f"relief[{k},{t}]")
    budget_cons = []
    for n in range(N):
        budget_cons.append(model.addConstr(q[n, :] @ x[n, :] / float(P[n]) <= H[n], name=f"budget[{n}]"))

    model.setObjective(gp.quicksum(C[n, t] * x[n, t] for n in range(N) for t in range(T)) + float(M_penalty) * s.sum(), GRB.MINIMIZE)
    t0 = time.time()
    model.optimize()
    runtime = time.time() - t0
    status = _status(model)
    has_sol = model.SolCount > 0
    x_sol = np.asarray(x.X, dtype=float).reshape(N, T) if has_sol else np.zeros((N, T))
    s_sol = np.asarray(s.X, dtype=float).reshape(K, T) if has_sol else np.zeros((K, T))

    mu = np.zeros(N, dtype=float)
    lam = np.zeros((K, T), dtype=float)
    if model.Status == GRB.OPTIMAL:
        for n, con in enumerate(budget_cons):
            mu[n] = abs(float(con.Pi))
        for (k, t), con in relief_cons.items():
            lam[k, t] = abs(float(con.Pi))

    cost_curtail = float(np.sum(C * x_sol))
    slack_total = float(np.sum(s_sol))
    cost_slack = float(M_penalty) * slack_total
    used = np.sum(q * x_sol / P[:, None], axis=1)
    util = float(np.mean(used / np.maximum(H, 1e-9)))
    obj = cost_curtail + cost_slack
    model.dispose()
    env.dispose()
    return NetLPResult(x_sol, s_sol, mu, lam, obj, cost_curtail, cost_slack, slack_total, util, status, runtime)


def _solve_network_touched_milp(
    A: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    Hbar: np.ndarray,
    Pbar: np.ndarray,
    M_penalty: float,
    *,
    quiet: bool = True,
    time_limit: float = 600.0,
) -> NetLPResult:
    if gp is None or GRB is None:
        raise RuntimeError("gurobipy is required for network-schema solves.")
    K, N, T = A.shape
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0 if quiet else 1)
    env.setParam("LogToConsole", 0 if quiet else 1)
    env.start()
    model = gp.Model("scarcity_boundary_network_m1", env=env)
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(getattr(cfg, "M1_MIP_GAP", 1e-4))

    x = model.addMVar((N, T), lb=0.0, ub=Pbar, name="x")
    y = model.addMVar((N, T), vtype=GRB.BINARY, name="y")
    s = model.addMVar((K, T), lb=0.0, name="s")
    for t in range(T):
        for k in range(K):
            model.addConstr(A[k, :, t] @ x[:, t] + s[k, t] >= R[k, t], name=f"relief[{k},{t}]")
    for n in range(N):
        for t in range(T):
            model.addConstr(x[n, t] <= Pbar[n, t] * y[n, t], name=f"link[{n},{t}]")
        model.addConstr(y[n, :].sum() <= math.floor(float(Hbar[n]) + 1e-9), name=f"budget[{n}]")
    model.setObjective(gp.quicksum(C[n, t] * x[n, t] for n in range(N) for t in range(T)) + float(M_penalty) * s.sum(), GRB.MINIMIZE)
    t0 = time.time()
    model.optimize()
    runtime = time.time() - t0
    status = _status(model)
    has_sol = model.SolCount > 0
    x_sol = np.asarray(x.X, dtype=float).reshape(N, T) if has_sol else np.zeros((N, T))
    s_sol = np.asarray(s.X, dtype=float).reshape(K, T) if has_sol else np.zeros((K, T))
    cost_curtail = float(np.sum(C * x_sol))
    slack_total = float(np.sum(s_sol))
    cost_slack = float(M_penalty) * slack_total
    obj = cost_curtail + cost_slack
    touched = (x_sol > 1e-7).sum(axis=1)
    util = float(np.mean(touched / np.maximum(Hbar, 1e-9)))
    gap = float(model.MIPGap) if has_sol and np.isfinite(model.MIPGap) else math.nan
    bound = float(model.ObjBound) if has_sol and hasattr(model, "ObjBound") else math.nan
    model.dispose()
    env.dispose()
    return NetLPResult(x_sol, s_sol, np.zeros(N), np.zeros((K, T)), obj, cost_curtail, cost_slack, slack_total, util, status, runtime, gap, bound)


# =============================================================================
# SOLVE + METRICS
# =============================================================================

def _solve_rule(arrays: Dict[str, Any], rule: str, q: Optional[np.ndarray], *, include_m1: bool, time_limit_lp: float, time_limit_milp: float) -> Any:
    mode = arrays["mode"]
    if mode == "province":
        R, C, H, Hbar, P, Pbar = arrays["R"], arrays["C"], arrays["H"], arrays["Hbar"], arrays["P"], arrays["Pbar"]
        if rule == "M1":
            if not include_m1:
                return None
            return solve_touched_time_milp(R=R, C=C, Hbar=Hbar, P=P, M_penalty=M_PENALTY, quiet=True, time_limit=time_limit_milp, Pbar=Pbar)
        if q is None:
            raise ValueError("Weighted-depth rule requires q.")
        return solve_weighted_depth_lp(R=R, C=C, q=q, H=H, P=P, M_penalty=M_PENALTY, quiet=True, time_limit=time_limit_lp, Pbar=Pbar)

    A, R, C, H, Hbar, P, Pbar = arrays["A"], arrays["R"], arrays["C"], arrays["H"], arrays["Hbar"], arrays["P"], arrays["Pbar"]
    if rule == "M1":
        if not include_m1:
            return None
        return _solve_network_touched_milp(A=A, R=R, C=C, Hbar=Hbar, Pbar=Pbar, M_penalty=M_PENALTY, quiet=True, time_limit=time_limit_milp)
    if q is None:
        raise ValueError("Weighted-depth rule requires q.")
    return _solve_network_weighted_lp(A=A, R=R, C=C, q=q, H=H, P=P, Pbar=Pbar, M_penalty=M_PENALTY, quiet=True, time_limit=time_limit_lp)


def _summarize(year: int, rho: float, rule: str, feature_set: str, variant: str, q_source: str, arrays: Dict[str, Any], result: Any, q: Optional[np.ndarray]) -> Dict[str, Any]:
    x = np.asarray(result.x, dtype=float)
    s = np.asarray(result.s, dtype=float)
    P = np.asarray(arrays["P"], dtype=float)
    H = np.asarray(arrays["H"], dtype=float)
    Hbar = np.asarray(arrays["Hbar"], dtype=float)
    R_total = float(np.sum(arrays["R"]))
    slack_total = float(np.sum(s))
    relief_served = max(R_total - slack_total, 0.0)

    physical_exp = np.sum(x / P[:, None], axis=1)
    if rule == "M1":
        debit_exp = np.sum(x > 1e-7, axis=1).astype(float)
        q_mean = q_std = q_min = q_max = q_annual_mean_min = q_annual_mean_max = math.nan
    else:
        assert q is not None
        debit_exp = np.sum(q * x / P[:, None], axis=1)
        q_mean = float(np.mean(q))
        q_std = float(np.std(q))
        q_min = float(np.min(q))
        q_max = float(np.max(q))
        q_ann = np.mean(q, axis=1)
        q_annual_mean_min = float(np.min(q_ann))
        q_annual_mean_max = float(np.max(q_ann))

    mu = np.asarray(getattr(result, "mu", np.zeros(len(P))), dtype=float)
    return {
        "year": int(year),
        "rho": float(rho),
        "rho_base": float(arrays.get("rho_base", math.nan)),
        "allowance_mode": str(arrays.get("allowance_mode", "unknown")),
        "mode": str(arrays["mode"]),
        "rule": rule,
        "feature_set": feature_set,
        "variant": variant,
        "q_source": q_source,
        "status": str(getattr(result, "status", "NA")),
        "cost_total": float(getattr(result, "cost_total", math.nan)),
        "cost_curtail": float(getattr(result, "cost_curtail", math.nan)),
        "cost_slack": float(getattr(result, "cost_slack", math.nan)),
        "slack_total": slack_total,
        "relief_required_total": R_total,
        "relief_served_total": relief_served,
        "relief_served_share": 1.0 - _safe_div(slack_total, R_total) if R_total > 1e-12 else math.nan,
        "budget_util_mean": float(getattr(result, "budget_util", math.nan)),
        "mu_positive_count": int(np.sum(mu > 1e-7)),
        "mu_positive_share": float(np.mean(mu > 1e-7)) if len(mu) else 0.0,
        "mu_mean": float(np.mean(mu)) if len(mu) else 0.0,
        "mu_max": float(np.max(mu)) if len(mu) else 0.0,
        "physical_exposure_total": float(np.sum(physical_exp)),
        "physical_exposure_gini": _gini(physical_exp),
        "physical_exposure_hhi": _hhi(physical_exp),
        "debit_exposure_total": float(np.sum(debit_exp)),
        "debit_exposure_gini": _gini(debit_exp),
        "debit_exposure_hhi": _hhi(debit_exp),
        "q_mean": q_mean,
        "q_std": q_std,
        "q_min": q_min,
        "q_max": q_max,
        "q_annual_mean_min": q_annual_mean_min,
        "q_annual_mean_max": q_annual_mean_max,
        "H_mean": float(np.mean(H)) if len(H) else math.nan,
        "H_min": float(np.min(H)) if len(H) else math.nan,
        "H_max": float(np.max(H)) if len(H) else math.nan,
        "Hbar_mean": float(np.mean(Hbar)) if len(Hbar) else math.nan,
        "mip_gap": float(getattr(result, "mip_gap", math.nan)),
        "obj_bound": float(getattr(result, "obj_bound", math.nan)),
        "runtime_sec": float(getattr(result, "runtime", math.nan)),
    }


def _pairwise(results: pd.DataFrame) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    for (year, rho), g in results.groupby(["year", "rho"], dropna=False):
        rows = {str(r.rule): r for r in g.itertuples(index=False)}
        if "M2" not in rows:
            continue
        base = rows["M2"]
        for rule in ["M1", "M3"]:
            if rule not in rows:
                continue
            comp = rows[rule]
            out.append({
                "year": int(year),
                "rho": float(rho),
                "comparison": f"{rule}_minus_M2",
                "cost_delta": float(comp.cost_total - base.cost_total),
                "cost_saving_vs_M2": float(base.cost_total - comp.cost_total),
                "slack_delta": float(comp.slack_total - base.slack_total),
                "relief_share_delta": float(comp.relief_served_share - base.relief_served_share),
                "mu_positive_share_delta": float(comp.mu_positive_share - base.mu_positive_share),
            })
    return pd.DataFrame(out)


# =============================================================================
# OUTPUTS
# =============================================================================

def _make_outputs(results: pd.DataFrame) -> None:
    results_path = RESULTS_DIR / "scarcity_boundary_results.csv"
    pair_path = RESULTS_DIR / "scarcity_boundary_pairwise.csv"
    results.to_csv(results_path, index=False)
    pairs = _pairwise(results)
    if not pairs.empty:
        pairs.to_csv(pair_path, index=False)

    summary = (
        results.groupby(["rule", "rho"], dropna=False)
        .agg(
            n_years=("year", "nunique"),
            cost_mean=("cost_total", "mean"),
            slack_mean=("slack_total", "mean"),
            relief_served_share_mean=("relief_served_share", "mean"),
            mu_positive_share_mean=("mu_positive_share", "mean"),
            budget_util_mean=("budget_util_mean", "mean"),
            q_std_mean=("q_std", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES_DIR / "T8_scarcity_boundary_summary.csv", index=False)
    try:
        summary.to_latex(TABLES_DIR / "T8_scarcity_boundary_summary.tex", index=False, float_format="%.4g")
    except Exception as exc:
        print(f"[WARN] Could not write LaTeX summary: {exc}")

    if not pairs.empty:
        pair_summary = (
            pairs.groupby(["comparison", "rho"], dropna=False)
            .agg(
                n=("year", "count"),
                cost_saving_mean=("cost_saving_vs_M2", "mean"),
                cost_saving_min=("cost_saving_vs_M2", "min"),
                cost_saving_max=("cost_saving_vs_M2", "max"),
                slack_delta_mean=("slack_delta", "mean"),
                relief_share_delta_mean=("relief_share_delta", "mean"),
            )
            .reset_index()
        )
        pair_summary.to_csv(TABLES_DIR / "T8_scarcity_boundary_pairwise.csv", index=False)
        try:
            pair_summary.to_latex(TABLES_DIR / "T8_scarcity_boundary_pairwise.tex", index=False, float_format="%.4g")
        except Exception:
            pass

    _make_figures(results, pairs)
    _diagnose(results, pairs)

    print(f"\n✓ Wrote {results_path}")
    if not pairs.empty:
        print(f"✓ Wrote {pair_path}")
    print(f"✓ Wrote {TABLES_DIR / 'T8_scarcity_boundary_summary.csv'}")


def _make_figures(results: pd.DataFrame, pairs: pd.DataFrame) -> None:
    if results.empty:
        return
    agg = (
        results.groupby(["rule", "rho"], dropna=False)
        .agg(cost=("cost_total", "mean"), mu_share=("mu_positive_share", "mean"), slack=("slack_total", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for rule, g in agg.groupby("rule", sort=False):
        g = g.sort_values("rho")
        ax.plot(g["rho"], g["cost"] / 1e6, marker="o", label=str(rule))
    ax.set_xlabel(r"Allowance generosity $\rho$")
    ax.set_ylabel("Mean annual operating cost (million units)")
    ax.set_title("Scarcity boundary: cost")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F10_scarcity_boundary_cost.pdf")
    fig.savefig(FIGURES_DIR / "F10_scarcity_boundary_cost.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for rule, g in agg.groupby("rule", sort=False):
        if rule == "M1":
            continue
        g = g.sort_values("rho")
        ax.plot(g["rho"], 100.0 * g["mu_share"], marker="o", label=str(rule))
    ax.set_xlabel(r"Allowance generosity $\rho$")
    ax.set_ylabel("Positive annual-budget multipliers (%)")
    ax.set_title("Scarcity boundary: active annual scarcity")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F10_scarcity_boundary_mu.pdf")
    fig.savefig(FIGURES_DIR / "F10_scarcity_boundary_mu.png", dpi=300)
    plt.close(fig)

    if not pairs.empty and "M3_minus_M2" in set(pairs["comparison"]):
        d = pairs[pairs["comparison"] == "M3_minus_M2"].copy()
        d_agg = d.groupby("rho", dropna=False)["cost_saving_vs_M2"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6.8, 3.0))
        ax.axhline(0.0, linewidth=1.0)
        ax.plot(d_agg["rho"], d_agg["cost_saving_vs_M2"] / 1e6, marker="o")
        ax.set_xlabel(r"Allowance generosity $\rho$")
        ax.set_ylabel("Mean M3 cost saving vs M2 (million units)")
        ax.set_title("Scarcity boundary: M3 value relative to M2")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "F10_scarcity_boundary_saving.pdf")
        fig.savefig(FIGURES_DIR / "F10_scarcity_boundary_saving.png", dpi=300)
        plt.close(fig)


def _diagnose(results: pd.DataFrame, pairs: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("SCARCITY-BOUNDARY DIAGNOSTIC")
    print("=" * 78)
    if results.empty:
        print("No results.")
        return
    m2 = results[results["rule"] == "M2"].copy()
    if not m2.empty:
        by_rho = m2.groupby("rho")["mu_positive_share"].mean().sort_index()
        active = by_rho[by_rho >= 0.05]
        if active.empty:
            print("M2 positive-mu share never reaches 5%. Your tested rho grid is too loose or R/P calibration is too weak.")
        else:
            print(f"First rho with mean M2 positive-mu share >= 5%: {float(active.index[0]):.3g}")
        print("M2 positive-mu share by rho:")
        for rho, val in by_rho.items():
            print(f"  rho={rho:.3g}: {100*val:.2f}%")

    m3 = results[results["rule"] == "M3"].copy()
    if not m3.empty:
        qstd = float(m3["q_std"].mean())
        print(f"Mean M3 q_std over all rho/year solves: {qstd:.4f}")
        if qstd < 0.05:
            print("  -> M3 is nearly flat. The weak results are likely from the learned q, not only from the LP regime.")

    if not pairs.empty and "M3_minus_M2" in set(pairs["comparison"]):
        d = pairs[pairs["comparison"] == "M3_minus_M2"]
        best = d.groupby("rho")["cost_saving_vs_M2"].mean().sort_values(ascending=False)
        if not best.empty:
            print(f"Best mean M3 saving vs M2 occurs at rho={float(best.index[0]):.3g}: {float(best.iloc[0]):.4g}")
            if float(best.iloc[0]) <= 0:
                print("  -> M3 does not beat M2 even in the best tested scarcity regime. Revisit proxy target/features or PTDF directionality.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Scarcity boundary rho-sensitivity for debit-rule paper.")
    parser.add_argument("--rho-grid", default="0.40,0.50,0.60,0.70,0.80,0.90,1.00")
    parser.add_argument("--years", default="", help="Comma-separated years. Default: cfg.EVAL_YEARS if set, else all LP years.")
    parser.add_argument("--feature-set", default=MAIN_FEATURE_SET)
    parser.add_argument("--variant", default=MAIN_M3_VARIANT)
    parser.add_argument("--include-m1", type=int, default=0, help="1=solve M1 MILP; default 0 because rho grid can be expensive.")
    parser.add_argument("--allowance-mode", choices=["scale-current", "reconstruct-province"], default="scale-current")
    parser.add_argument("--network-negative-policy", choices=["clip", "allow", "error"], default="clip")
    parser.add_argument("--strict-weights", type=int, default=1)
    parser.add_argument("--time-limit-lp", type=float, default=float(getattr(cfg, "LP_TIME_LIMIT", 180.0)))
    parser.add_argument("--time-limit-milp", type=float, default=float(getattr(cfg, "M1_TIME_LIMIT", 600.0)))
    args = parser.parse_args()

    rho_grid = _parse_csv_floats(args.rho_grid)
    if not rho_grid:
        raise ValueError("rho grid is empty.")
    all_years = _available_lp_years()
    if args.years.strip():
        years = _parse_csv_ints(args.years)
    else:
        years = [int(y) for y in getattr(cfg, "EVAL_YEARS", [])] or [int(y) for y in getattr(cfg, "TEST_YEARS", [])] or all_years
    missing_years = sorted(set(years).difference(all_years))
    if missing_years:
        raise FileNotFoundError(f"Requested years not present in LP_DIR: {missing_years}; available={all_years}")

    meta = _lp_meta()
    rho_base = _base_rho(meta)
    weights = _load_m3_weights()

    print("=" * 78)
    print("F: SCARCITY BOUNDARY")
    print("=" * 78)
    print(f"LP_DIR        : {LP_DIR}")
    print(f"M3 weights    : {M3_WEIGHTS_PATH}")
    print(f"Years         : {years}")
    print(f"Rho grid      : {rho_grid}")
    print(f"Base rho      : {rho_base}")
    print(f"Allowance mode: {args.allowance_mode}")
    print(f"Feature/var   : {args.feature_set} / {args.variant}")
    print(f"Include M1    : {bool(args.include_m1)}")
    print()

    rows: List[Dict[str, Any]] = []
    for year in years:
        print("-" * 78)
        print(f"Year {year}")
        lp = _load_lp_year(year)
        base_arrays = _annual_arrays(lp, negative_policy=args.network_negative_policy)
        print(f"  mode={base_arrays['mode']}, T={len(base_arrays['timestamps'])}, N={len(base_arrays['provinces'])}")
        if base_arrays["mode"] == "network":
            print(
                f"  K={len(base_arrays['branches'])}, a_col={base_arrays.get('a_column')}, "
                f"neg_seen={base_arrays.get('negative_coefficients_seen', 0)}, "
                f"neg_policy={base_arrays.get('negative_policy')}"
            )

        q2 = np.ones((len(base_arrays["provinces"]), len(base_arrays["timestamps"])), dtype=float)
        q3 = _q_matrix(
            weights=weights,
            year=year,
            feature_set=args.feature_set,
            variant=args.variant,
            timestamps=base_arrays["timestamps"],
            provinces=base_arrays["provinces"],
            strict=bool(args.strict_weights),
        )

        for rho in rho_grid:
            arrays = _apply_rho(base_arrays, rho=float(rho), rho_base=rho_base, allowance_mode=args.allowance_mode)
            print(f"  rho={rho:.3g}: H_mean={np.mean(arrays['H']):.4g}, Hbar_mean={np.mean(arrays['Hbar']):.4g}")

            if bool(args.include_m1):
                r1 = _solve_rule(arrays, "M1", None, include_m1=True, time_limit_lp=args.time_limit_lp, time_limit_milp=args.time_limit_milp)
                if r1 is not None:
                    rows.append(_summarize(year, rho, "M1", "none", "M1", "not_applicable", arrays, r1, None))
                    print(f"    M1 cost={r1.cost_total:.4g}, slack={r1.slack_total:.4g}, status={r1.status}")

            r2 = _solve_rule(arrays, "M2", q2, include_m1=False, time_limit_lp=args.time_limit_lp, time_limit_milp=args.time_limit_milp)
            rows.append(_summarize(year, rho, "M2", "uniform", "M2", "q_equals_one", arrays, r2, q2))
            print(f"    M2 cost={r2.cost_total:.4g}, slack={r2.slack_total:.4g}, mu+={np.mean(r2.mu > 1e-7):.3f}, status={r2.status}")

            r3 = _solve_rule(arrays, "M3", q3, include_m1=False, time_limit_lp=args.time_limit_lp, time_limit_milp=args.time_limit_milp)
            rows.append(_summarize(year, rho, "M3", args.feature_set, args.variant, "precomputed_m3_weights", arrays, r3, q3))
            print(f"    M3 cost={r3.cost_total:.4g}, slack={r3.slack_total:.4g}, mu+={np.mean(r3.mu > 1e-7):.3f}, q_std={np.std(q3):.4f}, status={r3.status}")

    results = pd.DataFrame(rows)
    _make_outputs(results)


if __name__ == "__main__":
    main()
