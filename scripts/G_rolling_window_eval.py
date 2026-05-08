#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G_rolling_window_eval.py
========================

Blocked annual rolling-window evaluation for the annual debit-rule paper.

Purpose
-------
This script adds the robustness experiment that IEEE TSG reviewers will expect:
performance is reported across several annual holdout years, not only one final
holdout year.

The script is designed to work with the current pipeline outputs:

    data/derived/lp_params/lp_params_{year}.parquet
    data/derived/lp_params/lp_meta.json
    data/derived/m3_debit_weights_by_year.parquet

It supports two LP schemas:

1) Province schema, already used by the current B/C/D scripts:
       province, timestamp_utc, R_value, c_value, H_i, Hbar_i, P_i, Pbar_value

2) PTDF/network schema, if produced by your PTDF bridge script:
       branch_id, province, timestamp_utc, A_value or a_relief,
       R_value, c_value, H_i, Hbar_i, P_i, Pbar_value

Important interpretation
------------------------
By default, this script uses the precomputed M3 weights already written by
C_train_m3.py. Therefore it is a blocked annual EVALUATION script, not a full
retraining script. It records q_source="precomputed_m3_weights" in the output.

That is still useful for finalizing the paper because it answers:
    "Are the operating conclusions stable across holdout years?"

If you need strict retraining per fold, run C_train_m3.py separately per fold
with environment-overridden TRAIN_YEARS/VAL_YEARS/TEST_YEARS and then run this
script on the corresponding saved weights. Do not silently call that strict
rolling-window evidence unless the weights were actually retrained per fold.

Run
---
    python scripts/G_rolling_window_eval.py

Options
-------
    --include-m1 0              skip M1 MILP if runtime is too high
    --feature-set F2            main M3 feature set
    --variant M3_LO             main M3 variant
    --network-negative-policy clip   {clip, allow, error}

Outputs
-------
    outputs/results/rolling_window_results.csv
    outputs/tables/T7_rolling_window_summary.csv
    outputs/tables/T7_rolling_window_summary.tex
    outputs/figures/F9_rolling_window_cost_slack.pdf/png
    outputs/figures/F9_rolling_window_scarcity.pdf/png
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
except Exception:  # pragma: no cover - only needed for network schema
    gp = None
    GRB = None


# =============================================================================
# CONFIG DEFAULTS
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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# RESULT OBJECTS FOR INTERNAL NETWORK SOLVERS
# =============================================================================

@dataclass
class NetLPResult:
    x: np.ndarray                    # (N,T)
    s: np.ndarray                    # (K,T)
    mu: np.ndarray                   # (N,)
    lambda_relief: np.ndarray        # (K,T)
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

def _as_int_list(x: Iterable[Any]) -> List[int]:
    return [int(v) for v in x]


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
        stem = p.stem.replace("lp_params_", "")
        try:
            years.append(int(stem))
        except ValueError:
            continue
    return sorted(set(years))


def _load_lp_year(year: int) -> pd.DataFrame:
    path = LP_DIR / f"lp_params_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing LP params for year {year}: {path}")
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


def _load_m3_weights() -> pd.DataFrame:
    if not M3_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing M3 weights: {M3_WEIGHTS_PATH}\n"
            "Run C_train_m3.py first, or pass a config with M3_WEIGHTS_PATH."
        )
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
    if df["debit_weight"].isna().any():
        raise ValueError("M3 weights contain NaN debit_weight values.")
    if (df["debit_weight"] <= 0).any():
        raise ValueError("M3 weights contain nonpositive debit_weight values.")
    return df


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
# PROVINCE-SCHEMA ARRAYS
# =============================================================================

def _province_arrays(lp_year: pd.DataFrame) -> Dict[str, Any]:
    provinces = NL_PROVINCES[:]
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(lp_year["timestamp_utc"].drop_duplicates(), utc=True).sort_values()
    )
    N, T = len(provinces), len(timestamps)

    required = {"province", "timestamp_utc", "R_value", "c_value", "H_i", "Hbar_i", "P_i"}
    missing = sorted(required.difference(lp_year.columns))
    if missing:
        raise KeyError(f"Province LP schema missing required columns: {missing}")

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
        "H": np.maximum(np.nan_to_num(H), 0.0),
        "Hbar": np.maximum(np.nan_to_num(Hbar), 0.0),
        "P": np.maximum(np.nan_to_num(P, nan=1.0), 1e-9),
        "Pbar": np.maximum(np.nan_to_num(Pbar), 0.0),
    }


# =============================================================================
# NETWORK-SCHEMA ARRAYS
# =============================================================================

def _network_arrays(lp_year: pd.DataFrame, negative_policy: str) -> Dict[str, Any]:
    if negative_policy not in {"clip", "allow", "error"}:
        raise ValueError("negative_policy must be one of: clip, allow, error")

    provinces = NL_PROVINCES[:]
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(lp_year["timestamp_utc"].drop_duplicates(), utc=True).sort_values()
    )
    branches = sorted(pd.to_numeric(lp_year["branch_id"], errors="coerce").dropna().astype(int).unique().tolist())
    if not branches:
        raise ValueError("Network LP schema has no valid branch_id values.")

    a_col = "a_relief" if "a_relief" in lp_year.columns else "A_value"
    required = {"branch_id", "province", "timestamp_utc", a_col, "R_value", "c_value", "H_i", "Hbar_i", "P_i"}
    missing = sorted(required.difference(lp_year.columns))
    if missing:
        raise KeyError(f"Network LP schema missing required columns: {missing}")

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

    # Branch relief requirements: same R[k,t] repeated over provinces; take max/first safely.
    rel = (
        df.groupby(["branch_id", "timestamp_utc"], observed=True)["R_value"]
        .max()
        .reset_index()
    )
    for r in rel.itertuples(index=False):
        bid = int(getattr(r, "branch_id"))
        ts = getattr(r, "timestamp_utc")
        if bid in b_to_k and ts in t_to_j:
            R[b_to_k[bid], t_to_j[ts]] = float(getattr(r, "R_value"))

    # Effectiveness coefficients. They may be constant over time or repeated over time.
    coeff = df[["branch_id", "province", "timestamp_utc", a_col]].dropna(subset=["branch_id", "province", "timestamp_utc"])
    neg_count = int((coeff[a_col] < -1e-12).sum())
    if neg_count and negative_policy == "error":
        raise ValueError(
            f"Found {neg_count} negative network coefficients in {a_col}. "
            "Use directional a_relief or run with --network-negative-policy clip/allow."
        )
    for r in coeff.itertuples(index=False):
        bid = int(getattr(r, "branch_id"))
        prov = str(getattr(r, "province"))
        ts = getattr(r, "timestamp_utc")
        val = float(getattr(r, a_col))
        if negative_policy == "clip":
            val = max(val, 0.0)
        if bid in b_to_k and prov in p_to_n and ts in t_to_j:
            A[b_to_k[bid], p_to_n[prov], t_to_j[ts]] = val

    # Province-time cost/capacity. Values are repeated over branches; use first.
    prov_time = (
        df.groupby(["province", "timestamp_utc"], observed=True)
        .agg(
            c_value=("c_value", "first"),
            Pbar_value=("Pbar_value", "first") if "Pbar_value" in df.columns else ("P_i", "first"),
        )
        .reset_index()
    )
    prov_static = (
        df.groupby("province", observed=True)
        .agg(H_i=("H_i", "first"), Hbar_i=("Hbar_i", "first"), P_i=("P_i", "first"))
        .reset_index()
    )
    for r in prov_static.itertuples(index=False):
        prov = str(getattr(r, "province"))
        if prov in p_to_n:
            n = p_to_n[prov]
            H[n] = float(pd.to_numeric(getattr(r, "H_i"), errors="coerce"))
            Hbar[n] = float(pd.to_numeric(getattr(r, "Hbar_i"), errors="coerce"))
            P[n] = float(pd.to_numeric(getattr(r, "P_i"), errors="coerce"))
    for r in prov_time.itertuples(index=False):
        prov = str(getattr(r, "province"))
        ts = getattr(r, "timestamp_utc")
        if prov in p_to_n and ts in t_to_j:
            n, t = p_to_n[prov], t_to_j[ts]
            C[n, t] = float(pd.to_numeric(getattr(r, "c_value"), errors="coerce"))
            Pbar[n, t] = float(pd.to_numeric(getattr(r, "Pbar_value"), errors="coerce"))

    # Fill missing Pbar with P.
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
        "H": np.maximum(np.nan_to_num(H), 0.0),
        "Hbar": np.maximum(np.nan_to_num(Hbar), 0.0),
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
# Q MATRIX
# =============================================================================

def _q_matrix(
    weights: pd.DataFrame,
    year: int,
    feature_set: str,
    variant: str,
    timestamps: Sequence[pd.Timestamp],
    provinces: Sequence[str],
    *,
    strict: bool = True,
) -> np.ndarray:
    q = np.ones((len(provinces), len(timestamps)), dtype=float)
    ts_order = pd.Index(pd.to_datetime(list(timestamps), utc=True))
    subset = weights[
        (weights["year"].astype(int) == int(year))
        & (weights["feature_set"] == feature_set)
        & (weights["variant"] == variant)
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
        if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
            raise ValueError(f"Invalid M3 weights for province={province}, year={year}.")
        q[n, :] = vals
    return q


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

    A = np.asarray(A, dtype=float)
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)
    q = np.asarray(q, dtype=float)
    H = np.asarray(H, dtype=float)
    P = np.asarray(P, dtype=float)
    Pbar = np.asarray(Pbar, dtype=float)

    K, N, T = A.shape
    if R.shape != (K, T):
        raise ValueError(f"R shape {R.shape}, expected {(K, T)}")
    if C.shape != (N, T) or q.shape != (N, T) or Pbar.shape != (N, T):
        raise ValueError("C, q, and Pbar must have shape (N,T).")

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0 if quiet else 1)
    env.setParam("LogToConsole", 0 if quiet else 1)
    env.start()
    model = gp.Model("weighted_depth_network_lp", env=env)
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

    if has_sol:
        x_sol = np.asarray(x.X, dtype=float).reshape(N, T)
        s_sol = np.asarray(s.X, dtype=float).reshape(K, T)
    else:
        x_sol = np.zeros((N, T), dtype=float)
        s_sol = np.zeros((K, T), dtype=float)

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

    A = np.asarray(A, dtype=float)
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)
    Hbar = np.asarray(Hbar, dtype=float)
    Pbar = np.asarray(Pbar, dtype=float)
    K, N, T = A.shape

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0 if quiet else 1)
    env.setParam("LogToConsole", 0 if quiet else 1)
    env.start()
    model = gp.Model("touched_time_network_milp", env=env)
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

    if has_sol:
        x_sol = np.asarray(x.X, dtype=float).reshape(N, T)
        s_sol = np.asarray(s.X, dtype=float).reshape(K, T)
    else:
        x_sol = np.zeros((N, T), dtype=float)
        s_sol = np.zeros((K, T), dtype=float)

    cost_curtail = float(np.sum(C * x_sol))
    slack_total = float(np.sum(s_sol))
    cost_slack = float(M_penalty) * slack_total
    obj = cost_curtail + cost_slack
    touched = (x_sol > 1e-7).sum(axis=1)
    util = float(np.mean(touched / np.maximum(Hbar, 1e-9)))
    gap = float(model.MIPGap) if hasattr(model, "MIPGap") and np.isfinite(model.MIPGap) else math.nan
    bound = float(model.ObjBound) if has_sol and hasattr(model, "ObjBound") else math.nan

    model.dispose()
    env.dispose()
    return NetLPResult(x_sol, s_sol, np.zeros(N), np.zeros((K, T)), obj, cost_curtail, cost_slack, slack_total, util, status, runtime, gap, bound)


# =============================================================================
# EVALUATION
# =============================================================================

def _summarize_solution(
    *,
    year: int,
    fold_id: str,
    train_years: Sequence[int],
    val_year: Optional[int],
    test_year: int,
    rule: str,
    comparison_group: str,
    feature_set: str,
    variant: str,
    q_source: str,
    mode: str,
    arrays: Dict[str, Any],
    result: Any,
    q: Optional[np.ndarray],
) -> Dict[str, Any]:
    x = np.asarray(result.x, dtype=float)
    s = np.asarray(result.s, dtype=float)
    P = np.asarray(arrays["P"], dtype=float)
    H = np.asarray(arrays["H"], dtype=float)
    Hbar = np.asarray(arrays["Hbar"], dtype=float)

    if mode == "province":
        R_total = float(np.sum(arrays["R"]))
    else:
        R_total = float(np.sum(arrays["R"]))

    slack_total = float(np.sum(s))
    relief_served = max(R_total - slack_total, 0.0)
    physical_exposure = np.sum(x / P[:, None], axis=1)

    if rule == "M1":
        debit_exposure = np.sum(x > 1e-7, axis=1).astype(float)
        q_mean = q_std = q_min = q_max = q_annual_mean_min = q_annual_mean_max = math.nan
    else:
        assert q is not None
        debit_exposure = np.sum(q * x / P[:, None], axis=1)
        q_mean = float(np.mean(q))
        q_std = float(np.std(q))
        q_min = float(np.min(q))
        q_max = float(np.max(q))
        q_ann = np.mean(q, axis=1)
        q_annual_mean_min = float(np.min(q_ann))
        q_annual_mean_max = float(np.max(q_ann))

    mu = np.asarray(getattr(result, "mu", np.zeros(len(P))), dtype=float)
    return {
        "fold_id": fold_id,
        "train_years": ",".join(str(int(y)) for y in train_years),
        "val_year": int(val_year) if val_year is not None else None,
        "test_year": int(test_year),
        "year": int(year),
        "mode": mode,
        "rule": rule,
        "comparison_group": comparison_group,
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
        "physical_exposure_total": float(np.sum(physical_exposure)),
        "physical_exposure_gini": _gini(physical_exposure),
        "physical_exposure_hhi": _hhi(physical_exposure),
        "debit_exposure_total": float(np.sum(debit_exposure)),
        "debit_exposure_gini": _gini(debit_exposure),
        "debit_exposure_hhi": _hhi(debit_exposure),
        "q_mean": q_mean,
        "q_std": q_std,
        "q_min": q_min,
        "q_max": q_max,
        "q_annual_mean_min": q_annual_mean_min,
        "q_annual_mean_max": q_annual_mean_max,
        "mip_gap": float(getattr(result, "mip_gap", math.nan)),
        "obj_bound": float(getattr(result, "obj_bound", math.nan)),
        "runtime_sec": float(getattr(result, "runtime", math.nan)),
        "H_mean": float(np.mean(H)) if len(H) else math.nan,
        "Hbar_mean": float(np.mean(Hbar)) if len(Hbar) else math.nan,
    }


def _solve_rule(
    arrays: Dict[str, Any],
    rule: str,
    q: Optional[np.ndarray],
    *,
    include_m1: bool,
    time_limit_lp: float,
    time_limit_milp: float,
) -> Any:
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

    # Network schema.
    A, R, C, H, Hbar, P, Pbar = arrays["A"], arrays["R"], arrays["C"], arrays["H"], arrays["Hbar"], arrays["P"], arrays["Pbar"]
    if rule == "M1":
        if not include_m1:
            return None
        return _solve_network_touched_milp(A=A, R=R, C=C, Hbar=Hbar, Pbar=Pbar, M_penalty=M_PENALTY, quiet=True, time_limit=time_limit_milp)
    if q is None:
        raise ValueError("Weighted-depth rule requires q.")
    return _solve_network_weighted_lp(A=A, R=R, C=C, q=q, H=H, P=P, Pbar=Pbar, M_penalty=M_PENALTY, quiet=True, time_limit=time_limit_lp)


def _default_folds(years: Sequence[int]) -> List[Tuple[List[int], Optional[int], int]]:
    years = sorted(int(y) for y in years)
    folds: List[Tuple[List[int], Optional[int], int]] = []
    if len(years) >= 5:
        # Expanding-window folds: train prefix, validate preceding year, test next year.
        for idx in range(2, len(years)):
            train = years[: idx - 1]
            val = years[idx - 1]
            test = years[idx]
            folds.append((train, val, test))
    elif len(years) >= 3:
        for idx in range(2, len(years)):
            folds.append((years[: idx - 1], years[idx - 1], years[idx]))
    else:
        raise ValueError("Need at least 3 LP years for rolling evaluation.")
    return folds


def _paired_differences(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold_id", "test_year", "year"]
    out: List[Dict[str, Any]] = []
    for key, g in rows.groupby(keys, dropna=False):
        vals = {str(r.rule): r for r in g.itertuples(index=False)}
        if "M2" not in vals:
            continue
        base = vals["M2"]
        for rule in ["M1", "M3"]:
            if rule not in vals:
                continue
            comp = vals[rule]
            out.append({
                "fold_id": key[0],
                "test_year": key[1],
                "year": key[2],
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

def _write_outputs(results: pd.DataFrame) -> None:
    out_path = RESULTS_DIR / "rolling_window_results.csv"
    diff_path = RESULTS_DIR / "rolling_window_pairwise_differences.csv"
    results.to_csv(out_path, index=False)

    diffs = _paired_differences(results)
    if not diffs.empty:
        diffs.to_csv(diff_path, index=False)

    summary = (
        results.groupby(["rule", "feature_set", "variant"], dropna=False)
        .agg(
            n_years=("year", "nunique"),
            cost_mean=("cost_total", "mean"),
            cost_std=("cost_total", "std"),
            slack_mean=("slack_total", "mean"),
            relief_served_share_mean=("relief_served_share", "mean"),
            mu_positive_share_mean=("mu_positive_share", "mean"),
            budget_util_mean=("budget_util_mean", "mean"),
            q_std_mean=("q_std", "mean"),
            runtime_sec_sum=("runtime_sec", "sum"),
        )
        .reset_index()
    )
    summary_path = TABLES_DIR / "T7_rolling_window_summary.csv"
    summary.to_csv(summary_path, index=False)
    try:
        summary.to_latex(TABLES_DIR / "T7_rolling_window_summary.tex", index=False, float_format="%.4g")
    except Exception as exc:
        print(f"[WARN] Could not write LaTeX table: {exc}")

    if diffs.empty:
        print("[WARN] No pairwise differences written because M2/M3 or M1 rows are missing.")
    else:
        diff_summary = (
            diffs.groupby("comparison", dropna=False)
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
        diff_summary.to_csv(TABLES_DIR / "T7_rolling_window_pairwise_summary.csv", index=False)
        try:
            diff_summary.to_latex(TABLES_DIR / "T7_rolling_window_pairwise_summary.tex", index=False, float_format="%.4g")
        except Exception:
            pass

    _make_figures(results)

    print(f"\n✓ Wrote {out_path}")
    if not diffs.empty:
        print(f"✓ Wrote {diff_path}")
    print(f"✓ Wrote {summary_path}")


def _make_figures(results: pd.DataFrame) -> None:
    if results.empty:
        return

    main = results[results["rule"].isin(["M1", "M2", "M3"])].copy()
    if main.empty:
        return

    # Figure 1: cost by holdout year.
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for rule, g in main.groupby("rule", sort=False):
        g = g.sort_values("year")
        ax.plot(g["year"], g["cost_total"] / 1e6, marker="o", label=str(rule))
    ax.set_xlabel("Holdout year")
    ax.set_ylabel("Annual operating cost (million units)")
    ax.set_title("Rolling-window holdout cost")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F9_rolling_window_cost.pdf")
    fig.savefig(FIGURES_DIR / "F9_rolling_window_cost.png", dpi=300)
    plt.close(fig)

    # Figure 2: scarcity activation by holdout year.
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for rule, g in main.groupby("rule", sort=False):
        if rule == "M1":
            continue
        g = g.sort_values("year")
        ax.plot(g["year"], 100.0 * g["mu_positive_share"], marker="o", label=str(rule))
    ax.set_xlabel("Holdout year")
    ax.set_ylabel("Positive annual-budget multipliers (%)")
    ax.set_title("Rolling-window annual scarcity diagnostic")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F9_rolling_window_scarcity.pdf")
    fig.savefig(FIGURES_DIR / "F9_rolling_window_scarcity.png", dpi=300)
    plt.close(fig)


def _diagnose(results: pd.DataFrame) -> None:
    if results.empty:
        print("[WARN] No results to diagnose.")
        return
    print("\n" + "=" * 78)
    print("ROLLING-WINDOW DIAGNOSTIC")
    print("=" * 78)
    m2 = results[results["rule"] == "M2"]
    m3 = results[results["rule"] == "M3"]
    if not m2.empty:
        scarcity = float(m2["mu_positive_share"].mean())
        print(f"M2 mean positive-mu share: {100*scarcity:.2f}%")
        if scarcity < 0.05:
            print("  -> Annual budgets are mostly slack. M3 cannot show much operational value in this regime.")
    if not m3.empty:
        qstd = float(m3["q_std"].mean())
        print(f"M3 mean q standard deviation: {qstd:.4f}")
        if qstd < 0.05:
            print("  -> Learned debit weights are nearly flat; this is effectively M2.")
    diffs = _paired_differences(results)
    if not diffs.empty and "M3_minus_M2" in set(diffs["comparison"]):
        d = diffs[diffs["comparison"] == "M3_minus_M2"]
        frac = float((d["cost_saving_vs_M2"] > 0).mean())
        avg = float(d["cost_saving_vs_M2"].mean())
        print(f"M3 beats M2 by cost in {100*frac:.1f}% of holdout years; mean saving={avg:.4g}.")
        if frac < 0.5:
            print("  -> Do not claim robust M3 dominance. Use the result to discuss regime dependence or proxy mismatch.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-window evaluation for debit-rule paper.")
    parser.add_argument("--feature-set", default=MAIN_FEATURE_SET)
    parser.add_argument("--variant", default=MAIN_M3_VARIANT)
    parser.add_argument("--include-m1", type=int, default=1, help="1=solve M1 MILP, 0=skip.")
    parser.add_argument("--time-limit-lp", type=float, default=float(getattr(cfg, "LP_TIME_LIMIT", 180.0)))
    parser.add_argument("--time-limit-milp", type=float, default=float(getattr(cfg, "M1_TIME_LIMIT", 600.0)))
    parser.add_argument("--network-negative-policy", choices=["clip", "allow", "error"], default="clip")
    parser.add_argument("--strict-weights", type=int, default=1)
    args = parser.parse_args()

    years = _available_lp_years()
    if not years:
        raise FileNotFoundError(f"No lp_params_*.parquet files found in {LP_DIR}.")
    folds = _default_folds(years)

    print("=" * 78)
    print("G: ROLLING-WINDOW EVALUATION")
    print("=" * 78)
    print(f"LP_DIR        : {LP_DIR}")
    print(f"M3 weights    : {M3_WEIGHTS_PATH}")
    print(f"Years         : {years}")
    print(f"Feature/var   : {args.feature_set} / {args.variant}")
    print(f"Include M1    : {bool(args.include_m1)}")
    print(f"Folds         : {[(tr, va, te) for tr, va, te in folds]}")
    print()

    weights = _load_m3_weights()
    meta = _lp_meta()
    print(f"LP meta mode  : {meta.get('mode', 'unknown')}")

    rows: List[Dict[str, Any]] = []
    for idx, (train_years, val_year, test_year) in enumerate(folds, start=1):
        fold_id = f"fold_{idx:02d}"
        print("-" * 78)
        print(f"{fold_id}: train={train_years}, val={val_year}, test={test_year}")
        lp = _load_lp_year(test_year)
        arrays = _annual_arrays(lp, negative_policy=args.network_negative_policy)
        mode = arrays["mode"]
        print(f"  mode={mode}, T={len(arrays['timestamps'])}, N={len(arrays['provinces'])}")
        if mode == "network":
            print(
                f"  K={len(arrays['branches'])}, a_col={arrays.get('a_column')}, "
                f"neg_seen={arrays.get('negative_coefficients_seen', 0)}, "
                f"neg_policy={arrays.get('negative_policy')}"
            )

        # M1.
        if bool(args.include_m1):
            print("  Solving M1 ...")
            r1 = _solve_rule(arrays, "M1", None, include_m1=True, time_limit_lp=args.time_limit_lp, time_limit_milp=args.time_limit_milp)
            if r1 is not None:
                rows.append(_summarize_solution(
                    year=test_year, fold_id=fold_id, train_years=train_years, val_year=val_year, test_year=test_year,
                    rule="M1", comparison_group="rolling_main", feature_set="none", variant="M1",
                    q_source="not_applicable", mode=mode, arrays=arrays, result=r1, q=None,
                ))
                print(f"    M1 status={r1.status}, cost={r1.cost_total:.4g}, slack={r1.slack_total:.4g}")

        # M2.
        print("  Solving M2 ...")
        q2 = np.ones((len(arrays["provinces"]), len(arrays["timestamps"])), dtype=float)
        r2 = _solve_rule(arrays, "M2", q2, include_m1=False, time_limit_lp=args.time_limit_lp, time_limit_milp=args.time_limit_milp)
        rows.append(_summarize_solution(
            year=test_year, fold_id=fold_id, train_years=train_years, val_year=val_year, test_year=test_year,
            rule="M2", comparison_group="rolling_main", feature_set="uniform", variant="M2",
            q_source="q_equals_one", mode=mode, arrays=arrays, result=r2, q=q2,
        ))
        print(f"    M2 status={r2.status}, cost={r2.cost_total:.4g}, slack={r2.slack_total:.4g}, mu+={np.mean(r2.mu > 1e-7):.3f}")

        # M3.
        print("  Solving M3 ...")
        q3 = _q_matrix(
            weights=weights,
            year=test_year,
            feature_set=args.feature_set,
            variant=args.variant,
            timestamps=arrays["timestamps"],
            provinces=arrays["provinces"],
            strict=bool(args.strict_weights),
        )
        r3 = _solve_rule(arrays, "M3", q3, include_m1=False, time_limit_lp=args.time_limit_lp, time_limit_milp=args.time_limit_milp)
        rows.append(_summarize_solution(
            year=test_year, fold_id=fold_id, train_years=train_years, val_year=val_year, test_year=test_year,
            rule="M3", comparison_group="rolling_main", feature_set=args.feature_set, variant=args.variant,
            q_source="precomputed_m3_weights", mode=mode, arrays=arrays, result=r3, q=q3,
        ))
        print(f"    M3 status={r3.status}, cost={r3.cost_total:.4g}, slack={r3.slack_total:.4g}, mu+={np.mean(r3.mu > 1e-7):.3f}, q_std={np.std(q3):.4f}")

    results = pd.DataFrame(rows)
    _write_outputs(results)
    _diagnose(results)


if __name__ == "__main__":
    main()
