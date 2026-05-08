"""
TU modular energy recourse experiment with exact Wasserstein reformulation.

Application:
    Modular backup / reinforcement planning on a radial feeder or EV charging corridor.

Second-stage recourse:
    After uncertain residual peak demand is realized at m feeder zones, the operator
    activates integer modular packages. Each package covers a contiguous interval of
    zones on the feeder. The interval-covering matrix W has the consecutive-ones
    property and is totally unimodular.

    Q(x,w) = min q^T y
             s.t. W y >= ceil(w - tau*x)
                  y in Z_+^n.

Because W is TU and the LP relaxation is exact for integer right-hand sides,

    Q(x,w) = sigma_Pi(ceil(w - tau*x)),
    Pi = {pi >= 0 : W^T pi <= q}.

The script compares:
    - SAA / nominal empirical optimization
    - exact positive-radius Wasserstein DRO for several eps values

Requires:
    pip install gurobipy numpy pandas matplotlib
"""

import itertools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB


# ============================================================
# Controls
# ============================================================

SEED = 20260501

R_REPLICATIONS = 30
N_TRAIN = 70
M_TEST = 20000

EPS_LIST = [0.00, 0.02, 0.05, 0.10, 0.20]

X_MIN = 3.0
X_MAX = 9.0
X_GRID_SIZE = 101

NEAR_THRESHOLD_TOL = 0.10

OUTDIR = Path("results_tu_modular_energy_dro")


# ============================================================
# TU interval-covering energy instance
# ============================================================

def build_instance():
    """
    m feeder zones / EV charging clusters on a radial corridor.

    Modular recourse packages cover contiguous intervals of zones.
    This gives an interval matrix W, which is TU by the consecutive-ones property.

    Packages:
        - local packages covering one zone,
        - adjacent two-zone packages,
        - three-zone feeder-section packages,
        - two larger corridor packages.

    The model is not separable SIR because one recourse package can cover
    multiple zones simultaneously.
    """

    m = 6

    intervals = []

    # Local emergency modules: expensive but guarantee complete recourse.
    for i in range(m):
        intervals.append((i, i, 1.35 + 0.07 * i))

    # Two-zone feeder-section modules.
    two_costs = [2.05, 1.95, 1.90, 2.00, 2.10]
    for i in range(m - 1):
        intervals.append((i, i + 1, two_costs[i]))

    # Three-zone modular packages.
    three_costs = [2.75, 2.60, 2.65, 2.85]
    for i in range(m - 2):
        intervals.append((i, i + 2, three_costs[i]))

    # Larger corridor packages.
    intervals.append((0, 3, 3.45))
    intervals.append((2, 5, 3.55))

    n = len(intervals)
    W = np.zeros((m, n), dtype=float)
    q = np.zeros(n, dtype=float)

    for j, (lo, hi, cost) in enumerate(intervals):
        W[lo:hi + 1, j] = 1.0
        q[j] = cost

    tau = np.array([0.90, 1.05, 1.00, 0.95, 1.10, 1.00], dtype=float)

    base_demand = np.array([8.4, 8.0, 8.2, 7.8, 8.5, 8.1], dtype=float)
    common_loading = np.array([0.80, 0.70, 0.75, 0.65, 0.90, 0.72], dtype=float)
    local_sigma = np.array([0.55, 0.50, 0.60, 0.45, 0.62, 0.55], dtype=float)

    first_stage_cost = 0.16

    return {
        "m": m,
        "n": n,
        "W": W,
        "q": q,
        "intervals": intervals,
        "tau": tau,
        "base_demand": base_demand,
        "common_loading": common_loading,
        "local_sigma": local_sigma,
        "first_stage_cost": first_stage_cost,
    }


def sample_true_demand(rng, n_samples, data):
    """
    Synthetic true distribution for peak modular residual demand.

    It contains:
        - common weather/traffic loading,
        - local hub-level noise,
        - rare stress events.
    """

    m = data["m"]

    common = rng.normal(0.0, 1.0, size=(n_samples, 1))
    local = rng.normal(0.0, 1.0, size=(n_samples, m)) * data["local_sigma"]

    stress_flag = rng.binomial(1, 0.08, size=(n_samples, 1))
    stress_size = rng.gamma(shape=2.0, scale=0.45, size=(n_samples, 1))
    stress_profile = np.array([0.7, 0.8, 0.9, 0.7, 1.0, 0.8], dtype=float)

    demand = (
        data["base_demand"]
        + common * data["common_loading"]
        + local
        + stress_flag * stress_size * stress_profile
    )

    return np.maximum(demand, 0.1)


# ============================================================
# Vertex enumeration for Pi = {pi >= 0 : W^T pi <= q}
# ============================================================

def enumerate_dual_vertices(data, tol=1e-8):
    W = data["W"]
    q = data["q"]
    m, n = W.shape

    A = []
    b = []

    # W^T pi <= q
    for j in range(n):
        A.append(W[:, j].copy())
        b.append(q[j])

    # -pi_i <= 0
    for i in range(m):
        row = np.zeros(m)
        row[i] = -1.0
        A.append(row)
        b.append(0.0)

    A = np.vstack(A)
    b = np.array(b)

    vertices = {}
    num_constraints = A.shape[0]

    for active in itertools.combinations(range(num_constraints), m):
        Aeq = A[list(active), :]
        beq = b[list(active)]

        if np.linalg.matrix_rank(Aeq) < m:
            continue

        try:
            pi = np.linalg.solve(Aeq, beq)
        except np.linalg.LinAlgError:
            continue

        if np.all(A @ pi <= b + 1e-7):
            pi[np.abs(pi) < 1e-9] = 0.0
            key = tuple(np.round(pi, 10))
            vertices[key] = pi

    if not vertices:
        raise RuntimeError("No vertices found for Pi. Check W and q.")

    V = np.array(list(vertices.values()), dtype=float)

    zero = np.zeros(m)
    if np.min(np.linalg.norm(V - zero, axis=1)) > 1e-8:
        V = np.vstack([V, zero])

    L_star = float(np.max(V))

    return {
        "Pi_vertices": V,
        "L_star": L_star,
        "num_vertices": V.shape[0],
    }


def all_binary_patterns(m):
    return np.array(list(itertools.product([0.0, 1.0], repeat=m)), dtype=float)


# ============================================================
# Exact recourse and exact DRO evaluation
# ============================================================

def c_delta_for_x(x, samples, data):
    a = samples - data["tau"] * x
    c = np.ceil(a)
    delta = c - a
    delta = np.clip(delta, 0.0, 1.0)
    return c, delta


def recourse_values_from_c(c, vertices):
    V = vertices["Pi_vertices"]
    return np.max(c @ V.T, axis=1)


def nominal_objective(x, samples, data, vertices):
    c, _ = c_delta_for_x(x, samples, data)
    rec = recourse_values_from_c(c, vertices)
    return float(data["first_stage_cost"] * x + np.mean(rec))


def robust_recourse_fixed_x(x, samples, eps, data, vertices, patterns):
    """
    Exact samplewise DRO reformulation.

    For lambda >= L*,

        Psi_s(x,lambda)
        = max_{u in {0,1}^m}
          { sigma_Pi(c^s(x)+u) - lambda delta^s(x)'u }.

    For fixed x, sigma_Pi(c+u) is evaluated from the vertices of Pi.
    Then the outer minimization over lambda is a small LP:

        min eps lambda + average(theta_s)
        s.t. theta_s + lambda delta_s'u >= sigma_Pi(c_s+u)
             for all samples s and binary jump patterns u.
    """

    if eps == 0.0:
        c, _ = c_delta_for_x(x, samples, data)
        rec = recourse_values_from_c(c, vertices)
        return {
            "robust_recourse": float(np.mean(rec)),
            "lambda_value": np.nan,
            "mean_jump_premium": 0.0,
            "mean_nominal_recourse": float(np.mean(rec)),
        }

    c, delta = c_delta_for_x(x, samples, data)
    V = vertices["Pi_vertices"]
    L_star = vertices["L_star"]

    N = samples.shape[0]
    P = patterns.shape[0]

    # sigma_Pi(c_s + u_p) = max_v v'(c_s+u_p)
    base = c @ V.T
    pattern_part = patterns @ V.T
    sigma_sp = np.max(base[:, None, :] + pattern_part[None, :, :], axis=2)

    slopes = delta @ patterns.T

    model = gp.Model("fixed_x_exact_wdro")
    model.Params.OutputFlag = 0
    model.Params.Method = 1

    lam = model.addVar(lb=L_star, name="lambda_value")
    theta = model.addVars(N, lb=0.0, name="theta")

    for s in range(N):
        for p in range(P):
            model.addConstr(theta[s] + float(slopes[s, p]) * lam >= float(sigma_sp[s, p]))

    model.setObjective(
        eps * lam + (1.0 / N) * gp.quicksum(theta[s] for s in range(N)),
        GRB.MINIMIZE,
    )

    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Gurobi did not solve fixed-x DRO LP. x={x}, eps={eps}, status={model.status}"
        )

    theta_val = np.array([theta[s].X for s in range(N)])
    nominal_per_sample = np.max(base, axis=1)

    out = {
        "robust_recourse": float(model.ObjVal),
        "lambda_value": float(lam.X),
        "mean_jump_premium": float(np.mean(theta_val - nominal_per_sample)),
        "mean_nominal_recourse": float(np.mean(nominal_per_sample)),
    }

    model.dispose()
    return out


def solve_grid(train_samples, eps, data, vertices, patterns, x_grid):
    best = None

    for x in x_grid:
        info = robust_recourse_fixed_x(
            x=x,
            samples=train_samples,
            eps=eps,
            data=data,
            vertices=vertices,
            patterns=patterns,
        )

        obj = data["first_stage_cost"] * x + info["robust_recourse"]

        if best is None or obj < best["train_obj"]:
            best = {
                "x": float(x),
                "train_obj": float(obj),
                "lambda_value": info["lambda_value"],
                "mean_jump_premium": info["mean_jump_premium"],
                "train_nominal_recourse": info["mean_nominal_recourse"],
            }

    return best


def evaluate_oos(x, test_samples, data, vertices):
    c, delta = c_delta_for_x(x, test_samples, data)
    rec = recourse_values_from_c(c, vertices)
    total = data["first_stage_cost"] * x + rec

    return {
        "oos_mean": float(np.mean(total)),
        "oos_std": float(np.std(total, ddof=1)),
        "oos_p90": float(np.percentile(total, 90)),
        "oos_p95": float(np.percentile(total, 95)),
        "oos_p99": float(np.percentile(total, 99)),
        "near_threshold": float(np.mean(delta <= NEAR_THRESHOLD_TOL)),
        "mean_recourse_only": float(np.mean(rec)),
    }


# ============================================================
# Replication experiment
# ============================================================

def summarize_results(df):
    rows = []

    order = ["SAA"] + [f"DRO_eps_{eps:g}" for eps in EPS_LIST if eps > 0.0]

    for method in order:
        g = df[df["method"] == method].copy()
        n = len(g)

        def se(col):
            return float(g[col].std(ddof=1) / math.sqrt(n))

        rows.append({
            "method": method,
            "mean_x": float(g["x"].mean()),
            "se_x": se("x"),
            "mean_train_obj": float(g["train_obj"].mean()),
            "mean_oos_cost": float(g["oos_mean"].mean()),
            "se_oos_cost": se("oos_mean"),
            "mean_oos_p90": float(g["oos_p90"].mean()),
            "mean_oos_p95": float(g["oos_p95"].mean()),
            "se_oos_p95": se("oos_p95"),
            "mean_oos_p99": float(g["oos_p99"].mean()),
            "mean_jump_premium": float(g["mean_jump_premium"].mean()),
            "mean_near_threshold": float(g["near_threshold"].mean()),
            "mean_lambda": (
                float(g["lambda_value"].dropna().mean())
                if g["lambda_value"].notna().any()
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def run_experiment():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    data = build_instance()
    vertices = enumerate_dual_vertices(data)
    patterns = all_binary_patterns(data["m"])
    x_grid = np.linspace(X_MIN, X_MAX, X_GRID_SIZE)

    print("TU modular energy Wasserstein DRO experiment")
    print("-------------------------------------------")
    print(f"Replications R:       {R_REPLICATIONS}")
    print(f"Training samples N:   {N_TRAIN}")
    print(f"Test samples M:       {M_TEST}")
    print(f"Zones m:              {data['m']}")
    print(f"Packages n:           {data['n']}")
    print(f"Dual vertices:        {vertices['num_vertices']}")
    print(f"Binary jump patterns: {patterns.shape[0]}")
    print(f"L*(Pi):               {vertices['L_star']:.4f}")
    print(f"x-grid size:          {len(x_grid)}")
    print()

    rows = []
    t0 = time.time()

    for r in range(R_REPLICATIONS):
        train = sample_true_demand(rng, N_TRAIN, data)
        test = sample_true_demand(rng, M_TEST, data)

        for eps in EPS_LIST:
            method = "SAA" if eps == 0.0 else f"DRO_eps_{eps:g}"

            sol = solve_grid(
                train_samples=train,
                eps=eps,
                data=data,
                vertices=vertices,
                patterns=patterns,
                x_grid=x_grid,
            )

            perf = evaluate_oos(
                x=sol["x"],
                test_samples=test,
                data=data,
                vertices=vertices,
            )

            rows.append({
                "replication": r + 1,
                "method": method,
                "eps": eps,
                "x": sol["x"],
                "train_obj": sol["train_obj"],
                "lambda_value": sol["lambda_value"],
                "mean_jump_premium": sol["mean_jump_premium"],
                "train_nominal_recourse": sol["train_nominal_recourse"],
                **perf,
            })

        print(f"Finished replication {r + 1:02d}/{R_REPLICATIONS}")

    results = pd.DataFrame(rows)
    summary = summarize_results(results)

    results.to_csv(OUTDIR / "replication_results.csv", index=False)
    summary.to_csv(OUTDIR / "summary.csv", index=False)

    elapsed = time.time() - t0

    print()
    print("Summary over replications")
    print("--------------------------")
    print(summary.to_string(index=False))
    print()
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print(f"Saved outputs to: {OUTDIR.resolve()}")

    try:
        make_plots(results, summary)
    except Exception as exc:
        print(f"Plotting failed, but CSV results were saved. Plot error: {exc}")

    return results, summary


# ============================================================
# Plots
# ============================================================

def boxplot_compat(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        plt.boxplot(data, labels=labels, showmeans=True)


def make_plots(results, summary):
    labels = summary["method"].tolist()

    plt.figure(figsize=(10, 5))
    data = [results.loc[results["method"] == lab, "oos_mean"].values for lab in labels]
    boxplot_compat(data, labels)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Out-of-sample total cost")
    plt.title("Out-of-sample cost across replications")
    plt.tight_layout()
    plt.savefig(OUTDIR / "boxplot_oos_mean_cost.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        np.arange(len(labels)),
        summary["mean_x"].values,
        yerr=summary["se_x"].values,
        fmt="o-",
        capsize=4,
    )
    plt.xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("Chosen first-stage capacity x")
    plt.title("Capacity selected by SAA and exact DRO")
    plt.tight_layout()
    plt.savefig(OUTDIR / "chosen_capacity_x.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(labels, summary["mean_oos_cost"].values, marker="o", label="Mean")
    plt.plot(labels, summary["mean_oos_p95"].values, marker="o", label="95th percentile")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Out-of-sample total cost")
    plt.title("Mean and tail out-of-sample cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "mean_and_p95_oos_cost.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(labels, summary["mean_jump_premium"].values, marker="o")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean samplewise jump premium")
    plt.title("Jump correction under Wasserstein DRO")
    plt.tight_layout()
    plt.savefig(OUTDIR / "mean_jump_premium.png", dpi=200)
    plt.close()


# ============================================================
# Optional diagnostic objective curve
# ============================================================

def objective_curve_diagnostic():
    rng = np.random.default_rng(SEED + 999)

    data = build_instance()
    vertices = enumerate_dual_vertices(data)
    patterns = all_binary_patterns(data["m"])
    x_grid = np.linspace(X_MIN, X_MAX, X_GRID_SIZE)

    train = sample_true_demand(rng, N_TRAIN, data)

    curves = []

    for eps in [0.0, 0.05, 0.10, 0.20]:
        method = "SAA" if eps == 0.0 else f"DRO_eps_{eps:g}"

        for x in x_grid:
            info = robust_recourse_fixed_x(
                x=x,
                samples=train,
                eps=eps,
                data=data,
                vertices=vertices,
                patterns=patterns,
            )

            obj = data["first_stage_cost"] * x + info["robust_recourse"]

            curves.append({
                "method": method,
                "x": x,
                "objective": obj,
                "jump_premium": info["mean_jump_premium"],
            })

    df = pd.DataFrame(curves)
    df.to_csv(OUTDIR / "objective_curve_diagnostic.csv", index=False)

    plt.figure(figsize=(9, 5))
    for method, g in df.groupby("method", sort=False):
        plt.plot(g["x"], g["objective"], label=method)
    plt.xlabel("First-stage capacity x")
    plt.ylabel("Training objective")
    plt.title("Objective curves: SAA versus exact Wasserstein DRO")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "objective_curves.png", dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_experiment()

    try:
        objective_curve_diagnostic()
    except Exception as exc:
        print(f"Objective-curve diagnostic failed, but main results were saved. Error: {exc}")