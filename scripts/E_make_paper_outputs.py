"""
E_make_paper_outputs.py
=======================

Generate paper tables and figures for the debit-rule paper.

Inputs
------
    cfg.BENCH_PATH      = outputs/results/benchmark_full_results.csv
    cfg.M3_LOG_PATH     = outputs/logs/m3_training_log.csv
    cfg.KKT_PATH        = outputs/results/kkt_equalization_diagnostic.csv
    cfg.M3_WEIGHTS_PATH = data/derived/m3_debit_weights_by_year.parquet

Outputs
-------
Tables:
    T1_main_benchmark.tex/csv
    T2_feature_ladder.tex/csv
    T3_m3_variants.tex/csv
    T4_kkt_summary.tex/csv
    T5_training_log.tex/csv
    T6_pairwise_summary.tex/csv

Figures:
    F1_main_cost_slack.pdf/png
    F2_main_budget_scarcity.pdf/png
    F3_feature_ladder.pdf/png
    F4_m3_variants.pdf/png
    F5_training_convergence.pdf/png
    F6_q_heatmap.pdf/png
    F7_cost_slack_tradeoff.pdf/png
    F8_kkt_equalization.pdf/png

Interpretation discipline
-------------------------
- Main comparison: M1, M2, M3 where M3 = cfg.MAIN_FEATURE_SET and cfg.MAIN_M3_VARIANT.
- Feature ladder: M3_feature_ladder rows.
- Architecture comparison: M3_variant_comparison rows.
- Cost savings: positive means the comparator is cheaper than the baseline named in the column.
- Do not interpret q_mean == 1 as no variation. Use q_std, q_min, q_max, and q_annual_mean_min/max.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipeline_config as cfg  # type: ignore


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

SM = 1e6
FW = 3.35
FWD = 6.80

EVAL_YEARS = list(getattr(cfg, "EVAL_YEARS", getattr(cfg, "TEST_YEARS", [])))
if not EVAL_YEARS:
    raise ValueError("cfg.EVAL_YEARS or cfg.TEST_YEARS must be defined.")

HOLDOUT_YEAR = int(EVAL_YEARS[-1])

MAIN_FEATURE_SET = str(getattr(cfg, "MAIN_FEATURE_SET", "F2"))
MAIN_M3_VARIANT = str(getattr(cfg, "MAIN_M3_VARIANT", "M3_LO"))
M3_VARIANTS = list(getattr(cfg, "M3_VARIANTS", ["M3_plain", "M3_OI", "M3_LO"]))

Q_MIN_CONTRACT = float(getattr(cfg, "M3_Q_MIN", 0.25))
Q_MAX_CONTRACT = float(getattr(cfg, "M3_Q_MAX", 4.0))

TABLES_DIR = Path(getattr(cfg, "TABLES_DIR", REPO_ROOT / "outputs" / "tables"))
FIGURES_DIR = Path(getattr(cfg, "FIGURES_DIR", REPO_ROOT / "outputs" / "figures"))

COLORS = {
    "M1": "#6B6B6B",
    "M2": "#1B4F8A",
    "M3": "#B63A2C",
}
MARKERS = {
    "M1": "s",
    "M2": "o",
    "M3": "^",
}
LABELS = {
    "M1": "M1 touched-time",
    "M2": "M2 uniform-depth",
    "M3": f"M3 {MAIN_FEATURE_SET}, {MAIN_M3_VARIANT}",
}

FS_ORDER = ["F0", "F1", "F2", "F3"]
FS_COLORS = {
    "F0": "#9ECAE1",
    "F1": "#6BAED6",
    "F2": "#1B4F8A",
    "F3": "#08306B",
}

VAR_COLORS = {
    "M3_plain": "#9ECAE1",
    "M3_OI": "#6BAED6",
    "M3_LO": "#1B4F8A",
}
VAR_MARKERS = {
    "M3_plain": "s",
    "M3_OI": "o",
    "M3_LO": "^",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
    }
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def _ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bench = _read_csv_required(Path(cfg.BENCH_PATH), "benchmark results")
    log = _read_csv_required(Path(cfg.M3_LOG_PATH), "M3 training log")

    kkt_path = Path(getattr(cfg, "KKT_PATH", REPO_ROOT / "outputs" / "results" / "kkt_equalization_diagnostic.csv"))
    kkt = pd.read_csv(kkt_path) if kkt_path.exists() else pd.DataFrame()

    weights_path = Path(getattr(cfg, "M3_WEIGHTS_PATH", REPO_ROOT / "data" / "derived" / "m3_debit_weights_by_year.parquet"))
    weights = pd.read_parquet(weights_path) if weights_path.exists() else pd.DataFrame()

    if "year" in bench.columns:
        bench["year"] = pd.to_numeric(bench["year"], errors="coerce").astype("Int64")
    if "year" in log.columns:
        log["year"] = pd.to_numeric(log["year"], errors="coerce").astype("Int64")
    if "year" in kkt.columns:
        kkt["year"] = pd.to_numeric(kkt["year"], errors="coerce").astype("Int64")
    if not weights.empty and "timestamp_utc" in weights.columns:
        weights["timestamp_utc"] = pd.to_datetime(weights["timestamp_utc"], utc=True)

    numeric_cols = [
        "cost_total",
        "cost_curtail",
        "cost_slack",
        "slack_total",
        "budget_util",
        "physical_proxy_exposure",
        "debit_proxy_exposure",
        "alignment_baseline",
        "exposure_gini",
        "exposure_hhi",
        "debit_gini",
        "debit_hhi",
        "mu_positive_share",
        "mu_positive_count",
        "mu_mean",
        "mu_max",
        "scarcity_active_share",
        "q_mean",
        "q_std",
        "q_min",
        "q_max",
        "q_annual_mean_min",
        "q_annual_mean_max",
        "mip_gap",
        "mip_bound",
        "solve_time",
    ]
    for col in numeric_cols:
        if col in bench.columns:
            bench[col] = pd.to_numeric(bench[col], errors="coerce")

    if "cost_total" in bench.columns:
        bench["cost_M"] = bench["cost_total"] / SM
    if "cost_slack" in bench.columns:
        bench["slack_cost_M"] = bench["cost_slack"] / SM

    return bench, log, kkt, weights


def _latex_escape(text: object) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def _safe_num(value: object, digits: int = 2, dash: str = "--") -> str:
    try:
        v = float(value)
    except Exception:
        return dash
    if not np.isfinite(v):
        return dash
    return f"{v:.{digits}f}"


def _safe_int(value: object, dash: str = "--") -> str:
    try:
        v = float(value)
    except Exception:
        return dash
    if not np.isfinite(v):
        return dash
    return str(int(round(v)))


def _write_tex_table(
    filename: str,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[str]],
    column_format: Optional[str] = None,
) -> None:
    if column_format is None:
        column_format = "l" + "r" * (len(columns) - 1)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    (TABLES_DIR / filename).write_text("\n".join(lines), encoding="utf-8")


def _save_fig(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {stem}")


def _main_rows(bench: pd.DataFrame) -> pd.DataFrame:
    out = bench[bench["rule"].isin(["M1", "M2", "M3"])].copy()
    out["_order"] = out["rule"].map({"M1": 0, "M2": 1, "M3": 2})
    return out.sort_values(["year", "_order"]).drop(columns="_order")


def _feature_rows(bench: pd.DataFrame) -> pd.DataFrame:
    out = bench[
        (bench["rule"] == "M3_feature_ladder")
        & (bench["variant"].astype(str) == MAIN_M3_VARIANT)
    ].copy()
    out["_order"] = out["feature_set"].map({fs: i for i, fs in enumerate(FS_ORDER)})
    return out.sort_values(["year", "_order"]).drop(columns="_order")


def _variant_rows(bench: pd.DataFrame) -> pd.DataFrame:
    out = bench[
        (bench["rule"] == "M3_variant_comparison")
        & (bench["feature_set"].astype(str) == MAIN_FEATURE_SET)
    ].copy()
    out["_order"] = out["variant"].map({v: i for i, v in enumerate(M3_VARIANTS)})
    return out.sort_values(["year", "_order"]).drop(columns="_order")


def _year_values(df: pd.DataFrame, key_col: str, value_col: str, key_order: list[str]) -> pd.DataFrame:
    piv = df.pivot_table(index=key_col, columns="year", values=value_col, aggfunc="first")
    piv = piv.reindex(key_order)
    years = [y for y in EVAL_YEARS if y in piv.columns]
    piv = piv[years]
    piv["Mean"] = piv[years].mean(axis=1)
    return piv


def _shade_holdout(ax: plt.Axes, years: list[int]) -> None:
    if HOLDOUT_YEAR not in years:
        return
    idx = years.index(HOLDOUT_YEAR)
    ax.axvspan(idx - 0.5, idx + 0.5, color="#B63A2C", alpha=0.07, zorder=0)
    ax.axvline(idx - 0.5, color="#B63A2C", lw=0.7, ls="--", alpha=0.45)


def _add_grid(ax: plt.Axes, axis: str = "y") -> None:
    if axis in {"y", "both"}:
        ax.yaxis.grid(True, alpha=0.30, lw=0.5)
    if axis in {"x", "both"}:
        ax.xaxis.grid(True, alpha=0.30, lw=0.5)
    ax.set_axisbelow(True)


def _first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


# =============================================================================
# TABLES
# =============================================================================

def table1_main_benchmark(bench: pd.DataFrame) -> None:
    main = _main_rows(bench)
    piv_cost = _year_values(main, "rule", "cost_M", ["M1", "M2", "M3"])
    years = [y for y in EVAL_YEARS if y in piv_cost.columns]

    m2_mean = float(piv_cost.loc["M2", "Mean"])
    piv_cost["Delta_vs_M2_pct"] = (piv_cost["Mean"] - m2_mean) / max(abs(m2_mean), 1e-12) * 100.0
    piv_cost.to_csv(TABLES_DIR / "T1_main_benchmark.csv")

    rows: list[list[str]] = []
    rule_label = {
        "M1": r"$M1$",
        "M2": r"$M2$",
        "M3": rf"$M3$ ({_latex_escape(MAIN_FEATURE_SET)}, {_latex_escape(MAIN_M3_VARIANT)})",
    }

    for rule in ["M1", "M2", "M3"]:
        row = [rule_label[rule]]
        row += [_safe_num(piv_cost.loc[rule, y], 2) for y in years]
        row += [_safe_num(piv_cost.loc[rule, "Mean"], 2)]
        row += ["--" if rule == "M2" else f"{float(piv_cost.loc[rule, 'Delta_vs_M2_pct']):+.1f}"]
        rows.append(row)

    _write_tex_table(
        "T1_main_benchmark.tex",
        rf"Main benchmark: annual operating cost in million EUR. "
        rf"$M3$ uses feature set {_latex_escape(MAIN_FEATURE_SET)} and architecture {_latex_escape(MAIN_M3_VARIANT)}.",
        "tab:main_benchmark",
        ["Rule", *[str(y) for y in years], "Mean", r"$\Delta$ vs. $M2$ (\%)"],
        rows,
    )
    print("  ✓ T1_main_benchmark")


def table2_feature_ladder(bench: pd.DataFrame) -> None:
    ladder = _feature_rows(bench)
    if ladder.empty:
        print("  - T2_feature_ladder skipped: no M3_feature_ladder rows")
        return

    hold = ladder[ladder["year"] == HOLDOUT_YEAR].copy()
    hold = hold.set_index("feature_set").reindex(FS_ORDER)

    if hold["cost_M"].isna().all():
        print("  - T2_feature_ladder skipped: no holdout ladder data")
        return

    base_fs = MAIN_FEATURE_SET if MAIN_FEATURE_SET in hold.index else "F2"
    base = float(hold.loc[base_fs, "cost_M"])
    hold["Delta_vs_main_pct"] = (hold["cost_M"] - base) / max(abs(base), 1e-12) * 100.0
    hold.to_csv(TABLES_DIR / "T2_feature_ladder.csv")

    rows: list[list[str]] = []
    for fs in FS_ORDER:
        if fs not in hold.index or pd.isna(hold.loc[fs, "cost_M"]):
            continue
        rows.append(
            [
                fs,
                _safe_num(hold.loc[fs, "cost_M"], 2),
                _safe_num(hold.loc[fs, "slack_total"], 1),
                _safe_num(hold.loc[fs, "budget_util"], 3),
                _safe_num(hold.loc[fs, "mu_positive_count"], 0),
                _safe_num(hold.loc[fs, "q_std"], 3),
                "--" if fs == base_fs else f"{float(hold.loc[fs, 'Delta_vs_main_pct']):+.1f}",
            ]
        )

    _write_tex_table(
        "T2_feature_ladder.tex",
        rf"Public feature-ladder comparison for holdout year {HOLDOUT_YEAR}; "
        rf"architecture {_latex_escape(MAIN_M3_VARIANT)}.",
        "tab:feature_ladder",
        ["Feature", "Cost (M)", "Slack", "Util.", r"$\mu>0$", r"$\sigma(q)$", r"$\Delta$ vs. main (\%)"],
        rows,
    )
    print("  ✓ T2_feature_ladder")


def table3_m3_variants(bench: pd.DataFrame) -> None:
    variants = _variant_rows(bench)
    if variants.empty:
        print("  - T3_m3_variants skipped: no M3_variant_comparison rows")
        return

    piv = _year_values(variants, "variant", "cost_M", M3_VARIANTS)
    years = [y for y in EVAL_YEARS if y in piv.columns]
    base_variant = MAIN_M3_VARIANT if MAIN_M3_VARIANT in piv.index else M3_VARIANTS[-1]
    base = float(piv.loc[base_variant, "Mean"])
    piv["Delta_vs_main_pct"] = (piv["Mean"] - base) / max(abs(base), 1e-12) * 100.0
    piv.to_csv(TABLES_DIR / "T3_m3_variants.csv")

    rows: list[list[str]] = []
    for variant in M3_VARIANTS:
        if variant not in piv.index:
            continue
        row = [_latex_escape(variant)]
        row += [_safe_num(piv.loc[variant, y], 2) for y in years]
        row += [_safe_num(piv.loc[variant, "Mean"], 2)]
        row += ["--" if variant == base_variant else f"{float(piv.loc[variant, 'Delta_vs_main_pct']):+.1f}"]
        rows.append(row)

    _write_tex_table(
        "T3_m3_variants.tex",
        rf"$M3$ training-architecture comparison at feature set {_latex_escape(MAIN_FEATURE_SET)}.",
        "tab:m3_variants",
        ["Variant", *[str(y) for y in years], "Mean", r"$\Delta$ vs. main (\%)"],
        rows,
    )
    print("  ✓ T3_m3_variants")


def table4_kkt(kkt: pd.DataFrame) -> None:
    if kkt.empty:
        print("  - T4_kkt skipped: KKT file missing or empty")
        return

    local = kkt.copy()
    for col in ["equalization_error", "n_interior", "mu", "R_mean", "R_std"]:
        if col in local.columns:
            local[col] = pd.to_numeric(local[col], errors="coerce")

    local = local[local["equalization_error"].notna()].copy()
    if local.empty:
        print("  - T4_kkt skipped: no finite equalization errors")
        return

    summary = (
        local.groupby(["rule", "year"], as_index=False)
        .agg(
            mean_error=("equalization_error", "mean"),
            max_error=("equalization_error", "max"),
            mean_interior=("n_interior", "mean"),
            median_mu=("mu", "median"),
        )
        .sort_values(["rule", "year"])
    )
    summary.to_csv(TABLES_DIR / "T4_kkt_summary.csv", index=False)

    rows: list[list[str]] = []
    for _, r in summary.iterrows():
        rows.append(
            [
                _latex_escape(r["rule"]),
                _safe_int(r["year"]),
                _safe_num(r["mean_error"], 3),
                _safe_num(r["max_error"], 3),
                _safe_num(r["mean_interior"], 1),
                _safe_num(r["median_mu"], 1),
            ]
        )

    _write_tex_table(
        "T4_kkt_summary.tex",
        r"KKT equalization diagnostics for weighted-depth rules. "
        r"The diagnostic compares $(P/q)(\lambda-c)$ with the annual-budget multiplier $\mu$ at interior intervals.",
        "tab:kkt_summary",
        ["Rule", "Year", "Mean err.", "Max err.", "Mean int.", r"Median $\mu$"],
        rows,
    )
    print("  ✓ T4_kkt_summary")


def table5_training_log(log: pd.DataFrame) -> None:
    if log.empty:
        print("  - T5_training_log skipped: empty log")
        return

    local = log.copy()
    for col in ["iter", "delta_theta_rel", "train_cost", "alpha_raw_mean", "alpha_raw_max", "alpha_positive_share"]:
        if col in local.columns:
            local[col] = pd.to_numeric(local[col], errors="coerce")

    local.to_csv(TABLES_DIR / "T5_training_log.csv", index=False)

    rows: list[list[str]] = []
    keep_cols = ["feature_set", "variant", "iter", "delta_theta_rel", "train_cost", "alpha_positive_share", "alpha_raw_max"]
    shown = local.sort_values(["feature_set", "variant", "iter"]).copy()

    for _, r in shown.iterrows():
        rows.append(
            [
                _latex_escape(r.get("feature_set", "")),
                _latex_escape(r.get("variant", "")),
                _safe_int(r.get("iter", np.nan)),
                _safe_num(r.get("delta_theta_rel", np.nan), 4),
                _safe_num(r.get("train_cost", np.nan), 2),
                _safe_num(r.get("alpha_positive_share", np.nan), 3),
                _safe_num(r.get("alpha_raw_max", np.nan), 2),
            ]
        )

    _write_tex_table(
        "T5_training_log.tex",
        r"Training diagnostics for the learned weighted-depth rules.",
        "tab:m3_training_log",
        ["Feat.", "Variant", "Iter.", r"$\Delta\theta$", "Train cost", r"$\Pr(\alpha>0)$", r"$\max\alpha$"],
        rows,
    )
    print("  ✓ T5_training_log")


def table6_pairwise(bench: pd.DataFrame) -> None:
    main = _main_rows(bench)
    if main.empty:
        print("  - T6_pairwise skipped: no main rows")
        return

    piv = main.pivot_table(index="year", columns="rule", values=["cost_M", "slack_total"], aggfunc="first")
    rows_out = []

    for y in EVAL_YEARS:
        if y not in piv.index:
            continue

        def val(metric: str, rule: str) -> float:
            try:
                return float(piv.loc[y, (metric, rule)])
            except Exception:
                return np.nan

        c1, c2, c3 = val("cost_M", "M1"), val("cost_M", "M2"), val("cost_M", "M3")
        s1, s2, s3 = val("slack_total", "M1"), val("slack_total", "M2"), val("slack_total", "M3")

        rows_out.append(
            {
                "year": y,
                "M2_minus_M3_cost_saving_M": c2 - c3,
                "M2_minus_M1_cost_saving_M": c2 - c1,
                "M1_minus_M3_cost_saving_M": c1 - c3,
                "M2_minus_M3_slack_reduction": s2 - s3,
                "M2_minus_M1_slack_reduction": s2 - s1,
                "M1_minus_M3_slack_reduction": s1 - s3,
            }
        )

    out = pd.DataFrame(rows_out)
    out.to_csv(TABLES_DIR / "T6_pairwise_summary.csv", index=False)

    rows: list[list[str]] = []
    for _, r in out.iterrows():
        rows.append(
            [
                _safe_int(r["year"]),
                _safe_num(r["M2_minus_M3_cost_saving_M"], 2),
                _safe_num(r["M2_minus_M1_cost_saving_M"], 2),
                _safe_num(r["M1_minus_M3_cost_saving_M"], 2),
                _safe_num(r["M2_minus_M3_slack_reduction"], 1),
            ]
        )

    _write_tex_table(
        "T6_pairwise_summary.tex",
        r"Paired benchmark differences. Positive cost-saving values mean the second named rule is cheaper than the first baseline rule.",
        "tab:pairwise_summary",
        ["Year", r"$M2-M3$ cost", r"$M2-M1$ cost", r"$M1-M3$ cost", r"$M2-M3$ slack"],
        rows,
    )
    print("  ✓ T6_pairwise_summary")


# =============================================================================
# FIGURES
# =============================================================================

def figure1_main_cost_slack(bench: pd.DataFrame) -> None:
    main = _main_rows(bench)
    years = [int(y) for y in EVAL_YEARS if y in set(main["year"].dropna().astype(int))]
    if not years:
        print("  - F1_main_cost_slack skipped: no main rows")
        return

    x = np.arange(len(years))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(FWD, 3.1))
    ax_cost, ax_slack = axes

    for j, rule in enumerate(["M1", "M2", "M3"]):
        sub = main[main["rule"] == rule].set_index("year")
        costs = [float(sub.loc[y, "cost_M"]) if y in sub.index else np.nan for y in years]
        slacks = [float(sub.loc[y, "slack_total"]) if y in sub.index else np.nan for y in years]
        offsets = x + (j - 1) * width
        ax_cost.bar(offsets, costs, width, color=COLORS[rule], alpha=0.88, label=LABELS[rule])
        ax_slack.bar(offsets, slacks, width, color=COLORS[rule], alpha=0.88, label=LABELS[rule])

    for ax in axes:
        _shade_holdout(ax, years)
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years])
        ax.set_xlabel("Year")
        _add_grid(ax, "y")

    ax_cost.set_ylabel("Annual cost (M EUR)")
    ax_cost.set_title("(a) Operating cost", loc="left")
    ax_slack.set_ylabel("Emergency slack")
    ax_slack.set_title("(b) Emergency slack", loc="left")
    ax_cost.legend(frameon=False, fontsize=7, loc="upper left")

    _save_fig(fig, "F1_main_cost_slack")


def figure2_main_budget_scarcity(bench: pd.DataFrame) -> None:
    main = _main_rows(bench)
    years = [int(y) for y in EVAL_YEARS if y in set(main["year"].dropna().astype(int))]
    if not years:
        print("  - F2_main_budget_scarcity skipped: no main rows")
        return

    x = np.arange(len(years))
    fig, axes = plt.subplots(1, 2, figsize=(FWD, 3.1))
    ax_util, ax_mu = axes

    for rule in ["M1", "M2", "M3"]:
        sub = main[main["rule"] == rule].set_index("year")
        util = [float(sub.loc[y, "budget_util"]) if y in sub.index else np.nan for y in years]
        ax_util.plot(x, util, color=COLORS[rule], marker=MARKERS[rule], label=LABELS[rule])

    for rule in ["M2", "M3"]:
        sub = main[main["rule"] == rule].set_index("year")
        mu_count_col = "mu_positive_count" if "mu_positive_count" in sub.columns else None
        if mu_count_col is not None:
            mu_vals = [float(sub.loc[y, mu_count_col]) if y in sub.index else np.nan for y in years]
            ylabel = "Number of scarce budgets"
        else:
            mu_vals = [float(sub.loc[y, "mu_positive_share"]) if y in sub.index else np.nan for y in years]
            ylabel = "Share of scarce budgets"
        ax_mu.plot(x, mu_vals, color=COLORS[rule], marker=MARKERS[rule], label=LABELS[rule])

    for ax in axes:
        _shade_holdout(ax, years)
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years])
        ax.set_xlabel("Year")
        _add_grid(ax, "y")

    ax_util.axhline(1.0, color="black", lw=0.6, ls=":", alpha=0.6)
    ax_util.set_ylabel("Budget utilisation")
    ax_util.set_ylim(0, max(1.15, float(main["budget_util"].max()) * 1.1))
    ax_util.set_title("(a) Allowance use", loc="left")

    ax_mu.set_ylabel(ylabel)
    ax_mu.set_title("(b) Annual scarcity", loc="left")
    ax_util.legend(frameon=False, fontsize=7, loc="lower left")

    _save_fig(fig, "F2_main_budget_scarcity")


def figure3_feature_ladder(bench: pd.DataFrame) -> None:
    ladder = _feature_rows(bench)
    if ladder.empty:
        print("  - F3_feature_ladder skipped: no ladder rows")
        return

    years = [int(y) for y in EVAL_YEARS if y in set(ladder["year"].dropna().astype(int))]
    if not years:
        print("  - F3_feature_ladder skipped: no years")
        return

    fig, axes = plt.subplots(1, 2, figsize=(FWD, 3.1))
    ax_cost, ax_delta = axes
    x = np.arange(len(years))

    for fs in FS_ORDER:
        sub = ladder[ladder["feature_set"] == fs].set_index("year")
        vals = [float(sub.loc[y, "cost_M"]) if y in sub.index else np.nan for y in years]
        ax_cost.plot(x, vals, color=FS_COLORS[fs], marker="o", label=fs)

    hold = ladder[ladder["year"] == HOLDOUT_YEAR].set_index("feature_set").reindex(FS_ORDER)
    if not hold.empty and MAIN_FEATURE_SET in hold.index:
        base = float(hold.loc[MAIN_FEATURE_SET, "cost_M"])
        deltas = []
        labels = []
        cols = []
        for fs in FS_ORDER:
            if fs in hold.index and np.isfinite(hold.loc[fs, "cost_M"]):
                deltas.append((float(hold.loc[fs, "cost_M"]) - base) / max(abs(base), 1e-12) * 100.0)
                labels.append(fs)
                cols.append(FS_COLORS[fs])
        y_pos = np.arange(len(labels))
        ax_delta.barh(y_pos, deltas, color=cols, alpha=0.9)
        ax_delta.axvline(0, color="black", lw=0.7)
        ax_delta.set_yticks(y_pos)
        ax_delta.set_yticklabels(labels)
        ax_delta.set_xlabel(r"$\Delta$ cost vs. main feature set (\%)")
        ax_delta.set_title(f"(b) Holdout {HOLDOUT_YEAR}", loc="left")
        _add_grid(ax_delta, "x")

    _shade_holdout(ax_cost, years)
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels([str(y) for y in years])
    ax_cost.set_xlabel("Year")
    ax_cost.set_ylabel("Annual cost (M EUR)")
    ax_cost.set_title(f"(a) Feature ladder, {MAIN_M3_VARIANT}", loc="left")
    ax_cost.legend(frameon=False, fontsize=7)
    _add_grid(ax_cost, "y")

    _save_fig(fig, "F3_feature_ladder")


def figure4_m3_variants(bench: pd.DataFrame) -> None:
    variants = _variant_rows(bench)
    if variants.empty:
        print("  - F4_m3_variants skipped: no variant rows")
        return

    years = [int(y) for y in EVAL_YEARS if y in set(variants["year"].dropna().astype(int))]
    if not years:
        print("  - F4_m3_variants skipped: no years")
        return

    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(FWD * 0.72, 3.1))

    for variant in M3_VARIANTS:
        sub = variants[variants["variant"] == variant].set_index("year")
        vals = [float(sub.loc[y, "cost_M"]) if y in sub.index else np.nan for y in years]
        ax.plot(
            x,
            vals,
            color=VAR_COLORS.get(variant, "#777777"),
            marker=VAR_MARKERS.get(variant, "o"),
            label=variant,
        )

    _shade_holdout(ax, years)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual cost (M EUR)")
    ax.set_title(f"M3 training architectures at {MAIN_FEATURE_SET}", loc="left")
    ax.legend(frameon=False, fontsize=8)
    _add_grid(ax, "y")

    _save_fig(fig, "F4_m3_variants")


def figure5_training_convergence(log: pd.DataFrame) -> None:
    if log.empty:
        print("  - F5_training_convergence skipped: empty log")
        return

    local = log.copy()
    for col in ["iter", "delta_theta_rel", "train_cost", "alpha_raw_max", "alpha_positive_share"]:
        if col in local.columns:
            local[col] = pd.to_numeric(local[col], errors="coerce")

    lo = local[local["variant"].astype(str) == "M3_LO"].copy()
    if lo.empty:
        lo = local.copy()

    if "feature_set" not in lo.columns or "iter" not in lo.columns:
        print("  - F5_training_convergence skipped: missing feature_set/iter")
        return

    fig, axes = plt.subplots(1, 2, figsize=(FWD, 3.1))
    ax_delta, ax_cost = axes

    feature_sets = [fs for fs in FS_ORDER if fs in set(lo["feature_set"].dropna().astype(str))]
    if not feature_sets:
        feature_sets = sorted(lo["feature_set"].dropna().astype(str).unique())

    for fs in feature_sets:
        sub = lo[lo["feature_set"].astype(str) == fs].sort_values("iter")
        col = FS_COLORS.get(fs, "#777777")

        if "delta_theta_rel" in sub.columns:
            ax_delta.plot(sub["iter"], sub["delta_theta_rel"], marker="o", color=col, label=fs)

        if "train_cost" in sub.columns:
            ax_cost.plot(sub["iter"], sub["train_cost"] / SM, marker="s", color=col, label=fs)

    ax_delta.set_yscale("log")
    ax_delta.set_xlabel("Iteration")
    ax_delta.set_ylabel(r"Relative parameter change")
    ax_delta.set_title("(a) Learn--optimize movement", loc="left")
    ax_delta.legend(frameon=False, fontsize=7)

    ax_cost.set_xlabel("Iteration")
    ax_cost.set_ylabel("Training cost (M EUR)")
    ax_cost.set_title("(b) Training objective value", loc="left")

    for ax in axes:
        _add_grid(ax, "y")

    _save_fig(fig, "F5_training_convergence")


def figure6_q_heatmap(weights: pd.DataFrame) -> None:
    if weights.empty:
        print("  - F6_q_heatmap skipped: M3 weights missing")
        return

    sub = weights[
        (weights["year"] == HOLDOUT_YEAR)
        & (weights["feature_set"].astype(str) == MAIN_FEATURE_SET)
        & (weights["variant"].astype(str) == MAIN_M3_VARIANT)
    ].copy()

    if sub.empty:
        print("  - F6_q_heatmap skipped: no holdout weights for main rule")
        return

    if "debit_weight" not in sub.columns:
        print("  - F6_q_heatmap skipped: debit_weight missing")
        return

    sub["hour"] = sub["timestamp_utc"].dt.hour
    sub["bucket"] = (sub["hour"] // 3) * 3
    sub["bucket_label"] = sub["bucket"].map(lambda h: f"{int(h):02d}h")

    buckets = [f"{h:02d}h" for h in range(0, 24, 3)]

    pivot = (
        sub.groupby(["province", "bucket_label"], observed=True)["debit_weight"]
        .mean()
        .unstack()
        .reindex(columns=buckets)
    )

    if hasattr(cfg, "NL_PROVINCES"):
        order = [p for p in cfg.NL_PROVINCES if p in pivot.index]
        pivot = pivot.reindex(order)

    pivot = pivot.dropna(how="all")
    if pivot.empty:
        print("  - F6_q_heatmap skipped: empty pivot")
        return

    pivot = pivot.fillna(1.0)

    from matplotlib.colors import TwoSlopeNorm

    vmin = min(float(np.nanmin(pivot.values)), Q_MIN_CONTRACT)
    vmax = max(float(np.nanmax(pivot.values)), Q_MAX_CONTRACT)
    if vmin >= 1.0:
        vmin = Q_MIN_CONTRACT
    if vmax <= 1.0:
        vmax = Q_MAX_CONTRACT
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(FWD * 0.82, max(2.8, 0.32 * len(pivot))))

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=norm,
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7.5)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(p)[:12] for p in pivot.index], fontsize=7.3)

    ax.set_xlabel("Hour bucket")

    variant_label = _latex_escape(MAIN_M3_VARIANT).replace("\\_", "_")
    ax.set_title(
        rf"Learned debit weights $\hat q_{{n,t}}$ — {HOLDOUT_YEAR}, "
        rf"{MAIN_FEATURE_SET}, {variant_label}",
        loc="left",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(r"$\hat q_{n,t}$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    _save_fig(fig, "F6_q_heatmap")


def figure7_cost_slack_tradeoff(bench: pd.DataFrame) -> None:
    main = _main_rows(bench)
    if main.empty:
        print("  - F7_cost_slack_tradeoff skipped: no main rows")
        return

    fig, ax = plt.subplots(figsize=(FW * 1.25, 3.2))

    for rule in ["M1", "M2", "M3"]:
        sub = main[main["rule"] == rule].copy()
        if sub.empty:
            continue
        ax.scatter(
            sub["slack_total"],
            sub["cost_M"],
            s=52,
            color=COLORS[rule],
            marker=MARKERS[rule],
            label=LABELS[rule],
            edgecolors="white",
            linewidths=0.4,
            alpha=0.90,
            zorder=3,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                str(int(r["year"])),
                (r["slack_total"], r["cost_M"]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6.5,
            )

    ax.set_xlabel("Emergency slack")
    ax.set_ylabel("Annual cost (M EUR)")
    ax.set_title("Cost--slack tradeoff across benchmark years", loc="left")
    ax.legend(frameon=False, fontsize=7)
    _add_grid(ax, "both")

    _save_fig(fig, "F7_cost_slack_tradeoff")


def figure8_kkt_equalization(kkt: pd.DataFrame) -> None:
    if kkt.empty:
        print("  - F8_kkt_equalization skipped: KKT file missing or empty")
        return

    local = kkt.copy()
    for col in ["mu", "R_mean", "equalization_error", "n_interior"]:
        if col in local.columns:
            local[col] = pd.to_numeric(local[col], errors="coerce")

    sub = local[
        (local["year"] == HOLDOUT_YEAR)
        & local["mu"].notna()
        & local["R_mean"].notna()
    ].copy()

    if sub.empty:
        print(f"  - F8_kkt_equalization skipped: no holdout KKT rows for {HOLDOUT_YEAR}")
        return

    fig, ax = plt.subplots(figsize=(FW * 1.25, 3.2))

    max_val = max(float(sub["mu"].max()), float(sub["R_mean"].max()), 1.0)
    upper = max_val * 1.08
    xs = np.linspace(0.0, upper, 200)

    ax.plot(xs, xs, color="black", lw=0.8, ls="--", alpha=0.75, label=r"$R(q)=\mu$")
    ax.fill_between(xs, 0.92 * xs, 1.08 * xs, color="gray", alpha=0.10, label=r"$\pm 8\%$ band")

    rules = sorted(sub["rule"].dropna().astype(str).unique())
    for rule in rules:
        rsub = sub[sub["rule"].astype(str) == rule]
        col = COLORS.get(rule, "#777777")
        marker = MARKERS.get(rule, "o")
        sizes = 28 + 2.0 * rsub["n_interior"].fillna(0).clip(0, 40)
        ax.scatter(
            rsub["mu"],
            rsub["R_mean"],
            s=sizes,
            color=col,
            marker=marker,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.4,
            label=rule,
            zorder=3,
        )

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Annual-budget multiplier $\mu_n$")
    ax.set_ylabel(r"Interior statistic $R_n(q)$")
    ax.set_title(f"KKT equalization diagnostic — holdout {HOLDOUT_YEAR}", loc="left")
    ax.legend(frameon=False, fontsize=7)
    _add_grid(ax, "both")

    _save_fig(fig, "F8_kkt_equalization")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    _ensure_dirs()
    bench, log, kkt, weights = _load_inputs()

    print("=" * 72)
    print("E: MAKE PAPER OUTPUTS")
    print("=" * 72)
    print(f"Benchmark : {cfg.BENCH_PATH}")
    print(f"Log       : {cfg.M3_LOG_PATH}")
    print(f"KKT       : {getattr(cfg, 'KKT_PATH', 'not configured')}")
    print(f"Weights   : {getattr(cfg, 'M3_WEIGHTS_PATH', 'not configured')}")
    print(f"Eval years: {EVAL_YEARS}")
    print(f"Holdout   : {HOLDOUT_YEAR}")
    print(f"Main M3   : feature_set={MAIN_FEATURE_SET}, variant={MAIN_M3_VARIANT}")

    print("\nTables")
    print("-" * 72)
    table1_main_benchmark(bench)
    table2_feature_ladder(bench)
    table3_m3_variants(bench)
    table4_kkt(kkt)
    table5_training_log(log)
    table6_pairwise(bench)

    print("\nFigures")
    print("-" * 72)
    figure1_main_cost_slack(bench)
    figure2_main_budget_scarcity(bench)
    figure3_feature_ladder(bench)
    figure4_m3_variants(bench)
    figure5_training_convergence(log)
    figure6_q_heatmap(weights)
    figure7_cost_slack_tradeoff(bench)
    figure8_kkt_equalization(kkt)

    print("\n" + "=" * 72)
    print("PAPER OUTPUTS COMPLETE")
    print("=" * 72)
    print(f"✓ Tables : {TABLES_DIR}")
    print(f"✓ Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()