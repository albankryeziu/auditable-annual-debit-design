#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E_make_advanced_figures.py
==========================

IEEE-TSG-ready figure and table generator for the annual TDTR debit-rule paper.

This script is aligned with the signed-off A -> B1/B -> C -> D pipeline.

Pipeline inputs
---------------
D_run_full_benchmark.py writes:
    outputs/results/benchmark_full_results.csv
    outputs/results/benchmark_pairwise_differences.csv
    outputs/results/kkt_equalization_diagnostic.csv
    outputs/results/benchmark_manifest.json
    outputs/results/benchmark_q_inventory.csv
    outputs/results/benchmark_q_audit.csv

C_train_m3.py writes:
    data/derived/m3_debit_weights_by_year.parquet
    data/derived/m3_debit_weights_by_year.csv
    outputs/results/m3_training_q_audit.csv
    outputs/logs/m3_training_log.csv
    data/derived/m3_training_manifest.json

Main-paper figures
------------------
F1  Main M1/M2/M3 operating tradeoff.
F2  Direct M2-vs-main-M3 pairwise improvement.
F3  Auditable information ladder F0--F3.
F4  KKT/scarcity mechanism.
F5  q audit and learned debit-rate structure.

Appendix figures
----------------
A1  Full method heatmap.
A2  M3 architecture comparison.
A3  M3 response curves.
A4  Training diagnostics.
A5  Operating frontier.

Main-paper tables
-----------------
Table I    Experimental setup and metric definitions.
Table II   Main annual benchmark.
Table III  M2-vs-M3 pairwise comparison.
Table IV   Sensitivity summary: feature ladder and architecture.

Appendix tables
---------------
Full benchmark, pairwise, KKT, q-audit, and solver-quality diagnostics.

Interpretation discipline
-------------------------
Relief-served share is a constructed-testbed metric:

    1 - total emergency slack / total constructed relief requirement.

It is not a measurement of actual Dutch redispatch performance.

The KKT/scarcity diagnostic used for learning and interpretation is:

    alpha_{n,t} = mu_n * x_{n,t}/P_n,

not

    mu_n * q_{n,t} * x_{n,t}/P_n.

M3 q is an accounting debit rate, not an electricity price, redispatch price,
or market price. Bounded and normalized q reallocates annual allowance
consumption; it does not create extra entitlement.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# REPO / OPTIONAL CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else Path.cwd()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import pipeline_config as cfg  # type: ignore
except Exception:
    cfg = None


def cfg_get(name: str, default: Any) -> Any:
    if cfg is not None and hasattr(cfg, name):
        value = getattr(cfg, name)
        if value is not None and value != "":
            return value
    return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def env_int_list(name: str, default: Iterable[int]) -> List[int]:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return [int(x) for x in default]
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def env_str_list(name: str, default: Iterable[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return [str(x) for x in default]
    return [x.strip() for x in str(raw).split(",") if x.strip()]


# =============================================================================
# PATHS AND GLOBAL SETTINGS
# =============================================================================

DERIVED_DIR = Path(os.getenv("DERIVED_DIR", str(cfg_get("DERIVED_DIR", REPO_ROOT / "data" / "derived"))))
RESULTS_DIR = Path(os.getenv("DEBIT_RESULTS_DIR", str(cfg_get("RESULTS_DIR", REPO_ROOT / "outputs" / "results"))))
LOG_DIR = Path(os.getenv("LOG_DIR", str(cfg_get("LOG_DIR", REPO_ROOT / "outputs" / "logs"))))

BASE_FIGURES_DIR = Path(os.getenv("FIGURES_DIR", str(cfg_get("FIGURES_DIR", REPO_ROOT / "outputs" / "figures"))))
BASE_TABLES_DIR = Path(os.getenv("TABLES_DIR", str(cfg_get("TABLES_DIR", REPO_ROOT / "outputs" / "tables"))))

MAIN_FIGURES_DIR = BASE_FIGURES_DIR / "main"
APPENDIX_FIGURES_DIR = BASE_FIGURES_DIR / "appendix"
MAIN_TABLES_DIR = BASE_TABLES_DIR / "main"
APPENDIX_TABLES_DIR = BASE_TABLES_DIR / "appendix"

INPUT_DIR = Path(
    os.getenv(
        "DEBIT_INPUT_DIR",
        str(cfg_get("DEBIT_OPT_INPUT_DIR", DERIVED_DIR / "debit_optimization_inputs")),
    )
)

BENCH_PATH = Path(os.getenv("BENCH_PATH", str(cfg_get("BENCH_PATH", RESULTS_DIR / "benchmark_full_results.csv"))))
PAIRWISE_PATH = Path(os.getenv("PAIRWISE_PATH", str(cfg_get("PAIRWISE_PATH", RESULTS_DIR / "benchmark_pairwise_differences.csv"))))
KKT_PATH = Path(os.getenv("KKT_PATH", str(cfg_get("KKT_PATH", RESULTS_DIR / "kkt_equalization_diagnostic.csv"))))
BENCH_MANIFEST_PATH = RESULTS_DIR / "benchmark_manifest.json"

M3_LOG_PATH = Path(
    os.getenv(
        "M3_LOG_PATH",
        str(cfg_get("M3_TRAINING_LOG_CSV", LOG_DIR / "m3_training_log.csv")),
    )
)

M3_Q_AUDIT_PATH = Path(
    os.getenv(
        "M3_Q_AUDIT_PATH",
        str(cfg_get("M3_TRAINING_Q_AUDIT_CSV", RESULTS_DIR / "m3_training_q_audit.csv")),
    )
)

M3_WEIGHTS_PATH = Path(
    os.getenv(
        "M3_Q_FILE",
        str(cfg_get("M3_WEIGHTS_PATH", DERIVED_DIR / "m3_debit_weights_by_year.parquet")),
    )
)

M3_WEIGHTS_WIDE_PATH = Path(
    os.getenv(
        "M3_Q_WIDE_FILE",
        str(cfg_get("M3_WEIGHTS_WIDE_PATH", DERIVED_DIR / "m3_debit_weights_wide_by_year.parquet")),
    )
)

STATE_PATH = Path(
    os.getenv(
        "STATE_PATH",
        str(cfg_get("STATE_PATH", DERIVED_DIR / "auditable_state_province_hour.csv")),
    )
)

DEFAULT_PROVINCES = [
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

NL_PROVINCES = list(cfg_get("NL_PROVINCES", DEFAULT_PROVINCES))

MAIN_FEATURE_SET = str(os.getenv("MAIN_FEATURE_SET", str(cfg_get("MAIN_FEATURE_SET", "F2")))).upper().strip()
MAIN_M3_VARIANT = str(os.getenv("MAIN_M3_VARIANT", str(cfg_get("MAIN_M3_VARIANT", "M3_LO")))).strip()

FEATURE_LADDER = [
    str(x).upper().strip()
    for x in env_str_list("M3_FEATURE_LADDER", cfg_get("FEATURE_LADDER", ["F0", "F1", "F2", "F3"]))
]

M3_VARIANTS = [
    str(x).strip()
    for x in env_str_list("M3_VARIANTS", cfg_get("M3_VARIANTS", ["M3_plain", "M3_OI", "M3_LO"]))
]

EVAL_YEARS = env_int_list("DEBIT_YEARS", cfg_get("ALL_YEARS", cfg_get("EVAL_YEARS", [2021, 2022, 2023, 2024, 2025])))
TEST_YEARS = env_int_list("TEST_YEARS", cfg_get("TEST_YEARS", []))

if TEST_YEARS:
    HOLDOUT_YEAR = int(TEST_YEARS[-1])
elif EVAL_YEARS:
    HOLDOUT_YEAR = int(EVAL_YEARS[-1])
else:
    HOLDOUT_YEAR = -1

SM = 1e6
GWH = 1000.0

FW = 3.35
FWD = 6.80

COL_MAIN = {
    "M1": "#6B6B6B",
    "M2": "#1F4E79",
    "M3": "#B03A2E",
}

COL_FS = {
    "F0": "#D8EAF7",
    "F1": "#8BBCE0",
    "F2": "#1F4E79",
    "F3": "#08306B",
}

COL_VAR = {
    "M3_plain": "#C7DCEF",
    "M3_OI": "#7EAED3",
    "M3_LO": "#1F4E79",
}

MARK_MAIN = {"M1": "s", "M2": "o", "M3": "^"}
MARK_VAR = {"M3_plain": "s", "M3_OI": "o", "M3_LO": "^"}
MARK_FS = {"F0": "s", "F1": "o", "F2": "^", "F3": "D"}

LABEL_COST = "Constructed operating objective (M EUR)"
LABEL_SLACK = "Emergency slack / unserved relief (GWh)"
LABEL_RELIEF = "Constructed relief served (%)"
LABEL_BUDGET = "Mean annual budget utilization"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.4,
    }
)

_MANIFEST: List[Dict[str, str]] = []


# =============================================================================
# SMALL UTILITIES
# =============================================================================

def _ensure_dirs() -> None:
    MAIN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, stem: str, note: str = "", out_dir: Optional[Path] = None) -> None:
    out_dir = MAIN_FIGURES_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    plt.close(fig)

    _MANIFEST.append(
        {
            "stem": stem,
            "figure_set": out_dir.name,
            "pdf": str(pdf_path),
            "png": str(png_path),
            "note": note,
        }
    )
    print(f"  ✓ {stem}")


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _as_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _read_table(path_or_stem: Path, required: bool = True) -> Optional[pd.DataFrame]:
    candidates: List[Path]

    if path_or_stem.suffix:
        candidates = [path_or_stem]
    else:
        candidates = [
            path_or_stem.with_suffix(".parquet"),
            path_or_stem.with_suffix(".csv"),
            Path(str(path_or_stem) + ".csv.gz"),
        ]

    for path in candidates:
        if path.exists():
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            if str(path).endswith(".csv.gz"):
                return pd.read_csv(path, compression="gzip", low_memory=False)
            return pd.read_csv(path, low_memory=False)

    if required:
        raise FileNotFoundError(f"Missing required table. Tried: {candidates}")
    return None


def _first_existing(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lower = {str(c).strip().lower(): str(c) for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        lc = str(c).strip().lower()
        if lc in lower:
            return lower[lc]
    if required:
        raise KeyError(f"Could not find any of {list(candidates)}. Available columns={list(df.columns)}")
    return None


def _canonical_variant(v: object) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    t = s.lower().replace("-", "_").replace(" ", "_").strip("_")

    if t in {"plain", "m3_plain", "q_m3_plain", "proxy", "proxy_plain"}:
        return "M3_plain"
    if t in {"oi", "m3_oi", "q_m3_oi", "one_shot", "oneshot", "optimization_informed", "opt_informed"}:
        return "M3_OI"
    if t in {"lo", "m3_lo", "q_m3_lo", "q_m3", "learn_optimize", "learn_opt", "alternating"}:
        return "M3_LO"

    if t.endswith("_plain"):
        return "M3_plain"
    if t.endswith("_oi"):
        return "M3_OI"
    if t.endswith("_lo"):
        return "M3_LO"

    return s


def _variant_short(v: object) -> str:
    c = _canonical_variant(v)
    if c == "M3_plain":
        return "plain"
    if c == "M3_OI":
        return "OI"
    if c == "M3_LO":
        return "LO"
    return str(v).replace("M3_", "")


def _canonical_feature(f: object) -> str:
    if f is None or pd.isna(f):
        return ""
    s = str(f).upper().strip()
    if s in {"0", "LEVEL0", "FEATURE0"}:
        return "F0"
    if s in {"1", "LEVEL1", "FEATURE1"}:
        return "F1"
    if s in {"2", "LEVEL2", "FEATURE2"}:
        return "F2"
    if s in {"3", "LEVEL3", "FEATURE3"}:
        return "F3"
    if "F0" in s:
        return "F0"
    if "F1" in s:
        return "F1"
    if "F2" in s:
        return "F2"
    if "F3" in s:
        return "F3"
    return s


def _hours_in_year_utc(year: int) -> int:
    t0 = pd.Timestamp(f"{int(year)}-01-01T00:00:00Z")
    t1 = pd.Timestamp(f"{int(year) + 1}-01-01T00:00:00Z")
    return int((t1 - t0) / pd.Timedelta(hours=1))


def _standardize_year_hour_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "year" not in out.columns:
        ts_col = _first_existing(out, ["timestamp_utc", "timestamp", "datetime_utc", "time"], required=False)
        if ts_col is None:
            raise KeyError("Table must contain either year or timestamp_utc/timestamp.")
        ts = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
        out["year"] = ts.dt.year

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["year"]).copy()
    out["year"] = out["year"].astype(int)

    ts_col = _first_existing(out, ["timestamp_utc", "timestamp", "datetime_utc", "time"], required=False)
    if ts_col is not None:
        if ts_col != "timestamp_utc":
            out = out.rename(columns={ts_col: "timestamp_utc"})
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True).dt.floor("h")

    h_col = _first_existing(out, ["hour_of_year", "hour", "hoy", "h"], required=False)
    if h_col is not None and h_col != "hour_of_year":
        out = out.rename(columns={h_col: "hour_of_year"})

    if "hour_of_year" not in out.columns:
        if "timestamp_utc" not in out.columns:
            raise KeyError("Need hour_of_year or timestamp_utc to standardize annual grid.")
        jan1 = pd.to_datetime(out["year"].astype(str) + "-01-01T00:00:00Z", utc=True)
        out["hour_of_year"] = ((out["timestamp_utc"] - jan1) / pd.Timedelta(hours=1)).round().astype(int) + 1

    out["hour_of_year"] = pd.to_numeric(out["hour_of_year"], errors="raise").astype(int)

    pieces: List[pd.DataFrame] = []
    for year, g in out.groupby("year", sort=False):
        gg = g.copy()
        H = _hours_in_year_utc(int(year))
        hmin = int(gg["hour_of_year"].min())
        hmax = int(gg["hour_of_year"].max())

        if hmin == 0 and hmax <= H - 1:
            gg["hour_of_year"] = gg["hour_of_year"] + 1

        if int(gg["hour_of_year"].min()) < 1 or int(gg["hour_of_year"].max()) > H:
            raise ValueError(
                f"Invalid hour_of_year range for year={year}: "
                f"min={gg['hour_of_year'].min()}, max={gg['hour_of_year'].max()}, expected 1..{H}."
            )

        if "timestamp_utc" not in gg.columns or gg["timestamp_utc"].isna().any():
            gg["timestamp_utc"] = (
                pd.Timestamp(f"{int(year)}-01-01T00:00:00Z")
                + pd.to_timedelta(gg["hour_of_year"] - 1, unit="h")
            )

        pieces.append(gg)

    out = pd.concat(pieces, ignore_index=True)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True).dt.floor("h")

    if out["timestamp_utc"].isna().any():
        raise ValueError("timestamp_utc still has missing values after standardization.")

    return out


def _grid(ax: plt.Axes, axis: str = "y") -> None:
    if axis in {"x", "both"}:
        ax.xaxis.grid(True, alpha=0.28, lw=0.45)
    if axis in {"y", "both"}:
        ax.yaxis.grid(True, alpha=0.28, lw=0.45)
    ax.set_axisbelow(True)


def _year_order_from(df: pd.DataFrame) -> List[int]:
    global EVAL_YEARS, HOLDOUT_YEAR

    if df is None or df.empty or "year" not in df.columns:
        return []

    years = sorted([int(y) for y in pd.to_numeric(df["year"], errors="coerce").dropna().unique()])

    if not EVAL_YEARS:
        EVAL_YEARS = years

    ordered = [int(y) for y in EVAL_YEARS if int(y) in years]
    if not ordered:
        ordered = years

    if HOLDOUT_YEAR < 0 and ordered:
        HOLDOUT_YEAR = int(ordered[-1])

    return ordered


def _holdout_shade(ax: plt.Axes, years: List[int]) -> None:
    if HOLDOUT_YEAR not in years:
        return
    j = years.index(HOLDOUT_YEAR)
    ax.axvspan(j - 0.5, j + 0.5, color=COL_MAIN["M3"], alpha=0.065, zorder=0)
    ax.axvline(j - 0.5, color=COL_MAIN["M3"], lw=0.6, ls="--", alpha=0.45)


def _latex_table(df: pd.DataFrame, path: Path, *, index: bool = False, float_format: str = "%.3f") -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_latex(path, index=index, escape=False, float_format=float_format)
    except Exception as exc:
        print(f"  [WARN] Could not write LaTeX table {path}: {exc}")


def _select_existing(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _write_csv_and_tex(df: pd.DataFrame, stem: str, out_dir: Path, *, float_format: str = "%.3f") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    tex_path = out_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    _latex_table(df, tex_path, float_format=float_format)
    print(f"  ✓ {stem}")


def _safe_pct_change(base: pd.Series, comp: pd.Series) -> pd.Series:
    base = pd.to_numeric(base, errors="coerce")
    comp = pd.to_numeric(comp, errors="coerce")
    denom = base.abs()
    return np.where(denom > 1e-9, 100.0 * (base - comp) / denom, np.nan)


# =============================================================================
# ROW FILTERS
# =============================================================================

def _main_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "comparison_group" in df.columns:
        out = df[(df["comparison_group"] == "main") & (df["rule"].isin(["M1", "M2", "M3"]))].copy()
    else:
        out = df[df["rule"].isin(["M1", "M2", "M3"])].copy()

    if out.empty:
        return out

    out["_order"] = out["rule"].map({"M1": 0, "M2": 1, "M3": 2}).fillna(99)
    out = out.sort_values(["year", "_order"]).drop(columns="_order")
    return out


def _variant_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "comparison_group" in df.columns:
        out = df[df["comparison_group"] == "variant_comparison"].copy()
    else:
        out = df[df["rule"].astype(str).str.contains("variant", case=False, na=False)].copy()

    if out.empty:
        return out

    if "feature_set" in out.columns:
        out["feature_set"] = out["feature_set"].map(_canonical_feature)
        out = out[out["feature_set"] == MAIN_FEATURE_SET].copy()

    if "variant" in out.columns:
        out["variant"] = out["variant"].map(_canonical_variant)

    order = {_canonical_variant(v): k for k, v in enumerate(M3_VARIANTS)}
    out["_order"] = out["variant"].map(order).fillna(99)
    out = out.sort_values(["year", "_order"]).drop(columns="_order")
    return out


def _ladder_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "comparison_group" in df.columns:
        out = df[df["comparison_group"] == "feature_ladder"].copy()
    else:
        out = df[df["rule"].astype(str).str.contains("feature", case=False, na=False)].copy()

    if out.empty:
        return out

    if "feature_set" in out.columns:
        out["feature_set"] = out["feature_set"].map(_canonical_feature)

    if "variant" in out.columns:
        out["variant"] = out["variant"].map(_canonical_variant)
        out = out[out["variant"] == _canonical_variant(MAIN_M3_VARIANT)].copy()

    out["_order"] = out["feature_set"].map({"F0": 0, "F1": 1, "F2": 2, "F3": 3}).fillna(99)
    out = out.sort_values(["year", "_order"]).drop(columns="_order")
    return out


# =============================================================================
# LOAD AND VALIDATE DATA
# =============================================================================

def _load_benchmark() -> pd.DataFrame:
    if not BENCH_PATH.exists():
        raise FileNotFoundError(
            f"Missing benchmark file:\n  {BENCH_PATH}\n"
            "Run scripts/D_run_full_benchmark.py first."
        )

    df = pd.read_csv(BENCH_PATH, low_memory=False)

    required = {"year", "rule", "status"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Benchmark file is missing required columns: {missing}")

    if "cost_total" not in df.columns:
        if "objective" in df.columns:
            df["cost_total"] = df["objective"]
        else:
            raise KeyError("Benchmark file must contain cost_total or objective.")

    if "slack_total" not in df.columns:
        if "total_slack" in df.columns:
            df["slack_total"] = df["total_slack"]
        else:
            raise KeyError("Benchmark file must contain slack_total or total_slack.")

    if "R_total" not in df.columns:
        if "total_relief_requirement" in df.columns:
            df["R_total"] = df["total_relief_requirement"]
        else:
            df["R_total"] = np.nan

    if "debit_total" not in df.columns:
        if "total_debit" in df.columns:
            df["debit_total"] = df["total_debit"]
        else:
            df["debit_total"] = np.nan

    if "budget_util" not in df.columns:
        if "mean_budget_utilization" in df.columns:
            df["budget_util"] = df["mean_budget_utilization"]
        else:
            df["budget_util"] = np.nan

    numeric_cols = [
        "year",
        "cost_total",
        "objective",
        "operating_cost_x",
        "slack_penalty_cost",
        "cost_slack",
        "slack_total",
        "total_slack",
        "R_total",
        "total_relief_requirement",
        "relief_served",
        "relief_served_share",
        "slack_share",
        "allowance_productivity",
        "budget_util",
        "mean_budget_utilization",
        "median_budget_utilization",
        "max_budget_utilization",
        "min_budget_utilization",
        "n_budget_binding",
        "physical_proxy_exposure",
        "debit_proxy_exposure",
        "physical_depth_total",
        "debit_total",
        "total_debit",
        "exposure_gini",
        "exposure_hhi",
        "exposure_top10_share",
        "debit_gini",
        "debit_hhi",
        "debit_top10_share",
        "mu_positive_share",
        "mu_positive_count",
        "mu_mean",
        "mu_max",
        "scarcity_active_share",
        "scarcity_alpha_sum",
        "scarcity_alpha_mean",
        "scarcity_alpha_max",
        "q_mean",
        "q_std",
        "q_min",
        "q_p01",
        "q_p05",
        "q_p50",
        "q_p95",
        "q_p99",
        "q_max",
        "q_nontrivial_share",
        "q_province_mean_min",
        "q_province_mean_max",
        "q_province_mean_std",
        "mip_gap",
        "best_bound",
        "node_count",
        "runtime_sec",
        "solve_time",
        "model_rows",
        "model_cols",
        "model_nnz",
    ]

    df = _as_numeric(df, numeric_cols)
    df["year"] = df["year"].astype(int)

    if "comparison_group" not in df.columns:
        df["comparison_group"] = np.where(df["rule"].isin(["M1", "M2", "M3"]), "main", "other")

    if "feature_set" in df.columns:
        df["feature_set"] = df["feature_set"].map(lambda x: np.nan if pd.isna(x) else _canonical_feature(x))
    else:
        df["feature_set"] = np.nan

    if "variant" in df.columns:
        df["variant"] = df["variant"].map(lambda x: np.nan if pd.isna(x) else _canonical_variant(x))
    else:
        df["variant"] = np.nan

    if "relief_served" not in df.columns or df["relief_served"].isna().all():
        df["relief_served"] = df["R_total"] - df["slack_total"]

    df["relief_served"] = pd.to_numeric(df["relief_served"], errors="coerce")
    df.loc[df["relief_served"] < 0, "relief_served"] = 0.0

    if "relief_served_share" not in df.columns or df["relief_served_share"].isna().all():
        df["relief_served_share"] = df["relief_served"] / df["R_total"].replace(0.0, np.nan)

    if "slack_share" not in df.columns or df["slack_share"].isna().all():
        df["slack_share"] = df["slack_total"] / df["R_total"].replace(0.0, np.nan)

    df["relief_served_share"] = pd.to_numeric(df["relief_served_share"], errors="coerce").clip(0.0, 1.0)
    df["slack_share"] = pd.to_numeric(df["slack_share"], errors="coerce").clip(0.0, 1.0)

    if "allowance_productivity" not in df.columns or df["allowance_productivity"].isna().all():
        df["allowance_productivity"] = df["relief_served"] / df["debit_total"].replace(0.0, np.nan)

    if "mu_positive_count" not in df.columns and "mu_positive_share" in df.columns and NL_PROVINCES:
        df["mu_positive_count"] = df["mu_positive_share"] * len(NL_PROVINCES)

    df["constructed_cost_MEUR"] = df["cost_total"] / SM
    df["emergency_slack_GWh"] = df["slack_total"] / GWH
    df["relief_served_pct"] = 100.0 * df["relief_served_share"]
    df["budget_util_pct"] = 100.0 * df["budget_util"]

    # Relative-to-M2 diagnostics by year, using main M2 only.
    main = _main_rows(df)
    m2 = (
        main[main["rule"] == "M2"][
            [
                "year",
                "cost_total",
                "slack_total",
                "relief_served_share",
                "physical_proxy_exposure",
                "debit_proxy_exposure",
            ]
        ]
        .drop_duplicates("year")
        .rename(
            columns={
                "cost_total": "cost_M2",
                "slack_total": "slack_M2",
                "relief_served_share": "served_M2",
                "physical_proxy_exposure": "physical_proxy_M2",
                "debit_proxy_exposure": "debit_proxy_M2",
            }
        )
    )

    df = df.merge(m2, on="year", how="left")

    df["cost_saving_vs_M2"] = df["cost_M2"] - df["cost_total"]
    df["cost_saving_vs_M2_pct"] = 100.0 * df["cost_saving_vs_M2"] / df["cost_M2"].replace(0.0, np.nan)

    df["slack_reduction_vs_M2"] = df["slack_M2"] - df["slack_total"]
    df["slack_reduction_vs_M2_pct_safe"] = np.where(
        df["slack_M2"].abs() > 1e-9,
        100.0 * df["slack_reduction_vs_M2"] / df["slack_M2"].abs(),
        np.nan,
    )

    df["served_gain_vs_M2_pctpt"] = 100.0 * (df["relief_served_share"] - df["served_M2"])
    df["physical_proxy_change_vs_M2"] = df["physical_proxy_exposure"] - df["physical_proxy_M2"]
    df["debit_proxy_change_vs_M2"] = df["debit_proxy_exposure"] - df["debit_proxy_M2"]

    return df


def _load_pairwise() -> pd.DataFrame:
    if not PAIRWISE_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(PAIRWISE_PATH, low_memory=False)

    numeric_cols = [
        "year",
        "cost_baseline",
        "cost_comparator",
        "cost_saving",
        "cost_saving_pct",
        "slack_baseline",
        "slack_comparator",
        "slack_reduction",
        "slack_reduction_pct",
        "relief_served_share_baseline",
        "relief_served_share_comparator",
        "relief_served_share_change",
        "allowance_productivity_baseline",
        "allowance_productivity_comparator",
        "allowance_productivity_change",
        "physical_proxy_baseline",
        "physical_proxy_comparator",
        "physical_proxy_change",
        "debit_proxy_baseline",
        "debit_proxy_comparator",
        "debit_proxy_change",
        "budget_util_baseline",
        "budget_util_comparator",
        "mu_positive_count_baseline",
        "mu_positive_count_comparator",
    ]

    df = _as_numeric(df, numeric_cols)

    if "year" in df.columns:
        df["year"] = df["year"].astype(int)

    if {"slack_baseline", "slack_comparator"}.issubset(df.columns):
        df["slack_reduction_pct_safe"] = np.where(
            df["slack_baseline"].abs() > 1e-9,
            100.0 * (df["slack_baseline"] - df["slack_comparator"]) / df["slack_baseline"].abs(),
            np.nan,
        )

    return df


def _load_log() -> pd.DataFrame:
    if not M3_LOG_PATH.exists():
        return pd.DataFrame()

    log = pd.read_csv(M3_LOG_PATH, low_memory=False)

    if "feature_set" not in log.columns and "feature_level" in log.columns:
        log["feature_set"] = log["feature_level"]

    if "feature_set" in log.columns:
        log["feature_set"] = log["feature_set"].map(_canonical_feature)
    else:
        log["feature_set"] = MAIN_FEATURE_SET

    if "variant" not in log.columns:
        if "architecture" in log.columns:
            log["variant"] = log["architecture"]
        else:
            log["variant"] = MAIN_M3_VARIANT

    log["variant"] = log["variant"].map(_canonical_variant)

    if "iter" not in log.columns and "iteration" in log.columns:
        log["iter"] = log["iteration"]

    numeric_cols = [
        "year",
        "iter",
        "iteration",
        "runtime_sec",
        "n_train_rows",
        "objective",
        "train_cost",
        "cost_total",
        "total_slack_MW_h",
        "slack_total",
        "served_requirement_share",
        "relief_served_share",
        "q_mean",
        "q_std",
        "q_min",
        "q_max",
        "alpha_raw_mean",
        "alpha_raw_max",
        "alpha_positive_share",
        "alpha_mean",
        "alpha_max",
        "mu_positive_share",
        "depth_positive_share",
        "delta_theta_rel",
        "sample_weight_mean",
        "sample_weight_max",
    ]

    log = _as_numeric(log, numeric_cols)

    if "year" in log.columns:
        log["year"] = pd.to_numeric(log["year"], errors="coerce").astype("Int64")

    return log


def _load_kkt() -> pd.DataFrame:
    if not KKT_PATH.exists():
        return pd.DataFrame()

    kkt = pd.read_csv(KKT_PATH, low_memory=False)

    # Normalize B/D KKT column names.
    if "mu_budget" not in kkt.columns:
        for c in ["mu_depth", "mu", "budget_dual", "annual_budget_dual"]:
            if c in kkt.columns:
                kkt["mu_budget"] = kkt[c]
                break

    if "scarcity_alpha" not in kkt.columns:
        for c in ["alpha_mu_x_over_P", "alpha"]:
            if c in kkt.columns:
                kkt["scarcity_alpha"] = kkt[c]
                break

    if "active_alpha_share" not in kkt.columns and "scarcity_alpha" in kkt.columns:
        # This is row-level file, so active share is computed later in aggregation.
        pass

    if "physical_depth" not in kkt.columns and {"x_mw", "Pctr_mw"}.issubset(kkt.columns):
        x = pd.to_numeric(kkt["x_mw"], errors="coerce")
        p = pd.to_numeric(kkt["Pctr_mw"], errors="coerce").replace(0.0, np.nan)
        kkt["physical_depth"] = (x / p).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "debit_total" not in kkt.columns:
        if {"debit_weight", "physical_depth"}.issubset(kkt.columns):
            kkt["debit"] = pd.to_numeric(kkt["debit_weight"], errors="coerce").fillna(1.0) * kkt["physical_depth"]
        elif "physical_depth" in kkt.columns:
            kkt["debit"] = kkt["physical_depth"]
        else:
            kkt["debit"] = np.nan

    numeric_cols = [
        "year",
        "hour_of_year",
        "x_mw",
        "Pctr_mw",
        "debit_weight",
        "mu_budget",
        "mu_depth",
        "mu",
        "physical_depth",
        "debit",
        "scarcity_alpha",
        "alpha",
        "alpha_mu_x_over_P",
        "scarcity_term_mu_q_over_P",
    ]

    kkt = _as_numeric(kkt, numeric_cols)

    if "year" in kkt.columns:
        kkt["year"] = pd.to_numeric(kkt["year"], errors="coerce").astype("Int64")

    if "rule" in kkt.columns:
        kkt["rule"] = kkt["rule"].astype(str)

    if "display_rule" in kkt.columns:
        # D writes both solver rule and display rule; display rule is cleaner.
        kkt["rule"] = kkt["display_rule"].fillna(kkt.get("rule", ""))

    if "variant" in kkt.columns:
        kkt["variant"] = kkt["variant"].map(lambda x: np.nan if pd.isna(x) else _canonical_variant(x))

    if "feature_set" in kkt.columns:
        kkt["feature_set"] = kkt["feature_set"].map(lambda x: np.nan if pd.isna(x) else _canonical_feature(x))

    return kkt


def _read_weights_file() -> pd.DataFrame:
    if M3_WEIGHTS_PATH.exists():
        raw = _read_table(M3_WEIGHTS_PATH, required=True)
    elif M3_WEIGHTS_WIDE_PATH.exists():
        raw = _read_table(M3_WEIGHTS_WIDE_PATH, required=True)
    else:
        return pd.DataFrame()

    assert raw is not None
    w = raw.copy()

    pcol = _first_existing(w, ["province", "provincie", "deployment_unit", "node"], required=False)
    if pcol is None:
        return pd.DataFrame()

    if pcol != "province":
        w = w.rename(columns={pcol: "province"})
    w["province"] = w["province"].astype(str).str.strip()

    w = _standardize_year_hour_timestamp(w)

    if "feature_set" not in w.columns:
        if "feature_level" in w.columns:
            w["feature_set"] = w["feature_level"]
        else:
            w["feature_set"] = MAIN_FEATURE_SET

    w["feature_set"] = w["feature_set"].map(_canonical_feature)

    if "feature_level" not in w.columns:
        w["feature_level"] = w["feature_set"]

    if "active_support" not in w.columns:
        w["active_support"] = 1

    # Preferred long format from C.
    if "variant" in w.columns and "debit_weight" in w.columns:
        out = w.copy()
        out["variant"] = out["variant"].map(_canonical_variant)
        out["debit_weight"] = pd.to_numeric(out["debit_weight"], errors="coerce")
    else:
        pieces: List[pd.DataFrame] = []

        wide_map = {
            "q_m3_plain": "M3_plain",
            "q_m3_OI": "M3_OI",
            "q_m3_LO": "M3_LO",
        }

        if "q_m3_LO" not in w.columns and "q_m3" in w.columns:
            wide_map["q_m3"] = "M3_LO"

        base_cols = ["year", "province", "timestamp_utc", "hour_of_year", "feature_set", "active_support"]
        base_cols = [c for c in base_cols if c in w.columns]

        for q_col, variant in wide_map.items():
            if q_col not in w.columns:
                continue
            tmp = w[base_cols + [q_col]].copy()
            tmp = tmp.rename(columns={q_col: "debit_weight"})
            tmp["variant"] = variant
            pieces.append(tmp)

        if not pieces:
            return pd.DataFrame()

        out = pd.concat(pieces, ignore_index=True)

    out = out[out["province"].isin(NL_PROVINCES)].copy()
    out["feature_set"] = out["feature_set"].map(_canonical_feature)
    out["variant"] = out["variant"].map(_canonical_variant)
    out["debit_weight"] = pd.to_numeric(out["debit_weight"], errors="coerce")
    out["active_support"] = pd.to_numeric(out.get("active_support", 1), errors="coerce").fillna(1).astype(int)

    out = out.dropna(subset=["debit_weight", "timestamp_utc"])
    out = out[out["debit_weight"] > 0].copy()

    return out.sort_values(["year", "feature_set", "variant", "province", "timestamp_utc"]).reset_index(drop=True)


def _load_weights(
    year: int,
    feature_set: str = MAIN_FEATURE_SET,
    variant: str = MAIN_M3_VARIANT,
    *,
    active_only: bool = False,
) -> pd.DataFrame:
    w = _read_weights_file()
    if w.empty:
        return w

    fs = _canonical_feature(feature_set)
    var = _canonical_variant(variant)

    out = w[
        (w["year"].astype(int) == int(year))
        & (w["feature_set"].map(_canonical_feature) == fs)
        & (w["variant"].map(_canonical_variant) == var)
    ].copy()

    if active_only and "active_support" in out.columns:
        out = out[out["active_support"].astype(int) == 1].copy()

    return out


def _load_state_year(year: int) -> pd.DataFrame:
    candidates = [
        STATE_PATH,
        INPUT_DIR / f"debit_opt_state_{int(year)}.parquet",
        INPUT_DIR / f"debit_opt_state_{int(year)}.csv",
        INPUT_DIR / f"debit_opt_state_{int(year)}.csv.gz",
    ]

    for path in candidates:
        if not path.exists():
            continue

        state = _read_table(path, required=False)
        if state is None or state.empty:
            continue

        if "province" not in state.columns:
            pcol = _first_existing(state, ["province", "provincie", "deployment_unit", "node"], required=False)
            if pcol is None:
                continue
            state = state.rename(columns={pcol: "province"})

        state["province"] = state["province"].astype(str).str.strip()

        try:
            state = _standardize_year_hour_timestamp(state)
        except Exception:
            if "timestamp_utc" not in state.columns:
                continue
            state["timestamp_utc"] = pd.to_datetime(state["timestamp_utc"], utc=True, errors="coerce")
            state["year"] = state["timestamp_utc"].dt.year

        state["year"] = pd.to_numeric(state["year"], errors="coerce").astype("Int64")
        state = state[state["year"].astype(int) == int(year)].copy()
        state = state[state["province"].isin(NL_PROVINCES)].copy()

        if not state.empty:
            return state

    return pd.DataFrame()


def _validate_d_outputs(bench: pd.DataFrame, pairwise: pd.DataFrame, weights: pd.DataFrame) -> None:
    required_main = {"M1", "M2", "M3"}
    main = _main_rows(bench)
    available = set(main["rule"].astype(str).unique())

    missing = required_main - available
    if missing:
        raise RuntimeError(
            f"Main benchmark is missing rules: {sorted(missing)}. "
            "Run B1 and D again before generating final figures."
        )

    required_cols = [
        "year",
        "rule",
        "comparison_group",
        "cost_total",
        "slack_total",
        "relief_served_share",
        "budget_util",
        "debit_total",
        "quality_flag",
    ]

    missing_cols = [c for c in required_cols if c not in bench.columns]
    if missing_cols:
        raise RuntimeError(f"benchmark_full_results.csv is missing required columns: {missing_cols}")

    if pairwise.empty:
        raise RuntimeError("benchmark_pairwise_differences.csv is missing or empty. Run D again.")

    if weights.empty:
        raise RuntimeError("M3 weights file is missing or empty. Run C again.")

    main_m3 = main[main["rule"] == "M3"].copy()
    if main_m3.empty:
        raise RuntimeError("No main M3 rows found.")

    if main_m3["feature_set"].dropna().astype(str).str.upper().nunique() > 1:
        print("[WARN] Main M3 contains multiple feature sets. Check D configuration.")

    if main_m3["variant"].dropna().astype(str).nunique() > 1:
        print("[WARN] Main M3 contains multiple variants. Check D configuration.")

    main_quality = main["quality_flag"].astype(str).str.lower()
    bad_quality = main_quality.str.contains("failed|gap_too_large|infeasible", regex=True, na=False)
    if bad_quality.any():
        print("[WARN] Some main rows have problematic quality_flag values:")
        print(main.loc[bad_quality, ["year", "rule", "quality_flag"]].to_string(index=False))


# =============================================================================
# MAIN FIGURE F1: MAIN TRADEOFF
# =============================================================================

def _annotate_holdout_m3_vs_m2(ax: plt.Axes, main: pd.DataFrame, metric_col: str, *, lower_is_better: bool = True) -> None:
    if HOLDOUT_YEAR < 0:
        return

    years = _year_order_from(main)
    if HOLDOUT_YEAR not in years:
        return

    h = main[main["year"] == HOLDOUT_YEAR]
    if h.empty:
        return

    m2 = h[h["rule"] == "M2"]
    m3 = h[h["rule"] == "M3"]
    if m2.empty or m3.empty:
        return

    b = _safe_float(m2.iloc[0].get(metric_col))
    c = _safe_float(m3.iloc[0].get(metric_col))

    if not np.isfinite(b) or not np.isfinite(c) or abs(b) < 1e-9:
        return

    pct = 100.0 * (b - c) / abs(b) if lower_is_better else 100.0 * (c - b) / abs(b)
    j = years.index(HOLDOUT_YEAR)

    ylim = ax.get_ylim()
    y = ylim[1] - 0.06 * (ylim[1] - ylim[0])

    ax.text(
        j,
        y,
        f"M3 vs M2: {pct:+.1f}%",
        ha="center",
        va="top",
        fontsize=6.7,
        color=COL_MAIN["M3"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.5),
    )


def figure1_main_tradeoff(bench: pd.DataFrame) -> None:
    main = _main_rows(bench)
    years = _year_order_from(main)

    if main.empty or not years:
        print("  F1 skipped: no main M1/M2/M3 rows.")
        return

    x = np.arange(len(years))
    width = 0.24

    rule_label = {
        "M1": "M1 touched-time",
        "M2": "M2 uniform-depth",
        "M3": rf"M3 {MAIN_FEATURE_SET}/{_variant_short(MAIN_M3_VARIANT)}",
    }

    fig, axs = plt.subplots(2, 2, figsize=(FWD, 4.75))
    ax_cost, ax_slack, ax_served, ax_budget = axs.ravel()

    # Cost bars.
    for j, rule in enumerate(["M1", "M2", "M3"]):
        sub = main[main["rule"] == rule].set_index("year")
        vals = [sub.loc[y, "constructed_cost_MEUR"] if y in sub.index else np.nan for y in years]

        for k, val in enumerate(vals):
            if not np.isfinite(val):
                continue
            hatch = "//" if years[k] == HOLDOUT_YEAR else None
            ax_cost.bar(
                x[k] + (j - 1) * width,
                val,
                width,
                color=COL_MAIN[rule],
                edgecolor="white",
                linewidth=0.45,
                hatch=hatch,
                alpha=0.90,
                label=rule_label[rule] if k == 0 else None,
            )

    _holdout_shade(ax_cost, years)
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels([str(y) for y in years])
    ax_cost.set_ylabel(LABEL_COST)
    ax_cost.set_title("(a) Annual constructed operating objective", loc="left")
    ax_cost.legend(frameon=False, ncol=1, loc="upper left", fontsize=6.9)
    _grid(ax_cost, "y")
    _annotate_holdout_m3_vs_m2(ax_cost, main, "constructed_cost_MEUR", lower_is_better=True)

    # Slack bars.
    for j, rule in enumerate(["M1", "M2", "M3"]):
        sub = main[main["rule"] == rule].set_index("year")
        vals = [sub.loc[y, "emergency_slack_GWh"] if y in sub.index else np.nan for y in years]

        for k, val in enumerate(vals):
            if not np.isfinite(val):
                continue
            hatch = "//" if years[k] == HOLDOUT_YEAR else None
            ax_slack.bar(
                x[k] + (j - 1) * width,
                val,
                width,
                color=COL_MAIN[rule],
                edgecolor="white",
                linewidth=0.45,
                hatch=hatch,
                alpha=0.88,
                label=rule_label[rule] if k == 0 else None,
            )

    _holdout_shade(ax_slack, years)
    ax_slack.set_xticks(x)
    ax_slack.set_xticklabels([str(y) for y in years])
    ax_slack.set_ylabel(LABEL_SLACK)
    ax_slack.set_title("(b) Emergency slack", loc="left")
    _grid(ax_slack, "y")
    _annotate_holdout_m3_vs_m2(ax_slack, main, "emergency_slack_GWh", lower_is_better=True)

    # Relief served lines.
    for rule in ["M1", "M2", "M3"]:
        sub = main[main["rule"] == rule].set_index("year")
        vals = [sub.loc[y, "relief_served_pct"] if y in sub.index else np.nan for y in years]
        ax_served.plot(
            x,
            vals,
            marker=MARK_MAIN[rule],
            color=COL_MAIN[rule],
            label=rule_label[rule],
            linewidth=1.5,
        )

    _holdout_shade(ax_served, years)
    ax_served.set_xticks(x)
    ax_served.set_xticklabels([str(y) for y in years])
    ax_served.set_ylabel(LABEL_RELIEF)
    ax_served.set_ylim(0, 103)
    ax_served.set_title("(c) Constructed relief served", loc="left")
    _grid(ax_served, "y")

    # Budget utilization lines.
    for rule in ["M1", "M2", "M3"]:
        sub = main[main["rule"] == rule].set_index("year")
        vals = [sub.loc[y, "budget_util"] if y in sub.index else np.nan for y in years]
        ax_budget.plot(
            x,
            vals,
            marker=MARK_MAIN[rule],
            color=COL_MAIN[rule],
            label=rule_label[rule],
            linewidth=1.5,
        )

    m3 = main[main["rule"] == "M3"].set_index("year")
    for k, y in enumerate(years):
        if y in m3.index and "mu_positive_count" in m3.columns:
            c = _safe_float(m3.loc[y, "mu_positive_count"])
            if np.isfinite(c):
                ax_budget.text(k, 0.035, rf"$\mu^+$={int(round(c))}", ha="center", va="bottom", fontsize=6.4)

    _holdout_shade(ax_budget, years)
    ax_budget.axhline(1.0, color="black", lw=0.7, ls=":", alpha=0.55)
    ax_budget.set_xticks(x)
    ax_budget.set_xticklabels([str(y) for y in years])
    ax_budget.set_ylabel(LABEL_BUDGET)
    ax_budget.set_ylim(0, max(1.16, float(np.nanmax(main["budget_util"])) * 1.1 if np.isfinite(main["budget_util"]).any() else 1.16))
    ax_budget.set_title("(d) Annual allowance pressure", loc="left")
    _grid(ax_budget, "y")

    fig.suptitle(
        "Main debit-rule comparison on the fixed constructed relief testbed",
        fontsize=9,
        y=1.02,
    )

    _save(
        fig,
        "F1_main_tradeoff",
        (
            "Main M1/M2/M3 benchmark. Cost is the constructed operating objective; "
            "emergency slack is unserved constructed relief; M1 uses touched-token utilization, "
            "whereas M2/M3 use depth-budget utilization."
        ),
        out_dir=MAIN_FIGURES_DIR,
    )


# =============================================================================
# MAIN FIGURE F2: DIRECT M2 VS M3 PAIRWISE IMPROVEMENT
# =============================================================================

def figure2_m2_m3_pairwise(pairwise: pd.DataFrame) -> None:
    if pairwise.empty:
        print("  F2 skipped: no pairwise table.")
        return

    df = pairwise[pairwise["comparison"] == "M2_minus_M3"].copy()
    if df.empty:
        print("  F2 skipped: no M2_minus_M3 rows.")
        return

    df = df.sort_values("year")
    years = [int(y) for y in df["year"].tolist()]
    x = np.arange(len(years))

    if "slack_reduction_pct_safe" not in df.columns:
        df["slack_reduction_pct_safe"] = np.where(
            df["slack_baseline"].abs() > 1e-9,
            100.0 * df["slack_reduction"] / df["slack_baseline"].abs(),
            np.nan,
        )

    fig, axs = plt.subplots(1, 3, figsize=(FWD, 2.75))
    ax_cost, ax_slack, ax_relief = axs

    ax_cost.bar(
        x,
        df["cost_saving_pct"],
        color=COL_MAIN["M3"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_cost.axhline(0.0, color="black", lw=0.7, ls=":", alpha=0.7)
    ax_cost.set_ylabel("Cost saving vs M2 (%)")
    ax_cost.set_title("(a) Cost saving", loc="left")

    ax_slack.bar(
        x,
        df["slack_reduction"] / GWH,
        color=COL_MAIN["M3"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_slack.axhline(0.0, color="black", lw=0.7, ls=":", alpha=0.7)
    ax_slack.set_ylabel("Slack reduction vs M2 (GWh)")
    ax_slack.set_title("(b) Emergency-slack reduction", loc="left")

    if "relief_served_share_change" in df.columns:
        ax_relief.bar(
            x,
            100.0 * df["relief_served_share_change"],
            color=COL_MAIN["M3"],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
        )
        ax_relief.axhline(0.0, color="black", lw=0.7, ls=":", alpha=0.7)
        ax_relief.set_ylabel("Relief-served gain (p.p.)")
        ax_relief.set_title("(c) Relief-served change", loc="left")
    else:
        ax_relief.axis("off")
        ax_relief.set_title("(c) Relief-served change unavailable", loc="left")

    for ax in axs:
        _holdout_shade(ax, years)
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years])
        _grid(ax, "y")

    fig.suptitle(
        rf"Main M3 improvement relative to M2: {MAIN_FEATURE_SET}/{_variant_short(MAIN_M3_VARIANT)}",
        fontsize=9,
        y=1.03,
    )

    _save(
        fig,
        "F2_m2_m3_pairwise_improvement",
        "Direct pairwise comparison of main M3 against M2.",
        out_dir=MAIN_FIGURES_DIR,
    )


# =============================================================================
# MAIN FIGURE F3: INFORMATION LADDER
# =============================================================================

def figure3_information_ladder(bench: pd.DataFrame) -> None:
    lad = _ladder_rows(bench)
    years = _year_order_from(bench)

    if lad.empty or not years:
        print("  F3 skipped: no feature-ladder rows.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(FWD, 2.8))
    ax_cost, ax_served, ax_slack = axs

    for fs in ["F0", "F1", "F2", "F3"]:
        sub = lad[lad["feature_set"] == fs].set_index("year")
        if sub.empty:
            continue

        xs = np.arange(len(years))
        vals_cost = [sub.loc[y, "cost_saving_vs_M2_pct"] if y in sub.index else np.nan for y in years]
        vals_served = [sub.loc[y, "relief_served_pct"] if y in sub.index else np.nan for y in years]
        vals_slack = [sub.loc[y, "slack_reduction_vs_M2"] / GWH if y in sub.index else np.nan for y in years]

        ax_cost.plot(xs, vals_cost, marker=MARK_FS.get(fs, "o"), color=COL_FS.get(fs, "gray"), label=fs)
        ax_served.plot(xs, vals_served, marker=MARK_FS.get(fs, "o"), color=COL_FS.get(fs, "gray"), label=fs)
        ax_slack.plot(xs, vals_slack, marker=MARK_FS.get(fs, "o"), color=COL_FS.get(fs, "gray"), label=fs)

    for ax in axs:
        _holdout_shade(ax, years)
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels([str(y) for y in years])
        _grid(ax, "y")

    ax_cost.axhline(0.0, color="black", lw=0.65, ls=":", alpha=0.55)
    ax_slack.axhline(0.0, color="black", lw=0.65, ls=":", alpha=0.55)

    ax_cost.set_title("(a) Cost saving vs M2", loc="left")
    ax_served.set_title("(b) Constructed relief served", loc="left")
    ax_slack.set_title("(c) Slack reduction vs M2", loc="left")

    ax_cost.set_ylabel("Cost saving (%)")
    ax_served.set_ylabel("Relief served (%)")
    ax_slack.set_ylabel("Slack reduction (GWh)")

    ax_cost.legend(frameon=False, title="Feature set", title_fontsize=7.5, loc="best")

    fig.suptitle(
        rf"Auditable information ladder for $M3^{{\ell,\mathrm{{{_variant_short(MAIN_M3_VARIANT)}}}}}$",
        fontsize=9,
        y=1.02,
    )

    _save(
        fig,
        "F3_information_ladder_operational_value",
        (
            "Feature ladder F0-F3. Richer feature sets are evaluated empirically; "
            "the figure should not be read as a finite-sample monotonicity theorem."
        ),
        out_dir=MAIN_FIGURES_DIR,
    )


# =============================================================================
# MAIN FIGURE F4: KKT / SCARCITY MECHANISM
# =============================================================================

def figure4_kkt_scarcity_mechanism(bench: pd.DataFrame, kkt: pd.DataFrame) -> None:
    main = _main_rows(bench)
    years = _year_order_from(main)

    if main.empty or not years:
        print("  F4 skipped: no main benchmark rows.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(FWD, 2.85))
    ax_bind, ax_mu, ax_alpha = axs
    xs = np.arange(len(years))

    for rule in ["M2", "M3"]:
        sub = main[main["rule"] == rule].set_index("year")
        if sub.empty:
            continue

        bind_vals = [sub.loc[y, "mu_positive_count"] if y in sub.index else np.nan for y in years]
        mu_vals = [sub.loc[y, "mu_max"] if y in sub.index else np.nan for y in years]
        alpha_vals = [
            sub.loc[y, "scarcity_alpha_sum"] if y in sub.index and "scarcity_alpha_sum" in sub.columns else np.nan
            for y in years
        ]

        ax_bind.plot(xs, bind_vals, marker=MARK_MAIN[rule], color=COL_MAIN[rule], label=rule)
        ax_mu.plot(xs, mu_vals, marker=MARK_MAIN[rule], color=COL_MAIN[rule], label=rule)
        ax_alpha.plot(xs, alpha_vals, marker=MARK_MAIN[rule], color=COL_MAIN[rule], label=rule)

    for ax in axs:
        _holdout_shade(ax, years)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(y) for y in years])
        _grid(ax, "y")

    ax_bind.set_title("(a) Binding annual budgets", loc="left")
    ax_bind.set_ylabel("Number of provinces")

    ax_mu.set_title(r"(b) Maximum budget dual $\mu_n$", loc="left")
    ax_mu.set_ylabel(r"Max $\mu_n$")

    ax_alpha.set_title(r"(c) Aggregate scarcity signal", loc="left")
    ax_alpha.set_ylabel(r"$\sum_{n,t}\mu_n x_{n,t}/P_n$")

    ax_bind.legend(frameon=False, fontsize=7)

    fig.suptitle(
        r"Scarcity mechanism under annual depth accounting",
        fontsize=9,
        y=1.03,
    )

    _save(
        fig,
        "F4_kkt_scarcity_mechanism",
        r"Scarcity mechanism: binding budgets, budget duals, and alpha = mu*x/P.",
        out_dir=MAIN_FIGURES_DIR,
    )


# =============================================================================
# MAIN FIGURE F5: Q AUDIT AND STRUCTURE
# =============================================================================

def figure5_q_audit_and_structure(weights: pd.DataFrame) -> None:
    if weights.empty:
        print("  F5 skipped: no M3 weights.")
        return

    w = weights.copy()
    w["variant"] = w["variant"].map(_canonical_variant)
    w["feature_set"] = w["feature_set"].map(_canonical_feature)

    main_w = w[
        (w["feature_set"] == MAIN_FEATURE_SET)
        & (w["variant"] == _canonical_variant(MAIN_M3_VARIANT))
    ].copy()

    if main_w.empty:
        print("  F5 skipped: no main M3 weights.")
        return

    main_w = _standardize_year_hour_timestamp(main_w)

    if "active_support" in main_w.columns:
        audit_w = main_w[main_w["active_support"].astype(int) == 1].copy()
        if audit_w.empty:
            audit_w = main_w.copy()
    else:
        audit_w = main_w.copy()

    fig = plt.figure(figsize=(FWD, 4.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], width_ratios=[1.0, 1.0])
    ax_norm = fig.add_subplot(gs[0, 0])
    ax_bounds = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, :])

    gy = (
        audit_w.groupby(["year", "province"], observed=True)["debit_weight"]
        .agg(q_mean="mean", q_min="min", q_max="max", q_std="std")
        .reset_index()
    )

    years = sorted(gy["year"].dropna().astype(int).unique().tolist())
    xs = np.arange(len(years))

    norm_err = (
        gy.assign(abs_err=lambda d: (d["q_mean"] - 1.0).abs())
        .groupby("year", as_index=False)["abs_err"]
        .max()
        .set_index("year")
    )

    ax_norm.bar(
        xs,
        [norm_err.loc[y, "abs_err"] if y in norm_err.index else np.nan for y in years],
        color=COL_MAIN["M3"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_norm.set_xticks(xs)
    ax_norm.set_xticklabels([str(y) for y in years])
    ax_norm.set_ylabel(r"max $|\bar q_{n,y}-1|$")
    ax_norm.set_title("(a) Active-support normalization error", loc="left")
    _grid(ax_norm, "y")

    bounds = (
        gy.groupby("year", as_index=False)
        .agg(q_min=("q_min", "min"), q_max=("q_max", "max"))
        .set_index("year")
    )

    ax_bounds.plot(xs, [bounds.loc[y, "q_min"] if y in bounds.index else np.nan for y in years], marker="v", color="#1F4E79", label="min")
    ax_bounds.plot(xs, [bounds.loc[y, "q_max"] if y in bounds.index else np.nan for y in years], marker="^", color="#B03A2E", label="max")
    ax_bounds.axhline(1.0, color="black", lw=0.7, ls=":", alpha=0.6)
    ax_bounds.set_xticks(xs)
    ax_bounds.set_xticklabels([str(y) for y in years])
    ax_bounds.set_ylabel(r"Debit-weight range")
    ax_bounds.set_title("(b) Active-support q bounds", loc="left")
    ax_bounds.legend(frameon=False, fontsize=7)
    _grid(ax_bounds, "y")

    hw = main_w[main_w["year"] == HOLDOUT_YEAR].copy()
    if hw.empty and years:
        hw = main_w[main_w["year"] == max(years)].copy()

    if "active_support" in hw.columns:
        hw_active = hw[hw["active_support"].astype(int) == 1].copy()
        if not hw_active.empty:
            hw = hw_active

    hw["dow"] = hw["timestamp_utc"].dt.dayofweek
    hw["hour"] = hw["timestamp_utc"].dt.hour
    hw["hour_of_week"] = hw["dow"] * 24 + hw["hour"]

    pivot = (
        hw.groupby(["province", "hour_of_week"], observed=True)["debit_weight"]
        .mean()
        .unstack("hour_of_week")
        .reindex(columns=list(range(168)))
    )

    pivot = pivot.reindex([p for p in NL_PROVINCES if p in pivot.index])
    pivot = pivot.dropna(how="all").fillna(1.0)

    if pivot.empty:
        ax_heat.axis("off")
        ax_heat.set_title("(c) Holdout learned debit-weight structure unavailable", loc="left")
    else:
        vals = pivot.to_numpy(dtype=float)
        vmin = max(0.25, float(np.nanmin(vals)))
        vmax = min(4.0, float(np.nanmax(vals)))
        if vmin >= 1.0:
            vmin = 0.25
        if vmax <= 1.0:
            vmax = 4.0

        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
        im = ax_heat.imshow(vals, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")

        ax_heat.set_xticks([0, 24, 48, 72, 96, 120, 144])
        ax_heat.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax_heat.set_yticks(np.arange(len(pivot.index)))
        ax_heat.set_yticklabels([str(p)[:16] for p in pivot.index])
        ax_heat.set_xlabel("Hour of week")
        ax_heat.set_ylabel("Province")
        ax_heat.set_title(
            rf"(c) Holdout learned debit weights $\widehat q_{{n,t}}$, {MAIN_FEATURE_SET}/{_variant_short(MAIN_M3_VARIANT)}",
            loc="left",
        )

        for h in [24, 48, 72, 96, 120, 144]:
            ax_heat.axvline(h - 0.5, color="black", lw=0.35, alpha=0.35)

        cbar = fig.colorbar(im, ax=ax_heat, shrink=0.78, pad=0.012)
        cbar.set_label(r"Debit weight $\widehat q_{n,t}$")

    fig.suptitle("Auditability and structure of learned M3 debit rates", fontsize=9, y=1.02)

    _save(
        fig,
        "F5_q_audit_and_structure",
        "Audit of q normalization, q bounds, and holdout hour-of-week debit-rate structure.",
        out_dir=MAIN_FIGURES_DIR,
    )


# =============================================================================
# APPENDIX FIGURE A1: FULL METHOD HEATMAP
# =============================================================================

def _method_panel_for_heatmap(bench: pd.DataFrame) -> pd.DataFrame:
    pieces: List[pd.DataFrame] = []

    main = _main_rows(bench)
    if not main.empty:
        sub = main[main["rule"].isin(["M1", "M2", "M3"])].copy()
        sub["method"] = sub["rule"].map(
            {
                "M1": "Main: M1 touched-time",
                "M2": "Main: M2 uniform-depth",
                "M3": rf"Main: M3 {MAIN_FEATURE_SET}/{_variant_short(MAIN_M3_VARIANT)}",
            }
        )
        sub["family"] = "main"
        pieces.append(sub)

    var = _variant_rows(bench)
    if not var.empty:
        sub = var.copy()
        sub["method"] = sub["variant"].map(lambda v: rf"Architecture: {_variant_short(v)}/{MAIN_FEATURE_SET}")
        sub["family"] = "architecture"
        pieces.append(sub)

    lad = _ladder_rows(bench)
    if not lad.empty:
        sub = lad.copy()
        short_variant = _variant_short(MAIN_M3_VARIANT)
        sub["method"] = "Information: " + sub["feature_set"].astype(str) + f"/{short_variant}"
        sub["family"] = "information"
        pieces.append(sub)

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True)

    order = [
        rf"Architecture: plain/{MAIN_FEATURE_SET}",
        rf"Architecture: OI/{MAIN_FEATURE_SET}",
        rf"Architecture: LO/{MAIN_FEATURE_SET}",
        f"Information: F0/{_variant_short(MAIN_M3_VARIANT)}",
        f"Information: F1/{_variant_short(MAIN_M3_VARIANT)}",
        f"Information: F2/{_variant_short(MAIN_M3_VARIANT)}",
        f"Information: F3/{_variant_short(MAIN_M3_VARIANT)}",
        "Main: M1 touched-time",
        "Main: M2 uniform-depth",
        rf"Main: M3 {MAIN_FEATURE_SET}/{_variant_short(MAIN_M3_VARIANT)}",
    ]

    out["_order"] = out["method"].map({m: i for i, m in enumerate(order)}).fillna(99)
    out = out.sort_values(["_order", "year"])
    return out.drop(columns="_order")


def figure_A1_method_heatmap(bench: pd.DataFrame) -> None:
    panel = _method_panel_for_heatmap(bench)
    years = _year_order_from(bench)

    if panel.empty or not years:
        print("  A1 skipped: no method panel data.")
        return

    pivot = panel.pivot_table(
        index="method",
        columns="year",
        values="cost_saving_vs_M2_pct",
        aggfunc="first",
    )

    pivot = pivot[[y for y in years if y in pivot.columns]]
    pivot = pivot.dropna(how="all")

    if pivot.empty:
        print("  A1 skipped: empty heatmap.")
        return

    vals = pivot.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    vmax = max(5.0, float(np.nanpercentile(np.abs(finite), 95))) if finite.size else 5.0
    vmax = min(max(vmax, 10.0), 100.0)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(FWD, max(3.2, 0.32 * len(pivot.index) + 1.0)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", norm=norm)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title("(a) Cost saving relative to M2 across debit-rule specifications", loc="left")
    ax.set_xlabel("Year")
    ax.set_ylabel("Specification")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                color = "white" if abs(val) > 0.55 * vmax else "black"
                ax.text(j, i, f"{val:+.0f}", ha="center", va="center", fontsize=6.5, color=color)

    if HOLDOUT_YEAR in list(pivot.columns):
        j = list(pivot.columns).index(HOLDOUT_YEAR)
        ax.axvline(j - 0.5, color="black", lw=0.8, ls="--", alpha=0.55)
        ax.axvline(j + 0.5, color="black", lw=0.8, ls="--", alpha=0.55)

    cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.015)
    cbar.set_label("Cost saving vs M2 (%)")

    _save(
        fig,
        "A1_method_cost_saving_heatmap",
        "Appendix heatmap of cost saving versus M2 for main, architecture, and information-ladder specifications.",
        out_dir=APPENDIX_FIGURES_DIR,
    )


# =============================================================================
# APPENDIX FIGURE A2: ARCHITECTURE COMPARISON
# =============================================================================

def figure_A2_training_architectures(bench: pd.DataFrame) -> None:
    var = _variant_rows(bench)
    years = _year_order_from(bench)

    if var.empty or not years:
        print("  A2 skipped: no M3 variant rows.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(FWD, 2.85))
    ax_cost, ax_served, ax_q = axs

    for variant in [_canonical_variant(v) for v in M3_VARIANTS]:
        sub = var[var["variant"] == variant].set_index("year")
        if sub.empty:
            continue

        xs = np.arange(len(years))
        vals_cost = [sub.loc[y, "cost_saving_vs_M2_pct"] if y in sub.index else np.nan for y in years]
        vals_served = [sub.loc[y, "relief_served_pct"] if y in sub.index else np.nan for y in years]
        vals_qstd = [sub.loc[y, "q_std"] if y in sub.index and "q_std" in sub.columns else np.nan for y in years]

        label = _variant_short(variant)
        ax_cost.plot(xs, vals_cost, marker=MARK_VAR.get(variant, "o"), color=COL_VAR.get(variant, "gray"), label=label)
        ax_served.plot(xs, vals_served, marker=MARK_VAR.get(variant, "o"), color=COL_VAR.get(variant, "gray"), label=label)
        ax_q.plot(xs, vals_qstd, marker=MARK_VAR.get(variant, "o"), color=COL_VAR.get(variant, "gray"), label=label)

    for ax in axs:
        _holdout_shade(ax, years)
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels([str(y) for y in years])
        _grid(ax, "y")

    ax_cost.axhline(0.0, color="black", lw=0.65, ls=":", alpha=0.55)

    ax_cost.set_title("(a) Cost saving vs M2", loc="left")
    ax_served.set_title("(b) Relief served", loc="left")
    ax_q.set_title("(c) Debit-weight dispersion", loc="left")

    ax_cost.set_ylabel("Cost saving (%)")
    ax_served.set_ylabel("Relief served (%)")
    ax_q.set_ylabel(r"Std. dev. of $\widehat q$")

    ax_cost.legend(frameon=False, title="Architecture", title_fontsize=7.5)

    fig.suptitle(
        f"Architecture sensitivity at feature set {MAIN_FEATURE_SET}",
        fontsize=9,
        y=1.02,
    )

    _save(
        fig,
        "A2_m3_architecture_comparison",
        "Appendix comparison of M3_plain, M3_OI, and M3_LO at the main feature set.",
        out_dir=APPENDIX_FIGURES_DIR,
    )


# =============================================================================
# APPENDIX FIGURE A3: RESPONSE CURVES
# =============================================================================

def _binned_curve(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 50:
        return np.array([]), np.array([]), np.array([]), np.array([])

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))

    if len(edges) < 4:
        edges = np.linspace(float(np.min(x)), float(np.max(x)), min(n_bins, 10) + 1)
        edges = np.unique(edges)

    if len(edges) < 3:
        return np.array([]), np.array([]), np.array([]), np.array([])

    centers: List[float] = []
    med: List[float] = []
    lo: List[float] = []
    hi: List[float] = []

    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue

        m = (x >= a) & (x <= b)
        if m.sum() < 10:
            continue

        centers.append(float(np.median(x[m])))
        med.append(float(np.median(y[m])))
        lo.append(float(np.quantile(y[m], 0.25)))
        hi.append(float(np.quantile(y[m], 0.75)))

    return np.array(centers), np.array(med), np.array(lo), np.array(hi)


def figure_A3_m3_response_curves() -> None:
    w = _load_weights(HOLDOUT_YEAR, MAIN_FEATURE_SET, MAIN_M3_VARIANT, active_only=True)
    state = _load_state_year(HOLDOUT_YEAR)

    if w.empty or state.empty:
        print("  A3 skipped: missing weights or state panel.")
        return

    state["timestamp_utc"] = pd.to_datetime(state["timestamp_utc"], utc=True, errors="coerce").dt.floor("h")
    state["province"] = state["province"].astype(str).str.strip()

    candidate_vars = [
        ("s_plus_norm", "Public stress $s^+$"),
        ("rl_plus_norm", "Residual-load pressure"),
        ("price_da_norm", "Day-ahead price state"),
        ("burden_proxy_norm", "Burden proxy"),
        ("relief_value_proxy_norm", "Relief-value proxy"),
        ("ptdf_effectivity_norm", "Effectivity proxy"),
        ("restriction_cost_norm", "Restriction-cost proxy"),
    ]

    use_cols = ["province", "timestamp_utc"]
    for col, _ in candidate_vars:
        if col in state.columns and col not in use_cols:
            use_cols.append(col)

    merged = w.merge(state[use_cols], on=["province", "timestamp_utc"], how="left")
    merged = merged.dropna(subset=["debit_weight"]).copy()

    available: List[Tuple[str, str]] = []
    seen_labels = set()

    for col, lab in candidate_vars:
        if col not in merged.columns or lab in seen_labels:
            continue

        z = pd.to_numeric(merged[col], errors="coerce")
        if z.notna().sum() < 100:
            continue
        if z.nunique(dropna=True) < 10:
            continue
        if float(z.std(skipna=True)) <= 1e-8:
            continue

        available.append((col, lab))
        seen_labels.add(lab)

    if not available:
        print("  A3 skipped: no response variables with sufficient variation.")
        return

    n_panels = min(4, len(available))
    fig, axs = plt.subplots(1, n_panels, figsize=(FWD, 2.85))
    if n_panels == 1:
        axs = np.array([axs])

    rng = np.random.default_rng(7)

    for idx_ax, (ax, (col, label)) in enumerate(zip(axs, available[:n_panels])):
        x = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(merged["debit_weight"], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        idx = np.where(mask)[0]

        if len(idx) > 12000:
            idx = rng.choice(idx, size=12000, replace=False)

        ax.scatter(x[idx], y[idx], s=4.0, alpha=0.10, color=COL_MAIN["M2"], edgecolor="none")

        cx, med, lo, hi = _binned_curve(x, y, n_bins=22)
        if len(cx):
            ax.plot(cx, med, color=COL_MAIN["M3"], lw=1.5, marker="o", ms=3.0, label="median")
            ax.fill_between(cx, lo, hi, color=COL_MAIN["M3"], alpha=0.15, label="IQR")

        ax.axhline(1.0, color="black", lw=0.7, ls=":", alpha=0.6)
        ax.set_xlabel(label)
        ax.set_title(f"({chr(97 + idx_ax)}) q response", loc="left")
        _grid(ax, "y")

    axs[0].set_ylabel(r"Learned debit weight $\widehat q$")
    axs[0].legend(frameon=False, fontsize=6.5, loc="best")

    fig.suptitle(
        rf"M3 active-support response curves: {MAIN_FEATURE_SET}, {_variant_short(MAIN_M3_VARIANT)}, {HOLDOUT_YEAR}",
        fontsize=9,
        y=1.02,
    )

    _save(
        fig,
        "A3_m3_response_curves",
        "Appendix response curves linking learned debit weights to public stress and proxy variables.",
        out_dir=APPENDIX_FIGURES_DIR,
    )


# =============================================================================
# APPENDIX FIGURE A4: TRAINING DIAGNOSTICS
# =============================================================================

def _training_target_used_flag(s: pd.Series) -> float:
    text = " ".join(str(x).lower() for x in s.dropna().tolist())
    if "alpha_blended" in text or "optimization" in text or "weighted_loss" in text:
        return 1.0
    return 0.0


def _heatmap_with_labels(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    cbar_label: str,
    cmap: str = "Blues",
    fmt: str = "{:.2f}",
    center: Optional[float] = None,
) -> None:
    if data.empty:
        ax.axis("off")
        ax.set_title(title + " (missing)", loc="left")
        return

    vals = data.to_numpy(dtype=float)

    if center is not None:
        finite = vals[np.isfinite(vals)]
        span = float(np.nanmax(np.abs(finite - center))) if finite.size else 1.0
        if span <= 0:
            span = 1.0
        norm = mcolors.TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)
    else:
        norm = None

    im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(list(data.columns), rotation=30, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(list(data.index))
    ax.set_title(title, loc="left")

    finite = vals[np.isfinite(vals)]
    threshold = np.nanmedian(finite) if finite.size else 0.0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iloc[i, j]
            if np.isfinite(v):
                color = "white" if v > threshold and cmap not in {"RdBu_r", "RdBu", "Greens"} else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.4, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.015)
    cbar.set_label(cbar_label)


def figure_A4_training_diagnostics(log: pd.DataFrame, weights: pd.DataFrame) -> None:
    if log.empty:
        print("  A4 skipped: no training log.")
        return

    log2 = log.copy()
    log2["feature_set"] = log2["feature_set"].map(_canonical_feature)
    log2["variant"] = log2["variant"].map(_canonical_variant)

    feature_order = [f for f in ["F0", "F1", "F2", "F3"] if f in set(log2["feature_set"])]
    variant_order = [_canonical_variant(v) for v in M3_VARIANTS if _canonical_variant(v) in set(log2["variant"])]

    if not feature_order or not variant_order:
        print("  A4 skipped: no feature/variant ordering.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(FWD, 4.65))
    ax_rows, ax_signal, ax_runtime, ax_q = axs.ravel()

    if "n_train_rows" in log2.columns:
        train_rows = (
            log2.pivot_table(index="feature_set", columns="variant", values="n_train_rows", aggfunc="max")
            .reindex(index=feature_order, columns=variant_order)
        )
        _heatmap_with_labels(ax_rows, train_rows, "(a) Training rows", "rows", cmap="Blues", fmt="{:.0f}")
    else:
        ax_rows.axis("off")
        ax_rows.set_title("(a) Training rows unavailable", loc="left")

    if "target_source" in log2.columns:
        target = (
            log2.groupby(["feature_set", "variant"], observed=True)["target_source"]
            .apply(_training_target_used_flag)
            .reset_index(name="optimization_signal_used")
        )
        signal = (
            target.pivot_table(index="feature_set", columns="variant", values="optimization_signal_used", aggfunc="max")
            .reindex(index=feature_order, columns=variant_order)
        )
        _heatmap_with_labels(ax_signal, signal, "(b) Optimization signal used", "1=yes", cmap="Greens", fmt="{:.0f}")
    else:
        ax_signal.axis("off")
        ax_signal.set_title("(b) Optimization signal unavailable", loc="left")

    if "runtime_sec" in log2.columns:
        runtime = (
            log2.pivot_table(index="feature_set", columns="variant", values="runtime_sec", aggfunc="sum")
            .reindex(index=feature_order, columns=variant_order)
        )
        _heatmap_with_labels(ax_runtime, runtime, "(c) Training runtime", "seconds", cmap="Purples", fmt="{:.0f}")
    else:
        ax_runtime.axis("off")
        ax_runtime.set_title("(c) Runtime unavailable", loc="left")

    if not weights.empty:
        w = weights.copy()
        w["feature_set"] = w["feature_set"].map(_canonical_feature)
        w["variant"] = w["variant"].map(_canonical_variant)
        if "active_support" in w.columns:
            w = w[w["active_support"].astype(int) == 1].copy()

        qstd = (
            w.groupby(["feature_set", "variant"], observed=True)["debit_weight"]
            .std()
            .reset_index(name="q_std")
            .pivot_table(index="feature_set", columns="variant", values="q_std", aggfunc="mean")
            .reindex(index=feature_order, columns=variant_order)
        )
        _heatmap_with_labels(ax_q, qstd, "(d) Active-support q dispersion", r"std(q)", cmap="Oranges", fmt="{:.2f}")
    else:
        ax_q.axis("off")
        ax_q.set_title("(d) q dispersion unavailable", loc="left")

    fig.suptitle("Optimization-informed M3 training diagnostics", fontsize=9, y=1.02)

    _save(
        fig,
        "A4_training_diagnostics",
        "Appendix training diagnostics from C_train_m3.py.",
        out_dir=APPENDIX_FIGURES_DIR,
    )


# =============================================================================
# APPENDIX FIGURE A5: OPERATING FRONTIER
# =============================================================================

def figure_A5_operating_frontier(bench: pd.DataFrame) -> None:
    main = _main_rows(bench)
    var = _variant_rows(bench)
    lad = _ladder_rows(bench)

    pieces: List[pd.DataFrame] = []

    if not main.empty:
        m = main.copy()
        m["display"] = m["rule"]
        m["group"] = "Main"
        pieces.append(m)

    if not var.empty:
        v = var.copy()
        v["display"] = v["variant"].astype(str).map(_variant_short)
        v["group"] = "M3 architecture"
        pieces.append(v)

    if not lad.empty:
        l = lad.copy()
        l["display"] = l["feature_set"].astype(str)
        l["group"] = "M3 information"
        pieces.append(l)

    if not pieces:
        print("  A5 skipped: no frontier data.")
        return

    df = pd.concat(pieces, ignore_index=True)
    df = df[np.isfinite(df["constructed_cost_MEUR"]) & np.isfinite(df["relief_served_pct"])].copy()

    if df.empty:
        print("  A5 skipped: invalid frontier values.")
        return

    fig, ax = plt.subplots(figsize=(FWD * 0.72, 3.55))

    group_style = {
        "Main": ("#222222", "o"),
        "M3 architecture": (COL_MAIN["M2"], "^"),
        "M3 information": (COL_MAIN["M3"], "s"),
    }

    for group, sub in df.groupby("group", observed=True):
        color, marker = group_style.get(group, ("gray", "o"))
        non_hold = sub[sub["year"] != HOLDOUT_YEAR]
        hold = sub[sub["year"] == HOLDOUT_YEAR]

        ax.scatter(
            non_hold["constructed_cost_MEUR"],
            non_hold["relief_served_pct"],
            s=28,
            color=color,
            alpha=0.25,
            marker=marker,
            edgecolor="none",
            label=f"{group}, other years",
        )

        ax.scatter(
            hold["constructed_cost_MEUR"],
            hold["relief_served_pct"],
            s=70,
            color=color,
            alpha=0.92,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            label=f"{group}, holdout",
            zorder=4,
        )

        for _, r in hold.iterrows():
            ax.annotate(
                str(r["display"]),
                (r["constructed_cost_MEUR"], r["relief_served_pct"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6.5,
            )

    ax.set_xlabel(LABEL_COST)
    ax.set_ylabel(LABEL_RELIEF)
    ax.set_title("(a) Operating frontier: lower cost and higher served relief are better", loc="left")
    _grid(ax, "both")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right", ncol=1)

    _save(
        fig,
        "A5_cost_relief_frontier",
        "Appendix operating frontier: constructed annual cost versus constructed relief served.",
        out_dir=APPENDIX_FIGURES_DIR,
    )


# =============================================================================
# MAIN TABLES
# =============================================================================

def write_table_experimental_setup() -> None:
    rows = [
        {
            "Component": "Physical layer",
            "Definition": "Fixed constructed branch-hour relief testbed",
            "Paper meaning": "Held fixed across M1/M2/M3",
        },
        {
            "Component": "M1",
            "Definition": r"Touched-time debit $d_{n,t}=\mathbf{1}\{x_{n,t}>0\}$",
            "Paper meaning": "Publication M1 imported from B1",
        },
        {
            "Component": "M2",
            "Definition": r"Uniform-depth debit $d_{n,t}=x_{n,t}/P_n$",
            "Paper meaning": "Solved by B through weighted-depth LP with $q=1$",
        },
        {
            "Component": "M3",
            "Definition": r"Weighted-depth debit $d_{n,t}=q_{n,t}x_{n,t}/P_n$",
            "Paper meaning": "q learned by C; final benchmark solved by D through B",
        },
        {
            "Component": "Main M3",
            "Definition": rf"{MAIN_FEATURE_SET}/{_variant_short(MAIN_M3_VARIANT)}",
            "Paper meaning": "Main learned auditable debit rule",
        },
        {
            "Component": "Relief-served share",
            "Definition": r"$1-\sum s/\sum R$",
            "Paper meaning": "Constructed-testbed metric, not real redispatch performance",
        },
        {
            "Component": "Scarcity diagnostic",
            "Definition": r"$\alpha_{n,t}=\mu_n x_{n,t}/P_n$",
            "Paper meaning": "Learning-relevant marginal accounting signal",
        },
    ]

    df = pd.DataFrame(rows)
    _write_csv_and_tex(df, "Table_I_experimental_setup", MAIN_TABLES_DIR, float_format="%.3f")


def write_table_main_annual_results(bench: pd.DataFrame) -> None:
    main = _main_rows(bench).copy()
    if main.empty:
        return

    out = pd.DataFrame(
        {
            "Year": main["year"].astype(int),
            "Rule": main["rule"].astype(str),
            "Cost (M EUR)": main["cost_total"] / SM,
            "Slack (GWh)": main["slack_total"] / GWH,
            "Relief served (%)": 100.0 * main["relief_served_share"],
            "Debit": main["debit_total"],
            "Budget util.": main["budget_util"],
            "Binding budgets": main["mu_positive_count"],
        }
    )

    out = out.sort_values(["Year", "Rule"])
    _write_csv_and_tex(out, "Table_II_main_annual_results", MAIN_TABLES_DIR, float_format="%.2f")


def write_table_m2_m3_pairwise(pairwise: pd.DataFrame) -> None:
    if pairwise.empty:
        return

    df = pairwise[pairwise["comparison"] == "M2_minus_M3"].copy()
    if df.empty:
        return

    if "slack_reduction_pct_safe" not in df.columns:
        df["slack_reduction_pct_safe"] = np.where(
            df["slack_baseline"].abs() > 1e-9,
            100.0 * df["slack_reduction"] / df["slack_baseline"].abs(),
            np.nan,
        )

    out = pd.DataFrame(
        {
            "Year": df["year"].astype(int),
            "Cost saving (M EUR)": df["cost_saving"] / SM,
            "Cost saving (%)": df["cost_saving_pct"],
            "Slack reduction (GWh)": df["slack_reduction"] / GWH,
            "Slack reduction (%)": df["slack_reduction_pct_safe"],
            "Relief-served gain (p.p.)": 100.0 * df["relief_served_share_change"],
            "Productivity change": df["allowance_productivity_change"],
        }
    )

    out = out.sort_values("Year")
    _write_csv_and_tex(out, "Table_III_M2_vs_M3_pairwise", MAIN_TABLES_DIR, float_format="%.2f")


def write_table_sensitivity_summary(bench: pd.DataFrame) -> None:
    pieces: List[pd.DataFrame] = []

    lad = _ladder_rows(bench).copy()
    if not lad.empty:
        lad["Specification"] = lad["feature_set"].astype(str) + "/" + lad["variant"].map(_variant_short)
        lad["Block"] = "Information ladder"
        pieces.append(lad)

    var = _variant_rows(bench).copy()
    if not var.empty:
        var["Specification"] = var["feature_set"].astype(str) + "/" + var["variant"].map(_variant_short)
        var["Block"] = "Architecture"
        pieces.append(var)

    if not pieces:
        return

    df = pd.concat(pieces, ignore_index=True)

    summary = (
        df.groupby(["Block", "Specification"], observed=True)
        .agg(
            mean_cost_saving_pct=("cost_saving_vs_M2_pct", "mean"),
            mean_slack_reduction_GWh=("slack_reduction_vs_M2", lambda s: float(np.nanmean(s) / GWH)),
            mean_relief_served_pct=("relief_served_share", lambda s: 100.0 * float(np.nanmean(s))),
            mean_q_std=("q_std", "mean"),
        )
        .reset_index()
    )

    hold = df[df["year"] == HOLDOUT_YEAR].copy()
    if not hold.empty:
        hold2 = (
            hold.groupby(["Block", "Specification"], observed=True)
            .agg(
                holdout_cost_saving_pct=("cost_saving_vs_M2_pct", "mean"),
                holdout_slack_reduction_GWh=("slack_reduction_vs_M2", lambda s: float(np.nanmean(s) / GWH)),
                holdout_relief_served_pct=("relief_served_share", lambda s: 100.0 * float(np.nanmean(s))),
            )
            .reset_index()
        )
        summary = summary.merge(hold2, on=["Block", "Specification"], how="left")

    _write_csv_and_tex(summary, "Table_IV_sensitivity_summary", MAIN_TABLES_DIR, float_format="%.2f")


# =============================================================================
# APPENDIX TABLES
# =============================================================================

def write_appendix_tables(bench: pd.DataFrame, pairwise: pd.DataFrame, kkt: pd.DataFrame, weights: pd.DataFrame) -> None:
    # Full benchmark.
    cols = _select_existing(
        bench,
        [
            "year",
            "comparison_group",
            "rule",
            "feature_set",
            "variant",
            "cost_total",
            "operating_cost_x",
            "slack_penalty_cost",
            "slack_total",
            "R_total",
            "relief_served_share",
            "debit_total",
            "physical_depth_total",
            "allowance_productivity",
            "budget_util",
            "n_budget_binding",
            "physical_proxy_exposure",
            "debit_proxy_exposure",
            "exposure_hhi",
            "debit_hhi",
            "mu_positive_count",
            "mu_mean",
            "mu_max",
            "scarcity_alpha_sum",
            "scarcity_alpha_mean",
            "scarcity_alpha_max",
            "q_mean",
            "q_std",
            "q_min",
            "q_p01",
            "q_p50",
            "q_p99",
            "q_max",
            "runtime_sec",
            "quality_flag",
            "status",
        ],
    )
    _write_csv_and_tex(bench[cols].copy(), "Table_A_full_benchmark_results", APPENDIX_TABLES_DIR, float_format="%.3f")

    # Full pairwise.
    if not pairwise.empty:
        _write_csv_and_tex(pairwise.copy(), "Table_A_full_pairwise_differences", APPENDIX_TABLES_DIR, float_format="%.3f")

    # KKT summary.
    if not kkt.empty and "mu_budget" in kkt.columns:
        kk = kkt.copy()
        kk["mu_positive"] = pd.to_numeric(kk["mu_budget"], errors="coerce").fillna(0.0) > 1e-9

        if "scarcity_alpha" not in kk.columns:
            if "alpha_mu_x_over_P" in kk.columns:
                kk["scarcity_alpha"] = kk["alpha_mu_x_over_P"]
            elif "alpha" in kk.columns:
                kk["scarcity_alpha"] = kk["alpha"]

        group_cols = ["rule"]
        if "comparison_group" in kk.columns:
            group_cols = ["comparison_group", "rule"]
        if "feature_set" in kk.columns:
            group_cols.append("feature_set")
        if "variant" in kk.columns:
            group_cols.append("variant")

        agg_cols = {
            "mu_budget": "mean",
            "mu_positive": "mean",
        }

        if "scarcity_alpha" in kk.columns:
            agg_cols["scarcity_alpha"] = ["mean", "max", "sum"]
        if "physical_depth" in kk.columns:
            agg_cols["physical_depth"] = "sum"
        if "debit" in kk.columns:
            agg_cols["debit"] = "sum"

        out = kk.groupby(group_cols, observed=True).agg(agg_cols)
        out.columns = ["_".join([str(x) for x in c if str(x) != ""]) for c in out.columns.to_flat_index()]
        out = out.reset_index()
        out = out.rename(columns={"mu_positive_mean": "mu_positive_share"})
        _write_csv_and_tex(out, "Table_A_kkt_scarcity_summary", APPENDIX_TABLES_DIR, float_format="%.3f")

    # q audit.
    if not weights.empty:
        w = weights.copy()
        w["feature_set"] = w["feature_set"].map(_canonical_feature)
        w["variant"] = w["variant"].map(_canonical_variant)
        if "active_support" in w.columns:
            w_active = w[w["active_support"].astype(int) == 1].copy()
            if not w_active.empty:
                w = w_active

        out = (
            w.groupby(["feature_set", "variant"], observed=True)["debit_weight"]
            .agg(
                q_mean="mean",
                q_std="std",
                q_min="min",
                q_p01=lambda s: float(np.quantile(s, 0.01)),
                q_p50="median",
                q_p99=lambda s: float(np.quantile(s, 0.99)),
                q_max="max",
                n_rows="size",
            )
            .reset_index()
            .sort_values(["feature_set", "variant"])
        )

        _write_csv_and_tex(out, "Table_A_q_audit", APPENDIX_TABLES_DIR, float_format="%.3f")


def write_table_solver_quality(bench: pd.DataFrame) -> None:
    cols = _select_existing(
        bench,
        [
            "year",
            "comparison_group",
            "rule",
            "feature_set",
            "variant",
            "status",
            "quality_flag",
            "mip_gap",
            "best_bound",
            "runtime_sec",
            "model_rows",
            "model_cols",
            "model_nnz",
            "notes",
        ],
    )

    out = bench[cols].copy()
    _write_csv_and_tex(out, "Table_A_solver_quality", APPENDIX_TABLES_DIR, float_format="%.3f")


# =============================================================================
# CAPTIONS AND INTERPRETATION NOTES
# =============================================================================

def write_figure_captions() -> None:
    captions = {
        "F1_main_tradeoff": (
            "Main debit-rule comparison on the fixed constructed relief testbed. "
            "Cost is the constructed operating objective, emergency slack is unserved constructed relief, "
            "and relief-served share equals one minus total emergency slack divided by total constructed relief requirement. "
            "M1 uses touched-time accounting, whereas M2 and M3 use depth-based annual accounting."
        ),
        "F2_m2_m3_pairwise_improvement": (
            "Direct improvement of the main learned weighted-depth rule relative to the uniform-depth rule M2. "
            "Positive values indicate lower constructed operating objective, lower emergency slack, or higher relief-served share under M3."
        ),
        "F3_information_ladder_operational_value": (
            "Auditable information ladder for M3. F0 is calendar-only, F1 adds public load/price stress, "
            "F2 adds burden/relief/cost proxies, and F3 adds constructed PTDF-testbed effectivity/exposure proxies. "
            "The comparison is empirical; richer information is not guaranteed to improve finite-sample performance."
        ),
        "F4_kkt_scarcity_mechanism": (
            r"Scarcity mechanism under annual depth accounting. The learning-relevant diagnostic is "
            r"$\alpha_{n,t}=\mu_n x_{n,t}/P_n$, not $\mu_n q_{n,t}x_{n,t}/P_n$."
        ),
        "F5_q_audit_and_structure": (
            "Auditability and structure of the learned M3 debit rates. The learned rates are bounded and province-year normalized, "
            "so M3 reallocates allowance consumption across hours rather than creating additional annual entitlement."
        ),
    }

    path = MAIN_TABLES_DIR / "main_figure_captions.md"
    text = "\n\n".join([f"## {k}\n\n{v}" for k, v in captions.items()])
    path.write_text(text, encoding="utf-8")
    print(f"  ✓ {path.name}")


def write_interpretation_notes() -> None:
    path = MAIN_TABLES_DIR / "figure_and_table_interpretation_notes.md"

    text = f"""# IEEE TSG figure and table interpretation notes

## Core empirical question

The experiment compares annual debit rules for non-firm grid access on a fixed constructed physical-relief testbed. The physical relief layer is held fixed across rules. What changes is the accounting rule that consumes the annual allowance and therefore changes which future restrictions remain feasible.

The comparison is not a reconstruction of actual Dutch operator redispatch. It is an auditable model-side experiment for studying how different allowance-debit designs perform under the same constructed branch-hour relief requirements.

## Main rule definitions

- **M1 touched-time:** a positive restriction consumes one token, independently of depth.
- **M2 uniform-depth:** debit equals normalized physical restriction depth, $x/P$.
- **M3 weighted-depth:** debit equals $q x/P$, where $q$ is a bounded, province-year-normalized, auditable debit rate.

The main M3 specification in the figures is:

- feature set: **{MAIN_FEATURE_SET}**
- training architecture: **{_canonical_variant(MAIN_M3_VARIANT)}**

## Main figures

### F1 Main operating tradeoff

Use this figure for the first empirical claim: how M1, M2, and M3 differ in constructed operating objective, emergency slack, constructed relief served, and annual budget pressure.

### F2 M2-vs-M3 pairwise improvement

Use this as the central M2-versus-M3 evidence. It separates cost saving, emergency-slack reduction, and relief-served gain.

### F3 Information ladder

This is the main figure for the auditable-information argument. It answers how much operational value is obtained as the public/auditable feature set is enriched from F0 to F3. The correct interpretation is empirical and out-of-sample. It is not a theorem that F3 dominates F2 or F1 in finite samples.

### F4 KKT/scarcity mechanism

The scarcity diagnostic uses:

$$
\\alpha_{{n,t}} = \\mu_n \\frac{{x_{{n,t}}}}{{P_n}}.
$$

This is the derivative of the annual budget constraint with respect to the debit rate $q_{{n,t}}$. Do not replace it by $\\mu_n q_{{n,t}}x_{{n,t}}/P_n$ when discussing learning signals.

### F5 q audit and learned structure

This is the contract/audit figure. It verifies that the learned rates remain bounded and approximately province-year-normalized. It supports the claim that M3 reallocates accounting rates within a controlled annual normalization rather than creating hidden additional allowance.

## Claims to avoid

Avoid the following overclaims:

1. Do not say M3 reconstructs operator dispatch.
2. Do not say richer information always improves performance.
3. Do not say the learned q is a market price.
4. Do not say relief-served share is actual energy served in the Dutch grid.
5. Do not say the KKT diagnostic proves fairness by itself; it is a scarcity and marginal-accounting diagnostic.

## Strong paper framing

The strongest empirical framing is:

> Holding the constructed physical relief problem fixed, the annual accounting rule materially changes which restrictions remain feasible over time. Weighted-depth debit rules can use auditable public information to shift allowance consumption toward restrictions that deliver more operational value per unit of contract debit, while preserving ex ante boundedness and annual normalization of the debit rates.

"""

    path.write_text(text, encoding="utf-8")
    print(f"  ✓ {path.name}")


def write_manifest_files() -> None:
    manifest = pd.DataFrame(_MANIFEST)

    main_manifest = MAIN_FIGURES_DIR / "figure_manifest.csv"
    appendix_manifest = APPENDIX_FIGURES_DIR / "figure_manifest.csv"
    full_manifest = BASE_FIGURES_DIR / "figure_manifest_all.csv"

    BASE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not manifest.empty:
        manifest.to_csv(full_manifest, index=False)
        manifest[manifest["figure_set"] == MAIN_FIGURES_DIR.name].to_csv(main_manifest, index=False)
        manifest[manifest["figure_set"] == APPENDIX_FIGURES_DIR.name].to_csv(appendix_manifest, index=False)

    print(f"  ✓ {full_manifest}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    _ensure_dirs()

    print("=" * 78)
    print("E: MAKE IEEE-TSG-READY FIGURES AND TABLES")
    print("=" * 78)
    print(f"Repo root     : {REPO_ROOT}")
    print(f"Benchmark     : {BENCH_PATH}")
    print(f"Pairwise      : {PAIRWISE_PATH}")
    print(f"KKT           : {KKT_PATH}")
    print(f"M3 log        : {M3_LOG_PATH}")
    print(f"M3 weights    : {M3_WEIGHTS_PATH}")
    print(f"State         : {STATE_PATH}")
    print(f"Main figures  : {MAIN_FIGURES_DIR}")
    print(f"Appendix figs : {APPENDIX_FIGURES_DIR}")
    print(f"Main tables   : {MAIN_TABLES_DIR}")
    print(f"Appendix tabs : {APPENDIX_TABLES_DIR}")
    print(f"Main M3       : feature_set={MAIN_FEATURE_SET}, variant={_canonical_variant(MAIN_M3_VARIANT)}")
    print("=" * 78)

    bench = _load_benchmark()
    pairwise = _load_pairwise()
    log = _load_log()
    kkt = _load_kkt()
    weights = _read_weights_file()

    _year_order_from(bench)
    _validate_d_outputs(bench, pairwise, weights)

    print(f"Years         : {EVAL_YEARS}")
    print(f"Holdout       : {HOLDOUT_YEAR}")
    print(f"Benchmark rows: {len(bench):,}")
    print(f"Pairwise rows : {len(pairwise):,}")
    print(f"KKT rows      : {len(kkt):,}")
    print(f"Weight rows   : {len(weights):,}")

    print("\nMain-paper figures:")
    main_tasks = [
        ("F1", lambda: figure1_main_tradeoff(bench)),
        ("F2", lambda: figure2_m2_m3_pairwise(pairwise)),
        ("F3", lambda: figure3_information_ladder(bench)),
        ("F4", lambda: figure4_kkt_scarcity_mechanism(bench, kkt)),
        ("F5", lambda: figure5_q_audit_and_structure(weights)),
    ]

    for name, fn in main_tasks:
        try:
            fn()
        except Exception as exc:
            print(f"  ! {name} failed: {exc}")

    print("\nAppendix figures:")
    appendix_tasks = [
        ("A1", lambda: figure_A1_method_heatmap(bench)),
        ("A2", lambda: figure_A2_training_architectures(bench)),
        ("A3", figure_A3_m3_response_curves),
        ("A4", lambda: figure_A4_training_diagnostics(log, weights)),
        ("A5", lambda: figure_A5_operating_frontier(bench)),
    ]

    for name, fn in appendix_tasks:
        try:
            fn()
        except Exception as exc:
            print(f"  ! {name} failed: {exc}")

    print("\nMain-paper tables:")
    try:
        write_table_experimental_setup()
        write_table_main_annual_results(bench)
        write_table_m2_m3_pairwise(pairwise)
        write_table_sensitivity_summary(bench)
        write_figure_captions()
        write_interpretation_notes()
    except Exception as exc:
        print(f"  ! Main table generation failed: {exc}")

    print("\nAppendix tables:")
    try:
        write_appendix_tables(bench, pairwise, kkt, weights)
        write_table_solver_quality(bench)
    except Exception as exc:
        print(f"  ! Appendix table generation failed: {exc}")

    print("\nManifests:")
    write_manifest_files()

    print("\nDone.")
    print(f"✓ Main figures     → {MAIN_FIGURES_DIR}")
    print(f"✓ Appendix figures → {APPENDIX_FIGURES_DIR}")
    print(f"✓ Main tables      → {MAIN_TABLES_DIR}")
    print(f"✓ Appendix tables  → {APPENDIX_TABLES_DIR}")


if __name__ == "__main__":
    main()