#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D_run_full_benchmark.py
=======================

Signed-off final annual benchmark runner for the annual debit-rule design paper
in a Dutch TDTR-like non-firm grid-access setting.

This script is deliberately strict. It is the final source of truth for paper
benchmark tables after A, B1, B, and C have been rerun.

Pipeline discipline
-------------------
A_build_debit_optimization_inputs.py
    Builds the fixed constructed annual physical relief testbed:
        data/derived/debit_optimization_inputs/

B1_run_m1_touched_only.py
    Source of truth for M1 touched-time accounting:
        outputs/results/m1_only/m1_only_summary.csv
        outputs/results/m1_only/solutions/M1_province_hour_{year}.csv.gz
        outputs/results/m1_only/solutions/M1_budget_summary_{year}.csv

B_run_annual_lp.py
    Source-of-truth weighted-depth LP/QP solver API for M2 and M3:
        load_year_input
        solve_weighted_depth_lp
        kkt_equalization_diagnostic
        result_summary_row
        save_result

C_train_m3.py
    Source of learned M3 debit weights:
        data/derived/m3_debit_weights_by_year.parquet/.csv
        data/derived/m3_debit_weights_wide_by_year.parquet/.csv
        data/derived/m3_training_manifest.json
        outputs/results/m3_training_q_audit.csv

D_run_full_benchmark.py
    Final benchmark runner:
        imports publication M1 from B1,
        solves M2 through B with q=1,
        solves M3 through B using C q weights,
        validates that main M3 q blocks exist and are normalized,
        writes final benchmark tables.

Core modelling discipline
-------------------------
1. The physical relief layer is fixed across M1/M2/M3.
2. D does not rebuild costs, Pbar, relief requirements, coefficients, support,
   or budgets.
3. M1 comes from B1 publication-grade touched-time outputs.
4. M2 is uniform-depth: q=1, debit=x/P.
5. M3 is weighted-depth: debit=q*x/P using C q weights.
6. Missing main M3 weights are a hard error; D never replaces missing M3 q by 1.
7. q is an accounting debit rate, not a market price, redispatch price, or
   electricity price.
8. The learning/scarcity diagnostic is alpha=mu*x/P, not mu*q*x/P.
9. Final claims should be based on this D output, not on B baselines or C
   training summaries.

Outputs
-------
outputs/results/benchmark_full_results.csv
outputs/results/kkt_equalization_diagnostic.csv
outputs/results/benchmark_pairwise_differences.csv
outputs/results/benchmark_manifest.json
outputs/results/benchmark_q_inventory.csv
outputs/results/benchmark_q_audit.csv
outputs/results/benchmark_q_validation.csv
outputs/results/solutions/
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Path setup and imports
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else Path.cwd()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import pipeline_config as cfg  # type: ignore
except Exception:  # pragma: no cover
    cfg = None

try:
    import B_run_annual_lp as B  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "D_run_full_benchmark.py requires scripts/B_run_annual_lp.py to be importable. "
        "Keep D in the scripts/ folder and run from the repository root."
    ) from exc


# =============================================================================
# Configuration helpers
# =============================================================================

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
# Global configuration
# =============================================================================

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

DERIVED_DIR = Path(os.getenv(
    "DERIVED_DIR",
    str(cfg_get("DERIVED_DIR", REPO_ROOT / "data" / "derived")),
))

RESULTS_DIR = Path(os.getenv(
    "DEBIT_RESULTS_DIR",
    str(cfg_get("RESULTS_DIR", REPO_ROOT / "outputs" / "results")),
))

LOG_DIR = Path(os.getenv(
    "LOG_DIR",
    str(cfg_get("LOG_DIR", REPO_ROOT / "outputs" / "logs")),
))

INPUT_DIR = Path(os.getenv(
    "DEBIT_INPUT_DIR",
    str(cfg_get("DEBIT_OPT_INPUT_DIR", DERIVED_DIR / "debit_optimization_inputs")),
))

M1_ONLY_RESULTS_DIR = Path(os.getenv(
    "M1_ONLY_RESULTS_DIR",
    str(RESULTS_DIR / "m1_only"),
))

M1_ONLY_SOLUTION_DIR = Path(os.getenv(
    "M1_ONLY_SOLUTION_DIR",
    str(M1_ONLY_RESULTS_DIR / "solutions"),
))

SOLUTION_DIR = Path(os.getenv(
    "DEBIT_SOLUTION_DIR",
    str(RESULTS_DIR / "solutions"),
))

M3_WEIGHTS_PATH = Path(os.getenv(
    "M3_Q_FILE",
    str(cfg_get("M3_WEIGHTS_PATH", DERIVED_DIR / "m3_debit_weights_by_year.parquet")),
))

M3_WEIGHTS_WIDE_PATH = Path(os.getenv(
    "M3_Q_WIDE_FILE",
    str(cfg_get("M3_WEIGHTS_WIDE_PATH", DERIVED_DIR / "m3_debit_weights_wide_by_year.parquet")),
))

C_TRAINING_MANIFEST_PATH = Path(os.getenv(
    "C_TRAINING_MANIFEST_PATH",
    str(cfg_get("M3_TRAINING_MANIFEST_PATH", DERIVED_DIR / "m3_training_manifest.json")),
))

YEARS = env_int_list(
    "DEBIT_YEARS",
    cfg_get("ALL_YEARS", cfg_get("EVAL_YEARS", [2021, 2022, 2023, 2024, 2025])),
)

PROVINCES = list(cfg_get("NL_PROVINCES", DEFAULT_PROVINCES))

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

PRIMARY_BURDEN = str(os.getenv("PRIMARY_BURDEN", str(cfg_get("PRIMARY_BURDEN", "burden_proxy_norm")))).strip()

RUN_M2 = env_bool("D_RUN_M2", True)
RUN_MAIN_M3 = env_bool("D_RUN_MAIN_M3", True)
IMPORT_B1_M1 = env_bool("D_IMPORT_B1_M1", True)
RUN_VARIANT_COMPARISON = env_bool("D_RUN_VARIANT_COMPARISON", True)
RUN_FEATURE_LADDER = env_bool("D_RUN_FEATURE_LADDER", True)

SAVE_MAIN_SOLUTIONS = env_bool("D_SAVE_MAIN_SOLUTIONS", True)
SAVE_OPTIONAL_SOLUTIONS = env_bool("D_SAVE_OPTIONAL_SOLUTIONS", False)

# Strictness flags. Defaults are intentionally paper-grade.
D_REQUIRE_M1 = env_bool("D_REQUIRE_M1", True)
D_REQUIRE_M2 = env_bool("D_REQUIRE_M2", True)
D_REQUIRE_MAIN_M3 = env_bool("D_REQUIRE_MAIN_M3", True)
D_REQUIRE_C_MANIFEST = env_bool("D_REQUIRE_C_MANIFEST", True)
D_REQUIRE_MAIN_M3_ALPHA = env_bool("D_REQUIRE_MAIN_M3_ALPHA", True)
D_STRICT_OPTIONAL_Q_BLOCKS = env_bool("D_STRICT_OPTIONAL_Q_BLOCKS", False)
D_FAIL_ON_MAIN_SOLVE_ERROR = env_bool("D_FAIL_ON_MAIN_SOLVE_ERROR", True)

Q_MEAN_TOL = float(os.getenv("D_Q_ACTIVE_MEAN_TOL", "1e-5"))
OFF_SUPPORT_Q_TOL = float(os.getenv("D_OFF_SUPPORT_Q_TOL", "1e-9"))
Q_POSITIVE_TOL = float(os.getenv("D_Q_POSITIVE_TOL", "1e-12"))

GUROBI_OUTPUT_FLAG = int(os.getenv("GUROBI_OUTPUT_FLAG_D", os.getenv("GUROBI_OUTPUT_FLAG", "0")))

TOUCH_TOL = float(os.getenv("TOUCH_TOL", "1e-7"))
POSITIVE_SLACK_TOL = float(os.getenv("POSITIVE_SLACK_TOL", "1e-7"))
BUDGET_BINDING_TOL = float(os.getenv("BUDGET_BINDING_TOL", "1e-5"))

BENCH_PATH = RESULTS_DIR / "benchmark_full_results.csv"
KKT_PATH = RESULTS_DIR / "kkt_equalization_diagnostic.csv"
PAIRWISE_PATH = RESULTS_DIR / "benchmark_pairwise_differences.csv"
MANIFEST_PATH = RESULTS_DIR / "benchmark_manifest.json"
Q_INVENTORY_PATH = RESULTS_DIR / "benchmark_q_inventory.csv"
Q_AUDIT_PATH = RESULTS_DIR / "benchmark_q_audit.csv"
Q_VALIDATION_PATH = RESULTS_DIR / "benchmark_q_validation.csv"


# =============================================================================
# Small utilities
# =============================================================================

def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTION_DIR.mkdir(parents=True, exist_ok=True)


def read_table(path_or_stem: Path, required: bool = True) -> Optional[pd.DataFrame]:
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


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path}")


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print(f"[SAVE] {path}")


def first_existing(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lower = {str(c).strip().lower(): str(c) for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        lc = str(c).strip().lower()
        if lc in lower:
            return lower[lc]
    if required:
        raise KeyError(
            f"Could not find any of {list(candidates)}. Available columns={list(df.columns)}"
        )
    return None


def canonical_variant(v: Any) -> str:
    s = str(v).strip()
    t = s.lower().replace("-", "_").replace(" ", "_").replace("__", "_").strip("_")

    if t in {"plain", "m3_plain", "q_m3_plain", "proxy", "proxy_plain"}:
        return "M3_plain"
    if t in {"oi", "m3_oi", "q_m3_oi", "one_shot", "oneshot", "optimization_informed", "opt_informed"}:
        return "M3_OI"
    if t in {"lo", "m3_lo", "q_m3_lo", "q_m3", "learn_optimize", "learn_opt", "alternating"}:
        return "M3_LO"

    return s


def canonical_feature(f: Any) -> str:
    s = str(f).strip().upper()
    s = s.replace("FEATURE_", "").replace("FEATURE", "").replace("LEVEL_", "").replace("LEVEL", "")
    s = s.replace("-", "_").replace(" ", "")
    if s in {"0", "_0"}:
        return "F0"
    if s in {"1", "_1"}:
        return "F1"
    if s in {"2", "_2"}:
        return "F2"
    if s in {"3", "_3"}:
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


def hours_in_year_utc(year: int) -> int:
    t0 = pd.Timestamp(f"{int(year)}-01-01T00:00:00Z")
    t1 = pd.Timestamp(f"{int(year) + 1}-01-01T00:00:00Z")
    return int((t1 - t0) / pd.Timedelta(hours=1))


def standardize_year_hour_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "year" not in df.columns:
        ts_col = first_existing(df, ["timestamp_utc", "timestamp", "datetime_utc", "time"], required=False)
        if ts_col is None:
            raise KeyError("Table must contain either year or timestamp_utc/timestamp.")
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df["year"] = ts.dt.year

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    ts_col = first_existing(df, ["timestamp_utc", "timestamp", "datetime_utc", "time"], required=False)
    if ts_col is not None:
        if ts_col != "timestamp_utc":
            df = df.rename(columns={ts_col: "timestamp_utc"})
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True).dt.floor("h")

    h_col = first_existing(df, ["hour_of_year", "hour", "hoy", "h", "hour_index", "time_index"], required=False)
    if h_col is not None and h_col != "hour_of_year":
        df = df.rename(columns={h_col: "hour_of_year"})

    if "hour_of_year" not in df.columns:
        if "timestamp_utc" not in df.columns:
            raise KeyError("Need hour_of_year or timestamp_utc to standardize annual grid.")
        jan1 = pd.to_datetime(df["year"].astype(str) + "-01-01T00:00:00Z", utc=True)
        df["hour_of_year"] = ((df["timestamp_utc"] - jan1) / pd.Timedelta(hours=1)).round().astype(int) + 1

    df["hour_of_year"] = pd.to_numeric(df["hour_of_year"], errors="raise").astype(int)

    pieces: List[pd.DataFrame] = []
    for year, g in df.groupby("year", sort=False):
        gg = g.copy()
        H = hours_in_year_utc(int(year))
        hmin = int(gg["hour_of_year"].min())
        hmax = int(gg["hour_of_year"].max())

        if hmin == 0 and hmax <= H - 1:
            gg["hour_of_year"] = gg["hour_of_year"] + 1

        if int(gg["hour_of_year"].min()) < 1 or int(gg["hour_of_year"].max()) > H:
            raise ValueError(
                f"Invalid hour_of_year range for year={year}: "
                f"min={gg['hour_of_year'].min()}, max={gg['hour_of_year'].max()}, H={H}"
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


def read_json_if_exists(path: Path, required: bool = False) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required JSON file: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if required:
            raise RuntimeError(f"Could not parse JSON file {path}: {exc}") from exc
        warnings.warn(f"Could not parse JSON file {path}: {exc}")
        return {}


# =============================================================================
# C manifest and split-role audit
# =============================================================================

def split_roles_from_c_manifest(c_manifest: Mapping[str, Any]) -> Dict[int, str]:
    roles: Dict[int, str] = {}
    years_obj = c_manifest.get("years", {}) if isinstance(c_manifest, Mapping) else {}
    if isinstance(years_obj, Mapping):
        for y in years_obj.get("train_years", []) or []:
            roles[int(y)] = "train"
        val = years_obj.get("validation_year", None)
        if val is not None and str(val) != "":
            roles[int(val)] = "validation"
        test = years_obj.get("test_year", None)
        if test is not None and str(test) != "":
            roles[int(test)] = "test"
        for y in years_obj.get("apply_years", []) or []:
            roles.setdefault(int(y), "apply")
    return roles


def split_role_for_year(year: int, weights: pd.DataFrame, role_map: Mapping[int, str]) -> str:
    if int(year) in role_map:
        return str(role_map[int(year)])
    if "split_role" in weights.columns:
        vals = weights.loc[weights["year"].astype(int) == int(year), "split_role"].dropna().astype(str).unique()
        if len(vals) == 1:
            return str(vals[0])
        if len(vals) > 1:
            return "mixed"
    return "unclassified"


def find_main_training_meta(c_manifest: Mapping[str, Any], feature_set: str, variant: str) -> Dict[str, Any]:
    f = canonical_feature(feature_set)
    v = canonical_variant(variant)
    metas = c_manifest.get("training_meta", []) if isinstance(c_manifest, Mapping) else []
    if not isinstance(metas, list):
        return {}

    matches: List[Dict[str, Any]] = []
    for obj in metas:
        if not isinstance(obj, Mapping):
            continue
        if canonical_feature(obj.get("feature_set", "")) == f and canonical_variant(obj.get("variant", "")) == v:
            matches.append(dict(obj))

    if not matches:
        return {}

    # Prefer the last match because LO/OI metadata may be appended after plain.
    return matches[-1]


def alpha_available_from_meta(meta: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Return whether OI/LO metadata confirms a usable alpha signal."""
    if not meta:
        return False, {"reason": "missing_training_meta"}

    if "alpha_meta" in meta and isinstance(meta["alpha_meta"], Mapping):
        am = dict(meta["alpha_meta"])
        return bool(am.get("alpha_available", False)), am

    iterations = meta.get("iterations", [])
    if isinstance(iterations, list) and iterations:
        # For LO, require the final iteration to have alpha_available=True.
        last = iterations[-1]
        if isinstance(last, Mapping) and isinstance(last.get("alpha_meta"), Mapping):
            am = dict(last["alpha_meta"])
            return bool(am.get("alpha_available", False)), am

    return False, {"reason": "no_alpha_meta_in_training_meta"}


def validate_c_manifest(c_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    if not c_manifest:
        if D_REQUIRE_C_MANIFEST:
            raise FileNotFoundError(
                f"C manifest is required but was not found or could not be read: {C_TRAINING_MANIFEST_PATH}"
            )
        return {"available": False}

    out: Dict[str, Any] = {"available": True}
    main_cfg = c_manifest.get("main_configuration", {}) if isinstance(c_manifest, Mapping) else {}
    if isinstance(main_cfg, Mapping):
        c_main_f = canonical_feature(main_cfg.get("main_feature_set", ""))
        c_main_v = canonical_variant(main_cfg.get("main_variant", ""))
        out["c_main_feature_set"] = c_main_f
        out["c_main_variant"] = c_main_v
        if c_main_f and c_main_f != canonical_feature(MAIN_FEATURE_SET):
            warnings.warn(
                f"D main feature set {canonical_feature(MAIN_FEATURE_SET)} differs from C manifest main feature {c_main_f}. "
                "This is allowed only if you intentionally override the main benchmark."
            )
        if c_main_v and c_main_v != canonical_variant(MAIN_M3_VARIANT):
            warnings.warn(
                f"D main variant {canonical_variant(MAIN_M3_VARIANT)} differs from C manifest main variant {c_main_v}. "
                "This is allowed only if you intentionally override the main benchmark."
            )

    main_meta = find_main_training_meta(c_manifest, MAIN_FEATURE_SET, MAIN_M3_VARIANT)
    out["main_training_meta_found"] = bool(main_meta)
    out["main_training_meta"] = main_meta

    main_variant = canonical_variant(MAIN_M3_VARIANT)
    if main_variant in {"M3_OI", "M3_LO"}:
        ok, alpha_meta = alpha_available_from_meta(main_meta)
        out["main_alpha_available"] = bool(ok)
        out["main_alpha_meta"] = alpha_meta
        if D_REQUIRE_MAIN_M3_ALPHA and not ok:
            raise RuntimeError(
                "Main M3 variant is optimization-informed but C manifest does not confirm usable alpha.\n"
                f"main_feature_set={canonical_feature(MAIN_FEATURE_SET)}, main_variant={main_variant}\n"
                f"alpha_meta={alpha_meta}\n"
                "Rerun the signed-off C_train_m3.py after the latest B fix, or deliberately set "
                "D_REQUIRE_MAIN_M3_ALPHA=0 only for a proxy-only sensitivity run."
            )
    else:
        out["main_alpha_available"] = None
        out["main_alpha_meta"] = None

    return out


# =============================================================================
# M3 weights loading and validation
# =============================================================================

def load_m3_weights() -> pd.DataFrame:
    if M3_WEIGHTS_PATH.exists():
        raw = read_table(M3_WEIGHTS_PATH, required=True)
        source = str(M3_WEIGHTS_PATH)
    elif M3_WEIGHTS_WIDE_PATH.exists():
        raw = read_table(M3_WEIGHTS_WIDE_PATH, required=True)
        source = str(M3_WEIGHTS_WIDE_PATH)
    else:
        raise FileNotFoundError(
            "Missing M3 weights. Run C_train_m3.py first.\n"
            f"Tried:\n  {M3_WEIGHTS_PATH}\n  {M3_WEIGHTS_WIDE_PATH}"
        )

    assert raw is not None
    raw = raw.copy()
    raw["_weights_source_file"] = source

    pcol = first_existing(raw, ["province", "provincie", "deployment_unit", "node", "area"], required=True)
    if pcol != "province":
        raw = raw.rename(columns={pcol: "province"})
    raw["province"] = raw["province"].astype(str).str.strip()

    raw = standardize_year_hour_timestamp(raw)

    if "feature_set" not in raw.columns:
        if "feature_level" in raw.columns:
            raw["feature_set"] = raw["feature_level"]
        else:
            raw["feature_set"] = MAIN_FEATURE_SET

    raw["feature_set"] = raw["feature_set"].map(canonical_feature)

    if "feature_level" not in raw.columns:
        raw["feature_level"] = raw["feature_set"]
    raw["feature_level"] = raw["feature_level"].map(canonical_feature)

    optional_cols = []
    for c in ["active_support", "split_role", "architecture", "_weights_source_file"]:
        if c in raw.columns:
            optional_cols.append(c)

    if "debit_weight" in raw.columns:
        if "variant" not in raw.columns:
            if "architecture" in raw.columns:
                raw["variant"] = raw["architecture"]
            else:
                raw["variant"] = MAIN_M3_VARIANT

        base_cols = [
            "year", "province", "timestamp_utc", "hour_of_year",
            "feature_set", "feature_level", "variant", "debit_weight",
        ] + [c for c in optional_cols if c not in {"architecture"}]
        out = raw[base_cols].copy()

    else:
        q_cols: Dict[str, str] = {}
        lower_cols = {str(c).lower(): c for c in raw.columns}
        for canonical_col, variant in [
            ("q_m3_plain", "M3_plain"),
            ("q_m3_oi", "M3_OI"),
            ("q_m3_lo", "M3_LO"),
        ]:
            if canonical_col in lower_cols:
                q_cols[str(lower_cols[canonical_col])] = variant
        if "q_m3" in lower_cols and all(c.lower() != "q_m3_lo" for c in q_cols):
            q_cols[str(lower_cols["q_m3"])] = "M3_LO"

        if not q_cols:
            raise KeyError(
                "M3 weights file must contain either long-format debit_weight "
                "or wide-format q_m3_plain/q_m3_OI/q_m3_LO/q_m3 columns. "
                f"Available columns={list(raw.columns)}"
            )

        base_cols = ["year", "province", "timestamp_utc", "hour_of_year", "feature_set", "feature_level"]
        base_cols += [c for c in optional_cols if c not in base_cols]
        pieces = []
        for q_col, variant in q_cols.items():
            tmp = raw[base_cols + [q_col]].copy()
            tmp = tmp.rename(columns={q_col: "debit_weight"})
            tmp["variant"] = variant
            pieces.append(tmp)

        out = pd.concat(pieces, ignore_index=True)

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["province"] = out["province"].astype(str).str.strip()
    out["feature_set"] = out["feature_set"].map(canonical_feature)
    out["feature_level"] = out["feature_level"].map(canonical_feature)
    out["variant"] = out["variant"].map(canonical_variant)
    out["debit_weight"] = pd.to_numeric(out["debit_weight"], errors="coerce")

    if "active_support" in out.columns:
        out["active_support"] = pd.to_numeric(out["active_support"], errors="coerce").fillna(0).astype(int)
    if "split_role" in out.columns:
        out["split_role"] = out["split_role"].astype(str)
    else:
        out["split_role"] = "unclassified"

    if out.empty:
        raise ValueError("M3 weights are empty.")

    if out["debit_weight"].isna().any():
        bad = int(out["debit_weight"].isna().sum())
        raise ValueError(f"M3 weights contain {bad} missing debit_weight values.")

    if (out["debit_weight"] <= Q_POSITIVE_TOL).any():
        bad = int((out["debit_weight"] <= Q_POSITIVE_TOL).sum())
        raise ValueError(f"M3 weights contain {bad} nonpositive or near-zero debit_weight values.")

    key = ["year", "province", "hour_of_year", "feature_set", "variant"]
    dup = out.duplicated(key, keep=False)
    if dup.any():
        sample = out.loc[dup, key + ["debit_weight"]].head(20)
        raise ValueError(
            "Duplicate M3 weights for same year/province/hour/feature/variant.\n"
            + sample.to_string(index=False)
        )

    return out.sort_values(
        ["year", "feature_set", "variant", "province", "hour_of_year"]
    ).reset_index(drop=True)


def q_inventory(weights: pd.DataFrame) -> pd.DataFrame:
    agg = weights.groupby(["year", "feature_set", "variant"], observed=True).agg(
        n_rows=("debit_weight", "size"),
        n_provinces=("province", "nunique"),
        n_hours=("hour_of_year", "nunique"),
        q_mean=("debit_weight", "mean"),
        q_std=("debit_weight", "std"),
        q_min=("debit_weight", "min"),
        q_p01=("debit_weight", lambda s: float(np.quantile(s, 0.01))),
        q_p50=("debit_weight", "median"),
        q_p99=("debit_weight", lambda s: float(np.quantile(s, 0.99))),
        q_max=("debit_weight", "max"),
        q_nontrivial_share=("debit_weight", lambda s: float(np.mean(np.abs(np.asarray(s, dtype=float) - 1.0) > 1e-6))),
    ).reset_index()

    if "active_support" in weights.columns:
        active = weights[weights["active_support"].astype(int) == 1].copy()
        if not active.empty:
            agg_active = active.groupby(["year", "feature_set", "variant"], observed=True).agg(
                active_rows=("debit_weight", "size"),
                active_q_mean=("debit_weight", "mean"),
                active_q_std=("debit_weight", "std"),
                active_q_min=("debit_weight", "min"),
                active_q_max=("debit_weight", "max"),
            ).reset_index()
            agg = agg.merge(agg_active, on=["year", "feature_set", "variant"], how="left")

    return agg.sort_values(["year", "feature_set", "variant"]).reset_index(drop=True)


def q_audit(weights: pd.DataFrame) -> pd.DataFrame:
    work = weights.copy()
    if "active_support" in work.columns:
        work["audit_support"] = work["active_support"].astype(int)
    else:
        work["audit_support"] = 1

    rows = []
    for support_label, sub in [("all_rows", work), ("active_support", work[work["audit_support"] == 1])]:
        if sub.empty:
            continue
        tmp = sub.groupby(["year", "feature_set", "variant", "province"], observed=True).agg(
            n_rows=("debit_weight", "size"),
            q_mean=("debit_weight", "mean"),
            q_std=("debit_weight", "std"),
            q_min=("debit_weight", "min"),
            q_p01=("debit_weight", lambda s: float(np.quantile(s, 0.01))),
            q_p50=("debit_weight", "median"),
            q_p99=("debit_weight", lambda s: float(np.quantile(s, 0.99))),
            q_max=("debit_weight", "max"),
            q_nontrivial_share=("debit_weight", lambda s: float(np.mean(np.abs(np.asarray(s, dtype=float) - 1.0) > 1e-6))),
        ).reset_index()
        tmp["support_scope"] = support_label
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True).sort_values(
        ["support_scope", "year", "feature_set", "variant", "province"]
    ).reset_index(drop=True)


def has_weight_block(weights: pd.DataFrame, year: int, feature_set: str, variant: str) -> bool:
    fs = canonical_feature(feature_set)
    var = canonical_variant(variant)
    sub = weights[
        (weights["year"].astype(int) == int(year))
        & (weights["feature_set"].astype(str).map(canonical_feature) == fs)
        & (weights["variant"].map(canonical_variant) == var)
    ]
    return not sub.empty


def raw_weight_block(weights: pd.DataFrame, year: int, feature_set: str, variant: str) -> pd.DataFrame:
    fs = canonical_feature(feature_set)
    var = canonical_variant(variant)
    sub = weights[
        (weights["year"].astype(int) == int(year))
        & (weights["feature_set"].astype(str).map(canonical_feature) == fs)
        & (weights["variant"].map(canonical_variant) == var)
    ].copy()

    if sub.empty:
        raise ValueError(f"Missing M3 q block for year={year}, feature_set={fs}, variant={var}.")

    return sub.copy()


def active_support_from_state(state: pd.DataFrame) -> pd.Series:
    if "tdtr_support" in state.columns:
        support = pd.to_numeric(state["tdtr_support"], errors="coerce").fillna(0.0).astype(float) > 0.5
    else:
        support = pd.Series(True, index=state.index)
    pbar = pd.to_numeric(state.get("pbar_mw", 0.0), errors="coerce").fillna(0.0).astype(float)
    return support & (pbar > TOUCH_TOL)


def validate_weight_block_against_year(
    *,
    weights: pd.DataFrame,
    yi: Any,
    feature_set: str,
    variant: str,
    comparison_group: str,
    enforce: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate a year-feature-variant q block against the exact annual state grid."""
    year = int(yi.year)
    fs = canonical_feature(feature_set)
    var = canonical_variant(variant)
    block = raw_weight_block(weights, year, fs, var)

    state = yi.state[["province", "hour_of_year"]].copy()
    state["province"] = state["province"].astype(str).str.strip()
    state["hour_of_year"] = pd.to_numeric(state["hour_of_year"], errors="raise").astype(int)
    state["state_active_support"] = active_support_from_state(yi.state).astype(int).to_numpy()

    block["province"] = block["province"].astype(str).str.strip()
    block["hour_of_year"] = pd.to_numeric(block["hour_of_year"], errors="raise").astype(int)
    block["debit_weight"] = pd.to_numeric(block["debit_weight"], errors="coerce").astype(float)

    key = ["province", "hour_of_year"]
    if block.duplicated(key).any():
        dup = int(block.duplicated(key).sum())
        msg = f"q block has {dup} duplicate province-hour keys for year={year}, feature={fs}, variant={var}."
        if enforce:
            raise ValueError(msg)
        warnings.warn(msg)
        block = block.groupby(key, as_index=False).agg(debit_weight=("debit_weight", "mean"))

    merged = state.merge(block[key + ["debit_weight"]], on=key, how="left", validate="one_to_one")
    missing = int(merged["debit_weight"].isna().sum())

    extra = block.merge(state[key], on=key, how="left", indicator=True)
    n_extra = int((extra["_merge"] == "left_only").sum())

    active = merged["state_active_support"].astype(int) == 1
    off = ~active

    if missing > 0:
        sample = merged.loc[merged["debit_weight"].isna(), key].head(10).to_dict("records")
        msg = (
            f"Missing q weights for {missing} state province-hours in year={year}, feature={fs}, variant={var}. "
            f"No q=1 fallback is allowed. Sample={sample}"
        )
        if enforce:
            raise ValueError(msg)
        warnings.warn(msg)

    if n_extra > 0:
        sample = extra.loc[extra["_merge"] == "left_only", key].head(10).to_dict("records")
        msg = f"q block has {n_extra} province-hour keys not present in state for year={year}, feature={fs}, variant={var}. Sample={sample}"
        if enforce:
            raise ValueError(msg)
        warnings.warn(msg)

    if merged["debit_weight"].notna().any() and (merged["debit_weight"].dropna() <= Q_POSITIVE_TOL).any():
        msg = f"q block contains nonpositive weights for year={year}, feature={fs}, variant={var}."
        if enforce:
            raise ValueError(msg)
        warnings.warn(msg)

    active_sub = merged.loc[active & merged["debit_weight"].notna()].copy()
    off_sub = merged.loc[off & merged["debit_weight"].notna()].copy()

    if active_sub.empty:
        active_mean_max_abs = np.nan
        active_mean_min = np.nan
        active_mean_max = np.nan
    else:
        prov_means = active_sub.groupby("province")["debit_weight"].mean()
        active_mean_max_abs = float(np.max(np.abs(prov_means.to_numpy(dtype=float) - 1.0)))
        active_mean_min = float(prov_means.min())
        active_mean_max = float(prov_means.max())
        if active_mean_max_abs > Q_MEAN_TOL:
            msg = (
                f"Active-support province mean q is not one within tolerance for year={year}, feature={fs}, variant={var}: "
                f"max_abs_mean_minus_one={active_mean_max_abs:.3e}, tol={Q_MEAN_TOL:.3e}."
            )
            if enforce:
                raise ValueError(msg)
            warnings.warn(msg)

    if off_sub.empty:
        off_support_max_abs_q_minus_one = np.nan
    else:
        off_support_max_abs_q_minus_one = float(np.max(np.abs(off_sub["debit_weight"].to_numpy(dtype=float) - 1.0)))
        if off_support_max_abs_q_minus_one > OFF_SUPPORT_Q_TOL:
            msg = (
                f"Off-support q differs from one for year={year}, feature={fs}, variant={var}: "
                f"max_abs_q_minus_one={off_support_max_abs_q_minus_one:.3e}, tol={OFF_SUPPORT_Q_TOL:.3e}."
            )
            if enforce:
                raise ValueError(msg)
            warnings.warn(msg)

    audit = {
        "year": year,
        "comparison_group": comparison_group,
        "feature_set": fs,
        "variant": var,
        "n_state_rows": int(len(state)),
        "n_q_rows": int(len(block)),
        "n_missing_state_q": missing,
        "n_extra_q_keys": n_extra,
        "n_active_state_rows": int(active.sum()),
        "n_off_support_state_rows": int(off.sum()),
        "active_q_mean_min_by_province": active_mean_min,
        "active_q_mean_max_by_province": active_mean_max,
        "active_q_max_abs_province_mean_minus_one": active_mean_max_abs,
        "off_support_max_abs_q_minus_one": off_support_max_abs_q_minus_one,
        "q_min_on_state": float(merged["debit_weight"].min(skipna=True)) if merged["debit_weight"].notna().any() else np.nan,
        "q_mean_on_state": float(merged["debit_weight"].mean(skipna=True)) if merged["debit_weight"].notna().any() else np.nan,
        "q_max_on_state": float(merged["debit_weight"].max(skipna=True)) if merged["debit_weight"].notna().any() else np.nan,
        "enforced": bool(enforce),
    }

    q_for_b = merged[key + ["debit_weight"]].copy()
    if q_for_b["debit_weight"].isna().any():
        # This path only occurs when enforce=False. We still do not solve with
        # filled q. The caller should skip the optional block.
        raise ValueError(
            f"Cannot solve with incomplete q block for year={year}, feature={fs}, variant={var}."
        )

    return q_for_b, audit


# =============================================================================
# Metric helpers
# =============================================================================

def choose_proxy_column(state: pd.DataFrame) -> Optional[str]:
    candidates = [
        PRIMARY_BURDEN,
        "burden_proxy_norm",
        "B_proxy_baseline",
        "B_proxy",
        "burden_proxy",
        "relief_value_proxy_norm",
        "relief_proxy_baseline",
        "price_da_norm",
        "ptdf_effectivity_norm",
        "ptdf_requirement_exposure_MW",
        "structural_exposure",
    ]

    for c in candidates:
        if c in state.columns:
            return c

    return None


def merge_proxy(x_df: pd.DataFrame, yi: Any) -> Tuple[pd.DataFrame, str]:
    proxy_col = choose_proxy_column(yi.state)
    out = x_df.copy()

    if proxy_col is None:
        out["benchmark_proxy"] = 0.0
        return out, "none_zero_proxy"

    if "province" not in out.columns:
        pcol = first_existing(out, ["province", "provincie", "deployment_unit", "node"], required=True)
        out = out.rename(columns={pcol: "province"})

    if "hour_of_year" not in out.columns:
        hcol = first_existing(out, ["hour_of_year", "hour", "hoy", "h"], required=False)
        if hcol is not None:
            out = out.rename(columns={hcol: "hour_of_year"})
        elif "timestamp_utc" in out.columns:
            tmp_ts = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True).dt.floor("h")
            jan1 = pd.Timestamp(f"{int(yi.year)}-01-01T00:00:00Z")
            out["hour_of_year"] = ((tmp_ts - jan1) / pd.Timedelta(hours=1)).round().astype(int) + 1
        else:
            raise KeyError("Solution table must contain hour_of_year or timestamp_utc.")

    out["province"] = out["province"].astype(str).str.strip()
    out["hour_of_year"] = pd.to_numeric(out["hour_of_year"], errors="raise").astype(int)

    st = yi.state[["province", "hour_of_year", proxy_col]].copy()
    st["province"] = st["province"].astype(str).str.strip()
    st["hour_of_year"] = pd.to_numeric(st["hour_of_year"], errors="raise").astype(int)
    st["_benchmark_proxy_from_state"] = (
        pd.to_numeric(st[proxy_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    st = st[["province", "hour_of_year", "_benchmark_proxy_from_state"]].drop_duplicates(
        ["province", "hour_of_year"], keep="first"
    )

    if "_benchmark_proxy_from_state" in out.columns:
        out = out.drop(columns=["_benchmark_proxy_from_state"])

    out = out.merge(st, on=["province", "hour_of_year"], how="left", validate="many_to_one")

    missing = int(out["_benchmark_proxy_from_state"].isna().sum())
    if missing > 0:
        warnings.warn(
            f"Proxy merge produced {missing:,} missing province-hour proxy values "
            f"for proxy_col={proxy_col!r}; filling them with zero."
        )

    out["benchmark_proxy"] = (
        pd.to_numeric(out["_benchmark_proxy_from_state"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    out = out.drop(columns=["_benchmark_proxy_from_state"])
    return out, proxy_col


def concentration(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    v = np.maximum(v, 0.0)

    total = float(v.sum())
    if total <= 1e-12 or len(v) == 0:
        return {"gini": 0.0, "hhi": 0.0, "top10_share": 0.0}

    s = np.sort(v)
    n = len(s)
    shares = s / total

    gini = float(2.0 * np.sum(np.arange(1, n + 1) * s) / (n * total) - (n + 1) / n)
    hhi = float(np.sum(shares ** 2))

    k = max(1, int(math.ceil(0.10 * n)))
    top10_share = float(np.sum(s[-k:]) / total)

    return {"gini": gini, "hhi": hhi, "top10_share": top10_share}


def q_stats_from_x(x: pd.DataFrame) -> Dict[str, float]:
    nan_stats = {
        "q_mean": np.nan,
        "q_std": np.nan,
        "q_min": np.nan,
        "q_p01": np.nan,
        "q_p05": np.nan,
        "q_p50": np.nan,
        "q_p95": np.nan,
        "q_p99": np.nan,
        "q_max": np.nan,
        "q_nontrivial_share": np.nan,
        "q_province_mean_min": np.nan,
        "q_province_mean_max": np.nan,
        "q_province_mean_std": np.nan,
    }
    if "debit_weight" not in x.columns:
        return nan_stats

    q = pd.to_numeric(x["debit_weight"], errors="coerce")
    q = q[np.isfinite(q)]

    if q.empty:
        return nan_stats

    prov_mean = x.assign(_q=pd.to_numeric(x["debit_weight"], errors="coerce")).groupby("province")['_q'].mean()

    return {
        "q_mean": float(q.mean()),
        "q_std": float(q.std(ddof=0)),
        "q_min": float(q.min()),
        "q_p01": float(q.quantile(0.01)),
        "q_p05": float(q.quantile(0.05)),
        "q_p50": float(q.quantile(0.50)),
        "q_p95": float(q.quantile(0.95)),
        "q_p99": float(q.quantile(0.99)),
        "q_max": float(q.max()),
        "q_nontrivial_share": float(np.mean(np.abs(q.to_numpy(dtype=float) - 1.0) > 1e-6)),
        "q_province_mean_min": float(prov_mean.min()),
        "q_province_mean_max": float(prov_mean.max()),
        "q_province_mean_std": float(prov_mean.std(ddof=0)),
    }


def m2_m3_result_to_benchmark_row(
    *,
    result: Any,
    yi: Any,
    comparison_group: str,
    display_rule: str,
    feature_set: Optional[str],
    variant: Optional[str],
    split_role: str,
) -> Dict[str, Any]:
    x, proxy_col = merge_proxy(result.x, yi)

    x["x_mw"] = pd.to_numeric(x["x_mw"], errors="coerce").fillna(0.0).astype(float)
    x["Pctr_mw"] = pd.to_numeric(x["Pctr_mw"], errors="coerce").replace(0, np.nan).astype(float)
    x["physical_depth"] = (x["x_mw"] / x["Pctr_mw"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "debit" not in x.columns:
        if "debit_weight" in x.columns:
            x["debit"] = pd.to_numeric(x["debit_weight"], errors="coerce").fillna(1.0) * x["physical_depth"]
        else:
            x["debit"] = x["physical_depth"]
    x["debit"] = pd.to_numeric(x["debit"], errors="coerce").fillna(0.0).astype(float)

    by_prov = x.groupby("province", as_index=False).agg(
        physical_depth_total=("physical_depth", "sum"),
        debit_total=("debit", "sum"),
    )

    physical_proxy = float(np.sum(x["benchmark_proxy"].to_numpy(dtype=float) * x["physical_depth"].to_numpy(dtype=float)))
    debit_proxy = float(np.sum(x["benchmark_proxy"].to_numpy(dtype=float) * x["debit"].to_numpy(dtype=float)))

    exp_conc = concentration(by_prov["physical_depth_total"].to_numpy(dtype=float))
    debit_conc = concentration(by_prov["debit_total"].to_numpy(dtype=float))

    mean_budget_util = float(result.budget["budget_utilization"].mean()) if not result.budget.empty else np.nan
    max_budget_util = float(result.budget["budget_utilization"].max()) if not result.budget.empty else np.nan
    min_budget_util = float(result.budget["budget_utilization"].min()) if not result.budget.empty else np.nan
    median_budget_util = float(result.budget["budget_utilization"].median()) if not result.budget.empty else np.nan
    n_budget_binding = int(result.budget["is_budget_binding"].sum()) if "is_budget_binding" in result.budget.columns else 0

    if result.kkt is not None and not result.kkt.empty:
        mu_by_prov = (
            result.kkt[["province", "mu_depth"]]
            .drop_duplicates("province")
            .assign(mu_depth=lambda d: pd.to_numeric(d["mu_depth"], errors="coerce").fillna(0.0))
        )
        mu_vals = mu_by_prov["mu_depth"].to_numpy(dtype=float)
        alpha_source = "alpha_mu_x_over_P" if "alpha_mu_x_over_P" in result.kkt.columns else "alpha"
        alpha_vals = pd.to_numeric(result.kkt.get(alpha_source, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mu_positive_count = int(np.sum(mu_vals > 1e-9))
        mu_positive_share = float(np.mean(mu_vals > 1e-9)) if len(mu_vals) else np.nan
        mu_mean = float(np.mean(mu_vals)) if len(mu_vals) else np.nan
        mu_max = float(np.max(mu_vals)) if len(mu_vals) else np.nan
        scarcity_alpha_sum = float(np.sum(alpha_vals)) if len(alpha_vals) else 0.0
        scarcity_alpha_mean = float(np.mean(alpha_vals)) if len(alpha_vals) else 0.0
        scarcity_alpha_max = float(np.max(alpha_vals)) if len(alpha_vals) else 0.0
        scarcity_active_share = float(np.mean(alpha_vals > 1e-9)) if len(alpha_vals) else 0.0
    else:
        mu_positive_count = np.nan
        mu_positive_share = np.nan
        mu_mean = np.nan
        mu_max = np.nan
        scarcity_alpha_sum = np.nan
        scarcity_alpha_mean = np.nan
        scarcity_alpha_max = np.nan
        scarcity_active_share = np.nan

    q_stats = q_stats_from_x(x)

    return {
        "year": result.year,
        "split_role": split_role,
        "comparison_group": comparison_group,
        "rule": display_rule,
        "solver_rule_name": result.rule,
        "feature_set": feature_set,
        "variant": variant,
        "status": result.status,
        "status_code": result.status_code,
        "quality_flag": result.quality_flag,
        "objective": result.objective,
        "cost_total": result.objective,
        "operating_cost_x": result.operating_cost_x,
        "slack_penalty_cost": result.slack_penalty_cost,
        "cost_slack": result.slack_penalty_cost,
        "total_slack": result.total_slack,
        "slack_total": result.total_slack,
        "total_relief_requirement": result.total_relief_requirement,
        "R_total": result.total_relief_requirement,
        "relief_served": result.total_relief_requirement - result.total_slack,
        "relief_served_share": result.relief_served_share,
        "total_debit": result.total_debit,
        "debit_total": result.total_debit,
        "physical_depth_total": float(x["physical_depth"].sum()),
        "allowance_productivity": result.allowance_productivity,
        "mean_budget_utilization": mean_budget_util,
        "median_budget_utilization": median_budget_util,
        "max_budget_utilization": max_budget_util,
        "min_budget_utilization": min_budget_util,
        "budget_util": mean_budget_util,
        "n_budget_binding": n_budget_binding,
        "physical_proxy_exposure": physical_proxy,
        "debit_proxy_exposure": debit_proxy,
        "proxy_column": proxy_col,
        "exposure_gini": exp_conc["gini"],
        "exposure_hhi": exp_conc["hhi"],
        "exposure_top10_share": exp_conc["top10_share"],
        "debit_gini": debit_conc["gini"],
        "debit_hhi": debit_conc["hhi"],
        "debit_top10_share": debit_conc["top10_share"],
        "gini": exp_conc["gini"],
        "hhi": exp_conc["hhi"],
        "top10_share": exp_conc["top10_share"],
        "mu_positive_count": mu_positive_count,
        "mu_positive_share": mu_positive_share,
        "mu_mean": mu_mean,
        "mu_max": mu_max,
        "scarcity_alpha_sum": scarcity_alpha_sum,
        "scarcity_alpha_mean": scarcity_alpha_mean,
        "scarcity_alpha_max": scarcity_alpha_max,
        "scarcity_active_share": scarcity_active_share,
        "runtime_sec": result.runtime_sec,
        "solve_time": result.runtime_sec,
        "mip_gap": result.mip_gap,
        "best_bound": result.best_bound,
        "node_count": result.node_count,
        "solution_count": result.solution_count,
        "model_rows": result.model_rows,
        "model_cols": result.model_cols,
        "model_nnz": result.model_nnz,
        "notes": " | ".join(result.notes),
        **q_stats,
    }


def append_kkt_rows(
    rows: List[pd.DataFrame],
    *,
    result: Any,
    comparison_group: str,
    display_rule: str,
    feature_set: Optional[str],
    variant: Optional[str],
    split_role: str,
) -> None:
    kd = B.kkt_equalization_diagnostic(result)
    if kd is None or kd.empty:
        return

    kd = kd.copy()
    kd["comparison_group"] = comparison_group
    kd["display_rule"] = display_rule
    kd["feature_set"] = feature_set
    kd["variant"] = variant
    kd["split_role"] = split_role

    if "solver_rule" not in kd.columns:
        kd["solver_rule"] = result.rule
    kd["rule"] = display_rule

    # Guard against wrong alpha semantics entering final output.
    if "alpha" in kd.columns and "alpha_mu_x_over_P" in kd.columns:
        diff = pd.to_numeric(kd["alpha"], errors="coerce").fillna(0.0) - pd.to_numeric(kd["alpha_mu_x_over_P"], errors="coerce").fillna(0.0)
        if float(np.max(np.abs(diff.to_numpy(dtype=float)))) > 1e-8:
            raise ValueError("KKT diagnostic alpha differs from alpha_mu_x_over_P. B must keep alpha=mu*x/P.")

    rows.append(kd)


# =============================================================================
# M1 import from B1
# =============================================================================

def load_b1_summary() -> pd.DataFrame:
    path = M1_ONLY_RESULTS_DIR / "m1_only_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing B1 M1 summary:\n  {path}\n"
            "Run scripts/B1_run_m1_touched_only.py first."
        )

    df = pd.read_csv(path, low_memory=False)
    if "year" not in df.columns:
        raise KeyError(f"B1 summary has no year column: {path}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    if "rule" in df.columns:
        df = df[df["rule"].astype(str).str.upper() == "M1"].copy()

    dup = df.duplicated(["year"], keep=False)
    if dup.any():
        sample = df.loc[dup, ["year", "rule", "objective"] if "objective" in df.columns else ["year"]].head(20)
        raise ValueError("B1 M1 summary contains duplicate M1 rows per year.\n" + sample.to_string(index=False))

    return df


def load_m1_solution(year: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_path = M1_ONLY_SOLUTION_DIR / f"M1_province_hour_{year}.csv.gz"
    slack_path = M1_ONLY_SOLUTION_DIR / f"M1_positive_slack_{year}.csv.gz"
    budget_path = M1_ONLY_SOLUTION_DIR / f"M1_budget_summary_{year}.csv"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing B1 M1 province-hour solution: {x_path}")
    if not budget_path.exists():
        raise FileNotFoundError(f"Missing B1 M1 budget summary: {budget_path}")

    x = pd.read_csv(x_path, compression="gzip", low_memory=False)
    budget = pd.read_csv(budget_path, low_memory=False)

    if slack_path.exists():
        slack = pd.read_csv(slack_path, compression="gzip", low_memory=False)
    else:
        slack = pd.DataFrame()

    return x, slack, budget


def m1_benchmark_row(*, yi: Any, b1_summary: pd.DataFrame, split_role: str) -> Dict[str, Any]:
    year = int(yi.year)

    sub = b1_summary[b1_summary["year"].astype(int) == year].copy()
    if sub.empty:
        raise ValueError(f"B1 summary contains no M1 row for year={year}.")

    srow = sub.iloc[0].to_dict()
    x, _, budget = load_m1_solution(year)
    x, proxy_col = merge_proxy(x, yi)

    if "x_mw" not in x.columns:
        xcol = first_existing(x, ["x_mw", "x_MW", "restriction_mw", "x"], required=True)
        x = x.rename(columns={xcol: "x_mw"})

    if "Pctr_mw" not in x.columns:
        pcol = first_existing(x, ["Pctr_mw", "P_ctr_MW", "P_i", "Pctr", "P"], required=False)
        if pcol is not None:
            x = x.rename(columns={pcol: "Pctr_mw"})
        else:
            pmap = yi.participants.set_index("province")["Pctr_mw"].to_dict() if "Pctr_mw" in yi.participants.columns else {}
            if not pmap:
                pcol_part = first_existing(yi.participants, ["Pctr_mw", "P_ctr_MW", "P_i", "P_i_ctr_mw"], required=True)
                pmap = yi.participants.set_index("province")[pcol_part].to_dict()
            x["Pctr_mw"] = x["province"].map(pmap)

    if "debit" not in x.columns:
        if "touch_realized" in x.columns:
            x["debit"] = pd.to_numeric(x["touch_realized"], errors="coerce").fillna(0.0)
        else:
            x["debit"] = (pd.to_numeric(x["x_mw"], errors="coerce").fillna(0.0) > TOUCH_TOL).astype(float)

    x["x_mw"] = pd.to_numeric(x["x_mw"], errors="coerce").fillna(0.0).astype(float)
    x["Pctr_mw"] = pd.to_numeric(x["Pctr_mw"], errors="coerce").replace(0, np.nan).astype(float)
    x["physical_depth"] = (x["x_mw"] / x["Pctr_mw"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x["debit"] = pd.to_numeric(x["debit"], errors="coerce").fillna(0.0).astype(float)

    by_prov = x.groupby("province", as_index=False).agg(
        physical_depth_total=("physical_depth", "sum"),
        debit_total=("debit", "sum"),
    )

    physical_proxy = float(np.sum(x["benchmark_proxy"].to_numpy(dtype=float) * x["physical_depth"].to_numpy(dtype=float)))
    debit_proxy = float(np.sum(x["benchmark_proxy"].to_numpy(dtype=float) * x["debit"].to_numpy(dtype=float)))

    exp_conc = concentration(by_prov["physical_depth_total"].to_numpy(dtype=float))
    debit_conc = concentration(by_prov["debit_total"].to_numpy(dtype=float))

    total_slack = float(srow.get("total_slack", srow.get("slack_total", np.nan)))
    total_R = float(srow.get("total_relief_requirement", srow.get("R_total", yi.requirements["R_mw"].sum())))
    relief_served_share = float(srow.get("relief_served_share", 1.0 - total_slack / max(total_R, 1e-9)))
    relief_served = total_R - total_slack

    total_debit = float(srow.get("total_touched_tokens", srow.get("total_debit", x["debit"].sum())))
    allowance_productivity = float(srow.get("allowance_productivity", relief_served / max(total_debit, 1e-9)))

    mean_budget_util = float(srow.get(
        "mean_budget_utilization",
        budget["budget_utilization"].mean() if "budget_utilization" in budget.columns else np.nan,
    ))
    max_budget_util = float(srow.get(
        "max_budget_utilization",
        budget["budget_utilization"].max() if "budget_utilization" in budget.columns else np.nan,
    ))
    n_budget_binding = int(srow.get(
        "n_budget_binding",
        budget["is_budget_binding"].sum() if "is_budget_binding" in budget.columns else 0,
    ))

    return {
        "year": year,
        "split_role": split_role,
        "comparison_group": "main",
        "rule": "M1",
        "solver_rule_name": "B1_M1",
        "feature_set": None,
        "variant": None,
        "status": srow.get("status", "UNKNOWN"),
        "status_code": srow.get("status_code", np.nan),
        "quality_flag": srow.get("quality_flag", srow.get("algorithm", "B1_imported")),
        "objective": float(srow.get("objective", np.nan)),
        "cost_total": float(srow.get("objective", np.nan)),
        "operating_cost_x": float(srow.get("operating_cost_x", np.nan)),
        "slack_penalty_cost": float(srow.get("slack_penalty_cost", np.nan)),
        "cost_slack": float(srow.get("slack_penalty_cost", np.nan)),
        "total_slack": total_slack,
        "slack_total": total_slack,
        "total_relief_requirement": total_R,
        "R_total": total_R,
        "relief_served": relief_served,
        "relief_served_share": relief_served_share,
        "total_debit": total_debit,
        "debit_total": total_debit,
        "physical_depth_total": float(x["physical_depth"].sum()),
        "allowance_productivity": allowance_productivity,
        "mean_budget_utilization": mean_budget_util,
        "median_budget_utilization": float(budget["budget_utilization"].median()) if "budget_utilization" in budget.columns else np.nan,
        "max_budget_utilization": max_budget_util,
        "min_budget_utilization": float(budget["budget_utilization"].min()) if "budget_utilization" in budget.columns else np.nan,
        "budget_util": mean_budget_util,
        "n_budget_binding": n_budget_binding,
        "physical_proxy_exposure": physical_proxy,
        "debit_proxy_exposure": debit_proxy,
        "proxy_column": proxy_col,
        "exposure_gini": exp_conc["gini"],
        "exposure_hhi": exp_conc["hhi"],
        "exposure_top10_share": exp_conc["top10_share"],
        "debit_gini": debit_conc["gini"],
        "debit_hhi": debit_conc["hhi"],
        "debit_top10_share": debit_conc["top10_share"],
        "gini": exp_conc["gini"],
        "hhi": exp_conc["hhi"],
        "top10_share": exp_conc["top10_share"],
        "mu_positive_count": np.nan,
        "mu_positive_share": np.nan,
        "mu_mean": np.nan,
        "mu_max": np.nan,
        "scarcity_alpha_sum": np.nan,
        "scarcity_alpha_mean": np.nan,
        "scarcity_alpha_max": np.nan,
        "scarcity_active_share": np.nan,
        "runtime_sec": float(srow.get("runtime_sec", np.nan)),
        "solve_time": float(srow.get("runtime_sec", np.nan)),
        "mip_gap": float(srow.get("mip_gap", np.nan)) if pd.notna(srow.get("mip_gap", np.nan)) else np.nan,
        "best_bound": float(srow.get("best_bound", srow.get("best_lower_bound", np.nan))) if pd.notna(srow.get("best_bound", srow.get("best_lower_bound", np.nan))) else np.nan,
        "node_count": float(srow.get("node_count", np.nan)) if pd.notna(srow.get("node_count", np.nan)) else np.nan,
        "solution_count": srow.get("solution_count", np.nan),
        "model_rows": srow.get("model_rows", np.nan),
        "model_cols": srow.get("model_cols", np.nan),
        "model_nnz": srow.get("model_nnz", np.nan),
        "notes": str(srow.get("notes", "Imported from B1_run_m1_touched_only.py")),
        "q_mean": np.nan,
        "q_std": np.nan,
        "q_min": np.nan,
        "q_p01": np.nan,
        "q_p05": np.nan,
        "q_p50": np.nan,
        "q_p95": np.nan,
        "q_p99": np.nan,
        "q_max": np.nan,
        "q_nontrivial_share": np.nan,
        "q_province_mean_min": np.nan,
        "q_province_mean_max": np.nan,
        "q_province_mean_std": np.nan,
    }


# =============================================================================
# Pairwise differences
# =============================================================================

def pair_row(year: int, comparison: str, baseline: pd.Series, comparator: pd.Series) -> Dict[str, Any]:
    b_cost = float(baseline.get("cost_total", baseline.get("objective", np.nan)))
    c_cost = float(comparator.get("cost_total", comparator.get("objective", np.nan)))
    b_slack = float(baseline.get("total_slack", baseline.get("slack_total", np.nan)))
    c_slack = float(comparator.get("total_slack", comparator.get("slack_total", np.nan)))

    return {
        "year": int(year),
        "split_role": baseline.get("split_role", comparator.get("split_role", "unclassified")),
        "comparison": comparison,
        "baseline_rule": baseline.get("rule"),
        "comparator_rule": comparator.get("rule"),
        "baseline_feature_set": baseline.get("feature_set"),
        "comparator_feature_set": comparator.get("feature_set"),
        "baseline_variant": baseline.get("variant"),
        "comparator_variant": comparator.get("variant"),
        "cost_baseline": b_cost,
        "cost_comparator": c_cost,
        "cost_saving": b_cost - c_cost,
        "cost_saving_pct": 100.0 * (b_cost - c_cost) / max(abs(b_cost), 1e-9),
        "slack_baseline": b_slack,
        "slack_comparator": c_slack,
        "slack_reduction": b_slack - c_slack,
        "slack_reduction_pct": 100.0 * (b_slack - c_slack) / max(abs(b_slack), 1e-9),
        "relief_served_share_baseline": float(baseline.get("relief_served_share", np.nan)),
        "relief_served_share_comparator": float(comparator.get("relief_served_share", np.nan)),
        "relief_served_share_change": float(comparator.get("relief_served_share", np.nan)) - float(baseline.get("relief_served_share", np.nan)),
        "allowance_productivity_baseline": float(baseline.get("allowance_productivity", np.nan)),
        "allowance_productivity_comparator": float(comparator.get("allowance_productivity", np.nan)),
        "allowance_productivity_change": float(comparator.get("allowance_productivity", np.nan)) - float(baseline.get("allowance_productivity", np.nan)),
        "physical_proxy_baseline": float(baseline.get("physical_proxy_exposure", np.nan)),
        "physical_proxy_comparator": float(comparator.get("physical_proxy_exposure", np.nan)),
        "physical_proxy_change": float(comparator.get("physical_proxy_exposure", np.nan)) - float(baseline.get("physical_proxy_exposure", np.nan)),
        "debit_proxy_baseline": float(baseline.get("debit_proxy_exposure", np.nan)),
        "debit_proxy_comparator": float(comparator.get("debit_proxy_exposure", np.nan)),
        "debit_proxy_change": float(comparator.get("debit_proxy_exposure", np.nan)) - float(baseline.get("debit_proxy_exposure", np.nan)),
        "budget_util_baseline": float(baseline.get("budget_util", np.nan)),
        "budget_util_comparator": float(comparator.get("budget_util", np.nan)),
        "mu_positive_count_baseline": baseline.get("mu_positive_count", np.nan),
        "mu_positive_count_comparator": comparator.get("mu_positive_count", np.nan),
    }


def build_pairwise(bench: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for year, g in bench.groupby("year", observed=True):
        main = g[g["comparison_group"] == "main"].copy()

        def one_main(rule: str) -> Optional[pd.Series]:
            h = main[main["rule"].astype(str) == rule]
            return None if h.empty else h.iloc[0]

        m1 = one_main("M1")
        m2 = one_main("M2")
        m3 = one_main("M3")

        for label, a, b in [
            ("M1_minus_M2", m1, m2),
            ("M2_minus_M3", m2, m3),
            ("M1_minus_M3", m1, m3),
        ]:
            if a is not None and b is not None:
                rows.append(pair_row(int(year), label, a, b))

        variants = g[g["comparison_group"] == "variant_comparison"].copy()
        if not variants.empty:
            def one_variant(v: str) -> Optional[pd.Series]:
                h = variants[variants["variant"].map(canonical_variant) == canonical_variant(v)]
                return None if h.empty else h.iloc[0]

            plain = one_variant("M3_plain")
            oi = one_variant("M3_OI")
            lo = one_variant("M3_LO")

            for label, a, b in [
                ("M3_plain_minus_M3_OI", plain, oi),
                ("M3_plain_minus_M3_LO", plain, lo),
                ("M3_OI_minus_M3_LO", oi, lo),
            ]:
                if a is not None and b is not None:
                    rows.append(pair_row(int(year), label, a, b))

        ladder = g[g["comparison_group"] == "feature_ladder"].copy()
        if not ladder.empty:
            base = ladder[ladder["feature_set"].astype(str).str.upper() == "F0"]
            if not base.empty:
                f0 = base.iloc[0]
                for _, r in ladder.iterrows():
                    fs = str(r.get("feature_set"))
                    if fs.upper() == "F0":
                        continue
                    rows.append(pair_row(int(year), f"F0_minus_{fs}", f0, r))

            f2base = ladder[ladder["feature_set"].astype(str).str.upper() == "F2"]
            if not f2base.empty:
                f2 = f2base.iloc[0]
                for _, r in ladder.iterrows():
                    fs = str(r.get("feature_set"))
                    if fs.upper() == "F2":
                        continue
                    rows.append(pair_row(int(year), f"F2_minus_{fs}", f2, r))

    return pd.DataFrame(rows)


# =============================================================================
# Manifest
# =============================================================================

def write_manifest(
    *,
    started_at: float,
    bench: pd.DataFrame,
    kkt: pd.DataFrame,
    pairwise: pd.DataFrame,
    qinv: pd.DataFrame,
    qaudit: pd.DataFrame,
    q_validation: pd.DataFrame,
    skipped: List[Dict[str, Any]],
    c_manifest_audit: Dict[str, Any],
    split_role_map: Dict[int, str],
) -> None:
    manifest = {
        "script": "D_run_full_benchmark.py",
        "script_version": "signed_off_final_benchmark_runner",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "repo_root": str(REPO_ROOT),
        "input_dir": str(INPUT_DIR),
        "derived_dir": str(DERIVED_DIR),
        "results_dir": str(RESULTS_DIR),
        "solution_dir": str(SOLUTION_DIR),
        "m1_only_results_dir": str(M1_ONLY_RESULTS_DIR),
        "m1_only_solution_dir": str(M1_ONLY_SOLUTION_DIR),
        "m3_weights_path": str(M3_WEIGHTS_PATH),
        "m3_weights_wide_path": str(M3_WEIGHTS_WIDE_PATH),
        "c_training_manifest_path": str(C_TRAINING_MANIFEST_PATH),
        "years": YEARS,
        "split_role_map": {str(k): v for k, v in split_role_map.items()},
        "provinces": PROVINCES,
        "main_feature_set": canonical_feature(MAIN_FEATURE_SET),
        "main_m3_variant": canonical_variant(MAIN_M3_VARIANT),
        "feature_ladder": [canonical_feature(x) for x in FEATURE_LADDER],
        "m3_variants": [canonical_variant(x) for x in M3_VARIANTS],
        "primary_proxy_column_preference": PRIMARY_BURDEN,
        "strictness": {
            "require_m1": D_REQUIRE_M1,
            "require_m2": D_REQUIRE_M2,
            "require_main_m3": D_REQUIRE_MAIN_M3,
            "require_c_manifest": D_REQUIRE_C_MANIFEST,
            "require_main_m3_alpha": D_REQUIRE_MAIN_M3_ALPHA,
            "strict_optional_q_blocks": D_STRICT_OPTIONAL_Q_BLOCKS,
            "q_active_mean_tol": Q_MEAN_TOL,
            "off_support_q_tol": OFF_SUPPORT_Q_TOL,
        },
        "run_flags": {
            "run_m2": RUN_M2,
            "run_main_m3": RUN_MAIN_M3,
            "import_b1_m1": IMPORT_B1_M1,
            "run_variant_comparison": RUN_VARIANT_COMPARISON,
            "run_feature_ladder": RUN_FEATURE_LADDER,
            "save_main_solutions": SAVE_MAIN_SOLUTIONS,
            "save_optional_solutions": SAVE_OPTIONAL_SOLUTIONS,
        },
        "source_of_truth": {
            "M1": "B1_run_m1_touched_only.py outputs are imported; D does not recompute publication M1.",
            "M2_M3": "B_run_annual_lp.py solve_weighted_depth_lp is used.",
            "M3_weights": "C_train_m3.py long/wide M3 weight outputs are used.",
            "physical_layer": "A_build_debit_optimization_inputs.py annual inputs are loaded through B.load_year_input.",
            "final_claims": "Use D benchmark outputs, not C training summaries, for paper claims.",
        },
        "metric_notes": {
            "relief_served_share": "1 - total_slack / total_constructed_relief_requirement; constructed-testbed metric.",
            "physical_proxy_exposure": "proxy weighted by x/P.",
            "debit_proxy_exposure": "proxy weighted by accounting debit.",
            "scarcity_alpha": "mu*x/P, not mu*q*x/P.",
            "M3_q": "q is accounting debit rate, not a price.",
        },
        "c_manifest_audit": c_manifest_audit,
        "n_benchmark_rows": int(len(bench)),
        "n_kkt_rows": int(len(kkt)),
        "n_pairwise_rows": int(len(pairwise)),
        "n_q_inventory_rows": int(len(qinv)),
        "n_q_audit_rows": int(len(qaudit)),
        "n_q_validation_rows": int(len(q_validation)),
        "skipped": skipped,
        "runtime_sec": float(time.time() - started_at),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "outputs": {
            "benchmark": str(BENCH_PATH),
            "kkt": str(KKT_PATH),
            "pairwise": str(PAIRWISE_PATH),
            "manifest": str(MANIFEST_PATH),
            "q_inventory": str(Q_INVENTORY_PATH),
            "q_audit": str(Q_AUDIT_PATH),
            "q_validation": str(Q_VALIDATION_PATH),
        },
    }

    save_json(manifest, MANIFEST_PATH)


# =============================================================================
# Header and final completeness audit
# =============================================================================

def print_header() -> None:
    print("=" * 78)
    print("D: RUN FINAL FULL DEBIT-RULE BENCHMARK — SIGNED-OFF VERSION")
    print("=" * 78)
    print(f"Repo root              : {REPO_ROOT}")
    print(f"Input dir              : {INPUT_DIR}")
    print(f"Results dir            : {RESULTS_DIR}")
    print(f"Solution dir           : {SOLUTION_DIR}")
    print(f"M1 B1 results dir      : {M1_ONLY_RESULTS_DIR}")
    print(f"M3 weights             : {M3_WEIGHTS_PATH}")
    print(f"C manifest             : {C_TRAINING_MANIFEST_PATH}")
    print(f"Years                  : {YEARS}")
    print(f"Provinces              : {PROVINCES}")
    print(f"Main M3                : {canonical_feature(MAIN_FEATURE_SET)} / {canonical_variant(MAIN_M3_VARIANT)}")
    print(f"Feature ladder         : {[canonical_feature(x) for x in FEATURE_LADDER]}")
    print(f"M3 variants            : {[canonical_variant(x) for x in M3_VARIANTS]}")
    print(f"Proxy preference       : {PRIMARY_BURDEN}")
    print(f"Import B1 M1           : {IMPORT_B1_M1}")
    print(f"Run M2                 : {RUN_M2}")
    print(f"Run main M3            : {RUN_MAIN_M3}")
    print(f"Variant comparison     : {RUN_VARIANT_COMPARISON}")
    print(f"Feature ladder         : {RUN_FEATURE_LADDER}")
    print(f"Save main solutions    : {SAVE_MAIN_SOLUTIONS}")
    print(f"Save optional solns.   : {SAVE_OPTIONAL_SOLUTIONS}")
    print(f"Require C manifest     : {D_REQUIRE_C_MANIFEST}")
    print(f"Require main alpha     : {D_REQUIRE_MAIN_M3_ALPHA}")
    print(f"Q mean tol             : {Q_MEAN_TOL:g}")
    print("=" * 78)


def final_completeness_audit(bench: pd.DataFrame) -> None:
    missing: List[str] = []
    for year in YEARS:
        main = bench[(bench["year"].astype(int) == int(year)) & (bench["comparison_group"] == "main")]
        have = set(main["rule"].astype(str).tolist())
        if D_REQUIRE_M1 and "M1" not in have:
            missing.append(f"year={year}: missing main M1")
        if D_REQUIRE_M2 and "M2" not in have:
            missing.append(f"year={year}: missing main M2")
        if D_REQUIRE_MAIN_M3 and "M3" not in have:
            missing.append(f"year={year}: missing main M3")
    if missing:
        raise RuntimeError("Final benchmark is incomplete:\n" + "\n".join(missing))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    started_at = time.time()
    ensure_dirs()
    print_header()

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Missing debit optimization input directory:\n  {INPUT_DIR}\n"
            "Run A_build_debit_optimization_inputs.py first."
        )

    c_manifest = read_json_if_exists(C_TRAINING_MANIFEST_PATH, required=D_REQUIRE_C_MANIFEST)
    c_manifest_audit = validate_c_manifest(c_manifest)
    split_role_map = split_roles_from_c_manifest(c_manifest)

    weights = load_m3_weights()
    qinv = q_inventory(weights)
    qaudit = q_audit(weights)

    save_csv(qinv, Q_INVENTORY_PATH)
    save_csv(qaudit, Q_AUDIT_PATH)

    print("\n[M3 Q INVENTORY]")
    print(qinv.to_string(index=False))

    b1_summary: Optional[pd.DataFrame] = None
    if IMPORT_B1_M1:
        b1_summary = load_b1_summary()
        print(f"\n[M1-B1] Loaded {len(b1_summary)} rows from {M1_ONLY_RESULTS_DIR / 'm1_only_summary.csv'}")
    elif D_REQUIRE_M1:
        raise RuntimeError("D_REQUIRE_M1=1 but D_IMPORT_B1_M1=0. Publication benchmark requires imported B1 M1.")

    rows: List[Dict[str, Any]] = []
    kkt_pieces: List[pd.DataFrame] = []
    q_validation_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for year in YEARS:
        year = int(year)
        ystart = time.time()
        split_role = split_role_for_year(year, weights, split_role_map)

        print("\n" + "=" * 78)
        print(f"[YEAR] {year} | split_role={split_role}")
        print("=" * 78)

        yi = B.load_year_input(year, input_dir=INPUT_DIR)

        total_R = float(yi.requirements["R_mw"].sum())
        n_state = int(len(yi.state))
        n_req = int(len(yi.requirements))
        n_coeff = int(len(yi.coefficients))
        n_hours = int(yi.state["hour_of_year"].nunique())

        pctr_col = first_existing(yi.participants, ["Pctr_mw", "P_ctr_MW", "P_i", "P_i_ctr_mw"], required=True)
        H_col = first_existing(yi.participants, ["H_depth", "H_depth_hours", "H_i", "H"], required=True)
        Hb_col = first_existing(yi.participants, ["H_touched", "H_touch_tokens", "Hbar_i", "Hbar"], required=True)

        print(
            f"[DATA] state={n_state:,} | hours={n_hours:,} | "
            f"requirements={n_req:,} | coeffs={n_coeff:,} | total R={total_R:,.3f}"
        )
        print(
            f"[BUDGET] Pctr mean={pd.to_numeric(yi.participants[pctr_col], errors='coerce').mean():.3f} MW | "
            f"H mean={pd.to_numeric(yi.participants[H_col], errors='coerce').mean():.3f} | "
            f"Hbar mean={pd.to_numeric(yi.participants[Hb_col], errors='coerce').mean():.3f}"
        )

        # ------------------------------------------------------------------
        # M1 from B1
        # ------------------------------------------------------------------
        if IMPORT_B1_M1 and b1_summary is not None:
            print("[IMPORT] M1 from B1_run_m1_touched_only.py")
            try:
                row_m1 = m1_benchmark_row(yi=yi, b1_summary=b1_summary, split_role=split_role)
                rows.append(row_m1)
                print(
                    f"[M1] objective={row_m1['objective']:,.3f} | "
                    f"slack={row_m1['total_slack']:.6f} | "
                    f"served={row_m1['relief_served_share']:.6f} | "
                    f"quality={row_m1['quality_flag']} | "
                    f"gap={row_m1['mip_gap']}"
                )
            except Exception as exc:
                msg = f"B1 import failed: {exc}"
                skipped.append({"year": year, "comparison_group": "main", "rule": "M1", "reason": msg})
                if D_REQUIRE_M1:
                    raise RuntimeError(msg) from exc
                print(f"[SKIP] M1 import failed: {exc}")

        # ------------------------------------------------------------------
        # M2
        # ------------------------------------------------------------------
        if RUN_M2:
            print("[SOLVE] M2 uniform-depth LP")
            try:
                res_m2 = B.solve_weighted_depth_lp(
                    yi=yi,
                    q=1.0,
                    rule_name="M2",
                    regularized=False,
                    output_flag=GUROBI_OUTPUT_FLAG,
                    save=SAVE_MAIN_SOLUTIONS,
                )
                rows.append(
                    m2_m3_result_to_benchmark_row(
                        result=res_m2,
                        yi=yi,
                        comparison_group="main",
                        display_rule="M2",
                        feature_set=None,
                        variant=None,
                        split_role=split_role,
                    )
                )
                append_kkt_rows(
                    kkt_pieces,
                    result=res_m2,
                    comparison_group="main",
                    display_rule="M2",
                    feature_set=None,
                    variant=None,
                    split_role=split_role,
                )
                print(
                    f"[M2] objective={res_m2.objective:,.3f} | "
                    f"slack={res_m2.total_slack:.6f} | "
                    f"served={res_m2.relief_served_share:.6f} | "
                    f"debit={res_m2.total_debit:,.3f} | "
                    f"runtime={res_m2.runtime_sec:.1f}s"
                )
            except Exception as exc:
                msg = f"M2 solve failed: {exc}"
                skipped.append({"year": year, "comparison_group": "main", "rule": "M2", "reason": msg})
                if D_REQUIRE_M2 or D_FAIL_ON_MAIN_SOLVE_ERROR:
                    raise RuntimeError(msg) from exc
                print(f"[SKIP] {msg}")
        elif D_REQUIRE_M2:
            raise RuntimeError("D_REQUIRE_M2=1 but D_RUN_M2=0.")

        # ------------------------------------------------------------------
        # Main M3
        # ------------------------------------------------------------------
        if RUN_MAIN_M3:
            print(f"[SOLVE] Main M3 weighted-depth LP | {canonical_feature(MAIN_FEATURE_SET)} / {canonical_variant(MAIN_M3_VARIANT)}")
            try:
                q_main_df, audit = validate_weight_block_against_year(
                    weights=weights,
                    yi=yi,
                    feature_set=MAIN_FEATURE_SET,
                    variant=MAIN_M3_VARIANT,
                    comparison_group="main",
                    enforce=True,
                )
                q_validation_rows.append(audit)
                main_rule_name = f"M3_{canonical_feature(MAIN_FEATURE_SET)}_{canonical_variant(MAIN_M3_VARIANT)}"

                res_m3 = B.solve_weighted_depth_lp(
                    yi=yi,
                    q=q_main_df,
                    rule_name=main_rule_name,
                    regularized=False,
                    output_flag=GUROBI_OUTPUT_FLAG,
                    save=SAVE_MAIN_SOLUTIONS,
                )
                rows.append(
                    m2_m3_result_to_benchmark_row(
                        result=res_m3,
                        yi=yi,
                        comparison_group="main",
                        display_rule="M3",
                        feature_set=canonical_feature(MAIN_FEATURE_SET),
                        variant=canonical_variant(MAIN_M3_VARIANT),
                        split_role=split_role,
                    )
                )
                append_kkt_rows(
                    kkt_pieces,
                    result=res_m3,
                    comparison_group="main",
                    display_rule="M3",
                    feature_set=canonical_feature(MAIN_FEATURE_SET),
                    variant=canonical_variant(MAIN_M3_VARIANT),
                    split_role=split_role,
                )
                print(
                    f"[M3] objective={res_m3.objective:,.3f} | "
                    f"slack={res_m3.total_slack:.6f} | "
                    f"served={res_m3.relief_served_share:.6f} | "
                    f"debit={res_m3.total_debit:,.3f} | "
                    f"runtime={res_m3.runtime_sec:.1f}s"
                )
            except Exception as exc:
                msg = f"Main M3 solve/validation failed: {exc}"
                skipped.append({"year": year, "comparison_group": "main", "rule": "M3", "reason": msg})
                if D_REQUIRE_MAIN_M3 or D_FAIL_ON_MAIN_SOLVE_ERROR:
                    raise RuntimeError(msg) from exc
                print(f"[SKIP] {msg}")
        elif D_REQUIRE_MAIN_M3:
            raise RuntimeError("D_REQUIRE_MAIN_M3=1 but D_RUN_MAIN_M3=0.")

        # ------------------------------------------------------------------
        # Variant comparison at main feature set
        # ------------------------------------------------------------------
        if RUN_VARIANT_COMPARISON:
            print(f"[OPTIONAL] M3 architecture comparison at {canonical_feature(MAIN_FEATURE_SET)}")
            for variant in M3_VARIANTS:
                variant_c = canonical_variant(variant)
                try:
                    if not has_weight_block(weights, year, MAIN_FEATURE_SET, variant_c):
                        raise ValueError("missing q block")
                    q_var_df, audit = validate_weight_block_against_year(
                        weights=weights,
                        yi=yi,
                        feature_set=MAIN_FEATURE_SET,
                        variant=variant_c,
                        comparison_group="variant_comparison",
                        enforce=D_STRICT_OPTIONAL_Q_BLOCKS,
                    )
                    q_validation_rows.append(audit)
                    rule_name = f"M3_{canonical_feature(MAIN_FEATURE_SET)}_{variant_c}"
                    res_var = B.solve_weighted_depth_lp(
                        yi=yi,
                        q=q_var_df,
                        rule_name=rule_name,
                        regularized=False,
                        output_flag=GUROBI_OUTPUT_FLAG,
                        save=SAVE_OPTIONAL_SOLUTIONS,
                    )
                    rows.append(
                        m2_m3_result_to_benchmark_row(
                            result=res_var,
                            yi=yi,
                            comparison_group="variant_comparison",
                            display_rule="M3_variant",
                            feature_set=canonical_feature(MAIN_FEATURE_SET),
                            variant=variant_c,
                            split_role=split_role,
                        )
                    )
                    append_kkt_rows(
                        kkt_pieces,
                        result=res_var,
                        comparison_group="variant_comparison",
                        display_rule="M3_variant",
                        feature_set=canonical_feature(MAIN_FEATURE_SET),
                        variant=variant_c,
                        split_role=split_role,
                    )
                    print(
                        f"[VAR {variant_c}] objective={res_var.objective:,.3f} | "
                        f"slack={res_var.total_slack:.6f} | "
                        f"served={res_var.relief_served_share:.6f} | "
                        f"runtime={res_var.runtime_sec:.1f}s"
                    )
                except Exception as exc:
                    skipped.append({
                        "year": year,
                        "comparison_group": "variant_comparison",
                        "feature_set": canonical_feature(MAIN_FEATURE_SET),
                        "variant": variant_c,
                        "reason": str(exc),
                    })
                    if D_STRICT_OPTIONAL_Q_BLOCKS:
                        raise
                    print(f"[SKIP] variant={variant_c}: {exc}")

        # ------------------------------------------------------------------
        # Feature ladder at main variant
        # ------------------------------------------------------------------
        if RUN_FEATURE_LADDER:
            main_var = canonical_variant(MAIN_M3_VARIANT)
            print(f"[OPTIONAL] M3 feature ladder at {main_var}")
            for feature_set in FEATURE_LADDER:
                fs = canonical_feature(feature_set)
                try:
                    if not has_weight_block(weights, year, fs, main_var):
                        raise ValueError("missing q block")
                    q_fs_df, audit = validate_weight_block_against_year(
                        weights=weights,
                        yi=yi,
                        feature_set=fs,
                        variant=main_var,
                        comparison_group="feature_ladder",
                        enforce=D_STRICT_OPTIONAL_Q_BLOCKS,
                    )
                    q_validation_rows.append(audit)
                    rule_name = f"M3_{fs}_{main_var}"
                    res_fs = B.solve_weighted_depth_lp(
                        yi=yi,
                        q=q_fs_df,
                        rule_name=rule_name,
                        regularized=False,
                        output_flag=GUROBI_OUTPUT_FLAG,
                        save=SAVE_OPTIONAL_SOLUTIONS,
                    )
                    rows.append(
                        m2_m3_result_to_benchmark_row(
                            result=res_fs,
                            yi=yi,
                            comparison_group="feature_ladder",
                            display_rule="M3_feature",
                            feature_set=fs,
                            variant=main_var,
                            split_role=split_role,
                        )
                    )
                    append_kkt_rows(
                        kkt_pieces,
                        result=res_fs,
                        comparison_group="feature_ladder",
                        display_rule="M3_feature",
                        feature_set=fs,
                        variant=main_var,
                        split_role=split_role,
                    )
                    print(
                        f"[LAD {fs}] objective={res_fs.objective:,.3f} | "
                        f"slack={res_fs.total_slack:.6f} | "
                        f"served={res_fs.relief_served_share:.6f} | "
                        f"runtime={res_fs.runtime_sec:.1f}s"
                    )
                except Exception as exc:
                    skipped.append({
                        "year": year,
                        "comparison_group": "feature_ladder",
                        "feature_set": fs,
                        "variant": main_var,
                        "reason": str(exc),
                    })
                    if D_STRICT_OPTIONAL_Q_BLOCKS:
                        raise
                    print(f"[SKIP] feature={fs}: {exc}")

        print(f"[YEAR DONE] {year} in {time.time() - ystart:.1f}s")

    bench = pd.DataFrame(rows)
    if bench.empty:
        raise RuntimeError("No benchmark rows were produced.")

    final_completeness_audit(bench)

    kkt = pd.concat(kkt_pieces, ignore_index=True) if kkt_pieces else pd.DataFrame()
    pairwise = build_pairwise(bench)
    q_validation = pd.DataFrame(q_validation_rows)

    save_csv(bench, BENCH_PATH)
    save_csv(kkt, KKT_PATH)
    save_csv(pairwise, PAIRWISE_PATH)
    save_csv(q_validation, Q_VALIDATION_PATH)

    write_manifest(
        started_at=started_at,
        bench=bench,
        kkt=kkt,
        pairwise=pairwise,
        qinv=qinv,
        qaudit=qaudit,
        q_validation=q_validation,
        skipped=skipped,
        c_manifest_audit=c_manifest_audit,
        split_role_map=split_role_map,
    )

    print("\n" + "=" * 78)
    print("D BENCHMARK COMPLETE")
    print("=" * 78)
    print(f"Benchmark   : {BENCH_PATH}")
    print(f"KKT         : {KKT_PATH}")
    print(f"Pairwise    : {PAIRWISE_PATH}")
    print(f"Manifest    : {MANIFEST_PATH}")
    print(f"Q inventory : {Q_INVENTORY_PATH}")
    print(f"Q audit     : {Q_AUDIT_PATH}")
    print(f"Q validation: {Q_VALIDATION_PATH}")

    if skipped:
        print("\n[SKIPPED]")
        print(pd.DataFrame(skipped).to_string(index=False))

    main = bench[bench["comparison_group"] == "main"].copy()
    if not main.empty:
        show_cols = [
            "year",
            "split_role",
            "rule",
            "feature_set",
            "variant",
            "objective",
            "total_slack",
            "relief_served_share",
            "total_debit",
            "allowance_productivity",
            "budget_util",
            "n_budget_binding",
            "mu_positive_count",
            "mu_max",
            "q_mean",
            "q_std",
            "q_min",
            "q_max",
            "mip_gap",
            "runtime_sec",
            "quality_flag",
        ]
        show_cols = [c for c in show_cols if c in main.columns]
        print("\n[MAIN RESULT SNAPSHOT]")
        print(main[show_cols].sort_values(["year", "rule"]).to_string(index=False))

    if not pairwise.empty:
        show_cols = [
            "year",
            "split_role",
            "comparison",
            "cost_saving",
            "cost_saving_pct",
            "slack_reduction",
            "relief_served_share_change",
            "allowance_productivity_change",
        ]
        show_cols = [c for c in show_cols if c in pairwise.columns]
        print("\n[PAIRWISE SNAPSHOT]")
        print(pairwise[show_cols].sort_values(["year", "comparison"]).to_string(index=False))

    print("\nImportant interpretation:")
    print("  relief_served_share     = constructed-testbed relief served without emergency slack")
    print("  physical_proxy_exposure = proxy weighted by x/P")
    print("  debit_proxy_exposure    = proxy weighted by accounting debit")
    print("  M3 q                    = accounting debit rate, not a price")
    print("  scarcity_alpha_*        = mu*x/P, not mu*q*x/P")
    print("  final paper claims      = use D outputs, not C training summaries")
    print("=" * 78)


if __name__ == "__main__":
    main()
