"""
validate_pipeline.py
====================

Current-paper validator.

This validator checks only the artifacts and invariants that are valid for the
current paper. It intentionally does NOT check the old synthetic G0/G1/G2/G3
pipeline, old benchmark naming, or hardcoded narrative expectations.

Validation layers
-----------------
0. GOPACS raw archive
1. Crosswalk candidate-support layer
2. Province-hour announcement panel
3. GOPACS activity layer
4. Burden proxy layer
5. Auditable-state master panel
6. Operational testbed / participant map
7. Optional downstream benchmark artifacts
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pipeline_config as cfg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Could not import pipeline_config.py.") from exc


PASS = "[PASS]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def _get_path(*names: str, env_name: Optional[str] = None, default: Optional[Path] = None) -> Optional[Path]:
    for name in names:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if value:
                return Path(value)
    if env_name:
        env_value = os.environ.get(env_name)
        if env_value:
            return Path(env_value)
    if default is not None:
        return Path(default)
    return None


def _section(title: str) -> None:
    print("\n" + "─" * 72)
    print(title)
    print("─" * 72)


def _report(ok: bool, message: str, *, fatal: bool = False, errors: list[str], warnings: list[str]) -> None:
    if ok:
        print(f"{PASS} {message}")
        return
    if fatal:
        print(f"{FAIL} {message}")
        errors.append(message)
    else:
        print(f"{WARN} {message}")
        warnings.append(message)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []

    jsonl_path = _get_path("JSONL_PATH", "GOPACS_JSONL_PATH", env_name="JSONL_PATH")
    manifest_path = _get_path("GOPACS_MANIFEST_PATH", env_name="GOPACS_MANIFEST_PATH")
    window_log_path = _get_path("GOPACS_WINDOW_LOG_PATH", env_name="GOPACS_WINDOW_LOG_PATH")
    raw_summary_path = _get_path("GOPACS_SUMMARY_PATH", env_name="GOPACS_SUMMARY_PATH")
    crosswalk_path = _get_path("CROSSWALK_PATH", env_name="CROSSWALK_PATH")
    ann_panel_path = _get_path("ANN_PANEL_PATH", env_name="ANN_PANEL_PATH")
    activity_path = _get_path("ACTIVITY_PATH", env_name="ACTIVITY_PATH")
    burden_path = _get_path("BURDEN_PATH", env_name="BURDEN_PATH")
    state_path = _get_path("STATE_PATH", env_name="STATE_PATH")
    testbed_path = _get_path("TESTBED_PATH", env_name="TESTBED_PATH")
    pam_path = _get_path("PAM_PATH", "PARTICIPANT_MAP_PATH", env_name="PAM_PATH")
    testbed_meta_path = _get_path("TESTBED_METADATA_PATH", env_name="TESTBED_METADATA_PATH")
    bench_path = _get_path("BENCH_PATH", "BENCHMARK_PATH", env_name="BENCHMARK_PATH")
    m3_path = _get_path("M3_WEIGHTS_PATH", env_name="M3_WEIGHTS_PATH")

    _section("LAYER 0: GOPACS raw archive")
    for path, label in [
        (jsonl_path, "raw JSONL"),
        (manifest_path, "fetch manifest"),
        (window_log_path, "window log"),
        (raw_summary_path, "raw summary"),
    ]:
        _report(path is not None and path.exists(), f"{label} exists", fatal=True, errors=errors, warnings=warnings)

    if jsonl_path and jsonl_path.exists():
        ids: list[str] = []
        parse_errors = 0
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ids.append(str(rec.get("id", "")))
                except Exception:
                    parse_errors += 1
        _report(parse_errors == 0, f"raw JSONL lines parse cleanly (errors={parse_errors})", fatal=True, errors=errors, warnings=warnings)
        ids_clean = [x for x in ids if x]
        dup_count = len(ids_clean) - len(set(ids_clean))
        _report(dup_count == 0, f"raw JSONL has no duplicate retained ids (duplicates={dup_count})", fatal=True, errors=errors, warnings=warnings)

    _section("LAYER 1: Crosswalk candidate-support layer")
    _report(crosswalk_path is not None and crosswalk_path.exists(), "crosswalk exists", fatal=True, errors=errors, warnings=warnings)
    if crosswalk_path and crosswalk_path.exists():
        cw = pd.read_csv(crosswalk_path)
        required = {"location_string", "province", "match_type", "match_score", "source_field", "is_fallback", "ambiguity_degree"}
        _report(required.issubset(set(cw.columns)), f"crosswalk has required metadata columns", fatal=True, errors=errors, warnings=warnings)
        if "match_score" in cw.columns:
            _report(cw["match_score"].notna().all(), "crosswalk match_score populated", fatal=False, errors=errors, warnings=warnings)

    _section("LAYER 2: Announcement panel")
    _report(ann_panel_path is not None and ann_panel_path.exists(), "announcement panel exists", fatal=True, errors=errors, warnings=warnings)
    if ann_panel_path and ann_panel_path.exists():
        ann = pd.read_csv(ann_panel_path, parse_dates=["timestamp_utc"])
        ann["timestamp_utc"] = pd.to_datetime(ann["timestamp_utc"], utc=True)
        required = {"province", "timestamp_utc", "n_announcements", "has_announcement"}
        _report(required.issubset(set(ann.columns)), "announcement panel has required columns", fatal=True, errors=errors, warnings=warnings)
        if {"timestamp_utc", "n_announcements"}.issubset(ann.columns):
            per_hour = ann.groupby("timestamp_utc")["n_announcements"].sum()
            _report((per_hour >= 0).all(), "announcement panel has nonnegative province-hour count mass", fatal=True, errors=errors, warnings=warnings)

    _section("LAYER 3: Activity layer")
    _report(activity_path is not None and activity_path.exists(), "activity panel exists", fatal=True, errors=errors, warnings=warnings)
    if activity_path and activity_path.exists():
        activity = pd.read_csv(activity_path, parse_dates=["timestamp_utc"])
        activity["timestamp_utc"] = pd.to_datetime(activity["timestamp_utc"], utc=True)
        meta_candidates = [
            activity_path.with_name(f"{activity_path.stem}_metadata.json"),
            activity_path.with_suffix(".metadata.json"),
            activity_path.parent / "gopacs_activity_metadata.json",
        ]
        meta_path = next((p for p in meta_candidates if p.exists()), None)
        if meta_path:
            meta = _load_json(meta_path)
            _report("source_granularity" in meta, "activity metadata has source_granularity", fatal=False, errors=errors, warnings=warnings)
            if bool(meta.get("is_hourly_observed", False)):
                _report(True, "activity layer certified as observed hourly", fatal=False, errors=errors, warnings=warnings)
            else:
                _report(False, "activity layer is not certified as observed hourly; baseline B_agg is blocked", fatal=False, errors=errors, warnings=warnings)
        else:
            _report(False, "activity metadata file missing", fatal=False, errors=errors, warnings=warnings)

    _section("LAYER 4: Burden proxy layer")
    _report(burden_path is not None and burden_path.exists(), "burden panel exists", fatal=True, errors=errors, warnings=warnings)
    if burden_path and burden_path.exists():
        burden = pd.read_csv(burden_path, parse_dates=["timestamp_utc"])
        burden["timestamp_utc"] = pd.to_datetime(burden["timestamp_utc"], utc=True)
        required = {"province", "timestamp_utc", "gopacs_eur", "gopacs_mwh", "gopacs_activation", "B_agg", "E_exposure", "B_proxy_baseline"}
        _report(required.issubset(set(burden.columns)), "burden panel has required columns", fatal=True, errors=errors, warnings=warnings)
        if {"timestamp_utc", "E_exposure"}.issubset(burden.columns):
            sums = burden.groupby("timestamp_utc")["E_exposure"].sum()
            max_err = float(sums.sub(1.0).abs().max()) if len(sums) else 0.0
            _report(max_err < 1e-6 or max_err < 1e-3, f"E_exposure sums to one per hour (max abs err={max_err:.3e})", fatal=False, errors=errors, warnings=warnings)
        if {"B_agg", "gopacs_eur"}.issubset(burden.columns):
            b_expected = np.log1p(pd.to_numeric(burden["gopacs_eur"], errors="coerce").fillna(0.0).clip(lower=0.0))
            max_diff = float((pd.to_numeric(burden["B_agg"], errors="coerce").fillna(0.0) - b_expected).abs().max())
            _report(max_diff < 1e-9 or max_diff < 1e-6, f"B_agg matches log1p(gopacs_eur) (max diff={max_diff:.3e})", fatal=False, errors=errors, warnings=warnings)

    _section("LAYER 5: Auditable-state master panel")
    _report(state_path is not None and state_path.exists(), "state panel exists", fatal=True, errors=errors, warnings=warnings)
    if state_path and state_path.exists():
        state = pd.read_csv(state_path, parse_dates=["timestamp_utc"])
        state["timestamp_utc"] = pd.to_datetime(state["timestamp_utc"], utc=True)
        required_contract = {
            "province",
            "timestamp_utc",
            "year",
            "hour_of_year",
            "hour_of_week",
            "month",
            "is_weekend",
            "is_holiday",
            "price_da_eur_per_mwh",
            "price_da_norm",
            "rl_plus_norm",
            "rl_minus_norm",
            "s_plus_norm",
            "s_minus_norm",
            "structural_exposure",
            "gopacs_eur",
            "gopacs_mwh",
            "gopacs_activation",
            "B_agg",
            "E_exposure",
            "B_proxy_baseline",
        }
        _report(required_contract.issubset(set(state.columns)), "state panel satisfies the required data contract", fatal=True, errors=errors, warnings=warnings)
        _report(state["province"].nunique() == 12, f"state panel covers 12 provinces (found {state['province'].nunique()})", fatal=False, errors=errors, warnings=warnings)
        nan_majority = [
            c for c in required_contract
            if c in state.columns and state[c].isna().mean() > 0.5
        ]
        _report(not nan_majority, f"no required contract field is majority-NaN (flagged={nan_majority})", fatal=False, errors=errors, warnings=warnings)

    _section("LAYER 6: Operational testbed / participant map")
    _report(testbed_path is not None and testbed_path.exists(), "operational testbed parquet exists", fatal=True, errors=errors, warnings=warnings)
    _report(pam_path is not None and pam_path.exists(), "participant map exists", fatal=True, errors=errors, warnings=warnings)
    if pam_path and pam_path.exists():
        pam = pd.read_csv(pam_path)
        required = {"province", "participant_id", "constraint_id", "same_province_only"}
        _report(required.issubset(set(pam.columns)), "participant map has required columns", fatal=True, errors=errors, warnings=warnings)
        _report(len(pam) == pam["province"].nunique() == 12, "participant map has exactly one representative participant per province", fatal=False, errors=errors, warnings=warnings)
        if "same_province_only" in pam.columns:
            _report((pd.to_numeric(pam["same_province_only"], errors="coerce").fillna(0) == 1).all(), "participant map enforces same-province-only logic", fatal=False, errors=errors, warnings=warnings)

    if testbed_meta_path and testbed_meta_path.exists():
        meta = _load_json(testbed_meta_path)
        _report("feature_groups" in meta, "testbed metadata has feature_groups", fatal=False, errors=errors, warnings=warnings)
        fg = meta.get("feature_groups", {})
        _report(set(fg.keys()) >= {"F0", "F1", "F2", "F3"}, "testbed metadata uses current F0/F1/F2/F3 naming", fatal=False, errors=errors, warnings=warnings)

    _section("LAYER 7: Optional downstream artifacts")
    if m3_path and m3_path.exists():
        m3 = pd.read_parquet(m3_path)
        if "variant" in m3.columns:
            required_variants = {"M3_plain", "M3_OI", "M3_LO"}
            _report(required_variants.issubset(set(m3["variant"].unique())), "M3 variants plain/OI/LO present", fatal=False, errors=errors, warnings=warnings)
    else:
        _report(False, "M3 weights not found yet (optional at this stage)", fatal=False, errors=errors, warnings=warnings)

    if bench_path and bench_path.exists():
        bench = pd.read_csv(bench_path)
        if "rule" in bench.columns:
            expected_rules = {"M1", "M2", "M3"}
            _report(expected_rules.issubset(set(bench["rule"].unique())), "benchmark contains M1/M2/M3", fatal=False, errors=errors, warnings=warnings)
    else:
        _report(False, "benchmark results not found yet (optional at this stage)", fatal=False, errors=errors, warnings=warnings)

    _section("SUMMARY")
    print(f"Errors   : {len(errors)}")
    print(f"Warnings : {len(warnings)}")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")
        print("\n✗ Fix fatal errors before continuing downstream.")
    else:
        print("\n✓ No fatal validation errors.")
        if warnings:
            print("  Continue, but fix the warnings before final paper runs.")


if __name__ == "__main__":
    main()
