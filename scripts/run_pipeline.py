
"""
run_pipeline.py
===============

Master runner for the final paper pipeline.

This runner is strict:
- it uses scripts/ as the live execution location
- it does not default to synthetic data
- it follows the locked upstream-to-downstream order
- it stops when an upstream hard gate fails
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import pipeline_config as cfg  # type: ignore


def _script_path(filename: str) -> Path:
    candidates = [
        cfg.SCRIPTS_DIR / filename,
        BASE / filename,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


STAGES: Dict[str, dict] = {
    "step00": {
        "script": "step00_fetch_gopacs_announcements.py",
        "desc": "Fetch and audit GOPACS machine announcements",
        "output": cfg.JSONL_PATH,
    },
    "step01": {
        "script": "step01_crosswalk_real.py",
        "desc": "Build GOPACS location candidate-support crosswalk",
        "output": cfg.CROSSWALK_PATH,
    },
    "step02": {
        "script": "step02_announcement_panel_real.py",
        "desc": "Build province-hour announcement panel",
        "output": cfg.ANN_PANEL_PATH,
    },
    "step03": {
        "script": "step03_called_activity.py",
        "desc": "Build GOPACS called-activity panel",
        "output": cfg.ACTIVITY_PATH,
    },
    "step04": {
        "script": "step04_burden_proxy.py",
        "desc": "Build burden proxy panel",
        "output": cfg.BURDEN_PATH,
    },
    "step05": {
        "script": "step05_auditable_state.py",
        "desc": "Build auditable-state master panel",
        "output": cfg.STATE_PATH,
    },
    "step06": {
        "script": "step06_testbed.py",
        "desc": "Build operational testbed and participant map",
        "output": cfg.TESTBED_PATH,
    },
    "A": {
        "script": "A_construct_lp_params.py",
        "desc": "Construct annual LP parameter files",
        "output": cfg.LP_DIR / "lp_meta.json",
    },
    "C": {
        "script": "C_train_m3.py",
        "desc": "Train M3 variants and feature ladders",
        "output": cfg.M3_WEIGHTS_PATH,
    },
    "D": {
        "script": "D_run_full_benchmark.py",
        "desc": "Run annual benchmark and diagnostics",
        "output": cfg.BENCH_PATH,
    },
    "E": {
        "script": "E_make_paper_outputs.py",
        "desc": "Generate all paper tables and figures (T1-T5, F1-F7)",
        "output": cfg.TABLES_DIR / "T1_main_benchmark.tex",
    },
    "PDF": {
        "script": "build_results_pdf.py",
        "desc": "Build results briefing PDF",
        "output": cfg.ROOT / "results_briefing.pdf",
    },
    "validate": {
        "script": "validate_pipeline.py",
        "desc": "Validate pipeline artifacts",
        "output": cfg.VALIDATION_REPORT_PATH,
    },
}

DEFAULT_ORDER: List[str] = [
    "step00",
    "step01",
    "step02",
    "step03",
    "step04",
    "step05",
    "step06",
    "A",
    "C",
    "D",
    "E",
    "PDF",
    "validate",
]


def _run_script(stage_name: str, stage: dict) -> bool:
    script = _script_path(stage["script"])
    if not script.exists():
        print(f"[FAIL] Missing script for stage {stage_name}: {script}")
        return False

    print("\n" + "─" * 72)
    print(f"Running stage {stage_name}: {stage['desc']}")
    print(f"Script : {script}")
    print("─" * 72)

    t0 = time.time()
    result = subprocess.run([sys.executable, str(script)], cwd=str(BASE))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[FAIL] Stage {stage_name} exited with code {result.returncode} after {elapsed:.1f}s")
        return False

    expected = stage.get("output")
    if isinstance(expected, Path):
        if expected.exists():
            print(f"[OK] Output exists: {expected}")
        else:
            print(f"[WARN] Expected output missing: {expected}")

    print(f"[OK] Stage {stage_name} completed in {elapsed:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final debit-rule paper pipeline.")
    parser.add_argument("--from-stage", choices=DEFAULT_ORDER, default=None)
    parser.add_argument("--only", choices=DEFAULT_ORDER, default=None)
    args = parser.parse_args()

    if args.only is not None:
        order = [args.only]
    else:
        order = list(DEFAULT_ORDER)
        if args.from_stage is not None:
            idx = order.index(args.from_stage)
            order = order[idx:]

    print("=" * 72)
    print("FINAL PAPER SCRIPT — MASTER PIPELINE")
    print("=" * 72)
    print(f"Repo root : {BASE}")
    print(f"Stages    : {order}")

    failed: List[str] = []
    t_total = time.time()

    for stage_name in order:
        ok = _run_script(stage_name, STAGES[stage_name])
        if not ok:
            failed.append(stage_name)
            break

    elapsed = time.time() - t_total
    print("\n" + "=" * 72)
    if failed:
        print(f"PIPELINE FAILED at stage: {failed[-1]}")
        print(f"Re-run from there with:")
        print(f'  python run_pipeline.py --from-stage {failed[-1]}')
    else:
        print(f"PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
        print(f"Benchmark results : {cfg.BENCH_PATH}")
        print(f"Validation report : {cfg.VALIDATION_REPORT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
