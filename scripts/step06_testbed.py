"""
step06_testbed.py
=================

Build the operational testbed and the representative participant map.

Locked operating layer for the paper
------------------------------------
- one representative participant per province
- one monitored constraint per province
- same-province curtailment or slack only

This script therefore does NOT infer multiple participants from crosswalk
operators. Doing that would violate the paper's locked annual operating layer.

Inputs
------
- auditable_state_province_hour.csv

Outputs
-------
- operational_testbed_inputs.parquet
- participant_area_map.csv
- operational_testbed_metadata.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pipeline_config as cfg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        f"Could not import pipeline_config.py from {REPO_ROOT}. "
        "Run this script from the final repo structure."
    ) from exc


# =============================================================================
# PATH / COLUMN HELPERS
# =============================================================================
def _get_path(*names: str, env_name: Optional[str] = None, default: Optional[Path] = None) -> Path:
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
    raise AttributeError(f"Could not resolve path from names {names!r}.")


def _require_existing_path(label: str, path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n"
            f"  {path}\n\n"
            f"Fix pipeline_config.py so this path points to the correct file."
        )
    print(f"{label}: using -> {path}")
    return path


def _pick_column(columns: Iterable[str], candidates: Iterable[str], *, required: bool = True) -> Optional[str]:
    low = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    if required:
        raise KeyError(f"Could not find columns {list(candidates)!r}. Found: {list(columns)!r}")
    return None


def _assert_unique_keys(df: pd.DataFrame, keys: list[str], label: str) -> None:
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        sample = (
            df.loc[dup, keys]
            .drop_duplicates()
            .sort_values(keys)
            .head(20)
        )
        raise ValueError(
            f"[{label}] Duplicate rows found for keys {keys}.\n"
            f"Examples:\n{sample.to_string(index=False)}"
        )


def _safe_participant_token(province: str) -> str:
    return province.upper().replace("-", "_").replace(" ", "_")


# =============================================================================
# LOAD / VALIDATE STATE PANEL
# =============================================================================
def _load_state_panel(state_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(state_path, parse_dates=["timestamp_utc"])
    panel.columns = [str(c).strip() for c in panel.columns]
    panel["timestamp_utc"] = pd.to_datetime(panel["timestamp_utc"], utc=True, errors="coerce")

    required_cols = [
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
    ]
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise ValueError(
            f"[STATE] Missing required columns: {missing}\n"
            f"Found columns: {list(panel.columns)}"
        )

    panel["province"] = panel["province"].astype(str).str.strip()
    panel = panel.dropna(subset=["province", "timestamp_utc"]).copy()

    numeric_cols = [
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
    ]
    for col in numeric_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    _assert_unique_keys(panel, ["province", "timestamp_utc"], "STATE_PANEL")

    provinces_expected = list(getattr(cfg, "NL_PROVINCES", []))
    if provinces_expected:
        bad = sorted(set(panel["province"].unique()) - set(provinces_expected))
        if bad:
            raise ValueError(f"[STATE] Unexpected province names found: {bad}")

        missing_prov = sorted(set(provinces_expected) - set(panel["province"].unique()))
        if missing_prov:
            raise ValueError(f"[STATE] Missing expected provinces: {missing_prov}")

    # Province coverage per timestamp
    counts = (
        panel.groupby("timestamp_utc")["province"]
        .nunique()
        .rename("n_provinces")
        .reset_index()
    )
    expected_n = len(provinces_expected) if provinces_expected else int(panel["province"].nunique())
    bad_counts = counts[counts["n_provinces"] != expected_n]
    if not bad_counts.empty:
        raise ValueError(
            f"[STATE] Some timestamps do not have exactly {expected_n} provinces.\n"
            f"Examples:\n{bad_counts.head(20).to_string(index=False)}"
        )

    return panel.sort_values(["province", "timestamp_utc"]).reset_index(drop=True)


# =============================================================================
# BUILD PARTICIPANT MAP
# =============================================================================
def _build_participant_area_map(provinces: list[str], representative_capacity_mw: float) -> pd.DataFrame:
    pam = pd.DataFrame(
        {
            "province": provinces,
            "participant_id": [f"P_{_safe_participant_token(p)}" for p in provinces],
            "constraint_id": [f"K_{_safe_participant_token(p)}" for p in provinces],
            "capacity_mw": float(representative_capacity_mw),
            "participant_type": "representative_province",
            "same_province_only": 1,
        }
    )

    _assert_unique_keys(pam, ["province"], "PAM_PROVINCE")
    _assert_unique_keys(pam, ["participant_id"], "PAM_PARTICIPANT")
    _assert_unique_keys(pam, ["constraint_id"], "PAM_CONSTRAINT")
    return pam


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    state_path = _require_existing_path(
        "auditable state panel",
        _get_path(
            "STATE_PATH",
            "AUDITABLE_STATE_PATH",
            env_name="STATE_PATH",
            default=REPO_ROOT / "data" / "derived" / "auditable_state_province_hour.csv",
        ),
    )

    testbed_path = _get_path(
        "TESTBED_PATH",
        env_name="TESTBED_PATH",
        default=REPO_ROOT / "data" / "derived" / "operational_testbed_inputs.parquet",
    )

    pam_path = _get_path(
        "PAM_PATH",
        "PARTICIPANT_AREA_MAP_PATH",
        env_name="PAM_PATH",
        default=REPO_ROOT / "data" / "derived" / "participant_area_map.csv",
    )

    metadata_path = _get_path(
        "TESTBED_METADATA_PATH",
        env_name="TESTBED_METADATA_PATH",
        default=REPO_ROOT / "data" / "derived" / "operational_testbed_metadata.json",
    )

    qc_dir = _get_path(
        "QC_DIR",
        env_name="QC_DIR",
        default=REPO_ROOT / "qc",
    )
    qc_summary_path = qc_dir / "step06_testbed_summary.json"

    print("=" * 72)
    print("STEP 06: BUILD OPERATIONAL TESTBED + PARTICIPANT MAP")
    print("=" * 72)
    print(f"State panel : {state_path}")
    print(f"Testbed     : {testbed_path}")
    print(f"PAM         : {pam_path}")
    print(f"Metadata    : {metadata_path}")

    panel = _load_state_panel(state_path)

    provinces = list(getattr(cfg, "NL_PROVINCES", []))
    if not provinces:
        provinces = sorted(panel["province"].dropna().astype(str).unique().tolist())

    representative_capacity_mw = float(
        getattr(cfg, "REPRESENTATIVE_CAPACITY_MW", os.environ.get("REPRESENTATIVE_CAPACITY_MW", "50.0"))
    )

    pam = _build_participant_area_map(
        provinces=provinces,
        representative_capacity_mw=representative_capacity_mw,
    )

    # The operational testbed is the validated auditable-state panel.
    # Downstream optimization builders will add the paper's operating-layer
    # objects explicitly (R_{n,t}, c_{i,t}, H_i, Hbar_i, etc.).
    testbed = panel.copy()

    # Write outputs
    pam_path.parent.mkdir(parents=True, exist_ok=True)
    pam.to_csv(pam_path, index=False, encoding="utf-8")

    testbed_path.parent.mkdir(parents=True, exist_ok=True)
    testbed.to_parquet(testbed_path, index=False)

    metadata = {
        "input_state_path": str(state_path),
        "output_testbed_path": str(testbed_path),
        "output_pam_path": str(pam_path),
        "n_rows": int(len(testbed)),
        "n_cols": int(testbed.shape[1]),
        "provinces": provinces,
        "n_provinces": int(len(provinces)),
        "n_representative_participants": int(len(pam)),
        "n_monitored_constraints": int(len(pam)),
        "representative_capacity_mw": float(representative_capacity_mw),
        "time_start_utc": str(testbed["timestamp_utc"].min()),
        "time_end_utc": str(testbed["timestamp_utc"].max()),
        "locked_operating_layer": {
            "one_participant_per_province": True,
            "one_constraint_per_province": True,
            "same_province_curtailment_only": True,
        },
        "feature_groups": {
            "F0": ["year", "hour_of_year", "hour_of_week", "month", "is_weekend", "is_holiday"],
            "F1": ["price_da_norm"],
            "F2": ["rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm"],
            "F3": ["structural_exposure"],
        },
        "required_contract_fields": [
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
        ],
    }

    if "split" in testbed.columns:
        metadata["split_counts"] = {
            str(k): int(v)
            for k, v in testbed["split"].astype(str).value_counts(dropna=False).sort_index().to_dict().items()
        }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_summary = {
        "rows": int(len(testbed)),
        "n_provinces": int(testbed["province"].nunique()),
        "n_timestamps": int(testbed["timestamp_utc"].nunique()),
        "expected_rows_from_panel_balance": int(
            testbed["province"].nunique() * testbed["timestamp_utc"].nunique()
        ),
        "balanced_panel": bool(
            len(testbed) == testbed["province"].nunique() * testbed["timestamp_utc"].nunique()
        ),
        "representative_capacity_mw": float(representative_capacity_mw),
        "state_path": str(state_path),
        "testbed_path": str(testbed_path),
        "pam_path": str(pam_path),
        "metadata_path": str(metadata_path),
    }
    with qc_summary_path.open("w", encoding="utf-8") as f:
        json.dump(qc_summary, f, indent=2)

    print(f"\nRepresentative participants: {len(pam)}")
    print(f"Representative capacity MW: {representative_capacity_mw}")
    print(f"Testbed shape             : {testbed.shape}")
    print(f"Provinces                 : {testbed['province'].nunique()}")
    print(f"Time range                : {testbed['timestamp_utc'].min()} -> {testbed['timestamp_utc'].max()}")
    print(f"\n✓ Written: {pam_path}")
    print(f"✓ Written: {testbed_path}")
    print(f"✓ Written: {metadata_path}")
    print(f"✓ Written: {qc_summary_path}")


if __name__ == "__main__":
    main()