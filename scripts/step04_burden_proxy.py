from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
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
    raise AttributeError(f"Could not resolve any of path names {names!r}.")


def _get_bool(*names: str, env_name: Optional[str] = None, default: bool = False) -> bool:
    for name in names:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if isinstance(value, bool):
                return value
            if value is not None:
                s = str(value).strip().lower()
                if s in {"1", "true", "yes", "y"}:
                    return True
                if s in {"0", "false", "no", "n"}:
                    return False
    if env_name:
        env_value = os.environ.get(env_name)
        if env_value is not None:
            s = str(env_value).strip().lower()
            if s in {"1", "true", "yes", "y"}:
                return True
            if s in {"0", "false", "no", "n"}:
                return False
    return default


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
        raise KeyError(f"Could not find any of columns {list(candidates)!r}. Found: {list(columns)!r}")
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


def _minmax01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    lo = float(x.min())
    hi = float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=s.index)
    return (x - lo) / (hi - lo)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as tmp:
        df.to_csv(tmp.name, index=False, encoding="utf-8")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


# =============================================================================
# DATA CONTAINERS
# =============================================================================
@dataclass(frozen=True)
class NationalActivitySummary:
    national_hourly: pd.DataFrame
    source_path: Path
    mode: str
    observed_start_utc: pd.Timestamp
    observed_end_utc: pd.Timestamp
    raw_panel_start_utc: pd.Timestamp
    raw_panel_end_utc: pd.Timestamp
    metadata_path: Optional[Path]


@dataclass(frozen=True)
class LocalActivitySummary:
    local_hourly: pd.DataFrame
    source_path: Path
    mode: str


# =============================================================================
# STRESS / RL LOADER
# =============================================================================
def _load_stress_panel(stress_path: Path) -> tuple[pd.DataFrame, str]:
    """
    Accept either:
      A) a true stress panel with rl_plus_norm / s_plus_norm style columns, or
      B) a residual-load panel with RL_MW only, from which the directional stress
         columns are derived.

    Output contract always includes:
      province, timestamp_utc, rl_plus_norm, rl_minus_norm, s_plus_norm, s_minus_norm
    """
    stress = pd.read_csv(stress_path)
    stress.columns = [str(c).strip() for c in stress.columns]

    ts_col = _pick_column(
        stress.columns,
        ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc", "timestamp"],
    )
    province_col = _pick_column(stress.columns, ["province"])

    stress["timestamp_utc"] = pd.to_datetime(stress[ts_col], utc=True, errors="coerce")
    if stress["timestamp_utc"].isna().all():
        raise ValueError(f"[STRESS_PANEL] Could not parse any timestamps from {stress_path}")

    stress["province"] = stress[province_col].astype(str).str.strip()

    rl_plus_col = _pick_column(
        stress.columns,
        ["rl_plus_norm", "RL_plus_norm", "rlplus_norm"],
        required=False,
    )
    rl_minus_col = _pick_column(
        stress.columns,
        ["rl_minus_norm", "RL_minus_norm", "rlminus_norm"],
        required=False,
    )
    s_plus_col = _pick_column(
        stress.columns,
        ["s_plus_norm", "S_plus_norm", "stress_plus_norm"],
        required=False,
    )
    s_minus_col = _pick_column(
        stress.columns,
        ["s_minus_norm", "S_minus_norm", "stress_minus_norm"],
        required=False,
    )
    structural_col = _pick_column(
        stress.columns,
        ["structural_exposure", "Structural_exposure"],
        required=False,
    )

    # -------------------------------------------------------------------------
    # Case A: normalized stress columns already exist
    # -------------------------------------------------------------------------
    if rl_plus_col is not None and rl_minus_col is not None and s_plus_col is not None and s_minus_col is not None:
        keep = ["province", "timestamp_utc", rl_plus_col, rl_minus_col, s_plus_col, s_minus_col]
        rename = {
            rl_plus_col: "rl_plus_norm",
            rl_minus_col: "rl_minus_norm",
            s_plus_col: "s_plus_norm",
            s_minus_col: "s_minus_norm",
        }
        if structural_col is not None:
            keep.append(structural_col)
            rename[structural_col] = "structural_exposure"

        out = stress[keep].rename(columns=rename).copy()
        for col in ["rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        if "structural_exposure" in out.columns:
            out["structural_exposure"] = pd.to_numeric(out["structural_exposure"], errors="coerce")

        out = out.dropna(subset=["province", "timestamp_utc"]).copy()
        _assert_unique_keys(out, ["province", "timestamp_utc"], "STRESS_PANEL")
        return out.sort_values(["province", "timestamp_utc"]).reset_index(drop=True), "precomputed_stress_panel"

    # -------------------------------------------------------------------------
    # Case B: derive from residual load
    # -------------------------------------------------------------------------
    rl_col = _pick_column(
        stress.columns,
        ["RL_MW", "rl_mw", "residual_load_mw", "nat_rl_MW"],
        required=False,
    )
    if rl_col is None:
        raise ValueError(
            f"[STRESS_PANEL] Could not find either precomputed stress columns or an RL_MW column in:\n"
            f"  {stress_path}\n"
            f"Columns: {list(stress.columns)}"
        )

    out = stress[["province", "timestamp_utc", rl_col]].copy()
    out = out.rename(columns={rl_col: "RL_MW"})
    out["RL_MW"] = pd.to_numeric(out["RL_MW"], errors="coerce").fillna(0.0)

    # RL convention used here:
    #   RL_MW > 0  => shortage
    #   RL_MW < 0  => surplus
    out["rl_minus_raw"] = out["RL_MW"].clip(lower=0.0)
    out["rl_plus_raw"] = (-out["RL_MW"]).clip(lower=0.0)

    out["rl_minus_norm"] = (
        out.groupby(["province", out["timestamp_utc"].dt.year], sort=False)["rl_minus_raw"]
        .transform(_minmax01)
        .astype(float)
    )
    out["rl_plus_norm"] = (
        out.groupby(["province", out["timestamp_utc"].dt.year], sort=False)["rl_plus_raw"]
        .transform(_minmax01)
        .astype(float)
    )

    out["s_minus_norm"] = out["rl_minus_norm"]
    out["s_plus_norm"] = out["rl_plus_norm"]

    if structural_col is not None:
        out["structural_exposure"] = pd.to_numeric(stress[structural_col], errors="coerce")

    keep = [
        "province",
        "timestamp_utc",
        "rl_plus_norm",
        "rl_minus_norm",
        "s_plus_norm",
        "s_minus_norm",
        "structural_exposure" if "structural_exposure" in out.columns else None,
    ]
    keep = [c for c in keep if c is not None]

    out = out[keep].copy()
    out = out.dropna(subset=["province", "timestamp_utc"]).copy()
    _assert_unique_keys(out, ["province", "timestamp_utc"], "STRESS_PANEL")
    return out.sort_values(["province", "timestamp_utc"]).reset_index(drop=True), "derived_from_RL_MW"


def _stress_panel_span_info(stress: pd.DataFrame) -> dict[str, object]:
    years = sorted(int(y) for y in stress["timestamp_utc"].dt.year.dropna().unique().tolist())
    return {
        "stress_start_utc": str(pd.Timestamp(stress["timestamp_utc"].min())),
        "stress_end_utc": str(pd.Timestamp(stress["timestamp_utc"].max())),
        "stress_years_present": years,
    }


def _looks_like_future_scaffold(stress_path: Path, stress: pd.DataFrame) -> bool:
    path_s = str(stress_path).lower()
    years = sorted(int(y) for y in stress["timestamp_utc"].dt.year.dropna().unique().tolist())
    if not years:
        return False

    filename_signal = (
        "2025_2035" in path_s
        or "2025-2035" in path_s
        or "learned_optionb" in path_s
        or "optionb" in path_s
    )

    temporal_signal = (min(years) >= 2025 and max(years) >= 2030)

    return filename_signal or temporal_signal


# =============================================================================
# NATIONAL GOPACS ACTIVITY
# =============================================================================
def _activity_metadata_path(activity_path: Path) -> Path:
    return activity_path.with_name(f"{activity_path.stem}_metadata.json")


def _load_step03_metadata_if_available(activity_path: Path) -> dict | None:
    meta_path = _activity_metadata_path(activity_path)
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    return payload


def _infer_observed_window_from_series(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    nonzero = df[
        (pd.to_numeric(df["gopacs_eur"], errors="coerce").fillna(0.0) > 0)
        | (pd.to_numeric(df["gopacs_mwh"], errors="coerce").fillna(0.0) > 0)
        | (pd.to_numeric(df["gopacs_activation"], errors="coerce").fillna(0.0) > 0)
    ].copy()

    if nonzero.empty:
        return pd.Timestamp(df["timestamp_utc"].min()), pd.Timestamp(df["timestamp_utc"].max())

    return pd.Timestamp(nonzero["timestamp_utc"].min()), pd.Timestamp(nonzero["timestamp_utc"].max())


def _load_step03_activity_national(activity_path: Path) -> NationalActivitySummary:
    df = pd.read_csv(activity_path)
    df.columns = [str(c).strip() for c in df.columns]

    ts_col = _pick_column(df.columns, ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc"])
    df["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).copy()

    operator_col = _pick_column(df.columns, ["operator", "area", "scope"], required=False)
    eur_col = _pick_column(df.columns, ["gopacs_eur"], required=False)
    mwh_col = _pick_column(df.columns, ["gopacs_mwh"], required=False)
    activation_col = _pick_column(df.columns, ["gopacs_activation"], required=False)

    if eur_col is None and mwh_col is None and activation_col is None:
        raise ValueError(f"[STEP03_ACTIVITY] No usable GOPACS columns in {activity_path}")

    if operator_col is not None:
        tmp = df.copy()
        tmp[operator_col] = tmp[operator_col].astype(str).str.strip().str.lower()
        nat = tmp[tmp[operator_col] == "national"].copy()
        if nat.empty:
            nat = (
                tmp.groupby("timestamp_utc", as_index=False)[
                    [c for c in [eur_col, mwh_col, activation_col] if c is not None]
                ]
                .sum()
                .copy()
            )
    else:
        nat = df.copy()

    out = nat[["timestamp_utc"]].copy()
    out["gopacs_eur"] = pd.to_numeric(nat[eur_col], errors="coerce").fillna(0.0) if eur_col else 0.0
    out["gopacs_mwh"] = pd.to_numeric(nat[mwh_col], errors="coerce").fillna(0.0) if mwh_col else 0.0

    if activation_col is not None:
        out["gopacs_activation"] = pd.to_numeric(nat[activation_col], errors="coerce").fillna(0.0)
        out["gopacs_activation"] = (out["gopacs_activation"] > 0).astype(int)
    else:
        out["gopacs_activation"] = ((out["gopacs_eur"] > 0) | (out["gopacs_mwh"] > 0)).astype(int)

    out = (
        out.groupby("timestamp_utc", as_index=False)[["gopacs_eur", "gopacs_mwh", "gopacs_activation"]]
        .sum()
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    out["gopacs_activation"] = (out["gopacs_activation"] > 0).astype(int)

    raw_panel_start = pd.Timestamp(out["timestamp_utc"].min())
    raw_panel_end = pd.Timestamp(out["timestamp_utc"].max())

    meta = _load_step03_metadata_if_available(activity_path)
    meta_path = _activity_metadata_path(activity_path) if meta is not None else None

    if meta is not None and meta.get("observed_start_utc") and meta.get("observed_end_utc"):
        observed_start = pd.Timestamp(meta["observed_start_utc"])
        observed_end = pd.Timestamp(meta["observed_end_utc"])
    else:
        observed_start, observed_end = _infer_observed_window_from_series(out)

    out = out[
        (out["timestamp_utc"] >= observed_start) &
        (out["timestamp_utc"] <= observed_end)
    ].copy()

    if out.empty:
        raise ValueError(
            f"[STEP03_ACTIVITY] Observed activity window is empty after restriction.\n"
            f"activity_path={activity_path}\n"
            f"observed_start={observed_start}\n"
            f"observed_end={observed_end}"
        )

    return NationalActivitySummary(
        national_hourly=out,
        source_path=activity_path,
        mode="STEP03_ACTIVITY",
        observed_start_utc=observed_start,
        observed_end_utc=observed_end,
        raw_panel_start_utc=raw_panel_start,
        raw_panel_end_utc=raw_panel_end,
        metadata_path=meta_path,
    )


# =============================================================================
# LOCAL GOPACS ACTIVITY
# =============================================================================
def _load_step02_localized_panel(panel_path: Path) -> LocalActivitySummary:
    panel = pd.read_csv(panel_path)
    panel.columns = [str(c).strip() for c in panel.columns]

    ts_col = _pick_column(panel.columns, ["timestamp_utc", "utc_timestamp", "utc_time", "time_utc"])
    province_col = _pick_column(panel.columns, ["province"])

    panel["timestamp_utc"] = pd.to_datetime(panel[ts_col], utc=True, errors="coerce")
    panel["province"] = panel[province_col].astype(str).str.strip()

    eur_col = _pick_column(
        panel.columns,
        [
            "gopacs_eur",
            "announcement_eur",
            "eur",
            "eur_total",
            "cost_eur",
            "gopacs_cost_eur",
        ],
        required=False,
    )
    mwh_col = _pick_column(
        panel.columns,
        [
            "gopacs_mwh",
            "announcement_mwh",
            "announcement_mwh_equiv",
            "mwh",
            "mwh_equiv",
            "mwh_total",
            "gopacs_mwh_equiv",
        ],
        required=False,
    )
    activation_col = _pick_column(
        panel.columns,
        [
            "gopacs_activation",
            "activation",
            "announcement_activation",
            "is_active",
            "has_announcement",
        ],
        required=False,
    )

    if eur_col is None and mwh_col is None and activation_col is None:
        raise ValueError(
            f"[STEP02] Could not find usable GOPACS value columns in {panel_path}. "
            f"Columns: {list(panel.columns)}"
        )

    local = panel[["province", "timestamp_utc"]].copy()
    local["gopacs_eur_local"] = pd.to_numeric(panel[eur_col], errors="coerce").fillna(0.0) if eur_col else 0.0
    local["gopacs_mwh_local"] = pd.to_numeric(panel[mwh_col], errors="coerce").fillna(0.0) if mwh_col else 0.0

    if activation_col is not None:
        local["gopacs_activation_local"] = pd.to_numeric(panel[activation_col], errors="coerce").fillna(0.0)
        local["gopacs_activation_local"] = (local["gopacs_activation_local"] > 0).astype(int)
    else:
        local["gopacs_activation_local"] = (
            (local["gopacs_eur_local"] > 0) | (local["gopacs_mwh_local"] > 0)
        ).astype(int)

    local = (
        local.groupby(["province", "timestamp_utc"], as_index=False)[
            ["gopacs_eur_local", "gopacs_mwh_local", "gopacs_activation_local"]
        ]
        .sum()
        .sort_values(["province", "timestamp_utc"])
        .reset_index(drop=True)
    )
    local["gopacs_activation_local"] = (local["gopacs_activation_local"] > 0).astype(int)

    _assert_unique_keys(local, ["province", "timestamp_utc"], "STEP02_LOCAL")

    return LocalActivitySummary(
        local_hourly=local,
        source_path=panel_path,
        mode="STEP02_LOCALIZED",
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    allow_scenario_scaffold = _get_bool(
        "ALLOW_SCENARIO_STRESS_SCAFFOLD",
        env_name="ALLOW_SCENARIO_STRESS_SCAFFOLD",
        default=False,
    )

    configured_stress_path = _get_path(
        "HISTORICAL_PROVINCE_STRESS_PATH",
        "HISTORICAL_BACKBONE_PATH",
        "PAPER_BACKBONE_PATH",
        "PROVINCIAL_RESIDUAL_LOAD_SOURCE_PATH",
        "PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_PATH",
        "PROVINCE_STRESS_PATH",
        "PROVINCE_STRESS_PANEL_PATH",
        "RESIDUAL_LOAD_PANEL_PATH",
        "PUBLIC_STRESS_PATH",
        "STATE_BACKBONE_PATH",
        env_name="PROVINCIAL_RESIDUAL_LOAD_SOURCE_PATH",
        default=REPO_ROOT / "data" / "derived" / "provincial_residual_load_2025_2035_learned_optionB.csv",
    )

    configured_activity_path = _get_path(
        "ACTIVITY_PATH",
        "ACTIVITY_DIAGNOSTIC_PATH",
        "GOPACS_CALLED_ACTIVITY_PATH",
        env_name="ACTIVITY_PATH",
        default=REPO_ROOT / "data" / "derived" / "gopacs_called_activity_hour.csv",
    )

    configured_step02_path = _get_path(
        "GOPACS_ANNOUNCEMENT_PANEL_PATH",
        "ANNOUNCEMENT_PANEL_PATH",
        "STEP02_PANEL_PATH",
        env_name="GOPACS_ANNOUNCEMENT_PANEL_PATH",
        default=REPO_ROOT / "data" / "derived" / "gopacs_announcement_province_hour.csv",
    )

    output_path = _get_path(
        "BURDEN_PATH",
        "BURDEN_PROXY_PATH",
        env_name="BURDEN_PATH",
        default=REPO_ROOT / "data" / "derived" / "burden_proxy_province_hour.csv",
    )

    qc_dir = _get_path(
        "QC_DIR",
        env_name="QC_DIR",
        default=REPO_ROOT / "qc",
    )

    stress_path = _require_existing_path("province stress panel", configured_stress_path)

    use_step03 = configured_activity_path.exists()
    use_step02 = configured_step02_path.exists()

    if not use_step03 and not use_step02:
        raise FileNotFoundError(
            "Need at least one GOPACS activity input.\n"
            f"Missing step03 activity file:\n  {configured_activity_path}\n"
            f"Missing step02 announcement panel:\n  {configured_step02_path}"
        )

    print("=" * 72)
    print("STEP 04: BUILD BURDEN PROXY PANEL")
    print("=" * 72)
    print(f"Stress panel : {stress_path}")
    print(f"Step03 act.  : {configured_activity_path}")
    print(f"Step02 panel : {configured_step02_path}")
    print(f"Output       : {output_path}")

    stress, stress_mode = _load_stress_panel(stress_path)
    stress_span = _stress_panel_span_info(stress)
    stress_is_future_scaffold = _looks_like_future_scaffold(stress_path, stress)

    if stress_is_future_scaffold and not allow_scenario_scaffold:
        raise RuntimeError(
            "Selected stress panel looks like a future/scenario scaffold, not the historical paper backbone.\n"
            f"Path: {stress_path}\n"
            f"Years present: {stress_span['stress_years_present']}\n\n"
            "This would silently leak the 2025–2035 scaffold into the burden proxy panel.\n"
            "Set a true historical stress path in pipeline_config.py, or explicitly override with\n"
            "ALLOW_SCENARIO_STRESS_SCAFFOLD=true only for temporary scaffold diagnostics."
        )

    if use_step03:
        national_activity = _load_step03_activity_national(configured_activity_path)
        print(f"National GOPACS source: {configured_activity_path} [{national_activity.mode}]")
        if national_activity.metadata_path is not None:
            print(f"Step03 metadata       : {national_activity.metadata_path}")
    else:
        local_tmp = _load_step02_localized_panel(configured_step02_path)
        nat = (
            local_tmp.local_hourly.groupby("timestamp_utc", as_index=False)[
                ["gopacs_eur_local", "gopacs_mwh_local", "gopacs_activation_local"]
            ]
            .sum()
            .rename(
                columns={
                    "gopacs_eur_local": "gopacs_eur",
                    "gopacs_mwh_local": "gopacs_mwh",
                    "gopacs_activation_local": "gopacs_activation",
                }
            )
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )
        nat["gopacs_activation"] = (nat["gopacs_activation"] > 0).astype(int)

        if nat.empty:
            raise ValueError("[STEP02_LOCALIZED_NATIONAL_FALLBACK] Could not build any national GOPACS activity.")

        national_activity = NationalActivitySummary(
            national_hourly=nat,
            source_path=configured_step02_path,
            mode="STEP02_LOCALIZED_NATIONAL_FALLBACK",
            observed_start_utc=pd.Timestamp(nat["timestamp_utc"].min()),
            observed_end_utc=pd.Timestamp(nat["timestamp_utc"].max()),
            raw_panel_start_utc=pd.Timestamp(nat["timestamp_utc"].min()),
            raw_panel_end_utc=pd.Timestamp(nat["timestamp_utc"].max()),
            metadata_path=None,
        )
        print(f"National GOPACS source: {configured_step02_path} [{national_activity.mode}]")

    if use_step02:
        local_activity = _load_step02_localized_panel(configured_step02_path)
        print(f"Local GOPACS source   : {configured_step02_path} [{local_activity.mode}]")
    else:
        local_activity = LocalActivitySummary(
            local_hourly=pd.DataFrame(
                columns=["province", "timestamp_utc", "gopacs_eur_local", "gopacs_mwh_local", "gopacs_activation_local"]
            ),
            source_path=configured_activity_path,
            mode="NO_LOCAL_PANEL",
        )
        print("Local GOPACS source   : none [NO_LOCAL_PANEL]")

    gopacs_start = national_activity.observed_start_utc
    gopacs_end = national_activity.observed_end_utc

    print(f"GOPACS observed window: {gopacs_start} .. {gopacs_end}")
    print(f"Step03 raw panel span : {national_activity.raw_panel_start_utc} .. {national_activity.raw_panel_end_utc}")

    # Restrict stress to the observed GOPACS window, not to padded step03 support.
    stress = stress[
        (stress["timestamp_utc"] >= gopacs_start) &
        (stress["timestamp_utc"] <= gopacs_end)
    ].copy()

    if stress.empty:
        raise ValueError(
            "Stress panel has no overlap with the observed GOPACS activity window.\n"
            f"GOPACS observed window: {gopacs_start} .. {gopacs_end}\n"
            f"Stress panel path      : {stress_path}"
        )

    national_hourly = national_activity.national_hourly.copy()
    national_hourly = national_hourly[
        (national_hourly["timestamp_utc"] >= gopacs_start) &
        (national_hourly["timestamp_utc"] <= gopacs_end)
    ].copy()

    panel = stress.merge(
        national_hourly,
        on="timestamp_utc",
        how="inner",
        validate="many_to_one",
    )

    if panel.empty:
        raise ValueError(
            "No rows remained after intersecting stress panel with observed national GOPACS window."
        )

    if not local_activity.local_hourly.empty:
        local_hourly = local_activity.local_hourly.copy()
        local_hourly = local_hourly[
            (local_hourly["timestamp_utc"] >= gopacs_start) &
            (local_hourly["timestamp_utc"] <= gopacs_end)
        ].copy()

        panel = panel.merge(
            local_hourly,
            on=["province", "timestamp_utc"],
            how="left",
            validate="one_to_one",
        )
    else:
        panel["gopacs_eur_local"] = 0.0
        panel["gopacs_mwh_local"] = 0.0
        panel["gopacs_activation_local"] = 0

    for col in [
        "gopacs_eur",
        "gopacs_mwh",
        "gopacs_activation",
        "gopacs_eur_local",
        "gopacs_mwh_local",
        "gopacs_activation_local",
    ]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)

    panel["gopacs_activation"] = (panel["gopacs_activation"] > 0).astype(int)
    panel["gopacs_activation_local"] = (panel["gopacs_activation_local"] > 0).astype(int)

    # -------------------------------------------------------------------------
    # Burden aggregate:
    #   preferred locked baseline: log(1 + GOPACS_EUR_t)
    #   diagnostic fallback when EUR is unavailable: log(1 + GOPACS_MWH_t)
    # -------------------------------------------------------------------------
    if float(panel["gopacs_eur"].sum()) > 0:
        panel["B_agg"] = np.log1p(panel["gopacs_eur"].clip(lower=0.0))
        b_agg_basis = "EUR"
    else:
        panel["B_agg"] = np.log1p(panel["gopacs_mwh"].clip(lower=0.0))
        b_agg_basis = "MWH_FALLBACK_NO_EUR"

    exposure_argument = (
        panel["rl_plus_norm"].fillna(0.0) +
        panel["s_plus_norm"].fillna(0.0)
    ).clip(lower=-50.0, upper=50.0)

    panel["_raw_exposure"] = np.exp(exposure_argument)
    denom = panel.groupby("timestamp_utc")["_raw_exposure"].transform("sum")
    panel["E_exposure"] = np.where(denom > 0, panel["_raw_exposure"] / denom, 0.0)

    panel["B_proxy_baseline"] = panel["B_agg"] * panel["E_exposure"]
    panel["B_proxy_mwh"] = np.log1p(panel["gopacs_mwh"].clip(lower=0.0)) * panel["E_exposure"]
    panel["B_proxy_activation"] = panel["gopacs_activation"].astype(float) * panel["E_exposure"]

    panel["year"] = panel["timestamp_utc"].dt.year
    panel["month"] = panel["timestamp_utc"].dt.month

    _assert_unique_keys(panel, ["province", "timestamp_utc"], "STEP04_PANEL")

    keep = [
        "province",
        "timestamp_utc",
        "year",
        "month",
        "rl_plus_norm",
        "rl_minus_norm",
        "s_plus_norm",
        "s_minus_norm",
        "structural_exposure" if "structural_exposure" in panel.columns else None,
        "gopacs_eur",
        "gopacs_mwh",
        "gopacs_activation",
        "gopacs_eur_local",
        "gopacs_mwh_local",
        "gopacs_activation_local",
        "B_agg",
        "E_exposure",
        "B_proxy_baseline",
        "B_proxy_mwh",
        "B_proxy_activation",
    ]
    keep = [c for c in keep if c is not None]

    out = panel[keep].sort_values(["province", "timestamp_utc"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(output_path, out)

    qc_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": int(len(out)),
        "n_provinces": int(out["province"].nunique()),
        "stress_input_path": str(stress_path),
        "stress_mode": stress_mode,
        "stress_is_future_scaffold": bool(stress_is_future_scaffold),
        "allow_scenario_scaffold_override": bool(allow_scenario_scaffold),
        "stress_start_utc": stress_span["stress_start_utc"],
        "stress_end_utc": stress_span["stress_end_utc"],
        "stress_years_present": stress_span["stress_years_present"],
        "national_gopacs_input_path": str(national_activity.source_path),
        "national_gopacs_mode": national_activity.mode,
        "national_gopacs_metadata_path": str(national_activity.metadata_path) if national_activity.metadata_path else None,
        "national_gopacs_observed_start": str(national_activity.observed_start_utc),
        "national_gopacs_observed_end": str(national_activity.observed_end_utc),
        "national_gopacs_raw_panel_start": str(national_activity.raw_panel_start_utc),
        "national_gopacs_raw_panel_end": str(national_activity.raw_panel_end_utc),
        "local_gopacs_input_path": str(local_activity.source_path),
        "local_gopacs_mode": local_activity.mode,
        "panel_start": str(out["timestamp_utc"].min()) if len(out) else None,
        "panel_end": str(out["timestamp_utc"].max()) if len(out) else None,
        "gopacs_window_start": str(gopacs_start),
        "gopacs_window_end": str(gopacs_end),
        "B_agg_basis": b_agg_basis,
        "sum_E_exposure_max_abs_error": float(
            out.groupby("timestamp_utc")["E_exposure"].sum().sub(1.0).abs().max() if len(out) else 0.0
        ),
        "baseline_nonzero_share": float((out["B_proxy_baseline"] > 0).mean()) if len(out) else 0.0,
        "national_gopacs_eur_nonzero_share": float((out["gopacs_eur"] > 0).mean()) if len(out) else 0.0,
        "national_gopacs_mwh_nonzero_share": float((out["gopacs_mwh"] > 0).mean()) if len(out) else 0.0,
        "local_gopacs_mwh_nonzero_share": float((out["gopacs_mwh_local"] > 0).mean()) if len(out) else 0.0,
        "years_present": sorted(int(y) for y in out["year"].dropna().unique().tolist()) if len(out) else [],
    }

    qc_path = qc_dir / "step04_burden_proxy_summary.json"
    _atomic_write_json(qc_path, summary)

    print(f"\nOutput rows: {len(out):,}")
    print(f"Provinces  : {out['province'].nunique()}")
    print(f"Stress mode: {stress_mode}")
    print(f"B_agg basis: {b_agg_basis}")
    print(f"Panel start: {summary['panel_start']}")
    print(f"Panel end  : {summary['panel_end']}")
    print(f"\n✓ Written: {output_path}")
    print(f"✓ Written: {qc_path}")


if __name__ == "__main__":
    main()