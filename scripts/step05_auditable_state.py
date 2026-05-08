"""
step05_auditable_state.py
=========================

Build the final province-hour auditable-state master panel aligned to the
current locked paper pipeline and exact-path configuration.

Required output columns
-----------------------
province
timestamp_utc
year
hour_of_year
hour_of_week
month
is_weekend
is_holiday
price_da_eur_per_mwh
price_da_norm
rl_plus_norm
rl_minus_norm
s_plus_norm
s_minus_norm
structural_exposure
gopacs_eur
gopacs_mwh
gopacs_activation
B_agg
E_exposure
B_proxy_baseline

Optional
--------
split
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


# =============================================================================
# PRICE LOADER
# =============================================================================
def _load_price_panel(price_path: Path) -> pd.DataFrame:
    price = pd.read_csv(price_path, dtype=str)
    price.columns = [str(c).strip() for c in price.columns]

    ts_col = _pick_column(
        price.columns,
        [
            "timestamp_utc",
            "utc_timestamp",
            "utc_time",
            "datetime",
            "timestamp",
            "time",
            "mtu",
            "Datetime (UTC)",
            "datetime (utc)",
        ],
    )
    val_col = _pick_column(
        price.columns,
        [
            "price_da_eur_per_mwh",
            "price_da_eur_mwh",
            "price",
            "eur/mwh",
            "eur_mwh",
            "value",
            "day_ahead_price",
            "Price (EUR/MWhe)",
            "price (eur/mwhe)",
        ],
    )

    out = pd.DataFrame()
    out["timestamp_utc"] = pd.to_datetime(price[ts_col], errors="coerce", utc=True).dt.floor("h")
    out["price_da_eur_per_mwh"] = pd.to_numeric(price[val_col], errors="coerce")

    out = out.dropna(subset=["timestamp_utc"]).copy()
    out = (
        out.groupby("timestamp_utc", as_index=False)["price_da_eur_per_mwh"]
        .mean()
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    out["year_tmp"] = out["timestamp_utc"].dt.year
    grp = out.groupby("year_tmp")["price_da_eur_per_mwh"]
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0, np.nan)
    out["price_da_norm"] = (
        (out["price_da_eur_per_mwh"] - mean) / std
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out.drop(columns=["year_tmp"])

    _assert_unique_keys(out, ["timestamp_utc"], "PRICE_PANEL")
    return out


# =============================================================================
# STRESS LOADER
# =============================================================================
def _derive_norm_by_group(df: pd.DataFrame, group_cols: list[str], source_col: str) -> pd.Series:
    def _mm(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
        lo = float(x.min())
        hi = float(x.max())
        if hi <= lo + 1e-12:
            return pd.Series(np.zeros(len(x), dtype=float), index=s.index)
        return pd.Series(((x - lo) / (hi - lo)).clip(0.0, 1.0), index=s.index)

    return df.groupby(group_cols, group_keys=False)[source_col].apply(_mm)


def _load_stress_panel(stress_path: Path) -> tuple[pd.DataFrame, str]:
    stress = pd.read_csv(stress_path, parse_dates=["timestamp_utc"])
    stress.columns = [str(c).strip() for c in stress.columns]
    stress["timestamp_utc"] = pd.to_datetime(stress["timestamp_utc"], utc=True)

    province_col = _pick_column(stress.columns, ["province"])
    stress["province"] = stress[province_col].astype(str).str.strip()

    has_prebuilt = all(
        c in stress.columns
        for c in ["rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm"]
    )

    if has_prebuilt:
        structural_col = _pick_column(stress.columns, ["structural_exposure"], required=False)
        keep = ["province", "timestamp_utc", "rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm"]
        if structural_col is not None:
            keep.append(structural_col)

        out = stress[keep].copy()
        if structural_col is not None and structural_col != "structural_exposure":
            out = out.rename(columns={structural_col: "structural_exposure"})
        if "structural_exposure" not in out.columns:
            out["structural_exposure"] = 0.0

        mode = "prebuilt_stress_panel"
    else:
        rl_col = _pick_column(stress.columns, ["RL_MW", "rl_mw", "residual_load_mw"], required=False)
        if rl_col is None:
            raise ValueError(
                f"[STRESS_PANEL] Could not find either prebuilt stress columns or RL_MW in {stress_path}.\n"
                f"Columns were: {list(stress.columns)}"
            )

        out = stress[["province", "timestamp_utc", rl_col]].copy().rename(columns={rl_col: "RL_MW"})
        out["RL_MW"] = pd.to_numeric(out["RL_MW"], errors="coerce").fillna(0.0)

        out["rl_minus_raw"] = out["RL_MW"].clip(lower=0.0)
        out["rl_plus_raw"] = (-out["RL_MW"]).clip(lower=0.0)

        out["year_norm"] = out["timestamp_utc"].dt.year.astype(int)
        out["rl_minus_norm"] = _derive_norm_by_group(out, ["province", "year_norm"], "rl_minus_raw")
        out["rl_plus_norm"] = _derive_norm_by_group(out, ["province", "year_norm"], "rl_plus_raw")

        out["s_minus_norm"] = out["rl_minus_norm"]
        out["s_plus_norm"] = out["rl_plus_norm"]
        out["structural_exposure"] = 0.0

        out = out.drop(columns=["RL_MW", "rl_minus_raw", "rl_plus_raw", "year_norm"])
        mode = "derived_from_RL_MW"

    for col in ["rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm", "structural_exposure"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["structural_exposure"] = out["structural_exposure"].fillna(0.0)

    _assert_unique_keys(out, ["province", "timestamp_utc"], "STRESS_PANEL")
    out = out.sort_values(["province", "timestamp_utc"]).reset_index(drop=True)
    return out, mode


# =============================================================================
# BURDEN LOADER
# =============================================================================
def _load_burden_panel(path: Path) -> pd.DataFrame:
    burden = pd.read_csv(path, parse_dates=["timestamp_utc"])
    burden.columns = [str(c).strip() for c in burden.columns]
    burden["timestamp_utc"] = pd.to_datetime(burden["timestamp_utc"], utc=True)

    province_col = _pick_column(burden.columns, ["province"])
    burden["province"] = burden[province_col].astype(str).str.strip()

    required = [
        "gopacs_eur",
        "gopacs_mwh",
        "gopacs_activation",
        "B_agg",
        "E_exposure",
        "B_proxy_baseline",
    ]
    missing = [c for c in required if c not in burden.columns]
    if missing:
        raise ValueError(f"[BURDEN_PANEL] Missing required columns {missing}. Found: {list(burden.columns)}")

    keep = ["province", "timestamp_utc"] + required
    out = burden[keep].copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    _assert_unique_keys(out, ["province", "timestamp_utc"], "BURDEN_PANEL")
    return out.sort_values(["province", "timestamp_utc"]).reset_index(drop=True)


# =============================================================================
# CALENDAR HELPERS
# =============================================================================
def _holiday_series(timestamps_utc: pd.Series) -> pd.Series:
    try:
        import holidays  # type: ignore
    except Exception:
        return pd.Series(False, index=timestamps_utc.index)

    local = timestamps_utc.dt.tz_convert("Europe/Amsterdam")
    years = sorted(local.dt.year.dropna().unique().tolist())
    nl_holidays = holidays.Netherlands(years=years)
    return local.dt.date.apply(lambda d: d in nl_holidays)


def _compute_local_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    local = out["timestamp_utc"].dt.tz_convert("Europe/Amsterdam")

    out["year"] = local.dt.year.astype(int)
    out["month"] = local.dt.month.astype(int)
    out["hour_of_day"] = local.dt.hour.astype(int)
    out["weekday"] = local.dt.weekday.astype(int)

    jan1_local = pd.to_datetime(out["year"].astype(str) + "-01-01 00:00:00").dt.tz_localize("Europe/Amsterdam")
    out["hour_of_year"] = (((local - jan1_local).dt.total_seconds() // 3600).astype(int) + 1)

    out["hour_of_week"] = (out["weekday"] * 24 + out["hour_of_day"]).astype(int)
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)
    out["is_holiday"] = _holiday_series(out["timestamp_utc"]).astype(int)

    return out.drop(columns=["hour_of_day", "weekday"])


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    price_path = _require_existing_path(
        "day-ahead price panel",
        _get_path(
            "PRICES_PATH",
            "PRICE_DA_PATH",
            "DAY_AHEAD_PRICE_PATH",
            "DAY_AHEAD_PRICES_PATH",
            env_name="PRICE_DA_PATH",
            default=REPO_ROOT / "data" / "external" / "Netherlands_prices.csv",
        ),
    )

    stress_path = _require_existing_path(
        "province stress panel",
        _get_path(
            "PROVINCIAL_RESIDUAL_LOAD_SOURCE_PATH",
            "PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_PATH",
            "PROVINCE_STRESS_PATH",
            "PROVINCE_STRESS_PANEL_PATH",
            "RESIDUAL_LOAD_PANEL_PATH",
            "PUBLIC_STRESS_PATH",
            "STATE_BACKBONE_PATH",
            env_name="PROVINCIAL_RESIDUAL_LOAD_SOURCE_PATH",
            default=REPO_ROOT / "data" / "derived" / "provincial_residual_load_2025_2035_learned_optionB.csv",
        ),
    )

    burden_path = _require_existing_path(
        "burden proxy panel",
        _get_path(
            "BURDEN_PATH",
            "BURDEN_PROXY_PATH",
            env_name="BURDEN_PATH",
            default=REPO_ROOT / "data" / "derived" / "burden_proxy_province_hour.csv",
        ),
    )

    state_path = _get_path(
        "STATE_PATH",
        "AUDITABLE_STATE_PATH",
        env_name="STATE_PATH",
        default=REPO_ROOT / "data" / "derived" / "auditable_state_province_hour.csv",
    )

    splits_path = _get_path(
        "SPLITS_PATH",
        env_name="SPLITS_PATH",
        default=REPO_ROOT / "data" / "derived" / "paper_holdout_splits.csv",
    )

    qc_dir = _get_path(
        "QC_DIR",
        env_name="QC_DIR",
        default=REPO_ROOT / "qc",
    )

    print("=" * 72)
    print("STEP 05: BUILD AUDITABLE STATE MASTER PANEL")
    print("=" * 72)
    print(f"Prices  : {price_path}")
    print(f"Stress  : {stress_path}")
    print(f"Burden  : {burden_path}")
    print(f"Output  : {state_path}")

    price = _load_price_panel(price_path)
    stress, stress_mode = _load_stress_panel(stress_path)
    burden = _load_burden_panel(burden_path)

    panel = stress.merge(price, on="timestamp_utc", how="left", validate="many_to_one")
    panel = panel.merge(
        burden,
        on=["province", "timestamp_utc"],
        how="inner",
        validate="one_to_one",
    )

    panel = _compute_local_calendar(panel)

    numeric_cols = [
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

    for col in ["gopacs_eur", "gopacs_mwh", "gopacs_activation", "B_agg", "E_exposure", "B_proxy_baseline"]:
        panel[col] = panel[col].fillna(0.0)

    panel = panel.sort_values(["province", "timestamp_utc"]).reset_index(drop=True)
    for col in ["price_da_eur_per_mwh", "price_da_norm", "rl_plus_norm", "rl_minus_norm", "s_plus_norm", "s_minus_norm"]:
        panel[col] = panel.groupby("province")[col].transform(lambda s: s.ffill().bfill())

    required_nonnull = [
        "price_da_eur_per_mwh",
        "price_da_norm",
        "rl_plus_norm",
        "rl_minus_norm",
        "s_plus_norm",
        "s_minus_norm",
    ]
    bad_required = {c: int(panel[c].isna().sum()) for c in required_nonnull if panel[c].isna().any()}
    if bad_required:
        raise ValueError(f"[STATE_PANEL] Missing required values after alignment/fill: {bad_required}")

    train_years = set(getattr(cfg, "TRAIN_YEARS", []))
    val_years = set(getattr(cfg, "VAL_YEARS", []))
    test_years = set(getattr(cfg, "TEST_YEARS", []))

    def _split(year: int) -> str:
        if year in train_years:
            return "train"
        if year in val_years:
            return "val"
        if year in test_years:
            return "test"
        return "out_of_scope"

    if train_years or val_years or test_years:
        panel["split"] = panel["year"].astype(int).map(_split)

    final_cols = [
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
    if "split" in panel.columns:
        final_cols.append("split")

    out = panel[final_cols].sort_values(["province", "timestamp_utc"]).reset_index(drop=True)

    missing_required = [c for c in final_cols if c != "split" and c not in out.columns]
    if missing_required:
        raise RuntimeError(f"Missing required output columns: {missing_required}")

    _assert_unique_keys(out, ["province", "timestamp_utc"], "AUDITABLE_STATE_PANEL")

    state_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(state_path, index=False, encoding="utf-8")

    if "split" in out.columns:
        splits = out[["province", "timestamp_utc", "year", "split"]].copy()
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(splits_path, index=False, encoding="utf-8")

    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_summary = {
        "rows": int(len(out)),
        "n_provinces": int(out["province"].nunique()),
        "price_input_path": str(price_path),
        "stress_input_path": str(stress_path),
        "stress_mode": stress_mode,
        "burden_input_path": str(burden_path),
        "output_path": str(state_path),
        "panel_start": str(out["timestamp_utc"].min()) if len(out) else None,
        "panel_end": str(out["timestamp_utc"].max()) if len(out) else None,
        "years_present": sorted(int(y) for y in out["year"].dropna().unique().tolist()) if len(out) else [],
        "hour_of_year_min": int(out["hour_of_year"].min()) if len(out) else None,
        "hour_of_year_max": int(out["hour_of_year"].max()) if len(out) else None,
        "price_missing_after_fill": int(out["price_da_eur_per_mwh"].isna().sum()),
        "baseline_nonzero_share": float((out["B_proxy_baseline"] > 0).mean()) if len(out) else 0.0,
    }
    qc_path = qc_dir / "step05_auditable_state_summary.json"
    with qc_path.open("w", encoding="utf-8") as f:
        json.dump(qc_summary, f, indent=2)

    print(f"\nOutput shape : {out.shape}")
    print(f"Provinces    : {out['province'].nunique()}")
    print(f"Stress mode  : {stress_mode}")
    print(f"Time range   : {out['timestamp_utc'].min()} -> {out['timestamp_utc'].max()}")
    print(f"\n✓ Written: {state_path}")
    print(f"✓ Written: {qc_path}")
    if "split" in out.columns:
        print(f"✓ Written: {splits_path}")


if __name__ == "__main__":
    main()