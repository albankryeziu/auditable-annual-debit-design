#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_province_hour_public_stress_panel.py

Purpose
-------
Build the historical paper-backbone province-hour public stress panel used by
step04_burden_proxy.py.

This is the missing script between:
    - national/provincial public load structure, and
    - burden-proxy construction.

Important distinction
---------------------
This script is for the MAIN PAPER branch.

It does NOT use the future 2025-2035 residual-load scenario scaffold as the
stress backbone. Instead it builds an auditable, public-data province-hour
stress panel over the paper years using:

    1) national hourly load templates from public historical load files,
    2) public annual national totals,
    3) provincial HH/CO annual share structure,
    4) public day-ahead prices.

Output contract
---------------
Main output:
    data/derived/province_hour_public_stress_panel.csv

QC output:
    qc/build_province_hour_public_stress_panel_summary.json

Required output columns for downstream step04:
    province
    timestamp_utc
    year
    month
    hour_of_year
    nat_load_MW
    prov_load_MW
    province_load_share
    price_da_eur_mwh
    price_da_norm
    rl_plus_norm
    rl_minus_norm
    s_plus_norm
    s_minus_norm
    structural_exposure

Design logic
------------
- This panel is a PUBLIC auditable stress proxy, not a physical redispatch/PTDF
  reconstruction.
- Province-hour load is built from national hourly load and provincial HH/CO
  annual shares.
- Directional stress is then constructed from:
      high-load pressure  -> rl_plus_norm
      low-load slack      -> rl_minus_norm
      price-weighted high-load pressure -> s_plus_norm
      price-weighted low-load slack     -> s_minus_norm
- structural_exposure is a province-year structural load-share proxy.

Paper scope
-----------
This script targets cfg.ALL_YEARS by default. Under your current pipeline_config,
that means:
    train = 2021, 2022, 2023
    val   = 2024
    test  = 2025

No recursive file search is allowed.
Exact paths only:
    pipeline_config.py -> environment variable -> hard default
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
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
except Exception as exc:
    raise RuntimeError(
        f"Could not import pipeline_config.py from {REPO_ROOT}. "
        "Run this script from the final repo structure."
    ) from exc


# =============================================================================
# HELPERS
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


def _hours_in_year_utc(year: int) -> int:
    t0 = pd.Timestamp(f"{year}-01-01T00:00:00Z")
    t1 = pd.Timestamp(f"{year+1}-01-01T00:00:00Z")
    return int((t1 - t0) / pd.Timedelta(hours=1))


def _build_hourly_index(year: int) -> pd.DataFrame:
    H = _hours_in_year_utc(year)
    ts = pd.date_range(
        pd.Timestamp(f"{year}-01-01T00:00:00Z"),
        periods=H,
        freq="h",
    )
    out = pd.DataFrame({"timestamp_utc": ts})
    out["year"] = year
    out["hour_of_year"] = np.arange(1, H + 1, dtype=int)
    out["month"] = out["timestamp_utc"].dt.month
    return out


def _assign_template_years(available_years: list[int], target_years: list[int]) -> dict[int, int]:
    """
    Match leap years to leap years and non-leap years to non-leap years, cycling if needed.
    """
    available_years = sorted(int(y) for y in available_years)
    leap = [y for y in available_years if _hours_in_year_utc(y) == 8784]
    non_leap = [y for y in available_years if _hours_in_year_utc(y) == 8760]

    if not available_years:
        raise ValueError("No template years available.")

    out: dict[int, int] = {}
    i_leap = 0
    i_non = 0

    for y in target_years:
        if _hours_in_year_utc(y) == 8784 and leap:
            out[y] = leap[i_leap % len(leap)]
            i_leap += 1
        elif _hours_in_year_utc(y) == 8760 and non_leap:
            out[y] = non_leap[i_non % len(non_leap)]
            i_non += 1
        else:
            out[y] = available_years[(y - target_years[0]) % len(available_years)]
    return out


# =============================================================================
# DATA CONTAINERS
# =============================================================================
@dataclass(frozen=True)
class NationalHourlyTemplates:
    by_year: dict[int, pd.DataFrame]
    source_path: Path


# =============================================================================
# LOAD NATIONAL HOURLY TEMPLATE YEARS
# =============================================================================
def _load_national_hourly_templates(path: Path) -> NationalHourlyTemplates:
    """
    Use public historical hourly national load templates from the profile file.

    Preferred rule:
      - actual if available
      - otherwise forecast
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    ts_col = _pick_column(df.columns, ["utc_timestamp", "timestamp_utc", "utc_time", "time_utc"])
    actual_col = _pick_column(
        df.columns,
        ["NL_load_actual_entsoe_transparency", "nat_load_MW", "load_actual_mw"],
        required=False,
    )
    forecast_col = _pick_column(
        df.columns,
        ["NL_load_forecast_entsoe_transparency", "load_forecast_mw"],
        required=False,
    )

    if actual_col is None and forecast_col is None:
        raise ValueError(
            f"[NATIONAL_TEMPLATE] Could not find an actual/forecast national load column in {path}."
        )

    df["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.floor("h")
    df = df.dropna(subset=["timestamp_utc"]).copy()
    df["year"] = df["timestamp_utc"].dt.year

    if actual_col is not None:
        df["load_actual"] = pd.to_numeric(df[actual_col], errors="coerce")
    else:
        df["load_actual"] = np.nan

    if forecast_col is not None:
        df["load_forecast"] = pd.to_numeric(df[forecast_col], errors="coerce")
    else:
        df["load_forecast"] = np.nan

    df["nat_load_template_MW"] = df["load_actual"]
    miss = df["nat_load_template_MW"].isna()
    df.loc[miss, "nat_load_template_MW"] = df.loc[miss, "load_forecast"]

    df = df.dropna(subset=["nat_load_template_MW"]).copy()

    templates: dict[int, pd.DataFrame] = {}
    for year, g in df.groupby("year", sort=True):
        g = g[["timestamp_utc", "nat_load_template_MW"]].copy()
        g = (
            g.groupby("timestamp_utc", as_index=False)["nat_load_template_MW"]
            .mean()
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )

        idx = _build_hourly_index(int(year))[["timestamp_utc", "hour_of_year"]]
        merged = idx.merge(g, on="timestamp_utc", how="left", validate="one_to_one")

        missing_share = float(merged["nat_load_template_MW"].isna().mean())
        if missing_share > 0.02:
            continue

        merged["nat_load_template_MW"] = (
            merged["nat_load_template_MW"]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

        if merged["nat_load_template_MW"].isna().any():
            continue

        merged["year"] = int(year)
        templates[int(year)] = merged[["timestamp_utc", "year", "hour_of_year", "nat_load_template_MW"]].copy()

    if not templates:
        raise ValueError(f"[NATIONAL_TEMPLATE] No usable template years survived QC from {path}.")

    return NationalHourlyTemplates(by_year=templates, source_path=path)


# =============================================================================
# LOAD ANNUAL NATIONAL TOTALS
# =============================================================================
def _load_annual_national_totals(
    sector_mix_path: Path,
    annual_forecast_path: Path,
    target_years: list[int],
) -> pd.DataFrame:
    """
    Build annual national totals (MWh) for the paper years.

    Priority:
      - observed totals in national_sector_mix file, if present
      - else forecast totals from national_annual_load_2025_2035_forecast.csv
    """
    mix = pd.read_csv(sector_mix_path)
    mix.columns = [str(c).strip() for c in mix.columns]

    mix_year_col = _pick_column(mix.columns, ["year"])
    mix["year"] = pd.to_numeric(mix[mix_year_col], errors="coerce").astype("Int64")

    total_obs_col = _pick_column(
        mix.columns,
        ["nat_total_electricity_MWh_year_obs", "nat_total_electricity_mwh_year_obs"],
        required=False,
    )
    omega_hh_col = _pick_column(mix.columns, ["omega_hh"], required=False)
    omega_co_col = _pick_column(mix.columns, ["omega_co"], required=False)

    observed = mix[["year"]].copy()
    if total_obs_col is not None:
        observed["nat_annual_mwh"] = pd.to_numeric(mix[total_obs_col], errors="coerce")
    else:
        observed["nat_annual_mwh"] = np.nan

    if omega_hh_col is not None:
        observed["omega_hh"] = pd.to_numeric(mix[omega_hh_col], errors="coerce")
    else:
        observed["omega_hh"] = np.nan

    if omega_co_col is not None:
        observed["omega_co"] = pd.to_numeric(mix[omega_co_col], errors="coerce")
    else:
        observed["omega_co"] = np.nan

    forecast = pd.read_csv(annual_forecast_path)
    forecast.columns = [str(c).strip() for c in forecast.columns]
    fc_year_col = _pick_column(forecast.columns, ["year"])
    fc_total_col = _pick_column(
        forecast.columns,
        ["nat_annual_mwh_hat", "nat_annual_mwh", "nat_total_mwh_hat"],
    )

    forecast["year"] = pd.to_numeric(forecast[fc_year_col], errors="coerce").astype("Int64")
    forecast["nat_annual_mwh_fc"] = pd.to_numeric(forecast[fc_total_col], errors="coerce")

    out = pd.DataFrame({"year": target_years})
    out = out.merge(observed[["year", "nat_annual_mwh", "omega_hh", "omega_co"]], on="year", how="left")
    out = out.merge(forecast[["year", "nat_annual_mwh_fc"]], on="year", how="left")

    out["nat_annual_mwh_target"] = out["nat_annual_mwh"]
    miss = out["nat_annual_mwh_target"].isna()
    out.loc[miss, "nat_annual_mwh_target"] = out.loc[miss, "nat_annual_mwh_fc"]

    # Fill omega from nearest available if needed.
    if out["omega_hh"].isna().any():
        available = observed.dropna(subset=["omega_hh"]).sort_values("year")
        if available.empty:
            raise ValueError("[ANNUAL_TOTALS] No omega_hh values available in sector mix file.")
        for idx in out.index[out["omega_hh"].isna()]:
            y = int(out.loc[idx, "year"])
            nearest = available.iloc[(available["year"] - y).abs().argsort().iloc[0]]
            out.loc[idx, "omega_hh"] = float(nearest["omega_hh"])

    if out["omega_co"].isna().any():
        out["omega_co"] = 1.0 - out["omega_hh"]

    missing_years = out.loc[out["nat_annual_mwh_target"].isna(), "year"].tolist()
    if missing_years:
        raise ValueError(
            f"[ANNUAL_TOTALS] Missing national annual totals for years: {missing_years}"
        )

    out["nat_annual_mwh_target"] = pd.to_numeric(out["nat_annual_mwh_target"], errors="coerce")
    out["omega_hh"] = pd.to_numeric(out["omega_hh"], errors="coerce")
    out["omega_co"] = pd.to_numeric(out["omega_co"], errors="coerce")

    return out.sort_values("year").reset_index(drop=True)


# =============================================================================
# BUILD NATIONAL HOURLY PAPER YEARS
# =============================================================================
def _build_national_hourly_paper_years(
    templates: NationalHourlyTemplates,
    annual_totals: pd.DataFrame,
    target_years: list[int],
) -> tuple[pd.DataFrame, dict[int, int]]:
    template_map = _assign_template_years(list(templates.by_year.keys()), target_years)
    pieces = []

    for year in target_years:
        template_year = template_map[year]
        tpl = templates.by_year[template_year].copy()

        H_target = _hours_in_year_utc(year)
        H_template = len(tpl)
        if H_target != H_template:
            raise ValueError(
                f"[NATIONAL_HOURLY] Template year {template_year} has {H_template} hours but "
                f"target year {year} has {H_target}. Leap/non-leap mismatch slipped through."
            )

        target_row = annual_totals.loc[annual_totals["year"] == year]
        if target_row.empty:
            raise ValueError(f"[NATIONAL_HOURLY] Missing annual total row for year {year}.")

        nat_total_mwh = float(target_row["nat_annual_mwh_target"].iloc[0])
        omega_hh = float(target_row["omega_hh"].iloc[0])
        omega_co = float(target_row["omega_co"].iloc[0])

        w = tpl["nat_load_template_MW"].to_numpy(dtype=float)
        wsum = float(w.sum())
        if not np.isfinite(wsum) or wsum <= 0:
            raise ValueError(f"[NATIONAL_HOURLY] Nonpositive template sum for year {template_year}.")

        shape = w / wsum

        out = _build_hourly_index(year)
        out["template_year_load"] = int(template_year)
        out["nat_load_MW"] = nat_total_mwh * shape
        out["nat_hh_load_MW"] = omega_hh * out["nat_load_MW"]
        out["nat_co_load_MW"] = omega_co * out["nat_load_MW"]

        # exact annual consistency
        err = abs(float(out["nat_load_MW"].sum()) - nat_total_mwh)
        if err > 1e-6:
            raise ValueError(f"[NATIONAL_HOURLY] Annual exactness failed for year {year}. err={err}")

        pieces.append(out)

    full = pd.concat(pieces, ignore_index=True).sort_values(["year", "hour_of_year"]).reset_index(drop=True)
    return full, template_map


# =============================================================================
# LOAD PROVINCIAL HH/CO SHARES
# =============================================================================
def _load_provincial_sector_shares(path: Path, target_years: list[int], provinces: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    year_col = _pick_column(df.columns, ["year"])
    province_col = _pick_column(df.columns, ["province"])

    # Prefer observed shares when available for historical years, otherwise hats.
    hh_col_obs = _pick_column(df.columns, ["hh_share_obs"], required=False)
    co_col_obs = _pick_column(df.columns, ["co_share_obs"], required=False)
    hh_col_hat = _pick_column(df.columns, ["hh_share_hat"], required=False)
    co_col_hat = _pick_column(df.columns, ["co_share_hat"], required=False)

    if hh_col_hat is None or co_col_hat is None:
        raise ValueError(
            f"[PROV_SHARES] Need hh_share_hat and co_share_hat in {path}."
        )

    df["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df["province"] = df[province_col].astype(str).str.strip()

    df["hh_share_final"] = pd.to_numeric(df[hh_col_hat], errors="coerce")
    df["co_share_final"] = pd.to_numeric(df[co_col_hat], errors="coerce")

    if hh_col_obs is not None:
        obs = pd.to_numeric(df[hh_col_obs], errors="coerce")
        df.loc[obs.notna(), "hh_share_final"] = obs.loc[obs.notna()]
    if co_col_obs is not None:
        obs = pd.to_numeric(df[co_col_obs], errors="coerce")
        df.loc[obs.notna(), "co_share_final"] = obs.loc[obs.notna()]

    df = df[df["year"].isin(target_years)].copy()
    df = df[df["province"].isin(provinces)].copy()

    full_idx = pd.MultiIndex.from_product([target_years, provinces], names=["year", "province"])
    df = (
        df[["year", "province", "hh_share_final", "co_share_final"]]
        .drop_duplicates(subset=["year", "province"], keep="last")
        .set_index(["year", "province"])
        .reindex(full_idx)
        .reset_index()
    )

    if df[["hh_share_final", "co_share_final"]].isna().any().any():
        missing = df[df[["hh_share_final", "co_share_final"]].isna().any(axis=1)].head(20)
        raise ValueError(
            "[PROV_SHARES] Missing province-year sector shares.\n"
            f"Examples:\n{missing.to_string(index=False)}"
        )

    for year, g in df.groupby("year", sort=True):
        hh_sum = float(g["hh_share_final"].sum())
        co_sum = float(g["co_share_final"].sum())
        if abs(hh_sum - 1.0) > 1e-8:
            raise ValueError(f"[PROV_SHARES] year={year}: HH shares sum to {hh_sum}, not 1.")
        if abs(co_sum - 1.0) > 1e-8:
            raise ValueError(f"[PROV_SHARES] year={year}: CO shares sum to {co_sum}, not 1.")

    return df.sort_values(["year", "province"]).reset_index(drop=True)


# =============================================================================
# BUILD PROVINCE-HOUR LOAD PANEL
# =============================================================================
def _build_province_hour_load_panel(
    nat_hourly: pd.DataFrame,
    prov_shares: pd.DataFrame,
) -> pd.DataFrame:
    panel = nat_hourly.merge(prov_shares, on="year", how="inner", validate="many_to_many")

    panel["prov_hh_load_MW"] = panel["nat_hh_load_MW"] * panel["hh_share_final"]
    panel["prov_co_load_MW"] = panel["nat_co_load_MW"] * panel["co_share_final"]
    panel["prov_load_MW"] = panel["prov_hh_load_MW"] + panel["prov_co_load_MW"]

    # exact conservation check
    chk = (
        panel.groupby(["year", "timestamp_utc", "hour_of_year"], as_index=False)[
            ["prov_hh_load_MW", "prov_co_load_MW", "prov_load_MW", "nat_hh_load_MW", "nat_co_load_MW", "nat_load_MW"]
        ]
        .sum()
        .sort_values(["year", "hour_of_year"])
        .reset_index(drop=True)
    )

    n_prov = int(panel["province"].nunique())
    chk["nat_hh_load_MW"] /= n_prov
    chk["nat_co_load_MW"] /= n_prov
    chk["nat_load_MW"] /= n_prov

    max_gap = max(
        float(np.abs(chk["prov_hh_load_MW"] - chk["nat_hh_load_MW"]).max()),
        float(np.abs(chk["prov_co_load_MW"] - chk["nat_co_load_MW"]).max()),
        float(np.abs(chk["prov_load_MW"] - chk["nat_load_MW"]).max()),
    )
    if max_gap > 1e-6:
        raise ValueError(f"[PROV_LOAD_PANEL] Conservation failed. max_gap={max_gap}")

    return panel.sort_values(["province", "timestamp_utc"]).reset_index(drop=True)


# =============================================================================
# PRICE LOADER
# =============================================================================
def _parse_price_timestamp_from_mtu(series: pd.Series) -> pd.Series:
    """
    Parse ENTSO-E / market-style MTU strings such as:
        '01.01.2021 00:00 - 01.01.2021 01:00'
        '2021-01-01 00:00 - 2021-01-01 01:00'
        '01/01/2021 00:00 - 01/01/2021 01:00'

    We interpret the LEFT endpoint as the delivery-start time in Europe/Amsterdam,
    then convert to UTC.
    """
    s = series.astype(str).str.strip()

    left = s.str.extract(r"^\s*([0-9]{1,4}[-./][0-9]{1,2}[-./][0-9]{1,4}\s+[0-9]{1,2}:[0-9]{2})")[0]
    if left.isna().all():
        # fallback: split on hyphen and take left side
        left = s.str.split(r"\s*-\s*", n=1, regex=True).str[0].str.strip()

    # Try day-first first, then generic
    ts_local = pd.to_datetime(left, errors="coerce", dayfirst=True)
    bad = ts_local.isna()

    if bad.any():
        ts_local_2 = pd.to_datetime(left[bad], errors="coerce", dayfirst=False)
        ts_local.loc[bad] = ts_local_2

    if ts_local.isna().all():
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

    # Local Dutch market time -> UTC
    # ambiguous='infer' works for sorted DST transitions in many files;
    # nonexistent='shift_forward' handles spring-forward gaps conservatively.
    try:
        ts_local = ts_local.dt.tz_localize(
            "Europe/Amsterdam",
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    except Exception:
        # safer row-wise fallback
        tmp = []
        for x in ts_local:
            if pd.isna(x):
                tmp.append(pd.NaT)
                continue
            try:
                tmp.append(
                    pd.Timestamp(x).tz_localize(
                        "Europe/Amsterdam",
                        ambiguous=False,
                        nonexistent="shift_forward",
                    )
                )
            except Exception:
                tmp.append(pd.NaT)
        ts_local = pd.Series(tmp, index=series.index)

    return pd.to_datetime(ts_local, utc=True, errors="coerce").dt.floor("h")


def _build_price_timestamp_column(df: pd.DataFrame, label: str) -> pd.Series:
    """
    Robust timestamp builder for Dutch day-ahead price files.

    Accepted patterns:
      1) direct timestamp columns already in UTC or parseable
      2) ENTSO-E style MTU / interval columns in Europe/Amsterdam time
      3) date + hour in Europe/Amsterdam time
    """
    # ------------------------------------------------------------------
    # 1) direct timestamp columns
    # ------------------------------------------------------------------
    ts_col = _pick_column(
        df.columns,
        [
            "timestamp_utc",
            "utc_timestamp",
            "utc_time",
            "time_utc",
            "timestamp",
            "datetime",
            "datetime (utc)",
            "Datetime (UTC)",
            "datetime utc",
            "datehour",
            "delivery_start_utc",
        ],
        required=False,
    )
    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if ts.notna().any():
            return ts.dt.floor("h")

    # ------------------------------------------------------------------
    # 2) MTU / interval columns in market local time
    # ------------------------------------------------------------------
    mtu_col = _pick_column(
        df.columns,
        [
            "MTU (CET/CEST)",
            "MTU",
            "mtu (cet/cest)",
            "mtu",
            "Delivery period",
            "delivery period",
            "Time interval",
            "time interval",
            "Period",
            "period",
        ],
        required=False,
    )
    if mtu_col is not None:
        ts = _parse_price_timestamp_from_mtu(df[mtu_col])
        if ts.notna().any():
            return ts

    # ------------------------------------------------------------------
    # 3) local datetime column
    # ------------------------------------------------------------------
    local_dt_col = _pick_column(
        df.columns,
        [
            "datetime (local)",
            "Datetime (Local)",
            "datetime local",
            "local_datetime",
        ],
        required=False,
    )
    if local_dt_col is not None:
        local_ts = pd.to_datetime(df[local_dt_col], errors="coerce", dayfirst=True)
        if local_ts.notna().any():
            try:
                local_ts = local_ts.dt.tz_localize(
                    "Europe/Amsterdam",
                    ambiguous="infer",
                    nonexistent="shift_forward",
                )
            except Exception:
                tmp = []
                for x in local_ts:
                    if pd.isna(x):
                        tmp.append(pd.NaT)
                        continue
                    try:
                        tmp.append(
                            pd.Timestamp(x).tz_localize(
                                "Europe/Amsterdam",
                                ambiguous=False,
                                nonexistent="shift_forward",
                            )
                        )
                    except Exception:
                        tmp.append(pd.NaT)
                local_ts = pd.Series(tmp, index=df.index)

            ts = pd.to_datetime(local_ts, utc=True, errors="coerce")
            if ts.notna().any():
                return ts.dt.floor("h")

    # ------------------------------------------------------------------
    # 4) date + hour columns in local Dutch market time
    # ------------------------------------------------------------------
    date_col = _pick_column(
        df.columns,
        ["date", "datum", "day", "delivery day", "trading day"],
        required=False,
    )
    hour_col = _pick_column(
        df.columns,
        [
            "hour",
            "uur",
            "hour_from",
            "hour from",
            "delivery hour",
            "position",
            "period",
            "hour ending",
        ],
        required=False,
    )

    if date_col is not None and hour_col is not None:
        d = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        h = pd.to_numeric(df[hour_col], errors="coerce")

        if h.notna().any():
            h_clean = h.copy()
            h_min = float(h_clean.min())
            h_max = float(h_clean.max())

            if h_min >= 1 and h_max <= 24:
                h_clean = h_clean - 1.0
            elif h_min >= 0 and h_max <= 23:
                pass
            else:
                raise ValueError(
                    f"[{label}] Could not interpret hour column {hour_col!r}. "
                    f"Observed range: min={h_min}, max={h_max}"
                )

            ts_local_naive = pd.to_datetime(
                d.dt.strftime("%Y-%m-%d"),
                errors="coerce",
            ) + pd.to_timedelta(h_clean.fillna(0), unit="h")

            try:
                ts_local = ts_local_naive.dt.tz_localize(
                    "Europe/Amsterdam",
                    ambiguous="infer",
                    nonexistent="shift_forward",
                )
            except Exception:
                tmp = []
                for x in ts_local_naive:
                    if pd.isna(x):
                        tmp.append(pd.NaT)
                        continue
                    try:
                        tmp.append(
                            pd.Timestamp(x).tz_localize(
                                "Europe/Amsterdam",
                                ambiguous=False,
                                nonexistent="shift_forward",
                            )
                        )
                    except Exception:
                        tmp.append(pd.NaT)
                ts_local = pd.Series(tmp, index=df.index)

            ts = pd.to_datetime(ts_local, utc=True, errors="coerce")
            if ts.notna().any():
                return ts.dt.floor("h")

    raise ValueError(
        f"[{label}] Could not build timestamps from price file. "
        f"Columns were: {list(df.columns)}"
    )




def _load_price_hourly(path: Path, target_years: list[int]) -> tuple[pd.DataFrame, dict[int, int]]:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    df["timestamp_utc"] = _build_price_timestamp_column(df, label="PRICE")
    df = df.dropna(subset=["timestamp_utc"]).copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce").dt.floor("h")

    price_col = _pick_column(
        df.columns,
        [
            "price_da_eur_mwh",
            "day_ahead_price_eur_mwh",
            "eur_mwh",
            "price_eur_mwh",
            "price",
            "value",
            "Price (EUR/MWhe)",
            "price (eur/mwhe)",
            "Price (EUR/MWh)",
            "price (eur/mwh)",
            "NL",
            "Netherlands",
            "Day-ahead Price [EUR/MWh]",
            "Day-ahead price [EUR/MWh]",
            "Day-ahead Price (EUR/MWh)",
            "Day-ahead price (EUR/MWh)",
        ],
        required=False,
    )

    if price_col is None:
        blacklist = {
            "year", "month", "day", "hour", "uur",
            "timestamp_utc", "utc_timestamp", "utc_time", "time_utc", "timestamp", "datetime",
            "datetime (utc)", "datetime (local)",
            "date", "datum", "delivery day", "trading day",
            "mtu", "mtu (cet/cest)", "period", "delivery period", "time interval",
            "country", "iso3 code",
        }
        numeric_candidates = []
        for c in df.columns:
            if str(c).strip().lower() in blacklist:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                numeric_candidates.append(c)

        if len(numeric_candidates) == 1:
            price_col = numeric_candidates[0]
        else:
            raise ValueError(
                f"[PRICE] Could not identify price column in {path}. "
                f"Numeric candidates were: {numeric_candidates}"
            )

    df["price_da_eur_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["price_da_eur_mwh"]).copy()

    df = (
        df.groupby("timestamp_utc", as_index=False)["price_da_eur_mwh"]
        .mean()
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    df["year"] = df["timestamp_utc"].dt.year.astype(int)

    available_years = sorted(df["year"].unique().tolist())
    if not available_years:
        raise ValueError("[PRICE] No usable price years found after parsing.")

    template_map = _assign_template_years(available_years, target_years)

    pieces = []
    for year in target_years:
        idx = _build_hourly_index(year)

        if year in available_years:
            g = df[df["year"] == year].copy()
            merged = idx.merge(
                g[["timestamp_utc", "price_da_eur_mwh"]],
                on="timestamp_utc",
                how="left",
                validate="one_to_one",
            )
            merged["template_year_price"] = int(year)
        else:
            ty = template_map[year]
            g = df[df["year"] == ty].copy().sort_values("timestamp_utc").reset_index(drop=True)
            tpl_idx = _build_hourly_index(ty)

            if len(g) != len(tpl_idx):
                # align on hour_of_year if direct timestamp counts differ because of DST formatting quirks
                g = tpl_idx[["hour_of_year"]].merge(
                    pd.DataFrame({
                        "hour_of_year": np.arange(1, min(len(g), len(tpl_idx)) + 1, dtype=int),
                        "price_da_eur_mwh": g["price_da_eur_mwh"].iloc[:min(len(g), len(tpl_idx))].to_numpy(dtype=float),
                    }),
                    on="hour_of_year",
                    how="left",
                )
            else:
                g = pd.DataFrame({
                    "hour_of_year": tpl_idx["hour_of_year"],
                    "price_da_eur_mwh": g["price_da_eur_mwh"].to_numpy(dtype=float),
                })

            merged = idx.merge(
                g,
                on="hour_of_year",
                how="left",
                validate="one_to_one",
            )
            merged["template_year_price"] = int(ty)

        merged["price_da_eur_mwh"] = (
            pd.to_numeric(merged["price_da_eur_mwh"], errors="coerce")
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

        if merged["price_da_eur_mwh"].isna().any():
            raise ValueError(f"[PRICE] Unresolved price gaps remained for year {year}.")

        pieces.append(merged[["timestamp_utc", "year", "hour_of_year", "price_da_eur_mwh", "template_year_price"]])

    out = pd.concat(pieces, ignore_index=True).sort_values(["year", "hour_of_year"]).reset_index(drop=True)
    return out, template_map
# =============================================================================
# BUILD PUBLIC STRESS PANEL
# =============================================================================
def _build_public_stress_panel(
    prov_load_panel: pd.DataFrame,
    price_hourly: pd.DataFrame,
) -> pd.DataFrame:
    panel = prov_load_panel.merge(
        price_hourly[["timestamp_utc", "price_da_eur_mwh"]],
        on="timestamp_utc",
        how="left",
        validate="many_to_one",
    )

    if panel["price_da_eur_mwh"].isna().any():
        raise ValueError("[PUBLIC_STRESS] Missing price rows after merge.")

    panel["year"] = panel["timestamp_utc"].dt.year.astype(int)
    panel["month"] = panel["timestamp_utc"].dt.month.astype(int)

    # Province-year load share and structural exposure
    nat_annual = (
        panel.groupby("year", as_index=False)["nat_load_MW"]
        .first()
        .rename(columns={"nat_load_MW": "dummy"})
    )
    del nat_annual  # not needed directly, kept only to show intentional separation

    prov_annual = (
        panel.groupby(["province", "year"], as_index=False)["prov_load_MW"]
        .sum()
        .rename(columns={"prov_load_MW": "prov_annual_mwh_proxy"})
    )
    nat_annual_sum = (
        panel.groupby("year", as_index=False)["nat_load_MW"]
        .sum()
        .rename(columns={"nat_load_MW": "nat_annual_mwh_proxy_times_12"})
    )
    # nat_load_MW repeats for each province row; divide by province count
    n_prov = int(panel["province"].nunique())
    nat_annual_sum["nat_annual_mwh_proxy"] = nat_annual_sum["nat_annual_mwh_proxy_times_12"] / n_prov
    nat_annual_sum = nat_annual_sum[["year", "nat_annual_mwh_proxy"]]

    panel = panel.merge(prov_annual, on=["province", "year"], how="left", validate="many_to_one")
    panel = panel.merge(nat_annual_sum, on="year", how="left", validate="many_to_one")

    panel["structural_exposure"] = np.where(
        panel["nat_annual_mwh_proxy"] > 0,
        panel["prov_annual_mwh_proxy"] / panel["nat_annual_mwh_proxy"],
        0.0,
    )

    panel["province_load_share"] = np.where(
        panel["nat_load_MW"] > 0,
        panel["prov_load_MW"] / panel["nat_load_MW"],
        0.0,
    )

    # High-load and low-load proxies by province-year
    panel["prov_load_norm"] = (
        panel.groupby(["province", "year"], sort=False)["prov_load_MW"]
        .transform(_minmax01)
        .astype(float)
    )

    panel["prov_load_inverse_norm"] = (
        panel.groupby(["province", "year"], sort=False)["prov_load_MW"]
        .transform(lambda s: _minmax01(-pd.to_numeric(s, errors="coerce").fillna(0.0)))
        .astype(float)
    )

    # Price normalization: high-price and low-price variants
    panel["price_da_norm"] = (
        panel.groupby("year", sort=False)["price_da_eur_mwh"]
        .transform(_minmax01)
        .astype(float)
    )
    panel["price_da_low_norm"] = (
        panel.groupby("year", sort=False)["price_da_eur_mwh"]
        .transform(lambda s: _minmax01(-pd.to_numeric(s, errors="coerce").fillna(0.0)))
        .astype(float)
    )

    # Final directional public stress proxies
    # plus  = high-load / high-price stress
    # minus = low-load / low-price slack
    panel["rl_plus_norm"] = panel["prov_load_norm"]
    panel["rl_minus_norm"] = panel["prov_load_inverse_norm"]

    panel["s_plus_norm"] = (
        0.65 * panel["prov_load_norm"]
        + 0.25 * panel["price_da_norm"]
        + 0.10 * panel["structural_exposure"]
    ).clip(lower=0.0)

    panel["s_minus_norm"] = (
        0.65 * panel["prov_load_inverse_norm"]
        + 0.25 * panel["price_da_low_norm"]
        + 0.10 * (1.0 - panel["structural_exposure"])
    ).clip(lower=0.0)

    # Re-normalize S terms to [0,1] within province-year
    panel["s_plus_norm"] = (
        panel.groupby(["province", "year"], sort=False)["s_plus_norm"]
        .transform(_minmax01)
        .astype(float)
    )
    panel["s_minus_norm"] = (
        panel.groupby(["province", "year"], sort=False)["s_minus_norm"]
        .transform(_minmax01)
        .astype(float)
    )

    keep = [
        "province",
        "timestamp_utc",
        "year",
        "month",
        "hour_of_year",
        "nat_load_MW",
        "prov_load_MW",
        "province_load_share",
        "price_da_eur_mwh",
        "price_da_norm",
        "rl_plus_norm",
        "rl_minus_norm",
        "s_plus_norm",
        "s_minus_norm",
        "structural_exposure",
    ]
    out = panel[keep].copy()
    out = out.sort_values(["province", "timestamp_utc"]).reset_index(drop=True)

    _assert_unique_keys(out, ["province", "timestamp_utc"], "PUBLIC_STRESS_PANEL")
    return out


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    target_years = sorted(int(y) for y in getattr(cfg, "ALL_YEARS", []))
    if not target_years:
        raise ValueError("cfg.ALL_YEARS is empty. Set paper years in pipeline_config.py.")

    provinces = list(getattr(cfg, "NL_PROVINCES", []))
    if not provinces:
        raise ValueError("cfg.NL_PROVINCES is empty in pipeline_config.py.")

    national_template_path = _get_path(
        "NATIONAL_LOAD_PROFILE_FILE_PATH",
        "TIME_SERIES_PROFILES_2015_2020_PATH",
        env_name="NATIONAL_LOAD_PROFILE_FILE_PATH",
        default=REPO_ROOT / "data" / "external" / "time_series_profiles_2015_2020.csv",
    )

    sector_mix_path = _get_path(
        "NATIONAL_SECTOR_MIX_PATH",
        "NATIONAL_SECTOR_MIX_UNIFIED_PATH",
        env_name="NATIONAL_SECTOR_MIX_PATH",
        default=REPO_ROOT / "data" / "derived" / "national_sector_mix_2012_2035.csv",
    )

    annual_forecast_path = _get_path(
        "NATIONAL_ANNUAL_LOAD_FORECAST_PATH",
        env_name="NATIONAL_ANNUAL_LOAD_FORECAST_PATH",
        default=REPO_ROOT / "data" / "derived" / "national_annual_load_2025_2035_forecast.csv",
    )

    provincial_shares_path = _get_path(
        "PROVINCIAL_SECTOR_SHARES_PATH",
        env_name="PROVINCIAL_SECTOR_SHARES_PATH",
        default=REPO_ROOT / "data" / "derived" / "provincial_sector_shares_2012_2035.csv",
    )

    price_path = _get_path(
        "PRICE_DA_PATH",
        "DAY_AHEAD_PRICE_PATH",
        "DAY_AHEAD_PRICES_PATH",
        env_name="PRICE_DA_PATH",
        default=REPO_ROOT / "data" / "external" / "Netherlands_prices.csv",
    )

    output_path = _get_path(
        "PROVINCE_STRESS_PATH",
        "PROVINCE_STRESS_PANEL_PATH",
        "PUBLIC_STRESS_PATH",
        "STATE_BACKBONE_PATH",
        env_name="PROVINCE_STRESS_PATH",
        default=REPO_ROOT / "data" / "derived" / "province_hour_public_stress_panel.csv",
    )

    qc_dir = _get_path(
        "QC_DIR",
        env_name="QC_DIR",
        default=REPO_ROOT / "qc",
    )

    national_template_path = _require_existing_path("national hourly template file", national_template_path)
    sector_mix_path = _require_existing_path("national sector mix file", sector_mix_path)
    annual_forecast_path = _require_existing_path("national annual forecast file", annual_forecast_path)
    provincial_shares_path = _require_existing_path("provincial sector shares file", provincial_shares_path)
    price_path = _require_existing_path("day-ahead price file", price_path)

    print("=" * 72)
    print("BUILD PROVINCE-HOUR PUBLIC STRESS PANEL")
    print("=" * 72)
    print(f"Paper years    : {target_years}")
    print(f"Templates      : {national_template_path}")
    print(f"Sector mix     : {sector_mix_path}")
    print(f"Annual forecast: {annual_forecast_path}")
    print(f"Prov shares    : {provincial_shares_path}")
    print(f"Prices         : {price_path}")
    print(f"Output         : {output_path}")

    templates = _load_national_hourly_templates(national_template_path)
    annual_totals = _load_annual_national_totals(
        sector_mix_path=sector_mix_path,
        annual_forecast_path=annual_forecast_path,
        target_years=target_years,
    )
    nat_hourly, template_assignment = _build_national_hourly_paper_years(
        templates=templates,
        annual_totals=annual_totals,
        target_years=target_years,
    )
    prov_shares = _load_provincial_sector_shares(
        path=provincial_shares_path,
        target_years=target_years,
        provinces=provinces,
    )
    prov_load_panel = _build_province_hour_load_panel(
        nat_hourly=nat_hourly,
        prov_shares=prov_shares,
    )
    price_hourly, price_template_assignment = _load_price_hourly(
        path=price_path,
        target_years=target_years,
    )
    out = _build_public_stress_panel(
        prov_load_panel=prov_load_panel,
        price_hourly=price_hourly,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")

    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_path = qc_dir / "build_province_hour_public_stress_panel_summary.json"

    exposure_error = (
        out.groupby("year", as_index=False)["structural_exposure"]
        .first()
    )
    del exposure_error  # explicit no-op; structural_exposure is year-province not a simplex over hours

    summary = {
        "output_path": str(output_path),
        "rows": int(len(out)),
        "n_provinces": int(out["province"].nunique()),
        "years_present": sorted(int(y) for y in out["year"].unique().tolist()),
        "panel_start": str(out["timestamp_utc"].min()) if len(out) else None,
        "panel_end": str(out["timestamp_utc"].max()) if len(out) else None,
        "nat_template_source_path": str(templates.source_path),
        "nat_load_template_assignment": {str(k): int(v) for k, v in template_assignment.items()},
        "price_template_assignment": {str(k): int(v) for k, v in price_template_assignment.items()},
        "annual_total_mwh_by_year": {
            str(int(r["year"])): float(r["nat_annual_mwh_target"])
            for _, r in annual_totals.iterrows()
        },
        "rl_plus_norm_min": float(out["rl_plus_norm"].min()) if len(out) else None,
        "rl_plus_norm_max": float(out["rl_plus_norm"].max()) if len(out) else None,
        "s_plus_norm_min": float(out["s_plus_norm"].min()) if len(out) else None,
        "s_plus_norm_max": float(out["s_plus_norm"].max()) if len(out) else None,
        "structural_exposure_min": float(out["structural_exposure"].min()) if len(out) else None,
        "structural_exposure_max": float(out["structural_exposure"].max()) if len(out) else None,
        "notes": [
            "This is the historical public stress backbone for the paper branch.",
            "It does not use the future 2025-2035 residual-load scenario scaffold as the main stress input.",
            "Directional stress is a public-data proxy built from province-hour load pressure and day-ahead prices.",
            "step04_burden_proxy.py should read this file via PROVINCE_STRESS_PATH / PROVINCE_STRESS_PANEL_PATH.",
        ],
    }

    with qc_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutput rows: {len(out):,}")
    print(f"Provinces  : {out['province'].nunique()}")
    print(f"Years      : {sorted(out['year'].unique().tolist())}")
    print(f"Panel start: {summary['panel_start']}")
    print(f"Panel end  : {summary['panel_end']}")
    print(f"\n✓ Written: {output_path}")
    print(f"✓ Written: {qc_path}")


if __name__ == "__main__":
    main()