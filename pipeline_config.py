"""
pipeline_config.py
==================

Single source of truth for the final Windows-native repo.

This file is intentionally aligned to the locked paper logic:
- final repo root = "Final Paper Script"
- one representative participant per province
- 12 Dutch provinces
- main feature set = F2
- main M3 variant = M3_LO
- baseline generosity rho = 0.70 with robustness {0.60, 0.80}
- baseline operating layer:
      R_{n,t} = eta_R * max(0, stress_signal)
      c_{i,t} = eta_c * P_i * max(0.05, 1 + price_da_norm)

Important design rule
---------------------
This config pins exact paths. Downstream scripts should not recursively search
the repo for "candidate" files when a configured file is missing. If a path is
wrong, fix it here.

Main-paper rule
---------------
The historical paper pipeline must use a true historical province-hour public
stress / residual-load backbone. Scenario or learned future scaffolds are
retained only as explicit scenario artifacts and must not become the default
backbone.

All paths are root-relative and Windows-safe through pathlib.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, List


# =============================================================================
# HELPERS
# =============================================================================
def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _env_path(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default)))


def _parse_year_list(env_name: str, default: str) -> List[int]:
    raw = os.environ.get(env_name, default)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(env_name: str, default: str) -> List[float]:
    raw = os.environ.get(env_name, default)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_bool(env_name: str, default: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return default


# =============================================================================
# REPO ROOTS
# =============================================================================
ROOT: Final[Path] = _repo_root()
REPO_ROOT: Final[Path] = ROOT

SCRIPTS_DIR: Final[Path] = ROOT / "scripts"
ARCHIVE_DIR: Final[Path] = ROOT / "archive_old_scripts"

DATA_DIR: Final[Path] = ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
DERIVED_DIR: Final[Path] = DATA_DIR / "derived"
EXTERNAL_DIR: Final[Path] = DATA_DIR / "external"

GOPACS_DIR: Final[Path] = DATA_DIR / "GOPACS"
GOPACS_RAW_DIR: Final[Path] = GOPACS_DIR / "raw"

MANIFESTS_DIR: Final[Path] = ROOT / "manifests"
QC_DIR: Final[Path] = ROOT / "qc"

OUTPUTS_DIR: Final[Path] = ROOT / "outputs"
TABLES_DIR: Final[Path] = OUTPUTS_DIR / "tables"
FIGURES_DIR: Final[Path] = OUTPUTS_DIR / "figures"
LOGS_DIR: Final[Path] = OUTPUTS_DIR / "logs"
RESULTS_DIR: Final[Path] = OUTPUTS_DIR / "results"

LP_DIR: Final[Path] = DERIVED_DIR / "lp_params"

for _p in [
    DATA_DIR,
    RAW_DIR,
    DERIVED_DIR,
    EXTERNAL_DIR,
    GOPACS_DIR,
    GOPACS_RAW_DIR,
    MANIFESTS_DIR,
    QC_DIR,
    OUTPUTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    LP_DIR,
]:
    _p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GEOGRAPHY
# =============================================================================
NL_PROVINCES: Final[List[str]] = [
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

DSO_TO_PROVINCES = {
    "tennet": NL_PROVINCES,
    "tennet tso b.v.": NL_PROVINCES,
    "alliander": ["Noord-Holland", "Gelderland", "Friesland", "Flevoland"],
    "alliander n.v.": ["Noord-Holland", "Gelderland", "Friesland", "Flevoland"],
    "liander": ["Noord-Holland", "Gelderland", "Friesland", "Flevoland"],
    "enexis": ["Groningen", "Drenthe", "Overijssel", "Noord-Brabant", "Limburg"],
    "enexis netbeheer b.v.": ["Groningen", "Drenthe", "Overijssel", "Noord-Brabant", "Limburg"],
    "stedin": ["Zuid-Holland", "Zeeland", "Utrecht"],
    "stedin netbeheer b.v.": ["Zuid-Holland", "Zeeland", "Utrecht"],
    "coteq": ["Overijssel", "Drenthe"],
    "coteq netbeheer": ["Overijssel", "Drenthe"],
    "rendo": ["Drenthe"],
    "rendo netwerken": ["Drenthe"],
    "westland infra": ["Zuid-Holland"],
    "westland": ["Zuid-Holland"],
}


# =============================================================================
# RAW / EXTERNAL INPUTS
# =============================================================================
PRICES_PATH: Final[Path] = _env_path("PRICE_DA_PATH", EXTERNAL_DIR / "Netherlands_prices.csv")
PRICE_DA_PATH: Final[Path] = PRICES_PATH
DAY_AHEAD_PRICE_PATH: Final[Path] = PRICES_PATH
DAY_AHEAD_PRICES_PATH: Final[Path] = PRICES_PATH

JSONL_PATH: Final[Path] = GOPACS_RAW_DIR / "gopacs_announcements_raw.jsonl"
WINDOW_LOG_PATH: Final[Path] = MANIFESTS_DIR / "gopacs_fetch_windows_latest.csv"
FETCH_MANIFEST_PATH: Final[Path] = MANIFESTS_DIR / "gopacs_fetch_manifest_latest.json"
RAW_SUMMARY_PATH: Final[Path] = QC_DIR / "gopacs_fetch_summary_latest.json"

EUR_PATH: Final[Path] = _env_path("GOPACS_EUR_PATH", EXTERNAL_DIR / "gopacs_called_orders_eur_public.csv")
MWH_PATH: Final[Path] = _env_path("GOPACS_MWH_PATH", EXTERNAL_DIR / "gopacs_called_volumes_mwh_public.csv")


# =============================================================================
# HISTORICAL PUBLIC STRESS / RESIDUAL-LOAD BACKBONE
# =============================================================================
# This is the main-paper backbone. It must be historical and aligned to the
# observed overlap window used in the empirical evidence.
PROVINCE_STRESS_PATH: Final[Path] = _env_path(
    "PROVINCE_STRESS_PATH",
    DERIVED_DIR / "province_hour_public_stress_panel.csv",
)

HISTORICAL_PROVINCE_STRESS_PATH: Final[Path] = _env_path(
    "HISTORICAL_PROVINCE_STRESS_PATH",
    PROVINCE_STRESS_PATH,
)

HISTORICAL_BACKBONE_PATH: Final[Path] = HISTORICAL_PROVINCE_STRESS_PATH
PAPER_BACKBONE_PATH: Final[Path] = HISTORICAL_PROVINCE_STRESS_PATH

PROVINCE_STRESS_PANEL_PATH: Final[Path] = HISTORICAL_PROVINCE_STRESS_PATH
PUBLIC_STRESS_PATH: Final[Path] = HISTORICAL_PROVINCE_STRESS_PATH
RESIDUAL_LOAD_PANEL_PATH: Final[Path] = HISTORICAL_PROVINCE_STRESS_PATH
STATE_BACKBONE_PATH: Final[Path] = HISTORICAL_PROVINCE_STRESS_PATH

# Exact residual-load / stress source used by main-paper downstream scripts.
# This must point to the historical backbone by default.
PROVINCIAL_RESIDUAL_LOAD_SOURCE_PATH: Final[Path] = _env_path(
    "PROVINCIAL_RESIDUAL_LOAD_SOURCE_PATH",
    HISTORICAL_PROVINCE_STRESS_PATH,
)

# Explicit guardrail: scenario scaffolds are not allowed in the main-paper
# pipeline unless deliberately overridden for temporary diagnostics.
ALLOW_SCENARIO_STRESS_SCAFFOLD: Final[bool] = _parse_bool(
    "ALLOW_SCENARIO_STRESS_SCAFFOLD",
    False,
)


# =============================================================================
# NATIONAL LOAD HISTORY INPUTS / OUTPUTS
# =============================================================================
NATIONAL_LOAD_OLD_FILE_PATH: Final[Path] = _env_path(
    "NATIONAL_LOAD_OLD_FILE_PATH",
    EXTERNAL_DIR / "time_series_2006_2018.csv",
)

NATIONAL_LOAD_PROFILE_FILE_PATH: Final[Path] = _env_path(
    "NATIONAL_LOAD_PROFILE_FILE_PATH",
    EXTERNAL_DIR / "time_series_profiles_2015_2020.csv",
)

NATIONAL_LOAD_HISTORY_PATH: Final[Path] = DERIVED_DIR / "national_hourly_load_history.csv"
NATIONAL_LOAD_OVERLAP_DIAG_PATH: Final[Path] = DERIVED_DIR / "national_hourly_load_overlap_diagnostics_2015_2018.csv"
NATIONAL_LOAD_HISTORY_QC_PATH: Final[Path] = DERIVED_DIR / "national_hourly_load_history_qc_by_year.csv"
NATIONAL_LOAD_HISTORY_META_PATH: Final[Path] = DERIVED_DIR / "national_hourly_load_history_build_meta.json"


# =============================================================================
# DERIVED PIPELINE ARTIFACTS
# =============================================================================
# Step 01
CROSSWALK_PATH: Final[Path] = DERIVED_DIR / "gopacs_location_candidate_crosswalk.csv"

# Step 02
ANN_PANEL_PATH: Final[Path] = DERIVED_DIR / "gopacs_announcement_province_hour.csv"
GOPACS_ANNOUNCEMENT_PANEL_PATH: Final[Path] = ANN_PANEL_PATH
ANNOUNCEMENT_PANEL_PATH: Final[Path] = ANN_PANEL_PATH
STEP02_PANEL_PATH: Final[Path] = ANN_PANEL_PATH

# Step 03 diagnostic only
ACTIVITY_PATH: Final[Path] = DERIVED_DIR / "gopacs_called_activity_hour.csv"
ACTIVITY_DIAGNOSTIC_PATH: Final[Path] = ACTIVITY_PATH
GOPACS_CALLED_ACTIVITY_PATH: Final[Path] = ACTIVITY_PATH

# Step 04
BURDEN_PATH: Final[Path] = DERIVED_DIR / "burden_proxy_province_hour.csv"
BURDEN_PROXY_PATH: Final[Path] = BURDEN_PATH

# Step 05
STATE_PATH: Final[Path] = DERIVED_DIR / "auditable_state_province_hour.csv"
AUDITABLE_STATE_PATH: Final[Path] = STATE_PATH

# Step 05/06 infrastructure
SPLITS_PATH: Final[Path] = DERIVED_DIR / "paper_holdout_splits.csv"
PAM_PATH: Final[Path] = DERIVED_DIR / "participant_area_map.csv"

# Step 06
TESTBED_PATH: Final[Path] = DERIVED_DIR / "operational_testbed_inputs.parquet"
TESTBED_METADATA_PATH: Final[Path] = DERIVED_DIR / "operational_testbed_metadata.json"

# Optimization / results
M3_WEIGHTS_PATH: Final[Path] = DERIVED_DIR / "m3_debit_weights_by_year.parquet"
M3_LOG_PATH: Final[Path] = LOGS_DIR / "m3_training_log.csv"

BENCH_PATH: Final[Path] = RESULTS_DIR / "benchmark_full_results.csv"
KKT_PATH: Final[Path] = RESULTS_DIR / "kkt_equalization_diagnostic.csv"

VALIDATION_REPORT_PATH: Final[Path] = QC_DIR / "pipeline_validation_report.json"


# =============================================================================
# SAMPLE WINDOW / HOLDOUT DESIGN
# =============================================================================
TRAIN_YEARS: Final[List[int]] = _parse_year_list("TRAIN_YEARS", "2021,2022,2023")
VAL_YEARS: Final[List[int]] = _parse_year_list("VAL_YEARS", "2024")
TEST_YEARS: Final[List[int]] = _parse_year_list("TEST_YEARS", "2025")
ALL_YEARS: Final[List[int]] = sorted(set(TRAIN_YEARS + VAL_YEARS + TEST_YEARS))
EVAL_YEARS: Final[List[int]] = _parse_year_list(
    "EVAL_YEARS",
    ",".join(str(y) for y in TEST_YEARS),
)

MIN_COMPLETE_YEAR: Final[int] = int(os.environ.get("MIN_COMPLETE_YEAR", min(ALL_YEARS)))
MAX_COMPLETE_YEAR: Final[int] = int(os.environ.get("MAX_COMPLETE_YEAR", max(ALL_YEARS)))


# =============================================================================
# CBS STATLINE INPUTS
# =============================================================================
CBS_COMPANIES_PATH: Final[Path] = _env_path(
    "CBS_COMPANIES_PATH",
    EXTERNAL_DIR / "82538NED_CBS_COMPANIES.csv",
)
CBS_HOUSEHOLDS_PATH: Final[Path] = _env_path(
    "CBS_HOUSEHOLDS_PATH",
    EXTERNAL_DIR / "81528NED_CBS_HOUSEHOLDS.csv",
)
CBS_DWELLINGS_PATH: Final[Path] = _env_path(
    "CBS_DWELLINGS_PATH",
    EXTERNAL_DIR / "83704NED_CBS_DWELLINGS.csv",
)


# =============================================================================
# PROVINCE-SECTOR TARGETS / SHARES
# =============================================================================
PROVINCIAL_SECTOR_OBS_PATH: Final[Path] = DERIVED_DIR / "provincial_sector_observed_annual_2012_2024.csv"
PROVINCIAL_SECTOR_SHARES_PATH: Final[Path] = DERIVED_DIR / "provincial_sector_shares_2012_2035.csv"
NATIONAL_SECTOR_MIX_PATH: Final[Path] = DERIVED_DIR / "national_sector_mix_2012_2035.csv"
NATIONAL_SECTOR_MIX_UNIFIED_PATH: Final[Path] = NATIONAL_SECTOR_MIX_PATH
PROVINCIAL_SECTOR_SHARE_FACTORS_PATH: Final[Path] = DERIVED_DIR / "provincial_sector_share_factors_2012_2035.csv"
PROVINCIAL_SECTOR_DIAG_PATH: Final[Path] = DERIVED_DIR / "provincial_sector_targets_and_factors_diagnostics.json"


# =============================================================================
# LEGACY PROVINCE-FACTOR BRANCH
# =============================================================================
NATIONAL_HOURLY_LOAD_LEGACY_PATH: Final[Path] = DERIVED_DIR / "national_hourly_load_2025_2035.csv"
PROVINCE_SHARE_FACTORS_OBS_PATH: Final[Path] = DERIVED_DIR / "province_share_factors_2012_2024_observed.csv"
PROVINCE_SHARE_FACTORS_FORECAST_PATH: Final[Path] = DERIVED_DIR / "province_share_factors_2025_2035_forecast.csv"
PROVINCIAL_HOURLY_LOAD_LEGACY_PATH: Final[Path] = DERIVED_DIR / "provincial_hourly_load_2025_2035.csv"


# =============================================================================
# OPTION-B RES TARGET INPUTS
# =============================================================================
OFFSHORE_LANDING_PANEL_PATH: Final[Path] = DERIVED_DIR / "panel_offshore_landing_cap_mw_2025_2035_v1.csv"
NL_RES_TARGETS_PATH: Final[Path] = DERIVED_DIR / "nl_res_targets_twh_2025_2035_v1.csv"


# =============================================================================
# NATIONAL MIX LEGACY FALLBACKS
# =============================================================================
NATIONAL_SECTOR_MIX_OBS_LEGACY_PATH: Final[Path] = DERIVED_DIR / "national_sector_mix_observed_2012_2024.csv"
NATIONAL_SECTOR_MIX_FORECAST_LEGACY_PATH: Final[Path] = DERIVED_DIR / "national_sector_mix_2025_2035_forecast.csv"


# =============================================================================
# EXTERNAL TIME-SERIES INPUTS
# =============================================================================
TIME_SERIES_PROFILES_2015_2020_PATH: Final[Path] = EXTERNAL_DIR / "time_series_profiles_2015_2020.csv"


# =============================================================================
# CLEANED CBS 85004 FILE
# =============================================================================
CBS_85004_PROVINCE_LONG_PATH: Final[Path] = DERIVED_DIR / "cbs_85004NED_province_long_2018_2024_v1.csv"


# =============================================================================
# LEARNED OPTION-B RESIDUAL-LOAD BRANCH
# =============================================================================
# These are explicit scenario artifacts, not the main-paper backbone.
PROVINCE_RES_SHARES_LEARNED_OPTIONB_PATH: Final[Path] = DERIVED_DIR / "province_res_shares_2025_2035_learned_optionB.csv"
PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_PATH: Final[Path] = DERIVED_DIR / "provincial_residual_load_2025_2035_learned_optionB.csv"
PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_QC_PATH: Final[Path] = DERIVED_DIR / "provincial_residual_load_2025_2035_learned_optionB_qc_by_year.csv"
PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_DIAG_PATH: Final[Path] = DERIVED_DIR / "provincial_residual_load_2025_2035_learned_optionB_diagnostics.json"


# =============================================================================
# NATIONAL ANNUAL / SHAPE OUTPUTS
# =============================================================================
NATIONAL_ANNUAL_LOAD_OBS_PATH: Final[Path] = DERIVED_DIR / "national_annual_load_observed_hist.csv"
NATIONAL_ANNUAL_LOAD_FORECAST_PATH: Final[Path] = DERIVED_DIR / "national_annual_load_2025_2035_forecast.csv"
NATIONAL_SECTOR_TOTALS_OBS_OVERLAP_PATH: Final[Path] = DERIVED_DIR / "national_sector_totals_observed_overlap.csv"
NATIONAL_SECTOR_TOTALS_FORECAST_PATH: Final[Path] = DERIVED_DIR / "national_sector_totals_2025_2035_forecast.csv"
NATIONAL_LOAD_SHAPES_DIAG_PATH: Final[Path] = DERIVED_DIR / "national_load_and_sector_shapes_diagnostics.json"
NATIONAL_SHAPE_FEATURE_COEFS_PATH: Final[Path] = DERIVED_DIR / "national_shape_feature_coefficients.csv"


# =============================================================================
# LEARNED NATIONAL / PROVINCIAL HOURLY LOAD BRANCH
# =============================================================================
NATIONAL_HOURLY_LOAD_LEARNED_PATH: Final[Path] = DERIVED_DIR / "national_hourly_load_2025_2035_learned.csv"
PROVINCIAL_HOURLY_LOAD_LEARNED_PATH: Final[Path] = DERIVED_DIR / "provincial_hourly_load_2025_2035_learned.csv"
PROVINCIAL_HOURLY_LOAD_LEARNED_QC_PATH: Final[Path] = DERIVED_DIR / "provincial_hourly_load_2025_2035_qc_by_year.csv"
PROVINCIAL_HOURLY_LOAD_LEARNED_DIAG_PATH: Final[Path] = DERIVED_DIR / "provincial_hourly_load_2025_2035_diagnostics.json"


# =============================================================================
# LOCKED EMPIRICAL / OPERATING CHOICES
# =============================================================================
REPRESENTATIVE_CAPACITY_MW: Final[float] = float(os.environ.get("REPRESENTATIVE_CAPACITY_MW", "50.0"))

ETA_R: Final[float] = float(os.environ.get("ETA_R", "25.0"))
ETA_C: Final[float] = float(os.environ.get("ETA_C", "1.0"))
M_PENALTY: Final[float] = float(os.environ.get("M_PENALTY", "5000.0"))

BASELINE_RHO: Final[float] = float(os.environ.get("BASELINE_RHO", "0.70"))
ROBUSTNESS_RHO_VALUES: Final[List[float]] = _parse_float_list("ROBUSTNESS_RHO_VALUES", "0.60,0.80")

MAIN_FEATURE_SET: Final[str] = os.environ.get("MAIN_FEATURE_SET", "F2")
MAIN_M3_VARIANT: Final[str] = os.environ.get("MAIN_M3_VARIANT", "M3_LO")
M3_VARIANTS: Final[List[str]] = ["M3_plain", "M3_OI", "M3_LO"]


# =============================================================================
# FEATURE GROUPS
# =============================================================================
F0_FEATURES: Final[List[str]] = [
    "hour_of_year",
    "hour_of_week",
    "month",
    "is_weekend",
    "is_holiday",
]

F1_FEATURES: Final[List[str]] = F0_FEATURES + [
    "price_da_norm",
]

F2_FEATURES: Final[List[str]] = F1_FEATURES + [
    "rl_plus_norm",
    "rl_minus_norm",
    "s_plus_norm",
    "s_minus_norm",
]

F3_FEATURES: Final[List[str]] = F2_FEATURES + [
    "structural_exposure",
]

FEATURE_GROUPS = {
    "F0": F0_FEATURES,
    "F1": F1_FEATURES,
    "F2": F2_FEATURES,
    "F3": F3_FEATURES,
}


# =============================================================================
# BURDEN PROXY CONTRACT
# =============================================================================
PRIMARY_BURDEN: Final[str] = "B_proxy_baseline"

BURDEN_COLS: Final[List[str]] = [
    "B_proxy_baseline",
    "B_proxy_mwh",
    "B_proxy_activation",
]

BURDEN_AGG_COL: Final[str] = "B_agg"
BURDEN_EXPOSURE_COL: Final[str] = "E_exposure"


# =============================================================================
# GOPACS FIELD NAMES
# =============================================================================
GOPACS_FIELDS = dict(
    id="id",
    type="type",
    state="announcementState",
    org="organisationName",
    buy_areas="requestAreaDescriptionBuyOrders",
    sell_areas="requestAreaDescriptionSellOrders",
    prob_area="problemAreaDescription",
    profile_mw="remainingProblemProfileInMW",
    compliance="complianceType",
    created_ts="createdTimestamp",
    updated_ts="lastUpdatedTimestamp",
    prob_period="problemPeriod",
    start_time="startTime",
    end_time="endTime",
    n_quarters="numberOfQuartersInTimeSpan",
    day="day",
    message="message",
)

DELETED_STATE: Final[str] = "ANNOUNCEMENT_DELETED"
TYPE_TA: Final[str] = "TRANSPORTPROBLEM"
TYPE_CA: Final[str] = "CONGESTIONMANAGEMENT"
TYPE_FT: Final[str] = "FREE_TEXT"