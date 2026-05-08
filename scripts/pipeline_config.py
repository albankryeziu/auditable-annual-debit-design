# -*- coding: utf-8 -*-
"""
pipeline_config.py
==================

Single source of truth for the final Windows-native repo.

This configuration is aligned with the current paper logic:

1. The main paper uses a historical, auditable province-hour public-stress
   backbone over the empirical years 2021--2025.

2. TDTR candidate-hour support is constructed upstream by
   G_build_tdtr_candidate_hour_support.py. The main scenario is "medium",
   corresponding to approximately 15% of annual hours. Additional scenarios
   "restrictive" and "wide" are retained for sensitivity analysis.

3. H_export_ptdf_for_debit_paper.py builds a fixed constructed PTDF-style
   relief layer. It may use the candidate-hour support as the relief-volume
   base and/or as an hour mask, depending on configuration.

4. A_build_debit_optimization_inputs.py imposes the selected TDTR support on
   Pbar and, defensively, filters requirements to the same support.

5. B1_run_m1_touched_only.py solves M1 using a support-aware formulation:
      - exact LP when M1 binaries are redundant under the selected TDTR support;
      - compact support-restricted MILP when support is wider than the touched
        budget;
      - legacy column generation only if explicitly requested.

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
    here = Path(__file__).resolve().parent
    if here.name.lower() == "scripts":
        return here.parent
    return here


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
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
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
# SAMPLE WINDOW / HOLDOUT DESIGN
# =============================================================================
TRAIN_YEARS: Final[List[int]] = _parse_year_list("TRAIN_YEARS", "2021,2022,2023")
VAL_YEARS: Final[List[int]] = _parse_year_list("VAL_YEARS", "2024")
TEST_YEARS: Final[List[int]] = _parse_year_list("TEST_YEARS", "2025")

ALL_YEARS: Final[List[int]] = sorted(set(TRAIN_YEARS + VAL_YEARS + TEST_YEARS))
EVAL_YEARS: Final[List[int]] = _parse_year_list(
    "EVAL_YEARS",
    ",".join(str(y) for y in ALL_YEARS),
)

MIN_COMPLETE_YEAR: Final[int] = int(os.environ.get("MIN_COMPLETE_YEAR", min(ALL_YEARS)))
MAX_COMPLETE_YEAR: Final[int] = int(os.environ.get("MAX_COMPLETE_YEAR", max(ALL_YEARS)))


# =============================================================================
# HISTORICAL PUBLIC STRESS / RESIDUAL-LOAD BACKBONE
# =============================================================================
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

PROVINCIAL_RESIDUAL_LOAD_SOURCE_PATH: Final[Path] = _env_path(
    "PROVINCIAL_RESIDUAL_LOAD_SOURCE_PATH",
    HISTORICAL_PROVINCE_STRESS_PATH,
)

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

TIME_SERIES_PROFILES_2015_2020_PATH: Final[Path] = NATIONAL_LOAD_PROFILE_FILE_PATH

NATIONAL_LOAD_HISTORY_PATH: Final[Path] = DERIVED_DIR / "national_hourly_load_history.csv"
NATIONAL_LOAD_OVERLAP_DIAG_PATH: Final[Path] = DERIVED_DIR / "national_hourly_load_overlap_diagnostics_2015_2018.csv"
NATIONAL_LOAD_HISTORY_QC_PATH: Final[Path] = DERIVED_DIR / "national_hourly_load_history_qc_by_year.csv"
NATIONAL_LOAD_HISTORY_META_PATH: Final[Path] = DERIVED_DIR / "national_hourly_load_history_build_meta.json"


# =============================================================================
# TDTR CANDIDATE-HOUR SUPPORT
# =============================================================================
USE_TDTR_CANDIDATE_SUPPORT: Final[bool] = _parse_bool(
    "USE_TDTR_CANDIDATE_SUPPORT",
    True,
)

TDTR_SUPPORT_SCENARIO: Final[str] = os.environ.get("TDTR_SUPPORT_SCENARIO", "medium").strip()
TDTR_MAIN_SUPPORT_SCENARIO: Final[str] = TDTR_SUPPORT_SCENARIO

TDTR_SUPPORT_REQUIRED: Final[bool] = _parse_bool("TDTR_SUPPORT_REQUIRED", True)

TDTR_SUPPORT_APPLY_TO_REQUIREMENTS: Final[bool] = _parse_bool(
    "TDTR_SUPPORT_APPLY_TO_REQUIREMENTS",
    True,
)

TDTR_SUPPORT_APPLY_TO_PBAR: Final[bool] = _parse_bool(
    "TDTR_SUPPORT_APPLY_TO_PBAR",
    True,
)

TDTR_CANDIDATE_SUPPORT_PATH: Final[Path] = _env_path(
    "TDTR_CANDIDATE_SUPPORT_PATH",
    DERIVED_DIR / "tdtr_candidate_hour_support.csv",
)

TDTR_SUPPORT_PATH: Final[Path] = _env_path(
    "TDTR_SUPPORT_PATH",
    TDTR_CANDIDATE_SUPPORT_PATH,
)

TDTR_SUPPORT_SCENARIO_PATH: Final[Path] = _env_path(
    "TDTR_SUPPORT_SCENARIO_PATH",
    DERIVED_DIR / f"tdtr_candidate_hour_support__{TDTR_SUPPORT_SCENARIO}.csv",
)

TDTR_SUPPORT_ALL_SCENARIOS_PATH: Final[Path] = _env_path(
    "TDTR_SUPPORT_ALL_SCENARIOS_PATH",
    DERIVED_DIR / "tdtr_candidate_hour_support_all_scenarios.csv",
)

TDTR_CANDIDATE_SUPPORT_ALL_SCENARIOS_PATH: Final[Path] = TDTR_SUPPORT_ALL_SCENARIOS_PATH

TDTR_CANDIDATE_SUPPORT_PROVINCE_SCORE_PATH: Final[Path] = _env_path(
    "TDTR_CANDIDATE_SUPPORT_PROVINCE_SCORE_PATH",
    DERIVED_DIR / "tdtr_candidate_hour_support_province_scores.csv",
)

TDTR_CANDIDATE_SUPPORT_PROVINCE_SCORE_ALL_SCENARIOS_PATH: Final[Path] = _env_path(
    "TDTR_CANDIDATE_SUPPORT_PROVINCE_SCORE_ALL_SCENARIOS_PATH",
    DERIVED_DIR / "tdtr_candidate_hour_support_province_scores_all_scenarios.csv",
)

TDTR_CANDIDATE_SUPPORT_QC_PATH: Final[Path] = _env_path(
    "TDTR_CANDIDATE_SUPPORT_QC_PATH",
    QC_DIR / "tdtr_candidate_hour_support_summary.json",
)

TDTR_CANDIDATE_SCORE_MODE: Final[str] = os.environ.get(
    "TDTR_CANDIDATE_SCORE_MODE",
    "max_directional_public_stress",
)

TDTR_CANDIDATE_HOUR_AGGREGATION: Final[str] = os.environ.get(
    "TDTR_CANDIDATE_HOUR_AGGREGATION",
    "blend",
)

TDTR_CANDIDATE_ROUNDING: Final[str] = os.environ.get(
    "TDTR_CANDIDATE_ROUNDING",
    "floor",
)

TDTR_CANDIDATE_STRICT_FULL_YEAR_COVERAGE: Final[bool] = _parse_bool(
    "TDTR_CANDIDATE_STRICT_FULL_YEAR_COVERAGE",
    True,
)

TDTR_CANDIDATE_SCENARIOS: Final[List[str]] = [
    x.strip()
    for x in os.environ.get("TDTR_CANDIDATE_SCENARIOS", "restrictive,medium,wide").split(",")
    if x.strip()
]

TDTR_CANDIDATE_ALPHA_RESTRICTIVE: Final[float] = float(os.environ.get("TDTR_CANDIDATE_ALPHA_RESTRICTIVE", "0.10"))
TDTR_CANDIDATE_ALPHA_MEDIUM: Final[float] = float(os.environ.get("TDTR_CANDIDATE_ALPHA_MEDIUM", "0.15"))
TDTR_CANDIDATE_ALPHA_WIDE: Final[float] = float(os.environ.get("TDTR_CANDIDATE_ALPHA_WIDE", "0.20"))

TDTR_CANDIDATE_ALPHA_BY_SCENARIO = {
    "restrictive": TDTR_CANDIDATE_ALPHA_RESTRICTIVE,
    "medium": TDTR_CANDIDATE_ALPHA_MEDIUM,
    "wide": TDTR_CANDIDATE_ALPHA_WIDE,
}

TDTR_SUPPORT_FIGURE_DIR: Final[Path] = FIGURES_DIR / "tdtr_support"


# =============================================================================
# PTDF / H-EXPORT SETTINGS
# =============================================================================
PTDF_PUBLICATION_DIR: Final[Path] = _env_path(
    "PTDF_PUBLICATION_DIR",
    ROOT / "PTDF_OUTPUT_PUBLICATION",
)

PROVINCE_BUS_WEIGHTS_PATH: Final[Path] = _env_path(
    "PROVINCE_BUS_WEIGHTS_PATH",
    PTDF_PUBLICATION_DIR / "province_to_bus_weights.csv",
)

PTDF_DEBIT_INPUT_DIR: Final[Path] = _env_path(
    "PTDF_DEBIT_INPUT_DIR",
    ROOT / "PTDF_OUTPUT_PUBLICATION" / "debit_paper_inputs",
)

PTDF_RHO: Final[float] = float(os.environ.get("PTDF_RHO", "0.70"))
PTDF_MAX_ACTIVE_BRANCHES: Final[int] = int(os.environ.get("PTDF_MAX_ACTIVE_BRANCHES", "40"))
PTDF_SCREENING_SCORE: Final[str] = os.environ.get("PTDF_SCREENING_SCORE", "max_loading_frac")
PTDF_DEFAULT_RATING_MVA: Final[float] = float(os.environ.get("PTDF_DEFAULT_RATING_MVA", "1000.0"))
PTDF_USE_VOLTAGE_PROXY_RATINGS: Final[bool] = _parse_bool("PTDF_USE_VOLTAGE_PROXY_RATINGS", True)

PTDF_REQUIREMENT_SCALING_MODE: Final[str] = os.environ.get(
    "PTDF_REQUIREMENT_SCALING_MODE",
    "annual_public_load_fraction",
)

PTDF_ANNUAL_RELIEF_FRACTION: Final[float] = float(os.environ.get("PTDF_ANNUAL_RELIEF_FRACTION", "0.02"))

PTDF_REQUIREMENT_CAP_QUANTILE_RAW = os.environ.get("PTDF_REQUIREMENT_CAP_QUANTILE", "0.995")
PTDF_REQUIREMENT_CAP_QUANTILE = None if PTDF_REQUIREMENT_CAP_QUANTILE_RAW.strip().lower() in {
    "", "0", "none", "false", "no",
} else float(PTDF_REQUIREMENT_CAP_QUANTILE_RAW)

PTDF_MIN_REQUIREMENT_MW: Final[float] = float(os.environ.get("PTDF_MIN_REQUIREMENT_MW", "1e-9"))
PTDF_MIN_COEFFICIENT: Final[float] = float(os.environ.get("PTDF_MIN_COEFFICIENT", "1e-12"))

PTDF_USE_TDTR_SUPPORT: Final[bool] = _parse_bool("PTDF_USE_TDTR_SUPPORT", True)

PTDF_REQUIREMENT_VOLUME_BASE: Final[str] = os.environ.get(
    "PTDF_REQUIREMENT_VOLUME_BASE",
    "tdtr_support_public_load",
)

PTDF_APPLY_SUPPORT_HOUR_MASK: Final[bool] = _parse_bool(
    "PTDF_APPLY_SUPPORT_HOUR_MASK",
    True,
)


# =============================================================================
# DERIVED PIPELINE ARTIFACTS
# =============================================================================
CROSSWALK_PATH: Final[Path] = DERIVED_DIR / "gopacs_location_candidate_crosswalk.csv"

ANN_PANEL_PATH: Final[Path] = DERIVED_DIR / "gopacs_announcement_province_hour.csv"
GOPACS_ANNOUNCEMENT_PANEL_PATH: Final[Path] = ANN_PANEL_PATH
ANNOUNCEMENT_PANEL_PATH: Final[Path] = ANN_PANEL_PATH
STEP02_PANEL_PATH: Final[Path] = ANN_PANEL_PATH

ACTIVITY_PATH: Final[Path] = DERIVED_DIR / "gopacs_called_activity_hour.csv"
ACTIVITY_DIAGNOSTIC_PATH: Final[Path] = ACTIVITY_PATH
GOPACS_CALLED_ACTIVITY_PATH: Final[Path] = ACTIVITY_PATH

BURDEN_PATH: Final[Path] = DERIVED_DIR / "burden_proxy_province_hour.csv"
BURDEN_PROXY_PATH: Final[Path] = BURDEN_PATH

STATE_PATH: Final[Path] = DERIVED_DIR / "auditable_state_province_hour.csv"
AUDITABLE_STATE_PATH: Final[Path] = STATE_PATH
AUDITABLE_STATE_PANEL_PATH: Final[Path] = STATE_PATH
STATE_PANEL_PATH: Final[Path] = STATE_PATH

SPLITS_PATH: Final[Path] = DERIVED_DIR / "paper_holdout_splits.csv"
PAM_PATH: Final[Path] = DERIVED_DIR / "participant_area_map.csv"

TESTBED_PATH: Final[Path] = DERIVED_DIR / "operational_testbed_inputs.parquet"
TESTBED_METADATA_PATH: Final[Path] = DERIVED_DIR / "operational_testbed_metadata.json"

DEBIT_OPT_INPUT_DIR: Final[Path] = _env_path(
    "DEBIT_OPT_INPUT_DIR",
    DERIVED_DIR / "debit_optimization_inputs",
)

M3_WEIGHTS_PATH: Final[Path] = DERIVED_DIR / "m3_debit_weights_by_year.parquet"
M3_LOG_PATH: Final[Path] = LOGS_DIR / "m3_training_log.csv"

BENCH_PATH: Final[Path] = RESULTS_DIR / "benchmark_full_results.csv"
KKT_PATH: Final[Path] = RESULTS_DIR / "kkt_equalization_diagnostic.csv"

VALIDATION_REPORT_PATH: Final[Path] = QC_DIR / "pipeline_validation_report.json"


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
# LEGACY / SCENARIO BRANCHES
# =============================================================================
NATIONAL_HOURLY_LOAD_LEGACY_PATH: Final[Path] = DERIVED_DIR / "national_hourly_load_2025_2035.csv"
PROVINCE_SHARE_FACTORS_OBS_PATH: Final[Path] = DERIVED_DIR / "province_share_factors_2012_2024_observed.csv"
PROVINCE_SHARE_FACTORS_FORECAST_PATH: Final[Path] = DERIVED_DIR / "province_share_factors_2025_2035_forecast.csv"
PROVINCIAL_HOURLY_LOAD_LEGACY_PATH: Final[Path] = DERIVED_DIR / "provincial_hourly_load_2025_2035.csv"

OFFSHORE_LANDING_PANEL_PATH: Final[Path] = DERIVED_DIR / "panel_offshore_landing_cap_mw_2025_2035_v1.csv"
NL_RES_TARGETS_PATH: Final[Path] = DERIVED_DIR / "nl_res_targets_twh_2025_2035_v1.csv"

NATIONAL_SECTOR_MIX_OBS_LEGACY_PATH: Final[Path] = DERIVED_DIR / "national_sector_mix_observed_2012_2024.csv"
NATIONAL_SECTOR_MIX_FORECAST_LEGACY_PATH: Final[Path] = DERIVED_DIR / "national_sector_mix_2025_2035_forecast.csv"

CBS_85004_PROVINCE_LONG_PATH: Final[Path] = DERIVED_DIR / "cbs_85004NED_province_long_2018_2024_v1.csv"

PROVINCE_RES_SHARES_LEARNED_OPTIONB_PATH: Final[Path] = DERIVED_DIR / "province_res_shares_2025_2035_learned_optionB.csv"
PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_PATH: Final[Path] = DERIVED_DIR / "provincial_residual_load_2025_2035_learned_optionB.csv"
PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_QC_PATH: Final[Path] = DERIVED_DIR / "provincial_residual_load_2025_2035_learned_optionB_qc_by_year.csv"
PROVINCIAL_RESIDUAL_LOAD_LEARNED_OPTIONB_DIAG_PATH: Final[Path] = DERIVED_DIR / "provincial_residual_load_2025_2035_learned_optionB_diagnostics.json"

NATIONAL_ANNUAL_LOAD_OBS_PATH: Final[Path] = DERIVED_DIR / "national_annual_load_observed_hist.csv"
NATIONAL_ANNUAL_LOAD_FORECAST_PATH: Final[Path] = DERIVED_DIR / "national_annual_load_2025_2035_forecast.csv"
NATIONAL_SECTOR_TOTALS_OBS_OVERLAP_PATH: Final[Path] = DERIVED_DIR / "national_sector_totals_observed_overlap.csv"
NATIONAL_SECTOR_TOTALS_FORECAST_PATH: Final[Path] = DERIVED_DIR / "national_sector_totals_2025_2035_forecast.csv"
NATIONAL_LOAD_SHAPES_DIAG_PATH: Final[Path] = DERIVED_DIR / "national_load_and_sector_shapes_diagnostics.json"
NATIONAL_SHAPE_FEATURE_COEFS_PATH: Final[Path] = DERIVED_DIR / "national_shape_feature_coefficients.csv"

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
# TDTR / DEBIT ACCOUNTING PARAMETERS
# =============================================================================
USE_TDTR85_BUDGETS: Final[bool] = _parse_bool("USE_TDTR85_BUDGETS", True)
TDTR_AVAILABILITY_SHARE: Final[float] = float(os.environ.get("TDTR_AVAILABILITY_SHARE", "0.85"))
TDTR_RESTRICTION_SHARE: Final[float] = 1.0 - TDTR_AVAILABILITY_SHARE

DEBIT_DEPTH_BUDGET_HOURS: Final[float] = float(os.environ.get("DEBIT_DEPTH_BUDGET_HOURS", "300.0"))
DEBIT_TOUCH_BUDGET_HOURS: Final[float] = float(os.environ.get("DEBIT_TOUCH_BUDGET_HOURS", "300.0"))

DEBIT_CONTRACT_CAPACITY_MODE: Final[str] = os.environ.get(
    "DEBIT_CONTRACT_CAPACITY_MODE",
    "fraction_of_p99_pressure",
)

DEBIT_CONTRACT_CAPACITY_FRACTION: Final[float] = float(os.environ.get("DEBIT_CONTRACT_CAPACITY_FRACTION", "0.30"))
DEBIT_CONTRACT_CAPACITY_MIN_MW: Final[float] = float(os.environ.get("DEBIT_CONTRACT_CAPACITY_MIN_MW", "25.0"))

DEBIT_CONTRACT_CAPACITY_MAX_MW_RAW = os.environ.get("DEBIT_CONTRACT_CAPACITY_MAX_MW", "")
DEBIT_CONTRACT_CAPACITY_MAX_MW = (
    None
    if DEBIT_CONTRACT_CAPACITY_MAX_MW_RAW.strip().lower() in {"", "none", "null"}
    else float(DEBIT_CONTRACT_CAPACITY_MAX_MW_RAW)
)

DEBIT_PARTICIPANT_CAPACITY_PATH_RAW = os.environ.get("DEBIT_PARTICIPANT_CAPACITY_PATH", "")
DEBIT_PARTICIPANT_CAPACITY_PATH = (
    None
    if DEBIT_PARTICIPANT_CAPACITY_PATH_RAW.strip().lower() in {"", "none", "null"}
    else Path(DEBIT_PARTICIPANT_CAPACITY_PATH_RAW)
)

DEBIT_TARIFF_TABLE_PATH_RAW = os.environ.get("DEBIT_TARIFF_TABLE_PATH", "")
DEBIT_TARIFF_TABLE_PATH = (
    None
    if DEBIT_TARIFF_TABLE_PATH_RAW.strip().lower() in {"", "none", "null"}
    else Path(DEBIT_TARIFF_TABLE_PATH_RAW)
)

DEBIT_TARIFF_REFERENCE_YEAR: Final[int] = int(os.environ.get("DEBIT_TARIFF_REFERENCE_YEAR", "2026"))

FIRM_CAPACITY_MULTIPLIER: Final[float] = float(os.environ.get("FIRM_CAPACITY_MULTIPLIER", "1.0"))
TDTR_CAPACITY_MULTIPLIER: Final[float] = float(os.environ.get("TDTR_CAPACITY_MULTIPLIER", "0.0"))

DEBIT_SLACK_PENALTY: Final[float] = float(os.environ.get("DEBIT_SLACK_PENALTY", str(M_PENALTY)))
DEBIT_MIN_REQUIREMENT_MW: Final[float] = float(os.environ.get("DEBIT_MIN_REQUIREMENT_MW", "1e-9"))
DEBIT_MIN_COEFFICIENT: Final[float] = float(os.environ.get("DEBIT_MIN_COEFFICIENT", "1e-12"))
DEBIT_RESTRICTION_COST_MIN: Final[float] = float(os.environ.get("DEBIT_RESTRICTION_COST_MIN", "1e-6"))
DEBIT_PRICE_COST_WINSOR_Q: Final[float] = float(os.environ.get("DEBIT_PRICE_COST_WINSOR_Q", "0.995"))


# =============================================================================
# M1 / B1 SOLVER SETTINGS
# =============================================================================
M1_SOLVE_MODE: Final[str] = os.environ.get("M1_SOLVE_MODE", "auto").strip().lower()
M1_TIGHTEN_USEFUL_PBAR: Final[bool] = _parse_bool("M1_TIGHTEN_USEFUL_PBAR", False)

M1_COMPACT_MIP_TIME_LIMIT: Final[float] = float(os.environ.get("M1_COMPACT_MIP_TIME_LIMIT", "1800"))
M1_COMPACT_MIP_GAP: Final[float] = float(os.environ.get("M1_COMPACT_MIP_GAP", "0.01"))
M1_COMPACT_MIP_OUTPUT_FLAG: Final[int] = int(os.environ.get("M1_COMPACT_MIP_OUTPUT_FLAG", "1"))
M1_COMPACT_MIP_FOCUS: Final[int] = int(os.environ.get("M1_COMPACT_MIP_FOCUS", "3"))
M1_COMPACT_MIP_HEURISTICS: Final[float] = float(os.environ.get("M1_COMPACT_MIP_HEURISTICS", "0.05"))
M1_COMPACT_MIP_CUTS: Final[int] = int(os.environ.get("M1_COMPACT_MIP_CUTS", "3"))
M1_COMPACT_MIP_PRESOLVE: Final[int] = int(os.environ.get("M1_COMPACT_MIP_PRESOLVE", "2"))
M1_COMPACT_NODEFILE_START: Final[float] = float(os.environ.get("M1_COMPACT_NODEFILE_START", "0.5"))
M1_REQUIRE_SUPPORT_REDUCTION: Final[bool] = _parse_bool("M1_REQUIRE_SUPPORT_REDUCTION", True)


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