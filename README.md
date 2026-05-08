# Annual Debit-Rule Design for Non-Firm Grid Access

**Replication package for:**  
*Novel Approaches for Optimization Problems Under Uncertainty*  
Alban Kryeziu · University of Groningen, Faculty of Economics and Business  
Supervised by Prof. Evrim Ursavas, Dr. Ward Romeijnders, Dr. Stuart X. Zhu

---

## What This Repository Does

This package implements and evaluates three annual debit-rule architectures for
TDTR-style non-firm electricity-network access contracts in the Dutch setting.

A **TDTR (Time-Dependent Transport Right)** is a non-firm grid-access product
under which the network operator may restrict a user's contracted transport
capacity during at most approximately 15% of annual hours, subject to advance
notice. When a restriction is imposed, it consumes part of the user's annual
non-firm entitlement. How that consumption is counted is the **annual debit rule**.

This repository compares three such rules:

- **M1 — Touched-time accounting.** Any positive restriction in an eligible
  hour costs one annual token, regardless of depth. Consistent with an
  annual-hours interpretation of the 85/15 TDTR clause.

- **M2 — Uniform-depth accounting.** Debit equals normalized restriction depth
  `x_{n,t} / P_n`. The state-independent, depth-sensitive benchmark.

- **M3 — Auditable weighted-depth accounting.** Debit equals
  `q_{n,t} · x_{n,t} / P_n`, where `q` is a bounded, province-year-normalized
  debit rate learned from public and reproducible proxy features.

All three rules are evaluated on the **same fixed constructed physical-relief
testbed** built from public Dutch data (GOPACS, ENTSO-E day-ahead prices, CBS
StatLine, ENTSOE load profiles). The physical layer does not change across rules;
only the annual accounting law changes.

The main empirical finding over 2021–2025: the learned F2/LO rule reduces the
constructed operating objective by 18.9% and emergency slack by 29.0% relative
to uniform-depth accounting. The holdout year (2025) alone shows a 26.2% cost
reduction and a 29.8% slack reduction.

---

## Understanding the TDTR Depth Budget

The TDTR contract has two distinct quantitative dimensions that this paper
carefully separates:

**Dimension 1 — Candidate-hour support (time eligibility).**  
The 85/15 rule identifies the *hours* in which restrictions may occur. For the
medium scenario used as the main benchmark, this is approximately 15% of annual
hours, or ~1,314 hours per year. This is built by `G_build_tdtr_candidate_hour_support.py`
and held fixed across M1, M2, and M3.

**Dimension 2 — Annual depth allowance (how much can be restricted inside eligible hours).**  
This is a separate question. A product that allows restrictions in 15% of hours
does not automatically permit full-depth restriction in all of those hours. The
depth allowance `H_depth` controls the cumulative full-power-equivalent hours of
restriction permitted per year per province. It is calibrated as:

```
H_depth  =  rho_depth  ×  |S_y|
```

where `|S_y|` is the number of selected TDTR candidate hours and `rho_depth`
is a scarcity-calibration parameter. The paper uses **`rho_depth = 0.02`**,
giving `H_depth ≈ 0.02 × 1314 ≈ 26 full-power-equivalent hours` per province
per year. This is intentionally tight to create meaningful annual scarcity for
the M2/M3 comparison.

**What is `DEBIT_DEPTH_BUDGET_HOURS = 300`?**  
This is a *legacy fallback* in `pipeline_config.py` that is only active when
`DEBIT_DEPTH_BUDGET_MODE = 'fixed'` — a mode that is **not used in any paper
run**. The paper always uses `DEBIT_DEPTH_BUDGET_MODE = 'rho_support'`, which
computes the budget dynamically from `rho_depth × |S_y|`. The 300-hour number
has no TDTR contractual meaning; it is simply a round number that makes the
budget non-binding in exploratory runs where you want to turn off depth scarcity
entirely. Do not use `fixed` mode for any result reported in the paper.

**Summary of budget parameters used in the paper:**

| Parameter | Value | Meaning |
|---|---|---|
| `TDTR_SUPPORT_SCENARIO` | `medium` | ~15% of annual hours selected |
| `TDTR_AVAILABILITY_SHARE` | `0.85` | 85/15 TDTR availability clause |
| `DEBIT_DEPTH_BUDGET_MODE` | `rho_support` | Budget = rho × support hours |
| `DEBIT_DEPTH_BUDGET_RHO` | `0.02` | Average 2% depth per support hour |
| `H_depth` (implied) | ~26 h | Full-power-equivalent hours per province per year |
| `H_touch` (M1, implied) | ~1314 h | Equals selected support hours |
| `DEBIT_DEPTH_BUDGET_HOURS` | `300` | **Fallback only. Never active in paper runs.** |

---

## Repository Structure

```
.
├── pipeline_config.py                  # Single source of truth for all paths and parameters
├── run_pipeline.py                     # Master pipeline runner
├── environment.yml                     # Conda environment specification
├── requirements.txt                    # pip fallback
│
├── scripts/
│   │
│   │  ── GOPACS data pipeline ──────────────────────────────────────────
│   ├── step00_fetch_gopacs_announcements.py    # Fetch raw GOPACS Machine Announcements API v7
│   ├── step01_crosswalk_real.py                # GOPACS location → province crosswalk
│   ├── step01b_plot_province_congestion_heatmaps.py
│   ├── step02_announcement_panel_real.py       # Province-hour announcement panel
│   ├── step03_called_activity_safe.py          # GOPACS called-activity panel
│   ├── step04_burden_proxy.py                  # Burden proxy (B_agg, E_exposure, B_proxy_baseline)
│   ├── step05_auditable_state.py               # Master auditable-state panel
│   ├── step06_testbed.py                       # Operational testbed + participant map
│   │
│   │  ── Load and residual-load backbone ────────────────────────────────
│   ├── build_national_load_history_master.py   # National hourly load 2006–2025
│   ├── learn_national_load_and_sector_shapes.py
│   ├── build_province_sector_targets_and_factors.py
│   ├── learn_province_factors.py               # OMEGA_MODE='freeze', CO_UNIT='mwh'
│   ├── build_provincial_hourly_load_from_learned_targets_updated.py
│   ├── build_provincial_residual_load_from_learned_load_optionB.py
│   ├── build_province_hour_public_stress_panel.py  # Province-hour public stress backbone
│   ├── build_nl_ptdf_and_stress.py
│   ├── make_targets_and_offshore_panel_from_table.py
│   │
│   │  ── TDTR support and physical relief layer ─────────────────────────
│   ├── G_build_tdtr_candidate_hour_support.py  # Candidate TDTR hours (G → H → A)
│   ├── H_export_ptdf_for_debit_paper.py        # Fixed physical relief layer
│   ├── F_scarcity_boundary.py                  # Scarcity boundary diagnostics
│   │
│   │  ── Annual debit-rule optimization pipeline ────────────────────────
│   ├── A_build_debit_optimization_inputs.py    # Annual LP inputs (support, budgets, costs)
│   ├── B_run_annual_lp.py                      # Weighted-depth LP/QP solver API (M2, M3)
│   ├── B1_run_m1_touched_only.py               # Support-aware M1 touched-time solver
│   ├── C_train_m3.py                           # Train F0–F3 × {plain, OI, LO} debit weights
│   ├── D_run_full_benchmark.py                 # Final benchmark (source of truth for paper)
│   ├── E_make_advanced_figures.py              # Paper figures and tables
│   └── validate_pipeline.py                    # Artifact validation
│
├── data/
│   ├── external/                       # Raw inputs (NOT tracked in git — see .gitignore)
│   │   ├── Netherlands_prices.csv      # ENTSO-E day-ahead prices 2021–2025
│   │   ├── time_series_profiles_2015_2020.csv   # ENTSO-E load profiles
│   │   ├── 82538NED_CBS_COMPANIES.csv  # CBS StatLine: companies
│   │   ├── 81528NED_CBS_HOUSEHOLDS.csv # CBS StatLine: households
│   │   └── 83704NED_CBS_DWELLINGS.csv  # CBS StatLine: dwellings
│   ├── GOPACS/raw/                     # GOPACS API responses (NOT tracked)
│   └── derived/                        # Built by pipeline (NOT tracked)
│
├── PTDF_OUTPUT_PUBLICATION/            # Constructed relief layer (NOT tracked in git)
│   └── debit_paper_inputs/
│       ├── active_branches_{year}.csv
│       ├── branch_hour_requirements_{year}.csv.gz
│       └── branch_hour_province_coefficients_{year}.csv.gz
│
├── outputs/
│   ├── results/                        # Benchmark CSVs and solution files (NOT tracked)
│   ├── figures/                        # Paper figures (NOT tracked)
│   └── tables/                         # LaTeX tables (NOT tracked)
│
├── qc/                                 # Quality-control summaries
└── manifests/                          # Fetch manifests and pipeline logs
```

---

## Prerequisites

### Software

| Tool | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | All scripts |
| Gurobi | ≥ 10.0 | LP/MIP solver (B, B1, C, D) |
| Anaconda / Miniconda | any | Recommended environment manager |
| Git | any | Version control |
| Git LFS | any | Large file handling (optional) |

A **Gurobi academic licence** is free for university researchers. Register at
[gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/)
and activate with `grbgetkey <your-key>`.

### Python packages

```
numpy >= 1.24
pandas >= 2.0
scipy >= 1.10
gurobipy >= 10.0
scikit-learn >= 1.3
matplotlib >= 3.7
pyarrow >= 12.0
requests >= 2.28
tqdm >= 4.65
```

---

## Data Requirements

The following external files must be placed in `data/external/` before running
the pipeline. None are tracked in git due to size or licensing.

| File | Source | Notes |
|---|---|---|
| `Netherlands_prices.csv` | [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Day-ahead prices NL, 2021–2025, hourly |
| `time_series_profiles_2015_2020.csv` | ENTSO-E / Open Power System Data | National load profiles 2015–2020 |
| `time_series_2006_2018.csv` | ENTSO-E historical | National load history |
| `82538NED_CBS_COMPANIES.csv` | [CBS StatLine](https://opendata.cbs.nl/statline/) | Dataset 82538NED |
| `81528NED_CBS_HOUSEHOLDS.csv` | CBS StatLine | Dataset 81528NED |
| `83704NED_CBS_DWELLINGS.csv` | CBS StatLine | Dataset 83704NED |

GOPACS Machine Announcements are fetched automatically by `step00` using the
GOPACS public API v7. No credentials are required for the Machine Announcements
endpoint as of 2025.

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/debit-rule-paper.git
cd debit-rule-paper
```

### Step 2 — Create the conda environment

```bash
conda env create -f environment.yml
conda activate debit-rule
```

Or with pip:

```bash
pip install -r requirements.txt
```

### Step 3 — Activate Gurobi

```bash
grbgetkey YOUR-GUROBI-KEY
```

Verify:

```bash
python -c "import gurobipy; gurobipy.Model()"
```

### Step 4 — Place external data

Copy your ENTSO-E and CBS files into `data/external/`. The paths are defined in
`pipeline_config.py` and can be overridden with environment variables.

### Step 5 — Verify configuration

```bash
python pipeline_config.py   # should run without errors and create directories
```

---

## Running the Full Pipeline

### Option A — Master runner (recommended)

```bash
python run_pipeline.py
```

This runs all stages in order. To resume from a failed stage:

```bash
python run_pipeline.py --from-stage G
```

To run a single stage:

```bash
python run_pipeline.py --only D
```

### Option B — Manual stage-by-stage (Windows)

The exact sequence used to produce the paper results. Run from the repo root.

```bat
:: ── Set TDTR scenario ─────────────────────────────────────────────────────
set TDTR_SUPPORT_SCENARIO=medium
set TDTR_SUPPORT_REQUIRED=1
set PTDF_APPLY_TDTR_SUPPORT_TO_REQUIREMENTS=1
set TDTR_SUPPORT_APPLY_TO_REQUIREMENTS=1
set TDTR_SUPPORT_APPLY_TO_PBAR=1

:: ── GOPACS data pipeline ───────────────────────────────────────────────────
python scripts\step00_fetch_gopacs_announcements.py
python scripts\step01_crosswalk_real.py
python scripts\step02_announcement_panel_real.py
python scripts\step03_called_activity_safe.py
python scripts\step04_burden_proxy.py
python scripts\step05_auditable_state.py

:: ── Load and stress backbone ───────────────────────────────────────────────
python scripts\build_national_load_history_master.py
python scripts\learn_national_load_and_sector_shapes.py
python scripts\build_province_sector_targets_and_factors.py
python scripts\learn_province_factors.py
python scripts\build_province_hour_public_stress_panel.py

:: ── Testbed construction ───────────────────────────────────────────────────
python scripts\step06_testbed.py

:: ── TDTR support and fixed relief layer ───────────────────────────────────
python scripts\G_build_tdtr_candidate_hour_support.py
python scripts\H_export_ptdf_for_debit_paper.py

:: ── Annual optimization inputs ────────────────────────────────────────────
python scripts\A_build_debit_optimization_inputs.py

:: ── M1 touched-time benchmark ─────────────────────────────────────────────
python scripts\B1_run_m1_touched_only.py

:: ── Train M3 debit weights ────────────────────────────────────────────────
python scripts\C_train_m3.py

:: ── Final benchmark (source of truth for paper) ───────────────────────────
python scripts\D_run_full_benchmark.py

:: ── Paper figures and tables ──────────────────────────────────────────────
python scripts\E_make_advanced_figures.py
```

### Scenario sensitivity loop (all three support scenarios)

```bat
python scripts\G_build_tdtr_candidate_hour_support.py

for %%S in (restrictive medium wide) do (
    set TDTR_SUPPORT_SCENARIO=%%S
    python scripts\H_export_ptdf_for_debit_paper.py
    python scripts\A_build_debit_optimization_inputs.py
    python scripts\B1_run_m1_touched_only.py
    python scripts\C_train_m3.py
    python scripts\D_run_full_benchmark.py
    python scripts\E_make_advanced_figures.py
)
```

**Important:** Always rerun C when changing the support scenario, because the LO
architecture's training weights depend on the scenario-specific operating layer.

---

## Key Configuration Parameters

All parameters live in `pipeline_config.py` and can be overridden with
environment variables.

### TDTR contract parameters

| Parameter | Default | Description |
|---|---|---|
| `TDTR_SUPPORT_SCENARIO` | `medium` | `restrictive` (10%), `medium` (15%), `wide` (20%) |
| `TDTR_AVAILABILITY_SHARE` | `0.85` | TDTR availability guarantee (85/15 rule) |
| `TDTR_RESTRICTION_SHARE` | `0.15` | Complement of availability share |

### Depth allowance calibration

| Parameter | Default | Description |
|---|---|---|
| `DEBIT_DEPTH_BUDGET_MODE` | `rho_support` | **Always use `rho_support` for paper runs.** Other modes: `rho_annual`, `fixed` |
| `DEBIT_DEPTH_BUDGET_RHO` | `0.02` | Average allowed depth per support hour. **Paper value: 0.02.** |
| `DEBIT_DEPTH_BUDGET_HOURS` | `300` | **Fallback only.** Active only when `DEPTH_BUDGET_MODE='fixed'`. Never used in paper runs. Implies 300 full-power-equivalent hours per province per year with no reference to the support size. |
| `DEBIT_TOUCH_BUDGET_FROM_SUPPORT` | `True` | M1 budget equals selected support hours |

### Solver and model

| Parameter | Default | Description |
|---|---|---|
| `MAIN_FEATURE_SET` | `F2` | Main M3 feature level |
| `MAIN_M3_VARIANT` | `M3_LO` | Main M3 training architecture |
| `M_PENALTY` | `5000.0` | Emergency slack penalty (EUR-eq./MWh) |
| `DEBIT_CONTRACT_CAPACITY_FRACTION` | `0.30` | Contracted capacity as fraction of p99 load pressure |

### Train/test split

| Parameter | Default | Description |
|---|---|---|
| `TRAIN_YEARS` | `2021,2022,2023` | Years used to fit M3 weights |
| `VAL_YEARS` | `2024` | Validation year |
| `TEST_YEARS` | `2025` | Holdout year (out-of-sample) |

---

## Output Files

After a full run, the following files contain the paper's results:

```
outputs/results/benchmark_full_results.csv        # Main annual comparison table
outputs/results/benchmark_pairwise_differences.csv # M1-M2, M2-M3, M1-M3 pairwise
outputs/results/kkt_equalization_diagnostic.csv   # Alpha = mu*x/P scarcity diagnostics
outputs/results/benchmark_manifest.json           # Full run metadata
outputs/results/benchmark_q_audit.csv             # Learned-rate audit statistics
outputs/results/solutions/                         # Province-hour solution files

outputs/figures/advanced/                          # Paper figures (F1–F10)
outputs/tables/                                    # LaTeX tables (T1–T5)

data/derived/m3_debit_weights_by_year.parquet     # Learned M3 debit weights
data/derived/m3_training_manifest.json            # Training metadata
```

**Important:** `D_run_full_benchmark.py` is the sole source of truth for any
numerical claim in the paper. Do not use C-stage training summaries as paper
results.

---

## Reproducing the Paper's Main Results

The reported results (Table 3) require:

1. `TDTR_SUPPORT_SCENARIO = medium`
2. `DEBIT_DEPTH_BUDGET_MODE = rho_support`
3. `DEBIT_DEPTH_BUDGET_RHO = 0.02`
4. `MAIN_FEATURE_SET = F2`, `MAIN_M3_VARIANT = M3_LO`
5. `TRAIN_YEARS = 2021,2022,2023`, `TEST_YEARS = 2025`
6. Gurobi solving all LPs to optimality (default tolerance)

Expected aggregate results (2021–2025, relative to M2):

| Metric | Value |
|---|---|
| Aggregate cost saving | 18.9% |
| Aggregate slack reduction | 29.0% |
| Holdout-year (2025) cost saving | 26.2% |
| Holdout-year slack reduction | 29.8% |

Note: years 2021–2023 are **in-sample** for the M3 regression. The holdout
evidence is 2024 (validation) and 2025 (test). Always report 2025 results
separately when making out-of-sample claims.

---

## Interpreting DEBIT_DEPTH_BUDGET_HOURS = 300 in Context

To be precise about where this number comes from and what it would imply if
activated:

With the paper's actual `rho_support` mode and `rho=0.02`:
```
H_depth = 0.02 × 1314 hours ≈ 26 full-power-equivalent hours per province
```
This creates genuine scarcity: budgets bind in 2022–2025 for all 12 provinces.

If someone were to switch to `DEPTH_BUDGET_MODE=fixed` with `HOURS=300`:
```
H_depth = 300 full-power-equivalent hours per province
```
This is roughly 11.5× more generous. At this level, the annual budget would
almost certainly be non-binding in all years, M2 and M3 would both solve to
the same unconstrained LP solution, and the M2–M3 comparison would be
uninformative. The 300 value is a safe exploratory default that intentionally
turns off depth scarcity so you can debug the physical relief layer in isolation.
It has no grounding in the 85/15 TDTR contract clause.

**Never use `fixed` mode in a paper run.**

---

## Citation

If you use this code, please cite:

```bibtex
@phdthesis{Kryeziu2026,
  author  = {Kryeziu, Alban},
  title   = {Novel Approaches for Optimization Problems Under Uncertainty},
  school  = {University of Groningen, Faculty of Economics and Business},
  year    = {2026},
  note    = {Supervised by Prof. E. Ursavas, Dr. W. Romeijnders, Dr. S. X. Zhu}
}
```

---

## License

© 2026 Alban Kryeziu, University of Groningen. All rights reserved.  
Code is provided for academic replication purposes only.  
Contact the author before any commercial use or redistribution.
