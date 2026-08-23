Auditable Annual Debit Design

Reproducibility and audit code for annual debit accounting of partial restrictions in non-firm transmission access, with a Dutch power-system test case and frozen out-of-sample evaluation.

This repository contains the optimization, validation, provenance, and post-freeze evidence scripts used to compare three annual debit-accounting designs:

M1 — touched-time accounting: debit depends on whether a participant is materially restricted in an hour.

M2 — uniform-depth accounting: debit is proportional to normalized restriction depth, x / capacity.

M3 — weighted-depth accounting: debit is proportional to q * x / capacity, where the participant-time weights q are learned and frozen before held-out evaluation.

The code is intentionally fail-closed. Frozen inputs, source files, cards, evaluation manifests, solver certificates, replay checks, and output hashes are validated before downstream evidence is accepted.

Scope. This repository is a research reproducibility package. It does not represent an operational TSO tool, an AC-OPF implementation, a complete N-1 security certificate, or an official statement about Dutch network-access decisions.

1. What is in this repository?

The current code base has three logical layers.

Layer A — upstream data construction

The earlier project version constructs the public-data backbone: GOPACS announcements, national/provincial load and residual-load histories, public stress indicators, candidate TDTR support, and the PTDF relief layer. These scripts produce the frozen network, request, and state artifacts consumed by the optimization layer.

Layer B — frozen M1/M2/M3 optimization study

The parent implementation validates the frozen Dutch test case, constructs M2 scarcity/KKT diagnostics, trains and freezes the M3 card, evaluates annual policies, evaluates rolling policies, independently replays solutions, and exports certified tables/figures/manifests.

Layer C — post-freeze TPWRS/TSG evidence

The updated evidence scripts operate read-only on the frozen parent study. They compile the annual M1 certificate, perform selected-state nonlinear AC consistency checks, compile frozen rate-bound sensitivity evidence, run an exact post-freeze M3 gamma path, and create the final manuscript figures.

The distinction between Layers B and C is important: the post-freeze scripts are deliberately not allowed to silently modify the frozen parent source tree or refit the selected M3 card.

2. Debit-accounting rules

Let x[n,t] denote the restriction imposed on participant n in hour t, and let P[n] be the participant's reference capacity.

Rule

Debit concept

Implementation

M1

Touched time

Counts materially restricted participant-hours; solved as a MILP because touch indicators are binary.

M1-LP

Continuous M1 lower bound

Projects the continuous relaxation to sum_t x[n,t]/xbar[n,t] <= H[n]; used only as a lower-bound formulation.

M2

Uniform depth

Uses sum_t x[n,t]/P[n] with a participant-specific annual allowance.

M3

Weighted depth

Uses sum_t q[n,t] x[n,t]/P[n]; q comes from a frozen three-rate card.

The common annual M1/M2 comparison uses a 15% duration/depth allowance. The main M3 reporting point is rho = 0.03, with the full frozen grid

0.15, 0.10, 0.075, 0.05, 0.03, 0.02

The optimization is lexicographic:

minimize aggregate directed thermal-limit relaxation; then

among Stage-I-optimal solutions, minimize restricted energy with a deterministic numerical tie-break.

The signed network convention is

background flow = H @ p_background
request flow    = background flow - G @ requested_transport
post flow       = request flow + G @ restriction

All retained signed rows are closed and independently replayed before a solution is accepted.

3. M3-OB-KKT card construction

M3 is not an unconstrained machine-learning policy. The card is trained and frozen under an explicit information and validation protocol.

The training pipeline:

solves M2 teacher problems on the training/validation period;

extracts deterministic selected annual-budget duals;

forms scarcity labels

alpha[n,t] = mu[n] * x[n,t] / P[n]

predicts within-class scarcity using only permitted features;

freezes class-specific score thresholds using training data only;

maps scores to three buckets;

solves a three-rate convex QP for each debit class;

enforces monotone rates, rate bounds, and reference-weighted mean-one neutrality;

selects the deployable candidate on the validation year; and

serializes the selected card, model bundle, thresholds, hashes, and validation metadata before held-out testing.

The three-rate problem minimizes a scarcity-weighted rate objective plus a small quadratic regularizer, subject to

q1 >= q2 >= q3
q_min <= qb <= q_max
reference-weighted mean(q) = 1

The selected score family, thresholds, and card are not refit on 2029–2031.

4. Frozen study split

The code uses the following predeclared year roles:

Role

Years

Training

2025–2027

Validation

2028

Held-out evaluation

2029–2030

Confirmatory evaluation

2031

The configured Dutch case is expected to contain:

45 buses;

114 participants;

847 monitored directed rows; and

8 retained contingencies.

These dimensions are validated, not silently imposed.

5. Repository structure

The updated script archive is logically organized as follows.

.
├── README.md
├── PACKAGE_MANIFEST.json              # Hash manifest for the post-freeze evidence subset
├── SCIENTIFIC_PROTOCOL.md             # Scientific scope of Runs 12–14
│
│   ── Parent frozen optimization implementation ─────────────────────────
├── __init__.py
├── config.py                           # Study years, paths, solver profiles, tolerances
├── io.py                               # Frozen-input discovery, hashes, alignment, manifests
├── model.py                            # Annual/rolling M1, M1-LP, M2, M3 + replay/certificates
├── m3_ob_kkt.py                        # KKT labels, score models, buckets, rate QP, frozen cards
├── reporting.py                        # Transactional outputs, tables, figures, manifests, ZIP export
│
│   ── Parent reproduction stages ────────────────────────────────────────
├── RUN_01_VALIDATE_INPUTS.py           # Validate/hash every frozen input partition
├── RUN_02_SCARCITY_DIAGNOSTIC.py       # M2 teachers, selected dual/KKT diagnostics
├── RUN_03_TRAIN_AND_FREEZE_CARD.py     # Train/validate M3 candidates and freeze selected card
├── RUN_04A_VALIDATE_M1_PROFILE.py      # Validate annual M1 continuous-relaxation lower bound
├── RUN_03B_FREEZE_EVALUATION_V2.py     # Freeze amended evaluation protocol; no card retraining
├── RUN_04_EVALUATE_ANNUAL.py           # Held-out/confirmatory annual M1-LP/M2/M3 evaluation
├── RUN_05_EVALUATE_ROLLING.py          # Rolling M1/M2/M3 evaluation and annual M1 certificate
├── RUN_06_EXPORT_OUTPUTS.py             # Tables, sanity checks, manifests, final package
│
│   ── Narrow post-freeze recovery/audit utilities ───────────────────────
├── RUN_05_RESUME_NUMERICAL_AMENDMENT.py # Controlled rolling-M1 numerical recovery driver
├── RUN_AC_ENGINEERING_AUDIT_V2.py       # Static AC engineering audit; superseded for policy states by V3
├── RUN_AC_POLICY_HOUR_VALIDATION_V3.py  # Selected policy-hour P/Q reconstruction + AC validation
│
│   ── Real post-freeze TPWRS/TSG evidence ───────────────────────────────
├── ieee_tsg_real_config.py             # Paths and controls for post-freeze experiments
├── real_experiment_common.py           # Read-only freeze validation and shared utilities
├── RUN_00_DIAGNOSE_PARENT_FREEZE.py    # Check tracked frozen hashes before post-freeze work
├── RUN_12_REAL_M1_BOUND_ANALYSIS.py     # Annual M1 certificate + chronology diagnostics
├── RUN_13_REAL_SELECTED_AC_VALIDATION.py# Compile/rerun official selected-state AC-V3 evidence
├── RUN_13B_OPTIONAL_FULL_YEAR_BACKGROUND_AC.py # Optional all-hour background AC reconstruction
├── RUN_14A_COMPILE_FROZEN_RATE_BOUND_EVIDENCE.py # Exact frozen tight/base/wide rate cases
├── RUN_14B_REAL_M3_GAMMA_PATH.py       # Exact gamma sensitivity; M3 + M2-MATCHED solves
├── RUN_ALL_IEEE_TSG_REAL_EXPERIMENTS.py# Convenience orchestrator; see note below
│
│   ── Publication figures ───────────────────────────────────────────────
├── PLOT_IEEE_TPWRS_FINAL_FIGURES.py     # Recommended final TPWRS figure generator
├── PLOT_IEEE_TSG_REAL_FIGURES_COLOR.py  # Alternative visual-only color rendering
│
│   ── Legacy/superseded figure utilities ────────────────────────────────
├── run_06_ieee_tsg_experiments.py       # Legacy extractor; not the current gamma workflow
└── plot_ieee_tsg_figures.py             # Legacy plotting script; do not use for final claims

Relation to the older public-data pipeline

If the upstream data-construction scripts are kept in the same GitHub repository, a clean top-level layout is:

.
├── README.md
├── upstream_data_pipeline/
│   ├── pipeline_config.py
│   ├── run_pipeline.py
│   ├── scripts/
│   │   ├── step00_fetch_gopacs_announcements.py
│   │   ├── step01_crosswalk_real.py
│   │   ├── step01b_plot_province_congestion_heatmaps.py
│   │   ├── step02_announcement_panel_real.py
│   │   ├── step03_called_activity_safe.py
│   │   ├── step04_burden_proxy.py
│   │   ├── step05_auditable_state.py
│   │   ├── step06_testbed.py
│   │   ├── build_national_load_history_master.py
│   │   ├── learn_national_load_and_sector_shapes.py
│   │   ├── build_province_sector_targets_and_factors.py
│   │   ├── learn_province_factors.py
│   │   ├── build_provincial_hourly_load_from_learned_targets_updated.py
│   │   ├── build_provincial_residual_load_from_learned_load_optionB.py
│   │   ├── build_province_hour_public_stress_panel.py
│   │   ├── build_nl_ptdf_and_stress.py
│   │   ├── make_targets_and_offshore_panel_from_table.py
│   │   ├── G_build_tdtr_candidate_hour_support.py
│   │   ├── H_export_ptdf_for_debit_paper.py
│   │   ├── F_scarcity_boundary.py
│   │   ├── A_build_debit_optimization_inputs.py
│   │   ├── B_run_annual_lp.py
│   │   ├── B1_run_m1_touched_only.py
│   │   ├── C_train_m3.py
│   │   ├── D_run_full_benchmark.py
│   │   ├── E_make_advanced_figures.py
│   │   └── validate_pipeline.py
│   └── ...
│
├── TPWRS_M3_OB_KKT_FINAL/              # Frozen parent optimization/evaluation package
└── IEEE_TSG_REAL_EXPERIMENTS_V2/       # Read-only post-freeze evidence package

This separation is preferable to mixing post-freeze experiment scripts into the frozen parent directory, because the freeze validator hashes the parent source tree.

6. Required frozen inputs

The optimization layer does not reconstruct the raw GOPACS/public-data pipeline. It expects a frozen Dutch case with the following artifacts.

frozen/
├── network/
│   ├── network_matrices.npz
│   └── network_rows.parquet
├── actions/
│   ├── participant_actions.npz
│   ├── participant_action_metadata.parquet
│   ├── redispatch_actions.npz
│   └── redispatch_action_metadata.parquet
├── contract/
│   ├── participants.parquet
│   ├── request_partition_manifest.json
│   └── requests/
│       └── participant_requests_hourly_{2025..2031}.parquet
└── states/
    └── dutch_nodal_pq_states_hourly_{2025..2031}.parquet

The associated frozen source-data root must also contain

hourly_system_state.parquet
retained_contingencies.parquet

Large raw, derived, frozen, result, cache, and solver-log files should normally remain outside Git unless they are intentionally published as a separate reproducibility release.

7. Software requirements

The current archive does not contain a fully pinned environment file, so exact scientific reproduction should use the same environment recorded for the frozen parent campaign whenever that record is available.

Core Python dependencies are:

numpy
pandas
scipy
pyarrow
joblib
scikit-learn
matplotlib
gurobipy
pandapower

seaborn is needed only by the superseded legacy plotting script.

A valid Gurobi license is required for the annual/rolling optimization and the M3 three-rate QP. The AC validation scripts require a compatible pandapower installation.

A typical environment setup is:

python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install numpy pandas scipy pyarrow joblib scikit-learn matplotlib pandapower gurobipy

Do not change solver versions, tolerances, thread settings, or frozen source files and then describe the run as a byte-for-byte reproduction of the frozen campaign.

8. Configuration

Parent optimization package

Edit config.py only when creating a new independent reproduction. In particular, verify:

FROZEN_ROOT = Path(r"...\results\frozen")
SOURCE_ROOT = Path(r"...\data\frozen_dutch_case")

The parent method freeze is protected by

ALLOW_PARENT_PIPELINE_REBUILD = False

Leave this False for the existing frozen study.

Post-freeze evidence package

Edit ieee_tsg_real_config.py so that

PROJECT_ROOT = Path(r"...\TPWRS_M3_OB_KKT_FINAL")
ADDENDUM_ROOT = Path(r"...\TPWRS_OPERATIONAL_VALIDATION_ADDENDUM")

point to the frozen parent project and AC-V3 addendum respectively.

Post-freeze outputs are written under

TPWRS_M3_OB_KKT_FINAL/results/ieee_tsg_real_experiments/

The post-freeze scripts themselves should remain outside TPWRS_M3_OB_KKT_FINAL.

9. Reproduction workflows

A. Reproduce the parent method from scratch

Use this only for a deliberate independent reproduction in a clean directory. It is not the normal way to extend an already frozen study.

python RUN_01_VALIDATE_INPUTS.py
python RUN_02_SCARCITY_DIAGNOSTIC.py
python RUN_03_TRAIN_AND_FREEZE_CARD.py
python RUN_04A_VALIDATE_M1_PROFILE.py
python RUN_03B_FREEZE_EVALUATION_V2.py
python RUN_04_EVALUATE_ANNUAL.py
python RUN_05_EVALUATE_ROLLING.py
python RUN_06_EXPORT_OUTPUTS.py

If METHOD_FREEZE.json already exists, Runs 01–03 are intentionally locked unless the user explicitly enables a separate full rebuild.

B. Continue an already frozen Evaluation-V2 study

The normal parent-side sequence is:

python RUN_04A_VALIDATE_M1_PROFILE.py
python RUN_03B_FREEZE_EVALUATION_V2.py
python RUN_04_EVALUATE_ANNUAL.py
python RUN_05_EVALUATE_ROLLING.py
python RUN_06_EXPORT_OUTPUTS.py

Do not regenerate the parent card or overwrite the original method freeze simply because a later diagnostic script was added.

C. Run the post-freeze manuscript evidence

First diagnose provenance:

python RUN_00_DIAGNOSE_PARENT_FREEZE.py

A changed or missing tracked frozen file is a hard failure. Newly added post-freeze scripts may appear as untracked extras and are reported separately.

Then run the low-risk evidence compilation:

python RUN_12_REAL_M1_BOUND_ANALYSIS.py
python RUN_13_REAL_SELECTED_AC_VALIDATION.py
python RUN_14A_COMPILE_FROZEN_RATE_BOUND_EVIDENCE.py

Run the exact gamma sensitivity when required:

python RUN_14B_REAL_M3_GAMMA_PATH.py

The default gamma grid is

0.25, 0.35, 0.45, 0.55, 0.65, 0.70, 0.80, 0.90, 1.00, 1.10,
1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 1.95, 2.00

with

gamma = 0 conceptually corresponding to the flat M2 card;

gamma = 1 reproducing the frozen selected M3 card; and

each gamma/year requiring an M3 solve and a participant-specific M2-MATCHED solve.

For a smoke test, temporarily set

GAMMA_OVERRIDE_GRID = (0.7, 1.0, 1.3)

then remove the smoke-test outputs, restore

GAMMA_OVERRIDE_GRID = None

and execute the full campaign.

D. Optional all-hour background AC reconstruction

This experiment is disabled by default:

RUN_FULL_YEAR_BACKGROUND_AC = False

To enable it, set for example

RUN_FULL_YEAR_BACKGROUND_AC = True
FULL_YEAR_AC_YEARS = (2031,)

and run

python RUN_13B_OPTIONAL_FULL_YEAR_BACKGROUND_AC.py

This tests the intact all-hour background-state reconstruction. It is not a replacement for the selected policy-state AC-V3 evidence and is not a full annual AC-feasibility proof.

E. Generate the final TPWRS figures

Use the final reporting-only plotter:

python PLOT_IEEE_TPWRS_FINAL_FIGURES.py \
  --results-root ".../TPWRS_M3_OB_KKT_FINAL/results/ieee_tsg_real_experiments" \
  --output-dir ".../figures"

It produces:

Fig1_M1_Certificate_TPWRS.pdf
Fig2_AC_Consistency_TPWRS.pdf
Fig3_M3_Gamma_Path_TPWRS.pdf
FIGURE_STATUS_TPWRS.json

PNG previews are generated as well.

10. Post-freeze output structure

results/ieee_tsg_real_experiments/
├── 01_m1_bound_analysis/
├── 02_selected_ac_validation/
├── 02b_full_year_background_ac/        # optional
├── 03a_frozen_rate_bound_evidence/
├── 03b_m3_gamma_path/
└── figures/

The parent study separately writes structured artifacts under

results/
├── manifests/
├── validation/
├── scarcity/
├── cards/
├── annual/
├── rolling/
├── replay/
├── ac/
├── tables/
├── figures/
├── cache/
└── logs/

Writes are transactional where appropriate, and status files include artifact hashes so incomplete runs are not silently treated as final evidence.

11. Main scientific certificates and diagnostics

Annual M1 certificate

The valid annual scalar relationship is

S_M2 <= S_M1-LP <= S_M1* <= S_rolling-M1

where

S_M1-LP is the exact continuous-relaxation lower bound;

S_M1* is the unknown annual integer M1 optimum; and

S_rolling-M1 is an implementable rolling M1 upper bound.

Hourly cumulative M1-LP and rolling-M1 curves are chronology diagnostics only. They are not pointwise lower/upper bounds on the unknown annual M1 optimum.

M2-MATCHED mechanism decomposition

For the selected M3 trajectory, the code solves an ex-post M2 problem with participant allowances matched to the M3 realized physical FDE. This separates

the entitlement/allocation effect; from

the state-shape effect under equal physical exposure.

Nonlinear AC consistency

The AC-V3 audit reconstructs actual selected policy-hour nodal P/Q states and frozen policy actions, then solves AC power flow on intact and exactly mapped configured-outage topologies.

The flow comparison is

DC active power (MW)  vs.  AC active power (MW)

not DC MW versus AC MVA.

Reported diagnostics include MAE, RMSE, identity-prediction R², Spearman rank correlation, sign agreement, voltage ranges, convergence, and restoration sensitivities.

Exact M3 gamma path

For the frozen selected rate vector q*, the sensitivity begins from

q_raw(gamma) = 1 + gamma * (q* - 1)

and projects non-endpoint values back to the feasible monotone, reference-neutral rate set inside [0.25, 2.0].

The main metrics are

relief = 100 * (S_M2 - S_M3) / S_M2

matched efficiency =
    100 * (S_M2 - S_M3) / (S_M2 - S_M2_MATCHED)

along with physical FDE and burden changes.

12. Interpretation rules

The following restrictions are part of the scientific protocol:

Do not describe hourly M1-LP and rolling-M1 trajectories as a pointwise optimality envelope.

Do not call the seven-day block-bootstrap band an optimization-gap confidence interval.

Do not independently bootstrap participants; they are coupled through the network and annual optimization.

Do not interpret AC convergence alone as AC feasibility or a complete security certificate.

Do not compare PTDF MW and AC MVA as if they were the same quantity.

Do not claim gamma stability, monotonicity, or a performance threshold unless the exact gamma solves establish it.

Report M3 security relief together with physical FDE, burden, M2-MATCHED, and the equal-exposure shape penalty.

Preserve negative results. A flat M3 card, positive security restoration, or a control outperforming the selected card is a scientific outcome, not a software failure.

13. Provenance and freeze discipline

The repository uses SHA-256 hashes and explicit freeze/status manifests to separate:

frozen parent source;

frozen input data;

frozen M3 cards and model bundles;

amended Evaluation-V2 logic;

completed annual/rolling result registries; and

post-freeze read-only evidence.

RUN_00_DIAGNOSE_PARENT_FREEZE.py validates the scientifically relevant invariant: every source file registered in the Evaluation-V2 freeze must still exist with its exact frozen hash. New post-freeze scripts are reported as untracked extras rather than being confused with modified frozen code.

PACKAGE_MANIFEST.json in the current archive hashes the post-freeze evidence subset. It should not be interpreted as a complete hash inventory of every Python file in the combined repository. If the entire combined source tree is published as one release, generate a separate top-level release manifest for all distributed files.

14. Current vs. legacy scripts

For the current paper/evidence workflow:

Use

RUN_12_REAL_M1_BOUND_ANALYSIS.py
RUN_13_REAL_SELECTED_AC_VALIDATION.py
RUN_14A_COMPILE_FROZEN_RATE_BOUND_EVIDENCE.py
RUN_14B_REAL_M3_GAMMA_PATH.py
PLOT_IEEE_TPWRS_FINAL_FIGURES.py

Treat as legacy/superseded

run_06_ieee_tsg_experiments.py
plot_ieee_tsg_figures.py

The legacy plotting script contains interpretations that are intentionally rejected by the current scientific protocol, including a pointwise M1 “optimality envelope” and an MW-versus-MVA parity presentation.

RUN_AC_ENGINEERING_AUDIT_V2.py remains useful as a static engineering audit, but RUN_AC_POLICY_HOUR_VALIDATION_V3.py is the relevant selected-policy-state nonlinear consistency check.

PLOT_IEEE_TSG_REAL_FIGURES_COLOR.py is an alternative visual-only renderer; it does not change numerical results.

One-command runner note

The current RUN_ALL_IEEE_TSG_REAL_EXPERIMENTS.py references

PLOT_IEEE_TSG_REAL_FIGURES.py

but that filename is not present in the supplied archive. Until the runner is updated, execute Runs 12–14 explicitly and then call PLOT_IEEE_TPWRS_FINAL_FIGURES.py (recommended) or PLOT_IEEE_TSG_REAL_FIGURES_COLOR.py directly.

15. Suggested .gitignore

The source code, configuration templates, scientific protocol, and small manifests belong in Git. Large/reconstructable data and generated solver outputs generally do not.

# Python
__pycache__/
*.py[cod]
.venv/
venv/

# IDE / OS
.idea/
.vscode/
.DS_Store
Thumbs.db

# Local data and frozen inputs
data/external/
data/derived/
GOPACS/raw/
results/frozen/

# Generated optimization artifacts
results/cache/
results/logs/
results/.transactions/
*.sol
*.mst
*.bas
*.lp
*.mps

# Large generated outputs
outputs/results/
outputs/figures/
outputs/tables/
PTDF_OUTPUT_PUBLICATION/

# Local machine configuration / temporary files
*.tmp

If selected result tables, figure PDFs, or audit manifests are intentionally part of the reproducibility release, explicitly whitelist those files rather than committing every generated artifact.

16. Reproducibility checklist

Before treating a run as comparable to the frozen study, verify that:

the frozen input root is complete;

the aggregate input hash matches the expected frozen value;

METHOD_FREEZE.json is unchanged;

EVALUATION_FREEZE_V2.json is unchanged;

the selected M3 card/model-bundle hashes match;

no confirmatory-year information enters training or candidate selection;

annual/rolling solver status is certified;

signed-row closure passes;

independent replay residuals are within tolerance;

annual budget primal/dual/complementarity checks pass where applicable;

Run-04 and Run-05 status registries are COMPLETE;

post-freeze scripts pass RUN_00_DIAGNOSE_PARENT_FREEZE.py; and

manuscript figures are generated only from committed experiment artifacts.

17. Research use and citation

This repository is intended to make the computational evidence auditable: assumptions, frozen decisions, optimization inputs, solver outputs, replay checks, sensitivity analyses, and publication artifacts are separated and hash-tracked.

If you use the code, please cite the associated paper once its final bibliographic information is available. The repository README should be updated with the final title, authors, journal, DOI, and archival release tag at publication time.

18. Contact / issues

For reproducibility problems, open a GitHub issue and include:

the script name;

Python/Gurobi/pandapower versions;

the relevant status or failure JSON;

the exact exception/solver status; and

whether the run is a fresh reproduction, continuation of the frozen parent study, or post-freeze analysis.

Do not resolve a freeze/hash failure by regenerating the freeze manifest. Restore the expected frozen source or run a clearly separated new reproduction instead.
