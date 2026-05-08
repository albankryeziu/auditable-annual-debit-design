# TDTR debit-rule pipeline run order

## Main medium scenario

From repo root:

```bat
cd "C:\Users\Alban\OneDrive - University of Groningen\Desktop\Final Paper Script"

set TDTR_SUPPORT_SCENARIO=medium
set TDTR_SUPPORT_REQUIRED=1
set PTDF_APPLY_TDTR_SUPPORT_TO_REQUIREMENTS=1
set TDTR_SUPPORT_APPLY_TO_REQUIREMENTS=1
set TDTR_SUPPORT_APPLY_TO_PBAR=1

python scripts\build_province_hour_public_stress_panel.py
REM Run your Step 01--05 scripts here if auditable_state_province_hour.csv is not already current.
REM Required target: data\derived\auditable_state_province_hour.csv

python scripts\G_build_tdtr_candidate_hour_support.py
python scripts\H_export_ptdf_for_debit_paper.py
python scripts\A_build_debit_optimization_inputs.py
python scripts\C_train_m3.py
python scripts\D_run_full_benchmark.py
python scripts\E_make_advanced_figures.py
```

## Scenario sensitivity loop

G can build all scenario files in one run. Then rerun H/A/C/D/E for the active scenario.

```bat
cd "C:\Users\Alban\OneDrive - University of Groningen\Desktop\Final Paper Script"

python scripts\G_build_tdtr_candidate_hour_support.py

for %%S in (restrictive medium wide) do (
    set TDTR_SUPPORT_SCENARIO=%%S
    set TDTR_SUPPORT_REQUIRED=1
    set PTDF_APPLY_TDTR_SUPPORT_TO_REQUIREMENTS=1
    set TDTR_SUPPORT_APPLY_TO_REQUIREMENTS=1
    set TDTR_SUPPORT_APPLY_TO_PBAR=1

    python scripts\H_export_ptdf_for_debit_paper.py
    python scripts\A_build_debit_optimization_inputs.py
    python scripts\C_train_m3.py
    python scripts\D_run_full_benchmark.py
    python scripts\E_make_advanced_figures.py
)
```

## Minimum rerun after changing only the TDTR support scenario

```bat
set TDTR_SUPPORT_SCENARIO=restrictive
python scripts\H_export_ptdf_for_debit_paper.py
python scripts\A_build_debit_optimization_inputs.py
python scripts\C_train_m3.py
python scripts\D_run_full_benchmark.py
python scripts\E_make_advanced_figures.py
```

For a theoretically clean scenario comparison, rerun C as well, because M3_LO uses optimization-informed solves and the learned weights can depend on the scenario-specific operating layer.
