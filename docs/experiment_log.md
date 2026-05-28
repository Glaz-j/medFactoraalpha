# Experiment Log

## 2026-05-22

### Medical QuantaAlpha Workflow Status

Confirmed that the medical QuantaAlpha workflow can run end-to-end on eICU LOS:

```text
factor_propose
-> factor_construct
-> factor_calculate
-> factor_backtest
-> feedback
```

The original PyHealth evaluator remains available:

```bash
MEDICAL_EVALUATOR=pyhealth
```

This route trains:

```text
QuantaAlphaPyHealthModel
-> PyHealth MultimodalRNN
-> conditions/procedures/drugs GRU branches
-> symbolic_factors linear branch
-> LOS multiclass classifier
```

Full-data smoke completed with:

```text
EICU_DEV=0
WORKFLOW_STEPS=5
EPOCHS=3
BATCH_SIZE=128
SEED=0
DEVICE=cuda:0
```

Scale:

```text
patients: 139,367
events: 19,425,700
LOS samples: 140,132
train/val/test: 112,249 / 13,880 / 14,003
```

Result:

```json
{
  "accuracy": 0.37256302220952653,
  "balanced_accuracy": 0.24937164506577614,
  "cohen_kappa": 0.1949373057042657,
  "f1_macro": 0.21416862887664823,
  "f1_weighted": 0.3407613902712229,
  "loss": 1.5754965088584207
}
```

Run artifacts:

```text
results/pyhealth_runs/eicu_los_quantaalpha_workflow_full_e8f78106/
results/quantaalpha_medical_workflow/workflow_factors_e8f78106.json
results/quantaalpha_medical_workflow/workflow_factor_summary_e8f78106.json
```

### Qlib-like Tabular Evaluator Added

Added a second medical evaluator that is closer to original QuantaAlpha's Qlib-style factor evaluation:

```bash
MEDICAL_EVALUATOR=tabular
```

New runner:

```text
baselines/QuantaAlpha/quantaalpha/medical/tabular_runner.py
```

Workflow entrypoint now selects evaluator through `MEDICAL_EVALUATOR`:

```text
experiments/quantaalpha_medical_workflow_eicu_los.py
```

The tabular runner evaluates each factor set as an explicit factor table with:

```text
baseline_logistic
baseline_gbdt
factors_logistic
factors_gbdt
combined_logistic
combined_gbdt
```

Because the current environment has `sklearn` but not `lightgbm` or `xgboost`, the GBDT route currently uses:

```text
sklearn.ensemble.HistGradientBoostingClassifier
```

The linear route uses:

```text
sklearn.linear_model.LogisticRegression
```

### Tabular Smoke Test

Command:

```bash
EICU_DEV=1 \
SEED=0 \
BATCH_SIZE=64 \
NUM_WORKERS=4 \
WORKFLOW_STEPS=5 \
LLM_MODEL=gpt-5.5 \
MEDICAL_EVALUATOR=tabular \
MEDICAL_GBDT_MAX_ITER=20 \
MEDICAL_TABULAR_PERMUTATION_IMPORTANCE=0 \
/home/lzp/anaconda3/envs/medfactoraalpha-pyhealth/bin/python \
  experiments/quantaalpha_medical_workflow_eicu_los.py
```

Result artifacts:

```text
results/pyhealth_runs/eicu_los_quantaalpha_tabular_dev_12ee0c10/final_test_scores.json
results/pyhealth_runs/eicu_los_quantaalpha_tabular_dev_12ee0c10/all_model_scores.json
results/pyhealth_runs/eicu_los_quantaalpha_tabular_dev_12ee0c10/feature_importance.json
```

Selected dev scores:

```text
baseline_logistic f1_macro: 0.0438
factors_logistic  f1_macro: 0.1294
combined_logistic f1_macro: 0.1294
factors_gbdt      f1_macro: 0.0796
combined_gbdt     f1_macro: 0.0796
```

Interpretation:

```text
The factor-only linear model already exceeds the simple tabular baseline on dev,
which suggests the LLM-generated medical factors contain interpretable signal.
The dev GBDT result is weaker, likely because the smoke used a tiny dev subset
and MEDICAL_GBDT_MAX_ITER=20.
```

### Architectural Decision

Keep both evaluators:

```text
MEDICAL_EVALUATOR=pyhealth  # strong EHR-sequence model, GRU-based
MEDICAL_EVALUATOR=tabular   # Qlib-like factor table + light models
```

The tabular route should be the main interpretability/QuantaAlpha-faithful evaluator.
The PyHealth GRU route should remain a strong predictive baseline and factor-enhanced
EHR model comparison.

Recommended next full tabular run:

```bash
EICU_DEV=0 \
SEED=0 \
BATCH_SIZE=128 \
NUM_WORKERS=4 \
WORKFLOW_STEPS=10 \
LLM_MODEL=gpt-5.5 \
MEDICAL_EVALUATOR=tabular \
MEDICAL_TABULAR_MODELS=logistic \
MEDICAL_TABULAR_FEATURE_SETS=factors,combined \
MEDICAL_TABULAR_PRIMARY_MODEL=combined_logistic \
MEDICAL_LOGISTIC_MAX_ITER=300 \
/home/lzp/anaconda3/envs/medfactoraalpha-pyhealth/bin/python \
  experiments/quantaalpha_medical_workflow_eicu_los.py
```

## 2026-05-22 Tabular Evaluator Bottleneck Fix

Problem:

```text
The first tabular implementation still used PyHealth set_task/task transforms
for each new factor hash. On full eICU this made each factor evaluation spend
many minutes rebuilding patient/sample caches before the lightweight model even
started fitting.
```

Changes:

```text
1. Use a shared tabular cache independent of factor hash.
2. Build a direct Polars eICU LOS base sample table once:
   results/pyhealth_cache/eicu_los_quantaalpha_tabular_full/tabular_base_samples_full.parquet
3. For each LLM factor set, reuse the base table and only materialize the
   factor matrix.
4. Parallelize factor matrix materialization with NUM_WORKERS processes.
5. Default tabular model changed to logistic regression; GBDT remains optional.
```

Final full workflow validation command:

```bash
EICU_DEV=0 \
SEED=0 \
BATCH_SIZE=128 \
NUM_WORKERS=4 \
WORKFLOW_STEPS=5 \
LLM_MODEL=gpt-5.5 \
MEDICAL_EVALUATOR=tabular \
MEDICAL_TABULAR_MODELS=logistic \
MEDICAL_TABULAR_FEATURE_SETS=factors,combined \
MEDICAL_TABULAR_PRIMARY_MODEL=combined_logistic \
MEDICAL_LOGISTIC_MAX_ITER=300 \
/home/lzp/anaconda3/envs/medfactoraalpha-pyhealth/bin/python \
  experiments/quantaalpha_medical_workflow_eicu_los.py
```

Final validation result:

```text
factor_hash: 270f8c53
sample_size: 140132
split_sizes: train=112195, val=14013, test=13924
primary_model: combined_logistic
f1_macro: 0.17765339735771488
accuracy: 0.2894283251939098
loss: 1.9813416371390242
```

Final evaluator timings:

```text
base_sample_cache_sec: 1.160
materialize_sec: 35.325
model_fit_and_score_sec: 13.769
split_sec: 0.226
total evaluator sec: 50.535
full 5-step workflow wall time: about 1 min 47 sec
```

Interpretation:

```text
The bottleneck is no longer model training. It is factor matrix materialization,
especially keyword matching across eICU text lists. After direct Polars base
sample caching and process-level parallelization, full-data tabular evaluation
is fast enough for iterative QuantaAlpha-style workflow loops.
```

## 2026-05-22 30-Step Tabular Workflow Review

Run shape:

```text
WORKFLOW_STEPS=30 means 6 complete workflow loops.
Evaluator: tabular logistic
Feature sets: factors, combined
Sample size: 140132
```

Loop scores:

```text
loop  hash      factors_f1  combined_f1
0     455116f0  0.137842    0.173239
1     f9cb3acf  0.133369    0.166725
2     1774cc7d  0.149513    0.173046
3     0e000a6d  0.108915    0.171771
4     dbab6241  0.134151    0.168832
5     ffa63a32  0.092544    0.168947
```

Standalone baseline-only logistic on the same split:

```text
baseline_logistic test f1_macro: 0.160592
```

Interpretation:

```text
The workflow did not show monotonic improvement. The best loop was loop 0
with combined_f1=0.173239. The final loop was lower at 0.168947. The symbolic
factor sets have weak standalone predictive power, and the combined model gains
only about +0.008 to +0.013 f1_macro over the baseline-count features.

Feature importance shows the combined model is still dominated by baseline
count/ratio features such as condition_log_count, drug_log_count, and
procedure_to_drug_ratio. Several LLM-generated factors are zero-match or too
sparse because the proposed clinical keywords do not reliably match the eICU
diagnosis, medication, and physical exam strings.
```

Action items:

```text
1. Always include baseline_logistic in tabular workflow reports.
2. Feed factor activity/nonzero-rate constraints back to the LLM more strongly.
3. Add an eICU vocabulary profiling step so the LLM proposes keywords from
   observed strings instead of plausible but absent clinical phrasing.
4. Optimize against incremental gain over baseline, not only absolute combined_f1.
```

Implementation update:

```text
Done.

1. The tabular runner now auto-adds baseline to MEDICAL_TABULAR_FEATURE_SETS
   by default, controlled by MEDICAL_TABULAR_ALWAYS_BASELINE=1.
2. Each tabular run writes baseline_deltas.json and includes baseline_deltas
   in exp.result, so feedback can judge combined-minus-baseline f1_macro.
3. The medical scenario now includes an observed eICU vocabulary profile in
   the LLM prompt by default, controlled by MEDICAL_VOCAB_PROFILE=1.
4. Proposal history now includes compact factor activity summaries and
   baseline deltas, so the next LLM round can avoid zero-activity keywords.
```

New artifacts:

```text
results/quantaalpha_medical_workflow/eicu_vocab_profile_full.json
results/quantaalpha_medical_workflow/eicu_vocab_profile_dev.json
results/pyhealth_runs/<run>/baseline_deltas.json
```

## 2026-05-22 Alpha-Style Medical Workflow

Goal:

```text
Move the medical workflow closer to original QuantaAlpha AlphaAgent behavior:
use richer hypothesis history, factor-construction prompts, regulator feedback,
LLM critic/refinement, and baseline-delta feedback while keeping only the data
adapter/evaluator medical-specific.
```

Implementation:

```text
1. Added quantaalpha.medical.alpha_proposal.MedicalAlphaHypothesisGen.
   This mirrors original AlphaAgentHypothesisGen: it uses history windows,
   prior feedback, SOTA/baseline deltas, factor activity, and a formal
   hypothesis JSON schema.

2. Added quantaalpha.medical.alpha_proposal.MedicalAlphaHypothesis2FactorExpression.
   This mirrors original AlphaAgentHypothesis2FactorExpression structure:
   construct factors, run a regulator, feed regulator feedback into repair
   attempts, and optionally call an LLM factor critic before execution.

3. Added MedicalDSLRegulator.
   It checks medical DSL validity, duplicate factor signatures, observed eICU
   vocabulary coverage, overly long factor definitions, and repeated factors
   from prior rounds.

4. Added MedicalAlphaTabularFactorSetting and made tabular workflow default to
   MEDICAL_WORKFLOW_FLAVOR=alpha. Set MEDICAL_WORKFLOW_FLAVOR=light to return
   to the previous lightweight medical DSL flow.

5. Added LLM usage JSONL logging through LLM_USAGE_LOG_PATH, defaulting to:
   results/quantaalpha_medical_workflow/logs/llm_usage.jsonl
```

Smoke test:

```bash
EICU_DEV=1 \
SEED=0 \
BATCH_SIZE=64 \
NUM_WORKERS=4 \
WORKFLOW_STEPS=5 \
LLM_MODEL=gpt-5.5 \
MEDICAL_EVALUATOR=tabular \
MEDICAL_WORKFLOW_FLAVOR=alpha \
MEDICAL_TABULAR_MODELS=logistic \
MEDICAL_TABULAR_FEATURE_SETS=factors,combined \
MEDICAL_TABULAR_PRIMARY_MODEL=combined_logistic \
MEDICAL_LOGISTIC_MAX_ITER=300 \
MEDICAL_ALPHA_REPAIR_ATTEMPTS=1 \
/home/lzp/anaconda3/envs/medfactoraalpha-pyhealth/bin/python \
  experiments/quantaalpha_medical_workflow_eicu_los.py
```

Smoke result:

```text
workflow_flavor: alpha
factor_hash: fb6abb02
models: baseline_logistic, factors_logistic, combined_logistic
baseline test f1_macro: 0.0564
factors test f1_macro: 0.0673
combined test f1_macro: 0.1162
combined-minus-baseline test f1_macro delta: +0.0598
wall time: about 2 min 36 sec for one dev loop
```

Interpretation:

```text
The alpha-style medical flow is now substantially more LLM-heavy than the
lightweight flow. Per loop it can call LLM for hypothesis generation, factor
construction, factor critic/refinement, and final feedback. It still keeps
factor execution deterministic through the medical DSL and tabular evaluator.
```

## 2026-05-22 Alpha Full Run Diagnosis And Fix

Observed full alpha run:

```text
WORKFLOW_STEPS=30 -> 6 loops
latest final factor_hash: d847addd
final combined f1_macro: 0.160589
baseline f1_macro: 0.160592
combined-minus-baseline f1_macro delta: -0.000003
```

Loop curve:

```text
loop  hash      factors_f1  combined_f1  delta_test  n_factors
0     fad86206  0.142395    0.167243     +0.006651   3
1     3d68fc6f  0.099887    0.164879     +0.004287   3
2     aa339c6a  0.132367    0.169894     +0.009303   3
3     9f288c50  0.092259    0.168767     +0.008176   3
4     58d9d815  0.078229    0.164342     +0.003750   1
5     d847addd  0.052040    0.160589     -0.000003   1
```

Diagnosis:

```text
Token usage is now clearly higher: factor construction prompts reached about
18k-20k tokens in the full run. The issue was not insufficient LLM usage.

The issue was an implementation mismatch with the intended AlphaAgent behavior:
the LLM hypothesis said to retain the prior compact core and add one hemodynamic
factor, but the medical runner evaluated only the current new factor batch. In
the last two loops the LLM/critic also became too conservative and generated
only one factor, so the workflow effectively tested a single incremental factor
instead of the intended SOTA-core-plus-new-factor set.
```

Fix:

```text
1. MEDICAL_ALPHA_MIN_FACTORS defaults to 3, preventing alpha construction from
   degenerating into one-factor batches.
2. MEDICAL_TABULAR_INCLUDE_BASED_FACTORS defaults to 1, so accepted historical
   experiments/SOTA factors are merged into the current tabular evaluation.
3. MEDICAL_TABULAR_MAX_EVAL_FACTORS defaults to 24 to cap accumulated factors.
4. Run artifacts now record based_factor_count, new_factor_count,
   evaluated_factor_count, and evaluated_factor_names.
```

Post-fix dev verification:

```text
WORKFLOW_STEPS=10 -> 2 loops
second loop factor_hash: e71291cb
based_factor_count: 3
new_factor_count: 3
evaluated_factor_count: 6
combined-minus-baseline test f1_macro delta: +0.090433
```

## 2026-05-22 Alpha Full Run Review: Inheritance vs Ablation

Observed full alpha run:

```text
WORKFLOW_STEPS=30 -> 6 loops
final factor_hash: 2a51b16d
baseline_logistic test f1_macro: 0.160592
combined_logistic test f1_macro: 0.178584
combined-minus-baseline test f1_macro delta: +0.017992
combined-minus-baseline val f1_macro delta: +0.020760
evaluated_factor_count: 12
based_factor_count: 9
new_factor_count: 3
```

Loop curve:

```text
loop  hash      evaluated  combined_f1  delta_test  delta_val
0     a55b4d2c  3          0.170046     +0.009454   +0.013094
1     1b8177b1  6          0.174374     +0.013782   +0.011781
2     c8f85e53  9          0.175935     +0.015343   +0.013589
3     98e1d2a9  12         0.177901     +0.017309   +0.017291
4     f1b372f5  12         0.176891     +0.016299   +0.011928
5     2a51b16d  12         0.178584     +0.017992   +0.020760
```

Interpretation:

```text
The performance signal is real and stable: final combined logistic improves
over the baseline by about +0.018 f1_macro on test and +0.021 on validation.
However, attribution is not clean. The final factor table retained legacy /
proxy factors that the hypothesis wanted to ablate, especially
acute_organ_failure_shock_conditions and the saturated
metabolic_electrolyte_insulin_drug_density factor (nonzero_rate about 0.904).
The LLM also asked for an exact 10-factor clean set, but the factor-construction
prompt still told it to generate only 2-3 factors per round.
```

Fix:

```text
1. MEDICAL_ALPHA_MAX_FACTORS now defaults to 12, so exact/clean-slate/full-list
   hypotheses can materialize a complete factor set.
2. The factor construction and critic prompts now preserve full requested sets
   for clean-slate, exact-name, ablation, or full factor-list hypotheses instead
   of always compressing back to 2-3 factors.
3. MedicalFactorExperiment stores target_hypothesis for the runner.
4. The tabular runner now supports an auto factor-selection policy: it still
   inherits historical/SOTA factors by default, but can infer requested and
   denied factor names from exact/clean-slate/remove/exclude hypotheses and
   filter the evaluated table accordingly.
5. Selection metadata is written into workflow_factors_*.json and exp.result.
```

## 2026-05-22 Requested-Filter Regression

Observed follow-up full alpha run:

```text
final factor_hash: 74063de8
final evaluated_factor_count: 2
final combined_logistic test f1_macro: 0.165236
combined-minus-baseline test f1_macro delta: +0.004644
```

The middle of the same run was much stronger:

```text
loop  hash      evaluated  combined_f1  delta_test  delta_val
0     3f7107be  3          0.171176     +0.010584   +0.012170
1     18c79c08  9          0.175524     +0.014932   +0.013671
2     3cd6ef60  14         0.183374     +0.022782   +0.020328
3     719b85d7  4          0.168356     +0.007764   +0.008318
4     7bb1b50d  24         0.186513     +0.025921   +0.027141
5     74063de8  2          0.165236     +0.004644   +0.005043
```

Diagnosis:

```text
The requested-factor allowlist was too aggressive. The final hypothesis
mentioned several requested factor names, but only two of those names were
present among the generated/inherited candidates. The runner therefore filtered
32 available candidates down to only invasive_ventilation_procedure_marker and
sedation_opioid_treatment_intensity, causing the final score to collapse.
```

Fix:

```text
1. Requested-name filtering is now applied only when at least
   MEDICAL_TABULAR_REQUESTED_MIN_MATCHES candidate names match; default is 3.
   Otherwise the runner applies denylist filtering only and keeps the broader
   inherited/new factor pool.
2. The selection policy now records active_requested_factor_names and
   requested_filter_applied.
3. Factor merging now deduplicates by factor name as well as signature; later
   generated factors replace older same-name factors, matching refinement
   behavior more closely.
```

## 2026-05-22 Empty Selection Crash Fix

Observed crash:

```text
ValueError: No factors left after applying tabular factor selection policy.
requested included baseline_logistic, f1_macro, log_count, keyword_count,
drug_log_count, procedure_log_count, phenotype_* names, and several real
factor names. denied overlapped with the remaining real names.
```

Diagnosis:

```text
The automatic text parser was still treating metric names, model names,
baseline feature names, and DSL operation names as possible factor names. In
one loop the requested/denied sets overlapped enough that all candidate factors
were filtered out before tabular evaluation.
```

Fix:

```text
1. Added a non-factor text-name filter for metrics, model keys, baseline
   feature names, DSL operation names, and phenotype_* experiment labels.
2. Empty automatic selections no longer crash by default. Unless
   MEDICAL_TABULAR_STRICT_SELECTION=1, the runner falls back to the current
   generated factor batch and records selection_fallback in artifacts.
3. Strict mode can still be enabled when an explicit allowlist/denylist test
   should fail fast.
```

## 2026-05-22 Long Alpha Accumulation Run

Command shape:

```text
EICU_DEV=0
WORKFLOW_STEPS=100
MEDICAL_WORKFLOW_FLAVOR=alpha
MEDICAL_TABULAR_MODELS=logistic
MEDICAL_ALPHA_MIN_FACTORS=4
MEDICAL_ALPHA_MAX_FACTORS=12
MEDICAL_TABULAR_MAX_EVAL_FACTORS=64
MEDICAL_TABULAR_REQUESTED_MIN_MATCHES=999
MEDICAL_TABULAR_STRICT_SELECTION=0
```

Final loop:

```text
factor_hash: 002aa3fb
evaluated_factor_count: 36
baseline_logistic test f1_macro: 0.160592
factors_logistic test f1_macro: 0.176726
combined_logistic test f1_macro: 0.186296
combined-minus-baseline test delta: +0.025705
combined-minus-baseline val delta: +0.027434
```

Best test loop:

```text
loop 5
factor_hash: be1dbf55
evaluated_factor_count: 33
factors_logistic test f1_macro: 0.175132
combined_logistic test f1_macro: 0.189402
combined-minus-baseline test delta: +0.028810
```

Best validation loop:

```text
loop 17
factor_hash: db124fef
evaluated_factor_count: 42
combined_logistic val f1_macro: 0.188079
combined_logistic test f1_macro: 0.185824
combined-minus-baseline val delta: +0.028528
```

Interpretation:

```text
Long accumulation helped. The symbolic factor library became predictive on its
own: factors-only logistic reached around 0.175-0.178 f1_macro in later loops,
clearly above the baseline logistic score of 0.1606. Combined models improved
to around 0.186-0.189 f1_macro, with best test delta about +0.0288.

The gain appears to plateau around 30-42 evaluated factors. Later loops did not
monotonically improve test score, but validation stayed strong.
```

Risks and next checks:

```text
Several high-activity factors remain broad treatment/documentation proxies,
including fluid_electrolyte_metabolic_drug_burden, analgesia_sedation_drug_burden,
and respiratory_failure_support_burden. Some generated names such as audit_v*
and aspirin_sentinel require cleanup before claiming interpretability. Next
step should be best-run preservation plus ablation/compression rather than only
longer accumulation.
```

## 2026-05-22 Temporal Symbolic Factor DSL

Implementation:

```text
1. Direct tabular eICU cache now has a temporal variant:
   tabular_base_samples_{dev,full}_temporal.parquet
2. The temporal cache keeps aligned event offset lists:
   conditions_offsets from diagnosisoffset
   procedures_offsets from physicalexamoffset
   drugs_offsets from coalesce(drugstartoffset, drugorderoffset)
3. Existing stay-level sources remain compatible:
   conditions, procedures, drugs
4. Added temporal DSL operations:
   temporal_keyword_count
   temporal_keyword_density
   first_keyword_offset
   early_late_keyword_delta
   keyword_persistence
5. Alpha prompts and scenario descriptions now expose temporal windows and
   current-stay offset constraints to the LLM.
```

Smoke tests:

```text
Manual temporal-factor dev runner completed on EICU_DEV=1 with 988 samples.
One-loop alpha dev workflow completed successfully with LLM-generated temporal
factors:
  shock_sepsis_condition_burden              keyword_count
  early_resp_support_procedure_density       temporal_keyword_density
  early_sedation_analgesia_drug_count        temporal_keyword_count
```

Notes:

```text
Temporal factors use only current-stay event offsets and do not use
unitdischargeoffset except for the task label construction. Recommended first
windows are [0,24], [24,72], and [72,168] hours.
```

## 2026-05-23 Temporal Alpha Long Run

Command shape:

```text
EICU_DEV=0
WORKFLOW_STEPS=100
MEDICAL_WORKFLOW_FLAVOR=alpha
MEDICAL_TEMPORAL_SOURCE=1
MEDICAL_TABULAR_MODELS=logistic
MEDICAL_ALPHA_MIN_FACTORS=4
MEDICAL_ALPHA_MAX_FACTORS=12
MEDICAL_TABULAR_MAX_EVAL_FACTORS=64
MEDICAL_TABULAR_REQUESTED_MIN_MATCHES=999
MEDICAL_TABULAR_STRICT_SELECTION=0
```

Final loop:

```text
factor_hash: 02616461
evaluated_factor_count: 64
temporal_factor_count: 61
baseline_logistic test f1_macro: 0.158898
factors_logistic test f1_macro: 0.192472
combined_logistic test f1_macro: 0.207779
combined-minus-baseline test delta: +0.048881
combined-minus-baseline val delta: +0.049154
```

Best test loop:

```text
loop 10
factor_hash: 678b8e27
evaluated_factor_count: 45
temporal_factor_count: 42
factors_logistic test f1_macro: 0.188186
combined_logistic test f1_macro: 0.212001
combined-minus-baseline test delta: +0.053103
```

Best validation loop:

```text
loop 17
factor_hash: 2cf0d371
evaluated_factor_count: 64
temporal_factor_count: 61
combined_logistic val f1_macro: 0.211758
combined_logistic test f1_macro: 0.210343
combined-minus-baseline val delta: +0.051981
```

Interpretation:

```text
Temporal symbolic factors produced the largest gain so far. Compared with the
previous non-temporal long-run best combined f1_macro of about 0.1894, temporal
alpha reached 0.2120 on the test split. Factors-only logistic also reached
0.1925 in the final loop, showing that the symbolic temporal library itself is
now stronger than the baseline count/ratio model.

The dominant learned signals are temporal renal/fluid-balance deltas,
electrolyte/potassium-magnesium persistence, early respiratory support density,
and metabolic/diuretic persistence. Activity rates are not degenerate; no
factor in the run had nonzero_rate > 0.8 or < 0.01.
```

Next checks:

```text
1. Save best-of-run summaries automatically instead of only last-loop summary.
2. Run ablation/compression on 678b8e27 and 2cf0d371 to find a compact
   temporal factor set.
3. Compare logistic vs GBDT on the best temporal factor sets.
```

## 2026-05-23 Numeric Temporal DSL And Expanded Sources

Implementation:

```text
1. Added numeric temporal cache variant:
   tabular_base_samples_{dev,full}_temporal_numeric.parquet
2. Added numeric temporal sources:
   vital_sao2, vital_heartrate, vital_respiration, vital_temperature,
   vital_systemicsystolic, vital_systemicdiastolic, vital_systemicmean,
   io_intake_total, io_output_total, io_net_total, io_dialysis_total,
   resp_fio2, resp_peep
3. Data tables:
   vitalPeriodic.csv -> vital_* sources using observationoffset
   intakeOutput.csv -> io_* sources using intakeoutputentryoffset/intakeoutputoffset
   respiratoryCharting.csv -> resp_fio2 and resp_peep using label-filtered respchartvalue
4. Added numeric temporal DSL operations:
   numeric_window_mean
   numeric_window_min
   numeric_window_max
   numeric_early_late_delta
   numeric_abnormal_fraction
   numeric_persistence
```

Smoke tests:

```text
Manual dev runner with 3 numeric temporal factors completed.
Factors:
  first24h_min_sao2                         numeric_window_min(vital_sao2)
  day1_to_day3_mean_hr_delta                numeric_early_late_delta(vital_heartrate)
  first72h_positive_net_fluid_persistence   numeric_persistence(io_net_total)

Dev scores:
  baseline_logistic test f1_macro: 0.103944
  factors_logistic test f1_macro: 0.193698
  combined_logistic test f1_macro: 0.239160
```

LLM alpha dev smoke:

```text
One-loop alpha dev workflow completed. The LLM generated:
  early_max_fio2                       numeric_window_max(resp_fio2)
  early_shock_sepsis_density           temporal_keyword_density
  sedation_analgesia_persistence       keyword_persistence

This verifies prompt -> regulator -> numeric temporal cache -> factor matrix ->
tabular logistic evaluation.
```

## 2026-05-23 Numeric Temporal Alpha Long Run

Setup:

```text
Dataset: full eICU LOS
Workflow flavor: alpha
Evaluator: tabular logistic
WORKFLOW_STEPS: 150
Theoretical loop budget: 30 loops, 5 workflow steps per loop
Completed evaluator runs observed: 27
Controller reached: loop_index=29, step_index=4, feedback
Primary metric: combined_logistic test f1_macro
```

Final loop:

```text
hash: f2797a4e
combined_logistic test f1_macro: 0.352735
combined_logistic test accuracy: 0.568795
combined_logistic test balanced_accuracy: 0.369705
combined_logistic test cohen_kappa: 0.464295
combined_logistic test f1_weighted: 0.569533
combined_logistic test loss: 1.184140

baseline_logistic test f1_macro: 0.156839
factors_logistic test f1_macro: 0.342184
combined-minus-baseline test f1_macro delta: +0.195895
```

Best observed evaluator run:

```text
hash: 7039a5f7
approx loop: 23
approx workflow step: 120
combined_logistic test f1_macro: 0.356396
combined_logistic test accuracy: 0.569011
combined_logistic test balanced_accuracy: 0.373413
combined_logistic test f1_weighted: 0.569897
combined-minus-baseline test f1_macro delta: +0.199557
```

Progression:

```text
loop 0-5   / step 5-30:   rapid rise, 0.1747 -> 0.2920
loop 6-13  / step 35-70:  first plateau, about 0.292-0.297
loop 14    / step 75:     major jump, 0.2929 -> 0.3476
loop 15-23 / step 80-120: slow improvement to best 0.3564
loop 24+   / step 125+:   no new high; small oscillation/regression
```

Interpretation:

```text
The expanded numeric temporal DSL produced a large step-change in performance.
Compared with the previous temporal symbolic long-run best of about 0.2120
macro-F1, the numeric temporal alpha run reached 0.3564. The factors-only
logistic model reached 0.3422 in the final loop, which means the learned
symbolic factor library now carries most of the predictive signal without
depending heavily on the six baseline count/ratio features.

The strongest learned signals are clinically coherent: persistent tachypnea,
late respiration burden, late positive fluid balance, MAP hypotension
persistence, SaO2 early-late change, net-fluid early-late change, high FiO2
persistence, PEEP burden, respiratory failure burden, and renal/dialysis
signals.

The run suggests that a 30-loop budget is reasonable for exploration, but most
of the useful gain appeared by about step 120. After that point, mutations such
as tightening PEEP from >6 to >8 did not improve the best test score.
```

Next checks:

```text
1. Save and report best-of-run artifacts automatically instead of only the last
   workflow summary.
2. Add an early-stop or patience rule around validation/test proxy improvement
   to avoid spending many loops after the step-120 plateau.
3. Compress/ablate the best factor panel from 7039a5f7 to identify a smaller
   stable factor set.
4. Raise logistic max_iter or tune the solver to remove convergence warnings.
5. Compare the best symbolic factor panel against stronger tabular learners
   after the logistic baseline is stable.
```

## 2026-05-23 Multi-Task Workflow Cleanup

Implemented shared infrastructure for eICU LOS, current-stay mortality, and
readmission workflows.

Changes:

```text
1. Added a shared task config layer:
   quantaalpha.medical.task_config

   It centralizes task name, cache/run slug, label key, label set, primary
   metric, best-of-run metrics, metric directions, and default observation
   horizon.

2. Added best-of-run artifact saving in the tabular runner:
   best_workflow_summary_by_f1_macro.json
   best_workflow_summary_by_balanced_accuracy.json
   best_workflow_summary_by_auroc.json
   best_workflow_summary_by_prauc.json
   best_workflow_summary_by_loss.json
   best_workflow_summary_primary.json

   The primary best file follows each task's primary metric:
     LOS -> f1_macro
     mortality/readmission -> auroc

3. Updated feedback to use task-specific metrics instead of always comparing
   f1_macro. Binary tasks now judge AUROC/PRAUC and f1_macro together, with
   baseline deltas still central.

4. Added observation-horizon leakage guard:
   mortality/readmission default to 48 hours via task config.
   MEDICAL_OBSERVATION_END_HOURS can override it.
   The alpha DSL regulator rejects temporal/numeric factors whose window
   endpoints exceed the configured horizon.
```

Smoke checks:

```text
1. py_compile passed for task_config, tabular_runner, scenario, alpha_proposal,
   feedback, and the three eICU workflow entrypoints.
2. Readmission dev manual factor run completed and wrote best-of-run artifacts.
3. Mortality regulator correctly rejected a numeric factor using a 24-72h window
   under the default 48h observation horizon.
```

## 2026-05-24 CoSTEER-Style Medical RAG

Updated the medical embedding RAG policy to follow the original QuantaAlpha /
CoSTEER retrieval discipline more closely.

Changes:

```text
1. MEDICAL_RAG_STYLE now defaults to costeer.
   Set MEDICAL_RAG_STYLE=legacy to recover the old global top-k behavior.

2. Retrieval is bucketed instead of dumping nearest neighbors:
   - MEDICAL_RAG_FAILURE_LIMIT=1
   - MEDICAL_RAG_SUCCESS_LIMIT=2
   - MEDICAL_RAG_ERROR_LIMIT=1

3. Retrieved memories are compressed to short design/error lessons via
   MEDICAL_RAG_MAX_CHARS_PER_ITEM, default 800 characters.

4. Factor-pattern signatures deduplicate retrieved examples by operation,
   source, numeric source, keywords, windows, and thresholds, and successful
   retrieved examples that duplicate previous trace factors are blocked by
   default.
```

Smoke check:

```text
Using the existing readmission Qwen vector store, a respiratory/hemodynamic
query returned two component-success memories and one error-regulator memory in
about 2.3k characters. py_compile passed for embedding_rag.py.
```

## 2026-05-24 Full-Mode Alignment Pass

Reviewed the current medical workflow against original QuantaAlpha and made the
first "full-mode" alignment pass.

Changes:

```text
1. Disabled hypothesis-text requested-factor filtering by default.
   Original QuantaAlpha accumulates accepted SOTA factors plus new factors; it
   does not turn words like "strict" or "exactly" in the hypothesis into a
   factor allowlist. The old medical behavior caused late readmission loops to
   evaluate only one leftover factor. Requested filtering is now opt-in via:
   MEDICAL_TABULAR_ENABLE_REQUESTED_FILTER=1

2. Expanded non-factor text-name filtering for DSL operation names, source
   names, and parameter names so they are not mistaken for factor names.

3. Added compact medical history rendering by default:
   MEDICAL_ALPHA_COMPACT_HISTORY=1
   This keeps report scores, baseline deltas, factor samples, activity samples,
   and feedback, but avoids repeatedly injecting full all_model_scores/history
   payloads that pushed prompts above 40k tokens.

4. Re-ranked CoSTEER-style RAG success memories with a small task-metric bonus
   in addition to embedding similarity:
   MEDICAL_RAG_PERFORMANCE_WEIGHT=0.25
   This makes strong prior memories more likely to be retrieved instead of only
   the nearest local respiratory/SaO2 ablation examples.
```

Smoke checks:

```text
1. py_compile passed for tabular_runner.py, alpha_proposal.py, and
   embedding_rag.py.
2. A strict/exclude hypothesis no longer enables requested-filter behavior by
   default; based factors continue to accumulate with new non-denied factors.
3. A readmission Qwen RAG query still returns a compact CoSTEER-style package
   of two success memories plus one error memory, around 2.4k characters.
```

## 2026-05-24 Medical Full QuantaAlpha Modules

Implemented the first full-version medical QuantaAlpha modules by following the
original CoSTEER and evolution-controller structure.

Changes:

```text
1. Added quantaalpha.medical.costeer.MedicalDSLCoSTEER.
   This replaces the lightweight parser as the medical coder. It keeps the
   implementation surface as safe JSON DSL, but adds CoSTEER-style task memory:
   duplicate task filtering, failed task trial limits, task signatures, and
   post-feedback success/mixed/failed records.

2. Added medical_dsl_costeer_knowledge.jsonl.
   Feedback now records each evaluated factor's task-level status, activity,
   score deltas, and error tags. Repeatedly failed signatures can be skipped
   before evaluation, analogous to CoSTEER's failed_task_info_set.

3. Added quantaalpha.medical.evolution.MedicalEvolutionController.
   It subclasses the original EvolutionController and maps medical primary
   metrics such as AUROC/F1/loss onto the controller's RankIC-style parent
   selection interface.

4. Added experiments/quantaalpha_medical_evolution_eicu_readmission.py.
   This is a full evolution entrypoint for readmission:
   original multi-direction trajectories -> mutation -> crossover, with each
   trajectory running the standard medical AlphaAgentLoop internally.
```

Smoke checks:

```text
1. py_compile passed for costeer.py, evolution.py, settings.py, feedback.py,
   and the new evolution entrypoint.
2. MedicalDSLCoSTEER kept one of two same-signature SaO2 factors and recorded
   the other as a duplicate task.
3. The evolution entrypoint initialized and wrote an empty smoke summary with
   MEDICAL_EVOLUTION_MAX_TASKS=0.
```

## 2026-05-24 Safe DSL Expansion

Extended the medical safe JSON DSL toward QuantaAlpha-style flexible factor
expressions while keeping strict legality checks and deterministic evaluation.

Changes:

```text
1. Added numeric window statistics:
   numeric_window_std, numeric_window_last, numeric_window_count,
   numeric_window_slope.

2. Added numeric_source_interaction.
   It combines two whitelisted numeric sources inside a legal observation
   window with whitelisted aggregations (mean/min/max/std/last/count),
   operators (add/sub/mul/ratio/max/min), and transforms
   (identity/log1p/abs/neg/sqrt_abs). This gives the LLM a safe expression-like
   surface without allowing arbitrary Python execution.

3. Propagated the new fields through MedicalFactorTask, AlphaAgent prompts,
   the deterministic regulator, CoSTEER task signatures, RAG memory records,
   and the fast tabular evaluator.

4. Updated numeric factor signatures to include windows and interaction fields,
   preventing accidental collisions between same-source factors with different
   temporal definitions.
```

Smoke checks:

```text
1. py_compile passed for dsl.py, tabular_runner.py, experiment.py,
   alpha_proposal.py, proposal.py, scenario.py, costeer.py, and embedding_rag.py.
2. Direct DSL compute passed for numeric_window_slope, numeric_window_std, and
   numeric_source_interaction.
3. Fast cached tabular evaluator matched direct DSL compute on
   numeric_window_last, transformed numeric_window_count, and an interaction
   factor.
```

## 2026-05-24 Keyword-Gated Numeric DSL

Added condition-gated numeric factors to make the safe DSL closer to flexible
QuantaAlpha factor code without allowing arbitrary Python execution.

Changes:

```text
1. Added keyword_gated_numeric.
   It returns a numeric window statistic only when source keywords are present
   in the same observation window; otherwise it returns 0.

2. Extended numeric aggregations with slope so gated factors can express
   trajectory changes such as renal-failure-gated fluid-balance slope.

3. Propagated the op through direct DSL compute, fast cached tabular evaluation,
   AlphaAgent prompts, the deterministic regulator, formula rendering, and the
   scenario interface.
```

Smoke checks:

```text
1. py_compile passed for dsl.py, tabular_runner.py, alpha_proposal.py,
   proposal.py, and scenario.py.
2. Direct DSL compute and cached tabular evaluator matched on a synthetic
   renal-failure-gated fluid-slope factor and an absent-gate zero factor.
3. MedicalDSLRegulator accepted a three-factor gated numeric payload.
```

## 2026-05-24 Expanded eICU Sources

Expanded the medical QuantaAlpha direct tabular data source beyond the original
diagnosis/physicalExam/medication plus selected vitals.

Changes:

```text
1. Text sources now merge additional current-stay tables:
   - conditions: diagnosis + admissionDx
   - procedures: physicalExam + treatment + respiratoryCare airway entries
   - drugs: medication + infusionDrug + admissionDrug

2. Numeric temporal sources now include:
   - vitalAperiodic non-invasive BP and cardiac output
   - lab chemistry, CBC, blood gas, lactate, renal, liver, electrolyte values
   - nurseCharting numeric values such as GCS, pain, RASS, SpO2, glucose
   - infusionDrug rates for vasopressors/norepinephrine/propofol/insulin/milrinone
   - APACHE APS early physiology variables with offset 0

3. Bumped the direct tabular numeric cache suffix to _temporal_numeric_v2 so
   older parquet caches cannot silently omit the new columns.
```

Smoke checks:

```text
1. py_compile passed for dsl.py, tabular_runner.py, alpha_proposal.py, and
   scenario.py.
2. A readmission dev base-sample build completed and wrote
   tabular_base_samples_readmission_dev_temporal_numeric_v2.parquet.
3. The dev frame had shape (1301, 127), including 59 numeric value columns.
   Confirmed new columns such as lab_creatinine_values, lab_lactate_values,
   vital_noninvasivemean_values, nurse_gcs_total_values,
   infusion_vasopressor_rate_values, and apache_creatinine_values.
```

## 2026-05-24 PyHealth Source Alignment

Changed the default medical workflow data source back to the PyHealth-standard
eICU task tables so future experiments are comparable with PyHealth baselines.
The expanded eICU table profile remains available, but must be enabled
explicitly with `MEDICAL_SOURCE_PROFILE=expanded_v2`.

Changes:

```text
1. Added MEDICAL_SOURCE_PROFILE with default pyhealth_standard.
   Default text sources now match PyHealth:
   - readmission/mortality: diagnosis.icd9code, physicalExam.physicalexampath,
     medication.drugname
   - LOS: diagnosis.diagnosisstring, physicalExam.physicalexamvalue,
     medication.drugname

2. Disabled numeric temporal DSL sources under pyhealth_standard. Numeric DSL
   remains available only under expanded_v2.

3. Changed the default mortality tabular task to match PyHealth
   MortalityPredictionEICU: current stay features predict whether the next
   hospital visit/stay discharge status is expired.

4. Made vocabulary profiling source-profile aware so prompts expose the same
   fields as the active PyHealth task profile.

5. Namespaced embedding-RAG memories by source profile so expanded_v2 memories
   do not silently contaminate PyHealth-standard runs.
```

Smoke checks:

```text
1. py_compile passed for source_profile.py, tabular_runner.py,
   alpha_proposal.py, scenario.py, vocab_profile.py, embedding_rag.py, and the
   task entrypoints.
2. Full readmission pyhealth_standard direct frame:
   shape=(38817, 9), labels={1: 21504, 0: 17313}, no numeric value columns.
3. Full mortality pyhealth_standard direct frame:
   shape=(38869, 9), labels={0: 35809, 1: 3060}, no numeric value columns.
4. The DSL regulator rejects numeric operations under pyhealth_standard.
```

## 2026-05-25 Safe Python Factor API

Added the first-stage QuantaAlpha-style generated-code factor path for medical
tabular experiments.

Changes:

```text
1. Added operation=safe_python, gated by MEDICAL_SAFE_PYTHON_FACTORS=1.

2. A safe_python factor must provide code defining exactly compute(sample).
   The generated code receives only current-stay conditions/procedures/drugs
   and their offsets; labels, patient_id, visit_id, filesystem, imports,
   subprocess, eval/exec, and unsafe attributes are unavailable.

3. Added AST validation plus restricted runtime helpers:
   get_texts, events, contains_any, count_keywords, density_keywords,
   first_offset_hours, persistence, safe_div, log1p.

4. Integrated safe_python through normalize_factors, Alpha proposal prompts,
   regulator validation, factor signatures, CoSTEER duplicate filtering,
   embedding/RAG records, MedicalFactorTask serialization, and direct tabular
   materialization.
```

Smoke checks:

```text
1. py_compile passed for safe_python.py, dsl.py, experiment.py, proposal.py,
   alpha_proposal.py, tabular_runner.py, costeer.py, embedding_rag.py, and
   scenario.py.
2. MedicalDSLRegulator accepted three distinct safe_python factors when
   MEDICAL_SAFE_PYTHON_FACTORS=1.
3. Unsafe code attempting file access was blocked by validation.
4. Direct tabular materialization computed a safe_python factor on a 64-sample
   readmission dev slice: base_x=(64, 6), factor_x=(64, 1).
```

Follow-up fix:

```text
1. The first full safe_python run exposed an API mismatch: the LLM naturally
   called persistence(sample, source, keywords, windows), while the helper only
   accepted persistence(events, keywords, windows).
2. The helper now supports both call styles.
3. Runtime errors in a safe_python factor now default to a 0.0 value per sample
   unless MEDICAL_SAFE_PYTHON_STRICT_RUNTIME=1, so a single brittle generated
   factor cannot abort a full materialization job.
4. The previously failing factor materialized successfully on a 256-sample
   full-readmission slice after the fix.
```

## 2026-05-25 Safe Python Readmission 6-Loop Result

Ran a full PyHealth-standard eICU readmission workflow with safe Python enabled.

Result:

```text
Output root:
results/quantaalpha_medical_workflow_readmission_pyhealth_safe_python_6loops_v2

Best combined_logistic:
AUROC 0.6218
PRAUC 0.6461
F1 macro 0.5814
Accuracy 0.5860
Loss 0.6733

Baseline logistic:
AUROC 0.5293
PRAUC 0.5794
F1 macro 0.5142
Accuracy 0.5215
Loss 0.6913

Best combined-minus-baseline:
AUROC +0.0925
PRAUC +0.0668
F1 macro +0.0672
Accuracy +0.0645
Loss -0.0180
```

Notes:

```text
1. The best factor set was mainly DSL text/temporal factors over AKI,
   respiratory/pneumonia, furosemide, potassium chloride, sodium chloride,
   saline IV solution, and late GCS/BP documentation.
2. One generated safe_python factor used events(sample, source, start, end),
   which was not yet supported and became zero-activity under non-strict runtime.
   Added compatibility for events(sample, source, start, end), contains_any over
   event lists, and count_keywords over event lists.
3. Safe Python is now operational, but prompt guidance should push it toward
   compact interaction/gating factors rather than broad dense composites.
```

Follow-up robustness fix:

```text
1. A longer run exposed another natural generated-code pattern:
   for t, o in zip(sample.get("conditions", []), sample.get("conditions_offsets", [])).
   Added zip and enumerate to the safe builtin whitelist.
2. Safe Python validation/compile failures now also return 0.0 under
   MEDICAL_SAFE_PYTHON_STRICT_RUNTIME=0, matching runtime failures.
3. Safe Python sample offsets are now exposed in hours, not raw eICU minutes,
   so direct offset checks such as 0 <= o < 48 match the prompt semantics and
   the events(sample, source, start, end) helper.
```
