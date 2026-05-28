# QuantaAlpha and PyHealth Integration Notes

Date: 2026-05-20

## Repository

QuantaAlpha was cloned into:

```text
medFactoraalpha/baselines/QuantaAlpha
```

Source:

```text
https://github.com/QuantaAlpha/QuantaAlpha.git
```

Local commit:

```text
be80873 Delete .github directory
```

## Compatibility

QuantaAlpha is now installed into the current PyHealth environment as an
editable package without pulling its full dependency tree:

```bash
/home/lzp/anaconda3/envs/medfactoraalpha-pyhealth/bin/python -m pip install \
  -e medFactoraalpha/baselines/QuantaAlpha --no-deps
```

The current PyHealth environment is:

```text
conda env: medfactoraalpha-pyhealth
python: 3.12
numpy: 2.2.6
pandas: 2.3.3
torch: 2.7.1+cu118
```

QuantaAlpha's original dependency list includes:

```text
numpy>=1.24,<2.0
rdagent==0.8.0
pyqlib
```

The `numpy<2.0` pin conflicts with PyHealth's `numpy~=2.2.0`. The Qlib/RD-Agent
stack is also much heavier than we need for PyHealth model training. Therefore
the current integration strategy is:

- keep QuantaAlpha source inside the PyHealth environment;
- make QuantaAlpha's PyHealth-facing imports lightweight;
- use PyHealth's dataset/task/trainer/evaluator as the experiment framework;
- load Qlib/RD-Agent components only if we explicitly work on the original
  financial runner later.

Two compatibility patches were added in the QuantaAlpha checkout:

- `quantaalpha.log` falls back to Python logging when `rdagent` is absent.
- `quantaalpha.pipeline.evolution` lazily imports mutation/crossover/controller
  so trajectory classes can be used without loading the full LLM/runtime stack.
- package metadata now keeps default dependencies lightweight
  (`filelock`, `numpy>=1.24,<3.0`, `pandas>=1.5,<3.0`, `scikit-learn>=1.3`);
  original agent/Qlib/PyHealth extras live in `requirements/agent.txt`,
  `requirements/qlib.txt`, and `requirements/pyhealth.txt`.

After reinstalling the editable package, `pip check` reports:

```text
No broken requirements found.
```

For LLM calls, QuantaAlpha uses an ignored local file:

```text
medFactoraalpha/baselines/QuantaAlpha/.env
```

The current `.env` is intentionally not tracked by git and contains the
OpenAI-compatible endpoint configuration for the local run. The `/v1/models`
probe confirmed that the endpoint exposes `gpt-5.5`.

## What Can Be Reused

QuantaAlpha is organized as a configurable loop:

```text
AlphaAgentLoop
-> hypothesis generator
-> hypothesis-to-factor-expression converter
-> coder/parser
-> runner/evaluator
-> feedback summarizer
-> trajectory pool and evolution controller
```

The reusable pieces are:

- `quantaalpha.pipeline.loop.AlphaAgentLoop`
- `quantaalpha.pipeline.evolution.*`
- `quantaalpha.core.proposal` abstractions
- prompt and trajectory structure
- factor library idea

The pieces that should be replaced are:

- `QlibAlphaAgentScenario`
- `AlphaAgentHypothesis2FactorExpression` prompts/function library
- `QlibFactorParser`
- `QlibFactorRunner`
- Qlib/RankIC-specific feedback and metric extraction

## Required Medical Adapter

The clean adapter should provide a new setting similar to
`AlphaAgentFactorBasePropSetting`, but pointing to medical components:

```text
scen                  -> PyHealthFactorScenario
hypothesis_gen        -> medical hypothesis generator
hypothesis2experiment -> medical factor expression generator
coder                 -> medical factor parser / safe executor
runner                -> PyHealthFactorRunner
summarizer            -> PyHealthFactorFeedback
```

The runner should use the PyHealth pipeline:

```text
BaseDataset
-> BaseTask with extra symbolic_factors tensor
-> split_by_patient
-> get_dataloader
-> MultimodalRNN / RETAIN / Transformer / custom model
-> Trainer.train()
-> Trainer.evaluate()
```

For eICU LOS, the factor-enhanced task shape is:

```python
input_schema = {
    "conditions": "sequence",
    "procedures": "sequence",
    "drugs": "sequence",
    "symbolic_factors": "tensor",
}
output_schema = {"los": "multiclass"}
```

`MultimodalRNN` already supports this: sequence features go through RNN branches,
and tensor features go through a linear embedding branch before concatenation.

The first PyHealth-facing QuantaAlpha model wrapper is:

```text
medFactoraalpha/baselines/QuantaAlpha/quantaalpha/pyhealth_model.py
```

It exposes:

```python
from quantaalpha.pyhealth_model import QuantaAlphaPyHealthModel

model = QuantaAlphaPyHealthModel(dataset=sample_dataset)
```

This makes QuantaAlpha usable from PyHealth scripts in the same style as
`RNN`, `RETAIN`, `Transformer`, or `MultimodalRNN`.

## Metric Changes

QuantaAlpha currently ranks trajectories with:

```text
RankIC
```

For PyHealth tasks, the primary metric should be task-dependent:

- LOS multiclass: `f1_macro` or `balanced_accuracy`
- mortality/readmission binary: `roc_auc` or `pr_auc`
- regression: `mse` or `mae` with `min` criterion

Therefore these should become configurable:

- `StrategyTrajectory.get_primary_metric()`
- `EvolutionController._extract_metrics()`
- feedback prompts and comparison logic

Temporary aliasing `f1_macro` to `RankIC` would work, but it is fragile and
should not be the long-term design.

## Smoke Test

A PyHealth-side smoke test was added:

```text
experiments/quantaalpha_pyhealth_eicu_los_factor_smoke.py
```

It does not call the QuantaAlpha LLM loop. It mimics QuantaAlpha output with a
small deterministic factor vector:

```text
symbolic_factors: tensor[8]
```

The smoke script now instantiates:

```python
QuantaAlphaPyHealthModel(dataset=sample_dataset)
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 \
EICU_DEV=1 \
EPOCHS=1 \
BATCH_SIZE=64 \
NUM_WORKERS=4 \
DEVICE=cuda:0 \
/home/lzp/anaconda3/envs/medfactoraalpha-pyhealth/bin/python \
  experiments/quantaalpha_pyhealth_eicu_los_factor_smoke.py
```

Result:

```json
{
  "accuracy": 0.3394495412844037,
  "balanced_accuracy": 0.11861471861471862,
  "cohen_kappa": 0.07822410147991532,
  "f1_macro": 0.08451576576576578,
  "f1_weighted": 0.24099615670716587,
  "loss": 2.0221118927001953
}
```

This score is only a smoke result on dev data and one epoch. The important result
is architectural: the QuantaAlpha package can be imported from the PyHealth
environment, PyHealth accepted symbolic factor tensors, and evaluation ran
through its normal Trainer on GPU.

Output:

```text
results/pyhealth_runs/eicu_los_quantaalpha_factor_smoke_dev/final_test_scores.json
```

## LLM Factor Probe

An actual LLM-backed probe was added:

```text
experiments/quantaalpha_llm_eicu_los_factor_probe.py
```

It runs:

```text
gpt-5.5 -> factor JSON -> safe factor DSL -> eICU LOS task
-> QuantaAlphaPyHealthModel -> PyHealth Trainer
```

The factor DSL is deliberately small and deterministic. The LLM is allowed to
propose only:

```text
log_count
count_ratio
keyword_any
keyword_count
keyword_density
```

No LLM-written Python code is executed.

Command:

```bash
EICU_DEV=1 \
EPOCHS=5 \
BATCH_SIZE=64 \
NUM_WORKERS=4 \
DEVICE=cuda:0 \
/home/lzp/anaconda3/envs/medfactoraalpha-pyhealth/bin/python \
  experiments/quantaalpha_llm_eicu_los_factor_probe.py
```

Generated factor file:

```text
results/quantaalpha_medical_factors/eicu_los_gpt55_factors.json
```

The first run generated 10 factors:

```text
infection_burden
respiratory_failure_burden
shock_vasopressor_use
renal_failure_burden
cardiac_instability
neurologic_severity
metabolic_derangement
medication_burden
procedure_complexity
intervention_to_diagnosis_ratio
```

Factor activity summary:

```text
results/quantaalpha_medical_factors/eicu_los_gpt55_factor_summary_7014c155.json
```

All generated factors were non-constant on the dev split. Example nonzero rates:

```text
infection_burden: 0.275
respiratory_failure_burden: 0.269
shock_vasopressor_use: 0.322
renal_failure_burden: 0.569
procedure_complexity: 0.531
intervention_to_diagnosis_ratio: 1.000
```

Five-epoch dev result:

```json
{
  "accuracy": 0.3177570093457944,
  "balanced_accuracy": 0.2274233972026841,
  "cohen_kappa": 0.06978682862927243,
  "f1_macro": 0.21921750321750322,
  "f1_weighted": 0.2624650284463369,
  "loss": 1.7986842393875122
}
```

Output:

```text
results/pyhealth_runs/eicu_los_quantaalpha_llm_gpt55_dev_7014c155/final_test_scores.json
```

This is still a small dev probe, not a full paper-grade experiment. But it
does confirm the intended bridge works: the LLM can propose clinically plausible
symbolic factors, those factors can be materialized from eICU samples, and
PyHealth can train/evaluate with them on GPU.

## QuantaAlpha Workflow

The medical components are now wired into QuantaAlpha's native `AlphaAgentLoop`:

```text
factor_propose
-> factor_construct
-> factor_calculate
-> factor_backtest
-> feedback
```

Added modules:

```text
quantaalpha/
  pyhealth_model.py         # model wrappers
  medical/
    scenario.py             # PyHealth task/data description
    experiment.py           # symbolic factor task/experiment objects
    proposal.py             # LLM hypothesis and factor proposal
    coder.py                # safe DSL parser/validator
    runner.py               # Trainer.train/evaluate runner
    feedback.py             # LLM feedback from PyHealth metrics
    settings.py             # medical component class paths
    dsl.py                  # safe factor operations
```

Workflow entrypoint:

```text
experiments/quantaalpha_medical_workflow_eicu_los.py
```

Command used for the first full workflow smoke:

```bash
EICU_DEV=1 \
EPOCHS=1 \
BATCH_SIZE=64 \
NUM_WORKERS=4 \
DEVICE=cuda:0 \
WORKFLOW_STEPS=5 \
/home/lzp/anaconda3/envs/medfactoraalpha-pyhealth/bin/python \
  experiments/quantaalpha_medical_workflow_eicu_los.py
```

This ran all five `AlphaAgentLoop` steps and wrote:

```text
results/quantaalpha_medical_workflow/last_workflow_summary.json
results/quantaalpha_medical_workflow/workflow_factors_47be22f6.json
results/quantaalpha_medical_workflow/workflow_factor_summary_47be22f6.json
results/pyhealth_runs/eicu_los_quantaalpha_workflow_dev_47be22f6/final_test_scores.json
```

The first workflow hypothesis focused on respiratory failure and mechanical
ventilation intensity. It produced factors such as:

```text
respiratory_failure_any
ventilation_procedure_count
airway_support_density
oxygen_escalation_any
shock_vasopressor_burden
renal_failure_dialysis_any
infection_sepsis_count
procedure_condition_complexity_ratio
neurologic_critical_illness_any
```

One-epoch dev result:

```json
{
  "accuracy": 0.2818181818181818,
  "balanced_accuracy": 0.1131336405529954,
  "cohen_kappa": 0.022607130806433506,
  "f1_macro": 0.08428826075884899,
  "f1_weighted": 0.19267541727434775,
  "loss": 1.9236724972724915
}
```

The LLM feedback step correctly flagged that the respiratory-focused hypothesis
was not empirically supported in this one-epoch dev run. It noted that
`oxygen_escalation_any` was inactive and suggested broadening the next round
toward multi-organ acuity signals.

## Recommended Next Step

Run `WORKFLOW_STEPS=10` to execute two consecutive `AlphaAgentLoop` rounds in a
single process. The second round will condition on the first round's PyHealth
metrics and LLM feedback.
