# Baseline Framework Selection

Date: 2026-05-20

## Recommendation

Use **PyHealth** as the main reproducible baseline framework outside REACT-LLM.

Repository:

```text
https://github.com/sunlabuiuc/PyHealth
```

Why:

- Strong community footprint: about 1.6k GitHub stars and 700+ forks as of 2026-05-20.
- Actively maintained.
- Healthcare-specific, not a one-off paper repo.
- Supports MIMIC-III, MIMIC-IV, eICU, OMOP, EHRShot, and custom EHR datasets.
- Supports common clinical prediction tasks: mortality, readmission, length-of-stay forecasting, hospitalization prediction, etc.
- Provides many reusable EHR models: RNN, LSTM, GRU, Transformer, RETAIN, AdaCare, ConCare, StageNet, GRASP, Deepr, TCN, SparcNet, Dr. Agent, and others.
- Has a KDD 2023 toolkit paper and a 2026 PyHealth 2.0 arXiv update, making it easier to cite as a reproducible framework.

## Role in medFactoraalpha

PyHealth should be our **strong framework baseline**, while REACT-LLM should be our **closest task/method baseline**.

Planned use:

```text
MIMIC/eICU data
-> PyHealth task pipeline
-> standard EHR models / ML baselines
-> compare against medFactoraalpha factors
```

For medFactoraalpha, we can use PyHealth in two ways:

1. Run standard PyHealth EHR sequence models as strong baselines.
2. Inject LLM-generated symbolic factors as additional features or as a parallel feature branch, then test whether they improve predictive performance and interpretability.

## Candidate Comparison

| Candidate | Stars/Forks checked on 2026-05-20 | Strength | Limitation | Decision |
| --- | ---: | --- | --- | --- |
| PyHealth | ~1587 / 773 | Broad healthcare ML toolkit; MIMIC/eICU support; many models/tasks; active | Need adapter work for our factor features | Main baseline framework |
| EHRSHOT benchmark | ~218 / 29 | Strong benchmark for EHR foundation model evaluation | Less directly aligned with MIMIC/eICU factor-mining experiments | Secondary reference |
| YerevaNN/mimic3-benchmarks | ~881 / 344 | Classic MIMIC-III benchmark suite | Older, mainly MIMIC-III and fixed tasks | Useful classic reference, not main |
| MIMIC-Extract | ~476 / 131 | Strong MIMIC-III extraction/preprocessing pipeline | Not a full model/baseline framework | Data-processing reference |
| REACT-LLM | parent repo ~3 stars; author fork ~1 star | Closest to our LLM + causal/feature prior framing; AAAI 2026 | Too new, low community footprint | Closest method baseline, not sole framework |

## First Implementation Plan

1. Install/test PyHealth in an isolated environment.
2. Reproduce a simple mortality/readmission/LOS task on a small MIMIC subset.
3. Add our derived/static data as a custom PyHealth dataset or build a bridge from our parquet files.
4. Run at least three baselines:
   - Logistic/LightGBM or XGBoost on tabular features.
   - RETAIN/Transformer/GRASP or ConCare from PyHealth.
   - medFactoraalpha symbolic-factor-enhanced model.
5. Keep REACT-LLM as the LLM prior/feature-selection baseline and compare its feature-list style prior against our formula-factor prior.

