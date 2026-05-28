# Clinical Prognostic Prediction Papers

Collected on 2026-05-18. Focus: recent clinical prognostic prediction, EHR/ICU LLMs, medical agents, and nearby LLM-guided feature/symbolic discovery methods.

## Suggested Reading Order

1. `2026_AAAI_REACT-LLM_clinical_prognostic_tasks.pdf` - closest top-conference anchor for LLMs on clinical prognostic tasks.
2. `2025_ICLR_Context_Clues_long_context_EHR_clinical_prediction.pdf` - long-context EHR prediction framing, useful for task setup and baselines.
3. `2026_AAAI_Forecasting_clinical_risk_from_textual_time_series.pdf` - risk prediction from textual time series; good task-definition reference.
4. `2024_NeurIPS_MIMIC-Instr_instruction_tuning_EHR_LLM.pdf` - MIMIC-based instruction benchmark/reference.
5. `2024_EMNLP_EHRAgent_code_agent_EHR_tabular_reasoning.pdf` and `2025_arXiv_EMR-AGENT_cohort_feature_extraction.pdf` - useful if we make an LLM/agent generate cohorts, variables, or search spaces.
6. `2025_AAAI_ELLM-FT_evolutionary_LLM_automated_feature_transformation.pdf`, `2024_NeurIPS_OCTree_LLM_guided_decision_tree_feature_generation.pdf`, and `2026_ICLR_SR-Scientist_agentic_equation_discovery.pdf` - method anchors for LLM-guided feature/formula search.

## Clinical Prediction / EHR LLM Papers

| File | Venue | Why it is here | Source |
| --- | --- | --- | --- |
| `2026_AAAI_REACT-LLM_clinical_prognostic_tasks.pdf` | AAAI 2026 | LLM for clinical prognostic tasks; direct framing anchor. | https://ojs.aaai.org/index.php/AAAI/article/view/39839 |
| `2026_AAAI_EAG-RL_EHR_reasoning_expert_attention_guidance.pdf` | AAAI 2026 | EHR reasoning with expert attention guidance; useful for medical-prior design. | https://ojs.aaai.org/index.php/AAAI/article/view/40325 |
| `2026_AAAI_Forecasting_clinical_risk_from_textual_time_series.pdf` | AAAI 2026 | Forecasts clinical risk from time-series text; close to prognostic prediction. | https://ojs.aaai.org/index.php/AAAI/article/view/41255 |
| `2025_AAAI_DearLLM_LLM_deduced_feature_correlations.pdf` | AAAI 2025 | Uses LLM-deduced feature correlations for medical time-series classification. | https://ojs.aaai.org/index.php/AAAI/article/view/32079 |
| `2025_ICLR_Context_Clues_long_context_EHR_clinical_prediction.pdf` | ICLR 2025 | Long-context EHR clinical prediction; strong benchmark/baseline reference. | https://openreview.net/forum?id=zg3ec1TdAP |
| `2024_NeurIPS_MIMIC-Instr_instruction_tuning_EHR_LLM.pdf` | NeurIPS 2024 Datasets & Benchmarks | MIMIC instruction dataset for EHR LLM evaluation. | https://proceedings.neurips.cc/paper_files/paper/2024/hash/62986e0a78780fe5f17b495aeded5bab-Abstract-Datasets_and_Benchmarks_Track.html |
| `2024_NeurIPS_EHRNoteQA_real_world_clinical_practice_EHR_QA.pdf` | NeurIPS 2024 Datasets & Benchmarks | Real-world EHR note QA benchmark; adjacent to clinical LLM evaluation. | https://proceedings.neurips.cc/paper_files/paper/2024/hash/e15c4afff22f12c4986c1fcb4e941e03-Abstract-Datasets_and_Benchmarks_Track.html |
| `2024_arXiv_ClinicalBench_LLM_vs_ML_clinical_prediction.pdf` | arXiv 2024 | Compares LLMs and traditional ML on clinical prediction tasks. | https://arxiv.org/abs/2411.06469 |
| `2024_arXiv_Prompting_LLMs_zero_shot_structured_longitudinal_EHR_prediction.pdf` | arXiv 2024 | Zero-shot prompting LLMs for structured longitudinal EHR prediction. | https://arxiv.org/abs/2402.01713 |
| `2024_PLOS_Digital_Health_CPLLM_clinical_prediction_with_LLMs.pdf` | PLOS Digital Health 2024 | Clinical prediction with LLMs; journal reference. | https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000680 |
| `2025_arXiv_LLMs_are_powerful_EHR_encoders.pdf` | arXiv 2025 | LLMs as EHR encoders; relevant representation baseline. | https://arxiv.org/abs/2502.17403 |
| `2024_npjDM_Probabilistic_medical_predictions_of_LLMs.pdf` | npj Digital Medicine 2024 | Evaluates probabilistic medical predictions by LLMs. | https://www.nature.com/articles/s41746-024-01366-4 |
| `2024_npjDM_Shared_EHR_foundation_model_multi_center_adaptability.pdf` | npj Digital Medicine 2024 | Multi-center EHR foundation model adaptability. | https://www.nature.com/articles/s41746-024-01166-w |
| `2024_npjDM_Zero_shot_health_trajectory_prediction_using_transformer.pdf` | npj Digital Medicine 2024 | Health trajectory prediction; useful as clinical time-series forecast framing. | https://www.nature.com/articles/s41746-024-01235-0 |
| `2025_npjDM_LLMs_forecast_patient_health_trajectories_digital_twins.pdf` | npj Digital Medicine 2025 | LLMs forecasting patient health trajectories / digital twins. | https://www.nature.com/articles/s41746-025-02004-3 |
| `2025_npjDM_Evaluating_LLM_workflows_triage_referral_diagnosis.pdf` | npj Digital Medicine 2025 | Evaluates LLM workflows for clinical decisions; useful for workflow design. | https://www.nature.com/articles/s41746-025-01684-1 |
| `2025_npjDM_MedS-Bench_versatile_medical_LLMs.pdf` | npj Digital Medicine 2025 | Medical LLM benchmark; broad but useful for evaluation language. | https://www.nature.com/articles/s41746-024-01390-4 |

## Medical Agent / ICU Decision Papers

| File | Venue | Why it is here | Source |
| --- | --- | --- | --- |
| `2026_AAAI_ICU_ventilator_doctor_agents_RL_benchmark.pdf` | AAAI 2026 | ICU doctor-agent / ventilator-control benchmark; agent + ICU anchor. | https://ojs.aaai.org/index.php/AAAI/article/view/39081 |
| `2026_AAAI_Delphi_neuro_symbolic_safe_interpretable_treatment_recommendation.pdf` | AAAI 2026 | Neuro-symbolic safe/interpretable treatment recommendation. | https://ojs.aaai.org/index.php/AAAI/article/view/39016 |
| `2024_EMNLP_EHRAgent_code_agent_EHR_tabular_reasoning.pdf` | EMNLP 2024 | Code agent for complex tabular EHR reasoning. | https://aclanthology.org/2024.emnlp-main.1245/ |
| `2025_arXiv_MedAgentBench_virtual_EHR_environment_medical_agents.pdf` | arXiv 2025 | Virtual EHR environment for medical agents. | https://arxiv.org/abs/2501.14654 |
| `2024_arXiv_AgentClinic_multimodal_agent_benchmark.pdf` | arXiv 2024 | Agent benchmark for clinical settings; not prognostic-only, but good agent reference. | https://arxiv.org/abs/2405.07960 |
| `2026_npjDM_Benchmarking_LLM_agent_systems_clinical_decision_tasks.pdf` | npj Digital Medicine 2026 | Journal benchmark of LLM-agent systems on clinical decision tasks. | https://www.nature.com/articles/s41746-026-02443-6 |
| `2025_arXiv_FHIR-AgentBench_interoperable_EHR_agent_QA.pdf` | arXiv 2025 | FHIR/EHR agent QA benchmark; relevant to interoperable EHR agents. | https://arxiv.org/abs/2509.19319 |
| `2026_arXiv_PhysicianBench_real_world_EHR_agents.pdf` | arXiv 2026 | Real-world EHR agent benchmark. | https://arxiv.org/abs/2605.02240 |
| `2025_arXiv_EMR-AGENT_cohort_feature_extraction.pdf` | arXiv 2025 | Agent for cohort and feature extraction from EMR databases; close to our data-building layer. | https://arxiv.org/abs/2510.00549 |

## Feature / Formula Search Method Anchors

| File | Venue | Why it is here | Source |
| --- | --- | --- | --- |
| `2023_NeurIPS_CAAFE_LLM_context_aware_automated_feature_engineering.pdf` | NeurIPS 2023 | LLM-based automated feature engineering baseline. | https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c2df4c35cdbee764ebb9e9d0acd5197-Abstract-Conference.html |
| `2024_NeurIPS_OCTree_LLM_guided_decision_tree_feature_generation.pdf` | NeurIPS 2024 | LLM-guided feature generation with model feedback. | https://papers.nips.cc/paper_files/paper/2024/hash/a7ebe2e8d8cfd2fcec6cd77f9e6fd34d-Abstract-Conference.html |
| `2023_NeurIPS_Reinforcement_enhanced_autoregressive_feature_transformation.pdf` | NeurIPS 2023 | RL-style feature transformation search; useful baseline framing. | https://proceedings.neurips.cc/paper_files/paper/2023/hash/8797d13e5998acfab387d4bf0a5b9b00-Abstract-Conference.html |
| `2025_AAAI_ELLM-FT_evolutionary_LLM_automated_feature_transformation.pdf` | AAAI 2025 | Evolutionary LLM for automated feature transformation; very relevant method anchor. | https://ojs.aaai.org/index.php/AAAI/article/view/33851 |
| `2025_ICLR_LLM-SR_scientific_equation_discovery.pdf` | ICLR 2025 | LLM-guided symbolic/equation discovery; important for formula search. | https://openreview.net/forum?id=m2nmp8P5in |
| `2025_ICML_LLM-SRBench_equation_discovery_benchmark.pdf` | ICML 2025 / arXiv | Benchmark for LLM-based symbolic regression. | https://arxiv.org/abs/2504.10415 |
| `2026_ICLR_SR-Scientist_agentic_equation_discovery.pdf` | ICLR 2026 | Agentic scientific equation discovery; directly relevant to LLM-agent formula search. | https://openreview.net/forum?id=KBN6oUx5uL |
| `2025_arXiv_DrSR_dual_reasoning_symbolic_regression.pdf` | arXiv 2025 | Dual-reasoning LLM symbolic regression. | https://arxiv.org/abs/2506.04282 |

## Notes

- The strongest clinical-task anchors for our project are REACT-LLM, Context Clues, Forecasting Clinical Risk from Textual Time Series, MIMIC-Instr, and the npj health-trajectory/foundation-model papers.
- The strongest method anchors for our proposed "LLM-guided interpretable medical factor search" are ELLM-FT, OCTree, LLM-SR, and SR-Scientist.
- The strongest agent/data-layer anchors are EHRAgent, EMR-AGENT, MedAgentBench, and PhysicianBench.
