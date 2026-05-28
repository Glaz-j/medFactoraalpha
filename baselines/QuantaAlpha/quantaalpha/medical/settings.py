"""Settings for the medical QuantaAlpha workflow."""

from __future__ import annotations

from quantaalpha.pipeline.settings import BasePropSetting


class MedicalPyHealthFactorSetting(BasePropSetting):
    scen: str = "quantaalpha.medical.scenario.PyHealthEICULOSScenario"
    hypothesis_gen: str = "quantaalpha.medical.proposal.MedicalHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.medical.proposal.MedicalHypothesis2FactorExpression"
    coder: str = "quantaalpha.medical.costeer.MedicalDSLCoSTEER"
    runner: str = "quantaalpha.medical.runner.PyHealthFactorRunner"
    summarizer: str = "quantaalpha.medical.feedback.MedicalPyHealthFeedback"
    evolving_n: int = 1


class MedicalTabularFactorSetting(MedicalPyHealthFactorSetting):
    runner: str = "quantaalpha.medical.tabular_runner.MedicalTabularFactorRunner"


class MedicalAlphaTabularFactorSetting(MedicalTabularFactorSetting):
    """AlphaAgent-style medical workflow with richer LLM proposal/refinement."""

    hypothesis_gen: str = "quantaalpha.medical.alpha_proposal.MedicalAlphaHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.medical.alpha_proposal.MedicalAlphaHypothesis2FactorExpression"
    summarizer: str = "quantaalpha.medical.feedback.MedicalPyHealthFeedback"
    evolving_n: int = 5


MEDICAL_PYHEALTH_FACTOR_SETTING = MedicalPyHealthFactorSetting()
MEDICAL_TABULAR_FACTOR_SETTING = MedicalTabularFactorSetting()
MEDICAL_ALPHA_TABULAR_FACTOR_SETTING = MedicalAlphaTabularFactorSetting()
