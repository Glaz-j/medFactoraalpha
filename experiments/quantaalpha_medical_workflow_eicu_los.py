"""
Run the QuantaAlpha workflow on PyHealth eICU length-of-stay.

This script uses QuantaAlpha's AlphaAgentLoop:

    factor_propose -> factor_construct -> factor_calculate
    -> factor_backtest -> feedback
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUANTAALPHA_ROOT = PROJECT_ROOT / "baselines" / "QuantaAlpha"


def main() -> None:
    env_path = Path(os.environ.get("QUANTAALPHA_ENV", QUANTAALPHA_ROOT / ".env"))
    explicit_env = {
        key: os.environ[key]
        for key in ("MAX_RETRY", "RETRY_WAIT_SECONDS")
        if key in os.environ
    }
    if env_path.exists():
        load_dotenv(env_path, override=True)
    os.environ.update(explicit_env)

    os.environ.setdefault("CHAT_MODEL", os.environ.get("LLM_MODEL", "gpt-5.5"))
    os.environ.setdefault("REASONING_MODEL", os.environ.get("LLM_MODEL", "gpt-5.5"))
    os.environ.setdefault("CHAT_STREAM", "False")
    os.environ.setdefault("LOG_LLM_CHAT_CONTENT", "False")
    os.environ.setdefault("MAX_RETRY", "2")
    os.environ.setdefault("RETRY_WAIT_SECONDS", "2")
    os.environ.setdefault("EICU_DEV", "1")
    os.environ.setdefault("EPOCHS", os.environ.get("MEDICAL_EPOCHS", "1"))
    os.environ.setdefault("BATCH_SIZE", "64")
    os.environ.setdefault("NUM_WORKERS", "4")
    os.environ.setdefault("DEVICE", "cuda:0")

    from quantaalpha.log import logger
    from quantaalpha.pipeline.loop import AlphaAgentLoop
    from quantaalpha.medical.settings import (
        MEDICAL_ALPHA_TABULAR_FACTOR_SETTING,
        MEDICAL_PYHEALTH_FACTOR_SETTING,
        MEDICAL_TABULAR_FACTOR_SETTING,
    )

    output_root = Path(
        os.environ.get(
            "MEDICAL_OUTPUT_ROOT",
            PROJECT_ROOT / "results" / "quantaalpha_medical_workflow",
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    logger.set_trace_path(log_root)
    os.environ.setdefault("LLM_USAGE_LOG_PATH", str(log_root / "llm_usage.jsonl"))

    direction = os.environ.get(
        "MEDICAL_DIRECTION",
        "Mine clinically meaningful symbolic factors for eICU ICU length-of-stay prediction.",
    )
    step_n = int(os.environ.get("WORKFLOW_STEPS", "5"))
    evaluator = os.environ.get("MEDICAL_EVALUATOR", "pyhealth").strip().lower()
    workflow_flavor = os.environ.get("MEDICAL_WORKFLOW_FLAVOR", "alpha").strip().lower()
    if evaluator in {"tabular", "gbdt", "linear", "qlib_like"}:
        if workflow_flavor in {"alpha", "full", "quantaalpha"}:
            setting = MEDICAL_ALPHA_TABULAR_FACTOR_SETTING
        elif workflow_flavor in {"light", "lightweight", "simple"}:
            setting = MEDICAL_TABULAR_FACTOR_SETTING
        else:
            raise ValueError(
                "MEDICAL_WORKFLOW_FLAVOR must be one of: alpha, full, "
                "quantaalpha, light, lightweight, simple"
            )
    elif evaluator in {"pyhealth", "gru", "rnn"}:
        setting = MEDICAL_PYHEALTH_FACTOR_SETTING
    else:
        raise ValueError(
            "MEDICAL_EVALUATOR must be one of: pyhealth, gru, rnn, "
            "tabular, gbdt, linear, qlib_like"
        )

    loop = AlphaAgentLoop(
        setting,
        potential_direction=direction,
        stop_event=None,
        use_local=True,
        quality_gate_config={
            "consistency_enabled": os.environ.get(
                "MEDICAL_CONSISTENCY_CHECK",
                "0",
            ).strip().lower()
            in {"1", "true", "yes", "y", "on"}
        },
    )
    loop.user_initial_direction = direction
    loop.run(step_n=step_n, stop_event=None)

    exp = loop._last_experiment
    feedback = loop._last_feedback
    hypothesis = loop._last_hypothesis

    summary = {
        "direction": direction,
        "evaluator": evaluator,
        "workflow_flavor": workflow_flavor,
        "steps": step_n,
        "hypothesis": str(hypothesis) if hypothesis is not None else None,
        "feedback": str(feedback) if feedback is not None else None,
        "result": getattr(exp, "result", None),
        "factors": [task.to_factor_dict() for task in exp.sub_tasks] if exp else [],
    }
    summary_path = output_root / "last_workflow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Workflow summary written to: {summary_path}")
    if summary.get("result"):
        print(json.dumps(summary["result"].get("scores", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
