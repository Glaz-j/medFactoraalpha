"""Run full medical QuantaAlpha evolution on eICU readmission.

This entrypoint uses the original QuantaAlpha EvolutionController shape:
original trajectories -> mutation -> crossover, while the inner loop remains
the medical AlphaAgent workflow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from quantaalpha_medical_workflow_eicu_los import PROJECT_ROOT, QUANTAALPHA_ROOT


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _direction_suffixes(task_name: str, count: int) -> list[str]:
    from quantaalpha.medical.evolution import default_medical_direction_suffixes

    raw = os.environ.get("MEDICAL_EVOLUTION_DIRECTIONS", "").strip()
    if raw:
        directions = [item.strip() for item in raw.split("||") if item.strip()]
    else:
        directions = default_medical_direction_suffixes(task_name)
    while len(directions) < count:
        directions.append(f"Original direction {len(directions)}: explore an orthogonal clinically interpretable factor family.")
    return directions[:count]


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

    os.environ.setdefault("MEDICAL_EICU_TASK", "readmission")
    os.environ.setdefault("MEDICAL_EVALUATOR", "tabular")
    os.environ.setdefault("MEDICAL_WORKFLOW_FLAVOR", "alpha")
    os.environ.setdefault("MEDICAL_TABULAR_DIRECT_SOURCE", "1")
    os.environ.setdefault("MEDICAL_SOURCE_PROFILE", "pyhealth_standard")
    os.environ.setdefault("MEDICAL_TEMPORAL_SOURCE", "1")
    os.environ.setdefault("MEDICAL_NUMERIC_TEMPORAL_SOURCE", "0")
    os.environ.setdefault("MEDICAL_DSL_COSTEER", "1")
    os.environ.setdefault("CHAT_MODEL", os.environ.get("LLM_MODEL", "gpt-5.5"))
    os.environ.setdefault("REASONING_MODEL", os.environ.get("LLM_MODEL", "gpt-5.5"))
    os.environ.setdefault("CHAT_STREAM", "False")
    os.environ.setdefault("LOG_LLM_CHAT_CONTENT", "False")
    os.environ.setdefault("MAX_RETRY", "2")
    os.environ.setdefault("RETRY_WAIT_SECONDS", "2")
    os.environ.setdefault("EICU_DEV", "1")
    os.environ.setdefault("BATCH_SIZE", "128")
    os.environ.setdefault("NUM_WORKERS", "4")
    os.environ.setdefault("MEDICAL_OBSERVATION_END_HOURS", "48")

    from quantaalpha.log import logger
    from quantaalpha.medical.evolution import EvolutionConfig, MedicalEvolutionController, RoundPhase
    from quantaalpha.medical.settings import MEDICAL_ALPHA_TABULAR_FACTOR_SETTING
    from quantaalpha.medical.task_config import get_task_config
    from quantaalpha.pipeline.loop import AlphaAgentLoop

    task_config = get_task_config()
    base_output_root = Path(
        os.environ.get(
            "MEDICAL_OUTPUT_ROOT",
            PROJECT_ROOT / "results" / "quantaalpha_medical_evolution_readmission",
        )
    )
    base_output_root.mkdir(parents=True, exist_ok=True)

    config = EvolutionConfig(
        num_directions=int(os.environ.get("MEDICAL_EVOLUTION_DIRECTIONS_N", "3")),
        max_rounds=int(os.environ.get("MEDICAL_EVOLUTION_MAX_ROUNDS", "3")),
        mutation_enabled=_bool_env("MEDICAL_EVOLUTION_MUTATION", "1"),
        crossover_enabled=_bool_env("MEDICAL_EVOLUTION_CROSSOVER", "1"),
        crossover_size=int(os.environ.get("MEDICAL_EVOLUTION_CROSSOVER_SIZE", "2")),
        crossover_n=int(os.environ.get("MEDICAL_EVOLUTION_CROSSOVER_N", "2")),
        parent_selection_strategy=os.environ.get("MEDICAL_EVOLUTION_PARENT_SELECTION", "best"),
        pool_save_path=str(base_output_root / "trajectory_pool.json"),
        fresh_start=_bool_env("MEDICAL_EVOLUTION_FRESH_START", "1"),
    )
    controller = MedicalEvolutionController(config)
    direction = os.environ.get(
        "MEDICAL_DIRECTION",
        (
            "Mine clinically meaningful symbolic factors for eICU readmission "
            "prediction using only the allowed observation window. Optimize "
            "AUROC/PRAUC/F1/loss with baseline+factor tabular evaluation."
        ),
    )
    direction_suffixes = _direction_suffixes(task_config.name, config.num_directions)
    step_n = int(os.environ.get("MEDICAL_EVOLUTION_WORKFLOW_STEPS", os.environ.get("WORKFLOW_STEPS", "25")))
    max_tasks = int(os.environ.get("MEDICAL_EVOLUTION_MAX_TASKS", "999"))
    completed = []

    task_counter = 0
    while task_counter < max_tasks:
        task = controller.get_next_task()
        if task is None:
            break
        phase = task["phase"]
        phase_value = phase.value if hasattr(phase, "value") else str(phase)
        suffix = task.get("strategy_suffix", "")
        if phase == RoundPhase.ORIGINAL:
            suffix = "\n\n" + direction_suffixes[task["direction_id"]]
        task_output_root = (
            base_output_root
            / f"round_{task['round_idx']:02d}_{phase_value}_dir_{task['direction_id']:02d}"
        )
        task_output_root.mkdir(parents=True, exist_ok=True)
        log_root = task_output_root / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        logger.set_trace_path(log_root)

        previous_env = {
            "MEDICAL_OUTPUT_ROOT": os.environ.get("MEDICAL_OUTPUT_ROOT"),
            "LLM_USAGE_LOG_PATH": os.environ.get("LLM_USAGE_LOG_PATH"),
        }
        os.environ["MEDICAL_OUTPUT_ROOT"] = str(task_output_root)
        os.environ["LLM_USAGE_LOG_PATH"] = str(log_root / "llm_usage.jsonl")
        try:
            loop = AlphaAgentLoop(
                MEDICAL_ALPHA_TABULAR_FACTOR_SETTING,
                potential_direction=direction,
                stop_event=None,
                use_local=True,
                strategy_suffix=suffix,
                evolution_phase=phase_value,
                trajectory_id="",
                parent_trajectory_ids=[p.trajectory_id for p in task.get("parent_trajectories", [])],
                direction_id=task["direction_id"],
                round_idx=task["round_idx"],
                quality_gate_config={
                    "consistency_enabled": _bool_env("MEDICAL_CONSISTENCY_CHECK", "0")
                },
            )
            loop.user_initial_direction = direction
            loop.run(step_n=step_n, stop_event=None)
            trajectory = controller.create_trajectory_from_loop_result(
                task,
                loop._last_hypothesis,
                loop._last_experiment,
                loop._last_feedback,
            )
            controller.report_task_complete(task, trajectory)
            completed.append(trajectory.to_dict())
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        task_counter += 1

    best = [traj.to_dict() for traj in controller.get_best_trajectories(top_n=10)]
    summary = {
        "task": task_config.name,
        "config": {
            "num_directions": config.num_directions,
            "max_rounds": config.max_rounds,
            "mutation_enabled": config.mutation_enabled,
            "crossover_enabled": config.crossover_enabled,
            "crossover_size": config.crossover_size,
            "crossover_n": config.crossover_n,
            "workflow_steps_per_task": step_n,
        },
        "completed_count": len(completed),
        "controller_state": controller.get_current_state(),
        "best_trajectories": best,
    }
    summary_path = base_output_root / "medical_evolution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Medical evolution summary written to: {summary_path}")
    if best:
        print(json.dumps(best[0].get("backtest_metrics", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
