"""
Evolution module for AlphaAgent.

The trajectory classes are lightweight and useful outside the original Qlib
workflow. Mutation/crossover/controller depend on LLM/runtime components, so
they are imported lazily to keep PyHealth-side imports small.
"""

from .trajectory import RoundPhase, StrategyTrajectory, TrajectoryPool

__all__ = [
    "StrategyTrajectory",
    "TrajectoryPool",
    "RoundPhase",
    "MutationOperator",
    "CrossoverOperator",
    "EvolutionController",
    "EvolutionConfig",
]


def __getattr__(name):
    if name == "MutationOperator":
        from .mutation import MutationOperator

        return MutationOperator
    if name == "CrossoverOperator":
        from .crossover import CrossoverOperator

        return CrossoverOperator
    if name in {"EvolutionController", "EvolutionConfig"}:
        from .controller import EvolutionConfig, EvolutionController

        return {"EvolutionController": EvolutionController, "EvolutionConfig": EvolutionConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
