"""
PyHealth integration models for QuantaAlpha.

This module keeps the PyHealth-facing surface lightweight: it does not import
Qlib or RD-Agent, and can therefore run inside the PyHealth experiment
environment. The expected data contract is that a PyHealth task has already
materialized QuantaAlpha-style symbolic factors as tensor features, for example
``symbolic_factors``.
"""

from __future__ import annotations

from typing import Iterable, Type

import torch

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel, MultimodalRNN


class QuantaAlphaPyHealthModel(BaseModel):
    """A PyHealth model wrapper for QuantaAlpha symbolic factors.

    The wrapper delegates clinical sequence modeling to a PyHealth backbone
    while making the QuantaAlpha factor tensor part of the model contract. This
    lets QuantaAlpha be used like a built-in PyHealth model in Trainer scripts.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        factor_keys: Iterable[str] | None = None,
        backbone_cls: Type[BaseModel] = MultimodalRNN,
        **backbone_kwargs,
    ):
        super().__init__(dataset=dataset)
        self.factor_keys = tuple(factor_keys or self._infer_factor_keys(dataset))
        if not self.factor_keys:
            raise ValueError(
                "QuantaAlphaPyHealthModel expects at least one tensor factor "
                "feature, e.g. `symbolic_factors`, in dataset.input_schema."
            )

        missing = [key for key in self.factor_keys if key not in dataset.input_schema]
        if missing:
            raise ValueError(f"Factor keys missing from dataset.input_schema: {missing}")

        self.backbone = backbone_cls(dataset=dataset, **backbone_kwargs)
        self.mode = getattr(self.backbone, "mode", self.mode)

    @staticmethod
    def _infer_factor_keys(dataset: SampleDataset) -> list[str]:
        keys = []
        for key, schema in dataset.input_schema.items():
            if isinstance(schema, str):
                schema_name = schema
            elif hasattr(schema, "__name__"):
                schema_name = schema.__name__
            else:
                schema_name = schema.__class__.__name__
            schema_name = schema_name.lower()
            if key.endswith("factors") or key == "symbolic_factors":
                keys.append(key)
            elif schema_name == "tensor":
                keys.append(key)
        return keys

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        return self.backbone(**kwargs)
