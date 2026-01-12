"""Surrogate models for Bayesian optimization."""

from folio.surrogates.base import Surrogate
from folio.surrogates.gp import SingleTaskGPSurrogate
from folio.surrogates.multitask_gp import MultiTaskGPSurrogate
from folio.surrogates.transforms import TaskStandardize

__all__ = [
    "Surrogate",
    "SingleTaskGPSurrogate",
    "MultiTaskGPSurrogate",
    "TaskStandardize",
]
