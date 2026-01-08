"""Surrogate models for Bayesian optimization."""

from folio.surrogates.base import Surrogate
from folio.surrogates.gp import SingleTaskGPSurrogate

__all__ = ["Surrogate", "SingleTaskGPSurrogate"]
