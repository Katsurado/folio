"""Surrogate models for Bayesian optimization."""

from folio.surrogates.base import Surrogate
from folio.surrogates.gp import GPSurrogate

__all__ = ["Surrogate", "GPSurrogate"]
