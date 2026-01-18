"""Experiment initialization strategies.

This module provides initializers that suggest initial experiments before
any observations are available. Unlike recommenders that use surrogate
models, initializers use other strategies like LLM-based literature
search or space-filling designs.
"""

from folio.recommenders.initializer.base import Initializer
from folio.recommenders.initializer.llm import (
    LLMBackend,
    LLMInitializer,
    OpenAIBackend,
)

__all__ = [
    "Initializer",
    "LLMBackend",
    "LLMInitializer",
    "OpenAIBackend",
]
