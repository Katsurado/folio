"""Extension modules for Folio demos."""

from extensions.custom_models import (
    NNEnsembleRecommender,
    NNEnsembleSurrogate,
    ProbabilityOfImprovement,
)

__all__ = ["NNEnsembleRecommender", "NNEnsembleSurrogate", "ProbabilityOfImprovement"]
