"""Extension modules for Folio demos."""

from extensions.custom_models import (
    NNEnsembleRecommender,
    NNEnsembleSurrogate,
    ProbabilityOfImprovement,
)
from extensions.r2_target import (
    R2Target,
    SternVolmerKsvTarget,
    SternVolmerR2Target,
)

__all__ = [
    "NNEnsembleRecommender",
    "NNEnsembleSurrogate",
    "ProbabilityOfImprovement",
    "R2Target",
    "SternVolmerKsvTarget",
    "SternVolmerR2Target",
]
