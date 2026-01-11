"""Target classes for extracting optimization objectives from observations."""

from folio.targets.base import ScalarTarget
from folio.targets.builtin import (
    DerivedTarget,
    DifferenceTarget,
    DirectTarget,
    DistanceTarget,
    RatioTarget,
    SlopeTarget,
)

__all__ = [
    "ScalarTarget",
    "DirectTarget",
    "DerivedTarget",
    "DistanceTarget",
    "RatioTarget",
    "DifferenceTarget",
    "SlopeTarget",
]
