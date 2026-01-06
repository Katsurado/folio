"""Target classes for extracting optimization objectives from observations."""

from folio.targets.builtin import (
    DerivedTarget,
    DifferenceTarget,
    DirectTarget,
    DistanceTarget,
    RatioTarget,
    SlopeTarget,
)

__all__ = [
    "DirectTarget",
    "DerivedTarget",
    "DistanceTarget",
    "RatioTarget",
    "DifferenceTarget",
    "SlopeTarget",
]
