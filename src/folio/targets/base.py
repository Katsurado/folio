"""Abstract base class for optimization targets."""

from abc import ABC, abstractmethod
from typing import Literal

from folio.core.observation import Observation


class ScalarTarget(ABC):
    """Abstract base class for extracting a scalar optimization target."""

    objective: Literal["maximize", "minimize"]

    @abstractmethod
    def compute(self, obs: Observation) -> float | None:
        """Extract scalar target value from an observation."""
