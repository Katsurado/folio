"""Abstract base class for optimization targets."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from folio.core.observation import Observation


class ScalarTarget(ABC):
    """Abstract base class for extracting a scalar optimization target.

    Targets transform observation outputs into a single scalar value for
    optimization. Subclasses implement different extraction strategies
    (direct value, ratio, difference, etc.).

    Attributes
    ----------
    objective : {"maximize", "minimize"}
        Optimization direction. Set by subclasses during initialization.

    Notes
    -----
    Subclasses must:
    - Set the `objective` attribute in __init__
    - Implement the `compute` method

    Examples
    --------
    Subclass implementation pattern:

    >>> class MyTarget(ScalarTarget):
    ...     def __init__(self, output_name: str, objective: str):
    ...         self.output_name = output_name
    ...         self.objective = objective
    ...
    ...     def compute(self, obs: "Observation") -> float | None:
    ...         return obs.outputs.get(self.output_name)
    """

    objective: Literal["maximize", "minimize"]

    @abstractmethod
    def compute(self, obs: "Observation") -> float | None:
        """Extract scalar target value from an observation.

        Parameters
        ----------
        obs : Observation
            Observation to extract target value from.

        Returns
        -------
        float | None
            The computed target value, or None if the value cannot be computed
            (e.g., missing outputs, division by zero, or other invalid states).
        """
