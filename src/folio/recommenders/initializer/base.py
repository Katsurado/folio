"""Base classes for experiment initialization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from folio.core.project import Project


class Initializer(ABC):
    """Abstract base class for experiment initializers.

    Initializers suggest initial experiments before any observations are
    available. Unlike recommenders that use surrogate models and acquisition
    functions, initializers use other strategies (e.g., LLM-based literature
    search, Latin hypercube sampling, factorial designs) to propose a
    diverse set of starting points.

    Notes
    -----
    Subclasses must implement the `suggest` method to provide their specific
    strategy for generating initial experiments.

    Examples
    --------
    >>> class LatinHypercubeInitializer(Initializer):
    ...     def suggest(self, project, n, description=None, existing=None):
    ...         # Generate Latin hypercube samples
    ...         ...
    """

    @abstractmethod
    def suggest(
        self,
        project: Project,
        n: int,
        description: str | None = None,
        existing: list[dict] | None = None,
    ) -> list[dict]:
        """Suggest initial experiments for the optimization problem.

        Parameters
        ----------
        project : Project
            The project defining inputs, outputs, and optimization targets.
        n : int
            Number of initial experiments to suggest.
        description : str | None, optional
            Natural language description providing additional context for
            the initializer (e.g., reaction type, constraints, prior knowledge).
            Some initializers (e.g., LLM-based) use this for better suggestions.
        existing : list[dict] | None, optional
            Previously run or planned experiments to avoid duplicating.
            Each dict maps parameter names to values.

        Returns
        -------
        list[dict]
            List of n suggested experiments. Each experiment is a dictionary
            mapping parameter names to suggested values. Continuous parameters
            have float values within bounds; categorical parameters have string
            values from the allowed choices.

        Notes
        -----
        - Implementations should ensure diversity across the parameter space
        - Returned values must be within specified bounds/levels
        - If existing experiments are provided, suggestions should complement them
        """
        ...
