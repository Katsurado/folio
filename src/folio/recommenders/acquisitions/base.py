"""Abstract base class for acquisition functions."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class Acquisition(ABC):
    """Abstract base class for acquisition functions in Bayesian optimization.

    Acquisition functions score candidate points based on surrogate predictions,
    balancing exploration (high uncertainty) and exploitation (high predicted value).
    Higher scores indicate more promising candidates for evaluation.

    The public `evaluate` method handles input validation and delegates to the
    abstract `_compute` method, which subclasses must implement with the actual
    scoring logic.

    Notes
    -----
    Subclasses must implement `_compute` to define the acquisition scoring logic.
    The `evaluate` method performs all validation, so `_compute` can assume inputs
    are valid.

    Validation performed by `evaluate`:

    - `mean` and `std` have matching shapes
    - `std` values are non-negative
    - No NaN values in `mean` or `std`
    - No Inf values in `mean` or `std`
    - `objective` is either "maximize" or "minimize"

    Examples
    --------
    Implementing a custom acquisition function:

    >>> class MyAcquisition(Acquisition):
    ...     def _compute(
    ...         self,
    ...         X: np.ndarray,
    ...         mean: np.ndarray,
    ...         std: np.ndarray,
    ...         y_best: float,
    ...         objective: Literal["maximize", "minimize"],
    ...     ) -> np.ndarray:
    ...         # Custom scoring logic here
    ...         return mean + 2.0 * std  # UCB-like

    Using an acquisition function:

    >>> acq = MyAcquisition()
    >>> scores = acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")
    >>> best_idx = np.argmax(scores)

    Reference: Frazier (2018), A Tutorial on Bayesian Optimization.
    """

    def evaluate(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: float,
        objective: Literal["maximize", "minimize"],
    ) -> np.ndarray:
        """Score candidate points using the acquisition function.

        Validates inputs and delegates to the abstract `_compute` method.
        Returns scores where higher values indicate more promising candidates.

        Parameters
        ----------
        X : np.ndarray, shape (n_candidates, n_features)
            Candidate points to score. Each row is a candidate point in the
            input space.
        mean : np.ndarray, shape (n_candidates,)
            Predicted mean values from the surrogate model at each candidate.
        std : np.ndarray, shape (n_candidates,)
            Predicted standard deviations (uncertainty) from the surrogate
            model at each candidate. Must be non-negative.
        y_best : float
            Best observed target value so far. Used as reference for
            improvement-based acquisition functions.
        objective : {"maximize", "minimize"}
            Optimization direction. "maximize" seeks higher target values,
            "minimize" seeks lower target values.

        Returns
        -------
        np.ndarray, shape (n_candidates,)
            Acquisition scores for each candidate. Higher scores indicate
            more promising candidates for evaluation.

        Raises
        ------
        ValueError
            If validation fails:

            - mean and std shapes don't match
            - std contains negative values
            - mean or std contains NaN values
            - mean or std contains Inf values
            - objective is not "maximize" or "minimize"

        Examples
        --------
        >>> acq = ExpectedImprovement()
        >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> mean = np.array([0.5, 0.8])
        >>> std = np.array([0.1, 0.3])
        >>> scores = acq.evaluate(X, mean, std, y_best=0.6, objective="maximize")
        >>> scores.shape
        (2,)
        """
        raise NotImplementedError

    @abstractmethod
    def _compute(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: float,
        objective: Literal["maximize", "minimize"],
    ) -> np.ndarray:
        """Compute acquisition scores for validated inputs.

        This method contains the actual acquisition function logic. It is called
        by `evaluate` after all input validation has passed, so implementations
        can assume inputs are valid.

        Parameters
        ----------
        X : np.ndarray, shape (n_candidates, n_features)
            Candidate points to score (validated).
        mean : np.ndarray, shape (n_candidates,)
            Predicted mean values (validated, no NaN/Inf).
        std : np.ndarray, shape (n_candidates,)
            Predicted standard deviations (validated, non-negative, no NaN/Inf).
        y_best : float
            Best observed target value so far.
        objective : {"maximize", "minimize"}
            Optimization direction (validated).

        Returns
        -------
        np.ndarray, shape (n_candidates,)
            Acquisition scores. Higher = more promising.

        Notes
        -----
        Implementations should handle the `objective` parameter to correctly
        score candidates whether maximizing or minimizing. For improvement-based
        functions, "maximize" means improvement is (mean - y_best) while
        "minimize" means improvement is (y_best - mean).

        Examples
        --------
        A simple UCB implementation:

        >>> def _compute(self, X, mean, std, y_best, objective):
        ...     if objective == "maximize":
        ...         return mean + self.kappa * std
        ...     else:
        ...         return -mean + self.kappa * std
        """
        ...
