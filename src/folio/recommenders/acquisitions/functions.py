"""Acquisition function implementations for Bayesian optimization."""

from typing import Literal

import numpy as np

from folio.recommenders.acquisitions.base import Acquisition


class ExpectedImprovement(Acquisition):
    """Expected Improvement acquisition function.

    EI balances exploration and exploitation by computing the expected value
    of improvement over the current best observation. Points with high predicted
    mean (exploitation) or high uncertainty (exploration) receive high scores.

    Parameters
    ----------
    xi : float, default=0.01
        Exploration-exploitation trade-off parameter. Higher values encourage
        more exploration by requiring larger expected improvements. A value of
        0 gives pure EI; positive values add a "margin" that must be exceeded.

    Attributes
    ----------
    xi : float
        The exploration parameter.

    Notes
    -----
    For maximization, the Expected Improvement is computed as:

        Z = (μ - y_best - ξ) / σ
        EI = (μ - y_best - ξ) · Φ(Z) + σ · φ(Z)

    For minimization, the improvement direction is reversed:

        Z = (y_best - μ - ξ) / σ
        EI = (y_best - μ - ξ) · Φ(Z) + σ · φ(Z)

    where Φ is the standard normal CDF and φ is the standard normal PDF.

    When σ = 0, EI returns 0 (no uncertainty means no expected improvement
    beyond the point estimate).

    Examples
    --------
    >>> ei = ExpectedImprovement(xi=0.01)
    >>> scores = ei.evaluate(X, mean, std, y_best=0.5, objective="maximize")
    >>> next_point = X[np.argmax(scores)]

    Reference: Jones, Schonlau, Welch (1998), Efficient Global Optimization
    of Expensive Black-Box Functions.
    """

    def __init__(self, xi: float = 0.01):
        """Initialize Expected Improvement acquisition function.

        Parameters
        ----------
        xi : float, default=0.01
            Exploration-exploitation trade-off parameter. Must be non-negative.
            Higher values favor exploration.

        Raises
        ------
        ValueError
            If xi is negative.
        """
        raise NotImplementedError

    def _compute(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: float,
        objective: Literal["maximize", "minimize"],
    ) -> np.ndarray:
        """Compute Expected Improvement scores.

        Parameters
        ----------
        X : np.ndarray, shape (n_candidates, n_features)
            Candidate points (unused in computation, included for interface).
        mean : np.ndarray, shape (n_candidates,)
            Predicted mean values.
        std : np.ndarray, shape (n_candidates,)
            Predicted standard deviations.
        y_best : float
            Best observed target value so far.
        objective : {"maximize", "minimize"}
            Optimization direction.

        Returns
        -------
        np.ndarray, shape (n_candidates,)
            EI scores. Higher values indicate more promising candidates.

        Notes
        -----
        For points with std=0, returns 0 (no expected improvement when there
        is no uncertainty).
        """
        raise NotImplementedError


class UpperConfidenceBound(Acquisition):
    """Upper Confidence Bound acquisition function.

    UCB provides an optimistic estimate of the objective value by adding
    a multiple of the standard deviation to the predicted mean. This naturally
    balances exploitation (high mean) with exploration (high uncertainty).

    Parameters
    ----------
    beta : float, default=2.0
        Exploration parameter controlling the width of the confidence bound.
        Higher values favor exploration (more weight on uncertainty).
        Common choices: 2.0 (default), or schedule based on iteration.

    Attributes
    ----------
    beta : float
        The exploration parameter.

    Notes
    -----
    For maximization:

        UCB = μ + β · σ

    For minimization (we want to find low values, so we flip the sign of μ
    but still add uncertainty to encourage exploration):

        UCB = -μ + β · σ

    In both cases, higher UCB scores indicate more promising candidates.

    Examples
    --------
    >>> ucb = UpperConfidenceBound(beta=2.0)
    >>> scores = ucb.evaluate(X, mean, std, y_best=0.5, objective="maximize")
    >>> next_point = X[np.argmax(scores)]

    Reference: Srinivas et al. (2010), Gaussian Process Optimization in the
    Bandit Setting: No Regret and Experimental Design.
    """

    def __init__(self, beta: float = 2.0):
        """Initialize Upper Confidence Bound acquisition function.

        Parameters
        ----------
        beta : float, default=2.0
            Exploration parameter. Must be non-negative. Higher values
            increase the influence of uncertainty on the score.

        Raises
        ------
        ValueError
            If beta is negative.
        """
        raise NotImplementedError

    def _compute(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: float,
        objective: Literal["maximize", "minimize"],
    ) -> np.ndarray:
        """Compute Upper Confidence Bound scores.

        Parameters
        ----------
        X : np.ndarray, shape (n_candidates, n_features)
            Candidate points (unused in computation, included for interface).
        mean : np.ndarray, shape (n_candidates,)
            Predicted mean values.
        std : np.ndarray, shape (n_candidates,)
            Predicted standard deviations.
        y_best : float
            Best observed target value (unused by UCB, included for interface).
        objective : {"maximize", "minimize"}
            Optimization direction.

        Returns
        -------
        np.ndarray, shape (n_candidates,)
            UCB scores. Higher values indicate more promising candidates.
        """
        raise NotImplementedError
