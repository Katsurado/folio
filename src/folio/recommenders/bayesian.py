"""Bayesian optimization recommender using GP surrogate and acquisition functions."""

from typing import TYPE_CHECKING, Literal

import numpy as np

from folio.recommenders.acquisitions import (
    Acquisition,
)
from folio.recommenders.base import Recommender
from folio.surrogates import MultiTaskGPSurrogate, SingleTaskGPSurrogate

if TYPE_CHECKING:
    from folio.core.observation import Observation
    from folio.core.project import Project


class BayesianRecommender(Recommender):
    """Bayesian optimization recommender using Gaussian Process surrogate.

    BayesianRecommender uses a Gaussian Process to model the objective function
    and an acquisition function to balance exploration and exploitation when
    suggesting the next experiment.

    The recommender operates in two phases:
    1. Initial phase (< n_initial observations): Returns random samples
    2. BO phase (>= n_initial observations): Fits GP, optimizes acquisition

    Parameters
    ----------
    project : Project
        The project defining input specifications, bounds, and recommender
        configuration. The recommender_config specifies:
        - surrogate: Surrogate model type ("gp" for SingleTaskGP,
          "multitask_gp" for MultiTaskGP)
        - acquisition: Acquisition function ("ei" or "ucb")
        - n_initial: Number of random samples before using BO
        - kwargs: Additional parameters (e.g., {"xi": 0.01}, {"beta": 2.0})

    Attributes
    ----------
    project : Project
        The project this recommender is configured for.
    _surrogate : SingleTaskGPSurrogate | MultiTaskGPSurrogate | None
        The fitted surrogate model. None until fit. The surrogate type is
        determined by project.recommender_config.surrogate.

    Examples
    --------
    >>> from folio.core.project import Project
    >>> from folio.core.schema import InputSpec, OutputSpec
    >>> from folio.core.config import TargetConfig, RecommenderConfig
    >>>
    >>> project = Project(
    ...     id=1,
    ...     name="example",
    ...     inputs=[
    ...         InputSpec("temperature", "continuous", bounds=(20.0, 100.0)),
    ...         InputSpec("pressure", "continuous", bounds=(1.0, 10.0)),
    ...     ],
    ...     outputs=[OutputSpec("yield")],
    ...     target_config=TargetConfig("yield", mode="maximize"),
    ...     recommender_config=RecommenderConfig(
    ...         type="bayesian",
    ...         surrogate="gp",
    ...         acquisition="ei",
    ...         n_initial=5,
    ...     ),
    ... )
    >>> recommender = BayesianRecommender(project)
    >>> # With few observations, returns random sample
    >>> next_inputs = recommender.recommend(few_observations)
    >>> # With enough observations, uses GP + acquisition
    >>> next_inputs = recommender.recommend(many_observations)

    Notes
    -----
    The GP surrogate uses a Matérn 2.5 kernel with ARD (Automatic Relevance
    Determination) by default. Input and output normalization are enabled
    for numerical stability.

    References
    ----------
    .. [1] Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient Global
           Optimization of Expensive Black-Box Functions. Journal of Global
           Optimization, 13(4), 455-492.
    .. [2] Srinivas, N., et al. (2010). Gaussian Process Optimization in the
           Bandit Setting: No Regret and Experimental Design. ICML.
    """

    _surrogate: SingleTaskGPSurrogate | MultiTaskGPSurrogate | None

    def __init__(self, project: "Project") -> None:
        """Initialize the Bayesian recommender with a project.

        Parameters
        ----------
        project : Project
            The project defining the experiment schema and recommender
            configuration.
        """
        raise NotImplementedError

    def recommend(self, observations: list["Observation"]) -> dict[str, float]:
        """Suggest next experiment inputs using Bayesian optimization.

        Extracts training data from observations and delegates to
        `recommend_from_data`. Returns random sample if fewer than
        n_initial valid observations; otherwise uses GP + acquisition.

        Parameters
        ----------
        observations : list[Observation]
            Previous experiment observations. Failed observations are
            automatically excluded from training data.

        Returns
        -------
        dict[str, float]
            Suggested input values for the next experiment.

        Examples
        --------
        >>> recommender = BayesianRecommender(project)
        >>> next_inputs = recommender.recommend(observations)
        >>> project.validate_inputs(next_inputs)  # Should not raise
        """
        raise NotImplementedError

    def recommend_from_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        objective: Literal["maximize", "minimize"],
    ) -> np.ndarray:
        """Suggest next experiment inputs from raw arrays.

        If fewer than n_initial samples, returns random sample within bounds.
        Otherwise, fits GP surrogate, builds acquisition function, and
        optimizes to find the next point.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training inputs from previous experiments.
        y : np.ndarray, shape (n_samples,)
            Training targets (scalar objective values).
        bounds : np.ndarray, shape (n_features, 2)
            Bounds for each input dimension. Each row is [lower, upper].
        objective : {"maximize", "minimize"}
            Optimization direction.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Suggested input values for the next experiment.

        Examples
        --------
        >>> X = np.array([[0.2, 0.3], [0.5, 0.7], [0.8, 0.1]])
        >>> y = np.array([1.0, 2.0, 1.5])
        >>> bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        >>> next_x = recommender.recommend_from_data(X, y, bounds, "maximize")
        """
        raise NotImplementedError

    def _random_sample(self) -> dict[str, float]:
        """Generate a random sample within input bounds.

        Returns
        -------
        dict[str, float]
            Randomly sampled input values. Keys are input names, values
            are floats uniformly sampled within each input's bounds.

        Notes
        -----
        This method is called during the initial exploration phase when
        there are fewer than n_initial observations.

        Examples
        --------
        >>> recommender = BayesianRecommender(project)
        >>> sample = recommender._random_sample()
        >>> project.validate_inputs(sample)  # Should not raise
        """
        raise NotImplementedError

    def _fit_surrogate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP surrogate model to training data.

        Creates and fits a surrogate model to the provided training data.
        The surrogate type is determined by project.recommender_config.surrogate:
        - "gp": SingleTaskGPSurrogate
        - "multitask_gp": MultiTaskGPSurrogate

        The fitted model is stored in self._surrogate for use by the
        acquisition function.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input features.
        y : np.ndarray, shape (n_samples,)
            Training target values.

        Notes
        -----
        After calling this method, self._surrogate will be a fitted
        surrogate instance that can be used for predictions.

        Examples
        --------
        >>> recommender = BayesianRecommender(project)
        >>> X, y = project.get_training_data(observations)
        >>> recommender._fit_surrogate(X, y)
        >>> assert recommender._surrogate._is_fitted
        """
        raise NotImplementedError

    def _optimize_acquisition(self, y: np.ndarray) -> dict[str, float]:
        """Optimize the acquisition function to find the next point.

        Uses BoTorch's optimize_acqf to find the input that maximizes
        the acquisition function value.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            Training target values, used to compute best_f for
            improvement-based acquisition functions.

        Returns
        -------
        dict[str, float]
            Optimal input values according to the acquisition function.

        Notes
        -----
        This method assumes _fit_surrogate has already been called and
        self._surrogate is a fitted GP model. The acquisition function
        is built using _build_acquisition().

        The optimization uses L-BFGS-B with multiple random restarts
        to find a global optimum of the acquisition function.

        Examples
        --------
        >>> recommender._fit_surrogate(X, y)
        >>> next_inputs = recommender._optimize_acquisition(y)
        """
        raise NotImplementedError

    def _build_acquisition(self) -> Acquisition:
        """Build the acquisition function based on recommender config.

        Creates an ExpectedImprovement or UpperConfidenceBound instance
        based on the project's recommender_config.acquisition setting.
        Passes through any relevant kwargs (xi for EI, beta for UCB).

        Returns
        -------
        Acquisition
            An acquisition function builder (ExpectedImprovement or
            UpperConfidenceBound) configured with kwargs from the
            recommender config.

        Examples
        --------
        >>> recommender = BayesianRecommender(project)  # acquisition="ei"
        >>> acq = recommender._build_acquisition()
        >>> isinstance(acq, ExpectedImprovement)
        True

        >>> recommender = BayesianRecommender(ucb_project)  # acquisition="ucb"
        >>> acq = recommender._build_acquisition()
        >>> isinstance(acq, UpperConfidenceBound)
        True
        """
        raise NotImplementedError
