"""Bayesian optimization recommender using GP surrogate and acquisition functions."""

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim.optimize import optimize_acqf

from folio.recommenders.acquisitions import (
    ExpectedImprovement,
    UpperConfidenceBound,
)
from folio.recommenders.base import Recommender
from folio.surrogates import MultiTaskGPSurrogate, SingleTaskGPSurrogate

if TYPE_CHECKING:
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
        super().__init__(project)
        self._surrogate = None

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
        bounds : np.ndarray, shape (2, n_features)
            Bounds for each input dimension. Row 0 contains lower bounds,
            row 1 contains upper bounds (BoTorch format).
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
        >>> bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> next_x = recommender.recommend_from_data(X, y, bounds, "maximize")
        """
        if len(X) < self.project.recommender_config.n_initial:
            return self.random_sample_from_bounds(bounds)

        self._fit_surrogate(X, y)
        return self._optimize_acquisition(y, bounds, objective)

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
        surrogate_type = self.project.recommender_config.surrogate
        kwargs = self.project.recommender_config.surrogate_kwargs

        if surrogate_type == "gp":
            self._surrogate = SingleTaskGPSurrogate(**kwargs)
        elif surrogate_type == "multitask_gp":
            self._surrogate = MultiTaskGPSurrogate(**kwargs)
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")

        self._surrogate.fit(X, y)

    def _optimize_acquisition(
        self, y: np.ndarray, bounds: np.ndarray, objective: str
    ) -> np.ndarray:
        """Optimize the acquisition function to find the next point.

        Uses BoTorch's optimize_acqf to find the input that maximizes
        the acquisition function value.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            Training target values, used to compute best_f for
            improvement-based acquisition functions.
        bounds : np.ndarray, shape (2, n_features)
            Bounds for optimization. Row 0 = lower, row 1 = upper.
        objective : {"maximize", "minimize"}
            Optimization direction, used to determine best_f.

        Returns
        -------
        np.ndarray, shape (n_features,)
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
        >>> next_x = recommender._optimize_acquisition(y, bounds, "maximize")
        """
        if objective == "maximize":
            best_f = float(y.max())
            maximize = True
        elif objective == "minimize":
            best_f = float(y.min())
            maximize = False
        else:
            raise ValueError(f"Unknown objective: {objective}")

        acq = self._build_acquisition(best_f, maximize)

        bounds = torch.tensor(bounds, dtype=torch.float64)

        candidates, acq_values = optimize_acqf(
            acq, bounds, q=1, num_restarts=5, raw_samples=5
        )

        return candidates[0].detach().numpy()

    def _build_acquisition(self, best_f: float, maximize: bool) -> AcquisitionFunction:
        """Build a BoTorch acquisition function based on recommender config.

        Creates and returns a ready-to-use BoTorch acquisition function
        using self._surrogate and the project's recommender_config.acquisition
        setting.

        Parameters
        ----------
        best_f : float
            Best observed target value so far.
        maximize : bool
            If True, seek higher values; if False, seek lower values.

        Returns
        -------
        AcquisitionFunction
            A BoTorch-compatible acquisition function ready for optimize_acqf.

        Notes
        -----
        This method assumes _fit_surrogate has already been called and
        self._surrogate is a fitted surrogate model.

        Examples
        --------
        >>> recommender = BayesianRecommender(project)  # acquisition="ei"
        >>> recommender._fit_surrogate(X, y)
        >>> acq_fn = recommender._build_acquisition(best_f=1.0, maximize=True)
        """
        acq_type = self.project.recommender_config.acquisition
        kwargs = self.project.recommender_config.acquisition_kwargs

        if acq_type == "ei":
            builder = ExpectedImprovement(**kwargs)
        elif acq_type == "ucb":
            builder = UpperConfidenceBound(**kwargs)
        else:
            raise ValueError(f"Unknown acquisition function: {acq_type}")

        return builder.build(self._surrogate.model, best_f, maximize)
