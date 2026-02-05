"""Bayesian optimization recommender using GP surrogate and acquisition functions."""

from typing import TYPE_CHECKING

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim.optimize import optimize_acqf

from folio.recommenders.acquisitions import (
    NEHVI,
    ExpectedImprovement,
    PosteriorVariance,
    UpperConfidenceBound,
)
from folio.recommenders.base import Recommender
from folio.surrogates import MultiTaskGPSurrogate, SingleTaskGPSurrogate
from folio.surrogates.base import Surrogate

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
    ...     target_configs=[TargetConfig(objective="yield", objective_mode="maximize")],
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

    @property
    def surrogate(self) -> SingleTaskGPSurrogate | MultiTaskGPSurrogate | None:
        """The fitted surrogate model from the last recommend_from_data call.

        Returns None if:
        - recommend_from_data has not been called yet
        - The last call had fewer than n_initial observations (random sampling)

        Returns
        -------
        SingleTaskGPSurrogate | MultiTaskGPSurrogate | None
            The fitted surrogate model, or None if not yet fitted.
        """
        return self._surrogate

    def recommend_from_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        maximize: list[bool],
        fixed_feature_indices: list[int] | None = None,
        fixed_feature_values: list[float] | None = None,
    ) -> np.ndarray:
        """Suggest next experiment inputs from raw arrays.

        If fewer than n_initial samples, returns random sample within bounds.
        Otherwise, fits GP surrogate, builds acquisition function, and
        optimizes to find the next point.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_all_features)
            Training inputs from previous experiments. Includes both
            optimizable and non-optimizable features.
        y : np.ndarray, shape (n_samples, n_objectives)
            Training targets. Shape (n, 1) for single-objective,
            (n, m) for m objectives.
        bounds : np.ndarray, shape (2, n_optimizable_features)
            Bounds for optimizable input dimensions only. Row 0 contains
            lower bounds, row 1 contains upper bounds (BoTorch format).
        maximize : list[bool]
            Whether to maximize each objective. True = maximize, False = minimize.
        fixed_feature_indices : list[int] | None, optional
            Indices of non-optimizable features in the X array.
        fixed_feature_values : list[float] | None, optional
            Current values for non-optimizable features.

        Returns
        -------
        np.ndarray, shape (n_optimizable_features,)
            Suggested input values for optimizable features only.

        Examples
        --------
        >>> X = np.array([[0.2, 0.3], [0.5, 0.7], [0.8, 0.1]])
        >>> y = np.array([[1.0], [2.0], [1.5]])
        >>> bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> next_x = recommender.recommend_from_data(X, y, bounds, [True])
        """
        if X.dtype != np.float64:
            raise ValueError(f"X must be float64, got {X.dtype}")
        if y.dtype != np.float64:
            raise ValueError(f"y must be float64, got {y.dtype}")

        if len(X) < self.project.recommender_config.n_initial:
            self._surrogate = None
            return self.random_sample_from_bounds(bounds)

        self._fit_surrogate(X, y)
        return self._optimize_acquisition(
            X, y, bounds, maximize, fixed_feature_indices, fixed_feature_values
        )

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
        y : np.ndarray, shape (n_samples, n_objectives)
            Training target values. Shape (n, 1) for single-objective,
            (n, m) for m objectives.

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
        self._surrogate = self._build_surrogate()
        self._surrogate.fit(X, y)

    def _optimize_acquisition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        maximize: list[bool],
        fixed_feature_indices: list[int] | None = None,
        fixed_feature_values: list[float] | None = None,
    ) -> np.ndarray:
        """Optimize the acquisition function to find the next point.

        Uses BoTorch's optimize_acqf to find the input that maximizes
        the acquisition function value.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_all_features)
            Training inputs, passed to acquisition builder for multi-objective.
        y : np.ndarray, shape (n_samples, n_objectives)
            Training target values. For single-objective, used to compute
            best_f for improvement-based acquisition functions.
        bounds : np.ndarray, shape (2, n_optimizable_features)
            Bounds for optimization (optimizable features only).
            Row 0 = lower, row 1 = upper.
        maximize : list[bool]
            Whether to maximize each objective. True = maximize, False = minimize.
        fixed_feature_indices : list[int] | None, optional
            Indices of non-optimizable features in the full X array.
        fixed_feature_values : list[float] | None, optional
            Current values for non-optimizable features.

        Returns
        -------
        np.ndarray, shape (n_optimizable_features,)
            Optimal input values for optimizable features only.

        Notes
        -----
        This method assumes _fit_surrogate has already been called and
        self._surrogate is a fitted GP model.

        The optimization uses L-BFGS-B with multiple random restarts
        to find a global optimum of the acquisition function.

        When fixed_feature_indices and fixed_feature_values are provided,
        BoTorch's fixed_features parameter is used to hold those dimensions
        fixed during optimization.

        Examples
        --------
        >>> recommender._fit_surrogate(X, y)
        >>> next_x = recommender._optimize_acquisition(X, y, bounds, [True])
        """
        acq = self._build_acquisition(X, y, maximize)

        # Build full bounds including fixed features
        n_all_features = X.shape[1]

        if fixed_feature_indices and fixed_feature_values:
            # Build full bounds array with fixed features included
            full_bounds = np.zeros((2, n_all_features))
            opt_idx = 0
            for i in range(n_all_features):
                if i in fixed_feature_indices:
                    # Fixed feature: set bounds to the fixed value
                    fixed_idx = fixed_feature_indices.index(i)
                    val = fixed_feature_values[fixed_idx]
                    full_bounds[0, i] = val
                    full_bounds[1, i] = val
                else:
                    # Optimizable feature: use provided bounds
                    full_bounds[0, i] = bounds[0, opt_idx]
                    full_bounds[1, i] = bounds[1, opt_idx]
                    opt_idx += 1

            bounds_tensor = torch.tensor(full_bounds, dtype=torch.float64)

            # Build fixed_features dict for optimize_acqf
            fixed_features = {
                idx: fixed_feature_values[fixed_feature_indices.index(idx)]
                for idx in fixed_feature_indices
            }

            candidates, acq_values = optimize_acqf(
                acq,
                bounds_tensor,
                q=1,
                num_restarts=5,
                raw_samples=5,
                fixed_features=fixed_features,
            )

            # Extract only the optimizable feature values from the result
            full_candidate = candidates[0].detach().numpy()
            opt_candidate = np.array(
                [
                    full_candidate[i]
                    for i in range(n_all_features)
                    if i not in fixed_feature_indices
                ]
            )
            return opt_candidate
        else:
            bounds_tensor = torch.tensor(bounds, dtype=torch.float64)

            candidates, acq_values = optimize_acqf(
                acq, bounds_tensor, q=1, num_restarts=5, raw_samples=5
            )

            return candidates[0].detach().numpy()

    def _build_surrogate(self) -> Surrogate:
        """Build the appropriate surrogate model based on project configuration.

        Dispatches to SingleTaskGPSurrogate for single-objective optimization
        or MultiTaskGPSurrogate for multi-objective optimization, based on
        project.is_multi_objective() and project.recommender_config.

        Returns
        -------
        Surrogate
            An unfitted surrogate model instance. For single-objective, returns
            SingleTaskGPSurrogate. For multi-objective, returns MultiTaskGPSurrogate.

        Notes
        -----
        The surrogate type is determined by:
        1. project.is_multi_objective() -> determines single vs multi-objective
        2. recommender_config.surrogate -> "gp" or "multitask_gp"

        For multi-objective:
        - "multitask_gp": Uses ICM kernel to model correlations between objectives

        Examples
        --------
        >>> # Single-objective project
        >>> surrogate = recommender._build_surrogate()
        >>> isinstance(surrogate, SingleTaskGPSurrogate)  # True

        >>> # Multi-objective project with multitask GP
        >>> surrogate = recommender._build_surrogate()
        >>> isinstance(surrogate, MultiTaskGPSurrogate)  # True
        """
        surrogate_type = self.project.recommender_config.surrogate
        surrogate_kwargs = self.project.recommender_config.surrogate_kwargs
        if surrogate_type == "multitask_gp":
            return MultiTaskGPSurrogate(**surrogate_kwargs)
        elif surrogate_type == "gp":
            return SingleTaskGPSurrogate(**surrogate_kwargs)
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")

    def _build_acquisition(
        self, X: np.ndarray, y: np.ndarray, maximize: list[bool]
    ) -> AcquisitionFunction:
        """Build the appropriate acquisition function based on project configuration.

        Dispatches to EI/UCB for single-objective or NEHVI for multi-objective
        optimization, based on project.is_multi_objective() and recommender_config.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features), dtype float64
            Training inputs, used by multi-objective acquisitions (e.g., NEHVI)
            to compute the Pareto frontier.
        y : np.ndarray, shape (n_samples, n_objectives), dtype float64
            Training targets. For single-objective, shape is (n, 1). For
            multi-objective, shape is (n, m) where m = number of objectives.
        maximize : list[bool]
            Whether to maximize each objective. True = maximize, False = minimize.

        Returns
        -------
        AcquisitionFunction
            A BoTorch-compatible acquisition function ready for optimize_acqf.
            For single-objective: EI or UCB. For multi-objective: NEHVI.

        Notes
        -----
        This method assumes _fit_surrogate has been called and self._surrogate
        is a fitted model.

        For single-objective:
        - Computes best_f from y (max or min depending on maximize[0])
        - Builds EI or UCB

        For multi-objective:
        - Uses reference_point from project.reference_point
        - Builds NEHVI with model, X_baseline, y, ref_point, maximize

        Examples
        --------
        >>> # Single-objective
        >>> acqf = recommender._build_acquisition(X, y, [True])

        >>> # Multi-objective
        >>> acqf = recommender._build_acquisition(X, y, [True, False])
        """
        acq_kwargs = self.project.recommender_config.acquisition_kwargs
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        # TODO TEMP AL branch for AL demo with claude light
        # TODO: clean up for proper AL/BO branching
        if self.project.recommender_config.acquisition == "variance":
            builder = PosteriorVariance()
            return builder.build(self._surrogate.model)

        if self.project.is_multi_objective():
            acq_type = self.project.recommender_config.mo_acquisition
            ref_pt = self.project.reference_point
            if acq_type == "nehvi":
                builder = NEHVI(**acq_kwargs)
                return builder.build(self._surrogate.model, X, y, ref_pt, maximize)
            else:
                raise ValueError(f"Unknown acquisition type: {acq_type}")
        else:
            acq_type = self.project.recommender_config.acquisition

            if maximize[0]:
                best_f = float(y.max())
            else:
                best_f = float(y.min())

            if acq_type == "ei":
                builder = ExpectedImprovement(**acq_kwargs)
            elif acq_type == "ucb":
                builder = UpperConfidenceBound(**acq_kwargs)
            else:
                raise ValueError(f"Unknown acquisition function: {acq_type}")

            return builder.build(self._surrogate.model, best_f, maximize[0])
