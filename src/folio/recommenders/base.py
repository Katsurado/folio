"""Abstract base class for experiment recommenders."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from folio.core.observation import Observation
    from folio.core.project import Project


class Recommender(ABC):
    """Abstract base class for experiment recommenders.

    Recommenders suggest the next experiment to run based on previous observations.
    They encapsulate different strategies for exploring the input space, from simple
    random sampling to sophisticated Bayesian optimization.

    The interface has two levels:
    - High-level: `recommend(observations)` works with Project and Observation objects
    - Low-level: `recommend_from_data(X, Y, bounds, objectives)` works with numpy arrays

    The high-level method extracts data from Project/Observations and delegates to
    the low-level method, which implements the actual recommendation logic.

    Parameters
    ----------
    project : Project
        The project defining input specifications, bounds, and optimization target.
        The recommender uses this to understand the search space.

    Attributes
    ----------
    project : Project
        The project this recommender is configured for.

    Notes
    -----
    Subclasses must implement `recommend_from_data` to provide their specific
    strategy for suggesting next experiments. The high-level `recommend` method
    is implemented in the base class and handles data extraction.

    Examples
    --------
    Implementing a custom recommender:

    >>> class MyRecommender(Recommender):
    ...     def __init__(self, project: Project):
    ...         super().__init__(project)
    ...
    ...     def recommend_from_data(
    ...         self,
    ...         X: np.ndarray,
    ...         y: np.ndarray,
    ...         bounds: np.ndarray,
    ...         maximize: list[bool],
    ...     ) -> np.ndarray:
    ...         # Custom logic to suggest next experiment
    ...         return np.array([0.5, 0.5])

    Using a recommender:

    >>> recommender = BayesianRecommender(project)
    >>> next_inputs = recommender.recommend(observations)
    >>> print(next_inputs)
    {"temperature": 85.0, "pressure": 3.5}

    Using the low-level interface directly:

    >>> X = np.array([[0.2, 0.3], [0.5, 0.7]])
    >>> y = np.array([[1.0], [2.0]])  # shape (n, m) for m objectives
    >>> bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    >>> next_x = recommender.recommend_from_data(X, y, bounds, [True])
    """

    def __init__(self, project: "Project") -> None:
        """Initialize the recommender with a project.

        Parameters
        ----------
        project : Project
            The project defining the experiment schema, including input
            specifications with bounds/levels and the optimization target.
        """
        self.project = project

    def recommend(
        self,
        observations: list["Observation"],
        fixed_inputs: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Suggest the next experiment inputs based on previous observations.

        This method extracts training data from observations using the project's
        target configuration, then delegates to `recommend_from_data` for the
        actual recommendation logic.

        Parameters
        ----------
        observations : list[Observation]
            Previous experiment observations. May be empty for the first
            recommendation. Failed observations (obs.failed=True) are
            automatically excluded from training data.
        fixed_inputs : dict[str, float] | None, optional
            Current values for non-optimizable inputs. Required if the project
            has inputs with `optimizable=False`. Keys are non-optimizable input
            names, values are the current values to hold fixed during acquisition
            optimization.

        Returns
        -------
        dict[str, float]
            Suggested input values for the next experiment. Keys are optimizable
            input names only (non-optimizable inputs are not included). Values
            are floats within the specified bounds.

        Raises
        ------
        ValueError
            If the project has non-optimizable inputs but fixed_inputs is not
            provided, or if fixed_inputs is missing required keys.

        Notes
        -----
        - The returned dict contains only optimizable input names
        - All values are within their respective bounds
        - Empty observation lists are handled gracefully (typically random sample)
        - Failed observations are filtered out before modeling

        Examples
        --------
        >>> recommender = BayesianRecommender(project)
        >>> # First recommendation with no prior data
        >>> first = recommender.recommend([])
        >>> # With non-optimizable inputs
        >>> next_inputs = recommender.recommend(
        ...     observations,
        ...     fixed_inputs={"hour": 14.0, "ambient_temp": 22.0},
        ... )
        """
        non_opt_inputs = self.project.get_non_optimizable_inputs()

        # Validate fixed_inputs is provided when needed
        if non_opt_inputs:
            if fixed_inputs is None:
                non_opt_names = [inp.name for inp in non_opt_inputs]
                raise ValueError(
                    f"fixed_inputs is required for non-optimizable inputs: "
                    f"{non_opt_names}"
                )
            # Check all required keys are provided
            missing = [
                inp.name for inp in non_opt_inputs if inp.name not in fixed_inputs
            ]
            if missing:
                raise ValueError(f"Missing fixed_inputs values: {missing}")

        X, y = self.project.get_training_data(observations)
        bounds = self.project.get_optimization_bounds()
        maximize = [
            cfg.objective_mode == "maximize" for cfg in self.project.target_configs
        ]

        # Build fixed feature indices and values for recommend_from_data
        fixed_indices = self.project.get_non_optimizable_indices()
        fixed_values = None
        if fixed_indices and fixed_inputs is not None:
            fixed_values = [fixed_inputs[inp.name] for inp in non_opt_inputs]

        candidate = self.recommend_from_data(
            X,
            y,
            bounds,
            maximize,
            fixed_feature_indices=fixed_indices,
            fixed_feature_values=fixed_values,
        )

        # Only return optimizable input names
        optimizable_inputs = self.project.get_optimizable_inputs()
        names = [inp.name for inp in optimizable_inputs]

        next_input = {names[i]: float(candidate[i]) for i in range(len(names))}

        return next_input

    @staticmethod
    def random_sample_from_bounds(bounds: np.ndarray) -> np.ndarray:
        """Sample uniformly at random within bounds.

        Parameters
        ----------
        bounds : np.ndarray, shape (2, d)
            Bounds for each input dimension. Row 0 contains lower bounds,
            row 1 contains upper bounds.

        Returns
        -------
        np.ndarray, shape (d,)
            Uniformly sampled values within bounds.

        Examples
        --------
        >>> bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        >>> sample = Recommender.random_sample_from_bounds(bounds)
        >>> sample.shape
        (2,)
        >>> np.all((bounds[0, :] <= sample) & (sample <= bounds[1, :]))
        True
        """
        return np.random.uniform(bounds[0, :], bounds[1, :])

    @abstractmethod
    def recommend_from_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        maximize: list[bool],
        fixed_feature_indices: list[int] | None = None,
        fixed_feature_values: list[float] | None = None,
    ) -> np.ndarray:
        """Suggest next experiment inputs from raw numpy arrays.

        This is the low-level interface that works directly with arrays,
        independent of Project and Observation objects. Subclasses must
        implement this method with their specific recommendation strategy.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_all_features)
            Training input features from previous experiments. Each row is
            an observation, each column is an input dimension. Includes both
            optimizable and non-optimizable features. May be empty
            (shape (0, n_features)) for the first recommendation.
        y : np.ndarray, shape (n_samples, n_objectives)
            Training target values corresponding to X. Each column is one
            objective. For single-objective, shape is (n, 1). Computed from
            outputs using the project's target configurations. May be empty
            for the first recommendation.
        bounds : np.ndarray, shape (2, n_optimizable_features)
            Bounds for each optimizable input dimension. Row 0 contains lower
            bounds, row 1 contains upper bounds (BoTorch format). Does NOT
            include bounds for non-optimizable features.
        maximize : list[bool]
            Whether to maximize each objective. True = maximize, False = minimize.
            Length must match y.shape[1].
        fixed_feature_indices : list[int] | None, optional
            Indices of non-optimizable features in the X array. Used to
            construct fixed_features for BoTorch's optimize_acqf.
        fixed_feature_values : list[float] | None, optional
            Current values for non-optimizable features, corresponding to
            fixed_feature_indices. These values are held fixed during
            acquisition optimization.

        Returns
        -------
        np.ndarray, shape (n_optimizable_features,)
            Suggested input values for optimizable features only. Each element
            corresponds to an optimizable input dimension and must be within
            the corresponding bounds.

        Notes
        -----
        - When X is empty, implementations typically return a random sample
        - All returned values must satisfy: bounds[i, 0] <= result[i] <= bounds[i, 1]
        - For single-objective, maximize has one element and y has shape (n, 1)
        - For multi-objective, maximize has m elements and y has shape (n, m)
        - The returned array has shape (n_optimizable_features,), NOT (n_all_features,)

        Examples
        --------
        >>> # Single-objective
        >>> X = np.array([[0.2, 0.3], [0.5, 0.7], [0.8, 0.1]])
        >>> y = np.array([[1.0], [2.0], [1.5]])
        >>> bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> next_x = recommender.recommend_from_data(X, y, bounds, [True])

        >>> # With non-optimizable features
        >>> X = np.array([[0.2, 0.5, 0.3], [0.5, 0.5, 0.7]])  # 3 features
        >>> bounds = np.array([[0.0, 0.0], [1.0, 1.0]])  # Only 2 optimizable
        >>> next_x = recommender.recommend_from_data(
        ...     X, y, bounds, [True],
        ...     fixed_feature_indices=[1],
        ...     fixed_feature_values=[0.5],
        ... )
        >>> next_x.shape  # (2,) - only optimizable features
        """
        ...
