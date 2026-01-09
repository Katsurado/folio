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
    - Low-level: `recommend_from_data(X, y, bounds, objective)` works with numpy arrays

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
    ...         objective: Literal["maximize", "minimize"],
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
    >>> y = np.array([1.0, 2.0])
    >>> bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    >>> next_x = recommender.recommend_from_data(X, y, bounds, "maximize")
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

    def recommend(self, observations: list["Observation"]) -> dict[str, float]:
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

        Returns
        -------
        dict[str, float]
            Suggested input values for the next experiment. Keys are input
            names matching the project's InputSpec names, values are the
            suggested settings. For continuous inputs, values are floats
            within the specified bounds.

        Notes
        -----
        - The returned dict contains all input names defined in the project
        - All values are within their respective bounds
        - Empty observation lists are handled gracefully (typically random sample)
        - Failed observations are filtered out before modeling

        Examples
        --------
        >>> recommender = BayesianRecommender(project)
        >>> # First recommendation with no prior data
        >>> first = recommender.recommend([])
        >>> # Subsequent recommendation using observed data
        >>> next_inputs = recommender.recommend(observations)
        """
        X, y = self.project.get_training_data(observations)
        bounds = self.project.get_optimization_bounds()
        objective = self.project.target_config.objective_mode
        candidate = self.recommend_from_data(X, y, bounds, objective)
        names = [inp.name for inp in self.project.inputs if inp.type == "continuous"]

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
        objective: str,
    ) -> np.ndarray:
        """Suggest next experiment inputs from raw numpy arrays.

        This is the low-level interface that works directly with arrays,
        independent of Project and Observation objects. Subclasses must
        implement this method with their specific recommendation strategy.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input features from previous experiments. Each row is
            an observation, each column is an input dimension. May be empty
            (shape (0, n_features)) for the first recommendation.
        y : np.ndarray, shape (n_samples,)
            Training target values corresponding to X. Computed from outputs
            using the project's target configuration (e.g., direct output,
            ratio, difference). May be empty for the first recommendation.
        bounds : np.ndarray, shape (2, n_features)
            Bounds for each input dimension. Row 0 contains lower bounds,
            row 1 contains upper bounds (BoTorch format).
        objective : {"maximize", "minimize"}
            Optimization direction. "maximize" seeks higher target values,
            "minimize" seeks lower values.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Suggested input values for the next experiment. Each element
            corresponds to an input dimension and must be within the
            corresponding bounds.

        Notes
        -----
        - When X is empty, implementations typically return a random sample
        - All returned values must satisfy: bounds[i, 0] <= result[i] <= bounds[i, 1]
        - The objective parameter determines whether to seek maxima or minima

        Examples
        --------
        >>> X = np.array([[0.2, 0.3], [0.5, 0.7], [0.8, 0.1]])
        >>> y = np.array([1.0, 2.0, 1.5])
        >>> bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        >>> next_x = recommender.recommend_from_data(X, y, bounds, "maximize")
        >>> next_x.shape
        (2,)
        >>> np.all((bounds[:, 0] <= next_x) & (next_x <= bounds[:, 1]))
        True
        """
        ...
