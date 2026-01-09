"""Random sampling recommender for baseline comparisons."""

from typing import TYPE_CHECKING, Literal

import numpy as np

from folio.recommenders.base import Recommender

if TYPE_CHECKING:
    from folio.core.observation import Observation
    from folio.core.project import Project


class RandomRecommender(Recommender):
    """Recommender that samples uniformly at random from the input space.

    RandomRecommender ignores all observations and simply samples uniformly
    within the bounds of each continuous input. This provides a baseline for
    comparing more sophisticated optimization strategies.

    This recommender is useful for:
    - Initial exploration before switching to Bayesian optimization
    - Baseline comparisons in benchmarking studies
    - Situations where no prior information is available or useful

    Parameters
    ----------
    project : Project
        The project defining input specifications and bounds.

    Attributes
    ----------
    project : Project
        The project this recommender is configured for.

    Examples
    --------
    >>> from folio.core.project import Project
    >>> from folio.core.schema import InputSpec, OutputSpec
    >>> from folio.core.config import TargetConfig
    >>>
    >>> project = Project(
    ...     id=1,
    ...     name="example",
    ...     inputs=[
    ...         InputSpec("temperature", "continuous", bounds=(20.0, 100.0)),
    ...         InputSpec("pressure", "continuous", bounds=(1.0, 10.0)),
    ...     ],
    ...     outputs=[OutputSpec("yield")],
    ...     target_config=TargetConfig("yield"),
    ... )
    >>> recommender = RandomRecommender(project)
    >>> next_inputs = recommender.recommend([])
    >>> 20.0 <= next_inputs["temperature"] <= 100.0
    True

    Notes
    -----
    The random sampling is uniform within bounds. For more sophisticated
    space-filling designs (Latin hypercube, Sobol sequences), consider
    using a different recommender or pre-generating initial samples.
    """

    def __init__(self, project: "Project") -> None:
        """Initialize the random recommender with a project.

        Parameters
        ----------
        project : Project
            The project defining the experiment schema, including input
            specifications with bounds for continuous inputs.
        """
        raise NotImplementedError

    def recommend(self, observations: list["Observation"]) -> dict[str, float]:
        """Suggest random input values, ignoring all observations.

        This method samples uniformly at random within the bounds of each
        continuous input, completely ignoring the provided observations.

        Parameters
        ----------
        observations : list[Observation]
            Previous experiment observations. Ignored by RandomRecommender.

        Returns
        -------
        dict[str, float]
            Randomly sampled input values. Keys are input names, values are
            floats uniformly sampled within each input's bounds.

        Examples
        --------
        >>> recommender = RandomRecommender(project)
        >>> inputs = recommender.recommend([])
        >>> all(bounds[0] <= inputs[name] <= bounds[1] for name, bounds in ...)
        True
        """
        raise NotImplementedError

    def recommend_from_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        objective: Literal["maximize", "minimize"],
    ) -> np.ndarray:
        """Sample uniformly at random within bounds, ignoring X and y.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training inputs. Ignored by RandomRecommender.
        y : np.ndarray, shape (n_samples,)
            Training targets. Ignored by RandomRecommender.
        bounds : np.ndarray, shape (n_features, 2)
            Bounds for each input dimension. Each row is [lower, upper].
        objective : {"maximize", "minimize"}
            Optimization direction. Ignored by RandomRecommender.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Uniformly sampled values within bounds.

        Examples
        --------
        >>> bounds = np.array([[0.0, 10.0], [-5.0, 5.0]])
        >>> x = recommender.recommend_from_data(
        ...     X=np.empty((0, 2)),
        ...     y=np.empty(0),
        ...     bounds=bounds,
        ...     objective="maximize",
        ... )
        >>> x.shape
        (2,)
        >>> np.all((bounds[:, 0] <= x) & (x <= bounds[:, 1]))
        True
        """
        raise NotImplementedError
