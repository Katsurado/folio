"""High-level API for Folio electronic lab notebook."""

import logging
from pathlib import Path

from folio.core import database
from folio.core.config import RecommenderConfig, TargetConfig
from folio.core.database import DEFAULT_DB_PATH
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.recommenders import BayesianRecommender, RandomRecommender, Recommender

logger = logging.getLogger(__name__)


class Folio:
    """High-level API for the Folio electronic lab notebook.

    Folio is the main entry point for users. It orchestrates all operations:
    creating/loading projects, recording observations, and getting experiment
    suggestions. Internally, it manages database access and recommender instances.

    The Folio class caches recommender instances per project to avoid rebuilding
    them on every call to `suggest()`. The cache is keyed by project name.

    Parameters
    ----------
    db_path : Path | str, optional
        Path to the SQLite database file. Defaults to ~/.folio/folio.db.
        Parent directories are created automatically if they don't exist.

    Attributes
    ----------
    db_path : Path
        Path to the database file.

    Examples
    --------
    Basic workflow:

    >>> folio = Folio()
    >>> folio.create_project(
    ...     name="yield_optimization",
    ...     inputs=[InputSpec("temperature", "continuous", bounds=(20.0, 100.0))],
    ...     outputs=[OutputSpec("yield")],
    ...     target_configs=[TargetConfig(objective="yield", objective_mode="maximize")],
    ... )
    >>> folio.add_observation(
    ...     project_name="yield_optimization",
    ...     inputs={"temperature": 50.0},
    ...     outputs={"yield": 75.0},
    ... )
    >>> next_experiment = folio.suggest("yield_optimization")

    Using a custom database path:

    >>> folio = Folio(db_path="./my_experiments.db")
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        """Initialize Folio with database path.

        Parameters
        ----------
        db_path : Path | str, optional
            Path to the SQLite database file. Defaults to ~/.folio/folio.db.
            Will be converted to Path if string is provided.
        """
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self._recommenders: dict[str, Recommender] = {}
        database.init_db(self.db_path)

    def create_project(
        self,
        name: str,
        inputs: list[InputSpec],
        outputs: list[OutputSpec],
        target_configs: list[TargetConfig],
        reference_point: list[float] | None = None,
        recommender_config: RecommenderConfig | None = None,
    ) -> None:
        """Create a new project in the database.

        Parameters
        ----------
        name : str
            Unique project name. Cannot be empty.
        inputs : list[InputSpec]
            Input variable specifications. Must have at least one.
        outputs : list[OutputSpec]
            Output variable specifications. Must have at least one.
        target_configs : list[TargetConfig]
            Optimization target configurations. For single-objective, provide one.
            For multi-objective, provide multiple and include reference_point.
        reference_point : list[float] | None, optional
            Reference point for multi-objective hypervolume calculation.
            Required when len(target_configs) > 1. Length must match target_configs.
        recommender_config : RecommenderConfig | None, optional
            Configuration for the experiment recommender. Defaults to Bayesian
            optimization with GP surrogate and EI acquisition.

        Raises
        ------
        ProjectExistsError
            If a project with the same name already exists.
        InvalidSchemaError
            If the project schema is invalid (empty name, no inputs/outputs,
            duplicate names, invalid bounds, missing reference_point for
            multi-objective, etc.).

        Examples
        --------
        Single-objective project:

        >>> folio.create_project(
        ...     name="yield_optimization",
        ...     inputs=[InputSpec("temperature", "continuous", bounds=(20.0, 100.0))],
        ...     outputs=[OutputSpec("yield")],
        ...     target_configs=[TargetConfig(objective="yield")],
        ... )

        Multi-objective project:

        >>> folio.create_project(
        ...     name="pareto_optimization",
        ...     inputs=[InputSpec("x", "continuous", bounds=(0.0, 1.0))],
        ...     outputs=[OutputSpec("yield"), OutputSpec("purity")],
        ...     target_configs=[
        ...         TargetConfig(objective="yield", objective_mode="maximize"),
        ...         TargetConfig(objective="purity", objective_mode="maximize"),
        ...     ],
        ...     reference_point=[0.0, 0.0],
        ... )
        """
        if recommender_config is None:
            recommender_config = RecommenderConfig()

        project = Project(
            id=None,
            name=name,
            inputs=inputs,
            outputs=outputs,
            target_configs=target_configs,
            reference_point=reference_point,
            recommender_config=recommender_config,
        )

        database.create_project(project, self.db_path)

    def list_projects(self) -> list[str]:
        """List all project names in the database.

        Returns
        -------
        list[str]
            Names of all projects, sorted alphabetically.

        Examples
        --------
        >>> folio.list_projects()
        ['optimization_v1', 'screening', 'yield_optimization']
        """
        return database.list_projects(self.db_path)

    def delete_project(self, name: str) -> None:
        """Delete a project and all its observations.

        Removes the project from the database along with all associated
        observations. Also removes the cached recommender if present.

        Parameters
        ----------
        name : str
            Name of the project to delete.

        Raises
        ------
        ProjectNotFoundError
            If no project with the given name exists.

        Notes
        -----
        This operation is irreversible. All observations associated with
        the project are deleted via CASCADE.

        Examples
        --------
        >>> folio.delete_project("old_experiment")
        """
        if self._recommenders.get(name) is not None:
            self._recommenders.pop(name)
        database.delete_project(name, self.db_path)

    def add_observation(
        self,
        project_name: str,
        inputs: dict[str, float | str],
        outputs: dict[str, float],
        tag: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Record a new observation for a project.

        Validates inputs and outputs against the project schema, then stores
        the observation in the database with the current timestamp.

        Parameters
        ----------
        project_name : str
            Name of the project to add the observation to.
        inputs : dict[str, float | str]
            Input values used in the experiment. Keys must match the project's
            InputSpec names. Values must be within bounds (continuous) or in
            valid levels (categorical).
        outputs : dict[str, float]
            Measured output values. Keys must match the project's OutputSpec names.
            All values must be numeric.
        tag : str | None, optional
            Category tag for grouping observations (e.g., "screening", "optimization").
        notes : str | None, optional
            Free-form notes about this experiment.

        Raises
        ------
        ProjectNotFoundError
            If no project with the given name exists.
        InvalidInputError
            If inputs are missing, extra, or have values outside valid ranges.
        InvalidOutputError
            If outputs are missing, extra, or have non-numeric values.

        Examples
        --------
        >>> folio.add_observation(
        ...     project_name="yield_optimization",
        ...     inputs={"temperature": 80.0, "solvent": "ethanol"},
        ...     outputs={"yield": 92.5, "purity": 98.0},
        ...     tag="optimization",
        ...     notes="Excellent result, consider replicating",
        ... )
        """
        project = database.get_project(project_name, self.db_path)
        project_id = project.id
        project.validate_inputs(inputs)
        project.validate_outputs(outputs)
        obs = Observation(
            project_id=project_id, inputs=inputs, outputs=outputs, tag=tag, notes=notes
        )
        database.add_observation(obs, self.db_path)

    def delete_observation(self, observation_id: int) -> None:
        """Delete an observation by its database ID.

        Parameters
        ----------
        observation_id : int
            Database ID of the observation to delete.

        Raises
        ------
        ValueError
            If no observation with the given ID exists.

        Examples
        --------
        >>> folio.delete_observation(42)
        """
        database.delete_observation(observation_id, self.db_path)

    def suggest(self, project_name: str) -> list[dict[str, float]]:
        """Get suggested next experiment inputs for a project.

        Uses the project's recommender to suggest the next experiment based
        on all recorded observations. For Bayesian optimization, this fits
        a surrogate model and optimizes an acquisition function.

        Parameters
        ----------
        project_name : str
            Name of the project to get suggestions for.

        Returns
        -------
        list[dict[str, float]]
            List of suggested input configurations. Each dict maps input names
            to suggested values. Currently returns a single suggestion, but
            the list format supports batch recommendations in the future.

        Raises
        ------
        ProjectNotFoundError
            If no project with the given name exists.

        Notes
        -----
        - If no observations exist, returns a random sample within bounds
        - If fewer observations than `n_initial` in RecommenderConfig, returns
          random samples (exploration phase)
        - Failed observations are excluded from model training

        Examples
        --------
        >>> suggestions = folio.suggest("yield_optimization")
        >>> next_experiment = suggestions[0]
        >>> print(next_experiment)
        {'temperature': 85.2, 'pressure': 3.7}
        """
        project = self.get_project(project_name)
        if self._recommenders.get(project_name) is None:
            self._recommenders[project_name] = self._build_recommender(project)
        rec = self._recommenders[project_name]
        obs = self.get_observations(project_name)
        suggestions = rec.recommend(obs)
        return [suggestions]

    def get_project(self, name: str) -> Project:
        """Get a project by name.

        Parameters
        ----------
        name : str
            Name of the project to retrieve.

        Returns
        -------
        Project
            The requested project object with full schema information.

        Raises
        ------
        ProjectNotFoundError
            If no project with the given name exists.

        Examples
        --------
        >>> project = folio.get_project("yield_optimization")
        >>> print(project.inputs)
        [InputSpec(name='temperature', type='continuous', bounds=(20.0, 100.0))]
        """
        return database.get_project(name, self.db_path)

    def get_observations(
        self,
        project_name: str,
        tag: str | None = None,
    ) -> list[Observation]:
        """Get observations for a project, optionally filtered by tag.

        Parameters
        ----------
        project_name : str
            Name of the project to get observations for.
        tag : str | None, optional
            If provided, only return observations with this tag.
            If None, return all observations.

        Returns
        -------
        list[Observation]
            Observations for the project, sorted chronologically by timestamp.

        Raises
        ------
        ProjectNotFoundError
            If no project with the given name exists.

        Examples
        --------
        Get all observations:

        >>> observations = folio.get_observations("yield_optimization")

        Get only screening observations:

        >>> screening = folio.get_observations("yield_optimization", tag="screening")
        """
        project = database.get_project(project_name, self.db_path)
        project_id = project.id
        observations = database.get_observations(project_id, self.db_path)
        if tag is not None:
            return [obs for obs in observations if obs.tag == tag]
        return observations

    def get_recommender(self, project_name: str) -> Recommender | None:
        """Get the cached recommender for a project.

        Returns the recommender instance if one has been built for this project
        (via `load_project()` or `suggest()`). Returns None if no recommender
        has been cached yet.

        Parameters
        ----------
        project_name : str
            Name of the project.

        Returns
        -------
        Recommender | None
            The cached recommender instance, or None if not yet built.
            Access the fitted surrogate via `recommender.surrogate` for
            BayesianRecommender.

        Notes
        -----
        This method does not create a recommender if one doesn't exist.
        Use `suggest()` to trigger recommender creation.

        Examples
        --------
        >>> folio.suggest("yield_optimization")  # Builds recommender
        >>> recommender = folio.get_recommender("yield_optimization")
        >>> if recommender is not None:
        ...     mean, std = recommender.surrogate.predict(X_test)
        """
        return self._recommenders.get(project_name)

    def _build_recommender(self, project: Project) -> Recommender:
        """Build a recommender instance for a project.

        Creates the appropriate recommender based on the project's
        recommender_config. This is an internal method used by `load_project()`
        and `suggest()`.

        Parameters
        ----------
        project : Project
            The project to build a recommender for.

        Returns
        -------
        Recommender
            A configured recommender instance (e.g., BayesianRecommender,
            RandomRecommender) ready to generate suggestions.

        Notes
        -----
        The recommender type is determined by project.recommender_config.type:
        - "bayesian": BayesianRecommender with configured surrogate/acquisition
        - "random": RandomRecommender for uniform sampling
        """
        rec_config = project.recommender_config
        if rec_config.type == "bayesian":
            rec = BayesianRecommender(project)
        elif rec_config.type == "random":
            rec = RandomRecommender(project)
        else:
            raise ValueError(f"unknown recommender: {rec_config.type}")
        return rec
