"""High-level API for Folio electronic lab notebook."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from folio.core import database
from folio.core.config import RecommenderConfig, TargetConfig
from folio.core.database import DEFAULT_DB_PATH
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.exceptions import ExecutorError
from folio.executors import ClaudeLightExecutor, Executor, HumanExecutor
from folio.recommenders import BayesianRecommender, RandomRecommender, Recommender
from folio.recommenders.initializer import LLMBackend, LLMInitializer, OpenAIBackend

logger = logging.getLogger(__name__)

# Registry mapping executor names to their classes
_EXECUTOR_REGISTRY: dict[str, type[Executor]] = {
    "human": HumanExecutor,
    "claude_light": ClaudeLightExecutor,
}


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

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        sync_url: str | None = None,
        auth_token: str | None = None,
    ) -> None:
        """Initialize Folio with database path and optional cloud sync.

        Parameters
        ----------
        db_path : Path | str, optional
            Path to the SQLite database file. Defaults to ~/.folio/folio.db.
            Will be converted to Path if string is provided.
        sync_url : str, optional
            libSQL sync URL for cloud synchronization (e.g., libsql://your-db.turso.io).
            If provided, enables cloud sync for collaborative workflows.
        auth_token : str, optional
            Authentication token for libSQL cloud sync. Required if sync_url
            is provided.
        """
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self.sync_url = sync_url
        self.auth_token = auth_token
        self._recommenders: dict[str, Recommender] = {}
        self.executor: Executor | None = None
        database.init_db(self.db_path, sync_url=sync_url, auth_token=auth_token)

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

        database.create_project(
            project, self.db_path, sync_url=self.sync_url, auth_token=self.auth_token
        )

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
        return database.list_projects(
            self.db_path, sync_url=self.sync_url, auth_token=self.auth_token
        )

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
        database.delete_project(
            name, self.db_path, sync_url=self.sync_url, auth_token=self.auth_token
        )

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
        project = database.get_project(
            project_name,
            self.db_path,
            sync_url=self.sync_url,
            auth_token=self.auth_token,
        )
        project_id = project.id
        project.validate_inputs(inputs)
        project.validate_outputs(outputs)
        obs = Observation(
            project_id=project_id, inputs=inputs, outputs=outputs, tag=tag, notes=notes
        )
        database.add_observation(
            obs, self.db_path, sync_url=self.sync_url, auth_token=self.auth_token
        )

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
        database.delete_observation(
            observation_id,
            self.db_path,
            sync_url=self.sync_url,
            auth_token=self.auth_token,
        )

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
        return database.get_project(
            name, self.db_path, sync_url=self.sync_url, auth_token=self.auth_token
        )

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
        project = database.get_project(
            project_name,
            self.db_path,
            sync_url=self.sync_url,
            auth_token=self.auth_token,
        )
        project_id = project.id
        observations = database.get_observations(
            project_id,
            self.db_path,
            sync_url=self.sync_url,
            auth_token=self.auth_token,
        )
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

    def build_executor(self, executor_name: str) -> Executor:
        """Build and cache an executor by name.

        Creates an executor instance based on the provided name and caches it
        in `self.executor` for use by `execute()`.

        Parameters
        ----------
        executor_name : str
            Name of the executor to build. Supported values:
            - "human": HumanExecutor for manual experiment entry
            - "claude_light": ClaudeLightExecutor for autonomous execution

        Returns
        -------
        Executor
            The created executor instance, also stored in `self.executor`.

        Raises
        ------
        ValueError
            If the executor name is not recognized.

        Examples
        --------
        >>> folio = Folio()
        >>> executor = folio.build_executor("human")
        >>> print(type(executor))
        <class 'folio.executors.human.HumanExecutor'>

        >>> folio.build_executor("claude_light")
        >>> folio.execute("my_project", n_iter=5)  # Uses cached executor
        """
        try:
            cls = _EXECUTOR_REGISTRY[executor_name]
        except KeyError:
            raise ValueError(f"Unknown executor: {executor_name}")

        self.executor = cls()
        return self.executor

    def execute(
        self,
        project_name: str,
        n_iter: int = 1,
        stop_on_error: bool = True,
        wait_between_runs: float = 0.0,
        executor: Executor | None = None,
    ) -> list[Observation]:
        """Run an automated experiment loop.

        Executes a closed-loop optimization workflow: for each iteration,
        gets a suggestion from the recommender, runs the experiment via the
        executor, and records the observation.

        Parameters
        ----------
        project_name : str
            Name of the project to run experiments for.
        n_iter : int, optional
            Number of experiment iterations to run. Defaults to 1.
        stop_on_error : bool, optional
            If True (default), re-raise any execution errors immediately.
            If False, log the error and continue with remaining iterations.
        wait_between_runs : float, optional
            Time in seconds to wait between iterations. Defaults to 0.0.
            Useful for rate-limiting or allowing system cooldown.
        executor : Executor | None, optional
            Executor to use for running experiments. If None, uses the
            executor cached in `self.executor` (set via `build_executor()`).

        Returns
        -------
        list[Observation]
            List of observations created during the execution loop.
            On partial failure (stop_on_error=False), returns only successful
            observations.

        Raises
        ------
        ExecutorError
            If no executor is provided and self.executor is None.
        ExecutorError
            If experiment execution fails (when stop_on_error=True).
        ProjectNotFoundError
            If the project doesn't exist.

        Examples
        --------
        Run 5 iterations with the cached executor:

        >>> folio.build_executor("human")
        >>> observations = folio.execute("my_project", n_iter=5)
        >>> print(f"Completed {len(observations)} experiments")

        Run with a specific executor and error handling:

        >>> from folio.executors import ClaudeLightExecutor
        >>> executor = ClaudeLightExecutor(api_url="http://localhost:8000")
        >>> observations = folio.execute(
        ...     "my_project",
        ...     n_iter=10,
        ...     stop_on_error=False,
        ...     wait_between_runs=1.0,
        ...     executor=executor,
        ... )

        Notes
        -----
        The execution loop follows this pattern for each iteration:
        1. Call `suggest()` to get recommended inputs
        2. Call `executor.execute()` with the suggestion
        3. Call `add_observation()` to record the result
        4. Sleep for `wait_between_runs` seconds (if > 0)
        """
        if self.executor is None and executor is None:
            raise ExecutorError("Executor is not configured")

        project = self.get_project(project_name)

        active_executor = self.executor if executor is None else executor

        observations = []

        for iteration in range(n_iter):
            suggestion = self.suggest(project_name)[0]
            try:
                next_obs = active_executor.execute(suggestion, project)
            except ExecutorError as e:
                if stop_on_error:
                    raise ExecutorError(f"{e}")
                else:
                    continue
            database.add_observation(
                next_obs,
                self.db_path,
                sync_url=self.sync_url,
                auth_token=self.auth_token,
            )
            observations.append(next_obs)
            time.sleep(wait_between_runs)

        return observations

    def initialize_from_llm(
        self,
        project_name: str,
        n: int = 5,
        description: str | None = None,
        backend: LLMBackend | None = None,
        prompt_template: str | Path | None = None,
        max_cost_per_call: float = 0.50,
    ) -> list[dict]:
        """Suggest initial experiments using LLM with literature search.

        Uses a large language model with web search capability to suggest
        initial experiments based on scientific literature and best practices.
        This is useful for starting an optimization campaign with informed
        initial conditions rather than random sampling.

        Parameters
        ----------
        project_name : str
            Name of the project to initialize.
        n : int, optional
            Number of initial experiments to suggest. Defaults to 5.
        description : str | None, optional
            Natural language description providing additional context for
            the LLM (e.g., reaction type, constraints, prior knowledge).
        backend : LLMBackend | None, optional
            LLM backend to use. Defaults to OpenAIBackend() which uses
            the OPENAI_API_KEY environment variable.
        prompt_template : str | Path | None, optional
            Custom prompt template. If None, uses the default template.
            If Path, loads template from file. If str, uses directly.
        max_cost_per_call : float, optional
            Maximum allowed cost per API call in USD. Defaults to 0.50.
            Raises CostLimitError if estimated cost exceeds this limit.

        Returns
        -------
        list[dict]
            List of suggested input configurations. Each dict maps input
            names to suggested values within the project's bounds.

        Raises
        ------
        ProjectNotFoundError
            If no project with the given name exists.
        CostLimitError
            If estimated API cost exceeds max_cost_per_call.
        LLMResponseError
            If LLM response cannot be parsed as valid suggestions.
        ValueError
            If OpenAI API key is not configured (when using default backend).

        Examples
        --------
        Basic usage with default OpenAI backend:

        >>> folio = Folio("my_experiments.db")
        >>> suggestions = folio.initialize_from_llm("polymer_optimization", n=5)
        >>> for s in suggestions:
        ...     result = run_experiment(s)
        ...     folio.add_observation("polymer_optimization", inputs=s, outputs=result)

        With custom description for better context:

        >>> suggestions = folio.initialize_from_llm(
        ...     "suzuki_coupling",
        ...     n=3,
        ...     description="Pd-catalyzed cross-coupling for biaryl synthesis. "
        ...                 "Using aryl bromide and phenylboronic acid.",
        ... )

        With custom prompt template:

        >>> suggestions = folio.initialize_from_llm(
        ...     "my_project",
        ...     n=5,
        ...     prompt_template=Path("custom_prompt.txt"),
        ...     max_cost_per_call=1.00,
        ... )

        Notes
        -----
        - Requires OPENAI_API_KEY environment variable for default backend
        - Web search is enabled by default for literature-informed suggestions
        - Cost estimation is performed before API call to prevent unexpected charges
        - Suggested values are validated against project schema and clamped to bounds
        """
        project = self.get_project(project_name)

        if backend is None:
            backend = OpenAIBackend()

        initializer = LLMInitializer(
            backend=backend,
            prompt_template=prompt_template,
            max_cost_per_call=max_cost_per_call,
        )

        # Print model and estimated cost before making API call
        prompt = initializer._build_prompt(project, n, description, existing=None)
        estimated_cost = backend.estimate_cost(prompt)
        model_name = getattr(backend, "model", "unknown")
        print(f"Model: {model_name}, Estimated cost: ${estimated_cost:.4f}")

        return initializer.suggest(project, n=n, description=description)
