"""Abstract base class for experiment executors."""

from abc import ABC, abstractmethod

from folio.core.observation import Observation
from folio.core.project import Project
from folio.exceptions import ExecutorError


class Executor(ABC):
    """Abstract base class for running experiments.

    Executors take suggested input values and produce observations by running
    experiments (either manually or automatically). The base class handles
    input validation; subclasses implement the actual experiment execution.

    Subclasses must implement `_run()` to define how experiments are executed.
    """

    def execute(self, suggestion: dict, project: Project) -> Observation:
        """Execute an experiment with the given inputs.

        Validates inputs against the project schema, then delegates to `_run()`
        for the actual execution.

        Parameters
        ----------
        suggestion : dict
            Input values for the experiment. Keys are input names, values are
            numeric (for continuous) or string (for categorical) inputs.
        project : Project
            Project defining the experiment schema.

        Returns
        -------
        Observation
            The resulting observation with id=None (database assigns on persist).

        Raises
        ------
        InvalidInputError
            If suggestion fails validation against project schema.
        ExecutorError
            If the experiment execution fails.
        """
        project.validate_inputs(suggestion)
        try:
            return self._run(suggestion, project)
        except Exception as e:
            raise ExecutorError(f"Execution failed: {e}") from e

    @abstractmethod
    def _run(self, suggestion: dict, project: Project) -> Observation:
        """Execute the experiment (subclass implementation).

        Parameters
        ----------
        suggestion : dict
            Validated input values for the experiment.
        project : Project
            Project defining the experiment schema.

        Returns
        -------
        Observation
            The resulting observation with id=None and project_id from project.
        """
        ...
