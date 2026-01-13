"""Human-in-the-loop executor for manual experiment entry."""

from typing import Callable

from folio.core.observation import Observation
from folio.core.project import Project
from folio.executors.base import Executor


class HumanExecutor(Executor):
    """Executor that prompts a human to run experiments and enter results.

    Displays suggested inputs to the user and collects output values via
    interactive prompts. Supports marking experiments as failed.

    Parameters
    ----------
    input_func : Callable[[str], str] | None, optional
        Function for collecting user input. Defaults to builtin `input()`.
        Override for testing or custom UI integration.

    Examples
    --------
    >>> executor = HumanExecutor()
    >>> obs = executor.execute({"temperature": 80.0}, project)
    # User is prompted for outputs interactively
    """

    def __init__(self, input_func: Callable[[str], str] | None = None):
        ...

    def _run(self, suggestion: dict, project: Project) -> Observation:
        """Prompt user to run experiment and enter results.

        Displays the suggested inputs, then prompts for each output value.
        Also asks if the experiment failed.

        Parameters
        ----------
        suggestion : dict
            Validated input values to display to user.
        project : Project
            Project defining expected outputs.

        Returns
        -------
        Observation
            Observation with user-entered outputs and failed flag.
        """
        ...
