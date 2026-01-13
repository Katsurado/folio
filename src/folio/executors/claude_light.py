"""Claude-Light executor for autonomous closed-loop experiments."""

from folio.core.observation import Observation
from folio.core.project import Project
from folio.executors.base import Executor


class ClaudeLightExecutor(Executor):
    """Executor that runs experiments via Claude-Light API.

    Sends input values to the Claude-Light API endpoint and parses the
    response into an Observation. Used for fully autonomous closed-loop
    optimization.

    Parameters
    ----------
    api_url : str, optional
        Base URL for the Claude-Light API.
        Defaults to "https://claude-light.cheme.cmu.edu/api".

    Examples
    --------
    >>> executor = ClaudeLightExecutor()
    >>> obs = executor.execute({"R": 255, "G": 128, "B": 64}, project)
    """

    DEFAULT_API_URL = "https://claude-light.cheme.cmu.edu/api"

    def __init__(self, api_url: str | None = None):
        ...

    def _run(self, suggestion: dict, project: Project) -> Observation:
        """Call Claude-Light API to run experiment.

        Parameters
        ----------
        suggestion : dict
            Validated input values to send to API.
        project : Project
            Project defining expected outputs.

        Returns
        -------
        Observation
            Observation with outputs from API response.

        Raises
        ------
        ExecutorError
            If API request fails or returns invalid response.
        """
        ...
