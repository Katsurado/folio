"""Claude-Light executor for autonomous closed-loop experiments."""

import ast
import getpass
import hashlib
import inspect
import socket

import requests

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

    def _get_class_hash(self):
        source = inspect.getsource(self.__class__)
        nomalized = ast.unparse(ast.parse(source))
        return nomalized, hashlib.sha256(nomalized.encode()).hexdigest()

    def __init__(self, api_url: str = DEFAULT_API_URL):
        self.api_url = api_url

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
        r_suggested = suggestion["R"]
        g_suggested = suggestion["G"]
        b_suggested = suggestion["B"]

        func, version = self._get_class_hash()
        user = getpass.getuser()
        hostname = socket.gethostname()

        res = requests.get(
            self.api_url, params={"R": r_suggested, "G": g_suggested, "B": b_suggested}
        ).json()

        inputs = {"R": res["in"][0], "G": res["in"][1], "B": res["in"][2]}

        metadata = {
            "func": func,
            "version": version,
            "user": user,
            "hostname": hostname,
        }

        outputs = res["out"]
        project_id = project.id

        obs = Observation(
            project_id=project_id,
            inputs=inputs,
            outputs=outputs,
            inputs_suggested=suggestion,
            metadata=metadata,
        )

        return obs
