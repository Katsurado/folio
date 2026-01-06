"""Observation dataclass for storing experiment results."""

from dataclasses import dataclass, field
from datetime import datetime

from folio.exceptions import InvalidInputError, InvalidOutputError


@dataclass
class Observation:
    """A single experiment observation recording inputs, outputs, and metadata.

    Observations are the fundamental data unit in Folio. Each observation captures
    the experimental conditions (inputs), measured results (outputs), and optional
    metadata like notes, tags, and links to raw data files.

    Parameters
    ----------
    project_id : int
        ID of the project this observation belongs to. Must be a positive integer.
    inputs : dict[str, float | str]
        Input variable values used in this experiment. Keys are input names,
        values are numeric (for continuous) or string (for categorical) inputs.
    outputs : dict[str, float]
        Measured output values from this experiment. Keys are output names,
        values are numeric measurements.
    timestamp : datetime, optional
        When the observation was recorded. Defaults to current time.
    id : int | None, optional
        Database ID assigned after persistence. None for new observations.
    notes : str | None, optional
        Free-form notes about this experiment (e.g., unusual observations).
    tag : str | None, optional
        Category tag for grouping observations (e.g., "screening", "optimization").
    raw_data_path : str | None, optional
        Path to raw data files associated with this observation.
    failed : bool, optional
        Whether this experiment failed. Failed observations are excluded from
        model training but kept for documentation. Defaults to False.

    Raises
    ------
    InvalidInputError
        If project_id is not a positive integer, inputs is not a dict,
        or timestamp is not a datetime.
    InvalidOutputError
        If outputs is not a dict.

    Examples
    --------
    >>> obs = Observation(
    ...     project_id=1,
    ...     inputs={"temperature": 80.0, "solvent": "ethanol"},
    ...     outputs={"yield": 85.2, "purity": 99.1},
    ...     notes="Reaction completed faster than expected",
    ... )
    """

    project_id: int
    inputs: dict[str, float | str]
    outputs: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    id: int | None = None
    notes: str | None = None
    tag: str | None = None
    raw_data_path: str | None = None
    failed: bool = False

    def __post_init__(self):
        if not isinstance(self.project_id, int) or self.project_id < 1:
            raise InvalidInputError(
                f"project_id must be a positive integer, got {self.project_id!r}."
            )
        if not isinstance(self.inputs, dict):
            raise InvalidInputError(
                f"inputs must be a dict, got {type(self.inputs).__name__}."
            )
        if not isinstance(self.outputs, dict):
            raise InvalidOutputError(
                f"outputs must be a dict, got {type(self.outputs).__name__}."
            )
        if not isinstance(self.timestamp, datetime):
            raise InvalidInputError(
                f"timestamp must be a datetime, got {type(self.timestamp).__name__}."
            )
