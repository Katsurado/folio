from dataclasses import dataclass, field
from datetime import datetime

from folio.exceptions import InvalidInputError, InvalidOutputError


@dataclass
class Observation:
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
