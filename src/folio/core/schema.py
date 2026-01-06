from dataclasses import dataclass

from folio.exceptions import InvalidSchemaError


@dataclass
class InputSpec:
    name: str
    type: str  # "continuous" or "categorical"
    bounds: tuple[float, float] | None = None
    levels: list[str] | None = None
    units: str | None = None

    def __post_init__(self):
        if self.type == "continuous":
            if self.bounds is None:
                raise InvalidSchemaError(
                    f"Continuous input '{self.name}' requires bounds."
                )
            if self.bounds[0] >= self.bounds[1]:
                raise InvalidSchemaError(
                    f"Continuous input '{self.name}' has invalid bounds: "
                    f"lower ({self.bounds[0]}) must be less than upper ({self.bounds[1]})."
                )
        elif self.type == "categorical":
            if self.levels is None or len(self.levels) < 2:
                raise InvalidSchemaError(
                    f"Categorical input '{self.name}' requires at least 2 levels."
                )


@dataclass
class OutputSpec:
    name: str
    units: str | None = None


@dataclass
class ConstantSpec:
    name: str
    value: float | str
    units: str | None = None
