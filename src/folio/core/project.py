"""Project definition and validation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from folio.core.schema import InputSpec, OutputSpec
from folio.exceptions import InvalidInputError, InvalidOutputError, InvalidSchemaError
from folio.targets import (
    DifferenceTarget,
    DirectTarget,
    RatioTarget,
    SlopeTarget,
)

if TYPE_CHECKING:
    from folio.core.observation import Observation


@dataclass
class TargetConfig:
    """Configuration for the optimization target."""

    name: str
    mode: str = "maximize"
    target_type: Literal["direct", "ratio", "difference", "slope"] = "direct"

    # used by ratio target
    numerator: str | None = None
    denominator: str | None = None

    # used by difference target
    first: str | None = None
    second: str | None = None

    # used by slope target
    slope_outputs: list[str] | None = None
    slope_x: list[float] | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("maximize", "minimize"):
            raise InvalidSchemaError(
                f"Target mode must be 'maximize' or 'minimize', got '{self.mode}'"
            )


@dataclass
class RecommenderConfig:
    """Configuration for the recommender."""

    type: str = "bayesian"
    # Surrogate model type for Bayesian optimization
    surrogate: str = "gp"
    # Acquisition function type
    acquisition: str = "ei"
    # Number of initial random samples before using surrogate
    n_initial: int = 5
    # Additional kwargs passed to recommender
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Project:
    """Experiment schema defining inputs, outputs, target, and recommender."""

    id: int | None
    name: str
    inputs: list[InputSpec]
    outputs: list[OutputSpec]
    target_config: TargetConfig
    recommender_config: RecommenderConfig = field(default_factory=RecommenderConfig)

    def __post_init__(self) -> None:
        self._validate_schema()

    def _validate_schema(self) -> None:
        """Validate project schema at construction."""
        if not self.name:
            raise InvalidSchemaError("Project name cannot be empty")
        if not self.inputs:
            raise InvalidSchemaError("Project must have at least one input")
        if not self.outputs:
            raise InvalidSchemaError("Project must have at least one output")

        # Check for duplicate input names
        input_names = [inp.name for inp in self.inputs]
        if len(input_names) != len(set(input_names)):
            raise InvalidSchemaError("Duplicate input names are not allowed")

        # Check for duplicate output names
        output_names = [out.name for out in self.outputs]
        if len(output_names) != len(set(output_names)):
            raise InvalidSchemaError("Duplicate output names are not allowed")

        # Validate each InputSpec
        for inp in self.inputs:
            if inp.type == "continuous":
                if inp.bounds is None:
                    raise InvalidSchemaError(
                        f"Continuous input '{inp.name}' must have bounds"
                    )
                if inp.bounds[0] >= inp.bounds[1]:
                    raise InvalidSchemaError(
                        f"Input '{inp.name}' bounds must have lower < upper, "
                        f"got {inp.bounds}"
                    )
            elif inp.type == "categorical":
                if not inp.levels:
                    raise InvalidSchemaError(
                        f"Categorical input '{inp.name}' must have levels"
                    )
            else:
                raise InvalidSchemaError(
                    f"Input '{inp.name}' type must be 'continuous' or 'categorical', "
                    f"got '{inp.type}'"
                )

        # Validate target references a valid output for direct targets
        if self.target_config.target_type == "direct":
            output_names_set = set(output_names)
            if self.target_config.name not in output_names_set:
                raise InvalidSchemaError(
                    f"Target '{self.target_config.name}' not in outputs. "
                    f"Available: {output_names}"
                )

    def validate_inputs(self, inputs: dict[str, float | str]) -> None:
        """Validate input values against schema."""
        expected = {inp.name for inp in self.inputs}
        provided = set(inputs.keys())

        missing = expected - provided
        if missing:
            raise InvalidInputError(f"Missing inputs: {missing}")

        extra = provided - expected
        if extra:
            raise InvalidInputError(f"Unexpected inputs: {extra}")

        for inp in self.inputs:
            value = inputs[inp.name]
            if inp.type == "continuous":
                if not isinstance(value, int | float):
                    raise InvalidInputError(
                        f"Input '{inp.name}' must be numeric, "
                        f"got {type(value).__name__}"
                    )
                if not inp.bounds[0] <= value <= inp.bounds[1]:
                    raise InvalidInputError(
                        f"Input '{inp.name}' value {value} outside bounds {inp.bounds}"
                    )
            elif inp.type == "categorical":
                if value not in inp.levels:
                    raise InvalidInputError(
                        f"Input '{inp.name}' value '{value}' not in levels {inp.levels}"
                    )

    def validate_outputs(self, outputs: dict[str, float]) -> None:
        """Validate output values against schema."""
        expected = {out.name for out in self.outputs}
        provided = set(outputs.keys())

        missing = expected - provided
        if missing:
            raise InvalidOutputError(f"Missing outputs: {missing}")

        extra = provided - expected
        if extra:
            raise InvalidOutputError(f"Unexpected outputs: {extra}")

        for out in self.outputs:
            value = outputs[out.name]
            if not isinstance(value, int | float):
                raise InvalidOutputError(
                    f"Output '{out.name}' must be numeric, got {type(value).__name__}"
                )

    def get_input_names(self) -> list[str]:
        """Return list of input names."""
        return [inp.name for inp in self.inputs]

    def get_output_names(self) -> list[str]:
        """Return list of output names."""
        return [out.name for out in self.outputs]

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return bounds for continuous inputs, in order."""
        return [inp.bounds for inp in self.inputs if inp.type == "continuous"]

    def get_target(self) -> DirectTarget | RatioTarget | DifferenceTarget | SlopeTarget:
        """Return the appropriate target instance based on target_config."""
        config = self.target_config
        if config.target_type == "direct":
            return DirectTarget(config.name, config.mode)
        elif config.target_type == "ratio":
            return RatioTarget(config.numerator, config.denominator, config.mode)
        elif config.target_type == "difference":
            return DifferenceTarget(config.first, config.second, config.mode)
        elif config.target_type == "slope":
            return SlopeTarget(config.slope_outputs, config.slope_x, config.mode)
        else:
            raise ValueError(f"Unknown target type: {config.target_type}")

    def get_training_data(
        self, observations: list["Observation"]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract training data from observations."""
        target = self.get_target()
        input_names = [inp.name for inp in self.inputs]
        X_rows = []
        y_values = []
        for obs in observations:
            if obs.failed:
                continue
            y = target.compute(obs)
            if y is None:
                continue
            row = [obs.inputs[name] for name in input_names]
            X_rows.append(row)
            y_values.append(y)
        return np.array(X_rows), np.array(y_values)
