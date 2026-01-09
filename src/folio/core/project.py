"""Project definition and validation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from folio.core.config import RecommenderConfig, TargetConfig
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
class Project:
    """Experiment schema defining inputs, outputs, target, and recommender.

    A Project defines the structure of an experiment series: what inputs can be
    varied, what outputs are measured, what target to optimize, and how to suggest
    new experiments. Projects are persisted to the database and can be retrieved
    by name.

    Parameters
    ----------
    id : int | None
        Database ID assigned after persistence. None for new projects.
    name : str
        Unique project name. Cannot be empty.
    inputs : list[InputSpec]
        Input variable specifications. Must have at least one input.
        Names must be unique within the project.
    outputs : list[OutputSpec]
        Output variable specifications. Must have at least one output.
        Names must be unique within the project.
    target_config : TargetConfig
        Configuration for the optimization target.
    recommender_config : RecommenderConfig, optional
        Configuration for the experiment recommender. Defaults to Bayesian
        optimization with GP surrogate and EI acquisition.

    Raises
    ------
    InvalidSchemaError
        If name is empty, no inputs/outputs defined, duplicate names exist,
        input bounds are invalid, or target references a non-existent output.

    Examples
    --------
    >>> project = Project(
    ...     id=None,
    ...     name="yield_optimization",
    ...     inputs=[
    ...         InputSpec("temperature", "continuous", bounds=(20.0, 100.0)),
    ...         InputSpec("solvent", "categorical", levels=["water", "ethanol"]),
    ...     ],
    ...     outputs=[OutputSpec("yield"), OutputSpec("purity")],
    ...     target_config=TargetConfig("yield", mode="maximize"),
    ... )
    """

    id: int | None
    name: str
    inputs: list[InputSpec]
    outputs: list[OutputSpec]
    target_config: TargetConfig
    recommender_config: RecommenderConfig = field(default_factory=RecommenderConfig)

    def __post_init__(self) -> None:
        self._validate_schema()

    def _validate_schema(self) -> None:
        """Validate project schema at construction.

        Checks that the project has a non-empty name, at least one input and
        output, no duplicate names, valid input bounds/levels, and that direct
        targets reference existing outputs.

        Raises
        ------
        InvalidSchemaError
            If any validation check fails.
        """
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
        """Validate input values against the project schema.

        Checks that all required inputs are provided, no extra inputs are given,
        and each value is valid for its input type (within bounds for continuous,
        in valid levels for categorical).

        Parameters
        ----------
        inputs : dict[str, float | str]
            Input values to validate. Keys are input names, values are numeric
            (for continuous) or string (for categorical).

        Raises
        ------
        InvalidInputError
            If required inputs are missing, unexpected inputs are provided,
            numeric values are outside bounds, or categorical values are not
            in the valid levels.
        """
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
        """Validate output values against the project schema.

        Checks that all required outputs are provided, no extra outputs are given,
        and all values are numeric.

        Parameters
        ----------
        outputs : dict[str, float]
            Output values to validate. Keys are output names, values are numeric.

        Raises
        ------
        InvalidOutputError
            If required outputs are missing, unexpected outputs are provided,
            or values are not numeric.
        """
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
        """Return list of input variable names in definition order.

        Returns
        -------
        list[str]
            Names of all input variables.
        """
        return [inp.name for inp in self.inputs]

    def get_output_names(self) -> list[str]:
        """Return list of output variable names in definition order.

        Returns
        -------
        list[str]
            Names of all output variables.
        """
        return [out.name for out in self.outputs]

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return bounds for continuous inputs in definition order.

        Categorical inputs are excluded since they don't have numeric bounds.

        Returns
        -------
        list[tuple[float, float]]
            List of (lower, upper) bound tuples for each continuous input.
        """
        return [inp.bounds for inp in self.inputs if inp.type == "continuous"]

    def get_optimization_bounds(self) -> np.ndarray:
        """Return bounds in BoTorch optimize_acqf format.

        Returns bounds as a 2D array suitable for BoTorch's optimize_acqf function.
        Row 0 contains lower bounds, row 1 contains upper bounds for each
        continuous input dimension.

        Returns
        -------
        np.ndarray, shape (2, d)
            Bounds array where d is the number of continuous inputs.
            bounds[0, :] are lower bounds, bounds[1, :] are upper bounds.

        Examples
        --------
        >>> project = Project(
        ...     id=None,
        ...     name="example",
        ...     inputs=[
        ...         InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
        ...         InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
        ...     ],
        ...     outputs=[OutputSpec("y")],
        ...     target_config=TargetConfig("y"),
        ... )
        >>> project.get_optimization_bounds()
        array([[ 0., -5.],
               [10.,  5.]])

        Notes
        -----
        Only continuous inputs are included. Categorical inputs are excluded
        since they don't have numeric bounds suitable for gradient-based
        optimization.
        """
        bounds = self.get_bounds()
        num_features = len(bounds)
        opt_bounds = np.zeros((2, num_features))

        for feature in range(num_features):
            opt_bounds[0, feature] = bounds[feature][0]
            opt_bounds[1, feature] = bounds[feature][1]

        return opt_bounds

    def get_target(self) -> DirectTarget | RatioTarget | DifferenceTarget | SlopeTarget:
        """Create the appropriate target instance based on target_config.

        Returns
        -------
        DirectTarget | RatioTarget | DifferenceTarget | SlopeTarget
            Target instance configured according to target_config.

        Raises
        ------
        ValueError
            If target_config.target_type is not a recognized type.
        """
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
        """Extract training data arrays from observations.

        Converts observations into (X, y) arrays suitable for model training.
        Automatically filters out failed observations and those with missing
        target values.

        Parameters
        ----------
        observations : list[Observation]
            Observations to extract training data from.

        Returns
        -------
        X : np.ndarray, shape (n_valid, n_inputs)
            Input values for valid observations. Columns are in input definition order.
        y : np.ndarray, shape (n_valid,)
            Target values computed from valid observations.

        Notes
        -----
        Observations are excluded from training data if:
        - The observation is marked as failed (obs.failed is True)
        - The target value cannot be computed (returns None)
        """
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
