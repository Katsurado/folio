"""Schema definitions for experiment inputs, outputs, and constants."""

from dataclasses import dataclass

from folio.exceptions import InvalidSchemaError


@dataclass
class InputSpec:
    """Specification for a single experiment input variable.

    Defines the name, type, and constraints for an input variable. Inputs can be
    either continuous (numeric with bounds) or categorical (discrete levels).

    Parameters
    ----------
    name : str
        Unique identifier for this input variable.
    type : str
        Either "continuous" for numeric inputs or "categorical" for discrete inputs.
    bounds : tuple[float, float] | None, optional
        Lower and upper bounds for continuous inputs as (min, max). Required if
        type is "continuous". Ignored for categorical inputs.
    levels : list[str] | None, optional
        Valid categories for categorical inputs. Required if type is "categorical"
        and must contain at least 2 levels. Ignored for continuous inputs.
    units : str | None, optional
        Physical units for the input (e.g., "mL", "°C"). For display purposes only.
    optimizable : bool, optional
        If True (default), this input is optimized over during acquisition function
        optimization. If False, this is a context variable: it is recorded and
        included in GP fitting, but held fixed during optimization. Use for
        non-controllable inputs like time-of-day or ambient conditions.

    Raises
    ------
    InvalidSchemaError
        If continuous input is missing bounds or has invalid bounds (lower >= upper).
        If categorical input is missing levels or has fewer than 2 levels.

    Examples
    --------
    >>> temp = InputSpec("temperature", "continuous", bounds=(20.0, 100.0))
    >>> solvent = InputSpec("solvent", "categorical", levels=["water", "ethanol"])
    >>> hour = InputSpec("hour", "continuous", bounds=(0.0, 24.0), optimizable=False)
    """

    name: str
    type: str
    bounds: tuple[float, float] | None = None
    levels: list[str] | None = None
    units: str | None = None
    optimizable: bool = True

    def __post_init__(self):
        if self.type == "continuous":
            if self.bounds is None:
                raise InvalidSchemaError(
                    f"Continuous input '{self.name}' requires bounds."
                )
            if self.bounds[0] >= self.bounds[1]:
                raise InvalidSchemaError(
                    f"Continuous input '{self.name}' has invalid bounds: "
                    f"lower ({self.bounds[0]}) must be less than "
                    f"upper ({self.bounds[1]})."
                )
        elif self.type == "categorical":
            if self.levels is None or len(self.levels) < 2:
                raise InvalidSchemaError(
                    f"Categorical input '{self.name}' requires at least 2 levels."
                )


@dataclass
class OutputSpec:
    """Specification for a single experiment output (measured result).

    Defines the name and units for an output variable. Outputs are always numeric
    values recorded after running an experiment.

    Parameters
    ----------
    name : str
        Unique identifier for this output variable.
    units : str | None, optional
        Physical units for the output (e.g., "mg", "%"). For display purposes only.

    Examples
    --------
    >>> yield_spec = OutputSpec("yield", units="%")
    >>> purity = OutputSpec("purity", units="%")
    """

    name: str
    units: str | None = None


@dataclass
class ConstantSpec:
    """Specification for a fixed experimental parameter.

    Constants are values that remain unchanged throughout an experiment series.
    They are recorded for documentation but not varied during optimization.

    Parameters
    ----------
    name : str
        Unique identifier for this constant.
    value : float | str
        The fixed value of the constant. Can be numeric or categorical.
    units : str | None, optional
        Physical units for the constant (e.g., "mL", "rpm"). For display purposes only.

    Examples
    --------
    >>> stirring = ConstantSpec("stirring_speed", 500, units="rpm")
    >>> catalyst = ConstantSpec("catalyst", "Pd/C")
    """

    name: str
    value: float | str
    units: str | None = None
