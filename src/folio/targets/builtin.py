"""Built-in target implementations."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np

from folio.targets.base import ScalarTarget

if TYPE_CHECKING:
    from folio.core.observation import Observation


class DirectTarget(ScalarTarget):
    """Target that extracts a single output value directly.

    The simplest target type: returns the value of one output unchanged.

    Parameters
    ----------
    output_name : str
        Name of the output to extract.
    objective : {"maximize", "minimize"}
        Optimization direction for this target.

    Examples
    --------
    >>> target = DirectTarget("yield", "maximize")
    >>> obs = Observation(project_id=1, inputs={}, outputs={"yield": 85.0})
    >>> target.compute(obs)
    85.0
    """

    def __init__(self, output_name: str, objective: Literal["maximize", "minimize"]):
        self.output_name = output_name
        self.objective = objective

    def compute(self, obs: "Observation") -> float | None:
        """Extract the output value from an observation.

        Parameters
        ----------
        obs : Observation
            Observation to extract value from.

        Returns
        -------
        float | None
            The output value, or None if the output is not present.
        """
        return obs.outputs.get(self.output_name)


class DerivedTarget(ScalarTarget):
    """Target computed from outputs via a custom function.

    Allows arbitrary transformations of outputs into a scalar target.
    Useful for complex objectives that can't be expressed as simple
    ratios or differences.

    Parameters
    ----------
    func : Callable[[dict], float]
        Function that takes the outputs dict and returns a scalar value.
    objective : {"maximize", "minimize"}
        Optimization direction for this target.

    Notes
    -----
    If the function raises any exception, compute() returns None.
    This allows graceful handling of missing or invalid output values.

    Examples
    --------
    >>> def selectivity(outputs):
    ...     return outputs["product_a"] / (outputs["product_a"] + outputs["product_b"])
    >>> target = DerivedTarget(selectivity, "maximize")
    """

    def __init__(
        self, func: Callable[[dict], float], objective: Literal["maximize", "minimize"]
    ):
        self.func = func
        self.objective = objective

    def compute(self, obs: "Observation") -> float | None:
        """Compute the target value using the custom function.

        Parameters
        ----------
        obs : Observation
            Observation to compute target from.

        Returns
        -------
        float | None
            The computed value, or None if the function raises an exception.
        """
        try:
            return self.func(obs.outputs)
        except Exception:
            return None


class DistanceTarget(ScalarTarget):
    """Target that minimizes distance from observed outputs to target values.

    Computes the distance between observed output values and specified target
    values. Always uses "minimize" objective since we want to minimize distance.

    Parameters
    ----------
    output_names : list[str]
        Names of outputs to include in distance calculation.
    default_target : list[float] | None, optional
        Target values to minimize distance to. Can be set later via set_target().
    metric : {"euclidean", "mse", "mae"}, optional
        Distance metric. Defaults to "euclidean".
        - euclidean: sqrt(sum((obs - target)^2))
        - mse: mean((obs - target)^2)
        - mae: mean(|obs - target|)

    Notes
    -----
    Normalize outputs to similar scales before using, or pre-divide target values
    by typical magnitudes to avoid one output dominating the distance.

    Examples
    --------
    >>> target = DistanceTarget(["temp", "pressure"], [100.0, 1.0])
    >>> obs = Observation(
    ...     project_id=1, inputs={}, outputs={"temp": 95.0, "pressure": 1.1}
    ... )
    >>> target.compute(obs)  # Distance from (95, 1.1) to (100, 1)
    """

    def __init__(
        self,
        output_names: list[str],
        default_target: list[float] | None = None,
        metric: Literal["euclidean", "mse", "mae"] = "euclidean",
    ):
        self.objective = "minimize"
        self.output_names = output_names
        self.target = default_target
        self.metric = metric

    def set_target(self, values: list[float]) -> None:
        """Set the target values to minimize distance to.

        Parameters
        ----------
        values : list[float]
            Target values, one per output in output_names order.
        """
        self.target = values

    def compute(self, obs: "Observation") -> float | None:
        """Compute distance from observation outputs to target values.

        Parameters
        ----------
        obs : Observation
            Observation to compute distance for.

        Returns
        -------
        float | None
            The distance value, or None if target is not set or any output
            is missing.

        Raises
        ------
        ValueError
            If metric is not a recognized value.
        """
        if self.target is None:
            return None
        observed = []
        for name in self.output_names:
            value = obs.outputs.get(name)
            if value is None:
                return None
            observed.append(value)
        if self.metric == "euclidean":
            return self._euclidean(observed, self.target)
        elif self.metric == "mse":
            return self._mse(observed, self.target)
        elif self.metric == "mae":
            return self._mae(observed, self.target)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    @staticmethod
    def _euclidean(observed: list[float], target: list[float]) -> float:
        """Compute Euclidean distance: sqrt(sum((obs - target)^2))."""
        return float(np.linalg.norm(np.array(observed) - np.array(target)))

    @staticmethod
    def _mse(observed: list[float], target: list[float]) -> float:
        """Compute Mean Squared Error: mean((obs - target)^2)."""
        return float(np.mean((np.array(observed) - np.array(target)) ** 2))

    @staticmethod
    def _mae(observed: list[float], target: list[float]) -> float:
        """Compute Mean Absolute Error: mean(|obs - target|)."""
        return float(np.mean(np.abs(np.array(observed) - np.array(target))))


class RatioTarget(ScalarTarget):
    """Target computed as ratio of two outputs (numerator / denominator).

    Useful for selectivity, efficiency, or yield calculations that are
    expressed as ratios of measured quantities.

    Parameters
    ----------
    numerator : str
        Name of the output to use as numerator.
    denominator : str
        Name of the output to use as denominator.
    objective : {"maximize", "minimize"}
        Optimization direction for this target.

    Notes
    -----
    Returns None if either output is missing or if the denominator is zero.

    Examples
    --------
    >>> target = RatioTarget("product", "starting_material", "maximize")
    >>> obs = Observation(project_id=1, inputs={},
    ...                   outputs={"product": 8.0, "starting_material": 10.0})
    >>> target.compute(obs)
    0.8
    """

    def __init__(
        self,
        numerator: str,
        denominator: str,
        objective: Literal["maximize", "minimize"],
    ):
        self.numerator = numerator
        self.denominator = denominator
        self.objective = objective

    def compute(self, obs: "Observation") -> float | None:
        """Compute the ratio of numerator to denominator outputs.

        Parameters
        ----------
        obs : Observation
            Observation to compute ratio from.

        Returns
        -------
        float | None
            The ratio value, or None if either output is missing or
            denominator is zero.
        """
        num = obs.outputs.get(self.numerator)
        denom = obs.outputs.get(self.denominator)
        if num is None or denom is None or denom == 0:
            return None
        return num / denom


class DifferenceTarget(ScalarTarget):
    """Target computed as difference of two outputs (first - second).

    Useful for optimization objectives that involve the difference between
    two measured quantities, such as profit margins or temperature differentials.

    Parameters
    ----------
    first : str
        Name of the first output (minuend).
    second : str
        Name of the second output (subtrahend).
    objective : {"maximize", "minimize"}
        Optimization direction for this target.

    Examples
    --------
    >>> target = DifferenceTarget("revenue", "cost", "maximize")
    >>> obs = Observation(project_id=1, inputs={},
    ...                   outputs={"revenue": 100.0, "cost": 60.0})
    >>> target.compute(obs)
    40.0
    """

    def __init__(
        self,
        first: str,
        second: str,
        objective: Literal["maximize", "minimize"],
    ):
        self.first = first
        self.second = second
        self.objective = objective

    def compute(self, obs: "Observation") -> float | None:
        """Compute the difference: first - second.

        Parameters
        ----------
        obs : Observation
            Observation to compute difference from.

        Returns
        -------
        float | None
            The difference value, or None if either output is missing.
        """
        a = obs.outputs.get(self.first)
        b = obs.outputs.get(self.second)
        if a is None or b is None:
            return None
        return a - b


class SlopeTarget(ScalarTarget):
    """Target computed as slope of linear fit to multiple outputs.

    Fits a line y = mx + b to the output values (as y) against provided x-values,
    and returns the slope m. Useful for rate optimization, kinetics studies, or
    any objective based on the trend across multiple measurements.

    Parameters
    ----------
    output_names : list[str]
        Names of outputs to use as y-values. Order must match x_values.
    x_values : list[float]
        X-values corresponding to each output. Often time points, concentrations,
        or other independent variables.
    objective : {"maximize", "minimize"}
        Optimization direction for this target.

    Raises
    ------
    ValueError
        If fewer than 3 output names provided (need degrees of freedom for fit).
        If output_names and x_values have different lengths.

    Notes
    -----
    Requires at least 3 points because 2 points always yield a perfect fit with
    no degrees of freedom to assess fit quality or uncertainty.

    Examples
    --------
    >>> target = SlopeTarget(["y_0", "y_1", "y_2"], [0.0, 1.0, 2.0], "maximize")
    >>> obs = Observation(project_id=1, inputs={},
    ...                   outputs={"y_0": 1.0, "y_1": 2.0, "y_2": 3.0})
    >>> target.compute(obs)
    1.0
    """

    def __init__(
        self,
        output_names: list[str],
        x_values: list[float],
        objective: Literal["maximize", "minimize"],
    ):
        if len(output_names) < 3:
            raise ValueError("SlopeTarget requires at least 3 output names.")
        if len(output_names) != len(x_values):
            raise ValueError(
                f"output_names length ({len(output_names)}) must match "
                f"x_values length ({len(x_values)})."
            )
        self.output_names = output_names
        self.x_values = x_values
        self.objective = objective

    def compute(self, obs: "Observation") -> float | None:
        """Compute the slope of a linear fit to the output values.

        Parameters
        ----------
        obs : Observation
            Observation containing the output values to fit.

        Returns
        -------
        float | None
            The slope of the linear fit, or None if any output is missing.
        """
        y_values = []
        for name in self.output_names:
            value = obs.outputs.get(name)
            if value is None:
                return None
            y_values.append(value)
        coeffs = np.polyfit(self.x_values, y_values, 1)
        return float(coeffs[0])
