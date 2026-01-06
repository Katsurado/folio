"""Built-in target implementations."""

from collections.abc import Callable
from typing import Literal

import numpy as np

from folio.core.observation import Observation
from folio.targets.base import ScalarTarget


class DirectTarget(ScalarTarget):
    """Target that extracts a single output value directly."""

    def __init__(self, output_name: str, objective: Literal["maximize", "minimize"]):
        self.output_name = output_name
        self.objective = objective

    def compute(self, obs: Observation) -> float | None:
        return obs.outputs.get(self.output_name)


class DerivedTarget(ScalarTarget):
    """Target computed from outputs via a custom function."""

    def __init__(
        self, func: Callable[[dict], float], objective: Literal["maximize", "minimize"]
    ):
        self.func = func
        self.objective = objective

    def compute(self, obs: Observation) -> float | None:
        try:
            return self.func(obs.outputs)
        except Exception:
            return None


class DistanceTarget(ScalarTarget):
    """Target that minimizes distance from observed outputs to target values.

    Users should normalize outputs to similar scales before using, or pre-divide
    target values by typical magnitudes to avoid one output dominating the distance.
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
        """Set the target values to minimize distance to."""
        self.target = values

    def compute(self, obs: Observation) -> float | None:
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
        return float(np.linalg.norm(np.array(observed) - np.array(target)))

    @staticmethod
    def _mse(observed: list[float], target: list[float]) -> float:
        return float(np.mean((np.array(observed) - np.array(target)) ** 2))

    @staticmethod
    def _mae(observed: list[float], target: list[float]) -> float:
        return float(np.mean(np.abs(np.array(observed) - np.array(target))))


class RatioTarget(ScalarTarget):
    """Target computed as ratio of two outputs."""

    def __init__(
        self,
        numerator: str,
        denominator: str,
        objective: Literal["maximize", "minimize"],
    ):
        self.numerator = numerator
        self.denominator = denominator
        self.objective = objective

    def compute(self, obs: Observation) -> float | None:
        num = obs.outputs.get(self.numerator)
        denom = obs.outputs.get(self.denominator)
        if num is None or denom is None or denom == 0:
            return None
        return num / denom


class DifferenceTarget(ScalarTarget):
    """Target computed as difference of two outputs."""

    def __init__(
        self,
        first: str,
        second: str,
        objective: Literal["maximize", "minimize"],
    ):
        self.first = first
        self.second = second
        self.objective = objective

    def compute(self, obs: Observation) -> float | None:
        a = obs.outputs.get(self.first)
        b = obs.outputs.get(self.second)
        if a is None or b is None:
            return None
        return a - b


class SlopeTarget(ScalarTarget):
    """Target computed as slope of linear fit to multiple outputs.

    Requires at least 3 points because 2 points always yield a perfect fit with no
    degrees of freedom to assess fit quality or uncertainty.
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

    def compute(self, obs: Observation) -> float | None:
        y_values = []
        for name in self.output_names:
            value = obs.outputs.get(name)
            if value is None:
                return None
            y_values.append(value)
        coeffs = np.polyfit(self.x_values, y_values, 1)
        return float(coeffs[0])
