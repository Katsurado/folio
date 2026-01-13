"""Custom R² target for Stern-Volmer linearity assessment.

This module demonstrates how to extend Folio's target system with custom
objectives. The R²Target computes the coefficient of determination (R²)
from a linear fit, useful for assessing how well data follows a linear
relationship like the Stern-Volmer equation.

Example
-------
>>> from extensions.r2_target import R2Target, SternVolmerR2Target
>>>
>>> # Generic R² target for any linear relationship
>>> r2_target = R2Target(
...     output_names=["y_0", "y_1", "y_2", "y_3"],
...     x_values=[0.0, 1.0, 2.0, 3.0],
... )
>>>
>>> # Specialized for Stern-Volmer I₀/I vs [O₂]
>>> sv_r2 = SternVolmerR2Target(
...     i0_name="I_0",
...     intensity_names=["I_5", "I_10", "I_15", "I_20"],
...     o2_concentrations=[5.0, 10.0, 15.0, 20.0],
... )
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from folio.targets.base import ScalarTarget

if TYPE_CHECKING:
    from folio.core.observation import Observation


class R2Target(ScalarTarget):
    """Target computed as R² (coefficient of determination) of a linear fit.

    Fits a line y = mx + b to the output values against provided x-values,
    and returns R² measuring how well the data fits a linear relationship.
    R² = 1 means perfect linear fit, R² = 0 means no linear relationship.

    Parameters
    ----------
    output_names : list[str]
        Names of outputs to use as y-values. Order must match x_values.
    x_values : list[float]
        X-values corresponding to each output.

    Raises
    ------
    ValueError
        If fewer than 3 output names provided (need degrees of freedom).
        If output_names and x_values have different lengths.

    Notes
    -----
    Always maximizes R² since higher values indicate better linearity.

    Examples
    --------
    >>> target = R2Target(["y_0", "y_1", "y_2"], [0.0, 1.0, 2.0])
    >>> obs = Observation(project_id=1, inputs={},
    ...                   outputs={"y_0": 1.0, "y_1": 2.0, "y_2": 3.0})
    >>> target.compute(obs)  # Perfect linear fit
    1.0
    """

    def __init__(
        self,
        output_names: list[str],
        x_values: list[float],
        objective: Literal["maximize", "minimize"] = "maximize",
    ):
        if len(output_names) < 3:
            raise ValueError("R2Target requires at least 3 output names.")
        if len(output_names) != len(x_values):
            raise ValueError(
                f"output_names length ({len(output_names)}) must match "
                f"x_values length ({len(x_values)})."
            )
        self.output_names = output_names
        self.x_values = np.array(x_values)
        self.objective = objective

    def compute(self, obs: "Observation") -> float | None:
        """Compute R² of a linear fit to the output values.

        Parameters
        ----------
        obs : Observation
            Observation containing the output values to fit.

        Returns
        -------
        float | None
            R² value (0 to 1), or None if any output is missing.
        """
        y_values = []
        for name in self.output_names:
            value = obs.outputs.get(name)
            if value is None:
                return None
            y_values.append(value)

        y = np.array(y_values)
        x = self.x_values

        # Linear fit: y = mx + b
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)

        # R² = 1 - SS_res / SS_tot
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            # All y values identical - perfect fit but undefined R²
            return 1.0

        r2 = 1 - ss_res / ss_tot
        return float(r2)


class SternVolmerR2Target(ScalarTarget):
    """R² target specialized for Stern-Volmer oxygen quenching analysis.

    Computes R² for the Stern-Volmer relationship: I₀/I = 1 + Ksv[O₂]

    This target transforms raw intensity measurements into the Stern-Volmer
    form (I₀/I vs [O₂]) before computing R². High R² indicates good linearity
    of the Stern-Volmer plot, which is desirable for sensor calibration.

    Parameters
    ----------
    i0_name : str
        Name of the output containing intensity at 0% O₂.
    intensity_names : list[str]
        Names of outputs containing intensities at non-zero O₂ concentrations.
        Order must match o2_concentrations.
    o2_concentrations : list[float]
        O₂ concentrations (%) corresponding to intensity_names.

    Notes
    -----
    The Stern-Volmer equation linearizes oxygen quenching:
        I₀/I = 1 + Ksv[O₂]

    Where:
        - I₀ = luminescence intensity at 0% O₂
        - I = intensity at a given O₂ concentration
        - Ksv = Stern-Volmer constant (sensitivity)
        - [O₂] = oxygen concentration

    A sensor with high R² exhibits ideal quenching behavior, making
    calibration straightforward and measurements reliable.

    Examples
    --------
    >>> target = SternVolmerR2Target(
    ...     i0_name="I_0",
    ...     intensity_names=["I_5", "I_10", "I_15", "I_20"],
    ...     o2_concentrations=[5.0, 10.0, 15.0, 20.0],
    ... )
    >>> obs = Observation(
    ...     project_id=1, inputs={},
    ...     outputs={"I_0": 100, "I_5": 80, "I_10": 67, "I_15": 57, "I_20": 50}
    ... )
    >>> r2 = target.compute(obs)
    """

    def __init__(
        self,
        i0_name: str,
        intensity_names: list[str],
        o2_concentrations: list[float],
    ):
        if len(intensity_names) < 3:
            raise ValueError(
                "SternVolmerR2Target requires at least 3 intensity measurements."
            )
        if len(intensity_names) != len(o2_concentrations):
            raise ValueError(
                f"intensity_names length ({len(intensity_names)}) must match "
                f"o2_concentrations length ({len(o2_concentrations)})."
            )
        self.i0_name = i0_name
        self.intensity_names = intensity_names
        self.o2_concentrations = np.array(o2_concentrations)
        self.objective: Literal["maximize", "minimize"] = "maximize"

    def compute(self, obs: "Observation") -> float | None:
        """Compute R² of the Stern-Volmer plot.

        Transforms intensities to I₀/I ratios and computes R² of the
        linear fit against O₂ concentration.

        Parameters
        ----------
        obs : Observation
            Observation containing I₀ and intensity measurements.

        Returns
        -------
        float | None
            R² value (0 to 1), or None if any intensity is missing or zero.
        """
        i0 = obs.outputs.get(self.i0_name)
        if i0 is None or i0 <= 0:
            return None

        # Compute I₀/I ratios (Stern-Volmer y-values)
        sv_ratios = []
        for name in self.intensity_names:
            intensity = obs.outputs.get(name)
            if intensity is None or intensity <= 0:
                return None
            sv_ratios.append(i0 / intensity)

        y = np.array(sv_ratios)
        x = self.o2_concentrations

        # Linear fit: I₀/I = 1 + Ksv*[O₂]
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)

        # R² = 1 - SS_res / SS_tot
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0

        r2 = 1 - ss_res / ss_tot
        return float(r2)


class SternVolmerKsvTarget(ScalarTarget):
    """Ksv target for Stern-Volmer oxygen quenching analysis.

    Computes the Stern-Volmer constant (Ksv) from intensity measurements.
    Ksv is the slope of the Stern-Volmer plot: I₀/I = 1 + Ksv[O₂]

    Higher Ksv means greater sensitivity to oxygen.

    Parameters
    ----------
    i0_name : str
        Name of the output containing intensity at 0% O₂.
    intensity_names : list[str]
        Names of outputs containing intensities at non-zero O₂ concentrations.
    o2_concentrations : list[float]
        O₂ concentrations (%) corresponding to intensity_names.

    Examples
    --------
    >>> target = SternVolmerKsvTarget(
    ...     i0_name="I_0",
    ...     intensity_names=["I_5", "I_10", "I_15", "I_20"],
    ...     o2_concentrations=[5.0, 10.0, 15.0, 20.0],
    ... )
    """

    def __init__(
        self,
        i0_name: str,
        intensity_names: list[str],
        o2_concentrations: list[float],
    ):
        if len(intensity_names) < 3:
            raise ValueError(
                "SternVolmerKsvTarget requires at least 3 intensity measurements."
            )
        if len(intensity_names) != len(o2_concentrations):
            raise ValueError(
                f"intensity_names length ({len(intensity_names)}) must match "
                f"o2_concentrations length ({len(o2_concentrations)})."
            )
        self.i0_name = i0_name
        self.intensity_names = intensity_names
        self.o2_concentrations = np.array(o2_concentrations)
        self.objective: Literal["maximize", "minimize"] = "maximize"

    def compute(self, obs: "Observation") -> float | None:
        """Compute Ksv from the Stern-Volmer plot slope.

        Parameters
        ----------
        obs : Observation
            Observation containing I₀ and intensity measurements.

        Returns
        -------
        float | None
            Ksv value (slope), or None if any intensity is missing or zero.
        """
        i0 = obs.outputs.get(self.i0_name)
        if i0 is None or i0 <= 0:
            return None

        sv_ratios = []
        for name in self.intensity_names:
            intensity = obs.outputs.get(name)
            if intensity is None or intensity <= 0:
                return None
            sv_ratios.append(i0 / intensity)

        y = np.array(sv_ratios)
        x = self.o2_concentrations

        # Linear fit: I₀/I = 1 + Ksv*[O₂], slope = Ksv
        coeffs = np.polyfit(x, y, 1)
        ksv = coeffs[0]

        return float(ksv)
