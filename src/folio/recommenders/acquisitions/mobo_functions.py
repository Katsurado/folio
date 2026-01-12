"""Multi-objective acquisition function implementations."""

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.model import Model

from folio.recommenders.acquisitions.mobo_base import MultiObjectiveAcquisition


class NEHVI(MultiObjectiveAcquisition):
    """Noisy Expected Hypervolume Improvement acquisition function builder.

    NEHVI measures the expected increase in the hypervolume dominated by the
    Pareto frontier when adding a new candidate point, accounting for observation
    noise. It naturally balances exploration and exploitation across all objectives
    simultaneously, favoring points that are likely to expand the Pareto frontier.

    This implementation wraps BoTorch's qLogNoisyExpectedHypervolumeImprovement,
    which handles noisy observations, supports batch acquisition (q > 1
    candidates per iteration), and has better numerical stability.

    Parameters
    ----------
    alpha : float, default=0.0
        Hyperparameter controlling the approximation of the Pareto frontier
        for the box decomposition. A value of 0.0 gives the exact hypervolume;
        small positive values can improve numerical stability.

    Notes
    -----
    **Objective Handling**: Pass original Y values and specify the maximize
    direction for each objective. The builder internally handles negation for
    objectives that should be minimized.

    **Reference Point Selection**: The reference point should be dominated by
    all points on the current Pareto frontier. A common heuristic is to use
    values slightly worse than the worst observed values for each objective.
    Poor reference point choices can lead to zero hypervolume improvement.

    **Computational Cost**: NEHVI scales poorly with the number of objectives
    (exponentially in the worst case) due to the box decomposition of the
    Pareto frontier. For more than 3-4 objectives, consider using ParEGO
    or other scalarization-based methods.

    Examples
    --------
    >>> nehvi = NEHVI(alpha=0.0)
    >>> acqf = nehvi.build(
    ...     model=fitted_mogp,
    ...     X_baseline=X_observed,
    ...     Y=Y_observed,  # Original values, negation handled internally
    ...     ref_point=[0.0, 0.0],
    ...     maximize=[True, False],  # Maximize obj 0, minimize obj 1
    ... )
    >>> candidates, values = optimize_acqf(acqf, bounds=bounds, q=1, ...)

    Reference: Daulton, Balandat, Bakshy (2021), Parallel Bayesian Optimization
    of Multiple Noisy Objectives with Expected Hypervolume Improvement.
    """

    def __init__(self, alpha: float = 0.0):
        """Initialize Noisy Expected Hypervolume Improvement builder.

        Parameters
        ----------
        alpha : float, default=0.0
            Approximation parameter for box decomposition. Use 0.0 for exact
            hypervolume computation; small positive values may improve stability.
        """
        self.alpha = alpha

    def build(
        self,
        model: Model,
        X_baseline: torch.Tensor,
        Y: torch.Tensor,
        ref_point: list[float],
        maximize: list[bool],
    ) -> AcquisitionFunction:
        """Build a BoTorch qLogNoisyExpectedHypervolumeImprovement acquisition function.

        Constructs an NEHVI acquisition function using the provided multi-output
        model and observed data. The acquisition function can then be optimized
        using BoTorch's optimize_acqf to find the next candidate(s).

        Parameters
        ----------
        model : Model
            A fitted BoTorch multi-output model (e.g., ModelListGP with one
            SingleTaskGP per objective, or a MultiTaskGP). Must have a
            posterior() method that returns predictions for all objectives.
        X_baseline : torch.Tensor
            Observed input points, shape (n_samples, n_features), dtype float64.
            Used to compute the current Pareto frontier for hypervolume calculation.
        Y : torch.Tensor
            Observed objective values, shape (n_samples, n_objectives), dtype
            float64. Pass original values; negation for minimization objectives
            is handled internally based on the maximize parameter.
        ref_point : list[float]
            Reference point for hypervolume calculation, length n_objectives.
            Must be dominated by all Pareto-optimal points. Pass original
            reference values; negation is handled internally.
        maximize : list[bool]
            Optimization direction for each objective, length n_objectives.
            True = maximize that objective, False = minimize.

        Returns
        -------
        AcquisitionFunction
            A BoTorch qLogNoisyExpectedHypervolumeImprovement instance configured
            with the provided model, baseline data, and reference point.

        Raises
        ------
        ValueError
            If X_baseline is empty (no observations).
        ValueError
            If len(maximize) != number of objectives in Y.
        ValueError
            If X_baseline or Y is not torch.float64.
        """
        self._validate_dtype(X_baseline, Y)

        if X_baseline.shape[0] == 0:
            raise ValueError(
                "X_baseline cannot be empty; at least one observation required"
            )

        n_objectives = Y.shape[1] if Y.dim() > 1 else 1
        if len(maximize) != n_objectives:
            raise ValueError(
                f"len(maximize)={len(maximize)} must "
                f"match number of objectives={n_objectives}"
            )

        y_max, ref_max = self._prepare_for_maximization(Y, ref_point, maximize)

        nehvi = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref_max, X_baseline=X_baseline, alpha=self.alpha
        )

        return nehvi
