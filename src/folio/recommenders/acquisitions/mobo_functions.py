"""Multi-objective acquisition function implementations."""

from typing import Literal

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model

from folio.recommenders.acquisitions.mobo_base import MultiObjectiveAcquisition


class NEHVI(MultiObjectiveAcquisition):
    """Noisy Expected Hypervolume Improvement acquisition function builder.

    NEHVI measures the expected increase in the hypervolume dominated by the
    Pareto frontier when adding a new candidate point, accounting for observation
    noise. It naturally balances exploration and exploitation across all objectives
    simultaneously, favoring points that are likely to expand the Pareto frontier.

    This implementation wraps BoTorch's qNoisyExpectedHypervolumeImprovement,
    which handles noisy observations and supports batch acquisition (q > 1
    candidates per iteration).

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
        """Build a BoTorch qNoisyExpectedHypervolumeImprovement acquisition function.

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
            Observed input points, shape (n_samples, n_features). Used to
            compute the current Pareto frontier for hypervolume calculation.
        Y : torch.Tensor
            Observed objective values, shape (n_samples, n_objectives).
            Pass original values; negation for minimization objectives is
            handled internally based on the maximize parameter.
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
            A BoTorch qNoisyExpectedHypervolumeImprovement instance configured
            with the provided model, baseline data, and reference point.
        """
        ...


class ParEGO(MultiObjectiveAcquisition):
    """ParEGO (Pareto Efficient Global Optimization) acquisition function builder.

    ParEGO converts multi-objective optimization into a series of single-objective
    problems using random scalarization. At each iteration, it samples random
    weights and combines the objectives into a single scalar value, then applies
    a standard single-objective acquisition function (typically Expected Improvement).

    This approach is simpler and more scalable than hypervolume-based methods,
    making it suitable for problems with many objectives (4+) where NEHVI becomes
    computationally expensive.

    Parameters
    ----------
    scalarization : {"chebyshev", "linear"}, default="chebyshev"
        Scalarization method for combining objectives:

        - "chebyshev": Augmented Chebyshev scalarization (original ParEGO).
          Uses max_i(w_i * |f_i - z_i|) + rho * sum_i(w_i * |f_i - z_i|)
          where z is the utopia point. Better at finding solutions in
          non-convex regions of the Pareto frontier.

        - "linear": Simple weighted sum. Uses sum_i(w_i * f_i).
          Faster but may miss solutions in non-convex Pareto regions.

    rho : float, default=0.05
        Augmentation parameter for Chebyshev scalarization. Controls the
        trade-off between the max term and sum term. Ignored for linear
        scalarization. Small positive values (0.01-0.1) are typical.

    Notes
    -----
    **Objective Handling**: Pass original Y values and specify the maximize
    direction for each objective. The builder internally handles negation for
    objectives that should be minimized.

    **Weight Sampling**: At each call to build(), new random weights are
    sampled from a Dirichlet distribution (ensuring they sum to 1). This
    randomization helps explore different trade-offs across the Pareto
    frontier over multiple iterations.

    **Advantages over NEHVI**:
    - Computational cost is independent of the number of objectives
    - Scales well to many-objective problems (4+ objectives)
    - Simpler implementation with fewer hyperparameters

    **Disadvantages**:
    - Single-objective acquisition may miss diverse Pareto-optimal solutions
    - Requires multiple iterations to explore the full frontier
    - Random weights may not efficiently cover the objective space

    Examples
    --------
    >>> parego = ParEGO(scalarization="chebyshev", rho=0.05)
    >>> acqf = parego.build(
    ...     model=fitted_mogp,
    ...     X_baseline=X_observed,
    ...     Y=Y_observed,  # Original values, negation handled internally
    ...     ref_point=[0.0, 0.0],
    ...     maximize=[True, False],  # Maximize obj 0, minimize obj 1
    ... )
    >>> candidates, values = optimize_acqf(acqf, bounds=bounds, q=1, ...)

    Reference: Knowles (2006), ParEGO: A Hybrid Algorithm With On-Line
    Landscape Approximation for Expensive Multiobjective Optimization Problems.
    """

    def __init__(
        self,
        scalarization: Literal["chebyshev", "linear"] = "chebyshev",
        rho: float = 0.05,
    ):
        """Initialize ParEGO builder.

        Parameters
        ----------
        scalarization : {"chebyshev", "linear"}, default="chebyshev"
            Method for scalarizing multiple objectives into a single value.
            "chebyshev" (augmented Tchebycheff) is the original ParEGO method;
            "linear" uses simple weighted sum.
        rho : float, default=0.05
            Augmentation coefficient for Chebyshev scalarization. Must be
            non-negative. Ignored when scalarization="linear".

        Raises
        ------
        ValueError
            If scalarization is not "chebyshev" or "linear".
        ValueError
            If rho is negative.
        """
        if scalarization not in ("chebyshev", "linear"):
            raise ValueError(
                f"scalarization must be 'chebyshev' or 'linear', got '{scalarization}'"
            )
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        self.scalarization = scalarization
        self.rho = rho

    def build(
        self,
        model: Model,
        X_baseline: torch.Tensor,
        Y: torch.Tensor,
        ref_point: list[float],
        maximize: list[bool],
    ) -> AcquisitionFunction:
        """Build a ParEGO acquisition function with random scalarization weights.

        Constructs a scalarized single-objective acquisition function by sampling
        random weights and combining the objectives into a single value.

        Parameters
        ----------
        model : Model
            A fitted BoTorch multi-output model (e.g., ModelListGP or
            MultiTaskGP). Must have a posterior() method.
        X_baseline : torch.Tensor
            Observed input points, shape (n_samples, n_features). Used to
            estimate the utopia point for Chebyshev scalarization.
        Y : torch.Tensor
            Observed objective values, shape (n_samples, n_objectives).
            Pass original values; negation for minimization objectives is
            handled internally based on the maximize parameter.
        ref_point : list[float]
            Reference point, length n_objectives. For ParEGO, this may be
            used to help estimate the utopia/nadir points for normalization.
        maximize : list[bool]
            Optimization direction for each objective, length n_objectives.
            True = maximize that objective, False = minimize.

        Returns
        -------
        AcquisitionFunction
            A BoTorch acquisition function that evaluates the scalarized
            objective. Can be used with optimize_acqf like any single-objective
            acquisition function.
        """
        ...
