"""Multi-objective acquisition function implementations."""

from typing import Literal

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model

from folio.recommenders.acquisitions.mobo_base import MultiObjectiveAcquisition


class EHVI(MultiObjectiveAcquisition):
    """Expected Hypervolume Improvement acquisition function builder.

    EHVI measures the expected increase in the hypervolume dominated by the
    Pareto frontier when adding a new candidate point. It naturally balances
    exploration and exploitation across all objectives simultaneously, favoring
    points that are likely to expand the Pareto frontier.

    This implementation wraps BoTorch's qNoisyExpectedHypervolumeImprovement
    (qNEHVI), which handles noisy observations and supports batch acquisition
    (q > 1 candidates per iteration).

    Parameters
    ----------
    alpha : float, default=0.0
        Hyperparameter controlling the approximation of the Pareto frontier
        for the box decomposition. A value of 0.0 gives the exact hypervolume;
        small positive values can improve numerical stability.
    prune_baseline : bool, default=True
        If True, prune dominated points from the baseline before computing
        the acquisition value. This can improve computational efficiency
        without affecting the result.
    cache_root : bool, default=True
        If True, cache the root decomposition of the Pareto frontier to
        speed up repeated evaluations during optimization.

    Attributes
    ----------
    alpha : float
        The approximation parameter.
    prune_baseline : bool
        Whether to prune dominated baseline points.
    cache_root : bool
        Whether to cache root decomposition.

    Notes
    -----
    **Objective Negation**: BoTorch always assumes maximization. For objectives
    where `maximize[i]` is False (minimization), you must negate the corresponding
    column in Y before passing to build(). The reference point should also be
    adjusted accordingly (negate the reference value for minimized objectives).

    **Reference Point Selection**: The reference point should be dominated by
    all points on the current Pareto frontier. A common heuristic is to use
    values slightly worse than the worst observed values for each objective.
    Poor reference point choices can lead to zero hypervolume improvement.

    **Computational Cost**: EHVI scales poorly with the number of objectives
    (exponentially in the worst case) due to the box decomposition of the
    Pareto frontier. For more than 3-4 objectives, consider using ParEGO
    or other scalarization-based methods.

    Examples
    --------
    >>> ehvi = EHVI(alpha=0.0, prune_baseline=True)
    >>> acqf = ehvi.build(
    ...     model=fitted_mogp,
    ...     X_baseline=X_observed,
    ...     Y=Y_observed,  # Negate columns for objectives to minimize!
    ...     ref_point=[0.0, 0.0],  # Adjust for negated objectives
    ...     maximize=[True, True],  # After negation, all are "maximize"
    ... )
    >>> candidates, values = optimize_acqf(acqf, bounds=bounds, q=1, ...)

    Reference: Emmerich, Giannakoglou, Naujoks (2006), Single- and Multiobjective
    Evolutionary Optimization Assisted by Gaussian Random Field Metamodels.

    See Also: Daulton, Balandat, Bakshy (2021), Parallel Bayesian Optimization
    of Multiple Noisy Objectives with Expected Hypervolume Improvement.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        prune_baseline: bool = True,
        cache_root: bool = True,
    ):
        """Initialize Expected Hypervolume Improvement builder.

        Parameters
        ----------
        alpha : float, default=0.0
            Approximation parameter for box decomposition. Use 0.0 for exact
            hypervolume computation; small positive values may improve stability.
        prune_baseline : bool, default=True
            If True, remove dominated points from the baseline Pareto frontier
            before computing acquisition values.
        cache_root : bool, default=True
            If True, cache the root decomposition for faster repeated evaluations.
        """
        self.alpha = alpha
        self.prune_baseline = prune_baseline
        self.cache_root = cache_root

    def build(
        self,
        model: Model,
        X_baseline: torch.Tensor,
        Y: torch.Tensor,
        ref_point: list[float],
        maximize: list[bool],
    ) -> AcquisitionFunction:
        """Build a BoTorch qNoisyExpectedHypervolumeImprovement acquisition function.

        Constructs an EHVI acquisition function using the provided multi-output
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
            **Important**: For objectives where maximize[i] is False, you must
            negate the corresponding column BEFORE calling this method, since
            BoTorch always maximizes. Example: if minimizing objective 1,
            pass -Y[:, 1] instead of Y[:, 1].
        ref_point : list[float]
            Reference point for hypervolume calculation, length n_objectives.
            Must be dominated by all Pareto-optimal points. For negated
            (minimization) objectives, negate the reference value as well.
        maximize : list[bool]
            Optimization direction for each objective, length n_objectives.
            After negating Y columns for minimization objectives, this should
            effectively be all True (since BoTorch maximizes). This parameter
            is included for interface consistency and validation.

        Returns
        -------
        AcquisitionFunction
            A BoTorch qNoisyExpectedHypervolumeImprovement instance configured
            with the provided model, baseline data, and reference point.

        Notes
        -----
        Implementation should:

        1. Create a NondominatedPartitioning from Y and ref_point
        2. Instantiate qNoisyExpectedHypervolumeImprovement with:
           - model=model
           - ref_point=ref_point (as tensor)
           - X_baseline=X_baseline
           - prune_baseline=self.prune_baseline
           - alpha=self.alpha
           - cache_root=self.cache_root
        3. Return the acquisition function

        Example imports needed:
            from botorch.acquisition.multi_objective.monte_carlo import (
                qNoisyExpectedHypervolumeImprovement,
            )
            from botorch.utils.multi_objective.box_decompositions.non_dominated import (
                NondominatedPartitioning,
            )
        """
        ...


class ParEGO(MultiObjectiveAcquisition):
    """ParEGO (Pareto Efficient Global Optimization) acquisition function builder.

    ParEGO converts multi-objective optimization into a series of single-objective
    problems using random scalarization. At each iteration, it samples random
    weights and combines the objectives into a single scalar value, then applies
    a standard single-objective acquisition function (typically Expected Improvement).

    This approach is simpler and more scalable than hypervolume-based methods,
    making it suitable for problems with many objectives (4+) where EHVI becomes
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

    Attributes
    ----------
    scalarization : str
        The scalarization method.
    rho : float
        The augmentation parameter.

    Notes
    -----
    **Objective Negation**: Like EHVI, BoTorch assumes maximization. For
    objectives where `maximize[i]` is False, negate the corresponding column
    in Y before calling build().

    **Weight Sampling**: At each call to build(), new random weights are
    sampled from a Dirichlet distribution (ensuring they sum to 1). This
    randomization helps explore different trade-offs across the Pareto
    frontier over multiple iterations.

    **Advantages over EHVI**:
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
    ...     Y=Y_observed,  # Negate columns for objectives to minimize!
    ...     ref_point=[0.0, 0.0],  # Used for utopia point estimation
    ...     maximize=[True, True],
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

        Constructs a scalarized single-objective acquisition function by:
        1. Sampling random weights from a Dirichlet distribution
        2. Creating a scalarization transform (Chebyshev or linear)
        3. Wrapping the multi-output model with the scalarization
        4. Building a standard single-objective acquisition (e.g., EI)

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
            **Important**: For objectives where maximize[i] is False, negate
            the corresponding column BEFORE calling this method.
        ref_point : list[float]
            Reference point, length n_objectives. For ParEGO, this may be
            used to help estimate the utopia/nadir points for normalization.
        maximize : list[bool]
            Optimization direction for each objective, length n_objectives.
            After negating Y columns for minimization objectives, this should
            effectively be all True.

        Returns
        -------
        AcquisitionFunction
            A BoTorch acquisition function that evaluates the scalarized
            objective. Can be used with optimize_acqf like any single-objective
            acquisition function.

        Notes
        -----
        Implementation should:

        1. Sample weights from Dirichlet(1, 1, ..., 1) distribution
        2. Compute utopia point (best value per objective from Y)
        3. For Chebyshev: use get_chebyshev_scalarization from BoTorch
           For linear: create weighted sum scalarization
        4. Create a posterior transform or model wrapper that applies scalarization
        5. Build ExpectedImprovement (or similar) on the scalarized model
        6. Return the acquisition function

        Example imports needed:
            from botorch.acquisition.objective import GenericMCObjective
            from botorch.utils.multi_objective.scalarization import (
                get_chebyshev_scalarization,
            )
            from torch.distributions import Dirichlet
        """
        ...
