"""Abstract base class for multi-objective acquisition function builders."""

from abc import ABC, abstractmethod

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model


class MultiObjectiveAcquisition(ABC):
    """Abstract base class for building multi-objective acquisition functions.

    Multi-objective acquisition functions guide the search for Pareto-optimal
    solutions when optimizing multiple conflicting objectives simultaneously.
    Unlike single-objective acquisition functions that optimize a scalar target,
    multi-objective variants work with the Pareto frontier and use metrics like
    hypervolume improvement to identify promising candidates.

    This class follows the builder pattern: instances store hyperparameters and
    provide a `build()` method that returns a BoTorch AcquisitionFunction ready
    for use with optimize_acqf. The builder pattern enables:

    - Reusing the same configuration across multiple optimization iterations
    - Swapping acquisition functions (EHVI, ParEGO, etc.) without changing the loop
    - Clean separation between user-facing config and BoTorch internals

    All tensor inputs (X_baseline, Y) must be torch.float64 for numerical stability
    with BoTorch's GP models and acquisition function optimization.

    Common multi-objective acquisition functions include:

    - **EHVI (Expected Hypervolume Improvement)**: Maximizes expected improvement
      in the hypervolume dominated by the Pareto frontier. Requires a reference
      point that is dominated by all Pareto-optimal solutions.

    - **ParEGO**: Scalarizes objectives using random weights and applies standard
      single-objective acquisition. Simple but effective, especially for many
      objectives where EHVI becomes expensive.

    - **NEHVI (Noisy Expected Hypervolume Improvement)**: Variant of EHVI that
      accounts for observation noise in the objectives.

    Notes
    -----
    Subclasses must implement `build()` to return an AcquisitionFunction that:

    - Inherits from botorch.acquisition.AcquisitionFunction
    - Implements forward(X: Tensor) -> Tensor
    - Accepts X with shape (batch, q, d) and returns shape (batch,)

    The reference point should be chosen to be dominated by all points on the
    Pareto frontier. A common heuristic is to use values slightly worse than
    the worst observed objective values.

    Examples
    --------
    Implementing a custom multi-objective acquisition:

    >>> class MyMOAcquisition(MultiObjectiveAcquisition):
    ...     def __init__(self, alpha: float = 0.5):
    ...         self.alpha = alpha
    ...
    ...     def build(self, model, X_baseline, Y, ref_point, maximize):
    ...         return _MyMOAcqf(model, X_baseline, Y, ref_point, maximize, self.alpha)

    Using a multi-objective acquisition in an optimization loop:

    >>> ehvi = ExpectedHypervolumeImprovement()
    >>> acqf = ehvi.build(
    ...     model=fitted_mogp,
    ...     X_baseline=X_observed,
    ...     Y=Y_observed,
    ...     ref_point=[0.0, 0.0],
    ...     maximize=[True, True],
    ... )
    >>> # Use acqf with optimize_acqf for gradient-based optimization
    >>> candidates, acq_values = optimize_acqf(acqf, bounds=bounds, ...)

    Reference: Daulton et al. (2020), Differentiable Expected Hypervolume
    Improvement for Parallel Multi-Objective Bayesian Optimization.
    """

    def _prepare_for_maximization(
        self,
        Y: torch.Tensor,
        ref_point: list[float],
        maximize: list[bool],
    ) -> tuple[torch.Tensor, list[float]]:
        """Convert mixed max/min objectives to all-maximization for BoTorch.

        BoTorch multi-objective acquisition functions assume maximization of all
        objectives. This helper negates columns of Y and corresponding elements
        of ref_point for any objective where maximize[i] is False, converting
        a mixed maximization/minimization problem to pure maximization.

        Parameters
        ----------
        Y : torch.Tensor
            Observed objective values, shape (n_samples, n_objectives).
            Original values as provided by the caller.
        ref_point : list[float]
            Reference point for hypervolume calculation, length n_objectives.
            Original values as provided by the caller.
        maximize : list[bool]
            Optimization direction for each objective, length n_objectives.
            True = maximize, False = minimize.

        Returns
        -------
        Y_max : torch.Tensor
            Copy of Y with columns negated where maximize[i] is False.
        ref_point_max : list[float]
            Copy of ref_point with elements negated where maximize[i] is False.

        Examples
        --------
        All maximization (no changes):

        >>> acq = SomeMultiObjectiveAcquisition()
        >>> Y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> ref_point = [0.0, 0.0]
        >>> maximize = [True, True]
        >>> Y_max, ref_max = acq._prepare_for_maximization(Y, ref_point, maximize)
        >>> Y_max  # unchanged
        tensor([[1., 2.], [3., 4.]])
        >>> ref_max  # unchanged
        [0.0, 0.0]

        Mixed optimization (minimize objective 1):

        >>> maximize = [False, True]
        >>> Y_max, ref_max = acq._prepare_for_maximization(Y, ref_point, maximize)
        >>> Y_max  # first column negated
        tensor([[-1., 2.], [-3., 4.]])
        >>> ref_max  # first element negated
        [-0.0, 0.0]
        """
        Y_max = Y.clone()
        ref_point_max = list(ref_point)

        for i, should_maximize in enumerate(maximize):
            if not should_maximize:
                Y_max[:, i] = -Y_max[:, i]
                ref_point_max[i] = -ref_point_max[i]

        return Y_max, ref_point_max

    def _validate_dtype(self, X_baseline: torch.Tensor, Y: torch.Tensor) -> None:
        """Validate that input tensors have float64 dtype.

        BoTorch GP models and acquisition functions require float64 tensors for
        numerical stability during optimization. This method raises an error if
        the inputs have a different dtype.

        Parameters
        ----------
        X_baseline : torch.Tensor
            Input feature tensor to validate.
        Y : torch.Tensor
            Objective values tensor to validate.

        Raises
        ------
        ValueError
            If X_baseline or Y is not torch.float64.
        """
        if X_baseline.dtype != torch.float64:
            raise ValueError(
                f"X_baseline must be torch.float64, got {X_baseline.dtype}"
            )
        if Y.dtype != torch.float64:
            raise ValueError(f"Y must be torch.float64, got {Y.dtype}")

    @abstractmethod
    def build(
        self,
        model: Model,
        X_baseline: torch.Tensor,
        Y: torch.Tensor,
        ref_point: list[float],
        maximize: list[bool],
    ) -> AcquisitionFunction:
        """Build a BoTorch-compatible multi-objective acquisition function.

        Constructs an acquisition function instance configured with the builder's
        hyperparameters and the provided model, observed data, and objective settings.

        Parameters
        ----------
        model : Model
            A fitted BoTorch multi-output model (e.g., ModelListGP or
            MultiTaskGP) with a posterior() method. The model should predict
            all objectives and be trained on observed data before calling build().
        X_baseline : torch.Tensor
            Observed input points, shape (n_samples, n_features), dtype float64.
            Used by some acquisition functions (like qNEHVI) to compute the
            current Pareto frontier and cell decomposition for hypervolume
            calculation.
        Y : torch.Tensor
            Observed objective values, shape (n_samples, n_objectives), dtype
            float64. Each row contains the objective values for one observation.
            Used to determine the current Pareto frontier and compute hypervolume
            improvement.
        ref_point : list[float]
            Reference point for hypervolume calculation, length n_objectives.
            Must be dominated by all Pareto-optimal points (i.e., worse than
            the worst acceptable objective values). The hypervolume is computed
            as the volume between the Pareto frontier and this reference point.
        maximize : list[bool]
            Optimization direction for each objective, length n_objectives.
            If maximize[i] is True, objective i is maximized; if False, minimized.
            Note: BoTorch internally assumes maximization, so objectives to be
            minimized should be negated before building the acquisition function.

        Returns
        -------
        AcquisitionFunction
            A BoTorch-compatible acquisition function ready for optimization.
            The returned function:
            - Has forward(X) accepting shape (batch, q, d)
            - Returns shape (batch,) with acquisition values
            - Can be used directly with optimize_acqf

        Examples
        --------
        >>> ehvi_builder = ExpectedHypervolumeImprovement()
        >>> acqf = ehvi_builder.build(
        ...     model=mogp,
        ...     X_baseline=torch.tensor([[0.2, 0.3], [0.5, 0.6], [0.8, 0.4]]),
        ...     Y=torch.tensor([[0.5, 0.3], [0.8, 0.6], [0.4, 0.9]]),
        ...     ref_point=[0.0, 0.0],
        ...     maximize=[True, True],
        ... )
        >>> # Evaluate acquisition at candidate points
        >>> X = torch.tensor([[[0.5, 0.5]]])  # shape (1, 1, 2)
        >>> acq_value = acqf(X)  # shape (1,)
        """
        ...
