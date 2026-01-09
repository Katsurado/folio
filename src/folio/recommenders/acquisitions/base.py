"""Abstract base class for acquisition function builders."""

from abc import ABC, abstractmethod

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model


class Acquisition(ABC):
    """Abstract base class for building BoTorch-compatible acquisition functions.

    Acquisition instances store hyperparameters (e.g., xi for EI, beta for UCB)
    and provide a `build()` method that returns a BoTorch AcquisitionFunction ready
    for use with optimize_acqf.

    The builder pattern separates configuration (hyperparameters) from construction
    (which requires a fitted model and best observed value). This enables:

    - Reusing the same configuration across multiple optimization iterations
    - Swapping acquisition functions without changing the optimization loop
    - Clean separation between user-facing config and BoTorch internals

    Notes
    -----
    Subclasses must implement `build()` to return an AcquisitionFunction that:

    - Inherits from botorch.acquisition.AcquisitionFunction
    - Implements forward(X: Tensor) -> Tensor
    - Accepts X with shape (batch, q, d) and returns shape (batch,)

    Examples
    --------
    Implementing a custom acquisition:

    >>> class MyAcquisition(Acquisition):
    ...     def __init__(self, kappa: float = 2.0):
    ...         if kappa < 0:
    ...             raise ValueError("kappa must be non-negative")
    ...         self.kappa = kappa
    ...
    ...     def build(self, model, best_f, maximize):
    ...         return _MyAcqf(model, best_f, self.kappa, maximize)

    Using an acquisition in an optimization loop:

    >>> ei = ExpectedImprovement(xi=0.01)
    >>> acqf = ei.build(model=fitted_gp, best_f=0.5, maximize=True)
    >>> # Use acqf with optimize_acqf for gradient-based optimization
    >>> candidates, acq_values = optimize_acqf(acqf, bounds=bounds, ...)

    Reference: Frazier (2018), A Tutorial on Bayesian Optimization.
    """

    @abstractmethod
    def build(
        self,
        model: Model,
        best_f: float,
        maximize: bool,
    ) -> AcquisitionFunction:
        """Build a BoTorch-compatible acquisition function.

        Constructs an acquisition function instance configured with the builder's
        hyperparameters and the provided model/best_f/maximize settings.

        Parameters
        ----------
        model : Model
            A fitted BoTorch model (e.g., SingleTaskGP) with a posterior() method.
            The model should be trained on observed data before calling build().
        best_f : float
            Best observed target value so far. Used by improvement-based acquisition
            functions (like EI) as the reference point for computing improvement.
        maximize : bool
            If True, optimize to find higher target values.
            If False, optimize to find lower target values.

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
        >>> ei_builder = ExpectedImprovement(xi=0.01)
        >>> acqf = ei_builder.build(model=gp, best_f=1.5, maximize=True)
        >>> # Evaluate acquisition at candidate points
        >>> X = torch.tensor([[[0.5, 0.5]]])  # shape (1, 1, 2)
        >>> acq_value = acqf(X)  # shape (1,)
        """
        ...
