"""Abstract base class for active learning acquisition function builders."""

from abc import ABC, abstractmethod

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model


class ActiveLearningAcquisition(ABC):
    """Abstract base class for building active learning acquisition functions.

    Active learning acquisition functions guide the search for informative
    observations to reduce model uncertainty, rather than optimizing a target
    value. These acquisitions are useful for:

    - Pure exploration / space-filling designs
    - Model calibration before optimization
    - Scientific discovery where understanding the response surface is the goal
    - Reducing epistemic uncertainty in specific regions

    Unlike optimization-focused acquisitions (EI, UCB) that balance exploitation
    and exploration, active learning acquisitions focus purely on exploration /
    uncertainty reduction. They do not require a best observed value or
    optimization direction.

    This class follows the builder pattern: instances store hyperparameters and
    provide a `build()` method that returns a BoTorch AcquisitionFunction ready
    for use with optimize_acqf. The builder pattern enables:

    - Reusing the same configuration across multiple iterations
    - Swapping acquisition functions without changing the sampling loop
    - Clean separation between user-facing config and BoTorch internals

    Notes
    -----
    Subclasses must implement `build()` to return an AcquisitionFunction that:

    - Inherits from botorch.acquisition.AcquisitionFunction
    - Implements forward(X: Tensor) -> Tensor
    - Accepts X with shape (batch, q, d) and returns shape (batch,)

    Examples
    --------
    Implementing a custom active learning acquisition:

    >>> class MyALAcquisition(ActiveLearningAcquisition):
    ...     def __init__(self, temperature: float = 1.0):
    ...         self.temperature = temperature
    ...
    ...     def build(self, model):
    ...         return _MyALAcqf(model, self.temperature)

    Using an active learning acquisition in a sampling loop:

    >>> pv = PosteriorVariance()
    >>> acqf = pv.build(model=fitted_gp)
    >>> # Use acqf with optimize_acqf for gradient-based optimization
    >>> candidates, acq_values = optimize_acqf(acqf, bounds=bounds, ...)

    Reference: Settles (2009), Active Learning Literature Survey.
    """

    @abstractmethod
    def build(self, model: Model) -> AcquisitionFunction:
        """Build a BoTorch-compatible active learning acquisition function.

        Constructs an acquisition function instance configured with the builder's
        hyperparameters and the provided model. Unlike optimization acquisitions,
        active learning acquisitions only need the model (no best_f or maximize).

        Parameters
        ----------
        model : Model
            A fitted BoTorch model (e.g., SingleTaskGP, MultiTaskGP) with a
            posterior() method. The model should be trained on observed data
            before calling build(). Can be single-output or multi-output.

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
        >>> pv_builder = PosteriorVariance()
        >>> acqf = pv_builder.build(model=fitted_gp)
        >>> # Evaluate acquisition at candidate points
        >>> X = torch.tensor([[[0.5, 0.5]]])  # shape (1, 1, 2)
        >>> acq_value = acqf(X)  # shape (1,)
        """
        ...
