"""Active learning acquisition function implementations."""

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch import Tensor

from folio.recommenders.acquisitions.al_base import ActiveLearningAcquisition


class _PosteriorVarianceAcquisition(AcquisitionFunction):
    """Inner BoTorch-compatible Posterior Variance acquisition function.

    This class implements the forward() method required by BoTorch's optimization
    routines. It should not be instantiated directly; use PosteriorVariance.build()
    instead.

    The acquisition value at each point is the sum of posterior variances across
    all model outputs. For single-output models, this is simply the variance.
    For multi-output models (e.g., MultiTaskGP), this sums variances across all
    tasks/outputs, encouraging exploration where any output is uncertain.

    Parameters
    ----------
    model : Model
        A fitted BoTorch model with posterior() method. Can be single-output
        (SingleTaskGP) or multi-output (MultiTaskGP, ModelListGP).

    Notes
    -----
    The posterior variance acquisition:
    - Is always non-negative (variance >= 0)
    - Is highest where the model is most uncertain
    - Decreases as observations are added nearby
    - Does not depend on any target value or optimization direction
    """

    def __init__(self, model: Model):
        """Initialize the Posterior Variance acquisition function.

        Parameters
        ----------
        model : Model
            A fitted BoTorch model with posterior() method.
        """
        super().__init__(model=model)

    def forward(self, X: Tensor) -> Tensor:
        """Compute sum of posterior variances at candidate points.

        Parameters
        ----------
        X : Tensor, shape (batch, q, d), dtype float64
            Candidate points to evaluate. batch is the number of batches,
            q is the number of candidates per batch (q-batch), d is input dimension.

        Returns
        -------
        Tensor, shape (batch,)
            Sum of posterior variances for each batch, summed over both the
            q dimension and output dimensions (for multi-output models).

        Raises
        ------
        ValueError
            If X is not torch.float64.

        Notes
        -----
        Implementation steps:

        1. Get posterior from model: posterior = self.model.posterior(X)
        2. Extract variance: variance = posterior.variance
           - Single-output: shape (batch, q, 1)
           - Multi-output: shape (batch, q, n_outputs)
        3. Sum over output dimension: variance.sum(dim=-1) -> shape (batch, q)
        4. Sum over q dimension: .sum(dim=-1) -> shape (batch,)
        """
        raise NotImplementedError


class PosteriorVariance(ActiveLearningAcquisition):
    """Posterior Variance acquisition function builder for active learning.

    PosteriorVariance selects points that maximize model uncertainty, measured
    as the sum of posterior variances across all outputs. This is a pure
    exploration strategy that does not consider any target value - it simply
    seeks to reduce epistemic uncertainty about the response surface.

    For multi-output models (e.g., MultiTaskGP with multiple tasks), the
    acquisition sums variances across all outputs, encouraging exploration
    where any output is uncertain. This is useful when you want to learn
    about all outputs simultaneously.

    Use Cases
    ---------
    - Initial space-filling before optimization (model calibration)
    - Scientific exploration where understanding the surface matters
    - Reducing model uncertainty in specific regions
    - Active learning for surrogate model construction

    Notes
    -----
    For single-output models, the acquisition is simply:

        acq(x) = Var[f(x)]

    For multi-output models with m outputs:

        acq(x) = sum_{i=1}^{m} Var[f_i(x)]

    Points near observed data will have low variance (GP interpolates
    through observations), while points far from observations will have
    high variance. The acquisition naturally decreases as observations
    are added.

    Examples
    --------
    >>> pv_builder = PosteriorVariance()
    >>> acqf = pv_builder.build(model=fitted_gp)
    >>> # Use with optimize_acqf
    >>> from botorch.optim import optimize_acqf
    >>> candidates, values = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5)

    Reference: MacKay (1992), Information-Based Objective Functions for
    Active Data Selection.
    """

    def build(self, model: Model) -> AcquisitionFunction:
        """Build a BoTorch-compatible Posterior Variance acquisition function.

        Parameters
        ----------
        model : Model
            A fitted BoTorch model (e.g., SingleTaskGP, MultiTaskGP, ModelListGP).
            Can be single-output or multi-output.

        Returns
        -------
        AcquisitionFunction
            A BoTorch-compatible acquisition function (_PosteriorVarianceAcquisition)
            that returns sum of posterior variances.
        """
        return _PosteriorVarianceAcquisition(model)
