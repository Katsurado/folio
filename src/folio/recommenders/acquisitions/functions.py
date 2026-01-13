"""BoTorch-compatible acquisition function implementations."""

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch import Tensor
from torch.distributions import Normal

from folio.recommenders.acquisitions.base import Acquisition


class _EIAcquisition(AcquisitionFunction):
    """Inner BoTorch-compatible Expected Improvement acquisition function.

    This class implements the forward() method required by BoTorch's optimization
    routines. It should not be instantiated directly; use ExpectedImprovement.build()
    instead.

    Parameters
    ----------
    model : Model
        A fitted BoTorch model with posterior() method.
    best_f : float
        Best observed target value so far.
    xi : float
        Exploration parameter (margin for improvement).
    maximize : bool
        If True, seek higher values; if False, seek lower values.

    Notes
    -----
    Use torch.distributions.Normal(0, 1) for computing:
    - Phi(Z): standard normal CDF via .cdf(Z)
    - phi(Z): standard normal PDF via .log_prob(Z).exp()

    Register best_f and xi as buffers using self.register_buffer().
    """

    def __init__(
        self,
        model: Model,
        best_f: float,
        xi: float,
        maximize: bool,
    ):
        """Initialize the EI acquisition function.

        Should call super().__init__(model=model) and register buffers for
        best_f and xi using self.register_buffer().
        """
        super().__init__(model=model)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("xi", torch.as_tensor(xi))
        self.maximize = maximize

    def forward(self, X: Tensor) -> Tensor:
        """Compute Expected Improvement at candidate points.

        Parameters
        ----------
        X : Tensor, shape (batch, q, d), dtype float64
            Candidate points to evaluate. batch is the number of batches,
            q is the number of candidates per batch (q-batch), d is input dimension.

        Returns
        -------
        Tensor, shape (batch,)
            EI values for each batch, summed over the q dimension.

        Raises
        ------
        ValueError
            If X is not torch.float64.

        Notes
        -----
        Implementation steps:

        1. Get posterior from model: posterior = self.model.posterior(X)
        2. Extract mean and std:
           - mean = posterior.mean.squeeze(-1)  # shape (batch, q)
           - std = posterior.variance.sqrt().squeeze(-1)  # shape (batch, q)

        3. Compute improvement direction:
           - maximize: improvement = mean - best_f - xi
           - minimize: improvement = best_f - mean - xi

        4. Compute Z = improvement / std (handle std near 0 with small epsilon)

        5. Compute EI = std * (Z * Phi(Z) + phi(Z))
           where Phi = CDF, phi = PDF of standard normal

        6. Set EI = 0 where std is effectively zero

        7. Sum over q dimension: return ei.sum(dim=-1)
        """
        if X.dtype != torch.float64:
            raise ValueError(f"X must be torch.float64, got {X.dtype}")

        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)

        if self.maximize:
            improvement = mean - self.best_f - self.xi
        else:
            improvement = self.best_f - mean - self.xi

        eps = 10e-9

        Z = improvement / (std + eps)

        norm = Normal(0, 1)

        ei = std * (Z * norm.cdf(Z) + norm.log_prob(Z).exp())
        ei = torch.where(std > 1e-6, ei, 0.0)
        ei = ei.sum(dim=-1)
        return ei


class ExpectedImprovement(Acquisition):
    """Expected Improvement acquisition function builder.

    EI balances exploration and exploitation by computing the expected value
    of improvement over the current best observation. Points with high predicted
    mean (exploitation) or high uncertainty (exploration) receive high scores.

    Parameters
    ----------
    xi : float, default=0.01
        Exploration-exploitation trade-off parameter. Higher values encourage
        more exploration by requiring larger expected improvements. A value of
        0 gives pure EI; positive values add a "margin" that must be exceeded.

    Attributes
    ----------
    xi : float
        The exploration parameter.

    Notes
    -----
    For maximization, the Expected Improvement is computed as:

        Z = (mu - best_f - xi) / sigma
        EI = sigma * (Z * Phi(Z) + phi(Z))

    For minimization, the improvement direction is reversed:

        Z = (best_f - mu - xi) / sigma
        EI = sigma * (Z * Phi(Z) + phi(Z))

    where Phi is the standard normal CDF and phi is the standard normal PDF.

    When sigma is near 0, EI returns 0 (no uncertainty means no expected
    improvement beyond the point estimate).

    Examples
    --------
    >>> ei_builder = ExpectedImprovement(xi=0.01)
    >>> acqf = ei_builder.build(model=fitted_gp, best_f=0.5, maximize=True)
    >>> # Use with optimize_acqf
    >>> from botorch.optim import optimize_acqf
    >>> candidates, values = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5)

    Reference: Jones, Schonlau, Welch (1998), Efficient Global Optimization
    of Expensive Black-Box Functions.
    """

    def __init__(self, xi: float = 0.01):
        """Initialize Expected Improvement builder.

        Parameters
        ----------
        xi : float, default=0.01
            Exploration-exploitation trade-off parameter. Must be non-negative.
            Higher values favor exploration.

        Raises
        ------
        ValueError
            If xi is negative.
        """
        if xi < 0:
            raise ValueError("xi must be non-negative")
        self.xi = xi

    def build(
        self,
        model: Model,
        best_f: float,
        maximize: bool,
    ) -> AcquisitionFunction:
        """Build a BoTorch-compatible EI acquisition function.

        Parameters
        ----------
        model : Model
            A fitted BoTorch model (e.g., SingleTaskGP).
        best_f : float
            Best observed target value so far.
        maximize : bool
            If True, seek higher values; if False, seek lower values.

        Returns
        -------
        AcquisitionFunction
            A BoTorch-compatible EI acquisition function (_EIAcquisition).
        """
        return _EIAcquisition(model, best_f, self.xi, maximize)


class _UCBAcquisition(AcquisitionFunction):
    """Inner BoTorch-compatible Upper Confidence Bound acquisition function.

    This class implements the forward() method required by BoTorch's optimization
    routines. It should not be instantiated directly; use UpperConfidenceBound.build()
    instead.

    Parameters
    ----------
    model : Model
        A fitted BoTorch model with posterior() method.
    beta : float
        Exploration parameter controlling confidence bound width.
    maximize : bool
        If True, seek higher values; if False, seek lower values.

    Notes
    -----
    Register beta as a buffer using self.register_buffer().
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        maximize: bool,
    ):
        """Initialize the UCB acquisition function.

        Should call super().__init__(model=model) and register buffer for beta.
        """
        super().__init__(model=model)
        self.maximize = maximize
        self.register_buffer("beta", torch.as_tensor(beta))

    def forward(self, X: Tensor) -> Tensor:
        """Compute Upper Confidence Bound at candidate points.

        Parameters
        ----------
        X : Tensor, shape (batch, q, d), dtype float64
            Candidate points to evaluate.

        Returns
        -------
        Tensor, shape (batch,)
            UCB values for each batch, summed over the q dimension.

        Raises
        ------
        ValueError
            If X is not torch.float64.
        """
        if X.dtype != torch.float64:
            raise ValueError(f"X must be torch.float64, got {X.dtype}")

        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)

        if self.maximize:
            ucb = mean + self.beta * std
        else:
            ucb = -mean + self.beta * std

        ucb = ucb.sum(-1)
        return ucb


class UpperConfidenceBound(Acquisition):
    """Upper Confidence Bound acquisition function builder.

    UCB provides an optimistic estimate of the objective value by adding
    a multiple of the standard deviation to the predicted mean. This naturally
    balances exploitation (high mean) with exploration (high uncertainty).

    Parameters
    ----------
    beta : float, default=2.0
        Exploration parameter controlling the width of the confidence bound.
        Higher values favor exploration (more weight on uncertainty).
        Common choices: 2.0 (default), or schedule based on iteration.

    Attributes
    ----------
    beta : float
        The exploration parameter.

    Notes
    -----
    For maximization:
        UCB = mu + beta * sigma

    For minimization (we want to find low values, so we flip the sign of mu
    but still add uncertainty to encourage exploration):
        UCB = -mu + beta * sigma

    In both cases, higher UCB scores indicate more promising candidates.

    Examples
    --------
    >>> ucb_builder = UpperConfidenceBound(beta=2.0)
    >>> acqf = ucb_builder.build(model=fitted_gp, best_f=0.5, maximize=True)
    >>> # Use with optimize_acqf
    >>> from botorch.optim import optimize_acqf
    >>> candidates, values = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5)

    Reference: Srinivas et al. (2010), Gaussian Process Optimization in the
    Bandit Setting: No Regret and Experimental Design.
    """

    def __init__(self, beta: float = 2.0):
        """Initialize Upper Confidence Bound builder.

        Parameters
        ----------
        beta : float, default=2.0
            Exploration parameter. Must be non-negative. Higher values
            increase the influence of uncertainty on the score.

        Raises
        ------
        ValueError
            If beta is negative.
        """
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.beta = beta

    def build(
        self,
        model: Model,
        best_f: float,
        maximize: bool,
    ) -> AcquisitionFunction:
        """Build a BoTorch-compatible UCB acquisition function.

        Parameters
        ----------
        model : Model
            A fitted BoTorch model (e.g., SingleTaskGP).
        best_f : float
            Best observed target value so far (unused by UCB, included for
            interface consistency).
        maximize : bool
            If True, seek higher values; if False, seek lower values.

        Returns
        -------
        AcquisitionFunction
            A BoTorch-compatible UCB acquisition function (_UCBAcquisition).
        """
        return _UCBAcquisition(model, self.beta, maximize)
