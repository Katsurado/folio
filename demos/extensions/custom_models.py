"""Custom surrogate, acquisition, and recommender implementations for Folio.

This module demonstrates how to extend Folio with custom models:
- NNEnsembleSurrogate: Neural network ensemble for uncertainty estimation
- ProbabilityOfImprovement: Classic PI acquisition function
- NNEnsembleRecommender: Complete recommender using NN ensemble (works with Folio API)

These can be used directly with Folio's optimization workflow or as templates
for implementing more sophisticated custom models.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch import Tensor
from torch.distributions import Normal

from folio.exceptions import NotFittedError
from folio.recommenders.acquisitions.base import Acquisition
from folio.recommenders.base import Recommender
from folio.surrogates.base import Surrogate

if TYPE_CHECKING:
    from folio.core.project import Project


class _MLP(nn.Module):
    """Simple 2-layer MLP for ensemble members.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        Number of hidden units.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (n_samples, input_dim)
            Input features.

        Returns
        -------
        Tensor, shape (n_samples, 1)
            Predictions.
        """
        return self.net(x)


class NNEnsembleSurrogate(Surrogate):
    """Neural network ensemble surrogate for Bayesian optimization.

    Uses an ensemble of small MLPs to provide predictions and uncertainty
    estimates. Uncertainty is computed from the disagreement (standard
    deviation) among ensemble members.

    Parameters
    ----------
    n_members : int, default=5
        Number of ensemble members.
    hidden_dim : int, default=32
        Hidden layer size for each MLP.
    n_epochs : int, default=200
        Training epochs per ensemble member.
    lr : float, default=0.01
        Learning rate for Adam optimizer.

    Attributes
    ----------
    _members : list[_MLP]
        Trained ensemble members. Empty until fit() is called.
    _input_dim : int | None
        Input dimensionality. Set during fit().

    Examples
    --------
    >>> surrogate = NNEnsembleSurrogate(n_members=5, hidden_dim=32)
    >>> surrogate.fit(X_train, y_train)
    >>> mean, std = surrogate.predict(X_test)
    >>> # std reflects ensemble disagreement
    """

    def __init__(
        self,
        n_members: int = 5,
        hidden_dim: int = 32,
        n_epochs: int = 200,
        lr: float = 0.01,
    ):
        super().__init__()
        self.n_members = n_members
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self._members: list[_MLP] = []
        self._input_dim: int | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NNEnsembleSurrogate":
        """Fit the ensemble to training data.

        Trains each ensemble member independently with different random
        initializations. Input features are standardized and target values
        are normalized for training stability.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input features.
        y : np.ndarray, shape (n_samples,) or (n_samples, 1)
            Training target values.

        Returns
        -------
        NNEnsembleSurrogate
            Returns self for method chaining.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples, "
                f"got {X.shape[0]} and {y.shape[0]}"
            )

        self._input_dim = X.shape[1]

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        # Normalize targets for training stability
        self._y_mean = float(y_t.mean())
        self._y_std = float(y_t.std()) + 1e-6
        y_normalized = (y_t - self._y_mean) / self._y_std

        self._members = []
        for _ in range(self.n_members):
            mlp = _MLP(self._input_dim, self.hidden_dim)
            optimizer = torch.optim.Adam(mlp.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()

            for _ in range(self.n_epochs):
                optimizer.zero_grad()
                pred = mlp(X_t)
                loss = loss_fn(pred, y_normalized)
                loss.backward()
                optimizer.step()

            mlp.eval()
            self._members.append(mlp)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and uncertainty at candidate points.

        Uncertainty is computed as the standard deviation across ensemble
        member predictions. Higher disagreement indicates higher uncertainty.

        Parameters
        ----------
        X : np.ndarray, shape (n_candidates, n_features)
            Candidate points to evaluate.

        Returns
        -------
        mean : np.ndarray, shape (n_candidates,)
            Mean prediction across ensemble members.
        std : np.ndarray, shape (n_candidates,)
            Standard deviation across ensemble members (uncertainty).

        Raises
        ------
        NotFittedError
            If called before fit().
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit() before predict()")

        X_t = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            preds = torch.stack([mlp(X_t) for mlp in self._members])

        # Shape: (n_members, n_candidates, 1) -> (n_members, n_candidates)
        preds = preds.squeeze(-1)

        # Denormalize predictions
        preds = preds * self._y_std + self._y_mean

        mean = preds.mean(dim=0).numpy()
        std = preds.std(dim=0).numpy()

        # Ensure std is never zero
        std = np.maximum(std, 1e-6)

        return mean, std


class _PIAcquisition(AcquisitionFunction):
    """Inner BoTorch-compatible Probability of Improvement acquisition.

    Parameters
    ----------
    model : Model
        A fitted BoTorch model with posterior() method.
    best_f : float
        Best observed target value so far.
    xi : float
        Exploration parameter (improvement threshold).
    maximize : bool
        If True, seek higher values; if False, seek lower values.
    """

    def __init__(
        self,
        model: Model,
        best_f: float,
        xi: float,
        maximize: bool,
    ):
        super().__init__(model=model)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("xi", torch.as_tensor(xi))
        self.maximize = maximize

    def forward(self, X: Tensor) -> Tensor:
        """Compute Probability of Improvement at candidate points.

        Parameters
        ----------
        X : Tensor, shape (batch, q, d)
            Candidate points to evaluate.

        Returns
        -------
        Tensor, shape (batch,)
            PI values for each batch, summed over q dimension.
        """
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)

        if self.maximize:
            improvement = mean - self.best_f - self.xi
        else:
            improvement = self.best_f - mean - self.xi

        eps = 1e-9
        Z = improvement / (std + eps)

        norm = Normal(0, 1)
        pi = norm.cdf(Z)

        # Set PI = 0 where std is effectively zero
        pi = torch.where(std > 1e-6, pi, torch.zeros_like(pi))

        return pi.sum(dim=-1)


class ProbabilityOfImprovement(Acquisition):
    """Probability of Improvement acquisition function builder.

    PI computes the probability that a point will improve over the current
    best observation. Unlike EI, it doesn't consider the magnitude of
    improvement, only the probability. This makes it more exploitative.

    Parameters
    ----------
    xi : float, default=0.0
        Exploration parameter. Positive values require improvements larger
        than xi to count, encouraging exploration.

    Notes
    -----
    For maximization:
        PI = Phi((mu - best_f - xi) / sigma)

    For minimization:
        PI = Phi((best_f - mu - xi) / sigma)

    where Phi is the standard normal CDF.

    Examples
    --------
    >>> pi_builder = ProbabilityOfImprovement(xi=0.01)
    >>> acqf = pi_builder.build(model=fitted_gp, best_f=0.5, maximize=True)
    >>> candidates, values = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5)

    Reference: Kushner (1964), A New Method of Locating the Maximum Point
    of an Arbitrary Multipeak Curve in the Presence of Noise.
    """

    def __init__(self, xi: float = 0.0):
        if xi < 0:
            raise ValueError("xi must be non-negative")
        self.xi = xi

    def build(
        self,
        model: Model,
        best_f: float,
        maximize: bool,
    ) -> AcquisitionFunction:
        """Build a BoTorch-compatible PI acquisition function.

        Parameters
        ----------
        model : Model
            A fitted BoTorch model.
        best_f : float
            Best observed target value so far.
        maximize : bool
            If True, seek higher values; if False, seek lower values.

        Returns
        -------
        AcquisitionFunction
            A BoTorch-compatible PI acquisition function.
        """
        return _PIAcquisition(model, best_f, self.xi, maximize)


class NNEnsembleRecommender(Recommender):
    """Custom recommender using neural network ensemble.

    This recommender can be used directly with Folio's high-level API by
    injecting it into the recommender cache. It uses an NN ensemble for
    uncertainty estimation and UCB acquisition for candidate selection.

    Parameters
    ----------
    project : Project
        The project defining input specs, bounds, and targets.
    n_members : int, default=5
        Number of ensemble members.
    hidden_dim : int, default=32
        Hidden layer size for each MLP.
    n_epochs : int, default=200
        Training epochs per member.
    n_initial : int, default=3
        Number of random samples before using the model.
    n_candidates : int, default=100
        Number of random candidates to evaluate for selection.
    beta : float, default=2.0
        UCB exploration parameter (higher = more exploration).

    Examples
    --------
    Using with Folio's high-level API:

    >>> from folio.api import Folio
    >>> from extensions.custom_models import NNEnsembleRecommender
    >>>
    >>> folio = Folio(db_path="test.db")
    >>> folio.create_project(...)
    >>>
    >>> # Inject custom recommender
    >>> project = folio.get_project("my_project")
    >>> folio._recommenders["my_project"] = NNEnsembleRecommender(project)
    >>>
    >>> # Now suggest() uses your custom recommender
    >>> suggestion = folio.suggest("my_project")
    """

    def __init__(
        self,
        project: "Project",
        n_members: int = 5,
        hidden_dim: int = 32,
        n_epochs: int = 200,
        n_initial: int = 3,
        n_candidates: int = 100,
        beta: float = 2.0,
    ):
        super().__init__(project)
        self.n_members = n_members
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.n_initial = n_initial
        self.n_candidates = n_candidates
        self.beta = beta
        self._surrogate: NNEnsembleSurrogate | None = None

    @property
    def surrogate(self) -> NNEnsembleSurrogate | None:
        """The fitted surrogate model, or None if not yet fitted."""
        return self._surrogate

    def recommend_from_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        maximize: list[bool],
    ) -> np.ndarray:
        """Suggest next experiment using NN ensemble + UCB.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training inputs from previous experiments.
        y : np.ndarray, shape (n_samples, n_objectives)
            Training targets. Currently only supports single-objective (n_objectives=1).
        bounds : np.ndarray, shape (2, n_features)
            Row 0 = lower bounds, row 1 = upper bounds.
        maximize : list[bool]
            Whether to maximize each objective.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Suggested input values.

        Notes
        -----
        Uses random sampling with UCB-based selection instead of gradient-based
        optimization, since the NN ensemble doesn't expose a BoTorch model.
        """
        # Return random sample if not enough data
        if len(X) < self.n_initial:
            self._surrogate = None
            return self.random_sample_from_bounds(bounds)

        # Only supports single-objective for simplicity
        if y.ndim == 2:
            y_flat = y[:, 0]
        else:
            y_flat = y

        # Fit surrogate
        self._surrogate = NNEnsembleSurrogate(
            n_members=self.n_members,
            hidden_dim=self.hidden_dim,
            n_epochs=self.n_epochs,
        )
        self._surrogate.fit(X, y_flat)

        # Generate random candidates
        candidates = np.random.uniform(
            bounds[0], bounds[1], size=(self.n_candidates, X.shape[1])
        )

        # Get predictions
        mean, std = self._surrogate.predict(candidates)

        # Compute UCB acquisition
        if maximize[0]:
            ucb = mean + self.beta * std
        else:
            ucb = -mean + self.beta * std

        # Select best candidate
        best_idx = np.argmax(ucb)
        return candidates[best_idx]
