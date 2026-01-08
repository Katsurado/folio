"""Gaussian Process surrogate model using BoTorch."""

from typing import Literal

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from folio.exceptions import NotFittedError
from folio.surrogates.base import Surrogate


class SingleTaskGPSurrogate(Surrogate):
    """Single-output Gaussian Process surrogate using BoTorch's SingleTaskGP.

    This surrogate wraps BoTorch's SingleTaskGP to provide a scikit-learn-style
    interface for Bayesian optimization. Supports configurable kernels and
    automatic input/output normalization. Use this for scalar optimization
    targets; for multi-output or mixed-task scenarios, use specialized GP
    surrogates (e.g., MultiTaskGP, HOGP).

    Parameters
    ----------
    kernel : {"matern", "rbf"}, default="matern"
        Covariance kernel type:
        - "matern": Matérn kernel (smoothness controlled by `nu`)
        - "rbf": Radial basis function (squared exponential) kernel
    nu : float, default=2.5
        Smoothness parameter for Matérn kernel. Common values:
        - 0.5: Equivalent to absolute exponential (rough, non-differentiable)
        - 1.5: Once differentiable
        - 2.5: Twice differentiable (default, good general choice)
        Ignored if kernel="rbf".
    ard : bool, default=True
        If True, use Automatic Relevance Determination (separate lengthscale
        per input dimension). If False, use a single shared lengthscale.
    normalize_inputs : bool, default=True
        If True, normalize inputs to [0, 1] range during fit/predict.
        Improves numerical stability for inputs with different scales.
    normalize_outputs : bool, default=True
        If True, standardize outputs to zero mean and unit variance.
        Improves optimization of GP hyperparameters.

    Attributes
    ----------
    kernel : str
        The kernel type ("matern" or "rbf").
    nu : float
        Matérn smoothness parameter.
    ard : bool
        Whether ARD is enabled.
    normalize_inputs : bool
        Whether input normalization is enabled.
    normalize_outputs : bool
        Whether output normalization is enabled.
    model : SingleTaskGP or None
        The fitted BoTorch GP model. None before fit() is called.
    n_features : int or None
        Number of input features. Set after fit().
    _is_fitted : bool
        Internal flag tracking whether fit() has been called.

    Examples
    --------
    Basic usage with defaults:

    >>> X_train = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    >>> y_train = np.sin(2 * np.pi * X_train).ravel()
    >>> gp = SingleTaskGPSurrogate()
    >>> gp.fit(X_train, y_train)
    <SingleTaskGPSurrogate object>
    >>> mean, std = gp.predict(np.array([[0.125], [0.375]]))

    Custom configuration:

    >>> gp = SingleTaskGPSurrogate(
    ...     kernel="rbf",
    ...     ard=False,
    ...     normalize_inputs=False,
    ... )

    Rough Matérn kernel for non-smooth functions:

    >>> gp = SingleTaskGPSurrogate(kernel="matern", nu=0.5)

    Notes
    -----
    The GP hyperparameters (lengthscales, outputscale, noise) are optimized
    during fit() using L-BFGS-B to maximize the marginal log-likelihood.

    References
    ----------
    .. [1] Balandat, M., et al. "BoTorch: A Framework for Efficient Monte-Carlo
           Bayesian Optimization." NeurIPS 2020.
    .. [2] Rasmussen, C. E. and Williams, C. K. I. "Gaussian Processes for
           Machine Learning." MIT Press, 2006.
    """

    def __init__(
        self,
        kernel: Literal["matern", "rbf"] = "matern",
        nu: float = 2.5,
        ard: bool = True,
        normalize_inputs: bool = True,
        normalize_outputs: bool = True,
    ):
        """Initialize the single-task GP surrogate.

        Parameters
        ----------
        kernel : {"matern", "rbf"}, default="matern"
            Covariance kernel type.
        nu : float, default=2.5
            Matérn smoothness parameter (0.5, 1.5, or 2.5).
        ard : bool, default=True
            Use Automatic Relevance Determination.
        normalize_inputs : bool, default=True
            Normalize inputs to [0, 1] range.
        normalize_outputs : bool, default=True
            Standardize outputs to zero mean, unit variance.

        Raises
        ------
        ValueError
            If kernel not in {"matern", "rbf"}.
        ValueError
            If nu not in {0.5, 1.5, 2.5}.
        """
        super().__init__()

        if kernel not in {"matern", "rbf"}:
            raise ValueError(
                f"Unknown kernel: {kernel}. Kernel should be 'matern' or 'rbf'"
            )
        if nu not in {0.5, 1.5, 2.5}:
            raise ValueError(f"Invalid nu: {nu}. nu should be in {{0.5, 1.5, 2.5}}")

        self.kernel = kernel
        self.nu = nu
        self.ard = ard
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SingleTaskGPSurrogate":
        """Fit the Gaussian Process model to training data.

        Converts numpy arrays to torch tensors, creates a SingleTaskGP model,
        and optimizes hyperparameters using marginal log-likelihood.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input features. Each row is an observation.
        y : np.ndarray, shape (n_samples,)
            Training target values.

        Returns
        -------
        SingleTaskGPSurrogate
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If X.shape[0] != y.shape[0] (different number of samples).
        ValueError
            If X.shape[0] < 1 (no training samples).

        Notes
        -----
        If normalize_inputs=True, input bounds are computed from training data
        and used to scale inputs to [0, 1]. If normalize_outputs=True, outputs
        are standardized to zero mean and unit variance.

        The model uses BoTorch's fit_gpytorch_mll to optimize hyperparameters
        via L-BFGS-B, maximizing the marginal log-likelihood.

        Examples
        --------
        >>> gp = SingleTaskGPSurrogate()
        >>> gp.fit(X_train, y_train)
        <SingleTaskGPSurrogate object>

        >>> # Method chaining
        >>> mean, std = SingleTaskGPSurrogate().fit(X_train, y_train).predict(X_test)
        """
        X = X[:, np.newaxis] if X.ndim == 1 else X

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples ({X.shape[0]}) "
                f"should match number of labels ({y.shape[0]})"
            )
        if X.shape[0] < 1:
            raise ValueError(f"Cannot fit model with {X.shape[0]} observations")

        X = torch.tensor(X, dtype=torch.float64)
        y = torch.tensor(y, dtype=torch.float64).unsqueeze(-1)

        self.n_features = X.shape[1]
        ard_features = self.n_features if self.ard else None

        if self.kernel == "matern":
            base_kernel = MaternKernel(nu=self.nu, ard_num_dims=ard_features)
        else:
            base_kernel = RBFKernel(ard_num_dims=ard_features)

        covar_module = ScaleKernel(base_kernel)

        if self.normalize_inputs:
            input_transform = Normalize(d=self.n_features)
        else:
            input_transform = None

        if self.normalize_outputs:
            output_transform = Standardize(m=1)
        else:
            output_transform = None

        self.model = SingleTaskGP(
            train_X=X,
            train_Y=y,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=output_transform,
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation at candidate points.

        Uses the fitted GP model to compute posterior mean and standard
        deviation at the given points.

        Parameters
        ----------
        X : np.ndarray, shape (n_candidates, n_features)
            Candidate points to evaluate. Must have same number of features
            as training data.

        Returns
        -------
        mean : np.ndarray, shape (n_candidates,)
            Posterior mean prediction at each candidate point.
        std : np.ndarray, shape (n_candidates,)
            Posterior standard deviation (uncertainty) at each point.
            Always non-negative.

        Raises
        ------
        NotFittedError
            If called before fit(). Must fit the model first.
        ValueError
            If X.shape[1] != n_features (feature dimension mismatch).

        Notes
        -----
        If normalize_inputs=True, inputs are normalized using the bounds
        computed during fit(). If normalize_outputs=True, predictions are
        transformed back to the original scale.

        Standard deviation is computed from the posterior variance:
        std = sqrt(variance).

        Examples
        --------
        >>> gp = SingleTaskGPSurrogate().fit(X_train, y_train)
        >>> mean, std = gp.predict(X_test)
        >>> mean.shape
        (10,)
        >>> np.all(std >= 0)
        True
        """
        X = X[:, np.newaxis] if X.ndim == 1 else X

        if not self._is_fitted:
            raise NotFittedError("Call fit() first before predict()")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: "
                f"fitted with {self.n_features} features, "
                f"but passed in new data with {X.shape[1]} features"
            )

        X = torch.tensor(X, dtype=torch.float64)

        posterior = self.model.posterior(X)
        mean = posterior.mean
        std = posterior.variance.sqrt()

        mean = mean.squeeze(-1).detach().numpy()
        std = std.squeeze(-1).detach().numpy()

        return mean, std
