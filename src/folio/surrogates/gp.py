"""Gaussian Process surrogate model using BoTorch."""

from typing import Literal

import numpy as np

from folio.surrogates.base import Surrogate


class GPSurrogate(Surrogate):
    """Gaussian Process surrogate model using BoTorch's SingleTaskGP.

    This surrogate wraps BoTorch's SingleTaskGP to provide a scikit-learn-style
    interface for Bayesian optimization. Supports configurable kernels and
    automatic input/output normalization.

    Parameters
    ----------
    noise : float, default=1e-4
        Observation noise variance added to the diagonal of the kernel matrix.
        Small positive values improve numerical stability. Larger values indicate
        more noisy observations.
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
    noise : float
        The observation noise variance.
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

    Examples
    --------
    Basic usage with defaults:

    >>> X_train = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    >>> y_train = np.sin(2 * np.pi * X_train).ravel()
    >>> gp = GPSurrogate()
    >>> gp.fit(X_train, y_train)
    <GPSurrogate object>
    >>> mean, std = gp.predict(np.array([[0.125], [0.375]]))

    Custom configuration:

    >>> gp = GPSurrogate(
    ...     noise=1e-3,
    ...     kernel="rbf",
    ...     ard=False,
    ...     normalize_inputs=False,
    ... )

    Rough Matérn kernel for non-smooth functions:

    >>> gp = GPSurrogate(kernel="matern", nu=0.5)

    Notes
    -----
    The GP hyperparameters (lengthscales, outputscale) are optimized during
    fit() using L-BFGS-B to maximize the marginal log-likelihood.

    References
    ----------
    .. [1] Balandat, M., et al. "BoTorch: A Framework for Efficient Monte-Carlo
           Bayesian Optimization." NeurIPS 2020.
    .. [2] Rasmussen, C. E. and Williams, C. K. I. "Gaussian Processes for
           Machine Learning." MIT Press, 2006.
    """

    def __init__(
        self,
        noise: float = 1e-4,
        kernel: Literal["matern", "rbf"] = "matern",
        nu: float = 2.5,
        ard: bool = True,
        normalize_inputs: bool = True,
        normalize_outputs: bool = True,
    ):
        """Initialize the GP surrogate.

        Parameters
        ----------
        noise : float, default=1e-4
            Observation noise variance. Must be non-negative.
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
            If noise is negative.
        ValueError
            If kernel is not "matern" or "rbf".
        ValueError
            If nu is not one of 0.5, 1.5, or 2.5.
        """
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
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
        GPSurrogate
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If X and y have incompatible shapes (different number of samples).
        ValueError
            If X has fewer than 1 sample.

        Notes
        -----
        If normalize_inputs=True, input bounds are computed from training data
        and used to scale inputs to [0, 1]. If normalize_outputs=True, outputs
        are standardized to zero mean and unit variance.

        The model uses BoTorch's fit_gpytorch_mll to optimize hyperparameters
        via L-BFGS-B, maximizing the marginal log-likelihood.

        Examples
        --------
        >>> gp = GPSurrogate()
        >>> gp.fit(X_train, y_train)
        <GPSurrogate object>

        >>> # Method chaining
        >>> mean, std = GPSurrogate().fit(X_train, y_train).predict(X_test)
        """
        raise NotImplementedError

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
            If X has wrong number of features (doesn't match training data).

        Notes
        -----
        If normalize_inputs=True, inputs are normalized using the bounds
        computed during fit(). If normalize_outputs=True, predictions are
        transformed back to the original scale.

        Standard deviation is computed from the posterior variance:
        std = sqrt(variance).

        Examples
        --------
        >>> gp = GPSurrogate().fit(X_train, y_train)
        >>> mean, std = gp.predict(X_test)
        >>> mean.shape
        (10,)
        >>> np.all(std >= 0)
        True
        """
        raise NotImplementedError
