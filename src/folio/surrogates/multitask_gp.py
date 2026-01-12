"""Multi-task Gaussian Process surrogate model using BoTorch."""

from typing import Literal

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from folio.exceptions import NotFittedError
from folio.surrogates.base import Surrogate
from folio.surrogates.transforms import TaskStandardize


class MultiTaskGPSurrogate(Surrogate):
    """Multi-output Gaussian Process surrogate using BoTorch's MultiTaskGP.

    This surrogate wraps BoTorch's MultiTaskGP to model correlated outputs via
    the Intrinsic Coregionalization Model (ICM) kernel. Use this for multi-output
    optimization where outputs are mechanistically linked (e.g., predicting
    molecular weight and conversion simultaneously).

    For single-output targets, use SingleTaskGPSurrogate instead.

    Parameters
    ----------
    kernel : {"matern", "rbf"}, default="matern"
        Covariance kernel type:
        - "matern": Matern kernel (smoothness controlled by `nu`)
        - "rbf": Radial basis function (squared exponential) kernel
    nu : float, default=2.5
        Smoothness parameter for Matern kernel. Common values:
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
        Matern smoothness parameter.
    ard : bool
        Whether ARD is enabled.
    normalize_inputs : bool
        Whether input normalization is enabled.
    normalize_outputs : bool
        Whether output normalization is enabled.
    model : MultiTaskGP or None
        The fitted BoTorch GP model. None before fit() is called.
    n_features : int or None
        Number of input features. Set after fit().
    n_tasks : int or None
        Number of output tasks. Set after fit().
    _is_fitted : bool
        Internal flag tracking whether fit() has been called.

    Examples
    --------
    Basic usage with correlated outputs:

    >>> X_train = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    >>> y_train = np.column_stack([
    ...     np.sin(2 * np.pi * X_train).ravel(),
    ...     np.cos(2 * np.pi * X_train).ravel()
    ... ])
    >>> gp = MultiTaskGPSurrogate()
    >>> gp.fit(X_train, y_train)
    <MultiTaskGPSurrogate object>
    >>> mean, std = gp.predict(np.array([[0.125], [0.375]]))
    >>> mean.shape
    (2, 2)

    Notes
    -----
    The MultiTaskGP models task correlations through the ICM kernel, which
    learns a task covariance matrix. This allows information sharing between
    tasks, improving predictions when outputs are correlated.

    References
    ----------
    .. [1] Bonilla, E. V., et al. "Multi-task Gaussian Process Prediction."
           NeurIPS 2008.
    .. [2] Balandat, M., et al. "BoTorch: A Framework for Efficient Monte-Carlo
           Bayesian Optimization." NeurIPS 2020.
    """

    def __init__(
        self,
        kernel: Literal["matern", "rbf"] = "matern",
        nu: float = 2.5,
        ard: bool = True,
        normalize_inputs: bool = True,
        normalize_outputs: bool = True,
    ):
        """Initialize the multi-task GP surrogate.

        Parameters
        ----------
        kernel : {"matern", "rbf"}, default="matern"
            Covariance kernel type.
        nu : float, default=2.5
            Matern smoothness parameter (0.5, 1.5, or 2.5).
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
        self.n_features = None
        self.n_tasks = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiTaskGPSurrogate":
        """Fit the multi-task Gaussian Process model to training data.

        Converts numpy arrays to torch tensors, reshapes data to multi-task
        format, creates a MultiTaskGP model, and optimizes hyperparameters
        using marginal log-likelihood.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input features. Each row is an observation.
        y : np.ndarray, shape (n_samples, n_tasks)
            Training target values. Each column is a different output task.
            Must have at least 2 tasks (columns).

        Returns
        -------
        MultiTaskGPSurrogate
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If y.ndim != 2 (must be 2D array).
        ValueError
            If y.shape[1] < 2 (use SingleTaskGPSurrogate for single output).
        ValueError
            If X.shape[0] != y.shape[0] (different number of samples).
        ValueError
            If X.shape[0] < 1 (no training samples).
        ValueError
            If X or y is not float64 dtype.

        Notes
        -----
        The data is reshaped to BoTorch's multi-task format where each
        observation-task pair becomes a separate row, with the task index
        appended as the last column of X.

        Examples
        --------
        >>> gp = MultiTaskGPSurrogate()
        >>> gp.fit(X_train, y_train)
        <MultiTaskGPSurrogate object>

        >>> # Method chaining
        >>> mean, std = MultiTaskGPSurrogate().fit(X_train, y_train).predict(X_test)
        """
        X = X[:, np.newaxis] if X.ndim == 1 else X

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples ({X.shape[0]}) "
                f"should match number of labels ({y.shape[0]})"
            )
        if X.shape[0] < 1:
            raise ValueError(f"Cannot fit model with {X.shape[0]} observations")
        if y.ndim != 2:
            raise ValueError("y must be an 2d array")
        if y.shape[1] < 2:
            raise ValueError("use SingleTaskGPSurrogate for single output")
        if X.dtype != np.float64:
            raise ValueError(f"X must be float64, got {X.dtype}")
        if y.dtype != np.float64:
            raise ValueError(f"y must be float64, got {y.dtype}")

        X = torch.tensor(X, dtype=torch.float64)
        y = torch.tensor(y, dtype=torch.float64)

        self.n_features = X.shape[1]
        ard_features = self.n_features if self.ard else None

        if self.kernel == "matern":
            base_kernel = MaternKernel(nu=self.nu, ard_num_dims=ard_features)
        else:
            base_kernel = RBFKernel(ard_num_dims=ard_features)

        self.n_tasks = y.shape[1]

        covar_module = ScaleKernel(base_kernel)

        if self.normalize_inputs:
            input_transform = Normalize(
                d=self.n_features + 1, indices=list(range(self.n_features))
            )
        else:
            input_transform = None

        if self.normalize_outputs:
            output_transform = TaskStandardize(num_tasks=self.n_tasks)
        else:
            output_transform = None

        X_mt, y_mt = self._to_multitask_format(X, y)

        self.model = MultiTaskGP(
            train_X=X_mt,
            train_Y=y_mt,
            task_feature=-1,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=output_transform,
            output_tasks=list(range(self.n_tasks)),
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation at candidate points for all tasks.

        Uses the fitted GP model to compute posterior mean and standard
        deviation at the given points for each output task.

        Parameters
        ----------
        X : np.ndarray, shape (n_candidates, n_features)
            Candidate points to evaluate. Must have same number of features
            as training data.

        Returns
        -------
        mean : np.ndarray, shape (n_candidates, n_tasks)
            Posterior mean prediction at each candidate point for each task.
        std : np.ndarray, shape (n_candidates, n_tasks)
            Posterior standard deviation (uncertainty) at each point for each task.
            Always non-negative.

        Raises
        ------
        NotFittedError
            If called before fit(). Must fit the model first.
        ValueError
            If X.shape[1] != n_features (feature dimension mismatch).
        ValueError
            If X is not float64 dtype.

        Notes
        -----
        Predictions are made by querying the posterior for each task index
        separately and combining the results.

        Examples
        --------
        >>> gp = MultiTaskGPSurrogate().fit(X_train, y_train)
        >>> mean, std = gp.predict(X_test)
        >>> mean.shape
        (10, 2)
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
        if X.dtype != np.float64:
            raise ValueError(f"X must be float64, got {X.dtype}")

        X = torch.tensor(X, dtype=torch.float64)

        posterior = self.model.posterior(X)
        mean = posterior.mean
        std = posterior.variance.sqrt()

        mean = mean.detach().numpy()
        std = std.detach().numpy()

        return mean, std

    def _to_multitask_format(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert standard arrays to BoTorch multi-task format.

        Reshapes data from (n_samples, n_tasks) to the stacked format expected
        by BoTorch's MultiTaskGP, where each observation-task pair is a separate
        row with the task index appended.

        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Input features tensor.
        y : torch.Tensor, shape (n, k)
            Output values tensor with k tasks.

        Returns
        -------
        X_mt : torch.Tensor, shape (n*k, d+1)
            Expanded input tensor with task index as last column.
            Row order: all task 0, then all task 1, etc.
        y_mt : torch.Tensor, shape (n*k, 1)
            Flattened output tensor.

        Examples
        --------
        >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
        >>> y = torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # shape (2, 2)
        >>> X_mt, y_mt = self._to_multitask_format(X, y)
        >>> X_mt.shape
        torch.Size([4, 3])
        >>> X_mt[:, -1]  # task indices
        tensor([0., 0., 1., 1.])
        """
        X_mt = []
        y_mt = []
        n_samples = X.shape[0]
        n_tasks = y.shape[1]

        for task in range(n_tasks):
            task_vector = torch.full((n_samples, 1), task, dtype=torch.float64)
            X_with_task = torch.cat((X, task_vector), dim=1)
            y_task = y[:, task]
            X_mt.append(X_with_task)
            y_mt.append(y_task)

        X_mt = torch.cat(X_mt, 0)
        y_mt = torch.cat(y_mt, 0).unsqueeze(-1)

        return X_mt, y_mt
