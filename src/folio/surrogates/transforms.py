"""Custom outcome transforms for multi-task GP models.

This module provides outcome transforms that handle per-task standardization
for multi-output Gaussian Processes where different tasks (objectives) have
vastly different scales.

The key issue: BoTorch's built-in Standardize computes global mean/std across
the entire stacked y vector. For multi-objective problems with different scales
(e.g., molecular weight ~10^5, PDI ~1-3, conversion ~0-1), this doesn't properly
balance the objectives. The GP ends up dominated by the high-magnitude objective.

TaskStandardize solves this by computing and applying per-task statistics,
ensuring each objective contributes equally to the model regardless of scale.
"""

from __future__ import annotations

import torch
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior
from torch import Tensor


class TaskStandardize(OutcomeTransform):
    """Per-task standardization for multi-task GPs with different objective scales.

    This transform computes separate mean and standard deviation for each task,
    then standardizes observations within each task independently. This ensures
    that objectives with different scales (e.g., MW ~10^5 vs PDI ~1-3) contribute
    equally to the GP model.

    The transform is "training-aware": on the first forward pass, it computes
    and stores the per-task statistics. Subsequent forward passes (and all
    untransform operations) use these frozen statistics.

    Parameters
    ----------
    num_tasks : int
        Number of tasks (objectives) in the multi-task GP.
    task_feature : int, optional
        Column index in X that contains the task identifier (0, 1, ..., num_tasks-1).
        Default is -1 (last column), following BoTorch convention for MultiTaskGP.

    Attributes
    ----------
    _means : Tensor or None
        Per-task means, shape (num_tasks,). None until first forward() call.
    _stds : Tensor or None
        Per-task standard deviations, shape (num_tasks,). None until first
        forward() call.
    _is_trained : bool
        Whether statistics have been computed and frozen.

    Examples
    --------
    >>> transform = TaskStandardize(num_tasks=3)
    >>> y_transformed, y_std = transform.forward(y, X)
    >>> y_original, _ = transform.untransform(y_transformed, X)

    Notes
    -----
    The transform follows BoTorch conventions:

    - forward() returns (y_transformed, per_row_stds) tuple
    - untransform() reverses the transformation
    - untransform_posterior() handles posterior predictive distributions

    Math:
        For each task t:
            mean_t = mean(y[task_ids == t])
            std_t = std(y[task_ids == t])
            y_transformed[task_ids == t] = (y[task_ids == t] - mean_t) / std_t

    Reference: Standard practice in multi-objective optimization to normalize
    objectives to comparable scales.
    """

    def __init__(self, num_tasks: int, task_feature: int = -1) -> None:
        """Initialize TaskStandardize transform.

        Parameters
        ----------
        num_tasks : int
            Number of tasks (objectives). Must be positive.
        task_feature : int, optional
            Column index in X containing task IDs. Default -1 (last column).

        Raises
        ------
        ValueError
            If num_tasks is not positive.
        """
        super().__init__()
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}")
        self.num_tasks = num_tasks
        self.task_feature = task_feature
        self._means: Tensor | None = None
        self._stds: Tensor | None = None
        self._is_trained: bool = False

    def forward(self, y: Tensor, X: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Transform y by standardizing per task.

        On the first call, computes and stores per-task mean and std from y.
        On subsequent calls, uses the stored statistics (frozen).

        Parameters
        ----------
        y : Tensor, shape (n, 1)
            Outcome values to transform. In MultiTaskGP format, this is a
            column vector with all observations stacked.
        X : Tensor, shape (n, d) or None
            Input features including task column. Required to identify which
            task each observation belongs to.

        Returns
        -------
        y_transformed : Tensor, shape (n, 1)
            Standardized outcome values.
        y_var : Tensor, shape (n, 1)
            Per-observation variance scaling (std^2 for each row's task).
            Used by BoTorch for noise modeling.

        Raises
        ------
        ValueError
            If X is None (task IDs required for per-task standardization).
        ValueError
            If y and X have different number of rows.
        ValueError
            If task IDs in X are out of range [0, num_tasks).
        """
        if X is None:
            raise ValueError("Need task IDs for per-task standardization")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Different number of observations: {X.shape[0]} "
                f"and labels: {y.shape[0]}"
            )

        task_ind = X[:, self.task_feature].long()

        if task_ind.min() < 0 or task_ind.max() >= self.num_tasks:
            raise ValueError(
                f"Task IDs must be in [0, {self.num_tasks}), "
                f"got range [{task_ind.min()}, {task_ind.max()}]"
            )

        if not self._is_trained:
            self._means = torch.zeros(self.num_tasks, dtype=y.dtype)
            self._stds = torch.zeros(self.num_tasks, dtype=y.dtype)

            for t in range(self.num_tasks):
                mask = task_ind == t
                self._means[t] = y[mask].mean()
                self._stds[t] = y[mask].std()

            self._is_trained = True

        eps = 10e-9

        mean = self._means[task_ind].unsqueeze(-1)
        std = self._stds[task_ind].unsqueeze(-1)

        transformed = (y - mean) / (std + eps)
        var = std.square()

        return transformed, var

    def untransform(self, y: Tensor, X: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Reverse the per-task standardization.

        Parameters
        ----------
        y : Tensor, shape (n, 1)
            Transformed outcome values to convert back to original scale.
        X : Tensor, shape (n, d) or None
            Input features including task column. Required to identify which
            task each observation belongs to.

        Returns
        -------
        y_original : Tensor, shape (n, 1)
            Outcome values in original scale.
        y_var : Tensor, shape (n, 1)
            Per-observation variance scaling.

        Raises
        ------
        ValueError
            If X is None.
        ValueError
            If transform has not been trained (forward() not called yet).
        """
        task_ind = X[:, self.task_feature].long()
        raise NotImplementedError

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        """Untransform a posterior distribution.

        This is required by BoTorch for acquisition function evaluation.
        The posterior mean and variance must be transformed back to the
        original scale for proper acquisition function computation.

        Parameters
        ----------
        posterior : Posterior
            A BoTorch posterior object with mean and variance in transformed
            (standardized) space.

        Returns
        -------
        Posterior
            A posterior object with mean and variance in original scale.

        Raises
        ------
        NotImplementedError
            If the posterior type is not supported.
        ValueError
            If transform has not been trained.

        Notes
        -----
        For a Gaussian posterior:
            mean_original = mean_transformed * std + mean
            var_original = var_transformed * std^2

        This method handles the complexity of mapping task-specific statistics
        to the batch/q-batch dimensions of the posterior.
        """
        raise NotImplementedError
