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
from botorch.posteriors import Posterior, TransformedPosterior
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
        self.eps = 10e-9

    def forward(
        self, Y: Tensor, X: Tensor | None = None, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Transform Y by standardizing per task.

        On the first call, computes and stores per-task mean and std from Y.
        On subsequent calls, uses the stored statistics (frozen).

        Parameters
        ----------
        Y : Tensor, shape (n, 1)
            Outcome values to transform. In MultiTaskGP format, this is a
            column vector with all observations stacked.
        X : Tensor, shape (n, d) or None
            Input features including task column. Required to identify which
            task each observation belongs to.
        Yvar : Tensor or None
            Observation noise variance. If provided, will be scaled by per-task
            variance.

        Returns
        -------
        Y_transformed : Tensor, shape (n, 1)
            Standardized outcome values.
        Yvar_transformed : Tensor or None
            Scaled observation variance if Yvar was provided, None otherwise.

        Raises
        ------
        ValueError
            If X is None (task IDs required for per-task standardization).
        ValueError
            If Y and X have different number of rows.
        ValueError
            If task IDs in X are out of range [0, num_tasks).
        """
        if X is None:
            raise ValueError("Need task IDs for per-task standardization")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Different number of observations: {X.shape[0]} "
                f"and labels: {Y.shape[0]}"
            )

        task_ind = X[:, self.task_feature].long()

        if task_ind.min() < 0 or task_ind.max() >= self.num_tasks:
            raise ValueError(
                f"Task IDs must be in [0, {self.num_tasks}), "
                f"got range [{task_ind.min()}, {task_ind.max()}]"
            )

        if not self._is_trained:
            self._means = torch.zeros(self.num_tasks, dtype=Y.dtype)
            self._stds = torch.zeros(self.num_tasks, dtype=Y.dtype)

            for t in range(self.num_tasks):
                mask = task_ind == t
                Y_task = Y[mask]
                self._means[t] = Y_task.mean()
                # Default to std=1.0 when only 1 sample (avoids division by zero in std)
                if Y_task.numel() <= 1:
                    self._stds[t] = 1.0
                else:
                    self._stds[t] = Y_task.std()

            self._is_trained = True

        mean = self._means[task_ind].unsqueeze(-1)
        std = self._stds[task_ind].unsqueeze(-1)

        transformed = (Y - mean) / (std + self.eps)

        # Only transform Yvar if it was provided (matches BoTorch Standardize behavior)
        if Yvar is not None:
            Yvar_transformed = Yvar / (std.square() + self.eps)
            return transformed, Yvar_transformed

        return transformed, None

    def untransform(
        self, Y: Tensor, X: Tensor | None = None, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Reverse the per-task standardization.

        Parameters
        ----------
        Y : Tensor, shape (n, 1)
            Transformed outcome values to convert back to original scale.
        X : Tensor, shape (n, d) or None
            Input features including task column. Required to identify which
            task each observation belongs to.
        Yvar : Tensor or None
            Observation noise variance (unused, for API compatibility).

        Returns
        -------
        Y_original : Tensor, shape (n, 1)
            Outcome values in original scale.
        Yvar_original : Tensor, shape (n, 1)
            Per-observation variance scaling.

        Raises
        ------
        ValueError
            If X is None.
        ValueError
            If transform has not been trained (forward() not called yet).
        """
        if X is None:
            raise ValueError("Need task IDs for per-task standardization")
        if not self._is_trained:
            raise ValueError("Need to call forward() before calling untransform()")

        task_ind = X[:, self.task_feature].long()
        mean = self._means[task_ind].unsqueeze(-1)
        std = self._stds[task_ind].unsqueeze(-1)

        untransformed = (Y * (std + self.eps)) + mean
        var = std.square()

        return untransformed, var

    def _sample_transform(self, sample):
        """s_original = µ_task + σ_task * s_standardized.

        Per-task stats have shape (num_tasks,) which broadcasts correctly
        against posterior shapes (..., num_tasks).
        """
        return self._means + self._stds * sample

    def _mean_transform(self, m, v):
        """µ_original = µ_task + σ_task * µ_standardized.

        Per-task stats have shape (num_tasks,) which broadcasts correctly
        against posterior mean shape (..., num_tasks).
        """
        return self._means + self._stds * m

    def _var_transform(self, m, v):
        """σ_original = σ_task^2 * σ_std^2.

        Per-task stats have shape (num_tasks,) which broadcasts correctly
        against posterior variance shape (..., num_tasks).
        """
        return self._stds.square() * v

    def untransform_posterior(
        self, posterior: Posterior, X: Tensor | None = None
    ) -> Posterior:
        """Untransform a posterior distribution.

        This is required by BoTorch for acquisition function evaluation.
        The posterior mean and variance must be transformed back to the
        original scale for proper acquisition function computation.

        Parameters
        ----------
        posterior : Posterior
            A BoTorch posterior object with mean and variance in transformed
            (standardized) space.
        X : Tensor or None
            Input features (unused, for API compatibility).

        Returns
        -------
        Posterior
            A posterior object with mean and variance in original scale.

        Raises
        ------
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
        if not self._is_trained:
            raise ValueError(
                "Need to call forward() before calling untransform_posterior()"
            )

        orginial = TransformedPosterior(
            posterior=posterior,
            sample_transform=self._sample_transform,
            mean_transform=self._mean_transform,
            variance_transform=self._var_transform,
        )

        return orginial
