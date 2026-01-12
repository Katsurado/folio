"""Tests for custom outcome transforms."""

import pytest
import torch

from folio.surrogates.transforms import TaskStandardize


class TestTaskStandardizeConstruction:
    """Test TaskStandardize constructor and initial state."""

    def test_default_task_feature(self):
        """Default task_feature is -1 (last column)."""
        transform = TaskStandardize(num_tasks=3)
        assert transform.task_feature == -1

    def test_custom_task_feature(self):
        """Custom task_feature can be specified."""
        transform = TaskStandardize(num_tasks=2, task_feature=0)
        assert transform.task_feature == 0

    def test_num_tasks_stored(self):
        """num_tasks is stored correctly."""
        transform = TaskStandardize(num_tasks=5)
        assert transform.num_tasks == 5

    def test_initial_means_is_none(self):
        """_means is None before training."""
        transform = TaskStandardize(num_tasks=3)
        assert transform._means is None

    def test_initial_stds_is_none(self):
        """_stds is None before training."""
        transform = TaskStandardize(num_tasks=3)
        assert transform._stds is None

    def test_initial_is_trained_false(self):
        """_is_trained is False before forward() is called."""
        transform = TaskStandardize(num_tasks=3)
        assert transform._is_trained is False

    def test_invalid_num_tasks_zero_raises(self):
        """num_tasks=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive|num_tasks|> 0"):
            TaskStandardize(num_tasks=0)

    def test_invalid_num_tasks_negative_raises(self):
        """Negative num_tasks raises ValueError."""
        with pytest.raises(ValueError, match="positive|num_tasks|> 0"):
            TaskStandardize(num_tasks=-1)


class TestTaskStandardizeForward:
    """Test TaskStandardize forward pass."""

    @pytest.fixture
    def simple_data(self):
        """Create simple multi-task data with known statistics.

        Task 0: values [0, 2, 4] -> mean=2, std=2 (population std with ddof=0)
        Task 1: values [10, 20, 30] -> mean=20, std~8.16

        Returns X with task IDs in last column, y with values.
        """
        X = torch.tensor(
            [
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [0.4, 1.0],
                [0.5, 1.0],
                [0.6, 1.0],
            ],
            dtype=torch.float64,
        )
        y = torch.tensor(
            [[0.0], [2.0], [4.0], [10.0], [20.0], [30.0]], dtype=torch.float64
        )
        return X, y

    def test_forward_returns_tuple(self, simple_data):
        """forward() returns (y_transformed, y_var) tuple."""
        X, y = simple_data
        transform = TaskStandardize(num_tasks=2)
        result = transform.forward(y, X)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_forward_output_shapes(self, simple_data):
        """forward() output shapes match input shapes."""
        X, y = simple_data
        transform = TaskStandardize(num_tasks=2)
        y_transformed, y_var = transform.forward(y, X)
        assert y_transformed.shape == y.shape
        assert y_var.shape == y.shape

    def test_forward_computes_correct_means(self, simple_data):
        """forward() computes correct per-task means."""
        X, y = simple_data
        transform = TaskStandardize(num_tasks=2)
        transform.forward(y, X)

        # Task 0: mean of [0, 2, 4] = 2.0
        # Task 1: mean of [10, 20, 30] = 20.0
        assert transform._means is not None
        torch.testing.assert_close(
            transform._means, torch.tensor([2.0, 20.0], dtype=torch.float64)
        )

    def test_forward_computes_correct_stds(self, simple_data):
        """forward() computes correct per-task standard deviations."""
        X, y = simple_data
        transform = TaskStandardize(num_tasks=2)
        transform.forward(y, X)

        # Task 0: std of [0, 2, 4] (sample std)
        # Task 1: std of [10, 20, 30] (sample std)
        assert transform._stds is not None
        expected_std_0 = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64).std()
        expected_std_1 = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64).std()
        torch.testing.assert_close(
            transform._stds,
            torch.tensor(
                [expected_std_0.item(), expected_std_1.item()], dtype=torch.float64
            ),
        )

    def test_forward_standardizes_correctly(self, simple_data):
        """forward() correctly standardizes each task."""
        X, y = simple_data
        transform = TaskStandardize(num_tasks=2)
        y_transformed, _ = transform.forward(y, X)

        # After standardization, each task should have mean~0, std~1
        task_0_transformed = y_transformed[:3]
        task_1_transformed = y_transformed[3:]

        # Means should be close to 0
        assert torch.abs(task_0_transformed.mean()) < 1e-6
        assert torch.abs(task_1_transformed.mean()) < 1e-6

        # Stds should be close to 1 (using sample std)
        assert torch.abs(task_0_transformed.std() - 1.0) < 1e-6
        assert torch.abs(task_1_transformed.std() - 1.0) < 1e-6

    def test_forward_sets_is_trained(self, simple_data):
        """forward() sets _is_trained to True."""
        X, y = simple_data
        transform = TaskStandardize(num_tasks=2)
        assert transform._is_trained is False
        transform.forward(y, X)
        assert transform._is_trained is True

    def test_forward_freezes_stats_after_first_call(self, simple_data):
        """forward() uses frozen stats on subsequent calls."""
        X, y = simple_data
        transform = TaskStandardize(num_tasks=2)
        transform.forward(y, X)

        # Store original stats
        original_means = transform._means.clone()
        original_stds = transform._stds.clone()

        # Call forward again with different data
        y_new = torch.tensor(
            [[100.0], [200.0], [300.0], [1000.0], [2000.0], [3000.0]],
            dtype=torch.float64,
        )
        transform.forward(y_new, X)

        # Stats should not have changed
        torch.testing.assert_close(transform._means, original_means)
        torch.testing.assert_close(transform._stds, original_stds)

    def test_forward_x_none_raises(self):
        """forward() raises ValueError if X is None."""
        y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        transform = TaskStandardize(num_tasks=2)
        with pytest.raises(ValueError, match="(?i)x|task|none|required"):
            transform.forward(y, None)

    def test_forward_shape_mismatch_raises(self, simple_data):
        """forward() raises ValueError if y and X have different n_samples."""
        X, y = simple_data
        y_wrong = y[:-1]
        transform = TaskStandardize(num_tasks=2)
        with pytest.raises(
            ValueError, match="(?i)shape|mismatch|samples|rows|observations|different"
        ):
            transform.forward(y_wrong, X)


class TestTaskStandardizeUntransform:
    """Test TaskStandardize untransform method."""

    @pytest.fixture
    def trained_transform(self):
        """Create a trained transform with known statistics."""
        X = torch.tensor(
            [
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [0.4, 1.0],
                [0.5, 1.0],
                [0.6, 1.0],
            ],
            dtype=torch.float64,
        )
        y = torch.tensor(
            [[0.0], [2.0], [4.0], [10.0], [20.0], [30.0]], dtype=torch.float64
        )
        transform = TaskStandardize(num_tasks=2)
        transform.forward(y, X)
        return transform, X, y

    def test_untransform_returns_tuple(self, trained_transform):
        """untransform() returns (y_original, y_var) tuple."""
        transform, X, y = trained_transform
        y_transformed, _ = transform.forward(y, X)
        result = transform.untransform(y_transformed, X)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_untransform_output_shapes(self, trained_transform):
        """untransform() output shapes match input shapes."""
        transform, X, y = trained_transform
        y_transformed, _ = transform.forward(y, X)
        y_untransformed, y_var = transform.untransform(y_transformed, X)
        assert y_untransformed.shape == y.shape
        assert y_var.shape == y.shape

    def test_round_trip(self, trained_transform):
        """untransform(forward(y)) recovers original y."""
        transform, X, y = trained_transform
        y_transformed, _ = transform.forward(y, X)
        y_recovered, _ = transform.untransform(y_transformed, X)
        torch.testing.assert_close(y_recovered, y)

    def test_untransform_before_training_raises(self):
        """untransform() raises ValueError if not trained."""
        transform = TaskStandardize(num_tasks=2)
        y = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
        X = torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.3, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="(?i)train|fit|forward"):
            transform.untransform(y, X)

    def test_untransform_x_none_raises(self, trained_transform):
        """untransform() raises ValueError if X is None."""
        transform, X, y = trained_transform
        y_transformed, _ = transform.forward(y, X)
        with pytest.raises(ValueError, match="(?i)x|task|none|required"):
            transform.untransform(y_transformed, None)


class TestTaskStandardizeUnequalTaskSizes:
    """Test TaskStandardize with unequal observations per task."""

    def test_unequal_task_sizes(self):
        """Transform works with different number of observations per task."""
        # Task 0: 2 observations [0, 4] -> mean=2, std=2
        # Task 1: 4 observations [10, 20, 30, 40] -> mean=25, std=11.18
        X = torch.tensor(
            [
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 1.0],
                [0.4, 1.0],
                [0.5, 1.0],
                [0.6, 1.0],
            ],
            dtype=torch.float64,
        )
        y = torch.tensor(
            [[0.0], [4.0], [10.0], [20.0], [30.0], [40.0]], dtype=torch.float64
        )

        transform = TaskStandardize(num_tasks=2)
        y_transformed, _ = transform.forward(y, X)

        # Check task 0 is standardized
        task_0_transformed = y_transformed[:2]
        assert torch.abs(task_0_transformed.mean()) < 1e-6

        # Check task 1 is standardized
        task_1_transformed = y_transformed[2:]
        assert torch.abs(task_1_transformed.mean()) < 1e-6

    def test_round_trip_unequal_sizes(self):
        """Round-trip works with unequal task sizes."""
        X = torch.tensor(
            [
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 1.0],
                [0.4, 1.0],
                [0.5, 1.0],
                [0.6, 1.0],
            ],
            dtype=torch.float64,
        )
        y = torch.tensor(
            [[0.0], [4.0], [10.0], [20.0], [30.0], [40.0]], dtype=torch.float64
        )

        transform = TaskStandardize(num_tasks=2)
        y_transformed, _ = transform.forward(y, X)
        y_recovered, _ = transform.untransform(y_transformed, X)

        torch.testing.assert_close(y_recovered, y)


class TestTaskStandardizeEdgeCases:
    """Test edge cases and error handling."""

    def test_single_observation_per_task(self):
        """Handles single observation per task (std=0 case)."""
        X = torch.tensor([[0.1, 0.0], [0.2, 1.0]], dtype=torch.float64)
        y = torch.tensor([[5.0], [100.0]], dtype=torch.float64)

        transform = TaskStandardize(num_tasks=2)
        # Should not raise, but behavior for std=0 is implementation-defined
        # (could use std=1 fallback or similar)
        y_transformed, _ = transform.forward(y, X)
        assert y_transformed.shape == y.shape

    def test_constant_values_per_task(self):
        """Handles constant values within a task (std=0 case)."""
        X = torch.tensor(
            [[0.1, 0.0], [0.2, 0.0], [0.3, 1.0], [0.4, 1.0]], dtype=torch.float64
        )
        y = torch.tensor([[5.0], [5.0], [100.0], [100.0]], dtype=torch.float64)

        transform = TaskStandardize(num_tasks=2)
        # Should not raise
        y_transformed, _ = transform.forward(y, X)
        assert y_transformed.shape == y.shape

    def test_task_ids_out_of_range_raises(self):
        """Raises error if task IDs exceed num_tasks."""
        X = torch.tensor([[0.1, 0.0], [0.2, 5.0]], dtype=torch.float64)
        y = torch.tensor([[1.0], [2.0]], dtype=torch.float64)

        transform = TaskStandardize(num_tasks=2)
        with pytest.raises(
            ValueError, match="(?i)task|out of range|invalid|bounds|0.*num_tasks"
        ):
            transform.forward(y, X)

    def test_negative_task_ids_raises(self):
        """Raises error for negative task IDs."""
        X = torch.tensor([[0.1, 0.0], [0.2, -1.0]], dtype=torch.float64)
        y = torch.tensor([[1.0], [2.0]], dtype=torch.float64)

        transform = TaskStandardize(num_tasks=2)
        with pytest.raises(
            ValueError, match="(?i)task|negative|invalid|out of range|bounds"
        ):
            transform.forward(y, X)

    def test_three_tasks(self):
        """Works with three tasks."""
        X = torch.tensor(
            [
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 1.0],
                [0.4, 1.0],
                [0.5, 2.0],
                [0.6, 2.0],
            ],
            dtype=torch.float64,
        )
        y = torch.tensor(
            [[0.0], [2.0], [100.0], [200.0], [1e6], [2e6]], dtype=torch.float64
        )

        transform = TaskStandardize(num_tasks=3)
        y_transformed, _ = transform.forward(y, X)

        # Each task should be standardized
        for task_id in range(3):
            mask = X[:, -1] == task_id
            task_transformed = y_transformed[mask]
            assert torch.abs(task_transformed.mean()) < 1e-5

        # Round-trip should work
        y_recovered, _ = transform.untransform(y_transformed, X)
        torch.testing.assert_close(y_recovered, y)


class TestTaskStandardizeUntransformPosterior:
    """Test untransform_posterior method."""

    def test_untransform_posterior_not_trained_raises(self):
        """untransform_posterior() raises if not trained."""
        from botorch.posteriors import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal

        transform = TaskStandardize(num_tasks=2)

        # Create a dummy posterior
        mean = torch.zeros(2)
        covar = torch.eye(2)
        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)

        with pytest.raises(ValueError, match="(?i)train|fit|forward"):
            transform.untransform_posterior(posterior)

    def test_untransform_posterior_returns_posterior(self):
        """untransform_posterior() returns a Posterior object."""
        from botorch.posteriors import GPyTorchPosterior, Posterior
        from gpytorch.distributions import MultivariateNormal

        # Train the transform first
        X = torch.tensor(
            [[0.1, 0.0], [0.2, 0.0], [0.3, 1.0], [0.4, 1.0]], dtype=torch.float64
        )
        y = torch.tensor([[0.0], [4.0], [10.0], [20.0]], dtype=torch.float64)
        transform = TaskStandardize(num_tasks=2)
        transform.forward(y, X)

        # Create a dummy posterior
        mean = torch.zeros(2, dtype=torch.float64)
        covar = torch.eye(2, dtype=torch.float64)
        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)

        result = transform.untransform_posterior(posterior)
        assert isinstance(result, Posterior)


class TestTaskStandardizeTaskFeatureColumn:
    """Test custom task_feature column specification."""

    def test_task_feature_first_column(self):
        """Works when task IDs are in first column."""
        # Task IDs in column 0
        X = torch.tensor(
            [[0.0, 0.1], [0.0, 0.2], [1.0, 0.3], [1.0, 0.4]], dtype=torch.float64
        )
        y = torch.tensor([[0.0], [4.0], [100.0], [200.0]], dtype=torch.float64)

        transform = TaskStandardize(num_tasks=2, task_feature=0)
        y_transformed, _ = transform.forward(y, X)

        # Check standardization happened correctly
        task_0_transformed = y_transformed[:2]
        task_1_transformed = y_transformed[2:]

        assert torch.abs(task_0_transformed.mean()) < 1e-6
        assert torch.abs(task_1_transformed.mean()) < 1e-6

    def test_task_feature_middle_column(self):
        """Works when task IDs are in a middle column."""
        # Task IDs in column 1 of 3
        X = torch.tensor(
            [
                [0.1, 0.0, 0.5],
                [0.2, 0.0, 0.6],
                [0.3, 1.0, 0.7],
                [0.4, 1.0, 0.8],
            ],
            dtype=torch.float64,
        )
        y = torch.tensor([[0.0], [4.0], [100.0], [200.0]], dtype=torch.float64)

        transform = TaskStandardize(num_tasks=2, task_feature=1)
        y_transformed, _ = transform.forward(y, X)

        # Verify round-trip
        y_recovered, _ = transform.untransform(y_transformed, X)
        torch.testing.assert_close(y_recovered, y)
