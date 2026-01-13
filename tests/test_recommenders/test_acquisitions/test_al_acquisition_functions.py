"""Tests for active learning acquisition functions."""

import numpy as np
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from folio.recommenders.acquisitions import (
    ActiveLearningAcquisition,
    PosteriorVariance,
)


@pytest.fixture
def fitted_single_task_gp():
    """Create a simple fitted SingleTaskGP for testing.

    Returns a GP fitted on 5 points from a simple quadratic function.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float64)
    y_train = (-((X_train - 0.5) ** 2) + 1).squeeze(-1)

    model = SingleTaskGP(
        train_X=X_train,
        train_Y=y_train.unsqueeze(-1),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


@pytest.fixture
def fitted_multi_task_gp():
    """Create a fitted MultiTaskGP with 2 outputs for testing.

    Returns a MultiTaskGP fitted on simple known data:
    - X = [[0], [1]] (2 points, 1D input)
    - Y = [[0, 0], [1, 1]] (2 outputs per point)

    This simple setup allows for predictable variance behavior.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    Y_train = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

    # MultiTaskGP requires task indices
    # Create augmented X with task indices
    n_tasks = 2
    n_samples = X_train.shape[0]

    # Stack X for each task and add task index
    X_augmented = []
    Y_flat = []
    for task_idx in range(n_tasks):
        task_indices = torch.full((n_samples, 1), task_idx, dtype=torch.float64)
        X_with_task = torch.cat([X_train, task_indices], dim=-1)
        X_augmented.append(X_with_task)
        Y_flat.append(Y_train[:, task_idx : task_idx + 1])

    X_all = torch.cat(X_augmented, dim=0)
    Y_all = torch.cat(Y_flat, dim=0)

    model = MultiTaskGP(
        train_X=X_all,
        train_Y=Y_all,
        task_feature=-1,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


# =============================================================================
# ActiveLearningAcquisition Base Class Tests
# =============================================================================


class TestActiveLearningAcquisitionBase:
    """Tests for ActiveLearningAcquisition base class."""

    def test_is_abstract(self):
        """Test that ActiveLearningAcquisition cannot be instantiated directly."""
        with pytest.raises(TypeError, match="(?i)abstract|instantiate"):
            ActiveLearningAcquisition()

    def test_subclass_must_implement_build(self):
        """Test that subclasses must implement build method."""

        class IncompleteALAcquisition(ActiveLearningAcquisition):
            pass

        with pytest.raises(TypeError, match="(?i)abstract|instantiate"):
            IncompleteALAcquisition()


# =============================================================================
# PosteriorVariance Tests
# =============================================================================


class TestPosteriorVarianceBuild:
    """Tests for PosteriorVariance.build."""

    def test_returns_acquisition_function(self, fitted_single_task_gp):
        """Test build returns a callable acquisition function."""
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        # Should be callable with shape (batch, q, d)
        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        result = acqf(X)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)

    def test_returns_acquisition_function_multitask(self, fitted_multi_task_gp):
        """Test build returns a callable acquisition function for multi-task GP."""
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_multi_task_gp)

        # MultiTaskGP expects X with task index as last column
        # Shape: (batch, q, d+1) where last dim is task index
        X = torch.tensor([[[0.5, 0.0]]], dtype=torch.float64)
        result = acqf(X)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)


class TestPosteriorVarianceShape:
    """Tests for PosteriorVariance output shapes."""

    def test_single_point_batch(self, fitted_single_task_gp):
        """Test PV with single point (batch=1, q=1)."""
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        result = acqf(X)

        assert result.shape == (1,)

    def test_multiple_batch(self, fitted_single_task_gp):
        """Test PV with multiple batches (batch > 1)."""
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        # 3 batches, 1 candidate each
        X = torch.tensor([[[0.1]], [[0.5]], [[0.9]]], dtype=torch.float64)
        result = acqf(X)

        assert result.shape == (3,)

    def test_q_batch(self, fitted_single_task_gp):
        """Test PV with q-batch (multiple candidates per batch)."""
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        # 1 batch, 2 candidates (q=2)
        X = torch.tensor([[[0.2], [0.8]]], dtype=torch.float64)
        result = acqf(X)

        # Result should be sum over q dimension
        assert result.shape == (1,)


class TestPosteriorVarianceNonNegative:
    """Tests for PosteriorVariance non-negativity."""

    def test_variance_always_nonnegative_single_task(self, fitted_single_task_gp):
        """Test that variance is always non-negative for single-task GP."""
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        # Test multiple points including edge cases
        X = torch.linspace(-0.5, 1.5, 20).unsqueeze(-1).unsqueeze(0)
        X = X.transpose(0, 1).double()
        results = torch.stack([acqf(x.unsqueeze(0)) for x in X])

        # Variance should always be >= 0
        assert (results >= -1e-10).all()

    def test_variance_always_nonnegative_multi_task(self, fitted_multi_task_gp):
        """Test that variance is always non-negative for multi-task GP."""
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_multi_task_gp)

        # Test multiple points (with task index)
        X_vals = torch.linspace(-0.5, 1.5, 10).unsqueeze(-1).double()
        task_idx = torch.zeros(10, 1, dtype=torch.float64)
        X = torch.cat([X_vals, task_idx], dim=-1).unsqueeze(1)

        results = torch.stack([acqf(x.unsqueeze(0)) for x in X])

        # Variance should always be >= 0
        assert (results >= -1e-10).all()


class TestPosteriorVarianceNumerical:
    """Numerical correctness tests for PosteriorVariance."""

    def test_matches_posterior_variance_single_output(self, fitted_single_task_gp):
        """Test PV output equals manual posterior variance computation.

        For single-output models:
        PV(x) = Var[f(x)]
        """
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        pv_value = acqf(X).item()

        # Manually compute variance from posterior
        posterior = fitted_single_task_gp.posterior(X)
        expected_variance = posterior.variance.sum().item()

        assert np.isclose(pv_value, expected_variance, rtol=1e-5)

    def test_matches_posterior_variance_multi_output(self, fitted_multi_task_gp):
        """Test PV output equals sum of variances across outputs.

        For multi-output models:
        PV(x) = sum_{i=1}^{m} Var[f_i(x)]
        """
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_multi_task_gp)

        # Test point with task index
        X = torch.tensor([[[0.5, 0.0]]], dtype=torch.float64)
        pv_value = acqf(X).item()

        # Manually compute variance from posterior
        posterior = fitted_multi_task_gp.posterior(X)
        expected_variance = posterior.variance.sum().item()

        assert np.isclose(pv_value, expected_variance, rtol=1e-5)

    def test_variance_higher_far_from_training_data(self, fitted_single_task_gp):
        """Test that variance is higher far from training data.

        GPs have low uncertainty near training points and high uncertainty
        far from them. This is a fundamental property we want to verify.
        """
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        # Near training data (training points are at 0, 0.25, 0.5, 0.75, 1.0)
        X_near = torch.tensor([[[0.5]]], dtype=torch.float64)
        # Far from training data
        X_far = torch.tensor([[[-1.0]]], dtype=torch.float64)

        pv_near = acqf(X_near).item()
        pv_far = acqf(X_far).item()

        # Variance should be higher far from training data
        assert pv_far > pv_near

    def test_variance_decreases_at_training_points(self, fitted_single_task_gp):
        """Test that variance is lowest at/near training points.

        GP interpolates through training data, so variance should be
        minimal at training points.
        """
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        # At a training point
        X_train = torch.tensor([[[0.5]]], dtype=torch.float64)
        # Between training points
        X_between = torch.tensor([[[0.375]]], dtype=torch.float64)

        pv_train = acqf(X_train).item()
        pv_between = acqf(X_between).item()

        # Variance at training point should be less than or equal
        # (allowing for numerical noise)
        assert pv_train <= pv_between + 1e-6


class TestPosteriorVarianceKnownData:
    """Tests using known data to verify numerical correctness."""

    def test_simple_gp_variance_computation(self):
        """Test variance computation on a minimal GP.

        Fit GP on X = [[0], [1]], Y = [[0], [1]] and verify variance
        at test points matches manual computation.
        """
        torch.manual_seed(42)

        X_train = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        Y_train = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

        model = SingleTaskGP(
            train_X=X_train,
            train_Y=Y_train,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        pv = PosteriorVariance()
        acqf = pv.build(model=model)

        # Test at midpoint (0.5) - should have higher variance than endpoints
        X_mid = torch.tensor([[[0.5]]], dtype=torch.float64)
        X_end = torch.tensor([[[0.0]]], dtype=torch.float64)

        pv_mid = acqf(X_mid).item()
        pv_end = acqf(X_end).item()

        # Midpoint should have higher variance than endpoint (near training)
        assert pv_mid > pv_end

        # Verify against manual posterior computation
        posterior_mid = model.posterior(X_mid)
        expected_var_mid = posterior_mid.variance.sum().item()
        assert np.isclose(pv_mid, expected_var_mid, rtol=1e-5)


class TestPosteriorVarianceDtype:
    """Tests for PV dtype validation."""

    def test_float32_raises(self, fitted_single_task_gp):
        """Test that float32 input raises ValueError."""
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        X = torch.tensor([[[0.3]]], dtype=torch.float32)
        with pytest.raises(ValueError, match="float64"):
            acqf(X)

    def test_float64_passes(self, fitted_single_task_gp):
        """Test that float64 input works."""
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        result = acqf(X)
        assert result.shape == (1,)


class TestPosteriorVarianceEdgeCases:
    """Edge case tests for PosteriorVariance."""

    def test_output_is_finite(self, fitted_single_task_gp):
        """Test that PV output contains no NaN or Inf values."""
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        # Test on multiple points including edge cases
        X = torch.tensor(
            [[[0.0]], [[0.5]], [[1.0]], [[-0.5]], [[1.5]]], dtype=torch.float64
        )
        results = torch.stack([acqf(x.unsqueeze(0)) for x in X])

        assert torch.all(torch.isfinite(results))

    def test_batch_and_q_dimension_handling(self, fitted_single_task_gp):
        """Test proper handling of batch and q dimensions."""
        pv = PosteriorVariance()
        acqf = pv.build(model=fitted_single_task_gp)

        # 2 batches, 3 candidates each
        X = torch.tensor(
            [[[0.1], [0.3], [0.5]], [[0.6], [0.8], [1.0]]], dtype=torch.float64
        )
        result = acqf(X)

        # Should sum over q dimension, return one value per batch
        assert result.shape == (2,)
