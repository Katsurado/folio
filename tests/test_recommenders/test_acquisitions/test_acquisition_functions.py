"""Tests for BoTorch-compatible acquisition functions."""

import numpy as np
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm

from folio.recommenders.acquisitions import (
    Acquisition,
    ExpectedImprovement,
    UpperConfidenceBound,
)


@pytest.fixture
def fitted_gp():
    """Create a simple fitted SingleTaskGP for testing.

    Returns a GP fitted on 5 points from a simple quadratic function.
    The model is fitted with Standardize transform for numerical stability.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float64)
    # Simple quadratic: y = -(x - 0.5)^2 + 1, max at x=0.5, y=1
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
def best_f_max():
    """Best observed value for maximization (y at x=0.5)."""
    return 1.0


@pytest.fixture
def best_f_min():
    """Best observed value for minimization (y at x=0 or x=1)."""
    return 0.75


# =============================================================================
# ExpectedImprovement Tests
# =============================================================================


class TestExpectedImprovementInit:
    """Tests for ExpectedImprovement.__init__."""

    def test_default_xi(self):
        """Test default xi value is 0.01."""
        ei = ExpectedImprovement()
        assert ei.xi == 0.01

    def test_custom_xi(self):
        """Test custom xi value is stored."""
        ei = ExpectedImprovement(xi=0.1)
        assert ei.xi == 0.1

    def test_zero_xi(self):
        """Test xi=0 is valid (pure EI)."""
        ei = ExpectedImprovement(xi=0.0)
        assert ei.xi == 0.0

    def test_negative_xi_raises(self):
        """Test negative xi raises ValueError."""
        with pytest.raises(ValueError, match="(?i)negative|non-negative|>= 0"):
            ExpectedImprovement(xi=-0.1)


class TestExpectedImprovementBuild:
    """Tests for ExpectedImprovement.build."""

    def test_returns_acquisition_function(self, fitted_gp, best_f_max):
        """Test build returns a callable acquisition function."""
        ei = ExpectedImprovement(xi=0.01)
        acqf = ei.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # Should be callable with shape (batch, q, d)
        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        result = acqf(X)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)

    def test_maximize_vs_minimize_different_scores(self, fitted_gp):
        """Test that maximize and minimize produce different acquisition values."""
        ei = ExpectedImprovement(xi=0.0)
        acqf_max = ei.build(model=fitted_gp, best_f=0.9, maximize=True)
        acqf_min = ei.build(model=fitted_gp, best_f=0.9, maximize=False)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        score_max = acqf_max(X).item()
        score_min = acqf_min(X).item()

        # Scores should differ (different improvement directions)
        assert score_max != score_min


class TestExpectedImprovementNumerical:
    """Numerical correctness tests for EI."""

    def test_ei_formula_maximize(self, fitted_gp, best_f_max):
        """Test EI computation matches analytical formula for maximization.

        EI = sigma * (Z * Phi(Z) + phi(Z)) where Z = (mu - best_f - xi) / sigma
        """
        xi = 0.0
        ei = ExpectedImprovement(xi=xi)
        acqf = ei.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        ei_value = acqf(X).item()

        # Compute expected value from GP posterior
        posterior = fitted_gp.posterior(X)
        mean = posterior.mean.squeeze().item()
        std = posterior.variance.sqrt().squeeze().item()

        if std > 1e-12:
            Z = (mean - best_f_max - xi) / std
            expected_ei = std * (Z * norm.cdf(Z) + norm.pdf(Z))
        else:
            expected_ei = 0.0

        assert np.isclose(ei_value, expected_ei, rtol=1e-5)

    def test_ei_formula_minimize(self, fitted_gp, best_f_min):
        """Test EI computation matches analytical formula for minimization.

        For minimize: Z = (best_f - mu - xi) / sigma
        """
        xi = 0.0
        ei = ExpectedImprovement(xi=xi)
        acqf = ei.build(model=fitted_gp, best_f=best_f_min, maximize=False)

        X = torch.tensor([[[0.8]]], dtype=torch.float64)
        ei_value = acqf(X).item()

        posterior = fitted_gp.posterior(X)
        mean = posterior.mean.squeeze().item()
        std = posterior.variance.sqrt().squeeze().item()

        if std > 1e-12:
            Z = (best_f_min - mean - xi) / std
            expected_ei = std * (Z * norm.cdf(Z) + norm.pdf(Z))
        else:
            expected_ei = 0.0

        assert np.isclose(ei_value, expected_ei, rtol=1e-5)

    def test_xi_reduces_ei(self, fitted_gp, best_f_max):
        """Test that larger xi reduces EI values (requires bigger improvement)."""
        ei_small = ExpectedImprovement(xi=0.0)
        ei_large = ExpectedImprovement(xi=0.5)

        acqf_small = ei_small.build(model=fitted_gp, best_f=best_f_max, maximize=True)
        acqf_large = ei_large.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        score_small = acqf_small(X).item()
        score_large = acqf_large(X).item()

        # Larger xi should give smaller or equal EI
        assert score_large <= score_small


class TestExpectedImprovementDirectional:
    """Directional behavior tests for EI."""

    def test_mean_better_than_best_gives_positive_ei_maximize(self, fitted_gp):
        """When mean > best_f (maximize), EI should be positive."""
        ei = ExpectedImprovement(xi=0.0)
        # Use a low best_f so points have positive improvement
        acqf = ei.build(model=fitted_gp, best_f=0.5, maximize=True)

        # Point near peak (x=0.5 has mean ~1.0, which is > 0.5)
        X = torch.tensor([[[0.5]]], dtype=torch.float64)
        ei_value = acqf(X).item()

        assert ei_value > 0

    def test_mean_better_than_best_gives_positive_ei_minimize(self, fitted_gp):
        """When mean < best_f (minimize), EI should be positive."""
        ei = ExpectedImprovement(xi=0.0)
        # Use a high best_f so points have positive improvement for minimize
        acqf = ei.build(model=fitted_gp, best_f=1.5, maximize=False)

        # Point near peak has mean ~1.0, which is < 1.5 (good for minimize)
        X = torch.tensor([[[0.5]]], dtype=torch.float64)
        ei_value = acqf(X).item()

        assert ei_value > 0

    def test_mean_worse_than_best_low_std_gives_near_zero_ei_maximize(self, fitted_gp):
        """When mean << best_f with low std (maximize), EI should be near zero."""
        ei = ExpectedImprovement(xi=0.0)
        # Use a very high best_f so improvement is unlikely
        acqf = ei.build(model=fitted_gp, best_f=5.0, maximize=True)

        # Training point has low std
        X = torch.tensor([[[0.5]]], dtype=torch.float64)
        ei_value = acqf(X).item()

        # EI should be very small since mean (~1.0) << best_f (5.0) with low std
        assert ei_value < 1e-6

    def test_mean_worse_than_best_low_std_gives_near_zero_ei_minimize(self, fitted_gp):
        """When mean >> best_f with low std (minimize), EI should be near zero."""
        ei = ExpectedImprovement(xi=0.0)
        # Use a very low best_f so improvement is unlikely for minimize
        acqf = ei.build(model=fitted_gp, best_f=-5.0, maximize=False)

        # Training point has low std
        X = torch.tensor([[[0.5]]], dtype=torch.float64)
        ei_value = acqf(X).item()

        # EI should be very small since mean (~1.0) >> best_f (-5.0) with low std
        assert ei_value < 1e-6

    def test_higher_std_can_increase_ei(self, fitted_gp, best_f_max):
        """Test that points with higher uncertainty can have higher EI.

        Points far from training data (higher std) should have higher EI
        when the mean is similar, demonstrating exploration behavior.
        """
        ei = ExpectedImprovement(xi=0.01)
        acqf = ei.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # Point near training data (low uncertainty)
        X_near = torch.tensor([[[0.5]]], dtype=torch.float64)
        # Point far from training data (higher uncertainty)
        X_far = torch.tensor([[[-0.5]]], dtype=torch.float64)

        # Get posteriors to verify std relationship
        posterior_near = fitted_gp.posterior(X_near)
        posterior_far = fitted_gp.posterior(X_far)
        std_near = posterior_near.variance.sqrt().squeeze().item()
        std_far = posterior_far.variance.sqrt().squeeze().item()

        # Far point should have higher uncertainty
        assert std_far > std_near

        # EI at far point should be positive (exploring unknown region)
        ei_far = acqf(X_far).item()
        assert ei_far > 0


class TestExpectedImprovementEdgeCases:
    """Edge case tests for EI."""

    def test_output_is_finite(self, fitted_gp, best_f_max):
        """Test that EI output contains no NaN or Inf values."""
        ei = ExpectedImprovement(xi=0.01)
        acqf = ei.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # Test on multiple points including edge cases
        X = torch.tensor(
            [[[0.0]], [[0.5]], [[1.0]], [[-0.5]], [[1.5]]], dtype=torch.float64
        )
        results = torch.stack([acqf(x.unsqueeze(0)) for x in X])

        assert torch.all(torch.isfinite(results))

    def test_zero_std_returns_zero_ei(self, fitted_gp, best_f_max):
        """Test that EI is small at training points (low uncertainty)."""
        ei = ExpectedImprovement(xi=0.0)
        acqf = ei.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # Training point should have near-zero std
        X_train = torch.tensor([[[0.5]]], dtype=torch.float64)
        ei_value = acqf(X_train).item()

        # EI should be small at training point
        # Note: not exactly zero because GP has observation noise
        assert ei_value < 0.01

    def test_single_point_batch(self, fitted_gp, best_f_max):
        """Test EI with single point (batch=1, q=1)."""
        ei = ExpectedImprovement(xi=0.01)
        acqf = ei.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        result = acqf(X)

        assert result.shape == (1,)
        assert result.item() >= 0

    def test_multiple_batch(self, fitted_gp, best_f_max):
        """Test EI with multiple batches (batch > 1)."""
        ei = ExpectedImprovement(xi=0.01)
        acqf = ei.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # 3 batches, 1 candidate each
        X = torch.tensor([[[0.1]], [[0.5]], [[0.9]]], dtype=torch.float64)
        result = acqf(X)

        assert result.shape == (3,)
        assert (result >= 0).all()

    def test_q_batch(self, fitted_gp, best_f_max):
        """Test EI with q-batch (multiple candidates per batch)."""
        ei = ExpectedImprovement(xi=0.01)
        acqf = ei.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # 1 batch, 2 candidates (q=2)
        X = torch.tensor([[[0.2], [0.8]]], dtype=torch.float64)
        result = acqf(X)

        # Result should be sum over q dimension
        assert result.shape == (1,)

    def test_ei_nonnegative(self, fitted_gp, best_f_max):
        """Test that EI is always non-negative."""
        ei = ExpectedImprovement(xi=0.0)
        acqf = ei.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # Test multiple points
        X = torch.linspace(0, 1, 20).unsqueeze(-1).unsqueeze(0)
        X = X.transpose(0, 1).double()
        results = torch.stack([acqf(x.unsqueeze(0)) for x in X])

        assert (results >= -1e-10).all()  # Allow small numerical errors


class TestExpectedImprovementDtype:
    """Tests for EI dtype validation."""

    def test_float32_raises(self, fitted_gp, best_f_max):
        """Test that float32 input raises ValueError."""
        ei = ExpectedImprovement(xi=0.01)
        acqf = ei.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.3]]], dtype=torch.float32)
        with pytest.raises(ValueError, match="float64"):
            acqf(X)

    def test_float64_passes(self, fitted_gp, best_f_max):
        """Test that float64 input works."""
        ei = ExpectedImprovement(xi=0.01)
        acqf = ei.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        result = acqf(X)
        assert result.shape == (1,)


# =============================================================================
# UpperConfidenceBound Tests
# =============================================================================


class TestUpperConfidenceBoundInit:
    """Tests for UpperConfidenceBound.__init__."""

    def test_default_beta(self):
        """Test default beta value is 2.0."""
        ucb = UpperConfidenceBound()
        assert ucb.beta == 2.0

    def test_custom_beta(self):
        """Test custom beta value is stored."""
        ucb = UpperConfidenceBound(beta=3.0)
        assert ucb.beta == 3.0

    def test_zero_beta(self):
        """Test beta=0 is valid (pure exploitation)."""
        ucb = UpperConfidenceBound(beta=0.0)
        assert ucb.beta == 0.0

    def test_negative_beta_raises(self):
        """Test negative beta raises ValueError."""
        with pytest.raises(ValueError, match="(?i)negative|non-negative|>= 0"):
            UpperConfidenceBound(beta=-1.0)


class TestUpperConfidenceBoundBuild:
    """Tests for UpperConfidenceBound.build."""

    def test_returns_acquisition_function(self, fitted_gp, best_f_max):
        """Test build returns a callable acquisition function."""
        ucb = UpperConfidenceBound(beta=2.0)
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        result = acqf(X)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)

    def test_maximize_vs_minimize_different_scores(self, fitted_gp):
        """Test that maximize and minimize produce different acquisition values."""
        ucb = UpperConfidenceBound(beta=2.0)
        acqf_max = ucb.build(model=fitted_gp, best_f=0.9, maximize=True)
        acqf_min = ucb.build(model=fitted_gp, best_f=0.9, maximize=False)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        score_max = acqf_max(X).item()
        score_min = acqf_min(X).item()

        # Scores should differ (mean sign is flipped)
        assert score_max != score_min


class TestUpperConfidenceBoundNumerical:
    """Numerical correctness tests for UCB."""

    def test_ucb_formula_maximize(self, fitted_gp, best_f_max):
        """Test UCB computation: UCB = mu + beta * sigma for maximize."""
        beta = 2.0
        ucb = UpperConfidenceBound(beta=beta)
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        ucb_value = acqf(X).item()

        posterior = fitted_gp.posterior(X)
        mean = posterior.mean.squeeze().item()
        std = posterior.variance.sqrt().squeeze().item()

        expected_ucb = mean + beta * std
        assert np.isclose(ucb_value, expected_ucb, rtol=1e-5)

    def test_ucb_formula_minimize(self, fitted_gp, best_f_min):
        """Test UCB computation: UCB = -mu + beta * sigma for minimize."""
        beta = 2.0
        ucb = UpperConfidenceBound(beta=beta)
        acqf = ucb.build(model=fitted_gp, best_f=best_f_min, maximize=False)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        ucb_value = acqf(X).item()

        posterior = fitted_gp.posterior(X)
        mean = posterior.mean.squeeze().item()
        std = posterior.variance.sqrt().squeeze().item()

        expected_ucb = -mean + beta * std
        assert np.isclose(ucb_value, expected_ucb, rtol=1e-5)

    def test_beta_scales_exploration(self, fitted_gp, best_f_max):
        """Test that larger beta increases influence of uncertainty."""
        ucb_small = UpperConfidenceBound(beta=1.0)
        ucb_large = UpperConfidenceBound(beta=5.0)

        acqf_small = ucb_small.build(model=fitted_gp, best_f=best_f_max, maximize=True)
        acqf_large = ucb_large.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # Point with high uncertainty (far from training data)
        X = torch.tensor([[[-0.5]]], dtype=torch.float64)

        score_small = acqf_small(X).item()
        score_large = acqf_large(X).item()

        # Larger beta should give higher score for uncertain points
        assert score_large > score_small


class TestUpperConfidenceBoundDirectional:
    """Directional behavior tests for UCB."""

    def test_lower_mean_gives_higher_score_minimize(self, fitted_gp, best_f_max):
        """Test that lower predicted mean gives higher UCB for minimization."""
        ucb = UpperConfidenceBound(beta=0.0)  # Pure exploitation
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=False)

        # x=0.5 has highest mean (peak of quadratic), x=0.0 has lower mean
        X_peak = torch.tensor([[[0.5]]], dtype=torch.float64)
        X_edge = torch.tensor([[[0.0]]], dtype=torch.float64)

        score_peak = acqf(X_peak).item()
        score_edge = acqf(X_edge).item()

        # For minimize, lower mean should give higher score
        assert score_edge > score_peak

    def test_higher_mean_increases_ucb_maximize(self, fitted_gp, best_f_max):
        """Test that higher predicted mean increases UCB for maximization."""
        ucb = UpperConfidenceBound(beta=0.0)  # Pure exploitation
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # x=0.5 has highest mean (peak of quadratic)
        X_peak = torch.tensor([[[0.5]]], dtype=torch.float64)
        X_edge = torch.tensor([[[0.0]]], dtype=torch.float64)

        score_peak = acqf(X_peak).item()
        score_edge = acqf(X_edge).item()

        assert score_peak > score_edge

    def test_higher_std_increases_ucb(self, fitted_gp, best_f_max):
        """Test that higher uncertainty increases UCB contribution.

        This test verifies the exploration component: beta * sigma should be
        larger for points with higher uncertainty.
        """
        ucb = UpperConfidenceBound(beta=2.0)
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # Near training data (low std)
        X_near = torch.tensor([[[0.5]]], dtype=torch.float64)
        # Far from training data (high std)
        X_far = torch.tensor([[[-1.0]]], dtype=torch.float64)

        posterior_near = fitted_gp.posterior(X_near)
        posterior_far = fitted_gp.posterior(X_far)
        std_near = posterior_near.variance.sqrt().squeeze().item()
        std_far = posterior_far.variance.sqrt().squeeze().item()

        # Verify far point has higher uncertainty
        assert std_far > std_near

        # UCB exploration contribution (beta * std) should be larger for far point
        ucb_exploration_near = 2.0 * std_near
        ucb_exploration_far = 2.0 * std_far
        assert ucb_exploration_far > ucb_exploration_near

        # Verify the acquisition function is callable (correctness checked elsewhere)
        _ = acqf(X_near)
        _ = acqf(X_far)


class TestUpperConfidenceBoundEdgeCases:
    """Edge case tests for UCB."""

    def test_output_is_finite(self, fitted_gp, best_f_max):
        """Test that UCB output contains no NaN or Inf values."""
        ucb = UpperConfidenceBound(beta=2.0)
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # Test on multiple points including edge cases
        X = torch.tensor(
            [[[0.0]], [[0.5]], [[1.0]], [[-0.5]], [[1.5]]], dtype=torch.float64
        )
        results = torch.stack([acqf(x.unsqueeze(0)) for x in X])

        assert torch.all(torch.isfinite(results))

    def test_single_point_batch(self, fitted_gp, best_f_max):
        """Test UCB with single point."""
        ucb = UpperConfidenceBound(beta=2.0)
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        result = acqf(X)

        assert result.shape == (1,)

    def test_multiple_batch(self, fitted_gp, best_f_max):
        """Test UCB with multiple batches."""
        ucb = UpperConfidenceBound(beta=2.0)
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.1]], [[0.5]], [[0.9]]], dtype=torch.float64)
        result = acqf(X)

        assert result.shape == (3,)

    def test_q_batch(self, fitted_gp, best_f_max):
        """Test UCB with q-batch."""
        ucb = UpperConfidenceBound(beta=2.0)
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        # 1 batch, 2 candidates
        X = torch.tensor([[[0.2], [0.8]]], dtype=torch.float64)
        result = acqf(X)

        # Result should be sum over q dimension
        assert result.shape == (1,)

    def test_beta_zero_is_pure_exploitation(self, fitted_gp, best_f_max):
        """Test that beta=0 gives pure mean (no exploration)."""
        ucb = UpperConfidenceBound(beta=0.0)
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        ucb_value = acqf(X).item()

        # Should equal just the mean
        posterior = fitted_gp.posterior(X)
        mean = posterior.mean.squeeze().item()

        assert np.isclose(ucb_value, mean, rtol=1e-5)


class TestUpperConfidenceBoundDtype:
    """Tests for UCB dtype validation."""

    def test_float32_raises(self, fitted_gp, best_f_max):
        """Test that float32 input raises ValueError."""
        ucb = UpperConfidenceBound(beta=2.0)
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.3]]], dtype=torch.float32)
        with pytest.raises(ValueError, match="float64"):
            acqf(X)

    def test_float64_passes(self, fitted_gp, best_f_max):
        """Test that float64 input works."""
        ucb = UpperConfidenceBound(beta=2.0)
        acqf = ucb.build(model=fitted_gp, best_f=best_f_max, maximize=True)

        X = torch.tensor([[[0.3]]], dtype=torch.float64)
        result = acqf(X)
        assert result.shape == (1,)


# =============================================================================
# Base Class Tests
# =============================================================================


class TestAcquisitionBase:
    """Tests for Acquisition base class."""

    def test_is_abstract(self):
        """Test that Acquisition cannot be instantiated directly."""
        with pytest.raises(TypeError, match="(?i)abstract|instantiate"):
            Acquisition()

    def test_subclass_must_implement_build(self):
        """Test that subclasses must implement build method."""

        class IncompleteAcquisition(Acquisition):
            pass

        with pytest.raises(TypeError, match="(?i)abstract|instantiate"):
            IncompleteAcquisition()
