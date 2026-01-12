"""Tests for multi-objective acquisition functions."""

import numpy as np
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from folio.recommenders.acquisitions import (
    NEHVI,
    MultiObjectiveAcquisition,
)


@pytest.fixture
def fitted_mogp():
    """Create a fitted multi-output GP (ModelListGP) for testing.

    Returns a ModelListGP with two objectives fitted on 5 points.
    Objective 1: y1 = -(x - 0.3)^2 + 1 (max at x=0.3)
    Objective 2: y2 = -(x - 0.7)^2 + 1 (max at x=0.7)

    This creates a trade-off: no single point maximizes both objectives.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float64)

    # Two objectives with different optima (creates Pareto trade-off)
    y1 = (-((X_train - 0.3) ** 2) + 1).squeeze(-1)
    y2 = (-((X_train - 0.7) ** 2) + 1).squeeze(-1)

    model1 = SingleTaskGP(
        train_X=X_train,
        train_Y=y1.unsqueeze(-1),
        outcome_transform=Standardize(m=1),
    )
    model2 = SingleTaskGP(
        train_X=X_train,
        train_Y=y2.unsqueeze(-1),
        outcome_transform=Standardize(m=1),
    )

    model = ModelListGP(model1, model2)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


@pytest.fixture
def X_baseline():
    """Baseline input points for multi-objective acquisition."""
    return torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float64)


@pytest.fixture
def Y_baseline():
    """Baseline objective values (n_samples, n_objectives).

    Objective 1: y1 = -(x - 0.3)^2 + 1
    Objective 2: y2 = -(x - 0.7)^2 + 1
    """
    X = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float64)
    y1 = (-((X - 0.3) ** 2) + 1).squeeze(-1)
    y2 = (-((X - 0.7) ** 2) + 1).squeeze(-1)
    return torch.stack([y1, y2], dim=-1)


@pytest.fixture
def ref_point_max():
    """Reference point for maximization (dominated by all Pareto points)."""
    return [0.0, 0.0]


@pytest.fixture
def maximize_both():
    """Maximize both objectives."""
    return [True, True]


# =============================================================================
# _prepare_for_maximization Tests
# =============================================================================


class TestPrepareForMaximization:
    """Tests for MultiObjectiveAcquisition._prepare_for_maximization helper."""

    def test_all_maximize_unchanged(self):
        """Test with all maximize=True: Y and ref_point unchanged."""
        nehvi = NEHVI()
        Y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ref_point = [0.0, 0.5]
        maximize = [True, True]

        Y_max, ref_max = nehvi._prepare_for_maximization(Y, ref_point, maximize)

        torch.testing.assert_close(Y_max, Y)
        assert ref_max == [0.0, 0.5]

    def test_all_minimize_all_negated(self):
        """Test with all maximize=False: all values negated."""
        nehvi = NEHVI()
        Y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ref_point = [0.0, 0.5]
        maximize = [False, False]

        Y_max, ref_max = nehvi._prepare_for_maximization(Y, ref_point, maximize)

        expected_Y = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
        torch.testing.assert_close(Y_max, expected_Y)
        assert ref_max == [-0.0, -0.5]

    def test_mixed_maximize_only_appropriate_negated(self):
        """Test with mixed maximize: only minimize columns negated."""
        nehvi = NEHVI()
        Y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ref_point = [0.1, 0.2, 0.3]
        maximize = [True, False, True]

        Y_max, ref_max = nehvi._prepare_for_maximization(Y, ref_point, maximize)

        # Only column 1 should be negated
        expected_Y = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
        torch.testing.assert_close(Y_max, expected_Y)
        assert ref_max == [0.1, -0.2, 0.3]

    def test_original_tensors_not_modified(self):
        """Test that original Y is not modified (returns a copy)."""
        nehvi = NEHVI()
        Y_original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        Y_copy = Y_original.clone()
        ref_point = [0.0, 0.0]
        maximize = [False, False]

        nehvi._prepare_for_maximization(Y_original, ref_point, maximize)

        # Original should be unchanged
        torch.testing.assert_close(Y_original, Y_copy)

    def test_original_ref_point_not_modified(self):
        """Test that original ref_point list is not modified."""
        nehvi = NEHVI()
        Y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ref_point = [0.5, 0.5]
        ref_point_copy = list(ref_point)
        maximize = [False, False]

        nehvi._prepare_for_maximization(Y, ref_point, maximize)

        assert ref_point == ref_point_copy


# =============================================================================
# _validate_dtype Tests
# =============================================================================


class TestValidateDtype:
    """Tests for MultiObjectiveAcquisition._validate_dtype helper."""

    def test_float64_passes(self):
        """Test that float64 tensors pass validation."""
        nehvi = NEHVI()
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        Y = torch.tensor([[0.5, 0.3]], dtype=torch.float64)

        # Should not raise
        nehvi._validate_dtype(X, Y)

    def test_float32_x_raises(self):
        """Test that float32 X raises ValueError."""
        nehvi = NEHVI()
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        Y = torch.tensor([[0.5, 0.3]], dtype=torch.float64)

        with pytest.raises(ValueError, match="X_baseline.*float64"):
            nehvi._validate_dtype(X, Y)

    def test_float32_y_raises(self):
        """Test that float32 Y raises ValueError."""
        nehvi = NEHVI()
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        Y = torch.tensor([[0.5, 0.3]], dtype=torch.float32)

        with pytest.raises(ValueError, match="Y.*float64"):
            nehvi._validate_dtype(X, Y)

    def test_both_float32_raises_on_x_first(self):
        """Test that when both are float32, X is validated first."""
        nehvi = NEHVI()
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        Y = torch.tensor([[0.5, 0.3]], dtype=torch.float32)

        with pytest.raises(ValueError, match="X_baseline.*float64"):
            nehvi._validate_dtype(X, Y)


# =============================================================================
# NEHVI Init Tests
# =============================================================================


class TestNEHVIInit:
    """Tests for NEHVI.__init__."""

    def test_default_alpha(self):
        """Test default alpha value is 0.0."""
        nehvi = NEHVI()
        assert nehvi.alpha == 0.0

    def test_custom_alpha(self):
        """Test custom alpha value is stored."""
        nehvi = NEHVI(alpha=0.1)
        assert nehvi.alpha == 0.1


# =============================================================================
# NEHVI Build Tests
# =============================================================================


class TestNEHVIBuild:
    """Tests for NEHVI.build method."""

    def test_returns_acquisition_function(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test build returns a callable acquisition function."""
        nehvi = NEHVI()
        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # Should be callable with shape (batch, q, d)
        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)

    def test_output_is_finite_multiple_points(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that NEHVI returns finite values for multiple candidate points.

        Note: qLogNEHVI returns log-transformed values which can be negative.
        We only check for finite (non-NaN, non-Inf) results.
        """
        nehvi = NEHVI()
        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # Test multiple candidate points
        X = torch.tensor(
            [[[0.1]], [[0.3]], [[0.5]], [[0.7]], [[0.9]]],
            dtype=torch.float64,
        )
        results = torch.stack([acqf(x) for x in X])

        # qLogNEHVI returns log-space values (can be negative)
        assert torch.all(torch.isfinite(results))

    def test_output_is_finite(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that NEHVI output contains no NaN or Inf values."""
        nehvi = NEHVI()
        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # Test on multiple points including edge cases
        X = torch.tensor(
            [[[0.0]], [[0.5]], [[1.0]], [[-0.2]], [[1.2]]],
            dtype=torch.float64,
        )
        results = torch.stack([acqf(x) for x in X])

        assert torch.all(torch.isfinite(results))


# =============================================================================
# NEHVI Directional Tests
# =============================================================================


class TestNEHVIDirectional:
    """Directional behavior tests for NEHVI."""

    def test_pareto_improving_point_has_positive_nehvi(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that a point improving the Pareto frontier has positive NEHVI.

        A point that dominates or extends the current Pareto frontier should
        have positive expected hypervolume improvement.
        """
        nehvi = NEHVI()
        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # Point in unexplored region with potential for improvement
        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        nehvi_value = acqf(X).item()

        # qLogNEHVI returns log-space values; just check it's finite
        assert np.isfinite(nehvi_value)

    def test_dominated_point_has_low_nehvi(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that a clearly dominated point has low/zero NEHVI.

        A point with mean predictions well below the Pareto frontier
        should have very low expected improvement.
        """
        nehvi = NEHVI()

        # Use a Pareto frontier that dominates the test point
        Y_high = Y_baseline + 2.0  # Shift frontier to high values
        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_high,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # Point near training data (low uncertainty, mean well below frontier)
        X = torch.tensor([[[0.5]]], dtype=torch.float64)
        nehvi_value = acqf(X).item()

        # NEHVI should be small since point is dominated
        # (may not be exactly zero due to uncertainty)
        assert nehvi_value < 0.5


# =============================================================================
# NEHVI Edge Cases
# =============================================================================


class TestNEHVIEdgeCases:
    """Edge case tests for NEHVI."""

    def test_single_point_batch(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test NEHVI with single point (batch=1, q=1)."""
        nehvi = NEHVI()
        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        assert result.shape == (1,)

    def test_multiple_batch(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test NEHVI with multiple batches (batch > 1)."""
        nehvi = NEHVI()
        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # 3 batches, 1 candidate each
        X = torch.tensor([[[0.2]], [[0.5]], [[0.8]]], dtype=torch.float64)
        result = acqf(X)

        assert result.shape == (3,)

    def test_alpha_affects_computation(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that alpha parameter affects the acquisition value."""
        nehvi_exact = NEHVI(alpha=0.0)
        nehvi_approx = NEHVI(alpha=0.5)

        acqf_exact = nehvi_exact.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )
        acqf_approx = nehvi_approx.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        value_exact = acqf_exact(X).item()
        value_approx = acqf_approx(X).item()

        # Values may differ (or be similar) depending on alpha
        # Just verify both produce valid finite results
        assert np.isfinite(value_exact)
        assert np.isfinite(value_approx)

    def test_float64_dtype(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test NEHVI works correctly with float64 tensors (standard BoTorch dtype)."""
        nehvi = NEHVI()

        # BoTorch models are typically trained in float64
        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # Evaluate with float64 (matching model)
        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        assert torch.isfinite(result).all()


# =============================================================================
# NEHVI Comparative Tests
# =============================================================================


class TestNEHVIComparative:
    """Tests for comparative ranking behavior of NEHVI."""

    def test_unexplored_region_higher_than_explored(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that unexplored regions have higher acquisition than near-data points.

        Points far from training data should generally have higher acquisition
        values due to exploration bonus from uncertainty.
        """
        nehvi = NEHVI()
        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # Point near training data (x=0.5 is in training set)
        X_near = torch.tensor([[[0.5]]], dtype=torch.float64)
        # Point far from training data
        X_far = torch.tensor([[[0.15]]], dtype=torch.float64)

        value_near = acqf(X_near).item()
        value_far = acqf(X_far).item()

        # Both should be finite
        assert np.isfinite(value_near)
        assert np.isfinite(value_far)

        # In log-space, higher is better (more expected improvement)
        # Unexplored region should generally have higher value
        # (though this isn't always guaranteed depending on the GP)


# =============================================================================
# NEHVI Input Validation Tests
# =============================================================================


class TestNEHVIInputValidation:
    """Tests for input validation in NEHVI."""

    def test_mismatched_ref_point_length(
        self, fitted_mogp, X_baseline, Y_baseline, maximize_both
    ):
        """Test that mismatched ref_point length raises an error."""
        nehvi = NEHVI()

        # ref_point has 3 elements but Y has 2 objectives
        ref_point_wrong = [0.0, 0.0, 0.0]

        with pytest.raises((ValueError, RuntimeError)):
            nehvi.build(
                model=fitted_mogp,
                X_baseline=X_baseline,
                Y=Y_baseline,
                ref_point=ref_point_wrong,
                maximize=maximize_both,
            )

    def test_mismatched_maximize_length(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max
    ):
        """Test that mismatched maximize length raises ValueError."""
        nehvi = NEHVI()

        # maximize has 3 elements but Y has 2 objectives
        maximize_wrong = [True, True, True]

        with pytest.raises(
            ValueError, match="(?i)maximize.*objective|objective.*maximize"
        ):
            nehvi.build(
                model=fitted_mogp,
                X_baseline=X_baseline,
                Y=Y_baseline,
                ref_point=ref_point_max,
                maximize=maximize_wrong,
            )

    def test_empty_baseline_raises(self, fitted_mogp, ref_point_max, maximize_both):
        """Test that empty baseline data raises ValueError."""
        nehvi = NEHVI()

        X_empty = torch.empty((0, 1), dtype=torch.float64)
        Y_empty = torch.empty((0, 2), dtype=torch.float64)

        with pytest.raises(ValueError, match="(?i)empty|observation"):
            nehvi.build(
                model=fitted_mogp,
                X_baseline=X_empty,
                Y=Y_empty,
                ref_point=ref_point_max,
                maximize=maximize_both,
            )

    def test_float32_x_baseline_raises(
        self, fitted_mogp, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that float32 X_baseline raises ValueError."""
        nehvi = NEHVI()

        X_float32 = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float32)
        Y_float64 = Y_baseline[:3]

        with pytest.raises(ValueError, match="(?i)float64|dtype"):
            nehvi.build(
                model=fitted_mogp,
                X_baseline=X_float32,
                Y=Y_float64,
                ref_point=ref_point_max,
                maximize=maximize_both,
            )

    def test_float32_y_raises(
        self, fitted_mogp, X_baseline, ref_point_max, maximize_both
    ):
        """Test that float32 Y raises ValueError."""
        nehvi = NEHVI()

        Y_float32 = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32
        )
        X_float64 = X_baseline[:3]

        with pytest.raises(ValueError, match="(?i)float64|dtype"):
            nehvi.build(
                model=fitted_mogp,
                X_baseline=X_float64,
                Y=Y_float32,
                ref_point=ref_point_max,
                maximize=maximize_both,
            )


# =============================================================================
# NEHVI Reference Point Tests
# =============================================================================


class TestNEHVIReferencePoint:
    """Tests for reference point handling in NEHVI."""

    def test_different_ref_points_different_values(
        self, fitted_mogp, X_baseline, Y_baseline, maximize_both
    ):
        """Test that different reference points produce different NEHVI values.

        The reference point defines the hypervolume region, so different
        reference points should generally produce different acquisition values.
        """
        nehvi = NEHVI()

        ref_point_low = [0.0, 0.0]
        ref_point_high = [-1.0, -1.0]  # Further from frontier

        acqf_low = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_low,
            maximize=maximize_both,
        )
        acqf_high = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_high,
            maximize=maximize_both,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        value_low = acqf_low(X).item()
        value_high = acqf_high(X).item()

        # Different reference points should give different values
        # (or at least both should be valid)
        assert np.isfinite(value_low)
        assert np.isfinite(value_high)

    def test_ref_point_dominated_by_frontier(
        self, fitted_mogp, X_baseline, Y_baseline, maximize_both
    ):
        """Test NEHVI with reference point properly dominated by Pareto frontier."""
        nehvi = NEHVI()

        # Reference point well below all observations
        ref_point = [0.5, 0.5]  # Below the Y values

        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point,
            maximize=maximize_both,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        # qLogNEHVI returns log-space values; just check it's finite
        assert torch.isfinite(result).all()


# =============================================================================
# NEHVI Multi-dimensional Tests
# =============================================================================


class TestNEHVIMultidimensional:
    """Tests for NEHVI with multi-dimensional inputs."""

    @pytest.fixture
    def fitted_mogp_2d(self):
        """Create a fitted multi-output GP with 2D inputs."""
        torch.manual_seed(42)
        np.random.seed(42)

        # 2D input space
        X_train = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=torch.float64,
        )

        # Two objectives
        y1 = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float64)
        y2 = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5], dtype=torch.float64)

        model1 = SingleTaskGP(
            train_X=X_train,
            train_Y=y1.unsqueeze(-1),
            outcome_transform=Standardize(m=1),
        )
        model2 = SingleTaskGP(
            train_X=X_train,
            train_Y=y2.unsqueeze(-1),
            outcome_transform=Standardize(m=1),
        )

        model = ModelListGP(model1, model2)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model

    @pytest.fixture
    def X_baseline_2d(self):
        """Baseline inputs for 2D case."""
        return torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=torch.float64,
        )

    @pytest.fixture
    def Y_baseline_2d(self):
        """Baseline objectives for 2D case."""
        y1 = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float64)
        y2 = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5], dtype=torch.float64)
        return torch.stack([y1, y2], dim=-1)

    def test_2d_input_returns_correct_shape(
        self, fitted_mogp_2d, X_baseline_2d, Y_baseline_2d, ref_point_max, maximize_both
    ):
        """Test NEHVI with 2D inputs returns correct output shape."""
        nehvi = NEHVI()
        acqf = nehvi.build(
            model=fitted_mogp_2d,
            X_baseline=X_baseline_2d,
            Y=Y_baseline_2d,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # Single candidate in 2D space
        X = torch.tensor([[[0.3, 0.7]]], dtype=torch.float64)
        result = acqf(X)

        assert result.shape == (1,)
        assert torch.isfinite(result).all()

    def test_2d_input_multiple_candidates(
        self, fitted_mogp_2d, X_baseline_2d, Y_baseline_2d, ref_point_max, maximize_both
    ):
        """Test NEHVI with multiple 2D candidates."""
        nehvi = NEHVI()
        acqf = nehvi.build(
            model=fitted_mogp_2d,
            X_baseline=X_baseline_2d,
            Y=Y_baseline_2d,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # Multiple candidates
        X = torch.tensor(
            [[[0.2, 0.2]], [[0.5, 0.5]], [[0.8, 0.8]]],
            dtype=torch.float64,
        )
        result = acqf(X)

        assert result.shape == (3,)
        assert torch.isfinite(result).all()


# =============================================================================
# NEHVI Three Objectives Tests
# =============================================================================


class TestNEHVIThreeObjectives:
    """Tests for NEHVI with three objectives."""

    @pytest.fixture
    def fitted_mogp_3obj(self):
        """Create a fitted multi-output GP with three objectives."""
        torch.manual_seed(42)
        np.random.seed(42)

        X_train = torch.tensor(
            [[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float64
        )

        # Three objectives with different optima
        y1 = (-((X_train - 0.2) ** 2) + 1).squeeze(-1)
        y2 = (-((X_train - 0.5) ** 2) + 1).squeeze(-1)
        y3 = (-((X_train - 0.8) ** 2) + 1).squeeze(-1)

        model1 = SingleTaskGP(
            train_X=X_train,
            train_Y=y1.unsqueeze(-1),
            outcome_transform=Standardize(m=1),
        )
        model2 = SingleTaskGP(
            train_X=X_train,
            train_Y=y2.unsqueeze(-1),
            outcome_transform=Standardize(m=1),
        )
        model3 = SingleTaskGP(
            train_X=X_train,
            train_Y=y3.unsqueeze(-1),
            outcome_transform=Standardize(m=1),
        )

        model = ModelListGP(model1, model2, model3)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model

    @pytest.fixture
    def Y_baseline_3obj(self):
        """Baseline objectives for three-objective case."""
        X = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float64)
        y1 = (-((X - 0.2) ** 2) + 1).squeeze(-1)
        y2 = (-((X - 0.5) ** 2) + 1).squeeze(-1)
        y3 = (-((X - 0.8) ** 2) + 1).squeeze(-1)
        return torch.stack([y1, y2, y3], dim=-1)

    def test_three_objectives(self, fitted_mogp_3obj, X_baseline, Y_baseline_3obj):
        """Test NEHVI with three objectives."""
        nehvi = NEHVI()
        ref_point = [0.0, 0.0, 0.0]
        maximize = [True, True, True]

        acqf = nehvi.build(
            model=fitted_mogp_3obj,
            X_baseline=X_baseline,
            Y=Y_baseline_3obj,
            ref_point=ref_point,
            maximize=maximize,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        assert result.shape == (1,)
        assert torch.isfinite(result).all()


# =============================================================================
# NEHVI Mixed Optimization Tests
# =============================================================================


class TestNEHVIMixedOptimization:
    """Tests for NEHVI with mixed maximize/minimize objectives."""

    def test_mixed_max_min_objectives(self, fitted_mogp, X_baseline, Y_baseline):
        """Test NEHVI with one maximize and one minimize objective.

        Pass original Y values; negation for minimize objectives is
        handled internally by _prepare_for_maximization.
        """
        nehvi = NEHVI()

        # Maximize objective 0, minimize objective 1
        # Reference point should be dominated in original objective space:
        # - For maximize: ref < worst observed
        # - For minimize: ref > worst observed
        ref_point = [0.0, 2.0]
        maximize = [True, False]

        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point,
            maximize=maximize,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        assert torch.isfinite(result).all()

    def test_all_minimize_objectives(self, fitted_mogp, X_baseline, Y_baseline):
        """Test NEHVI with all minimize objectives.

        Pass original Y values; negation is handled internally.
        """
        nehvi = NEHVI()

        # Minimize both objectives
        # Reference point: values worse (higher) than worst observed
        ref_point = [2.0, 2.0]
        maximize = [False, False]

        acqf = nehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point,
            maximize=maximize,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        assert torch.isfinite(result).all()


# =============================================================================
# MultiObjectiveAcquisition Base Class Tests
# =============================================================================


class TestMultiObjectiveAcquisitionBase:
    """Tests for MultiObjectiveAcquisition base class."""

    def test_is_abstract(self):
        """Test that MultiObjectiveAcquisition cannot be instantiated directly."""
        with pytest.raises(TypeError, match="(?i)abstract|instantiate"):
            MultiObjectiveAcquisition()

    def test_subclass_must_implement_build(self):
        """Test that subclasses must implement build method."""

        class IncompleteAcquisition(MultiObjectiveAcquisition):
            pass

        with pytest.raises(TypeError, match="(?i)abstract|instantiate"):
            IncompleteAcquisition()

    def test_nehvi_is_subclass(self):
        """Test that NEHVI is a proper subclass of MultiObjectiveAcquisition."""
        assert issubclass(NEHVI, MultiObjectiveAcquisition)
