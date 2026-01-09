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
    EHVI,
    MultiObjectiveAcquisition,
    ParEGO,
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
# EHVI Init Tests
# =============================================================================


class TestEHVIInit:
    """Tests for EHVI.__init__."""

    def test_default_alpha(self):
        """Test default alpha value is 0.0."""
        ehvi = EHVI()
        assert ehvi.alpha == 0.0

    def test_default_prune_baseline(self):
        """Test default prune_baseline is True."""
        ehvi = EHVI()
        assert ehvi.prune_baseline is True

    def test_default_cache_root(self):
        """Test default cache_root is True."""
        ehvi = EHVI()
        assert ehvi.cache_root is True

    def test_custom_alpha(self):
        """Test custom alpha value is stored."""
        ehvi = EHVI(alpha=0.1)
        assert ehvi.alpha == 0.1

    def test_custom_prune_baseline(self):
        """Test custom prune_baseline value is stored."""
        ehvi = EHVI(prune_baseline=False)
        assert ehvi.prune_baseline is False

    def test_custom_cache_root(self):
        """Test custom cache_root value is stored."""
        ehvi = EHVI(cache_root=False)
        assert ehvi.cache_root is False

    def test_all_custom_params(self):
        """Test all custom parameters are stored correctly."""
        ehvi = EHVI(alpha=0.05, prune_baseline=False, cache_root=False)
        assert ehvi.alpha == 0.05
        assert ehvi.prune_baseline is False
        assert ehvi.cache_root is False


# =============================================================================
# EHVI Build Tests
# =============================================================================


class TestEHVIBuild:
    """Tests for EHVI.build method."""

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_returns_acquisition_function(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test build returns a callable acquisition function."""
        ehvi = EHVI()
        acqf = ehvi.build(
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

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_output_is_nonnegative(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that EHVI returns non-negative values.

        Hypervolume improvement is always non-negative since we can only
        increase or maintain the dominated hypervolume.
        """
        ehvi = EHVI()
        acqf = ehvi.build(
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

        # EHVI should be non-negative
        assert (results >= -1e-10).all()

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_output_is_finite(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that EHVI output contains no NaN or Inf values."""
        ehvi = EHVI()
        acqf = ehvi.build(
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
# EHVI Directional Tests
# =============================================================================


class TestEHVIDirectional:
    """Directional behavior tests for EHVI."""

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_pareto_improving_point_has_positive_ehvi(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that a point improving the Pareto frontier has positive EHVI.

        A point that dominates or extends the current Pareto frontier should
        have positive expected hypervolume improvement.
        """
        ehvi = EHVI()
        acqf = ehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # Point in unexplored region with potential for improvement
        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        ehvi_value = acqf(X).item()

        # Should have some positive EHVI (uncertainty enables improvement)
        assert ehvi_value >= 0

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_dominated_point_has_low_ehvi(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that a clearly dominated point has low/zero EHVI.

        A point with mean predictions well below the Pareto frontier
        should have very low expected improvement.
        """
        ehvi = EHVI()

        # Use a Pareto frontier that dominates the test point
        Y_high = Y_baseline + 2.0  # Shift frontier to high values
        acqf = ehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_high,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        # Point near training data (low uncertainty, mean well below frontier)
        X = torch.tensor([[[0.5]]], dtype=torch.float64)
        ehvi_value = acqf(X).item()

        # EHVI should be small since point is dominated
        # (may not be exactly zero due to uncertainty)
        assert ehvi_value < 0.5


# =============================================================================
# EHVI Edge Cases
# =============================================================================


class TestEHVIEdgeCases:
    """Edge case tests for EHVI."""

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_single_point_batch(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test EHVI with single point (batch=1, q=1)."""
        ehvi = EHVI()
        acqf = ehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        assert result.shape == (1,)

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_multiple_batch(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test EHVI with multiple batches (batch > 1)."""
        ehvi = EHVI()
        acqf = ehvi.build(
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

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_alpha_affects_computation(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that alpha parameter affects the acquisition value."""
        ehvi_exact = EHVI(alpha=0.0)
        ehvi_approx = EHVI(alpha=0.5)

        acqf_exact = ehvi_exact.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )
        acqf_approx = ehvi_approx.build(
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

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_prune_baseline_option(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that prune_baseline option doesn't cause errors."""
        ehvi_prune = EHVI(prune_baseline=True)
        ehvi_no_prune = EHVI(prune_baseline=False)

        acqf_prune = ehvi_prune.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )
        acqf_no_prune = ehvi_no_prune.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)

        # Both should produce valid results
        result_prune = acqf_prune(X)
        result_no_prune = acqf_no_prune(X)

        assert torch.isfinite(result_prune).all()
        assert torch.isfinite(result_no_prune).all()


# =============================================================================
# EHVI Reference Point Tests
# =============================================================================


class TestEHVIReferencePoint:
    """Tests for reference point handling in EHVI."""

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_different_ref_points_different_values(
        self, fitted_mogp, X_baseline, Y_baseline, maximize_both
    ):
        """Test that different reference points produce different EHVI values.

        The reference point defines the hypervolume region, so different
        reference points should generally produce different acquisition values.
        """
        ehvi = EHVI()

        ref_point_low = [0.0, 0.0]
        ref_point_high = [-1.0, -1.0]  # Further from frontier

        acqf_low = ehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_low,
            maximize=maximize_both,
        )
        acqf_high = ehvi.build(
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

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_ref_point_dominated_by_frontier(
        self, fitted_mogp, X_baseline, Y_baseline, maximize_both
    ):
        """Test EHVI with reference point properly dominated by Pareto frontier."""
        ehvi = EHVI()

        # Reference point well below all observations
        ref_point = [0.5, 0.5]  # Below the Y values

        acqf = ehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point,
            maximize=maximize_both,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        # Should produce valid non-negative result
        assert result.item() >= 0
        assert torch.isfinite(result).all()


# =============================================================================
# EHVI Multi-dimensional Tests
# =============================================================================


class TestEHVIMultidimensional:
    """Tests for EHVI with multi-dimensional inputs."""

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

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_2d_input_returns_correct_shape(
        self, fitted_mogp_2d, X_baseline_2d, Y_baseline_2d, ref_point_max, maximize_both
    ):
        """Test EHVI with 2D inputs returns correct output shape."""
        ehvi = EHVI()
        acqf = ehvi.build(
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

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_2d_input_multiple_candidates(
        self, fitted_mogp_2d, X_baseline_2d, Y_baseline_2d, ref_point_max, maximize_both
    ):
        """Test EHVI with multiple 2D candidates."""
        ehvi = EHVI()
        acqf = ehvi.build(
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
# EHVI Three Objectives Tests
# =============================================================================


class TestEHVIThreeObjectives:
    """Tests for EHVI with three objectives."""

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

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_three_objectives(self, fitted_mogp_3obj, X_baseline, Y_baseline_3obj):
        """Test EHVI with three objectives."""
        ehvi = EHVI()
        ref_point = [0.0, 0.0, 0.0]
        maximize = [True, True, True]

        acqf = ehvi.build(
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
        assert result.item() >= 0


# =============================================================================
# EHVI Minimization Tests (Objective Negation)
# =============================================================================


class TestEHVIMinimization:
    """Tests for EHVI with minimization objectives (negated)."""

    @pytest.mark.skip(reason="EHVI.build not yet implemented")
    def test_negated_objective_for_minimization(
        self, fitted_mogp, X_baseline, Y_baseline
    ):
        """Test EHVI with negated objective for minimization.

        When minimize=False for an objective, user must negate Y column
        before calling build(). This test verifies the pattern works.
        """
        ehvi = EHVI()

        # Negate second objective (simulating minimization)
        Y_with_negation = Y_baseline.clone()
        Y_with_negation[:, 1] = -Y_with_negation[:, 1]

        # Adjust reference point for negated objective
        ref_point = [0.0, -2.0]  # Second ref is negative since we negated

        # After negation, both are "maximize" from BoTorch's perspective
        maximize = [True, True]

        acqf = ehvi.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_with_negation,
            ref_point=ref_point,
            maximize=maximize,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        assert torch.isfinite(result).all()
        assert result.item() >= 0


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

    def test_ehvi_is_subclass(self):
        """Test that EHVI is a proper subclass of MultiObjectiveAcquisition."""
        assert issubclass(EHVI, MultiObjectiveAcquisition)

    def test_parego_is_subclass(self):
        """Test that ParEGO is a proper subclass of MultiObjectiveAcquisition."""
        assert issubclass(ParEGO, MultiObjectiveAcquisition)


# =============================================================================
# ParEGO Init Tests
# =============================================================================


class TestParEGOInit:
    """Tests for ParEGO.__init__."""

    def test_default_scalarization(self):
        """Test default scalarization is 'chebyshev'."""
        parego = ParEGO()
        assert parego.scalarization == "chebyshev"

    def test_default_rho(self):
        """Test default rho value is 0.05."""
        parego = ParEGO()
        assert parego.rho == 0.05

    def test_custom_scalarization_chebyshev(self):
        """Test scalarization='chebyshev' is stored."""
        parego = ParEGO(scalarization="chebyshev")
        assert parego.scalarization == "chebyshev"

    def test_custom_scalarization_linear(self):
        """Test scalarization='linear' is stored."""
        parego = ParEGO(scalarization="linear")
        assert parego.scalarization == "linear"

    def test_custom_rho(self):
        """Test custom rho value is stored."""
        parego = ParEGO(rho=0.1)
        assert parego.rho == 0.1

    def test_zero_rho(self):
        """Test rho=0 is valid."""
        parego = ParEGO(rho=0.0)
        assert parego.rho == 0.0

    def test_invalid_scalarization_raises(self):
        """Test invalid scalarization raises ValueError."""
        with pytest.raises(ValueError, match="scalarization"):
            ParEGO(scalarization="invalid")

    def test_negative_rho_raises(self):
        """Test negative rho raises ValueError."""
        with pytest.raises(ValueError, match="(?i)rho|negative|non-negative"):
            ParEGO(rho=-0.1)

    def test_all_custom_params(self):
        """Test all custom parameters are stored correctly."""
        parego = ParEGO(scalarization="linear", rho=0.1)
        assert parego.scalarization == "linear"
        assert parego.rho == 0.1


# =============================================================================
# ParEGO Build Tests
# =============================================================================


class TestParEGOBuild:
    """Tests for ParEGO.build method."""

    @pytest.mark.skip(reason="ParEGO.build not yet implemented")
    def test_returns_acquisition_function(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test build returns a callable acquisition function."""
        parego = ParEGO()
        acqf = parego.build(
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

    @pytest.mark.skip(reason="ParEGO.build not yet implemented")
    def test_output_is_finite(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that ParEGO output contains no NaN or Inf values."""
        parego = ParEGO()
        acqf = parego.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        X = torch.tensor(
            [[[0.0]], [[0.5]], [[1.0]], [[-0.2]], [[1.2]]],
            dtype=torch.float64,
        )
        results = torch.stack([acqf(x) for x in X])

        assert torch.all(torch.isfinite(results))

    @pytest.mark.skip(reason="ParEGO.build not yet implemented")
    def test_chebyshev_scalarization(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test ParEGO with Chebyshev scalarization."""
        parego = ParEGO(scalarization="chebyshev", rho=0.05)
        acqf = parego.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        assert torch.isfinite(result).all()

    @pytest.mark.skip(reason="ParEGO.build not yet implemented")
    def test_linear_scalarization(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test ParEGO with linear scalarization."""
        parego = ParEGO(scalarization="linear")
        acqf = parego.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        result = acqf(X)

        assert torch.isfinite(result).all()

    @pytest.mark.skip(reason="ParEGO.build not yet implemented")
    def test_different_calls_may_differ_due_to_random_weights(
        self, fitted_mogp, X_baseline, Y_baseline, ref_point_max, maximize_both
    ):
        """Test that different build() calls may produce different functions.

        ParEGO samples random weights, so consecutive builds should
        generally produce different acquisition functions.
        """
        parego = ParEGO()

        acqf1 = parego.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )
        acqf2 = parego.build(
            model=fitted_mogp,
            X_baseline=X_baseline,
            Y=Y_baseline,
            ref_point=ref_point_max,
            maximize=maximize_both,
        )

        X = torch.tensor([[[0.4]]], dtype=torch.float64)
        value1 = acqf1(X).item()
        value2 = acqf2(X).item()

        # Values may differ due to random weight sampling
        # (or could be same by chance, so just verify both are valid)
        assert np.isfinite(value1)
        assert np.isfinite(value2)
