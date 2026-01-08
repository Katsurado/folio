"""Tests for Expected Improvement and Upper Confidence Bound acquisition functions."""

import numpy as np
import pytest
from scipy.stats import norm

from folio.recommenders.acquisitions.functions import (
    ExpectedImprovement,
    UpperConfidenceBound,
)

# =============================================================================
# Expected Improvement Tests
# =============================================================================


class TestExpectedImprovementContract:
    """Test EI satisfies basic contract requirements."""

    def test_output_shape_matches_input(self):
        """EI output shape matches number of candidates."""
        ei = ExpectedImprovement(xi=0.01)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mean = np.array([0.5, 0.8, 0.3])
        std = np.array([0.1, 0.2, 0.15])

        scores = ei.evaluate(X, mean, std, y_best=0.4, objective="maximize")

        assert scores.shape == (3,)

    def test_output_is_finite(self):
        """EI scores contain no NaN or Inf values."""
        ei = ExpectedImprovement(xi=0.01)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        mean = np.array([0.5, 0.8, 0.3, 1.0])
        std = np.array([0.1, 0.2, 0.0, 0.5])

        scores = ei.evaluate(X, mean, std, y_best=0.6, objective="maximize")

        assert np.all(np.isfinite(scores))

    def test_single_point_evaluation(self):
        """EI works with a single candidate point."""
        ei = ExpectedImprovement(xi=0.0)
        X = np.array([[1.0, 2.0]])
        mean = np.array([0.5])
        std = np.array([0.1])

        scores = ei.evaluate(X, mean, std, y_best=0.4, objective="maximize")

        assert scores.shape == (1,)
        assert np.isfinite(scores[0])

    def test_batch_evaluation(self):
        """EI works with many candidate points."""
        ei = ExpectedImprovement(xi=0.01)
        n = 100
        X = np.random.randn(n, 5)
        mean = np.random.randn(n)
        std = np.abs(np.random.randn(n))

        scores = ei.evaluate(X, mean, std, y_best=0.0, objective="maximize")

        assert scores.shape == (n,)
        assert np.all(np.isfinite(scores))

    def test_zero_std_returns_zero(self):
        """EI returns 0 when std=0 (no uncertainty)."""
        ei = ExpectedImprovement(xi=0.0)
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.0, 0.0])

        scores = ei.evaluate(X, mean, std, y_best=0.4, objective="maximize")

        assert scores[0] == 0.0
        assert scores[1] == 0.0

    def test_negative_xi_raises(self):
        """EI constructor raises ValueError for negative xi."""
        with pytest.raises(ValueError, match="(?i)xi|negative|non-negative|>= 0"):
            ExpectedImprovement(xi=-0.1)


class TestExpectedImprovementDirectional:
    """Test EI directional behavior (higher std, better mean, etc.)."""

    def test_higher_std_gives_higher_score(self):
        """Points with higher uncertainty get higher EI (all else equal)."""
        ei = ExpectedImprovement(xi=0.0)
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.5])
        std = np.array([0.1, 0.5])

        scores = ei.evaluate(X, mean, std, y_best=0.5, objective="maximize")

        assert scores[1] > scores[0]

    def test_mean_better_than_ybest_gives_positive_ei_maximize(self):
        """When mean > y_best (maximize), EI is positive."""
        ei = ExpectedImprovement(xi=0.0)
        X = np.array([[1.0]])
        mean = np.array([1.0])
        std = np.array([0.5])

        scores = ei.evaluate(X, mean, std, y_best=0.5, objective="maximize")

        assert scores[0] > 0

    def test_mean_better_than_ybest_gives_positive_ei_minimize(self):
        """When mean < y_best (minimize), EI is positive."""
        ei = ExpectedImprovement(xi=0.0)
        X = np.array([[1.0]])
        mean = np.array([0.0])
        std = np.array([0.5])

        scores = ei.evaluate(X, mean, std, y_best=0.5, objective="minimize")

        assert scores[0] > 0

    def test_mean_worse_than_ybest_low_std_gives_near_zero_ei_maximize(self):
        """When mean << y_best with low std (maximize), EI is near zero."""
        ei = ExpectedImprovement(xi=0.0)
        X = np.array([[1.0]])
        mean = np.array([0.0])
        std = np.array([0.01])

        scores = ei.evaluate(X, mean, std, y_best=1.0, objective="maximize")

        assert scores[0] < 1e-6

    def test_mean_worse_than_ybest_low_std_gives_near_zero_ei_minimize(self):
        """When mean >> y_best with low std (minimize), EI is near zero."""
        ei = ExpectedImprovement(xi=0.0)
        X = np.array([[1.0]])
        mean = np.array([2.0])
        std = np.array([0.01])

        scores = ei.evaluate(X, mean, std, y_best=1.0, objective="minimize")

        assert scores[0] < 1e-6

    def test_xi_affects_output(self):
        """Higher xi reduces EI score (requires larger improvement margin)."""
        ei_low = ExpectedImprovement(xi=0.0)
        ei_high = ExpectedImprovement(xi=0.5)
        X = np.array([[1.0]])
        mean = np.array([0.6])
        std = np.array([0.2])

        score_low = ei_low.evaluate(X, mean, std, y_best=0.5, objective="maximize")
        score_high = ei_high.evaluate(X, mean, std, y_best=0.5, objective="maximize")

        assert score_low[0] > score_high[0]


class TestExpectedImprovementNumerical:
    """Test EI numerical correctness against hand-computed values."""

    def test_ei_maximize_known_value_1(self):
        """
        EI maximize: y_best=3, μ=4, σ=1, ξ=0

        Formula (maximize):
            Z = (μ - y_best - ξ) / σ = (4 - 3 - 0) / 1 = 1
            EI = (μ - y_best - ξ) · Φ(Z) + σ · φ(Z)
               = (4 - 3 - 0) · Φ(1) + 1 · φ(1)
               = 1 · 0.8413... + 1 · 0.2420...
               ≈ 1.0833
        """
        ei = ExpectedImprovement(xi=0.0)
        X = np.array([[0.0]])
        mean = np.array([4.0])
        std = np.array([1.0])
        y_best = 3.0

        Z = 1.0
        expected = 1.0 * norm.cdf(Z) + 1.0 * norm.pdf(Z)

        scores = ei.evaluate(X, mean, std, y_best=y_best, objective="maximize")

        assert scores[0] == pytest.approx(expected, rel=1e-6)

    def test_ei_maximize_known_value_2(self):
        """
        EI maximize: y_best=0, μ=0.5, σ=0.5, ξ=0.1

        Formula (maximize):
            Z = (μ - y_best - ξ) / σ = (0.5 - 0 - 0.1) / 0.5 = 0.8
            EI = (μ - y_best - ξ) · Φ(Z) + σ · φ(Z)
               = 0.4 · Φ(0.8) + 0.5 · φ(0.8)
        """
        ei = ExpectedImprovement(xi=0.1)
        X = np.array([[0.0]])
        mean = np.array([0.5])
        std = np.array([0.5])
        y_best = 0.0

        Z = 0.8
        improvement = 0.4
        expected = improvement * norm.cdf(Z) + 0.5 * norm.pdf(Z)

        scores = ei.evaluate(X, mean, std, y_best=y_best, objective="maximize")

        assert scores[0] == pytest.approx(expected, rel=1e-6)

    def test_ei_maximize_known_value_3(self):
        """
        EI maximize: y_best=2, μ=1, σ=2, ξ=0 (mean worse than y_best)

        Formula (maximize):
            Z = (μ - y_best - ξ) / σ = (1 - 2 - 0) / 2 = -0.5
            EI = (μ - y_best - ξ) · Φ(Z) + σ · φ(Z)
               = -1 · Φ(-0.5) + 2 · φ(-0.5)
        """
        ei = ExpectedImprovement(xi=0.0)
        X = np.array([[0.0]])
        mean = np.array([1.0])
        std = np.array([2.0])
        y_best = 2.0

        Z = -0.5
        improvement = -1.0
        expected = improvement * norm.cdf(Z) + 2.0 * norm.pdf(Z)

        scores = ei.evaluate(X, mean, std, y_best=y_best, objective="maximize")

        assert scores[0] == pytest.approx(expected, rel=1e-6)

    def test_ei_minimize_known_value_1(self):
        """
        EI minimize: y_best=3, μ=2, σ=1, ξ=0

        Formula (minimize):
            Z = (y_best - μ - ξ) / σ = (3 - 2 - 0) / 1 = 1
            EI = (y_best - μ - ξ) · Φ(Z) + σ · φ(Z)
               = 1 · Φ(1) + 1 · φ(1)
               ≈ 1.0833
        """
        ei = ExpectedImprovement(xi=0.0)
        X = np.array([[0.0]])
        mean = np.array([2.0])
        std = np.array([1.0])
        y_best = 3.0

        Z = 1.0
        expected = 1.0 * norm.cdf(Z) + 1.0 * norm.pdf(Z)

        scores = ei.evaluate(X, mean, std, y_best=y_best, objective="minimize")

        assert scores[0] == pytest.approx(expected, rel=1e-6)

    def test_ei_minimize_known_value_2(self):
        """
        EI minimize: y_best=1, μ=0.5, σ=0.25, ξ=0

        Formula (minimize):
            Z = (y_best - μ - ξ) / σ = (1 - 0.5 - 0) / 0.25 = 2
            EI = (y_best - μ - ξ) · Φ(Z) + σ · φ(Z)
               = 0.5 · Φ(2) + 0.25 · φ(2)
        """
        ei = ExpectedImprovement(xi=0.0)
        X = np.array([[0.0]])
        mean = np.array([0.5])
        std = np.array([0.25])
        y_best = 1.0

        Z = 2.0
        improvement = 0.5
        expected = improvement * norm.cdf(Z) + 0.25 * norm.pdf(Z)

        scores = ei.evaluate(X, mean, std, y_best=y_best, objective="minimize")

        assert scores[0] == pytest.approx(expected, rel=1e-6)

    def test_ei_minimize_known_value_3(self):
        """
        EI minimize: y_best=0, μ=1, σ=0.5, ξ=0.1 (mean worse than y_best)

        Formula (minimize):
            Z = (y_best - μ - ξ) / σ = (0 - 1 - 0.1) / 0.5 = -2.2
            EI = (y_best - μ - ξ) · Φ(Z) + σ · φ(Z)
               = -1.1 · Φ(-2.2) + 0.5 · φ(-2.2)
        """
        ei = ExpectedImprovement(xi=0.1)
        X = np.array([[0.0]])
        mean = np.array([1.0])
        std = np.array([0.5])
        y_best = 0.0

        Z = -2.2
        improvement = -1.1
        expected = improvement * norm.cdf(Z) + 0.5 * norm.pdf(Z)

        scores = ei.evaluate(X, mean, std, y_best=y_best, objective="minimize")

        assert scores[0] == pytest.approx(expected, rel=1e-6)


# =============================================================================
# Upper Confidence Bound Tests
# =============================================================================


class TestUpperConfidenceBoundContract:
    """Test UCB satisfies basic contract requirements."""

    def test_output_shape_matches_input(self):
        """UCB output shape matches number of candidates."""
        ucb = UpperConfidenceBound(beta=2.0)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mean = np.array([0.5, 0.8, 0.3])
        std = np.array([0.1, 0.2, 0.15])

        scores = ucb.evaluate(X, mean, std, y_best=0.4, objective="maximize")

        assert scores.shape == (3,)

    def test_output_is_finite(self):
        """UCB scores contain no NaN or Inf values."""
        ucb = UpperConfidenceBound(beta=2.0)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        mean = np.array([0.5, 0.8, 0.3, 1.0])
        std = np.array([0.1, 0.2, 0.0, 0.5])

        scores = ucb.evaluate(X, mean, std, y_best=0.6, objective="maximize")

        assert np.all(np.isfinite(scores))

    def test_single_point_evaluation(self):
        """UCB works with a single candidate point."""
        ucb = UpperConfidenceBound(beta=2.0)
        X = np.array([[1.0, 2.0]])
        mean = np.array([0.5])
        std = np.array([0.1])

        scores = ucb.evaluate(X, mean, std, y_best=0.4, objective="maximize")

        assert scores.shape == (1,)
        assert np.isfinite(scores[0])

    def test_batch_evaluation(self):
        """UCB works with many candidate points."""
        ucb = UpperConfidenceBound(beta=2.0)
        n = 100
        X = np.random.randn(n, 5)
        mean = np.random.randn(n)
        std = np.abs(np.random.randn(n))

        scores = ucb.evaluate(X, mean, std, y_best=0.0, objective="maximize")

        assert scores.shape == (n,)
        assert np.all(np.isfinite(scores))

    def test_zero_std_returns_mean_based_score(self):
        """UCB with std=0 returns score based on mean only."""
        ucb = UpperConfidenceBound(beta=2.0)
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.0, 0.0])

        scores_max = ucb.evaluate(X, mean, std, y_best=0.0, objective="maximize")
        scores_min = ucb.evaluate(X, mean, std, y_best=0.0, objective="minimize")

        # maximize: μ + β·σ = μ + 0 = μ
        assert scores_max[0] == pytest.approx(0.5)
        assert scores_max[1] == pytest.approx(0.8)
        # minimize: -μ + β·σ = -μ + 0 = -μ
        assert scores_min[0] == pytest.approx(-0.5)
        assert scores_min[1] == pytest.approx(-0.8)

    def test_negative_beta_raises(self):
        """UCB constructor raises ValueError for negative beta."""
        with pytest.raises(ValueError, match="(?i)beta|negative|non-negative|>= 0"):
            UpperConfidenceBound(beta=-0.5)


class TestUpperConfidenceBoundDirectional:
    """Test UCB directional behavior."""

    def test_higher_std_gives_higher_score(self):
        """Points with higher uncertainty get higher UCB (all else equal)."""
        ucb = UpperConfidenceBound(beta=2.0)
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.5])
        std = np.array([0.1, 0.5])

        scores = ucb.evaluate(X, mean, std, y_best=0.0, objective="maximize")

        assert scores[1] > scores[0]

    def test_score_increases_with_beta(self):
        """Higher beta gives higher scores (for points with std > 0)."""
        ucb_low = UpperConfidenceBound(beta=1.0)
        ucb_high = UpperConfidenceBound(beta=3.0)
        X = np.array([[1.0]])
        mean = np.array([0.5])
        std = np.array([0.2])

        score_low = ucb_low.evaluate(X, mean, std, y_best=0.0, objective="maximize")
        score_high = ucb_high.evaluate(X, mean, std, y_best=0.0, objective="maximize")

        assert score_high[0] > score_low[0]

    def test_higher_mean_gives_higher_score_maximize(self):
        """Points with higher mean get higher UCB when maximizing."""
        ucb = UpperConfidenceBound(beta=2.0)
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 1.0])
        std = np.array([0.2, 0.2])

        scores = ucb.evaluate(X, mean, std, y_best=0.0, objective="maximize")

        assert scores[1] > scores[0]

    def test_lower_mean_gives_higher_score_minimize(self):
        """Points with lower mean get higher UCB when minimizing."""
        ucb = UpperConfidenceBound(beta=2.0)
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 1.0])
        std = np.array([0.2, 0.2])

        scores = ucb.evaluate(X, mean, std, y_best=0.0, objective="minimize")

        assert scores[0] > scores[1]


class TestUpperConfidenceBoundNumerical:
    """Test UCB numerical correctness against hand-computed values."""

    def test_ucb_maximize_known_value_1(self):
        """
        UCB maximize: μ=2, σ=0.5, β=2

        Formula (maximize):
            UCB = μ + β · σ = 2 + 2 · 0.5 = 3
        """
        ucb = UpperConfidenceBound(beta=2.0)
        X = np.array([[0.0]])
        mean = np.array([2.0])
        std = np.array([0.5])

        expected = 2.0 + 2.0 * 0.5

        scores = ucb.evaluate(X, mean, std, y_best=0.0, objective="maximize")

        assert scores[0] == pytest.approx(expected, rel=1e-9)

    def test_ucb_maximize_known_value_2(self):
        """
        UCB maximize: μ=-1, σ=3, β=1.5

        Formula (maximize):
            UCB = μ + β · σ = -1 + 1.5 · 3 = 3.5
        """
        ucb = UpperConfidenceBound(beta=1.5)
        X = np.array([[0.0]])
        mean = np.array([-1.0])
        std = np.array([3.0])

        expected = -1.0 + 1.5 * 3.0

        scores = ucb.evaluate(X, mean, std, y_best=0.0, objective="maximize")

        assert scores[0] == pytest.approx(expected, rel=1e-9)

    def test_ucb_maximize_known_value_3(self):
        """
        UCB maximize: μ=0.75, σ=0.25, β=4

        Formula (maximize):
            UCB = μ + β · σ = 0.75 + 4 · 0.25 = 1.75
        """
        ucb = UpperConfidenceBound(beta=4.0)
        X = np.array([[0.0]])
        mean = np.array([0.75])
        std = np.array([0.25])

        expected = 0.75 + 4.0 * 0.25

        scores = ucb.evaluate(X, mean, std, y_best=0.0, objective="maximize")

        assert scores[0] == pytest.approx(expected, rel=1e-9)

    def test_ucb_minimize_known_value_1(self):
        """
        UCB minimize: μ=2, σ=0.5, β=2

        Formula (minimize):
            UCB = -μ + β · σ = -2 + 2 · 0.5 = -1
        """
        ucb = UpperConfidenceBound(beta=2.0)
        X = np.array([[0.0]])
        mean = np.array([2.0])
        std = np.array([0.5])

        expected = -2.0 + 2.0 * 0.5

        scores = ucb.evaluate(X, mean, std, y_best=0.0, objective="minimize")

        assert scores[0] == pytest.approx(expected, rel=1e-9)

    def test_ucb_minimize_known_value_2(self):
        """
        UCB minimize: μ=-1, σ=3, β=1.5

        Formula (minimize):
            UCB = -μ + β · σ = -(-1) + 1.5 · 3 = 1 + 4.5 = 5.5
        """
        ucb = UpperConfidenceBound(beta=1.5)
        X = np.array([[0.0]])
        mean = np.array([-1.0])
        std = np.array([3.0])

        expected = 1.0 + 1.5 * 3.0

        scores = ucb.evaluate(X, mean, std, y_best=0.0, objective="minimize")

        assert scores[0] == pytest.approx(expected, rel=1e-9)

    def test_ucb_minimize_known_value_3(self):
        """
        UCB minimize: μ=0.75, σ=0.25, β=4

        Formula (minimize):
            UCB = -μ + β · σ = -0.75 + 4 · 0.25 = 0.25
        """
        ucb = UpperConfidenceBound(beta=4.0)
        X = np.array([[0.0]])
        mean = np.array([0.75])
        std = np.array([0.25])

        expected = -0.75 + 4.0 * 0.25

        scores = ucb.evaluate(X, mean, std, y_best=0.0, objective="minimize")

        assert scores[0] == pytest.approx(expected, rel=1e-9)

    def test_ucb_batch_known_values(self):
        """
        UCB batch: multiple points computed correctly.

        μ = [1, 2, 3], σ = [0.5, 1.0, 0.25], β = 2

        maximize: UCB = μ + β · σ = [1+1, 2+2, 3+0.5] = [2, 4, 3.5]
        minimize: UCB = -μ + β · σ = [-1+1, -2+2, -3+0.5] = [0, 0, -2.5]
        """
        ucb = UpperConfidenceBound(beta=2.0)
        X = np.array([[0.0], [1.0], [2.0]])
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 1.0, 0.25])

        scores_max = ucb.evaluate(X, mean, std, y_best=0.0, objective="maximize")
        scores_min = ucb.evaluate(X, mean, std, y_best=0.0, objective="minimize")

        assert scores_max[0] == pytest.approx(2.0, rel=1e-9)
        assert scores_max[1] == pytest.approx(4.0, rel=1e-9)
        assert scores_max[2] == pytest.approx(3.5, rel=1e-9)

        assert scores_min[0] == pytest.approx(0.0, abs=1e-9)
        assert scores_min[1] == pytest.approx(0.0, abs=1e-9)
        assert scores_min[2] == pytest.approx(-2.5, rel=1e-9)
