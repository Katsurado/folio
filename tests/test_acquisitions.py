"""Tests for acquisition function interface."""

from typing import Literal

import numpy as np
import pytest

from folio.recommenders.acquisitions.base import Acquisition


class ConcreteAcquisition(Acquisition):
    """Minimal concrete implementation for testing the interface contract.

    This mock acquisition returns the mean values as scores, allowing tests
    to verify that _compute is called correctly after validation.
    """

    def __init__(self):
        self.compute_called = False
        self.last_args = None

    def _compute(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: float,
        objective: Literal["maximize", "minimize"],
    ) -> np.ndarray:
        """Return mean as scores and record that _compute was called."""
        self.compute_called = True
        self.last_args = {
            "X": X,
            "mean": mean,
            "std": std,
            "y_best": y_best,
            "objective": objective,
        }
        return mean.copy()


class TestAcquisitionABC:
    """Test that Acquisition enforces the abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Acquisition ABC cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Acquisition()

    def test_must_implement_compute(self):
        """Subclass without _compute raises TypeError."""

        class NoCompute(Acquisition):
            def evaluate(self, X, mean, std, y_best, objective):
                return mean

        with pytest.raises(TypeError, match="abstract"):
            NoCompute()


class TestEvaluateValidation:
    """Test that evaluate validates inputs before delegating to _compute."""

    def test_shape_mismatch_raises(self):
        """evaluate() raises ValueError if mean and std shapes don't match."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.1, 0.2, 0.3])

        with pytest.raises(ValueError, match="shape"):
            acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")

    def test_negative_std_raises(self):
        """evaluate() raises ValueError if std contains negative values."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.1, -0.2])

        with pytest.raises(ValueError, match="negative|non-negative"):
            acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")

    def test_nan_in_mean_raises(self):
        """evaluate() raises ValueError if mean contains NaN."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, np.nan])
        std = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="NaN"):
            acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")

    def test_nan_in_std_raises(self):
        """evaluate() raises ValueError if std contains NaN."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([np.nan, 0.2])

        with pytest.raises(ValueError, match="NaN"):
            acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")

    def test_inf_in_mean_raises(self):
        """evaluate() raises ValueError if mean contains Inf."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, np.inf])
        std = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="Inf|infinite"):
            acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")

    def test_negative_inf_in_mean_raises(self):
        """evaluate() raises ValueError if mean contains -Inf."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([-np.inf, 0.8])
        std = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="Inf|infinite"):
            acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")

    def test_inf_in_std_raises(self):
        """evaluate() raises ValueError if std contains Inf."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.1, np.inf])

        with pytest.raises(ValueError, match="Inf|infinite"):
            acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")

    def test_invalid_objective_raises(self):
        """evaluate() raises ValueError if objective is not 'maximize' or 'minimize'."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="objective"):
            acq.evaluate(X, mean, std, y_best=0.5, objective="invalid")


class TestEvaluateDelegation:
    """Test that evaluate correctly delegates to _compute after validation."""

    def test_delegates_to_compute_on_valid_input(self):
        """evaluate() calls _compute when inputs are valid."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.1, 0.2])

        acq.evaluate(X, mean, std, y_best=0.6, objective="maximize")

        assert acq.compute_called

    def test_passes_all_arguments_to_compute(self):
        """evaluate() passes all arguments to _compute."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.1, 0.2])
        y_best = 0.6

        acq.evaluate(X, mean, std, y_best=y_best, objective="minimize")

        assert np.array_equal(acq.last_args["X"], X)
        assert np.array_equal(acq.last_args["mean"], mean)
        assert np.array_equal(acq.last_args["std"], std)
        assert acq.last_args["y_best"] == y_best
        assert acq.last_args["objective"] == "minimize"

    def test_returns_compute_result(self):
        """evaluate() returns the result from _compute."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0], [3.0]])
        mean = np.array([0.5, 0.8, 0.3])
        std = np.array([0.1, 0.2, 0.15])

        result = acq.evaluate(X, mean, std, y_best=0.4, objective="maximize")

        assert np.array_equal(result, mean)

    def test_does_not_call_compute_on_invalid_input(self):
        """evaluate() does not call _compute if validation fails."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, np.nan])
        std = np.array([0.1, 0.2])

        with pytest.raises(ValueError):
            acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")

        assert not acq.compute_called


class TestEvaluateOutputShape:
    """Test that evaluate returns correctly shaped output."""

    def test_output_shape_matches_input(self):
        """evaluate() returns array with shape (n_candidates,)."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        mean = np.array([0.5, 0.8, 0.3, 0.9])
        std = np.array([0.1, 0.2, 0.15, 0.25])

        result = acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")

        assert result.shape == (4,)

    def test_single_candidate(self):
        """evaluate() works with a single candidate point."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0, 2.0]])
        mean = np.array([0.5])
        std = np.array([0.1])

        result = acq.evaluate(X, mean, std, y_best=0.4, objective="minimize")

        assert result.shape == (1,)


class TestValidObjectives:
    """Test that both valid objectives work correctly."""

    def test_maximize_objective_accepted(self):
        """evaluate() accepts 'maximize' as objective."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.1, 0.2])

        acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")

        assert acq.last_args["objective"] == "maximize"

    def test_minimize_objective_accepted(self):
        """evaluate() accepts 'minimize' as objective."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.1, 0.2])

        acq.evaluate(X, mean, std, y_best=0.5, objective="minimize")

        assert acq.last_args["objective"] == "minimize"


class TestEdgeCases:
    """Test edge cases for acquisition evaluation."""

    def test_zero_std_accepted(self):
        """evaluate() accepts zero standard deviation (valid, no uncertainty)."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.0, 0.0])

        acq.evaluate(X, mean, std, y_best=0.5, objective="maximize")

        assert acq.compute_called

    def test_large_values_accepted(self):
        """evaluate() accepts large (but finite) values."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([1e10, -1e10])
        std = np.array([1e8, 1e8])

        acq.evaluate(X, mean, std, y_best=1e9, objective="maximize")

        assert acq.compute_called

    def test_negative_y_best_accepted(self):
        """evaluate() accepts negative y_best values."""
        acq = ConcreteAcquisition()
        X = np.array([[1.0], [2.0]])
        mean = np.array([0.5, 0.8])
        std = np.array([0.1, 0.2])

        acq.evaluate(X, mean, std, y_best=-100.0, objective="maximize")

        assert acq.last_args["y_best"] == -100.0
