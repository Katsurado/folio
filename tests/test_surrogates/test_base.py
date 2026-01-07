"""Tests for surrogate model interface."""

import numpy as np
import pytest

from folio.exceptions import NotFittedError
from folio.surrogates import Surrogate


class ConcreteSurrogate(Surrogate):
    """Minimal concrete implementation for testing the interface contract.

    This mock surrogate stores training data and returns simple predictions
    based on the training mean and a fixed uncertainty.
    """

    def __init__(self):
        self._is_fitted = False
        self._X_train = None
        self._y_train = None
        self._n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConcreteSurrogate":
        """Store training data and mark as fitted."""
        if X.shape[0] != y.shape[0]:
            raise ValueError("samples")

        self._X_train = X
        self._y_train = y
        self._is_fitted = True
        self._n_features = X.shape[1]

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return constant mean (training mean) and fixed std."""
        if not self._is_fitted:
            raise NotFittedError("fit")
        if X.shape[1] != self._n_features:
            raise ValueError("features")

        n = X.shape[0]
        return np.zeros(n), np.ones(n)


class TestSurrogateABC:
    """Test that Surrogate enforces the abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Surrogate ABC cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Surrogate()

    def test_must_implement_fit(self):
        """Subclass without fit raises TypeError."""

        class NoFit(Surrogate):
            def predict(self, X):
                return np.zeros(len(X)), np.ones(len(X))

        with pytest.raises(TypeError, match="abstract"):
            NoFit()

    def test_must_implement_predict(self):
        """Subclass without predict raises TypeError."""

        class NoPredict(Surrogate):
            def fit(self, X, y):
                return self

        with pytest.raises(TypeError, match="abstract"):
            NoPredict()


class TestFitMethod:
    """Test the fit method contract."""

    def test_fit_returns_self(self):
        """fit() must return self for method chaining."""
        surrogate = ConcreteSurrogate()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])

        result = surrogate.fit(X, y)

        assert result is surrogate

    def test_fit_enables_predict(self):
        """After fit(), predict() should work without error."""
        surrogate = ConcreteSurrogate()
        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([1.0, 2.0, 3.0])
        X_test = np.array([[1.5], [2.5]])

        surrogate.fit(X_train, y_train)
        mean, std = surrogate.predict(X_test)

        assert isinstance(mean, np.ndarray)
        assert isinstance(std, np.ndarray)

    def test_fit_method_chaining(self):
        """fit() and predict() can be chained in one expression."""
        surrogate = ConcreteSurrogate()
        X_train = np.array([[1.0], [2.0]])
        y_train = np.array([1.0, 2.0])
        X_test = np.array([[1.5]])

        mean, std = surrogate.fit(X_train, y_train).predict(X_test)

        assert mean.shape == (1,)
        assert std.shape == (1,)

    def test_fit_validates_shape_mismatch(self):
        """fit() should raise ValueError if X and y have different n_samples."""
        surrogate = ConcreteSurrogate()
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="samples"):
            surrogate.fit(X, y)


class TestPredictMethod:
    """Test the predict method contract."""

    def test_predict_returns_tuple_of_arrays(self):
        """predict() must return (mean, std) tuple of numpy arrays."""
        surrogate = ConcreteSurrogate()
        X_train = np.array([[1.0], [2.0]])
        y_train = np.array([1.0, 2.0])
        surrogate.fit(X_train, y_train)

        result = surrogate.predict(np.array([[1.5]]))

        assert isinstance(result, tuple)
        assert len(result) == 2
        mean, std = result
        assert isinstance(mean, np.ndarray)
        assert isinstance(std, np.ndarray)

    def test_predict_output_shapes(self):
        """mean and std must have shape (n_candidates,)."""
        surrogate = ConcreteSurrogate()
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_train = np.array([1.0, 2.0, 3.0])
        X_test = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]])
        surrogate.fit(X_train, y_train)

        mean, std = surrogate.predict(X_test)

        assert mean.shape == (4,)
        assert std.shape == (4,)

    def test_predict_std_nonnegative(self):
        """Standard deviation values must be non-negative."""
        surrogate = ConcreteSurrogate()
        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([1.0, 2.0, 3.0])
        X_test = np.array([[0.5], [1.5], [2.5], [3.5]])
        surrogate.fit(X_train, y_train)

        _, std = surrogate.predict(X_test)

        assert np.all(std >= 0)

    def test_predict_single_point(self):
        """predict() works with a single candidate point."""
        surrogate = ConcreteSurrogate()
        X_train = np.array([[1.0], [2.0]])
        y_train = np.array([1.0, 2.0])
        X_test = np.array([[1.5]])
        surrogate.fit(X_train, y_train)

        mean, std = surrogate.predict(X_test)

        assert mean.shape == (1,)
        assert std.shape == (1,)


class TestNotFittedError:
    """Test that predict raises NotFittedError before fit."""

    def test_predict_before_fit_raises(self):
        """predict() must raise NotFittedError if fit() not called."""
        surrogate = ConcreteSurrogate()
        X_test = np.array([[1.0], [2.0]])

        with pytest.raises(NotFittedError):
            surrogate.predict(X_test)

    def test_error_message_is_helpful(self):
        """NotFittedError message should mention fit()."""
        surrogate = ConcreteSurrogate()
        X_test = np.array([[1.0]])

        with pytest.raises(NotFittedError, match="fit"):
            surrogate.predict(X_test)

    def test_not_fitted_error_is_surrogate_error(self):
        """NotFittedError should be a subclass of SurrogateError."""
        from folio.exceptions import SurrogateError

        assert issubclass(NotFittedError, SurrogateError)


class TestFeatureValidation:
    """Test that predict validates feature dimensions."""

    def test_predict_wrong_n_features_raises(self):
        """predict() should raise ValueError if n_features doesn't match fit."""
        surrogate = ConcreteSurrogate()
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_train = np.array([1.0, 2.0])
        X_test = np.array([[1.0, 2.0, 3.0]])
        surrogate.fit(X_train, y_train)

        with pytest.raises(ValueError, match="features"):
            surrogate.predict(X_test)
