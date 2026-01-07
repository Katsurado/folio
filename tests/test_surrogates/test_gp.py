"""Tests for GPSurrogate Gaussian Process model."""

import numpy as np
import pytest

from folio.exceptions import NotFittedError
from folio.surrogates import GPSurrogate


@pytest.fixture
def sine_data():
    """Generate 5-point sine wave training data."""
    X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    y = np.sin(2 * np.pi * X).ravel()
    return X, y


@pytest.fixture
def multidim_data():
    """Generate 2D training data (5 points)."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    y = X[:, 0] + X[:, 1]
    return X, y


class TestGPSurrogateInit:
    """Test GPSurrogate constructor validation."""

    def test_default_parameters(self):
        """GPSurrogate initializes with sensible defaults."""
        gp = GPSurrogate()
        assert gp.noise == 1e-4
        assert gp.kernel == "matern"
        assert gp.nu == 2.5
        assert gp.ard is True
        assert gp.normalize_inputs is True
        assert gp.normalize_outputs is True

    def test_custom_noise(self):
        """Custom noise value is stored."""
        gp = GPSurrogate(noise=0.1)
        assert gp.noise == 0.1

    def test_negative_noise_raises(self):
        """Negative noise raises ValueError."""
        with pytest.raises(ValueError, match="noise"):
            GPSurrogate(noise=-0.1)

    def test_zero_noise_allowed(self):
        """Zero noise is allowed (though not recommended)."""
        gp = GPSurrogate(noise=0.0)
        assert gp.noise == 0.0

    def test_rbf_kernel(self):
        """RBF kernel can be specified."""
        gp = GPSurrogate(kernel="rbf")
        assert gp.kernel == "rbf"

    def test_matern_kernel(self):
        """Matérn kernel is default."""
        gp = GPSurrogate(kernel="matern")
        assert gp.kernel == "matern"

    def test_invalid_kernel_raises(self):
        """Invalid kernel name raises ValueError."""
        with pytest.raises(ValueError, match="kernel"):
            GPSurrogate(kernel="invalid")

    def test_nu_values(self):
        """Valid nu values (0.5, 1.5, 2.5) are accepted."""
        for nu in [0.5, 1.5, 2.5]:
            gp = GPSurrogate(nu=nu)
            assert gp.nu == nu

    def test_invalid_nu_raises(self):
        """Invalid nu value raises ValueError."""
        with pytest.raises(ValueError, match="nu"):
            GPSurrogate(nu=1.0)

    def test_ard_disabled(self):
        """ARD can be disabled."""
        gp = GPSurrogate(ard=False)
        assert gp.ard is False

    def test_normalization_disabled(self):
        """Normalization can be disabled."""
        gp = GPSurrogate(normalize_inputs=False, normalize_outputs=False)
        assert gp.normalize_inputs is False
        assert gp.normalize_outputs is False


class TestGPSurrogateFit:
    """Test GPSurrogate fit method."""

    def test_fit_returns_self(self, sine_data):
        """fit() returns self for method chaining."""
        X, y = sine_data
        gp = GPSurrogate()
        result = gp.fit(X, y)
        assert result is gp

    def test_fit_sets_n_features(self, sine_data):
        """fit() stores number of features."""
        X, y = sine_data
        gp = GPSurrogate()
        gp.fit(X, y)
        assert gp.n_features == 1

    def test_fit_multidim(self, multidim_data):
        """fit() works with multi-dimensional inputs."""
        X, y = multidim_data
        gp = GPSurrogate()
        gp.fit(X, y)
        assert gp.n_features == 2

    def test_fit_shape_mismatch_raises(self, sine_data):
        """fit() raises ValueError if X and y have different n_samples."""
        X, y = sine_data
        with pytest.raises(ValueError, match="samples"):
            GPSurrogate().fit(X, y[:-1])

    def test_fit_single_sample(self):
        """fit() works with a single training point."""
        X = np.array([[0.5]])
        y = np.array([1.0])
        gp = GPSurrogate()
        gp.fit(X, y)
        assert gp.n_features == 1

    def test_fit_method_chaining(self, sine_data):
        """fit() and predict() can be chained."""
        X, y = sine_data
        X_test = np.array([[0.125]])
        mean, std = GPSurrogate().fit(X, y).predict(X_test)
        assert mean.shape == (1,)

    def test_fit_with_rbf_kernel(self, sine_data):
        """fit() works with RBF kernel."""
        X, y = sine_data
        gp = GPSurrogate(kernel="rbf")
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (5,)

    def test_fit_with_matern_05(self, sine_data):
        """fit() works with Matérn nu=0.5."""
        X, y = sine_data
        gp = GPSurrogate(kernel="matern", nu=0.5)
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (5,)

    def test_fit_with_matern_15(self, sine_data):
        """fit() works with Matérn nu=1.5."""
        X, y = sine_data
        gp = GPSurrogate(kernel="matern", nu=1.5)
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (5,)

    def test_fit_without_ard(self, multidim_data):
        """fit() works without ARD."""
        X, y = multidim_data
        gp = GPSurrogate(ard=False)
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (5,)

    def test_fit_without_normalization(self, sine_data):
        """fit() works without normalization."""
        X, y = sine_data
        gp = GPSurrogate(normalize_inputs=False, normalize_outputs=False)
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (5,)

    def test_refit_updates_model(self, sine_data):
        """Calling fit() again updates the model."""
        X, y = sine_data
        gp = GPSurrogate()
        gp.fit(X, y)
        mean1, _ = gp.predict(np.array([[0.125]]))

        X2 = np.array([[0.0], [1.0]])
        y2 = np.array([0.0, 0.0])
        gp.fit(X2, y2)
        mean2, _ = gp.predict(np.array([[0.125]]))

        assert gp.n_features == 1
        # Predictions should differ after refit
        assert not np.allclose(mean1, mean2)


class TestGPSurrogatePredict:
    """Test GPSurrogate predict method."""

    def test_predict_returns_tuple(self, sine_data):
        """predict() returns (mean, std) tuple."""
        X, y = sine_data
        gp = GPSurrogate().fit(X, y)
        result = gp.predict(np.array([[0.125]]))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_output_shapes(self, sine_data):
        """predict() returns arrays with shape (n_candidates,)."""
        X, y = sine_data
        gp = GPSurrogate().fit(X, y)
        X_test = np.array([[0.1], [0.2], [0.3], [0.4]])
        mean, std = gp.predict(X_test)
        assert mean.shape == (4,)
        assert std.shape == (4,)

    def test_predict_single_point(self, sine_data):
        """predict() works with a single candidate."""
        X, y = sine_data
        gp = GPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.5]]))
        assert mean.shape == (1,)
        assert std.shape == (1,)

    def test_predict_std_nonnegative(self, sine_data):
        """Standard deviation is always non-negative."""
        X, y = sine_data
        gp = GPSurrogate().fit(X, y)
        X_test = np.linspace(0, 1, 20).reshape(-1, 1)
        _, std = gp.predict(X_test)
        assert np.all(std >= 0)

    def test_predict_returns_numpy_arrays(self, sine_data):
        """predict() returns numpy arrays, not torch tensors."""
        X, y = sine_data
        gp = GPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.5]]))
        assert isinstance(mean, np.ndarray)
        assert isinstance(std, np.ndarray)

    def test_predict_multidim(self, multidim_data):
        """predict() works with multi-dimensional inputs."""
        X, y = multidim_data
        gp = GPSurrogate().fit(X, y)
        X_test = np.array([[0.25, 0.25], [0.75, 0.75]])
        mean, std = gp.predict(X_test)
        assert mean.shape == (2,)
        assert std.shape == (2,)


class TestGPSurrogateNotFitted:
    """Test NotFittedError handling."""

    def test_predict_before_fit_raises(self):
        """predict() raises NotFittedError if fit() not called."""
        gp = GPSurrogate()
        X_test = np.array([[0.5]])
        with pytest.raises(NotFittedError):
            gp.predict(X_test)

    def test_error_message_mentions_fit(self):
        """NotFittedError message mentions fit()."""
        gp = GPSurrogate()
        X_test = np.array([[0.5]])
        with pytest.raises(NotFittedError, match="fit"):
            gp.predict(X_test)

    def test_model_is_none_before_fit(self):
        """model attribute is None before fit()."""
        gp = GPSurrogate()
        assert gp.model is None


class TestGPSurrogateFeatureValidation:
    """Test feature dimension validation."""

    def test_predict_wrong_n_features_raises(self, sine_data):
        """predict() raises ValueError if n_features doesn't match."""
        X, y = sine_data
        gp = GPSurrogate().fit(X, y)
        X_wrong = np.array([[0.5, 0.5]])  # 2 features instead of 1
        with pytest.raises(ValueError, match="features"):
            gp.predict(X_wrong)

    def test_predict_fewer_features_raises(self, multidim_data):
        """predict() raises ValueError with fewer features than training."""
        X, y = multidim_data
        gp = GPSurrogate().fit(X, y)
        X_wrong = np.array([[0.5]])  # 1 feature instead of 2
        with pytest.raises(ValueError, match="features"):
            gp.predict(X_wrong)


class TestGPSurrogateBehavior:
    """Test GP prediction behavior and properties."""

    def test_low_uncertainty_at_training_points(self, sine_data):
        """Uncertainty should be low at training points."""
        X, y = sine_data
        gp = GPSurrogate(noise=1e-6).fit(X, y)
        _, std = gp.predict(X)
        # Std at training points should be very small
        assert np.all(std < 0.1)

    def test_higher_uncertainty_away_from_data(self, sine_data):
        """Uncertainty should be higher far from training points."""
        X, y = sine_data
        gp = GPSurrogate().fit(X, y)
        _, std_train = gp.predict(X)
        _, std_far = gp.predict(np.array([[-1.0], [2.0]]))
        # Std should be higher away from training data
        assert np.mean(std_far) > np.mean(std_train)

    def test_predictions_close_to_training_values(self, sine_data):
        """Predictions at training points should match training values."""
        X, y = sine_data
        gp = GPSurrogate(noise=1e-6).fit(X, y)
        mean, _ = gp.predict(X)
        # Mean predictions should be close to training targets
        assert np.allclose(mean, y, atol=0.1)

    def test_interpolation_is_smooth(self, sine_data):
        """GP should provide smooth interpolation."""
        X, y = sine_data
        gp = GPSurrogate().fit(X, y)
        X_dense = np.linspace(0, 1, 100).reshape(-1, 1)
        mean, _ = gp.predict(X_dense)
        # Check predictions are within reasonable range
        assert np.all(mean >= -1.5)
        assert np.all(mean <= 1.5)

    def test_noise_affects_uncertainty(self, sine_data):
        """Higher noise should increase uncertainty at training points."""
        X, y = sine_data
        gp_low_noise = GPSurrogate(noise=1e-6).fit(X, y)
        gp_high_noise = GPSurrogate(noise=0.1).fit(X, y)

        _, std_low = gp_low_noise.predict(X)
        _, std_high = gp_high_noise.predict(X)

        # Higher noise should give higher uncertainty
        assert np.mean(std_high) > np.mean(std_low)


class TestGPSurrogateNormalization:
    """Test input/output normalization behavior."""

    def test_normalization_handles_different_scales(self):
        """Normalization should handle inputs with different scales."""
        X = np.array([[1.0, 1000.0], [2.0, 2000.0], [3.0, 3000.0]])
        y = X[:, 0] + X[:, 1] / 1000
        gp = GPSurrogate(normalize_inputs=True).fit(X, y)
        X_test = np.array([[1.5, 1500.0], [2.5, 2500.0]])
        mean, std = gp.predict(X_test)
        assert mean.shape == (2,)
        assert np.all(std >= 0)

    def test_output_normalization_preserves_scale(self):
        """Output normalization should preserve original scale in predictions."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([100.0, 150.0, 200.0])  # Large scale
        gp = GPSurrogate(normalize_outputs=True).fit(X, y)
        mean, _ = gp.predict(X)
        # Predictions should be on original scale
        assert np.all(mean > 50)
        assert np.all(mean < 250)

    def test_no_normalization_works(self):
        """GP works without any normalization."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.5, 1.0])
        gp = GPSurrogate(normalize_inputs=False, normalize_outputs=False).fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (3,)


class TestGPSurrogateEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_constant_target_values(self):
        """GP handles constant target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([1.0, 1.0, 1.0])
        gp = GPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.25]]))
        assert mean.shape == (1,)
        # Mean should be close to 1.0
        assert np.abs(mean[0] - 1.0) < 0.5

    def test_many_features(self):
        """GP handles many input features."""
        n_features = 10
        X = np.random.rand(20, n_features)
        y = np.sum(X, axis=1)
        gp = GPSurrogate().fit(X, y)
        X_test = np.random.rand(5, n_features)
        mean, std = gp.predict(X_test)
        assert mean.shape == (5,)
        assert gp.n_features == n_features

    def test_many_training_points(self):
        """GP handles many training points."""
        X = np.linspace(0, 1, 50).reshape(-1, 1)
        y = np.sin(2 * np.pi * X).ravel()
        gp = GPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.25], [0.75]]))
        assert mean.shape == (2,)

    def test_negative_target_values(self):
        """GP handles negative target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([-10.0, -5.0, 0.0])
        gp = GPSurrogate().fit(X, y)
        mean, _ = gp.predict(X)
        assert np.all(mean < 5)

    def test_large_target_values(self):
        """GP handles large target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([1e6, 2e6, 3e6])
        gp = GPSurrogate().fit(X, y)
        mean, _ = gp.predict(X)
        assert np.all(mean > 0)
