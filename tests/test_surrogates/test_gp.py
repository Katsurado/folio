"""Tests for SingleTaskGPSurrogate Gaussian Process model."""

import numpy as np
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from folio.exceptions import NotFittedError
from folio.surrogates import SingleTaskGPSurrogate


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


class TestSingleTaskGPSurrogateInit:
    """Test SingleTaskGPSurrogate constructor validation."""

    def test_default_parameters(self):
        """SingleTaskGPSurrogate initializes with sensible defaults."""
        gp = SingleTaskGPSurrogate()
        assert gp.kernel == "matern"
        assert gp.nu == 2.5
        assert gp.ard is True
        assert gp.normalize_inputs is True
        assert gp.normalize_outputs is True

    def test_rbf_kernel(self):
        """RBF kernel can be specified."""
        gp = SingleTaskGPSurrogate(kernel="rbf")
        assert gp.kernel == "rbf"

    def test_matern_kernel(self):
        """Matérn kernel is default."""
        gp = SingleTaskGPSurrogate(kernel="matern")
        assert gp.kernel == "matern"

    def test_invalid_kernel_raises(self):
        """Invalid kernel name raises ValueError."""
        with pytest.raises(ValueError, match="kernel"):
            SingleTaskGPSurrogate(kernel="invalid")

    def test_nu_values(self):
        """Valid nu values (0.5, 1.5, 2.5) are accepted."""
        for nu in [0.5, 1.5, 2.5]:
            gp = SingleTaskGPSurrogate(nu=nu)
            assert gp.nu == nu

    def test_invalid_nu_raises(self):
        """Invalid nu value raises ValueError."""
        with pytest.raises(ValueError, match="nu"):
            SingleTaskGPSurrogate(nu=1.0)

    def test_ard_disabled(self):
        """ARD can be disabled."""
        gp = SingleTaskGPSurrogate(ard=False)
        assert gp.ard is False

    def test_normalization_disabled(self):
        """Normalization can be disabled."""
        gp = SingleTaskGPSurrogate(normalize_inputs=False, normalize_outputs=False)
        assert gp.normalize_inputs is False
        assert gp.normalize_outputs is False


class TestSingleTaskGPSurrogateFit:
    """Test SingleTaskGPSurrogate fit method."""

    def test_fit_returns_self(self, sine_data):
        """fit() returns self for method chaining."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate()
        result = gp.fit(X, y)
        assert result is gp

    def test_fit_sets_n_features(self, sine_data):
        """fit() stores number of features."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate()
        gp.fit(X, y)
        assert gp.n_features == 1

    def test_fit_multidim(self, multidim_data):
        """fit() works with multi-dimensional inputs."""
        X, y = multidim_data
        gp = SingleTaskGPSurrogate()
        gp.fit(X, y)
        assert gp.n_features == 2

    def test_fit_shape_mismatch_raises(self, sine_data):
        """fit() raises ValueError if X and y have different n_samples."""
        X, y = sine_data
        with pytest.raises(ValueError, match="samples"):
            SingleTaskGPSurrogate().fit(X, y[:-1])

    def test_fit_y_2d_raises(self):
        """fit() raises ValueError if y is 2D."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([[0.0], [0.5], [1.0]])
        with pytest.raises(ValueError, match="(?i)1d|1-d|dimension|ndim"):
            SingleTaskGPSurrogate().fit(X, y)

    def test_fit_single_sample(self):
        """fit() works with a single training point."""
        X = np.array([[0.5]])
        y = np.array([1.0])
        gp = SingleTaskGPSurrogate()
        gp.fit(X, y)
        assert gp.n_features == 1

    def test_fit_method_chaining(self, sine_data):
        """fit() and predict() can be chained."""
        X, y = sine_data
        X_test = np.array([[0.125]])
        mean, std = SingleTaskGPSurrogate().fit(X, y).predict(X_test)
        assert mean.shape == (1,)

    def test_fit_with_rbf_kernel(self, sine_data):
        """fit() works with RBF kernel."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate(kernel="rbf")
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (5,)

    def test_fit_with_matern_05(self, sine_data):
        """fit() works with Matérn nu=0.5."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate(kernel="matern", nu=0.5)
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (5,)

    def test_fit_with_matern_15(self, sine_data):
        """fit() works with Matérn nu=1.5."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate(kernel="matern", nu=1.5)
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (5,)

    def test_fit_without_ard(self, multidim_data):
        """fit() works without ARD."""
        X, y = multidim_data
        gp = SingleTaskGPSurrogate(ard=False)
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (5,)

    def test_fit_without_normalization(self, sine_data):
        """fit() works without normalization."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate(normalize_inputs=False, normalize_outputs=False)
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (5,)

    def test_refit_updates_model(self):
        """Calling fit() again updates the model."""
        # Use more training points for robust fitting
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = np.sin(2 * np.pi * X).ravel()

        gp = SingleTaskGPSurrogate()
        gp.fit(X, y)
        # Predict at x=0.25 where sine = 1 (peak)
        mean1, _ = gp.predict(np.array([[0.25]]))

        X2 = np.array([[0.0], [0.5], [1.0]])
        y2 = np.array([0.0, 0.0, 0.0])
        gp.fit(X2, y2)
        mean2, _ = gp.predict(np.array([[0.25]]))

        assert gp.n_features == 1
        # Predictions should differ: mean1 ≈ 1.0, mean2 ≈ 0.0
        assert not np.allclose(mean1, mean2)


class TestSingleTaskGPSurrogateDtype:
    """Test SingleTaskGPSurrogate dtype validation."""

    def test_fit_float32_x_raises(self):
        """fit() raises ValueError for float32 X."""
        X = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
        y = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="float64"):
            SingleTaskGPSurrogate().fit(X, y)

    def test_fit_float32_y_raises(self):
        """fit() raises ValueError for float32 y."""
        X = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)
        y = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="float64"):
            SingleTaskGPSurrogate().fit(X, y)

    def test_predict_float32_x_raises(self):
        """predict() raises ValueError for float32 X."""
        X = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)
        y = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        gp = SingleTaskGPSurrogate().fit(X, y)
        X_test = np.array([[0.25]], dtype=np.float32)
        with pytest.raises(ValueError, match="float64"):
            gp.predict(X_test)

    def test_float64_passes(self):
        """fit() and predict() work with float64."""
        X = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)
        y = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        gp = SingleTaskGPSurrogate().fit(X, y)
        X_test = np.array([[0.25]], dtype=np.float64)
        mean, std = gp.predict(X_test)
        assert mean.shape == (1,)


class TestSingleTaskGPSurrogatePredict:
    """Test SingleTaskGPSurrogate predict method."""

    def test_predict_returns_tuple(self, sine_data):
        """predict() returns (mean, std) tuple."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        result = gp.predict(np.array([[0.125]]))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_output_shapes(self, sine_data):
        """predict() returns arrays with shape (n_candidates,)."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        X_test = np.array([[0.1], [0.2], [0.3], [0.4]])
        mean, std = gp.predict(X_test)
        assert mean.shape == (4,)
        assert std.shape == (4,)

    def test_predict_single_point(self, sine_data):
        """predict() works with a single candidate."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.5]]))
        assert mean.shape == (1,)
        assert std.shape == (1,)

    def test_predict_std_nonnegative(self, sine_data):
        """Standard deviation is always non-negative."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        X_test = np.linspace(0, 1, 20).reshape(-1, 1)
        _, std = gp.predict(X_test)
        assert np.all(std >= 0)

    def test_predict_returns_numpy_arrays(self, sine_data):
        """predict() returns numpy arrays, not torch tensors."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.5]]))
        assert isinstance(mean, np.ndarray)
        assert isinstance(std, np.ndarray)

    def test_predict_multidim(self, multidim_data):
        """predict() works with multi-dimensional inputs."""
        X, y = multidim_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        X_test = np.array([[0.25, 0.25], [0.75, 0.75]])
        mean, std = gp.predict(X_test)
        assert mean.shape == (2,)
        assert std.shape == (2,)


class TestSingleTaskGPSurrogateNotFitted:
    """Test NotFittedError handling."""

    def test_predict_before_fit_raises(self):
        """predict() raises NotFittedError if fit() not called."""
        gp = SingleTaskGPSurrogate()
        X_test = np.array([[0.5]])
        with pytest.raises(NotFittedError):
            gp.predict(X_test)

    def test_error_message_mentions_fit(self):
        """NotFittedError message mentions fit()."""
        gp = SingleTaskGPSurrogate()
        X_test = np.array([[0.5]])
        with pytest.raises(NotFittedError, match="fit"):
            gp.predict(X_test)

    def test_model_is_none_before_fit(self):
        """model attribute is None before fit()."""
        gp = SingleTaskGPSurrogate()
        assert gp.model is None


class TestSingleTaskGPSurrogateFeatureValidation:
    """Test feature dimension validation."""

    def test_predict_wrong_n_features_raises(self, sine_data):
        """predict() raises ValueError if n_features doesn't match."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        X_wrong = np.array([[0.5, 0.5]])  # 2 features instead of 1
        with pytest.raises(ValueError, match="features"):
            gp.predict(X_wrong)

    def test_predict_fewer_features_raises(self, multidim_data):
        """predict() raises ValueError with fewer features than training."""
        X, y = multidim_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        X_wrong = np.array([[0.5]])  # 1 feature instead of 2
        with pytest.raises(ValueError, match="features"):
            gp.predict(X_wrong)


class TestSingleTaskGPSurrogateBehavior:
    """Test GP prediction behavior and properties."""

    def test_low_uncertainty_at_training_points(self, sine_data):
        """Uncertainty should be low at training points."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        _, std = gp.predict(X)
        # Std at training points should be relatively small
        assert np.all(std < 0.5)

    def test_higher_uncertainty_away_from_data(self, sine_data):
        """Uncertainty should be higher far from training points."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        _, std_train = gp.predict(X)
        _, std_far = gp.predict(np.array([[-1.0], [2.0]]))
        # Std should be higher away from training data
        assert np.mean(std_far) > np.mean(std_train)

    def test_predictions_close_to_training_values(self, sine_data):
        """Predictions at training points should match training values."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        mean, _ = gp.predict(X)
        # Mean predictions should be close to training targets
        assert np.allclose(mean, y, atol=0.3)

    def test_interpolation_is_smooth(self, sine_data):
        """GP should provide smooth interpolation."""
        X, y = sine_data
        gp = SingleTaskGPSurrogate().fit(X, y)
        X_dense = np.linspace(0, 1, 100).reshape(-1, 1)
        mean, _ = gp.predict(X_dense)
        # Check predictions are within reasonable range
        assert np.all(mean >= -1.5)
        assert np.all(mean <= 1.5)


class TestSingleTaskGPSurrogateNormalization:
    """Test input/output normalization behavior."""

    def test_normalization_handles_different_scales(self):
        """Normalization should handle inputs with different scales."""
        X = np.array([[1.0, 1000.0], [2.0, 2000.0], [3.0, 3000.0]])
        y = X[:, 0] + X[:, 1] / 1000
        gp = SingleTaskGPSurrogate(normalize_inputs=True).fit(X, y)
        X_test = np.array([[1.5, 1500.0], [2.5, 2500.0]])
        mean, std = gp.predict(X_test)
        assert mean.shape == (2,)
        assert np.all(std >= 0)

    def test_output_normalization_preserves_scale(self):
        """Output normalization should preserve original scale in predictions."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([100.0, 150.0, 200.0])  # Large scale
        gp = SingleTaskGPSurrogate(normalize_outputs=True).fit(X, y)
        mean, _ = gp.predict(X)
        # Predictions should be on original scale
        assert np.all(mean > 50)
        assert np.all(mean < 250)

    def test_no_normalization_works(self):
        """GP works without any normalization."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.5, 1.0])
        gp = SingleTaskGPSurrogate(normalize_inputs=False, normalize_outputs=False).fit(
            X, y
        )
        mean, std = gp.predict(X)
        assert mean.shape == (3,)


class TestSingleTaskGPSurrogateEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_constant_target_values(self):
        """GP handles constant target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([1.0, 1.0, 1.0])
        gp = SingleTaskGPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.25]]))
        assert mean.shape == (1,)
        # Mean should be close to 1.0
        assert np.abs(mean[0] - 1.0) < 0.5

    def test_many_features(self):
        """GP handles many input features."""
        n_features = 10
        X = np.random.rand(20, n_features)
        y = np.sum(X, axis=1)
        gp = SingleTaskGPSurrogate().fit(X, y)
        X_test = np.random.rand(5, n_features)
        mean, std = gp.predict(X_test)
        assert mean.shape == (5,)
        assert gp.n_features == n_features

    def test_many_training_points(self):
        """GP handles many training points."""
        X = np.linspace(0, 1, 50).reshape(-1, 1)
        y = np.sin(2 * np.pi * X).ravel()
        gp = SingleTaskGPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.25], [0.75]]))
        assert mean.shape == (2,)

    def test_negative_target_values(self):
        """GP handles negative target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([-10.0, -5.0, 0.0])
        gp = SingleTaskGPSurrogate().fit(X, y)
        mean, _ = gp.predict(X)
        assert np.all(mean < 5)

    def test_large_target_values(self):
        """GP handles large target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([1e6, 2e6, 3e6])
        gp = SingleTaskGPSurrogate().fit(X, y)
        mean, _ = gp.predict(X)
        assert np.all(mean > 0)


class TestSingleTaskGPSurrogate1DInput:
    """Test 1D input support (scikit-learn convention)."""

    def test_fit_1d_X(self):
        """fit() accepts 1D X array and treats as single feature."""
        X = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = np.sin(2 * np.pi * X)
        gp = SingleTaskGPSurrogate().fit(X, y)
        assert gp.n_features == 1

    def test_predict_1d_X(self):
        """predict() accepts 1D X array."""
        X_train = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y_train = np.sin(2 * np.pi * X_train)
        gp = SingleTaskGPSurrogate().fit(X_train, y_train)

        X_test = np.array([0.125, 0.375])
        mean, std = gp.predict(X_test)
        assert mean.shape == (2,)
        assert std.shape == (2,)

    def test_1d_and_2d_X_equivalent(self):
        """1D and 2D X produce identical results."""
        X_1d = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        X_2d = X_1d.reshape(-1, 1)
        y = np.sin(2 * np.pi * X_1d)

        gp_1d = SingleTaskGPSurrogate().fit(X_1d, y)
        gp_2d = SingleTaskGPSurrogate().fit(X_2d, y)

        X_test_1d = np.array([0.125, 0.375])
        X_test_2d = X_test_1d.reshape(-1, 1)

        mean_1d, std_1d = gp_1d.predict(X_test_1d)
        mean_2d, std_2d = gp_2d.predict(X_test_2d)

        np.testing.assert_allclose(mean_1d, mean_2d)
        np.testing.assert_allclose(std_1d, std_2d)


class TestSingleTaskGPSurrogateCorrectness:
    """Test GP prediction correctness beyond just shape/contract checks."""

    def test_interpolates_linear_function(self):
        """GP accurately interpolates a linear function."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.5, 1.0])
        gp = SingleTaskGPSurrogate().fit(X, y)

        X_test = np.array([[0.25], [0.75]])
        mean, _ = gp.predict(X_test)

        np.testing.assert_allclose(mean, [0.25, 0.75], atol=0.15)

    def test_interpolates_sine_function(self):
        """GP accurately interpolates a sine function at training points."""
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = np.sin(2 * np.pi * X).ravel()
        gp = SingleTaskGPSurrogate().fit(X, y)

        mean, _ = gp.predict(X)
        np.testing.assert_allclose(mean, y, atol=0.2)

    def test_extrapolation_increases_uncertainty(self):
        """Uncertainty increases for extrapolation beyond training range."""
        X = np.array([[0.2], [0.4], [0.6], [0.8]])
        y = np.array([0.2, 0.4, 0.6, 0.8])
        gp = SingleTaskGPSurrogate().fit(X, y)

        _, std_interp = gp.predict(np.array([[0.5]]))
        _, std_extrap = gp.predict(np.array([[1.5]]))

        assert std_extrap[0] > std_interp[0]

    def test_prediction_mean_within_bounds(self):
        """GP predictions stay reasonable near training data."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 1.0, 0.0])
        gp = SingleTaskGPSurrogate().fit(X, y)

        X_test = np.linspace(0, 1, 20).reshape(-1, 1)
        mean, _ = gp.predict(X_test)

        # Predictions should not wildly exceed training range
        assert np.all(mean >= -0.5)
        assert np.all(mean <= 1.5)

    def test_more_data_reduces_uncertainty(self):
        """More training data reduces prediction uncertainty."""
        X_sparse = np.array([[0.0], [1.0]])
        X_dense = np.linspace(0, 1, 10).reshape(-1, 1)
        y_sparse = np.array([0.0, 1.0])
        y_dense = np.linspace(0, 1, 10)

        gp_sparse = SingleTaskGPSurrogate().fit(X_sparse, y_sparse)
        gp_dense = SingleTaskGPSurrogate().fit(X_dense, y_dense)

        X_test = np.array([[0.5]])
        _, std_sparse = gp_sparse.predict(X_test)
        _, std_dense = gp_dense.predict(X_test)

        assert std_dense[0] < std_sparse[0]


def _fit_reference_botorch_gp(X, y, normalize_inputs=True, normalize_outputs=True):
    """Fit a reference BoTorch SingleTaskGP for comparison.

    Must match SingleTaskGPSurrogate defaults exactly:
    - kernel: Matérn 2.5 with ARD
    - covar_module: ScaleKernel wrapping the base kernel
    """
    from gpytorch.kernels import MaternKernel, ScaleKernel

    X_torch = torch.tensor(X, dtype=torch.float64)
    y_torch = torch.tensor(y, dtype=torch.float64).unsqueeze(-1)

    input_transform = Normalize(d=X_torch.shape[-1]) if normalize_inputs else None
    outcome_transform = Standardize(m=1) if normalize_outputs else None

    # Match implementation: ScaleKernel(MaternKernel(nu=2.5, ard))
    base_kernel = MaternKernel(nu=2.5, ard_num_dims=X_torch.shape[-1])
    covar_module = ScaleKernel(base_kernel)

    model = SingleTaskGP(
        X_torch,
        y_torch,
        covar_module=covar_module,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


def _predict_reference_botorch_gp(model, X):
    """Get predictions from reference BoTorch GP."""
    X_torch = torch.tensor(X, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X_torch)
        mean = posterior.mean.squeeze(-1).numpy()
        std = posterior.variance.squeeze(-1).sqrt().numpy()
    return mean, std


class TestSingleTaskGPSurrogateVsReference:
    """Test that SingleTaskGPSurrogate matches reference BoTorch implementation."""

    def test_matches_reference_simple_linear(self):
        """Output matches BoTorch on simple linear data."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.5, 1.0])
        X_test = np.array([[0.25], [0.75]])

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_sine(self):
        """Output matches BoTorch on sine wave data."""
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = np.sin(2 * np.pi * X).ravel()
        X_test = np.array([[0.15], [0.35], [0.55], [0.85]])

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_multidim(self):
        """Output matches BoTorch on multi-dimensional input."""
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
        X_test = np.array([[0.25, 0.25], [0.75, 0.75]])

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_no_normalization(self):
        """Output matches BoTorch without normalization."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.5, 1.0])
        X_test = np.array([[0.25], [0.75]])

        gp = SingleTaskGPSurrogate(normalize_inputs=False, normalize_outputs=False).fit(
            X, y
        )
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(
            X, y, normalize_inputs=False, normalize_outputs=False
        )
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_at_training_points(self):
        """Output matches BoTorch when predicting at training points."""
        X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        y = np.sin(2 * np.pi * X).ravel()

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)


class TestSingleTaskGPSurrogateVsReferenceEdgeCases:
    """Test reference matching on edge cases and weird inputs."""

    def test_matches_reference_single_point(self):
        """Output matches BoTorch with single training point.

        Note: Both implementations emit gpytorch NumericalWarning due to
        covariance matrix instability with n=1. This is expected behavior.
        """
        X = np.array([[0.5]])
        y = np.array([1.0])
        X_test = np.array([[0.0], [0.25], [0.75], [1.0]])

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_constant_y(self):
        """Output matches BoTorch with constant target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([1.0, 1.0, 1.0])
        X_test = np.array([[0.25], [0.75]])

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_large_y_values(self):
        """Output matches BoTorch with large target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([1e6, 2e6, 3e6])
        X_test = np.array([[0.25], [0.75]])

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_negative_y_values(self):
        """Output matches BoTorch with negative target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([-10.0, -5.0, 0.0])
        X_test = np.array([[0.25], [0.75]])

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_different_scales(self):
        """Output matches BoTorch with inputs at different scales."""
        X = np.array([[1.0, 1000.0], [2.0, 2000.0], [3.0, 3000.0]])
        y = X[:, 0] + X[:, 1] / 1000
        X_test = np.array([[1.5, 1500.0], [2.5, 2500.0]])

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_extrapolation(self):
        """Output matches BoTorch when extrapolating beyond training range."""
        X = np.array([[0.2], [0.4], [0.6], [0.8]])
        y = np.array([0.2, 0.4, 0.6, 0.8])
        X_test = np.array([[-0.5], [1.5], [2.0]])

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_many_features(self):
        """Output matches BoTorch with many input features."""
        np.random.seed(42)
        n_features = 10
        X = np.random.rand(20, n_features)
        y = np.sum(X, axis=1)
        X_test = np.random.rand(5, n_features)

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_many_points(self):
        """Output matches BoTorch with many training points."""
        X = np.linspace(0, 1, 50).reshape(-1, 1)
        y = np.sin(2 * np.pi * X).ravel()
        X_test = np.array([[0.15], [0.35], [0.55], [0.85]])

        gp = SingleTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_gp(ref_model, X_test)

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)
