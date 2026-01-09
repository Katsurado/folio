"""Tests for MultiTaskGPSurrogate multi-output Gaussian Process model."""

import numpy as np
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from folio.exceptions import NotFittedError
from folio.surrogates.multitask_gp import MultiTaskGPSurrogate


@pytest.fixture
def correlated_data():
    """Generate correlated multi-output training data.

    y1 = sin(x), y2 = sin(x) + 0.3*cos(x)
    """
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y1 = np.sin(2 * np.pi * X).ravel()
    y2 = np.sin(2 * np.pi * X).ravel() + 0.3 * np.cos(2 * np.pi * X).ravel()
    y = np.column_stack([y1, y2])
    return X, y


@pytest.fixture
def three_task_data():
    """Generate 3-task multi-output training data."""
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y1 = np.sin(2 * np.pi * X).ravel()
    y2 = np.cos(2 * np.pi * X).ravel()
    y3 = X.ravel() ** 2
    y = np.column_stack([y1, y2, y3])
    return X, y


@pytest.fixture
def multidim_input_data():
    """Generate multi-dimensional input, multi-output data."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    y1 = X[:, 0] + X[:, 1]
    y2 = X[:, 0] * X[:, 1]
    y = np.column_stack([y1, y2])
    return X, y


class TestMultiTaskGPSurrogateInit:
    """Test MultiTaskGPSurrogate constructor validation."""

    def test_default_parameters(self):
        """MultiTaskGPSurrogate initializes with sensible defaults."""
        gp = MultiTaskGPSurrogate()
        assert gp.kernel == "matern"
        assert gp.nu == 2.5
        assert gp.ard is True
        assert gp.normalize_inputs is True
        assert gp.normalize_outputs is True

    def test_rbf_kernel(self):
        """RBF kernel can be specified."""
        gp = MultiTaskGPSurrogate(kernel="rbf")
        assert gp.kernel == "rbf"

    def test_matern_kernel(self):
        """Matern kernel is default."""
        gp = MultiTaskGPSurrogate(kernel="matern")
        assert gp.kernel == "matern"

    def test_invalid_kernel_raises(self):
        """Invalid kernel name raises ValueError."""
        with pytest.raises(ValueError, match="kernel"):
            MultiTaskGPSurrogate(kernel="invalid")

    def test_nu_values(self):
        """Valid nu values (0.5, 1.5, 2.5) are accepted."""
        for nu in [0.5, 1.5, 2.5]:
            gp = MultiTaskGPSurrogate(nu=nu)
            assert gp.nu == nu

    def test_invalid_nu_raises(self):
        """Invalid nu value raises ValueError."""
        with pytest.raises(ValueError, match="nu"):
            MultiTaskGPSurrogate(nu=1.0)

    def test_ard_disabled(self):
        """ARD can be disabled."""
        gp = MultiTaskGPSurrogate(ard=False)
        assert gp.ard is False

    def test_normalization_disabled(self):
        """Normalization can be disabled."""
        gp = MultiTaskGPSurrogate(normalize_inputs=False, normalize_outputs=False)
        assert gp.normalize_inputs is False
        assert gp.normalize_outputs is False


class TestMultiTaskGPSurrogateFit:
    """Test MultiTaskGPSurrogate fit method."""

    def test_fit_predict_correlated(self, correlated_data):
        """fit() and predict() work with correlated outputs, shapes are correct."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate()
        gp.fit(X, y)

        X_test = np.array([[0.15], [0.35], [0.55]])
        mean, std = gp.predict(X_test)

        assert mean.shape == (3, 2)
        assert std.shape == (3, 2)
        assert np.all(std >= 0)

    def test_method_chaining(self, correlated_data):
        """fit() returns self for method chaining."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate()
        result = gp.fit(X, y)
        assert result is gp

    def test_fit_sets_n_features(self, correlated_data):
        """fit() stores number of features."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate()
        gp.fit(X, y)
        assert gp.n_features == 1

    def test_fit_sets_n_tasks(self, correlated_data):
        """fit() stores number of tasks."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate()
        gp.fit(X, y)
        assert gp.n_tasks == 2

    def test_three_tasks(self, three_task_data):
        """fit() and predict() work with 3 tasks."""
        X, y = three_task_data
        gp = MultiTaskGPSurrogate()
        gp.fit(X, y)

        X_test = np.array([[0.15], [0.35]])
        mean, std = gp.predict(X_test)

        assert mean.shape == (2, 3)
        assert std.shape == (2, 3)
        assert gp.n_tasks == 3

    def test_fit_multidim_input(self, multidim_input_data):
        """fit() works with multi-dimensional inputs."""
        X, y = multidim_input_data
        gp = MultiTaskGPSurrogate()
        gp.fit(X, y)
        assert gp.n_features == 2
        assert gp.n_tasks == 2

    def test_fit_shape_mismatch_raises(self, correlated_data):
        """fit() raises ValueError if X and y have different n_samples."""
        X, y = correlated_data
        with pytest.raises(ValueError, match="(?i)samples|shape|mismatch"):
            MultiTaskGPSurrogate().fit(X, y[:-1])

    def test_fit_with_rbf_kernel(self, correlated_data):
        """fit() works with RBF kernel."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate(kernel="rbf")
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (10, 2)

    def test_fit_without_ard(self, correlated_data):
        """fit() works without ARD."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate(ard=False)
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (10, 2)

    def test_fit_without_normalization(self, correlated_data):
        """fit() works without normalization."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate(normalize_inputs=False, normalize_outputs=False)
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (10, 2)


class TestMultiTaskGPSurrogatePredict:
    """Test MultiTaskGPSurrogate predict method."""

    def test_predict_returns_tuple(self, correlated_data):
        """predict() returns (mean, std) tuple."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate().fit(X, y)
        result = gp.predict(np.array([[0.125]]))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_output_shapes(self, correlated_data):
        """predict() returns arrays with shape (n_candidates, n_tasks)."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate().fit(X, y)
        X_test = np.array([[0.1], [0.2], [0.3], [0.4]])
        mean, std = gp.predict(X_test)
        assert mean.shape == (4, 2)
        assert std.shape == (4, 2)

    def test_predict_single_point(self, correlated_data):
        """predict() works with a single candidate."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.5]]))
        assert mean.shape == (1, 2)
        assert std.shape == (1, 2)

    def test_predict_std_nonnegative(self, correlated_data):
        """Standard deviation is always non-negative."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate().fit(X, y)
        X_test = np.linspace(0, 1, 20).reshape(-1, 1)
        _, std = gp.predict(X_test)
        assert np.all(std >= 0)

    def test_predict_returns_numpy_arrays(self, correlated_data):
        """predict() returns numpy arrays, not torch tensors."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.5]]))
        assert isinstance(mean, np.ndarray)
        assert isinstance(std, np.ndarray)

    def test_predict_multidim(self, multidim_input_data):
        """predict() works with multi-dimensional inputs."""
        X, y = multidim_input_data
        gp = MultiTaskGPSurrogate().fit(X, y)
        X_test = np.array([[0.25, 0.25], [0.75, 0.75]])
        mean, std = gp.predict(X_test)
        assert mean.shape == (2, 2)
        assert std.shape == (2, 2)


class TestMultiTaskGPSurrogateValidation:
    """Test input validation and error handling."""

    def test_raises_if_y_1d(self):
        """fit() raises ValueError when y is 1D."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="(?i)2d|2-d|dimension|ndim"):
            MultiTaskGPSurrogate().fit(X, y)

    def test_raises_if_single_task(self):
        """fit() raises ValueError when y.shape[1] == 1."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([[0.0], [0.5], [1.0]])
        with pytest.raises(ValueError, match="(?i)single|task|column|SingleTaskGP"):
            MultiTaskGPSurrogate().fit(X, y)

    def test_raises_not_fitted(self):
        """predict() raises NotFittedError when called before fit()."""
        gp = MultiTaskGPSurrogate()
        X_test = np.array([[0.5]])
        with pytest.raises(NotFittedError, match="(?i)fit"):
            gp.predict(X_test)

    def test_raises_feature_mismatch(self, correlated_data):
        """predict() raises ValueError when X has wrong n_features."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate().fit(X, y)
        X_wrong = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="(?i)feature"):
            gp.predict(X_wrong)

    def test_model_is_none_before_fit(self):
        """model attribute is None before fit()."""
        gp = MultiTaskGPSurrogate()
        assert gp.model is None

    def test_raises_if_zero_observations(self):
        """fit() raises ValueError when X has zero observations."""
        X = np.array([]).reshape(0, 2)
        y = np.array([]).reshape(0, 2)
        with pytest.raises(ValueError, match="(?i)observation|sample|empty|0"):
            MultiTaskGPSurrogate().fit(X, y)


class TestMultiTaskGPSurrogateToMultitaskFormat:
    """Test _to_multitask_format helper method."""

    def test_to_multitask_format_shapes(self, correlated_data):
        """_to_multitask_format produces correct output shapes."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate()

        X_t = torch.tensor(X, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64)

        X_mt, y_mt = gp._to_multitask_format(X_t, y_t)

        n, d = X.shape
        k = y.shape[1]

        assert X_mt.shape == (n * k, d + 1)
        assert y_mt.shape == (n * k, 1)

    def test_to_multitask_format_task_indices(self):
        """_to_multitask_format produces correct task indices."""
        gp = MultiTaskGPSurrogate()

        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        y = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)

        X_mt, y_mt = gp._to_multitask_format(X, y)

        assert X_mt.shape == (4, 3)
        assert y_mt.shape == (4, 1)

        task_indices = X_mt[:, -1]
        expected_tasks = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(task_indices, expected_tasks)

    def test_to_multitask_format_values(self):
        """_to_multitask_format preserves input values correctly."""
        gp = MultiTaskGPSurrogate()

        X = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        y = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float64)

        X_mt, y_mt = gp._to_multitask_format(X, y)

        expected_y = torch.tensor([[10.0], [30.0], [20.0], [40.0]], dtype=torch.float64)
        torch.testing.assert_close(y_mt, expected_y)

        expected_X_features = torch.tensor(
            [[1.0], [2.0], [1.0], [2.0]], dtype=torch.float64
        )
        torch.testing.assert_close(X_mt[:, :-1], expected_X_features)

    def test_to_multitask_format_three_tasks(self):
        """_to_multitask_format works with 3 tasks."""
        gp = MultiTaskGPSurrogate()

        X = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        y = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=torch.float64)

        X_mt, y_mt = gp._to_multitask_format(X, y)

        assert X_mt.shape == (6, 2)
        assert y_mt.shape == (6, 1)

        task_indices = X_mt[:, -1]
        expected_tasks = torch.tensor(
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=torch.float64
        )
        torch.testing.assert_close(task_indices, expected_tasks)


class TestMultiTaskGPSurrogateBehavior:
    """Test GP prediction behavior and properties."""

    def test_low_uncertainty_at_training_points(self, correlated_data):
        """Uncertainty should be low at training points."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate().fit(X, y)
        _, std = gp.predict(X)
        assert np.all(std < 0.5)

    def test_predictions_close_to_training_values(self, correlated_data):
        """Predictions at training points should match training values."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean, _ = gp.predict(X)
        assert np.allclose(mean, y, atol=0.3)

    def test_task_correlation_improves_predictions(self):
        """Multi-task GP should leverage task correlations.

        When tasks are correlated, predictions should be reasonable even
        with limited data, as information is shared between tasks.
        """
        X = np.array([[0.0], [0.5], [1.0]])
        y1 = np.array([0.0, 1.0, 0.0])
        y2 = np.array([0.0, 1.0, 0.0])
        y = np.column_stack([y1, y2])

        gp = MultiTaskGPSurrogate().fit(X, y)
        X_test = np.array([[0.25], [0.75]])
        mean, _ = gp.predict(X_test)

        assert mean.shape == (2, 2)
        assert np.all(mean >= -0.5)
        assert np.all(mean <= 1.5)


class TestMultiTaskGPSurrogateEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_two_samples_two_tasks(self):
        """GP handles minimum viable data (2 samples, 2 tasks)."""
        X = np.array([[0.0], [1.0]])
        y = np.array([[0.0, 1.0], [1.0, 0.0]])
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.5]]))
        assert mean.shape == (1, 2)
        assert std.shape == (1, 2)

    def test_many_tasks(self):
        """GP handles many output tasks."""
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        n_tasks = 5
        y = np.column_stack([np.sin(2 * np.pi * X + i) for i in range(n_tasks)])
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.25]]))
        assert mean.shape == (1, n_tasks)
        assert gp.n_tasks == n_tasks

    def test_many_features(self):
        """GP handles many input features."""
        n_features = 5
        np.random.seed(42)
        X = np.random.rand(15, n_features)
        y = np.column_stack([np.sum(X, axis=1), np.prod(X, axis=1)])
        gp = MultiTaskGPSurrogate().fit(X, y)
        X_test = np.random.rand(3, n_features)
        mean, std = gp.predict(X_test)
        assert mean.shape == (3, 2)
        assert gp.n_features == n_features

    def test_negative_target_values(self):
        """GP handles negative target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([[-10.0, -5.0], [-5.0, -2.5], [0.0, 0.0]])
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean, _ = gp.predict(X)
        assert np.all(mean < 5)

    def test_large_target_values(self):
        """GP handles large target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([[1e6, 2e6], [1.5e6, 2.5e6], [2e6, 3e6]])
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean, _ = gp.predict(X)
        assert np.all(mean > 0)

    def test_constant_task_values(self):
        """GP handles constant values in one task."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean, std = gp.predict(np.array([[0.25]]))
        assert mean.shape == (1, 2)
        assert np.abs(mean[0, 0] - 1.0) < 0.5


class TestMultiTaskGPSurrogateCorrectness:
    """Test GP prediction correctness beyond just shape/contract checks."""

    def test_interpolates_linear_functions_small_data(self):
        """GP interpolates linear functions with small data (relaxed tolerance)."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        gp = MultiTaskGPSurrogate().fit(X, y)

        X_test = np.array([[0.25], [0.75]])
        mean, _ = gp.predict(X_test)

        # Relaxed tolerance for sparse data
        np.testing.assert_allclose(mean[:, 0], [0.25, 0.75], atol=0.35)
        np.testing.assert_allclose(mean[:, 1], [0.75, 0.25], atol=0.35)

    def test_interpolates_linear_functions_more_data(self):
        """GP accurately interpolates linear functions with more data."""
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = np.column_stack([X.ravel(), 1 - X.ravel()])
        gp = MultiTaskGPSurrogate().fit(X, y)

        X_test = np.array([[0.25], [0.75]])
        mean, _ = gp.predict(X_test)

        # Strict tolerance for dense data
        np.testing.assert_allclose(mean[:, 0], [0.25, 0.75], atol=0.15)
        np.testing.assert_allclose(mean[:, 1], [0.75, 0.25], atol=0.15)

    def test_interpolates_at_training_points(self, correlated_data):
        """GP accurately interpolates at training points."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate().fit(X, y)

        mean, _ = gp.predict(X)
        np.testing.assert_allclose(mean, y, atol=0.2)

    def test_extrapolation_increases_uncertainty(self, correlated_data):
        """Uncertainty increases for extrapolation beyond training range."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate().fit(X, y)

        _, std_interp = gp.predict(np.array([[0.5]]))
        _, std_extrap = gp.predict(np.array([[1.5]]))

        assert np.all(std_extrap > std_interp)

    def test_prediction_mean_within_bounds(self, correlated_data):
        """GP predictions stay reasonable near training data."""
        X, y = correlated_data
        gp = MultiTaskGPSurrogate().fit(X, y)

        X_test = np.linspace(0, 1, 20).reshape(-1, 1)
        mean, _ = gp.predict(X_test)

        y_min, y_max = y.min(), y.max()
        assert np.all(mean >= y_min - 0.5)
        assert np.all(mean <= y_max + 0.5)

    def test_more_data_reduces_uncertainty(self):
        """More training data reduces prediction uncertainty."""
        X_sparse = np.array([[0.0], [1.0]])
        y_sparse = np.array([[0.0, 1.0], [1.0, 0.0]])

        X_dense = np.linspace(0, 1, 10).reshape(-1, 1)
        y_dense = np.column_stack([X_dense.ravel(), 1 - X_dense.ravel()])

        gp_sparse = MultiTaskGPSurrogate().fit(X_sparse, y_sparse)
        gp_dense = MultiTaskGPSurrogate().fit(X_dense, y_dense)

        X_test = np.array([[0.5]])
        _, std_sparse = gp_sparse.predict(X_test)
        _, std_dense = gp_dense.predict(X_test)

        assert np.all(std_dense < std_sparse)

    def test_correlated_tasks_share_information(self):
        """Correlated tasks should produce similar predictions.

        When tasks are perfectly correlated (y2 = y1), predictions for
        both tasks should be nearly identical.
        """
        X = np.linspace(0, 1, 5).reshape(-1, 1)
        y1 = np.sin(2 * np.pi * X).ravel()
        y = np.column_stack([y1, y1])

        gp = MultiTaskGPSurrogate().fit(X, y)
        X_test = np.array([[0.25], [0.75]])
        mean, _ = gp.predict(X_test)

        np.testing.assert_allclose(mean[:, 0], mean[:, 1], atol=0.2)


def _to_multitask_format_reference(X: np.ndarray, y: np.ndarray):
    """Convert to multi-task format for reference BoTorch implementation.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input features.
    y : np.ndarray, shape (n, k)
        Output values with k tasks.

    Returns
    -------
    X_mt : torch.Tensor, shape (n*k, d+1)
        Expanded input with task index as last column.
    y_mt : torch.Tensor, shape (n*k, 1)
        Flattened outputs.
    """
    n, d = X.shape
    k = y.shape[1]

    X_mt_list = []
    y_mt_list = []

    for task_idx in range(k):
        task_indices = np.full((n, 1), task_idx, dtype=np.float64)
        X_with_task = np.hstack([X, task_indices])
        X_mt_list.append(X_with_task)
        y_mt_list.append(y[:, task_idx : task_idx + 1])

    X_mt = np.vstack(X_mt_list)
    y_mt = np.vstack(y_mt_list)

    return torch.tensor(X_mt, dtype=torch.float64), torch.tensor(
        y_mt, dtype=torch.float64
    )


def _fit_reference_botorch_multitask_gp(
    X: np.ndarray,
    y: np.ndarray,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
):
    """Fit a reference BoTorch MultiTaskGP for comparison.

    Uses same settings as MultiTaskGPSurrogate defaults.
    """
    X_mt, y_mt = _to_multitask_format_reference(X, y)

    n_tasks = y.shape[1]
    d = X.shape[1]

    if normalize_inputs:
        input_transform = Normalize(d=d + 1, indices=list(range(d)))
    else:
        input_transform = None
    outcome_transform = Standardize(m=1) if normalize_outputs else None

    # Use same kernel as MultiTaskGPSurrogate (Matern 2.5 with ARD)
    base_kernel = MaternKernel(nu=2.5, ard_num_dims=d)
    covar_module = ScaleKernel(base_kernel)

    model = MultiTaskGP(
        train_X=X_mt,
        train_Y=y_mt,
        task_feature=-1,
        covar_module=covar_module,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
        output_tasks=list(range(n_tasks)),
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


def _predict_reference_botorch_multitask_gp(
    model: MultiTaskGP, X: np.ndarray, n_tasks: int
):
    """Get predictions from reference BoTorch MultiTaskGP.

    Parameters
    ----------
    model : MultiTaskGP
        Fitted BoTorch model.
    X : np.ndarray, shape (n, d)
        Test points.
    n_tasks : int
        Number of tasks.

    Returns
    -------
    mean : np.ndarray, shape (n, n_tasks)
        Predicted means.
    std : np.ndarray, shape (n, n_tasks)
        Predicted standard deviations.
    """
    n = X.shape[0]
    mean_list = []
    std_list = []

    model.eval()
    with torch.no_grad():
        for task_idx in range(n_tasks):
            task_indices = np.full((n, 1), task_idx, dtype=np.float64)
            X_with_task = np.hstack([X, task_indices])
            X_torch = torch.tensor(X_with_task, dtype=torch.float64)

            posterior = model.posterior(X_torch)
            mean_list.append(posterior.mean.squeeze(-1).numpy())
            std_list.append(posterior.variance.squeeze(-1).sqrt().numpy())

    mean = np.column_stack(mean_list)
    std = np.column_stack(std_list)

    return mean, std


class TestMultiTaskGPSurrogateVsReference:
    """Test that MultiTaskGPSurrogate matches reference BoTorch implementation."""

    def test_matches_reference_correlated_data(self, correlated_data):
        """Output matches BoTorch on correlated output data."""
        X, y = correlated_data
        X_test = np.array([[0.15], [0.35], [0.55], [0.85]])

        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_at_training_points(self, correlated_data):
        """Output matches BoTorch when predicting at training points."""
        X, y = correlated_data

        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X)

        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_three_tasks(self, three_task_data):
        """Output matches BoTorch with 3 tasks."""
        X, y = three_task_data
        X_test = np.array([[0.15], [0.55]])

        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=3
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_multidim_input(self, multidim_input_data):
        """Output matches BoTorch on multi-dimensional input."""
        X, y = multidim_input_data
        X_test = np.array([[0.25, 0.25], [0.75, 0.75]])

        # Set seed for reproducible hyperparameter optimization
        torch.manual_seed(42)
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        torch.manual_seed(42)
        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_no_normalization(self):
        """Output matches BoTorch without normalization."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        X_test = np.array([[0.25], [0.75]])

        gp = MultiTaskGPSurrogate(normalize_inputs=False, normalize_outputs=False).fit(
            X, y
        )
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_multitask_gp(
            X, y, normalize_inputs=False, normalize_outputs=False
        )
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_extrapolation(self, correlated_data):
        """Output matches BoTorch when extrapolating beyond training range."""
        X, y = correlated_data
        X_test = np.array([[-0.5], [1.5]])

        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)


class TestMultiTaskGPSurrogateVsReferenceEdgeCases:
    """Test reference matching on edge cases."""

    def test_matches_reference_large_y_values(self):
        """Output matches BoTorch with large target values."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([[1e6, 2e6], [1.5e6, 2.5e6], [2e6, 3e6]])
        X_test = np.array([[0.25], [0.75]])

        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_negative_y_values_small_data(self):
        """Output matches BoTorch with negative values (small data)."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([[-10.0, -5.0], [-5.0, -2.5], [0.0, 0.0]])
        X_test = np.array([[0.25], [0.75]])

        # Set seed for reproducible hyperparameter optimization
        torch.manual_seed(42)
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        torch.manual_seed(42)
        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_negative_y_values_more_data(self):
        """Output matches BoTorch with negative values (more data)."""
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = np.column_stack([-10 + 10 * X.ravel(), -5 + 5 * X.ravel()])
        X_test = np.array([[0.25], [0.75]])

        # Set seed for reproducible hyperparameter optimization
        torch.manual_seed(42)
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        torch.manual_seed(42)
        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_different_scales_small_data(self):
        """Output matches BoTorch with different scales (small data)."""
        X = np.array([[1.0, 1000.0], [2.0, 2000.0], [3.0, 3000.0]])
        y = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        X_test = np.array([[1.5, 1500.0], [2.5, 2500.0]])

        # Set seed for reproducible hyperparameter optimization
        torch.manual_seed(42)
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        torch.manual_seed(42)
        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_different_scales_more_data(self):
        """Output matches BoTorch with different scales (more data)."""
        X = np.column_stack([np.linspace(1, 3, 10), np.linspace(1000, 3000, 10)])
        y = np.column_stack([np.linspace(1, 3, 10), np.linspace(100, 300, 10)])
        X_test = np.array([[1.5, 1500.0], [2.5, 2500.0]])

        # Set seed for reproducible hyperparameter optimization
        torch.manual_seed(42)
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        torch.manual_seed(42)
        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_many_features_small_data(self):
        """Output matches BoTorch with many features (small data)."""
        np.random.seed(42)
        n_features = 5
        X = np.random.rand(5, n_features)
        y = np.column_stack([np.sum(X, axis=1), np.mean(X, axis=1)])
        X_test = np.random.rand(3, n_features)

        # Set seed for reproducible hyperparameter optimization
        torch.manual_seed(42)
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        torch.manual_seed(42)
        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)

    def test_matches_reference_many_features_more_data(self):
        """Output matches BoTorch with many features (more data)."""
        np.random.seed(42)
        n_features = 5
        X = np.random.rand(20, n_features)
        y = np.column_stack([np.sum(X, axis=1), np.mean(X, axis=1)])
        X_test = np.random.rand(3, n_features)

        # Set seed for reproducible hyperparameter optimization
        torch.manual_seed(42)
        gp = MultiTaskGPSurrogate().fit(X, y)
        mean_ours, std_ours = gp.predict(X_test)

        torch.manual_seed(42)
        ref_model = _fit_reference_botorch_multitask_gp(X, y)
        mean_ref, std_ref = _predict_reference_botorch_multitask_gp(
            ref_model, X_test, n_tasks=2
        )

        np.testing.assert_allclose(mean_ours, mean_ref, rtol=0.1)
        np.testing.assert_allclose(std_ours, std_ref, rtol=0.1)
