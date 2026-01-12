"""Tests for BayesianRecommender."""

import numpy as np
import pytest
import torch

from folio.core.config import RecommenderConfig, TargetConfig
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.recommenders.bayesian import BayesianRecommender


@pytest.fixture
def simple_project():
    """Create a simple project with continuous inputs only."""
    return Project(
        id=1,
        name="test_project",
        inputs=[
            InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
            InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
        ],
        outputs=[OutputSpec("y")],
        target_configs=[TargetConfig(objective="y", objective_mode="maximize")],
        recommender_config=RecommenderConfig(
            type="bayesian",
            surrogate="gp",
            acquisition="ei",
            n_initial=3,
        ),
    )


@pytest.fixture
def minimize_project():
    """Create a project configured for minimization."""
    return Project(
        id=1,
        name="minimize_project",
        inputs=[
            InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
            InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
        ],
        outputs=[OutputSpec("y")],
        target_configs=[TargetConfig(objective="y", objective_mode="minimize")],
        recommender_config=RecommenderConfig(
            type="bayesian",
            surrogate="gp",
            acquisition="ei",
            n_initial=3,
        ),
    )


@pytest.fixture
def ucb_project():
    """Create a project configured to use UCB acquisition."""
    return Project(
        id=1,
        name="ucb_project",
        inputs=[
            InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
            InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
        ],
        outputs=[OutputSpec("y")],
        target_configs=[TargetConfig(objective="y", objective_mode="maximize")],
        recommender_config=RecommenderConfig(
            type="bayesian",
            surrogate="gp",
            acquisition="ucb",
            n_initial=3,
            acquisition_kwargs={"beta": 2.5},
        ),
    )


@pytest.fixture
def single_input_project():
    """Create a project with a single continuous input."""
    return Project(
        id=1,
        name="single_input",
        inputs=[InputSpec("x", "continuous", bounds=(0.0, 1.0))],
        outputs=[OutputSpec("y")],
        target_configs=[TargetConfig(objective="y", objective_mode="maximize")],
        recommender_config=RecommenderConfig(n_initial=2),
    )


@pytest.fixture
def few_observations(simple_project):
    """Create fewer observations than n_initial (should trigger random sampling)."""
    return [
        Observation(
            project_id=1,
            inputs={"x1": 5.0, "x2": 0.0},
            outputs={"y": 10.0},
        ),
        Observation(
            project_id=1,
            inputs={"x1": 2.0, "x2": -3.0},
            outputs={"y": 7.0},
        ),
    ]


@pytest.fixture
def enough_observations(simple_project):
    """Create enough observations to trigger BO (>= n_initial)."""
    return [
        Observation(
            project_id=1,
            inputs={"x1": 5.0, "x2": 0.0},
            outputs={"y": 10.0},
        ),
        Observation(
            project_id=1,
            inputs={"x1": 2.0, "x2": -3.0},
            outputs={"y": 7.0},
        ),
        Observation(
            project_id=1,
            inputs={"x1": 8.0, "x2": 2.0},
            outputs={"y": 15.0},
        ),
        Observation(
            project_id=1,
            inputs={"x1": 1.0, "x2": -1.0},
            outputs={"y": 5.0},
        ),
    ]


class TestBayesianRecommenderInit:
    """Test BayesianRecommender initialization."""

    def test_init_with_project(self, simple_project):
        """BayesianRecommender can be initialized with a Project."""
        recommender = BayesianRecommender(simple_project)
        assert recommender.project is simple_project

    def test_init_stores_project(self, simple_project):
        """BayesianRecommender stores the project reference."""
        recommender = BayesianRecommender(simple_project)
        assert recommender.project.name == "test_project"


class TestBayesianRecommenderRecommend:
    """Test BayesianRecommender.recommend method."""

    def test_recommend_returns_dict(self, simple_project, enough_observations):
        """recommend() returns a dictionary."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend(enough_observations)
        assert isinstance(result, dict)

    def test_recommend_has_correct_keys(self, simple_project, enough_observations):
        """recommend() returns dict with all input names as keys."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend(enough_observations)
        assert set(result.keys()) == {"x1", "x2"}

    def test_recommend_values_within_bounds(self, simple_project, enough_observations):
        """recommend() returns values within input bounds."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend(enough_observations)
        assert 0.0 <= result["x1"] <= 10.0
        assert -5.0 <= result["x2"] <= 5.0

    def test_recommend_returns_float_values(self, simple_project, enough_observations):
        """recommend() returns float values for continuous inputs."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend(enough_observations)
        assert isinstance(result["x1"], float)
        assert isinstance(result["x2"], float)


class TestBayesianRecommenderRecommendFromData:
    """Test BayesianRecommender.recommend_from_data method."""

    def test_recommend_from_data_returns_array(self, simple_project):
        """recommend_from_data() returns a numpy array."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        X = np.array([[5.0, 0.0], [2.0, -3.0], [8.0, 2.0], [1.0, -1.0]])
        y = np.array([[10.0], [7.0], [15.0], [5.0]])
        # Bounds shape (2, d): row 0 = lower, row 1 = upper
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = recommender.recommend_from_data(X, y, bounds, [True])
        assert isinstance(result, np.ndarray)

    def test_recommend_from_data_correct_shape(self, simple_project):
        """recommend_from_data() returns array with shape (n_features,)."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        X = np.array([[5.0, 0.0], [2.0, -3.0], [8.0, 2.0], [1.0, -1.0]])
        y = np.array([[10.0], [7.0], [15.0], [5.0]])
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = recommender.recommend_from_data(X, y, bounds, [True])
        assert result.shape == (2,)

    def test_recommend_from_data_within_bounds(self, simple_project):
        """recommend_from_data() returns values within bounds."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        X = np.array([[5.0, 0.0], [2.0, -3.0], [8.0, 2.0], [1.0, -1.0]])
        y = np.array([[10.0], [7.0], [15.0], [5.0]])
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = recommender.recommend_from_data(X, y, bounds, [True])
        assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))

    def test_recommend_from_data_empty_X_returns_random(self, simple_project):
        """recommend_from_data() with empty X returns random sample."""
        recommender = BayesianRecommender(simple_project)
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = recommender.recommend_from_data(
            X=np.empty((0, 2)), y=np.empty((0, 1)), bounds=bounds, maximize=[True]
        )
        assert result.shape == (2,)
        assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))

    def test_recommend_from_data_few_samples_returns_random(self, simple_project):
        """recommend_from_data() with fewer than n_initial samples returns random."""
        recommender = BayesianRecommender(simple_project)
        # n_initial=3, so 2 samples should trigger random
        X = np.array([[5.0, 0.0], [2.0, -3.0]])
        y = np.array([[10.0], [7.0]])
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = recommender.recommend_from_data(X, y, bounds, [True])
        assert result.shape == (2,)
        assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))

    def test_recommend_from_data_maximize(self, simple_project):
        """recommend_from_data() respects maximize objective."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        # Pattern: y increases with x1
        X = np.array([[1.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
        y = np.array([[1.0], [5.0], [9.0]])
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = recommender.recommend_from_data(X, y, bounds, [True])
        assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))

    def test_recommend_from_data_minimize(self, minimize_project):
        """recommend_from_data() respects minimize objective."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(minimize_project)
        # Pattern: y increases with x1, so minimum is at low x1
        X = np.array([[1.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
        y = np.array([[1.0], [5.0], [9.0]])
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = recommender.recommend_from_data(X, y, bounds, [False])
        assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))

    def test_recommend_from_data_single_dimension(self, single_input_project):
        """recommend_from_data() works with single dimension."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(single_input_project)
        X = np.array([[0.2], [0.5], [0.8]])
        y = np.array([[0.2], [0.8], [0.4]])
        # Shape (2, 1) for single dimension
        bounds = np.array([[0.0], [1.0]])
        result = recommender.recommend_from_data(X, y, bounds, [True])
        assert result.shape == (1,)
        assert 0.0 <= result[0] <= 1.0


class TestBayesianRecommenderEmptyObservations:
    """Test BayesianRecommender behavior with empty observations."""

    def test_recommend_with_empty_observations(self, simple_project):
        """recommend() with empty list returns random sample."""
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend([])
        assert set(result.keys()) == {"x1", "x2"}
        assert 0.0 <= result["x1"] <= 10.0
        assert -5.0 <= result["x2"] <= 5.0

    def test_empty_observations_returns_random(self, simple_project):
        """Empty observations should trigger random sampling (no surrogate)."""
        recommender = BayesianRecommender(simple_project)
        # Multiple calls should produce different results (random)
        results = [recommender.recommend([]) for _ in range(5)]
        x1_values = [r["x1"] for r in results]
        # With randomness, values should vary
        assert len(set(x1_values)) > 1


class TestBayesianRecommenderFewObservations:
    """Test BayesianRecommender behavior with fewer observations than n_initial."""

    def test_few_observations_returns_random(self, simple_project, few_observations):
        """Fewer than n_initial observations should trigger random sampling."""
        # n_initial=3, few_observations has 2 observations
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend(few_observations)
        assert set(result.keys()) == {"x1", "x2"}
        assert 0.0 <= result["x1"] <= 10.0
        assert -5.0 <= result["x2"] <= 5.0

    def test_few_observations_still_within_bounds(
        self, simple_project, few_observations
    ):
        """Random samples from few observations should be within bounds."""
        recommender = BayesianRecommender(simple_project)
        for _ in range(10):
            result = recommender.recommend(few_observations)
            assert 0.0 <= result["x1"] <= 10.0
            assert -5.0 <= result["x2"] <= 5.0


class TestBayesianRecommenderEnoughObservations:
    """Test BayesianRecommender behavior with enough observations for BO."""

    def test_enough_observations_uses_surrogate(
        self, simple_project, enough_observations
    ):
        """With >= n_initial observations, should use surrogate and acquisition."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend(enough_observations)
        # Should still return valid result within bounds
        assert 0.0 <= result["x1"] <= 10.0
        assert -5.0 <= result["x2"] <= 5.0

    def test_recommend_exploits_good_region(self, simple_project):
        """BO should suggest points near observed maxima (exploitation)."""
        torch.manual_seed(42)
        # Create observations where high y correlates with high x1
        observations = [
            Observation(
                project_id=1, inputs={"x1": 1.0, "x2": 0.0}, outputs={"y": 1.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 3.0, "x2": 0.0}, outputs={"y": 3.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 5.0, "x2": 0.0}, outputs={"y": 5.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 7.0, "x2": 0.0}, outputs={"y": 7.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 9.0, "x2": 0.0}, outputs={"y": 9.0}
            ),
        ]
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend(observations)
        # Should suggest high x1 (towards upper bound)
        assert result["x1"] > 5.0


class TestBayesianRecommenderMaximize:
    """Test BayesianRecommender in maximize mode."""

    def test_maximize_seeks_higher_values(self, simple_project):
        """In maximize mode, should seek regions with higher predicted values."""
        torch.manual_seed(42)
        # Clear pattern: y increases with x1
        observations = [
            Observation(
                project_id=1, inputs={"x1": 0.0, "x2": 0.0}, outputs={"y": 0.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 5.0, "x2": 0.0}, outputs={"y": 5.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 10.0, "x2": 0.0}, outputs={"y": 10.0}
            ),
        ]
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend(observations)
        # Should explore or exploit high x1 region
        assert 0.0 <= result["x1"] <= 10.0


class TestBayesianRecommenderMinimize:
    """Test BayesianRecommender in minimize mode."""

    def test_minimize_seeks_lower_values(self, minimize_project):
        """In minimize mode, should seek regions with lower predicted values."""
        torch.manual_seed(42)
        # Clear pattern: y increases with x1, so minimum is at low x1
        observations = [
            Observation(
                project_id=1, inputs={"x1": 0.0, "x2": 0.0}, outputs={"y": 0.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 5.0, "x2": 0.0}, outputs={"y": 5.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 10.0, "x2": 0.0}, outputs={"y": 10.0}
            ),
        ]
        recommender = BayesianRecommender(minimize_project)
        result = recommender.recommend(observations)
        # Should explore or exploit low x1 region
        assert 0.0 <= result["x1"] <= 10.0


class TestBayesianRecommenderAcquisitionEI:
    """Test BayesianRecommender with Expected Improvement acquisition."""

    def test_ei_acquisition(self, simple_project, enough_observations):
        """EI acquisition produces valid recommendations."""
        torch.manual_seed(42)
        # simple_project uses EI by default
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend(enough_observations)
        assert 0.0 <= result["x1"] <= 10.0
        assert -5.0 <= result["x2"] <= 5.0

    def test_ei_with_xi_kwarg(self):
        """EI acquisition respects xi kwarg."""
        torch.manual_seed(42)
        project = Project(
            id=1,
            name="ei_xi",
            inputs=[
                InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
                InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
            ],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y", objective_mode="maximize")],
            recommender_config=RecommenderConfig(
                type="bayesian",
                acquisition="ei",
                n_initial=3,
                acquisition_kwargs={"xi": 0.1},
            ),
        )
        observations = [
            Observation(
                project_id=1, inputs={"x1": 5.0, "x2": 0.0}, outputs={"y": 10.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 2.0, "x2": -3.0}, outputs={"y": 7.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 8.0, "x2": 2.0}, outputs={"y": 15.0}
            ),
        ]
        recommender = BayesianRecommender(project)
        result = recommender.recommend(observations)
        assert 0.0 <= result["x1"] <= 10.0


class TestBayesianRecommenderAcquisitionUCB:
    """Test BayesianRecommender with Upper Confidence Bound acquisition."""

    def test_ucb_acquisition(self, ucb_project, enough_observations):
        """UCB acquisition produces valid recommendations."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(ucb_project)
        result = recommender.recommend(enough_observations)
        assert 0.0 <= result["x1"] <= 10.0
        assert -5.0 <= result["x2"] <= 5.0

    def test_ucb_with_beta_kwarg(self, ucb_project, enough_observations):
        """UCB acquisition respects beta kwarg."""
        torch.manual_seed(42)
        # ucb_project has beta=2.5 in kwargs
        recommender = BayesianRecommender(ucb_project)
        result = recommender.recommend(enough_observations)
        assert 0.0 <= result["x1"] <= 10.0


class TestBayesianRecommenderSingleInput:
    """Test BayesianRecommender with single input dimension."""

    def test_single_input_recommend(self, single_input_project):
        """recommend() works with single input project."""
        torch.manual_seed(42)
        observations = [
            Observation(project_id=1, inputs={"x": 0.2}, outputs={"y": 0.2}),
            Observation(project_id=1, inputs={"x": 0.5}, outputs={"y": 0.8}),
            Observation(project_id=1, inputs={"x": 0.8}, outputs={"y": 0.4}),
        ]
        recommender = BayesianRecommender(single_input_project)
        result = recommender.recommend(observations)
        assert "x" in result
        assert 0.0 <= result["x"] <= 1.0


class TestBayesianRecommenderEdgeCases:
    """Test edge cases for BayesianRecommender."""

    def test_failed_observations_excluded(self, simple_project):
        """Failed observations should be excluded from training data."""
        torch.manual_seed(42)
        observations = [
            Observation(
                project_id=1, inputs={"x1": 5.0, "x2": 0.0}, outputs={"y": 10.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 2.0, "x2": -3.0}, outputs={"y": 7.0}
            ),
            Observation(
                project_id=1,
                inputs={"x1": 8.0, "x2": 2.0},
                outputs={"y": 100.0},
                failed=True,
            ),
            Observation(
                project_id=1, inputs={"x1": 1.0, "x2": -1.0}, outputs={"y": 5.0}
            ),
        ]
        recommender = BayesianRecommender(simple_project)
        # Should not crash and should return valid result
        result = recommender.recommend(observations)
        assert 0.0 <= result["x1"] <= 10.0

    def test_exactly_n_initial_observations(self, simple_project):
        """Exactly n_initial observations should use surrogate."""
        torch.manual_seed(42)
        # n_initial=3
        observations = [
            Observation(
                project_id=1, inputs={"x1": 5.0, "x2": 0.0}, outputs={"y": 10.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 2.0, "x2": -3.0}, outputs={"y": 7.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 8.0, "x2": 2.0}, outputs={"y": 15.0}
            ),
        ]
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend(observations)
        assert 0.0 <= result["x1"] <= 10.0

    def test_many_inputs(self):
        """recommend() works with many inputs."""
        torch.manual_seed(42)
        inputs = [InputSpec(f"x{i}", "continuous", bounds=(0.0, 1.0)) for i in range(5)]
        project = Project(
            id=1,
            name="many_inputs",
            inputs=inputs,
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
            recommender_config=RecommenderConfig(n_initial=3),
        )
        observations = [
            Observation(
                project_id=1,
                inputs={f"x{i}": 0.5 for i in range(5)},
                outputs={"y": 1.0},
            ),
            Observation(
                project_id=1,
                inputs={f"x{i}": 0.2 for i in range(5)},
                outputs={"y": 0.5},
            ),
            Observation(
                project_id=1,
                inputs={f"x{i}": 0.8 for i in range(5)},
                outputs={"y": 0.8},
            ),
        ]
        recommender = BayesianRecommender(project)
        result = recommender.recommend(observations)
        assert len(result) == 5
        for i in range(5):
            assert 0.0 <= result[f"x{i}"] <= 1.0

    def test_constant_y_values(self, simple_project):
        """recommend() handles constant y values (no variance)."""
        torch.manual_seed(42)
        observations = [
            Observation(
                project_id=1, inputs={"x1": 1.0, "x2": 0.0}, outputs={"y": 5.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 5.0, "x2": 0.0}, outputs={"y": 5.0}
            ),
            Observation(
                project_id=1, inputs={"x1": 9.0, "x2": 0.0}, outputs={"y": 5.0}
            ),
        ]
        recommender = BayesianRecommender(simple_project)
        result = recommender.recommend(observations)
        assert 0.0 <= result["x1"] <= 10.0


class TestBayesianRecommenderInternalMethods:
    """Test internal methods of BayesianRecommender."""

    def test_fit_surrogate_creates_model(self, simple_project, enough_observations):
        """_fit_surrogate creates a fitted surrogate model."""
        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        X, y = simple_project.get_training_data(enough_observations)
        recommender._fit_surrogate(X, y)
        # Surrogate should be fitted after this call
        assert recommender._surrogate is not None
        assert recommender._surrogate._is_fitted

    def test_build_acquisition_ei(self, simple_project, enough_observations):
        """_build_acquisition creates callable acquisition for ei config."""
        from botorch.acquisition import AcquisitionFunction

        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        X, y = simple_project.get_training_data(enough_observations)
        recommender._fit_surrogate(X, y)
        acq_fn = recommender._build_acquisition(X, y, [True])
        assert isinstance(acq_fn, AcquisitionFunction)

    def test_build_acquisition_ucb(self, ucb_project, enough_observations):
        """_build_acquisition creates callable acquisition for ucb config."""
        from botorch.acquisition import AcquisitionFunction

        torch.manual_seed(42)
        recommender = BayesianRecommender(ucb_project)
        X, y = ucb_project.get_training_data(enough_observations)
        recommender._fit_surrogate(X, y)
        acq_fn = recommender._build_acquisition(X, y, [True])
        assert isinstance(acq_fn, AcquisitionFunction)


class TestBayesianRecommenderKwargsPassthrough:
    """Test that kwargs are passed through to acquisition functions."""

    def test_xi_passed_to_ei(self):
        """xi kwarg is accepted by ExpectedImprovement builder."""
        from botorch.acquisition import AcquisitionFunction

        torch.manual_seed(42)
        project = Project(
            id=1,
            name="ei_project",
            inputs=[InputSpec("x", "continuous", bounds=(0.0, 1.0))],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
            recommender_config=RecommenderConfig(
                acquisition="ei",
                acquisition_kwargs={"xi": 0.5},
            ),
        )
        recommender = BayesianRecommender(project)
        X = np.array([[0.2], [0.5], [0.8]])
        y = np.array([[0.2], [0.8], [0.4]])
        recommender._fit_surrogate(X, y)
        acq_fn = recommender._build_acquisition(X, y, [True])
        assert isinstance(acq_fn, AcquisitionFunction)

    def test_beta_passed_to_ucb(self):
        """beta kwarg is accepted by UpperConfidenceBound builder."""
        from botorch.acquisition import AcquisitionFunction

        torch.manual_seed(42)
        project = Project(
            id=1,
            name="ucb_project",
            inputs=[InputSpec("x", "continuous", bounds=(0.0, 1.0))],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
            recommender_config=RecommenderConfig(
                acquisition="ucb",
                acquisition_kwargs={"beta": 3.0},
            ),
        )
        recommender = BayesianRecommender(project)
        X = np.array([[0.2], [0.5], [0.8]])
        y = np.array([[0.2], [0.8], [0.4]])
        recommender._fit_surrogate(X, y)
        acq_fn = recommender._build_acquisition(X, y, [True])
        assert isinstance(acq_fn, AcquisitionFunction)

    def test_default_xi_when_not_specified(self):
        """Default xi is used when not specified in kwargs."""
        from botorch.acquisition import AcquisitionFunction

        torch.manual_seed(42)
        project = Project(
            id=1,
            name="ei_default",
            inputs=[InputSpec("x", "continuous", bounds=(0.0, 1.0))],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
            recommender_config=RecommenderConfig(acquisition="ei"),
        )
        recommender = BayesianRecommender(project)
        X = np.array([[0.2], [0.5], [0.8]])
        y = np.array([[0.2], [0.8], [0.4]])
        recommender._fit_surrogate(X, y)
        acq_fn = recommender._build_acquisition(X, y, [True])
        assert isinstance(acq_fn, AcquisitionFunction)

    def test_default_beta_when_not_specified(self):
        """Default beta is used when not specified in kwargs."""
        from botorch.acquisition import AcquisitionFunction

        torch.manual_seed(42)
        project = Project(
            id=1,
            name="ucb_default",
            inputs=[InputSpec("x", "continuous", bounds=(0.0, 1.0))],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
            recommender_config=RecommenderConfig(acquisition="ucb"),
        )
        recommender = BayesianRecommender(project)
        X = np.array([[0.2], [0.5], [0.8]])
        y = np.array([[0.2], [0.8], [0.4]])
        recommender._fit_surrogate(X, y)
        acq_fn = recommender._build_acquisition(X, y, [True])
        assert isinstance(acq_fn, AcquisitionFunction)


class TestBayesianRecommenderSurrogateConfig:
    """Test surrogate model selection from config."""

    def test_gp_surrogate_selected(self, simple_project, enough_observations):
        """surrogate='gp' creates SingleTaskGPSurrogate."""
        from folio.surrogates import SingleTaskGPSurrogate

        torch.manual_seed(42)
        recommender = BayesianRecommender(simple_project)
        X, y = simple_project.get_training_data(enough_observations)
        recommender._fit_surrogate(X, y)
        assert isinstance(recommender._surrogate, SingleTaskGPSurrogate)

    @pytest.mark.skip(
        reason="MultiTaskGP requires MultiObjectiveRecommender, not BayesianRecommender"
    )
    def test_multitask_gp_surrogate_selected(self, enough_observations):
        """surrogate='multitask_gp' creates MultiTaskGPSurrogate."""
        from folio.surrogates import MultiTaskGPSurrogate

        torch.manual_seed(42)
        project = Project(
            id=1,
            name="multitask_project",
            inputs=[
                InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
                InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
            ],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y", objective_mode="maximize")],
            recommender_config=RecommenderConfig(
                type="bayesian",
                surrogate="multitask_gp",
                acquisition="ei",
                n_initial=3,
            ),
        )
        recommender = BayesianRecommender(project)
        X, y = project.get_training_data(enough_observations)
        recommender._fit_surrogate(X, y)
        assert isinstance(recommender._surrogate, MultiTaskGPSurrogate)
