"""Tests for Recommender abstract base class."""

import numpy as np
import pytest

from folio.core.config import RecommenderConfig, TargetConfig
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.recommenders.base import Recommender


class ConcreteRecommender(Recommender):
    """Minimal concrete implementation for testing the interface contract.

    This mock recommender records calls to recommend_from_data and returns
    a predictable result for testing the base class logic.
    """

    def __init__(self, project: Project) -> None:
        super().__init__(project)
        self.last_X = None
        self.last_y = None
        self.last_bounds = None
        self.last_objective = None
        self.call_count = 0

    def recommend_from_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        objective: str,
    ) -> np.ndarray:
        """Record arguments and return midpoint of bounds.

        Note: bounds has shape (2, d) where row 0 is lower bounds,
        row 1 is upper bounds.
        """
        self.last_X = X
        self.last_y = y
        self.last_bounds = bounds
        self.last_objective = objective
        self.call_count += 1

        # Return midpoint of each dimension
        # bounds[0, :] = lower bounds, bounds[1, :] = upper bounds
        return (bounds[0, :] + bounds[1, :]) / 2


@pytest.fixture
def simple_project():
    """Create a simple project with two continuous inputs."""
    return Project(
        id=1,
        name="test_project",
        inputs=[
            InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
            InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
        ],
        outputs=[OutputSpec("y")],
        target_config=TargetConfig("y", mode="maximize"),
        recommender_config=RecommenderConfig(),
    )


@pytest.fixture
def sample_observations():
    """Create sample observations for testing."""
    return [
        Observation(
            project_id=1,
            inputs={"x1": 2.0, "x2": -2.0},
            outputs={"y": 5.0},
        ),
        Observation(
            project_id=1,
            inputs={"x1": 8.0, "x2": 3.0},
            outputs={"y": 12.0},
        ),
        Observation(
            project_id=1,
            inputs={"x1": 5.0, "x2": 0.0},
            outputs={"y": 8.0},
        ),
    ]


class TestRandomSampleFromBounds:
    """Test the random_sample_from_bounds static method."""

    def test_returns_array(self):
        """random_sample_from_bounds returns numpy array."""
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = Recommender.random_sample_from_bounds(bounds)
        assert isinstance(result, np.ndarray)

    def test_correct_shape(self):
        """random_sample_from_bounds returns shape (d,)."""
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = Recommender.random_sample_from_bounds(bounds)
        assert result.shape == (2,)

    def test_within_bounds(self):
        """random_sample_from_bounds returns values within bounds."""
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        for _ in range(20):
            result = Recommender.random_sample_from_bounds(bounds)
            assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))

    def test_single_dimension(self):
        """random_sample_from_bounds works with single dimension."""
        bounds = np.array([[0.0], [1.0]])
        result = Recommender.random_sample_from_bounds(bounds)
        assert result.shape == (1,)
        assert 0.0 <= result[0] <= 1.0

    def test_many_dimensions(self):
        """random_sample_from_bounds works with many dimensions."""
        bounds = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        result = Recommender.random_sample_from_bounds(bounds)
        assert result.shape == (5,)
        assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))

    def test_negative_bounds(self):
        """random_sample_from_bounds works with negative bounds."""
        bounds = np.array([[-100.0], [-50.0]])
        for _ in range(10):
            result = Recommender.random_sample_from_bounds(bounds)
            assert -100.0 <= result[0] <= -50.0

    def test_produces_varied_samples(self):
        """random_sample_from_bounds produces different samples."""
        bounds = np.array([[0.0], [1.0]])
        samples = [Recommender.random_sample_from_bounds(bounds)[0] for _ in range(10)]
        assert len(set(samples)) > 1


class TestRecommenderABC:
    """Test that Recommender enforces the abstract interface."""

    def test_cannot_instantiate_abstract_class(self, simple_project):
        """Recommender ABC cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Recommender(simple_project)

    def test_must_implement_recommend_from_data(self, simple_project):
        """Subclass without recommend_from_data raises TypeError."""

        class IncompleteRecommender(Recommender):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteRecommender(simple_project)


class TestRecommenderInit:
    """Test Recommender initialization."""

    def test_init_stores_project(self, simple_project):
        """__init__ stores the project reference."""
        recommender = ConcreteRecommender(simple_project)
        assert recommender.project is simple_project

    def test_init_project_accessible(self, simple_project):
        """Project attributes are accessible after init."""
        recommender = ConcreteRecommender(simple_project)
        assert recommender.project.name == "test_project"
        assert len(recommender.project.inputs) == 2


class TestRecommendMethod:
    """Test the recommend() method implemented in the base class."""

    def test_recommend_returns_dict(self, simple_project, sample_observations):
        """recommend() returns a dictionary."""
        recommender = ConcreteRecommender(simple_project)
        result = recommender.recommend(sample_observations)
        assert isinstance(result, dict)

    def test_recommend_has_correct_keys(self, simple_project, sample_observations):
        """recommend() returns dict with all input names as keys."""
        recommender = ConcreteRecommender(simple_project)
        result = recommender.recommend(sample_observations)
        assert set(result.keys()) == {"x1", "x2"}

    def test_recommend_values_are_floats(self, simple_project, sample_observations):
        """recommend() returns float values."""
        recommender = ConcreteRecommender(simple_project)
        result = recommender.recommend(sample_observations)
        assert isinstance(result["x1"], float)
        assert isinstance(result["x2"], float)

    def test_recommend_calls_recommend_from_data(
        self, simple_project, sample_observations
    ):
        """recommend() delegates to recommend_from_data."""
        recommender = ConcreteRecommender(simple_project)
        recommender.recommend(sample_observations)
        assert recommender.call_count == 1

    def test_recommend_extracts_X_correctly(self, simple_project, sample_observations):
        """recommend() extracts X array from observations."""
        recommender = ConcreteRecommender(simple_project)
        recommender.recommend(sample_observations)

        expected_X = np.array(
            [
                [2.0, -2.0],
                [8.0, 3.0],
                [5.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(recommender.last_X, expected_X)

    def test_recommend_extracts_y_correctly(self, simple_project, sample_observations):
        """recommend() extracts y array from observations using target config."""
        recommender = ConcreteRecommender(simple_project)
        recommender.recommend(sample_observations)

        expected_y = np.array([5.0, 12.0, 8.0])
        np.testing.assert_array_equal(recommender.last_y, expected_y)

    def test_recommend_extracts_bounds_correctly(
        self, simple_project, sample_observations
    ):
        """recommend() extracts bounds from project.

        Bounds have shape (2, d) where row 0 is lower bounds, row 1 is upper bounds.
        """
        recommender = ConcreteRecommender(simple_project)
        recommender.recommend(sample_observations)

        # Shape (2, d): row 0 = lower bounds, row 1 = upper bounds
        expected_bounds = np.array(
            [
                [0.0, -5.0],
                [10.0, 5.0],
            ]
        )
        np.testing.assert_array_equal(recommender.last_bounds, expected_bounds)

    def test_recommend_passes_objective_correctly(
        self, simple_project, sample_observations
    ):
        """recommend() passes objective from target config."""
        recommender = ConcreteRecommender(simple_project)
        recommender.recommend(sample_observations)
        assert recommender.last_objective == "maximize"

    def test_recommend_passes_minimize_objective(self, sample_observations):
        """recommend() passes 'minimize' objective when configured."""
        project = Project(
            id=1,
            name="minimize_project",
            inputs=[
                InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
                InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
            ],
            outputs=[OutputSpec("y")],
            target_config=TargetConfig("y", mode="minimize"),
        )
        recommender = ConcreteRecommender(project)
        recommender.recommend(sample_observations)
        assert recommender.last_objective == "minimize"

    def test_recommend_converts_array_to_dict(
        self, simple_project, sample_observations
    ):
        """recommend() converts recommend_from_data array result to dict."""
        recommender = ConcreteRecommender(simple_project)
        result = recommender.recommend(sample_observations)

        # ConcreteRecommender returns midpoint of bounds
        # x1: (0.0 + 10.0) / 2 = 5.0
        # x2: (-5.0 + 5.0) / 2 = 0.0
        assert result["x1"] == 5.0
        assert result["x2"] == 0.0


class TestRecommendWithEmptyObservations:
    """Test recommend() behavior with empty observations."""

    def test_recommend_with_empty_list(self, simple_project):
        """recommend() works with empty observation list."""
        recommender = ConcreteRecommender(simple_project)
        result = recommender.recommend([])
        assert set(result.keys()) == {"x1", "x2"}

    def test_empty_observations_passes_empty_X(self, simple_project):
        """Empty observations results in empty X array."""
        recommender = ConcreteRecommender(simple_project)
        recommender.recommend([])

        # np.array([]) produces shape (0,) not (0, n_features)
        assert recommender.last_X.shape == (0,)

    def test_empty_observations_passes_empty_y(self, simple_project):
        """Empty observations results in empty y array."""
        recommender = ConcreteRecommender(simple_project)
        recommender.recommend([])

        assert recommender.last_y.shape == (0,)


class TestRecommendWithFailedObservations:
    """Test that failed observations are excluded."""

    def test_failed_observations_excluded_from_X(self, simple_project):
        """Failed observations are not included in X."""
        observations = [
            Observation(
                project_id=1,
                inputs={"x1": 2.0, "x2": -2.0},
                outputs={"y": 5.0},
            ),
            Observation(
                project_id=1,
                inputs={"x1": 8.0, "x2": 3.0},
                outputs={"y": 100.0},
                failed=True,
            ),
            Observation(
                project_id=1,
                inputs={"x1": 5.0, "x2": 0.0},
                outputs={"y": 8.0},
            ),
        ]
        recommender = ConcreteRecommender(simple_project)
        recommender.recommend(observations)

        # Only 2 non-failed observations
        assert recommender.last_X.shape == (2, 2)
        expected_X = np.array(
            [
                [2.0, -2.0],
                [5.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(recommender.last_X, expected_X)

    def test_failed_observations_excluded_from_y(self, simple_project):
        """Failed observations are not included in y."""
        observations = [
            Observation(
                project_id=1,
                inputs={"x1": 2.0, "x2": -2.0},
                outputs={"y": 5.0},
            ),
            Observation(
                project_id=1,
                inputs={"x1": 8.0, "x2": 3.0},
                outputs={"y": 100.0},
                failed=True,
            ),
            Observation(
                project_id=1,
                inputs={"x1": 5.0, "x2": 0.0},
                outputs={"y": 8.0},
            ),
        ]
        recommender = ConcreteRecommender(simple_project)
        recommender.recommend(observations)

        expected_y = np.array([5.0, 8.0])
        np.testing.assert_array_equal(recommender.last_y, expected_y)


class TestRecommendSingleInput:
    """Test recommend() with single input dimension."""

    def test_single_input_returns_correct_keys(self):
        """recommend() returns correct key for single input."""
        project = Project(
            id=1,
            name="single_input",
            inputs=[InputSpec("x", "continuous", bounds=(0.0, 1.0))],
            outputs=[OutputSpec("y")],
            target_config=TargetConfig("y"),
        )
        observations = [
            Observation(project_id=1, inputs={"x": 0.5}, outputs={"y": 1.0}),
        ]
        recommender = ConcreteRecommender(project)
        result = recommender.recommend(observations)

        assert set(result.keys()) == {"x"}

    def test_single_input_extracts_1d_X(self):
        """recommend() extracts X with shape (n, 1) for single input."""
        project = Project(
            id=1,
            name="single_input",
            inputs=[InputSpec("x", "continuous", bounds=(0.0, 1.0))],
            outputs=[OutputSpec("y")],
            target_config=TargetConfig("y"),
        )
        observations = [
            Observation(project_id=1, inputs={"x": 0.2}, outputs={"y": 1.0}),
            Observation(project_id=1, inputs={"x": 0.8}, outputs={"y": 2.0}),
        ]
        recommender = ConcreteRecommender(project)
        recommender.recommend(observations)

        assert recommender.last_X.shape == (2, 1)
        # Bounds shape is (2, d) where d=1
        assert recommender.last_bounds.shape == (2, 1)


class TestRecommendManyInputs:
    """Test recommend() with many input dimensions."""

    def test_many_inputs_returns_all_keys(self):
        """recommend() returns all input names for many inputs."""
        inputs = [InputSpec(f"x{i}", "continuous", bounds=(0.0, 1.0)) for i in range(5)]
        project = Project(
            id=1,
            name="many_inputs",
            inputs=inputs,
            outputs=[OutputSpec("y")],
            target_config=TargetConfig("y"),
        )
        observations = [
            Observation(
                project_id=1,
                inputs={f"x{i}": 0.5 for i in range(5)},
                outputs={"y": 1.0},
            ),
        ]
        recommender = ConcreteRecommender(project)
        result = recommender.recommend(observations)

        assert len(result) == 5
        for i in range(5):
            assert f"x{i}" in result

    def test_many_inputs_extracts_correct_shape(self):
        """recommend() extracts X with correct shape for many inputs."""
        inputs = [InputSpec(f"x{i}", "continuous", bounds=(0.0, 1.0)) for i in range(5)]
        project = Project(
            id=1,
            name="many_inputs",
            inputs=inputs,
            outputs=[OutputSpec("y")],
            target_config=TargetConfig("y"),
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
        ]
        recommender = ConcreteRecommender(project)
        recommender.recommend(observations)

        assert recommender.last_X.shape == (2, 5)
        # Bounds shape is (2, d) where d=5
        assert recommender.last_bounds.shape == (2, 5)
