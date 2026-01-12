"""Tests for RandomRecommender."""

import numpy as np
import pytest

from folio.core.config import RecommenderConfig, TargetConfig
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.recommenders.random import RandomRecommender


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
        recommender_config=RecommenderConfig(type="random"),
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
    )


@pytest.fixture
def sample_observations():
    """Create sample observations for testing."""
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


class TestRandomRecommenderInit:
    """Test RandomRecommender initialization."""

    def test_init_with_project(self, simple_project):
        """RandomRecommender can be initialized with a Project."""
        recommender = RandomRecommender(simple_project)
        assert recommender.project is simple_project

    def test_init_stores_project(self, simple_project):
        """RandomRecommender stores the project reference."""
        recommender = RandomRecommender(simple_project)
        assert recommender.project.name == "test_project"


class TestRandomRecommenderRecommend:
    """Test RandomRecommender.recommend method."""

    def test_recommend_returns_dict(self, simple_project, sample_observations):
        """recommend() returns a dictionary."""
        recommender = RandomRecommender(simple_project)
        result = recommender.recommend(sample_observations)
        assert isinstance(result, dict)

    def test_recommend_has_correct_keys(self, simple_project, sample_observations):
        """recommend() returns dict with all input names as keys."""
        recommender = RandomRecommender(simple_project)
        result = recommender.recommend(sample_observations)
        assert set(result.keys()) == {"x1", "x2"}

    def test_recommend_values_within_bounds(self, simple_project, sample_observations):
        """recommend() returns values within input bounds."""
        recommender = RandomRecommender(simple_project)
        # Test multiple times for randomness
        for _ in range(20):
            result = recommender.recommend(sample_observations)
            assert 0.0 <= result["x1"] <= 10.0
            assert -5.0 <= result["x2"] <= 5.0

    def test_recommend_with_empty_observations(self, simple_project):
        """recommend() works with empty observation list."""
        recommender = RandomRecommender(simple_project)
        result = recommender.recommend([])
        assert set(result.keys()) == {"x1", "x2"}
        assert 0.0 <= result["x1"] <= 10.0
        assert -5.0 <= result["x2"] <= 5.0

    def test_recommend_ignores_observations(self, simple_project):
        """RandomRecommender ignores observations (samples randomly regardless)."""
        recommender = RandomRecommender(simple_project)
        # Should work the same with any observations
        result1 = recommender.recommend([])
        result2 = recommender.recommend(
            [
                Observation(
                    project_id=1, inputs={"x1": 1.0, "x2": 1.0}, outputs={"y": 100.0}
                )
            ]
        )
        # Both should be valid (within bounds) - randomness means they'll differ
        assert 0.0 <= result1["x1"] <= 10.0
        assert 0.0 <= result2["x1"] <= 10.0

    def test_recommend_single_input(self, single_input_project):
        """recommend() works with single input project."""
        recommender = RandomRecommender(single_input_project)
        result = recommender.recommend([])
        assert "x" in result
        assert 0.0 <= result["x"] <= 1.0

    def test_recommend_returns_float_values(self, simple_project):
        """recommend() returns float values for continuous inputs."""
        recommender = RandomRecommender(simple_project)
        result = recommender.recommend([])
        assert isinstance(result["x1"], float)
        assert isinstance(result["x2"], float)


class TestRandomRecommenderRecommendFromData:
    """Test RandomRecommender.recommend_from_data method."""

    def test_recommend_from_data_returns_array(self, simple_project):
        """recommend_from_data() returns a numpy array."""
        recommender = RandomRecommender(simple_project)
        # Bounds shape (2, d): row 0 = lower, row 1 = upper
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = recommender.recommend_from_data(
            X=np.empty((0, 2)),
            y=np.empty(0),
            bounds=bounds,
            maximize=[True],
        )
        assert isinstance(result, np.ndarray)

    def test_recommend_from_data_correct_shape(self, simple_project):
        """recommend_from_data() returns array with shape (n_features,)."""
        recommender = RandomRecommender(simple_project)
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = recommender.recommend_from_data(
            X=np.empty((0, 2)),
            y=np.empty(0),
            bounds=bounds,
            maximize=[True],
        )
        assert result.shape == (2,)

    def test_recommend_from_data_within_bounds(self, simple_project):
        """recommend_from_data() returns values within bounds."""
        recommender = RandomRecommender(simple_project)
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        for _ in range(20):
            result = recommender.recommend_from_data(
                X=np.empty((0, 2)),
                y=np.empty(0),
                bounds=bounds,
                maximize=[True],
            )
            assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))

    def test_recommend_from_data_ignores_X_y(self, simple_project):
        """recommend_from_data() ignores X and y (samples randomly)."""
        recommender = RandomRecommender(simple_project)
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        X = np.array([[5.0, 0.0], [2.0, -3.0]])
        y = np.array([10.0, 7.0])
        result = recommender.recommend_from_data(
            X=X, y=y, bounds=bounds, maximize=[True]
        )
        # Should still return valid result within bounds
        assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))

    def test_recommend_from_data_ignores_objective(self, simple_project):
        """recommend_from_data() ignores objective (samples randomly)."""
        recommender = RandomRecommender(simple_project)
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        # Both objectives should produce valid results
        result_max = recommender.recommend_from_data(
            X=np.empty((0, 2)),
            y=np.empty(0),
            bounds=bounds,
            maximize=[True],
        )
        result_min = recommender.recommend_from_data(
            X=np.empty((0, 2)),
            y=np.empty(0),
            bounds=bounds,
            maximize=[False],
        )
        assert np.all((bounds[0, :] <= result_max) & (result_max <= bounds[1, :]))
        assert np.all((bounds[0, :] <= result_min) & (result_min <= bounds[1, :]))

    def test_recommend_from_data_single_dimension(self, single_input_project):
        """recommend_from_data() works with single dimension."""
        recommender = RandomRecommender(single_input_project)
        # Shape (2, 1) for single dimension
        bounds = np.array([[0.0], [1.0]])
        result = recommender.recommend_from_data(
            X=np.empty((0, 1)),
            y=np.empty(0),
            bounds=bounds,
            maximize=[True],
        )
        assert result.shape == (1,)
        assert 0.0 <= result[0] <= 1.0

    def test_recommend_from_data_many_dimensions(self):
        """recommend_from_data() works with many dimensions."""
        inputs = [
            InputSpec(f"x{i}", "continuous", bounds=(0.0, 1.0)) for i in range(10)
        ]
        project = Project(
            id=1,
            name="many_inputs",
            inputs=inputs,
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        recommender = RandomRecommender(project)
        # Shape (2, 10) for 10 dimensions
        bounds = np.array(
            [
                [0.0] * 10,
                [1.0] * 10,
            ]
        )
        result = recommender.recommend_from_data(
            X=np.empty((0, 10)),
            y=np.empty(0),
            bounds=bounds,
            maximize=[True],
        )
        assert result.shape == (10,)
        assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))


class TestRandomRecommenderRandomness:
    """Test that RandomRecommender produces varied samples."""

    def test_multiple_calls_produce_different_results(self, simple_project):
        """Multiple calls should produce different samples (with high probability)."""
        recommender = RandomRecommender(simple_project)
        results = [recommender.recommend([]) for _ in range(10)]
        x1_values = [r["x1"] for r in results]
        # At least some values should differ
        assert len(set(x1_values)) > 1

    def test_samples_cover_range(self, single_input_project):
        """Samples should cover the input range over many calls."""
        recommender = RandomRecommender(single_input_project)
        samples = [recommender.recommend([])["x"] for _ in range(100)]
        # Should have samples across the range
        assert min(samples) < 0.3
        assert max(samples) > 0.7

    def test_recommend_from_data_multiple_calls_differ(self, simple_project):
        """recommend_from_data() should produce different samples."""
        recommender = RandomRecommender(simple_project)
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        results = [
            recommender.recommend_from_data(
                X=np.empty((0, 2)), y=np.empty(0), bounds=bounds, maximize=[True]
            )
            for _ in range(10)
        ]
        x1_values = [r[0] for r in results]
        assert len(set(x1_values)) > 1


class TestRandomRecommenderEdgeCases:
    """Test edge cases for RandomRecommender."""

    def test_narrow_bounds(self):
        """recommend() works with very narrow bounds."""
        project = Project(
            id=1,
            name="narrow",
            inputs=[InputSpec("x", "continuous", bounds=(0.5, 0.50001))],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        recommender = RandomRecommender(project)
        result = recommender.recommend([])
        assert 0.5 <= result["x"] <= 0.50001

    def test_negative_bounds(self):
        """recommend() works with negative bounds."""
        project = Project(
            id=1,
            name="negative",
            inputs=[InputSpec("x", "continuous", bounds=(-100.0, -50.0))],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        recommender = RandomRecommender(project)
        result = recommender.recommend([])
        assert -100.0 <= result["x"] <= -50.0

    def test_large_bounds(self):
        """recommend() works with large bounds."""
        project = Project(
            id=1,
            name="large",
            inputs=[InputSpec("x", "continuous", bounds=(0.0, 1e6))],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        recommender = RandomRecommender(project)
        result = recommender.recommend([])
        assert 0.0 <= result["x"] <= 1e6

    def test_many_inputs(self):
        """recommend() works with many inputs."""
        inputs = [
            InputSpec(f"x{i}", "continuous", bounds=(0.0, 1.0)) for i in range(10)
        ]
        project = Project(
            id=1,
            name="many_inputs",
            inputs=inputs,
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        recommender = RandomRecommender(project)
        result = recommender.recommend([])
        assert len(result) == 10
        for i in range(10):
            assert f"x{i}" in result
            assert 0.0 <= result[f"x{i}"] <= 1.0
