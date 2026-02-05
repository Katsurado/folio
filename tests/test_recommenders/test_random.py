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


class TestRandomRecommenderMultiObjective:
    """Test RandomRecommender with multi-objective projects."""

    @pytest.fixture
    def multi_objective_project(self):
        """Create a multi-objective project with 2 targets."""
        return Project(
            id=1,
            name="multi_objective",
            inputs=[
                InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
                InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
            ],
            outputs=[OutputSpec("y1"), OutputSpec("y2")],
            target_configs=[
                TargetConfig(objective="y1", objective_mode="maximize"),
                TargetConfig(objective="y2", objective_mode="minimize"),
            ],
            reference_point=[0.0, 10.0],
            recommender_config=RecommenderConfig(type="random"),
        )

    @pytest.fixture
    def multi_objective_observations(self):
        """Create observations for multi-objective testing."""
        return [
            Observation(
                project_id=1,
                inputs={"x1": 1.0, "x2": -2.0},
                outputs={"y1": 5.0, "y2": 8.0},
            ),
            Observation(
                project_id=1,
                inputs={"x1": 5.0, "x2": 0.0},
                outputs={"y1": 10.0, "y2": 3.0},
            ),
        ]

    def test_multi_objective_recommend_returns_dict(
        self, multi_objective_project, multi_objective_observations
    ):
        """recommend() returns dict for multi-objective project."""
        recommender = RandomRecommender(multi_objective_project)
        result = recommender.recommend(multi_objective_observations)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"x1", "x2"}

    def test_multi_objective_recommend_within_bounds(
        self, multi_objective_project, multi_objective_observations
    ):
        """recommend() returns values within bounds for multi-objective."""
        recommender = RandomRecommender(multi_objective_project)
        for _ in range(10):
            result = recommender.recommend(multi_objective_observations)
            assert 0.0 <= result["x1"] <= 10.0
            assert -5.0 <= result["x2"] <= 5.0

    def test_multi_objective_recommend_from_data(self, multi_objective_project):
        """recommend_from_data works with multi-objective arrays."""
        recommender = RandomRecommender(multi_objective_project)
        X = np.array([[1.0, -2.0], [5.0, 0.0]])
        y = np.array([[5.0, 8.0], [10.0, 3.0]])
        bounds = np.array([[0.0, -5.0], [10.0, 5.0]])
        result = recommender.recommend_from_data(X, y, bounds, [True, False])
        assert result.shape == (2,)
        assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))

    def test_multi_objective_training_data_shape(
        self, multi_objective_project, multi_objective_observations
    ):
        """get_training_data returns correct shapes for multi-objective."""
        X, y = multi_objective_project.get_training_data(multi_objective_observations)
        assert X.shape == (2, 2)
        assert y.shape == (2, 2)

    def test_multi_objective_empty_observations(self, multi_objective_project):
        """recommend() works with empty observations for multi-objective."""
        recommender = RandomRecommender(multi_objective_project)
        result = recommender.recommend([])
        assert set(result.keys()) == {"x1", "x2"}
        assert 0.0 <= result["x1"] <= 10.0
        assert -5.0 <= result["x2"] <= 5.0


# -----------------------------------------------------------------------------
# Non-optimizable (context) inputs tests
# -----------------------------------------------------------------------------


class TestRandomRecommenderNonOptimizableInputs:
    """Tests for RandomRecommender with non-optimizable (context) inputs.

    Non-optimizable inputs should be excluded from random sampling - the
    recommender only samples over optimizable dimensions.
    """

    @pytest.fixture
    def project_with_non_optimizable(self):
        """Project with optimizable and non-optimizable inputs."""
        return Project(
            id=1,
            name="context_project",
            inputs=[
                InputSpec("R", "continuous", bounds=(0.0, 255.0)),
                InputSpec("G", "continuous", bounds=(0.0, 255.0)),
                InputSpec("B", "continuous", bounds=(0.0, 255.0)),
                InputSpec("hour", "continuous", bounds=(0.0, 24.0), optimizable=False),
                InputSpec("temp", "continuous", bounds=(15.0, 35.0), optimizable=False),
            ],
            outputs=[OutputSpec("intensity")],
            target_configs=[TargetConfig(objective="intensity")],
            recommender_config=RecommenderConfig(type="random"),
        )

    def test_recommend_returns_only_optimizable_keys(
        self, project_with_non_optimizable
    ):
        """recommend() returns dict with only optimizable input names."""
        recommender = RandomRecommender(project_with_non_optimizable)
        result = recommender.recommend([], fixed_inputs={"hour": 12.0, "temp": 22.0})
        assert set(result.keys()) == {"R", "G", "B"}
        assert "hour" not in result
        assert "temp" not in result

    def test_recommend_values_within_optimizable_bounds(
        self, project_with_non_optimizable
    ):
        """recommend() returns values within optimizable input bounds."""
        recommender = RandomRecommender(project_with_non_optimizable)
        for _ in range(20):
            result = recommender.recommend(
                [], fixed_inputs={"hour": 12.0, "temp": 22.0}
            )
            assert 0.0 <= result["R"] <= 255.0
            assert 0.0 <= result["G"] <= 255.0
            assert 0.0 <= result["B"] <= 255.0

    def test_recommend_from_data_samples_optimizable_only(
        self, project_with_non_optimizable
    ):
        """recommend_from_data() samples only optimizable dimensions."""
        recommender = RandomRecommender(project_with_non_optimizable)
        # Bounds only for optimizable (R, G, B)
        bounds = np.array([[0.0, 0.0, 0.0], [255.0, 255.0, 255.0]])
        result = recommender.recommend_from_data(
            X=np.empty((0, 5)),  # 5 total features in training
            y=np.empty(0),
            bounds=bounds,  # Only 3 optimizable
            maximize=[True],
        )
        # Result matches optimizable dimensions
        assert result.shape == (3,)
        assert np.all((bounds[0, :] <= result) & (result <= bounds[1, :]))

    def test_non_optimizable_ignored_for_sampling(self, project_with_non_optimizable):
        """Non-optimizable values don't affect random sampling."""
        recommender = RandomRecommender(project_with_non_optimizable)
        observations = [
            Observation(
                project_id=1,
                inputs={
                    "R": 100.0,
                    "G": 150.0,
                    "B": 200.0,
                    "hour": 10.0,
                    "temp": 22.0,
                },
                outputs={"intensity": 0.75},
            ),
        ]
        result = recommender.recommend(
            observations, fixed_inputs={"hour": 12.0, "temp": 22.0}
        )
        # Still only optimizable keys
        assert set(result.keys()) == {"R", "G", "B"}
