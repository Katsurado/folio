import numpy as np
import pytest

from folio.core.observation import Observation
from folio.core.project import Project, TargetConfig
from folio.core.schema import InputSpec, OutputSpec
from folio.exceptions import InvalidInputError, InvalidOutputError


@pytest.fixture
def sample_project():
    """Project with one continuous input, one categorical input, and one output."""
    return Project(
        id=None,
        name="test_project",
        inputs=[
            InputSpec(name="temperature", type="continuous", bounds=(20.0, 100.0)),
            InputSpec(name="solvent", type="categorical", levels=["water", "ethanol"]),
        ],
        outputs=[OutputSpec(name="yield")],
        target_configs=[TargetConfig(objective="yield")],
    )


class TestValidateInputs:
    def test_valid_continuous_within_bounds(self, sample_project):
        sample_project.validate_inputs({"temperature": 50.0, "solvent": "water"})

    def test_valid_continuous_at_lower_bound(self, sample_project):
        sample_project.validate_inputs({"temperature": 20.0, "solvent": "water"})

    def test_valid_continuous_at_upper_bound(self, sample_project):
        sample_project.validate_inputs({"temperature": 100.0, "solvent": "water"})

    def test_valid_categorical_value(self, sample_project):
        sample_project.validate_inputs({"temperature": 50.0, "solvent": "ethanol"})

    def test_categorical_value_not_in_levels_raises(self, sample_project):
        with pytest.raises(InvalidInputError, match="not in levels"):
            sample_project.validate_inputs({"temperature": 50.0, "solvent": "acetone"})

    def test_unknown_input_name_raises(self, sample_project):
        with pytest.raises(InvalidInputError, match="Unexpected inputs"):
            sample_project.validate_inputs(
                {
                    "temperature": 50.0,
                    "solvent": "water",
                    "pressure": 1.0,
                }
            )

    def test_missing_input_raises(self, sample_project):
        with pytest.raises(InvalidInputError, match="Missing inputs"):
            sample_project.validate_inputs({"temperature": 50.0})

    def test_continuous_below_bounds_raises(self, sample_project):
        with pytest.raises(InvalidInputError, match="outside bounds"):
            sample_project.validate_inputs({"temperature": 10.0, "solvent": "water"})

    def test_continuous_above_bounds_raises(self, sample_project):
        with pytest.raises(InvalidInputError, match="outside bounds"):
            sample_project.validate_inputs({"temperature": 150.0, "solvent": "water"})

    def test_continuous_wrong_type_raises(self, sample_project):
        with pytest.raises(InvalidInputError, match="must be numeric"):
            sample_project.validate_inputs({"temperature": "hot", "solvent": "water"})


class TestValidateOutputs:
    def test_valid_outputs(self, sample_project):
        sample_project.validate_outputs({"yield": 85.5})

    def test_missing_output_raises(self, sample_project):
        with pytest.raises(InvalidOutputError, match="Missing outputs"):
            sample_project.validate_outputs({})

    def test_unexpected_output_raises(self, sample_project):
        with pytest.raises(InvalidOutputError, match="Unexpected outputs"):
            sample_project.validate_outputs({"yield": 85.5, "purity": 99.0})

    def test_non_numeric_output_raises(self, sample_project):
        with pytest.raises(InvalidOutputError, match="must be numeric"):
            sample_project.validate_outputs({"yield": "high"})


class TestGetOptimizationBounds:
    """Test get_optimization_bounds method."""

    def test_returns_ndarray(self):
        """get_optimization_bounds returns a numpy array."""
        project = Project(
            id=None,
            name="test",
            inputs=[InputSpec("x", "continuous", bounds=(0.0, 1.0))],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_optimization_bounds()
        assert isinstance(result, np.ndarray)

    def test_shape_single_input(self):
        """Shape is (2, 1) for single continuous input."""
        project = Project(
            id=None,
            name="test",
            inputs=[InputSpec("x", "continuous", bounds=(0.0, 1.0))],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_optimization_bounds()
        assert result.shape == (2, 1)

    def test_shape_multiple_inputs(self):
        """Shape is (2, d) for d continuous inputs."""
        project = Project(
            id=None,
            name="test",
            inputs=[
                InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
                InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
                InputSpec("x3", "continuous", bounds=(100.0, 200.0)),
            ],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_optimization_bounds()
        assert result.shape == (2, 3)

    def test_row_0_is_lower_bounds(self):
        """Row 0 contains lower bounds."""
        project = Project(
            id=None,
            name="test",
            inputs=[
                InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
                InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
            ],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_optimization_bounds()
        np.testing.assert_array_equal(result[0, :], [0.0, -5.0])

    def test_row_1_is_upper_bounds(self):
        """Row 1 contains upper bounds."""
        project = Project(
            id=None,
            name="test",
            inputs=[
                InputSpec("x1", "continuous", bounds=(0.0, 10.0)),
                InputSpec("x2", "continuous", bounds=(-5.0, 5.0)),
            ],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_optimization_bounds()
        np.testing.assert_array_equal(result[1, :], [10.0, 5.0])

    def test_excludes_categorical_inputs(self, sample_project):
        """Categorical inputs are excluded from bounds."""
        # sample_project has 1 continuous (temperature) and 1 categorical (solvent)
        result = sample_project.get_optimization_bounds()
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result[0, :], [20.0])
        np.testing.assert_array_equal(result[1, :], [100.0])

    def test_preserves_input_order(self):
        """Bounds are in input definition order."""
        project = Project(
            id=None,
            name="test",
            inputs=[
                InputSpec("a", "continuous", bounds=(1.0, 2.0)),
                InputSpec("b", "continuous", bounds=(3.0, 4.0)),
                InputSpec("c", "continuous", bounds=(5.0, 6.0)),
            ],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_optimization_bounds()
        np.testing.assert_array_equal(result[0, :], [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(result[1, :], [2.0, 4.0, 6.0])

    def test_mixed_continuous_categorical_order(self):
        """Continuous inputs extracted in order, skipping categorical."""
        project = Project(
            id=None,
            name="test",
            inputs=[
                InputSpec("x1", "continuous", bounds=(0.0, 1.0)),
                InputSpec("cat1", "categorical", levels=["a", "b"]),
                InputSpec("x2", "continuous", bounds=(2.0, 3.0)),
                InputSpec("cat2", "categorical", levels=["c", "d"]),
                InputSpec("x3", "continuous", bounds=(4.0, 5.0)),
            ],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_optimization_bounds()
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result[0, :], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(result[1, :], [1.0, 3.0, 5.0])

    def test_negative_bounds(self):
        """Handles negative bounds correctly."""
        project = Project(
            id=None,
            name="test",
            inputs=[InputSpec("x", "continuous", bounds=(-100.0, -50.0))],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_optimization_bounds()
        np.testing.assert_array_equal(result, [[-100.0], [-50.0]])

    def test_large_bounds(self):
        """Handles large bounds correctly."""
        project = Project(
            id=None,
            name="test",
            inputs=[InputSpec("x", "continuous", bounds=(0.0, 1e6))],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_optimization_bounds()
        np.testing.assert_array_equal(result, [[0.0], [1e6]])


# -----------------------------------------------------------------------------
# Non-optimizable (context) inputs tests
# -----------------------------------------------------------------------------


@pytest.fixture
def project_with_non_optimizable():
    """Project with optimizable RGB inputs and non-optimizable context inputs."""
    return Project(
        id=1,
        name="claude_light_with_context",
        inputs=[
            InputSpec("R", "continuous", bounds=(0.0, 255.0)),
            InputSpec("G", "continuous", bounds=(0.0, 255.0)),
            InputSpec("B", "continuous", bounds=(0.0, 255.0)),
            InputSpec("hour", "continuous", bounds=(0.0, 24.0), optimizable=False),
            InputSpec(
                "ambient_temp", "continuous", bounds=(15.0, 35.0), optimizable=False
            ),
        ],
        outputs=[OutputSpec("intensity")],
        target_configs=[TargetConfig(objective="intensity", objective_mode="maximize")],
    )


@pytest.fixture
def observations_with_non_optimizable(project_with_non_optimizable):
    """Observations including non-optimizable context values."""
    return [
        Observation(
            project_id=1,
            inputs={
                "R": 100.0,
                "G": 150.0,
                "B": 200.0,
                "hour": 10.0,
                "ambient_temp": 22.0,
            },
            outputs={"intensity": 0.75},
        ),
        Observation(
            project_id=1,
            inputs={
                "R": 200.0,
                "G": 100.0,
                "B": 50.0,
                "hour": 14.0,
                "ambient_temp": 25.0,
            },
            outputs={"intensity": 0.60},
        ),
        Observation(
            project_id=1,
            inputs={
                "R": 50.0,
                "G": 200.0,
                "B": 150.0,
                "hour": 18.0,
                "ambient_temp": 20.0,
            },
            outputs={"intensity": 0.85},
        ),
    ]


class TestGetOptimizableInputs:
    """Tests for Project.get_optimizable_inputs() method."""

    def test_returns_list_of_input_specs(self, project_with_non_optimizable):
        """get_optimizable_inputs returns list of InputSpec."""
        result = project_with_non_optimizable.get_optimizable_inputs()
        assert isinstance(result, list)
        assert all(isinstance(inp, InputSpec) for inp in result)

    def test_excludes_non_optimizable_inputs(self, project_with_non_optimizable):
        """get_optimizable_inputs excludes inputs marked as non-optimizable."""
        result = project_with_non_optimizable.get_optimizable_inputs()
        names = [inp.name for inp in result]
        assert names == ["R", "G", "B"]
        assert "hour" not in names
        assert "ambient_temp" not in names

    def test_all_optimizable_when_none_marked(self, sample_project):
        """All continuous inputs returned when none are non-optimizable."""
        result = sample_project.get_optimizable_inputs()
        # sample_project has temperature (continuous) and solvent (categorical)
        names = [inp.name for inp in result]
        assert "temperature" in names

    def test_empty_when_all_non_optimizable(self):
        """Returns empty list when all inputs are non-optimizable."""
        project = Project(
            id=1,
            name="all_non_optimizable",
            inputs=[
                InputSpec("hour", "continuous", bounds=(0.0, 24.0), optimizable=False),
                InputSpec("temp", "continuous", bounds=(0.0, 40.0), optimizable=False),
            ],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_optimizable_inputs()
        assert result == []


class TestGetNonOptimizableInputs:
    """Tests for Project.get_non_optimizable_inputs() method."""

    def test_returns_list_of_input_specs(self, project_with_non_optimizable):
        """get_non_optimizable_inputs returns list of InputSpec."""
        result = project_with_non_optimizable.get_non_optimizable_inputs()
        assert isinstance(result, list)
        assert all(isinstance(inp, InputSpec) for inp in result)

    def test_returns_only_non_optimizable_inputs(self, project_with_non_optimizable):
        """get_non_optimizable_inputs returns only inputs marked as non-optimizable."""
        result = project_with_non_optimizable.get_non_optimizable_inputs()
        names = [inp.name for inp in result]
        assert names == ["hour", "ambient_temp"]
        assert "R" not in names

    def test_empty_when_all_optimizable(self, sample_project):
        """Returns empty list when no inputs are non-optimizable."""
        result = sample_project.get_non_optimizable_inputs()
        assert result == []

    def test_preserves_order(self, project_with_non_optimizable):
        """Non-optimizable inputs are returned in definition order."""
        result = project_with_non_optimizable.get_non_optimizable_inputs()
        names = [inp.name for inp in result]
        assert names == ["hour", "ambient_temp"]


class TestGetOptimizationBoundsWithNonOptimizable:
    """Tests for get_optimization_bounds with non-optimizable inputs."""

    def test_excludes_non_optimizable_from_bounds(self, project_with_non_optimizable):
        """Optimization bounds exclude non-optimizable inputs."""
        result = project_with_non_optimizable.get_optimization_bounds()
        # Only R, G, B (3 optimizable inputs)
        assert result.shape == (2, 3)

    def test_bounds_values_correct(self, project_with_non_optimizable):
        """Bounds values are correct for optimizable inputs only."""
        result = project_with_non_optimizable.get_optimization_bounds()
        # R, G, B all have bounds (0, 255)
        np.testing.assert_array_equal(result[0, :], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result[1, :], [255.0, 255.0, 255.0])

    def test_mixed_order_preserved(self):
        """Optimizable inputs extracted in order, skipping non-optimizable."""
        project = Project(
            id=1,
            name="mixed",
            inputs=[
                InputSpec("x1", "continuous", bounds=(0.0, 1.0)),
                InputSpec("ctx1", "continuous", bounds=(0.0, 10.0), optimizable=False),
                InputSpec("x2", "continuous", bounds=(2.0, 3.0)),
                InputSpec("ctx2", "continuous", bounds=(0.0, 20.0), optimizable=False),
                InputSpec("x3", "continuous", bounds=(4.0, 5.0)),
            ],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_optimization_bounds()
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result[0, :], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(result[1, :], [1.0, 3.0, 5.0])


class TestGetTrainingDataWithNonOptimizable:
    """Tests for get_training_data with non-optimizable inputs.

    The GP should be trained on ALL continuous features (optimizable + non-optimizable)
    so it learns the effect of context variables. The key difference is that
    optimization bounds only cover optimizable inputs.
    """

    def test_x_includes_non_optimizable_features(
        self, project_with_non_optimizable, observations_with_non_optimizable
    ):
        """Training X includes both optimizable and non-optimizable features."""
        X, y = project_with_non_optimizable.get_training_data(
            observations_with_non_optimizable
        )
        # R, G, B + hour + ambient_temp = 5 features
        assert X.shape == (3, 5)

    def test_x_values_correct(
        self, project_with_non_optimizable, observations_with_non_optimizable
    ):
        """Training X has correct values for all features."""
        X, y = project_with_non_optimizable.get_training_data(
            observations_with_non_optimizable
        )
        # First observation: R=100, G=150, B=200, hour=10, ambient_temp=22
        np.testing.assert_array_equal(X[0, :], [100.0, 150.0, 200.0, 10.0, 22.0])

    def test_y_shape_unchanged(
        self, project_with_non_optimizable, observations_with_non_optimizable
    ):
        """Training y shape is unaffected by non-optimizable inputs."""
        X, y = project_with_non_optimizable.get_training_data(
            observations_with_non_optimizable
        )
        assert y.shape == (3, 1)


class TestGetNonOptimizableIndices:
    """Tests for Project.get_non_optimizable_indices() method.

    Returns the indices of non-optimizable features in the full X array.
    Used by recommender to construct fixed_features dict for optimize_acqf.
    """

    def test_returns_list_of_ints(self, project_with_non_optimizable):
        """get_non_optimizable_indices returns list of integers."""
        result = project_with_non_optimizable.get_non_optimizable_indices()
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)

    def test_indices_correct(self, project_with_non_optimizable):
        """Indices correctly identify non-optimizable feature positions in X."""
        result = project_with_non_optimizable.get_non_optimizable_indices()
        # R(0), G(1), B(2), hour(3), ambient_temp(4)
        assert result == [3, 4]

    def test_empty_when_all_optimizable(self, sample_project):
        """Returns empty list when no non-optimizable inputs."""
        result = sample_project.get_non_optimizable_indices()
        assert result == []

    def test_indices_for_interleaved_non_optimizable(self):
        """Indices correct when non-optimizable inputs are interleaved."""
        project = Project(
            id=1,
            name="interleaved",
            inputs=[
                InputSpec("x1", "continuous", bounds=(0.0, 1.0)),
                InputSpec("ctx1", "continuous", bounds=(0.0, 10.0), optimizable=False),
                InputSpec("x2", "continuous", bounds=(2.0, 3.0)),
                InputSpec("ctx2", "continuous", bounds=(0.0, 20.0), optimizable=False),
            ],
            outputs=[OutputSpec("y")],
            target_configs=[TargetConfig(objective="y")],
        )
        result = project.get_non_optimizable_indices()
        # x1(0), ctx1(1), x2(2), ctx2(3)
        assert result == [1, 3]


class TestValidateInputsWithNonOptimizable:
    """Tests for validate_inputs with non-optimizable inputs."""

    def test_requires_non_optimizable_values(self, project_with_non_optimizable):
        """Validation requires non-optimizable input values to be provided."""
        with pytest.raises(InvalidInputError, match="Missing inputs"):
            project_with_non_optimizable.validate_inputs(
                {"R": 100.0, "G": 150.0, "B": 200.0}
            )

    def test_validates_non_optimizable_bounds(self, project_with_non_optimizable):
        """Non-optimizable values are validated against their bounds."""
        with pytest.raises(InvalidInputError, match="outside bounds"):
            project_with_non_optimizable.validate_inputs(
                {
                    "R": 100.0,
                    "G": 150.0,
                    "B": 200.0,
                    "hour": 25.0,  # Invalid: > 24
                    "ambient_temp": 22.0,
                }
            )

    def test_accepts_valid_non_optimizable_values(self, project_with_non_optimizable):
        """Valid non-optimizable values pass validation."""
        # Should not raise
        project_with_non_optimizable.validate_inputs(
            {
                "R": 100.0,
                "G": 150.0,
                "B": 200.0,
                "hour": 12.0,
                "ambient_temp": 22.0,
            }
        )
