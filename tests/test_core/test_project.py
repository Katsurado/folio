import numpy as np
import pytest

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
        target_config=TargetConfig(objective="yield"),
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
            target_config=TargetConfig(objective="y"),
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
            target_config=TargetConfig(objective="y"),
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
            target_config=TargetConfig(objective="y"),
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
            target_config=TargetConfig(objective="y"),
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
            target_config=TargetConfig(objective="y"),
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
            target_config=TargetConfig(objective="y"),
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
            target_config=TargetConfig(objective="y"),
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
            target_config=TargetConfig(objective="y"),
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
            target_config=TargetConfig(objective="y"),
        )
        result = project.get_optimization_bounds()
        np.testing.assert_array_equal(result, [[0.0], [1e6]])
