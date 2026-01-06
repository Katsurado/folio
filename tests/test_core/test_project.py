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
        target_config=TargetConfig(name="yield"),
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
