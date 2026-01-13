"""Tests for HumanExecutor."""

import pytest

from folio.core.config import TargetConfig
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.executors import HumanExecutor


@pytest.fixture
def rgb_project():
    """Project with RGB inputs and wavelength output."""
    return Project(
        id=1,
        name="rgb_wavelength",
        inputs=[
            InputSpec("R", "continuous", bounds=(0.0, 255.0)),
            InputSpec("G", "continuous", bounds=(0.0, 255.0)),
            InputSpec("B", "continuous", bounds=(0.0, 255.0)),
        ],
        outputs=[OutputSpec("wavelength")],
        target_configs=[TargetConfig(objective="wavelength", objective_mode="maximize")],
    )


@pytest.fixture
def multi_output_project():
    """Project with multiple outputs for testing."""
    return Project(
        id=2,
        name="multi_output",
        inputs=[InputSpec("x", "continuous", bounds=(0.0, 10.0))],
        outputs=[OutputSpec("y1"), OutputSpec("y2")],
        target_configs=[TargetConfig(objective="y1", objective_mode="maximize")],
    )


class TestHumanExecutorInit:
    """Test HumanExecutor initialization."""

    def test_default_input_func(self):
        """HumanExecutor uses builtin input by default."""
        executor = HumanExecutor()
        # Implementation should store input or use builtin
        assert executor is not None

    def test_custom_input_func(self):
        """HumanExecutor accepts custom input function."""
        mock_input = lambda prompt: "42.0"
        executor = HumanExecutor(input_func=mock_input)
        assert executor is not None


class TestHumanExecutorRun:
    """Test HumanExecutor experiment execution."""

    def test_run_with_mock_input(self, rgb_project):
        """HumanExecutor collects outputs via input function."""
        # Mock responses: wavelength value, then 'n' for not failed
        responses = iter(["550.0", "n"])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

        result = executor.execute(suggestion, rgb_project)

        assert isinstance(result, Observation)
        assert result.inputs == suggestion
        assert result.outputs["wavelength"] == 550.0
        assert result.failed is False

    def test_run_multiple_outputs(self, multi_output_project):
        """HumanExecutor prompts for each output."""
        # Mock responses: y1, y2, then 'n' for not failed
        responses = iter(["1.5", "2.5", "n"])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        suggestion = {"x": 5.0}

        result = executor.execute(suggestion, multi_output_project)

        assert result.outputs["y1"] == 1.5
        assert result.outputs["y2"] == 2.5

    def test_run_marks_failed_experiment(self, rgb_project):
        """HumanExecutor can mark experiments as failed."""
        # Mock responses: wavelength value, then 'y' for failed
        responses = iter(["0.0", "y"])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

        result = executor.execute(suggestion, rgb_project)

        assert result.failed is True

    def test_observation_has_correct_project_id(self, rgb_project):
        """Returned observation has project_id from passed project."""
        responses = iter(["550.0", "n"])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

        result = executor.execute(suggestion, rgb_project)

        assert result.project_id == rgb_project.id

    def test_observation_has_no_database_id(self, rgb_project):
        """Returned observation has id=None (DB assigns on persist)."""
        responses = iter(["550.0", "n"])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

        result = executor.execute(suggestion, rgb_project)

        assert result.id is None
