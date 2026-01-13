"""Tests for Executor base class."""

import pytest

from folio.core.config import TargetConfig
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.exceptions import InvalidInputError
from folio.executors import Executor


class ConcreteExecutor(Executor):
    """Minimal concrete implementation for testing the interface contract."""

    def __init__(self):
        self.run_called = False
        self.last_suggestion = None

    def _run(self, suggestion: dict, project: Project) -> Observation:
        """Store call info and return a mock observation."""
        self.run_called = True
        self.last_suggestion = suggestion
        return Observation(
            project_id=project.id,
            inputs=suggestion,
            outputs={out.name: 0.0 for out in project.outputs},
        )


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
        target_configs=[
            TargetConfig(objective="wavelength", objective_mode="maximize")
        ],
    )


class TestExecutorABC:
    """Test that Executor enforces the abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Executor ABC cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Executor()

    def test_must_implement_run(self):
        """Subclass without _run raises TypeError."""

        class NoRun(Executor):
            pass

        with pytest.raises(TypeError, match="abstract"):
            NoRun()


class TestExecuteMethod:
    """Test the execute() method contract."""

    def test_execute_validates_inputs_before_run(self, rgb_project):
        """execute() calls validate_inputs before _run."""
        executor = ConcreteExecutor()
        suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

        executor.execute(suggestion, rgb_project)

        assert executor.run_called
        assert executor.last_suggestion == suggestion

    def test_execute_raises_on_invalid_inputs(self, rgb_project):
        """execute() raises InvalidInputError for invalid inputs."""
        executor = ConcreteExecutor()
        # Missing 'B' input
        suggestion = {"R": 100.0, "G": 150.0}

        with pytest.raises(InvalidInputError, match="Missing|missing"):
            executor.execute(suggestion, rgb_project)

        assert not executor.run_called

    def test_execute_raises_on_out_of_bounds(self, rgb_project):
        """execute() raises InvalidInputError for out-of-bounds values."""
        executor = ConcreteExecutor()
        # R > 255
        suggestion = {"R": 300.0, "G": 150.0, "B": 200.0}

        with pytest.raises(InvalidInputError, match="outside|bounds"):
            executor.execute(suggestion, rgb_project)

        assert not executor.run_called

    def test_execute_raises_on_unexpected_inputs(self, rgb_project):
        """execute() raises InvalidInputError for extra inputs."""
        executor = ConcreteExecutor()
        suggestion = {"R": 100.0, "G": 150.0, "B": 200.0, "A": 50.0}

        with pytest.raises(InvalidInputError, match="Unexpected|unexpected|extra"):
            executor.execute(suggestion, rgb_project)

        assert not executor.run_called

    def test_execute_returns_observation(self, rgb_project):
        """execute() returns an Observation with correct project_id."""
        executor = ConcreteExecutor()
        suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

        result = executor.execute(suggestion, rgb_project)

        assert isinstance(result, Observation)
        assert result.project_id == rgb_project.id
        assert result.inputs == suggestion
        assert result.id is None

    def test_validation_error_does_not_call_run(self, rgb_project):
        """When validation fails, _run is not called."""
        executor = ConcreteExecutor()
        invalid_suggestion = {"R": -10.0, "G": 150.0, "B": 200.0}

        with pytest.raises(InvalidInputError):
            executor.execute(invalid_suggestion, rgb_project)

        assert not executor.run_called
