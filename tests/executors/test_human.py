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
        assert executor.input_func is input

    def test_custom_input_func(self):
        """HumanExecutor accepts custom input function."""
        mock_input = lambda prompt: "42.0"
        executor = HumanExecutor(input_func=mock_input)
        assert executor.input_func is mock_input


class TestHumanExecutorRun:
    """Test HumanExecutor experiment execution."""

    def test_run_collects_actual_inputs_and_outputs(self, rgb_project):
        """HumanExecutor collects actual inputs and outputs via input function."""
        # Sequence: R, G, B actual values, wavelength, failed, notes, tag
        responses = iter(["100.0", "150.0", "200.0", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

        result = executor.execute(suggestion, rgb_project)

        assert isinstance(result, Observation)
        assert result.inputs == {"R": 100.0, "G": 150.0, "B": 200.0}
        assert result.outputs["wavelength"] == 550.0
        assert result.failed is False

    def test_actual_inputs_can_differ_from_suggestion(self, rgb_project):
        """User can enter actual inputs different from suggested values."""
        # Actual values differ from suggestion
        responses = iter(["105.0", "148.0", "202.0", "555.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

        result = executor.execute(suggestion, rgb_project)

        # Actual inputs differ from suggestion
        assert result.inputs == {"R": 105.0, "G": 148.0, "B": 202.0}
        assert result.inputs_suggested == suggestion

    def test_inputs_suggested_stores_original_suggestion(self, rgb_project):
        """Observation.inputs_suggested contains the original suggestion."""
        responses = iter(["100.0", "150.0", "200.0", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

        result = executor.execute(suggestion, rgb_project)

        assert result.inputs_suggested == suggestion

    def test_run_multiple_outputs(self, multi_output_project):
        """HumanExecutor prompts for each output."""
        # Sequence: x actual, y1, y2, failed, notes, tag
        responses = iter(["5.0", "1.5", "2.5", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        suggestion = {"x": 5.0}

        result = executor.execute(suggestion, multi_output_project)

        assert result.outputs["y1"] == 1.5
        assert result.outputs["y2"] == 2.5

    def test_run_marks_failed_experiment(self, rgb_project):
        """HumanExecutor can mark experiments as failed."""
        responses = iter(["100.0", "150.0", "200.0", "550.0", "y", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

        result = executor.execute(suggestion, rgb_project)

        assert result.failed is True

    def test_failed_flag_yes_variations(self, rgb_project):
        """Various 'yes' responses mark experiment as failed."""
        for yes_response in ["y", "Y", "yes", "YES", "Yeah"]:
            responses = iter(["100.0", "150.0", "200.0", "550.0", yes_response, "", ""])
            mock_input = lambda prompt: next(responses)

            executor = HumanExecutor(input_func=mock_input)
            result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

            assert result.failed is True, f"'{yes_response}' should mark as failed"

    def test_failed_flag_no_variations(self, rgb_project):
        """Various 'no' responses do not mark experiment as failed."""
        for no_response in ["n", "N", "no", "NO", "nope", ""]:
            responses = iter(["100.0", "150.0", "200.0", "550.0", no_response, "", ""])
            mock_input = lambda prompt: next(responses)

            executor = HumanExecutor(input_func=mock_input)
            result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

            assert result.failed is False, f"'{no_response}' should not mark as failed"

    def test_notes_are_captured(self, rgb_project):
        """Notes entered by user are stored in observation."""
        responses = iter(["100.0", "150.0", "200.0", "550.0", "n", "Sample looked cloudy", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        assert result.notes == "Sample looked cloudy"

    def test_empty_notes_become_none(self, rgb_project):
        """Empty notes string becomes None."""
        responses = iter(["100.0", "150.0", "200.0", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        assert result.notes is None

    def test_whitespace_notes_become_none(self, rgb_project):
        """Whitespace-only notes become None."""
        responses = iter(["100.0", "150.0", "200.0", "550.0", "n", "   ", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        assert result.notes is None

    def test_tag_is_captured(self, rgb_project):
        """Tag entered by user is stored in observation."""
        responses = iter(["100.0", "150.0", "200.0", "550.0", "n", "", "screening"])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        assert result.tag == "screening"

    def test_empty_tag_becomes_none(self, rgb_project):
        """Empty tag string becomes None."""
        responses = iter(["100.0", "150.0", "200.0", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        assert result.tag is None

    def test_observation_has_correct_project_id(self, rgb_project):
        """Returned observation has project_id from passed project."""
        responses = iter(["100.0", "150.0", "200.0", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        assert result.project_id == rgb_project.id

    def test_observation_has_no_database_id(self, rgb_project):
        """Returned observation has id=None (DB assigns on persist)."""
        responses = iter(["100.0", "150.0", "200.0", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        assert result.id is None


class TestHumanExecutorInputValidation:
    """Test input validation and retry behavior."""

    def test_empty_input_retries(self, rgb_project, capsys):
        """Empty input for actual value triggers retry."""
        # First R is empty, then valid; rest are valid
        responses = iter(["", "100.0", "150.0", "200.0", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        captured = capsys.readouterr()
        assert "actual inputs" in captured.out.lower()
        assert result.inputs["R"] == 100.0

    def test_invalid_number_input_retries(self, rgb_project, capsys):
        """Non-numeric input triggers retry with error message."""
        # First R is invalid, then valid
        responses = iter(["abc", "100.0", "150.0", "200.0", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        captured = capsys.readouterr()
        assert "invalid" in captured.out.lower()
        assert result.inputs["R"] == 100.0

    def test_empty_output_retries(self, rgb_project, capsys):
        """Empty output value triggers retry."""
        # wavelength is empty first, then valid
        responses = iter(["100.0", "150.0", "200.0", "", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        captured = capsys.readouterr()
        assert "actual outputs" in captured.out.lower()
        assert result.outputs["wavelength"] == 550.0

    def test_invalid_number_output_retries(self, rgb_project, capsys):
        """Non-numeric output triggers retry with error message."""
        # wavelength is invalid first, then valid
        responses = iter(["100.0", "150.0", "200.0", "not_a_number", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        captured = capsys.readouterr()
        assert "invalid" in captured.out.lower()
        assert result.outputs["wavelength"] == 550.0

    def test_multiple_retries_before_valid_input(self, rgb_project, capsys):
        """Multiple invalid attempts followed by valid input."""
        # R: empty, invalid, valid; rest normal
        responses = iter(["", "abc", "100.0", "150.0", "200.0", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        captured = capsys.readouterr()
        assert "actual inputs" in captured.out.lower()
        assert "invalid" in captured.out.lower()
        assert result.inputs["R"] == 100.0

    def test_whitespace_input_treated_as_empty(self, rgb_project, capsys):
        """Whitespace-only input is treated as empty and triggers retry."""
        responses = iter(["   ", "100.0", "150.0", "200.0", "550.0", "n", "", ""])
        mock_input = lambda prompt: next(responses)

        executor = HumanExecutor(input_func=mock_input)
        result = executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        captured = capsys.readouterr()
        assert "actual inputs" in captured.out.lower()


class TestHumanExecutorPrompts:
    """Test that prompts contain expected information."""

    def test_input_prompt_shows_suggested_value(self, rgb_project):
        """Input prompt displays the suggested value."""
        prompts_received = []

        def capturing_input(prompt):
            prompts_received.append(prompt)
            # Return valid responses in sequence
            values = ["100.0", "150.0", "200.0", "550.0", "n", "", ""]
            return values[len(prompts_received) - 1]

        executor = HumanExecutor(input_func=capturing_input)
        executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        # First three prompts should show suggested values
        assert "100.0" in prompts_received[0]
        assert "150.0" in prompts_received[1]
        assert "200.0" in prompts_received[2]

    def test_input_prompt_shows_input_name(self, rgb_project):
        """Input prompt displays the input variable name."""
        prompts_received = []

        def capturing_input(prompt):
            prompts_received.append(prompt)
            values = ["100.0", "150.0", "200.0", "550.0", "n", "", ""]
            return values[len(prompts_received) - 1]

        executor = HumanExecutor(input_func=capturing_input)
        executor.execute({"R": 100.0, "G": 150.0, "B": 200.0}, rgb_project)

        # Check input names appear in prompts
        assert "R" in prompts_received[0]
        assert "G" in prompts_received[1]
        assert "B" in prompts_received[2]
