"""Tests for ClaudeLightExecutor."""

from unittest.mock import Mock, patch

import pytest

from folio.core.config import TargetConfig
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.exceptions import ExecutorError
from folio.executors import ClaudeLightExecutor

PATCH_TARGET = "folio.executors.claude_light.requests.get"


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


def make_api_response(r, g, b, wavelength):
    """Create mock API response in expected format."""
    return {"in": [r, g, b], "out": {"wavelength": wavelength}}


class TestClaudeLightExecutorInit:
    """Test ClaudeLightExecutor initialization."""

    def test_default_api_url(self):
        """ClaudeLightExecutor uses default API URL."""
        executor = ClaudeLightExecutor()
        assert executor.api_url == ClaudeLightExecutor.DEFAULT_API_URL

    def test_custom_api_url(self):
        """ClaudeLightExecutor accepts custom API URL."""
        custom_url = "https://custom.api/endpoint"
        executor = ClaudeLightExecutor(api_url=custom_url)
        assert executor.api_url == custom_url


class TestClaudeLightExecutorRun:
    """Test ClaudeLightExecutor API interaction."""

    def test_successful_api_call(self, rgb_project):
        """ClaudeLightExecutor returns observation from API response."""
        mock_response = Mock()
        mock_response.json.return_value = make_api_response(100.0, 150.0, 200.0, 550.0)

        with patch(PATCH_TARGET, return_value=mock_response) as mock_get:
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            result = executor.execute(suggestion, rgb_project)

            mock_get.assert_called_once()
            assert isinstance(result, Observation)
            assert result.outputs["wavelength"] == 550.0

    def test_api_call_includes_inputs_as_params(self, rgb_project):
        """ClaudeLightExecutor sends inputs as query parameters."""
        mock_response = Mock()
        mock_response.json.return_value = make_api_response(100.0, 150.0, 200.0, 550.0)

        with patch(PATCH_TARGET, return_value=mock_response) as mock_get:
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            executor.execute(suggestion, rgb_project)

            # Verify inputs were passed as params
            call_args = mock_get.call_args
            assert call_args.kwargs["params"] == {"R": 100.0, "G": 150.0, "B": 200.0}

    def test_inputs_from_api_response(self, rgb_project):
        """Observation.inputs contains actual values from API response."""
        # API returns slightly different actual values than suggested
        mock_response = Mock()
        mock_response.json.return_value = make_api_response(102.0, 148.0, 201.0, 550.0)

        with patch(PATCH_TARGET, return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            result = executor.execute(suggestion, rgb_project)

            # inputs come from API response, not suggestion
            assert result.inputs == {"R": 102.0, "G": 148.0, "B": 201.0}

    def test_inputs_suggested_stores_original(self, rgb_project):
        """Observation.inputs_suggested contains original suggestion."""
        mock_response = Mock()
        mock_response.json.return_value = make_api_response(102.0, 148.0, 201.0, 550.0)

        with patch(PATCH_TARGET, return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            result = executor.execute(suggestion, rgb_project)

            assert result.inputs_suggested == suggestion

    def test_observation_has_correct_project_id(self, rgb_project):
        """Returned observation has project_id from passed project."""
        mock_response = Mock()
        mock_response.json.return_value = make_api_response(100.0, 150.0, 200.0, 550.0)

        with patch(PATCH_TARGET, return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            result = executor.execute(suggestion, rgb_project)

            assert result.project_id == rgb_project.id

    def test_observation_has_no_database_id(self, rgb_project):
        """Returned observation has id=None (DB assigns on persist)."""
        mock_response = Mock()
        mock_response.json.return_value = make_api_response(100.0, 150.0, 200.0, 550.0)

        with patch(PATCH_TARGET, return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            result = executor.execute(suggestion, rgb_project)

            assert result.id is None

    def test_custom_api_url_is_used(self, rgb_project):
        """ClaudeLightExecutor uses custom API URL when provided."""
        mock_response = Mock()
        mock_response.json.return_value = make_api_response(100.0, 150.0, 200.0, 550.0)

        custom_url = "https://custom.api/endpoint"

        with patch(PATCH_TARGET, return_value=mock_response) as mock_get:
            executor = ClaudeLightExecutor(api_url=custom_url)
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            executor.execute(suggestion, rgb_project)

            # Verify the custom URL was used
            call_args = mock_get.call_args
            assert call_args.args[0] == custom_url

    def test_observation_has_metadata(self, rgb_project):
        """Returned observation includes metadata with version info."""
        mock_response = Mock()
        mock_response.json.return_value = make_api_response(100.0, 150.0, 200.0, 550.0)

        with patch(PATCH_TARGET, return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            result = executor.execute(suggestion, rgb_project)

            assert result.metadata is not None
            assert "version" in result.metadata
            assert "user" in result.metadata
            assert "hostname" in result.metadata


class TestClaudeLightExecutorErrorHandling:
    """Test error handling for API failures.

    Base class wraps _run() in try/except, converting exceptions to ExecutorError.
    """

    def test_api_connection_error_raises_executor_error(self, rgb_project):
        """ClaudeLightExecutor raises ExecutorError on connection failure."""
        with patch(PATCH_TARGET) as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            with pytest.raises(ExecutorError, match="(?i)fail|error|connection"):
                executor.execute(suggestion, rgb_project)

    def test_api_http_error_raises_executor_error(self, rgb_project):
        """ClaudeLightExecutor raises ExecutorError on HTTP error response."""
        import requests as req

        mock_response = Mock()
        mock_response.json.side_effect = req.HTTPError("404 Not Found")

        with patch(PATCH_TARGET, return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            with pytest.raises(ExecutorError, match="(?i)fail|error|http"):
                executor.execute(suggestion, rgb_project)

    def test_invalid_json_raises_executor_error(self, rgb_project):
        """ClaudeLightExecutor raises ExecutorError on invalid JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with patch(PATCH_TARGET, return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            with pytest.raises(ExecutorError, match="(?i)fail|error|json|invalid"):
                executor.execute(suggestion, rgb_project)

    def test_missing_response_fields_raises_executor_error(self, rgb_project):
        """ClaudeLightExecutor raises ExecutorError when response missing fields."""
        mock_response = Mock()
        # Missing "in" field
        mock_response.json.return_value = {"out": {"wavelength": 550.0}}

        with patch(PATCH_TARGET, return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            with pytest.raises(ExecutorError, match="(?i)fail|error|key"):
                executor.execute(suggestion, rgb_project)
