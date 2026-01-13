"""Tests for ClaudeLightExecutor."""

from unittest.mock import Mock, patch

import pytest

from folio.core.config import TargetConfig
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.exceptions import ExecutorError
from folio.executors import ClaudeLightExecutor


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


class TestClaudeLightExecutorInit:
    """Test ClaudeLightExecutor initialization."""

    def test_default_api_url(self):
        """ClaudeLightExecutor uses default API URL."""
        executor = ClaudeLightExecutor()
        assert executor is not None

    def test_custom_api_url(self):
        """ClaudeLightExecutor accepts custom API URL."""
        executor = ClaudeLightExecutor(api_url="https://custom.api/endpoint")
        assert executor is not None


class TestClaudeLightExecutorRun:
    """Test ClaudeLightExecutor API interaction."""

    def test_successful_api_call(self, rgb_project):
        """ClaudeLightExecutor returns observation from API response."""
        mock_response = Mock()
        mock_response.json.return_value = {"out": {"wavelength": 550.0}}
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            result = executor.execute(suggestion, rgb_project)

            mock_get.assert_called_once()
            assert isinstance(result, Observation)
            assert result.outputs["wavelength"] == 550.0

    def test_api_call_includes_inputs(self, rgb_project):
        """ClaudeLightExecutor sends inputs as query parameters."""
        mock_response = Mock()
        mock_response.json.return_value = {"out": {"wavelength": 550.0}}
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            executor.execute(suggestion, rgb_project)

            # Verify inputs were passed to the API
            call_kwargs = mock_get.call_args
            assert "params" in call_kwargs.kwargs or len(call_kwargs.args) > 1

    def test_api_error_raises_executor_error(self, rgb_project):
        """ClaudeLightExecutor raises ExecutorError on API failure."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            with pytest.raises(ExecutorError, match="(?i)api|request|connection|fail"):
                executor.execute(suggestion, rgb_project)

    def test_api_http_error_raises_executor_error(self, rgb_project):
        """ClaudeLightExecutor raises ExecutorError on HTTP error response."""
        import requests

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("requests.get", return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            with pytest.raises(ExecutorError, match="(?i)api|http|error|fail"):
                executor.execute(suggestion, rgb_project)

    def test_invalid_json_raises_executor_error(self, rgb_project):
        """ClaudeLightExecutor raises ExecutorError on invalid JSON response."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with patch("requests.get", return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            with pytest.raises(ExecutorError, match="(?i)json|response|parse|invalid"):
                executor.execute(suggestion, rgb_project)

    def test_observation_has_correct_project_id(self, rgb_project):
        """Returned observation has project_id from passed project."""
        mock_response = Mock()
        mock_response.json.return_value = {"out": {"wavelength": 550.0}}
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            result = executor.execute(suggestion, rgb_project)

            assert result.project_id == rgb_project.id

    def test_observation_has_no_database_id(self, rgb_project):
        """Returned observation has id=None (DB assigns on persist)."""
        mock_response = Mock()
        mock_response.json.return_value = {"out": {"wavelength": 550.0}}
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            executor = ClaudeLightExecutor()
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            result = executor.execute(suggestion, rgb_project)

            assert result.id is None

    def test_custom_api_url_is_used(self, rgb_project):
        """ClaudeLightExecutor uses custom API URL when provided."""
        mock_response = Mock()
        mock_response.json.return_value = {"out": {"wavelength": 550.0}}
        mock_response.raise_for_status = Mock()

        custom_url = "https://custom.api/endpoint"

        with patch("requests.get", return_value=mock_response) as mock_get:
            executor = ClaudeLightExecutor(api_url=custom_url)
            suggestion = {"R": 100.0, "G": 150.0, "B": 200.0}

            executor.execute(suggestion, rgb_project)

            # Verify the custom URL was used
            call_args = mock_get.call_args
            assert custom_url in str(call_args)
