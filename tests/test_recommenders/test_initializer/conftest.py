"""Fixtures for initializer tests."""

from __future__ import annotations

import json

import pytest

from folio.core.config import TargetConfig
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.recommenders.initializer.llm import LLMBackend


class MockLLMBackend(LLMBackend):
    """Mock LLM backend that returns canned responses."""

    def __init__(self, response: str | None = None) -> None:
        """Initialize with optional canned response."""
        self._response = response or json.dumps(
            [
                {"temperature": 80.0, "pressure": 2.5, "solvent": "ethanol"},
                {"temperature": 100.0, "pressure": 3.0, "solvent": "water"},
                {"temperature": 90.0, "pressure": 2.0, "solvent": "dmso"},
            ]
        )

    def complete(self, prompt: str) -> str:
        """Return canned response."""
        return self._response

    def estimate_cost(self, prompt: str, max_output_tokens: int = 4096) -> float:
        """Return fixed low cost."""
        return 0.01


@pytest.fixture
def sample_input_spec_continuous() -> InputSpec:
    """Continuous input spec for temperature (60-120)."""
    return InputSpec(
        name="temperature",
        type="continuous",
        bounds=(60.0, 120.0),
        units="°C",
    )


@pytest.fixture
def sample_input_spec_categorical() -> InputSpec:
    """Categorical input spec for solvent."""
    return InputSpec(
        name="solvent",
        type="categorical",
        levels=["water", "ethanol", "dmso"],
    )


@pytest.fixture
def sample_project(
    sample_input_spec_continuous: InputSpec,
    sample_input_spec_categorical: InputSpec,
) -> Project:
    """Project with 2 continuous + 1 categorical parameter."""
    return Project(
        id=None,
        name="test_optimization",
        inputs=[
            sample_input_spec_continuous,
            InputSpec("pressure", "continuous", bounds=(1.0, 5.0), units="atm"),
            sample_input_spec_categorical,
        ],
        outputs=[OutputSpec("yield", units="%")],
        target_configs=[TargetConfig(objective="yield", objective_mode="maximize")],
    )


@pytest.fixture
def mock_llm_backend() -> MockLLMBackend:
    """Mock LLM backend with default valid response."""
    return MockLLMBackend()


@pytest.fixture
def custom_prompt_template() -> str:
    """Simple custom template for testing."""
    return (
        "Suggest {n} experiments. Goal: {objective}. "
        "Params: {parameters}. Context: {description}"
    )
