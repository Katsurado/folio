"""Shared pytest fixtures for Folio tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_db():
    """Provide a temporary database path that's cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def sample_input_schema():
    """Sample input schema for testing."""
    return [
        {
            "name": "temperature",
            "type": "continuous",
            "bounds": [60, 120],
            "units": "C",
        },
        {"name": "pressure", "type": "continuous", "bounds": [1, 10], "units": "atm"},
        {"name": "solvent", "type": "categorical", "levels": ["THF", "DMF", "toluene"]},
    ]


@pytest.fixture
def sample_output_schema():
    """Sample output schema for testing."""
    return [
        {"name": "yield", "units": "%"},
        {"name": "purity", "units": "%"},
    ]


@pytest.fixture
def sample_observations():
    """Sample observations for testing."""
    return [
        {
            "inputs": {"temperature": 80, "pressure": 5, "solvent": "THF"},
            "outputs": {"yield": 72.5, "purity": 95.0},
        },
        {
            "inputs": {"temperature": 100, "pressure": 3, "solvent": "DMF"},
            "outputs": {"yield": 85.2, "purity": 91.0},
        },
        {
            "inputs": {"temperature": 60, "pressure": 8, "solvent": "toluene"},
            "outputs": {"yield": 55.0, "purity": 98.5},
        },
    ]
