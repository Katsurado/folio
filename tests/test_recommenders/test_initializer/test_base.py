"""Tests for initializer base classes."""

from __future__ import annotations

import pytest

from folio.core.project import Project
from folio.core.schema import InputSpec
from folio.recommenders.initializer.base import Initializer


class TestInitializerABC:
    """Tests for Initializer abstract base class."""

    def test_initializer_is_abstract(self) -> None:
        """Initializer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract|instantiate"):
            Initializer()

    def test_initializer_subclass_must_implement_suggest(self) -> None:
        """Subclass without suggest method cannot be instantiated."""

        class IncompleteInitializer(Initializer):
            pass

        with pytest.raises(TypeError, match="abstract|suggest"):
            IncompleteInitializer()

    def test_initializer_subclass_with_suggest_can_instantiate(
        self, sample_project: Project
    ) -> None:
        """Subclass with suggest method can be instantiated."""

        class ConcreteInitializer(Initializer):
            def suggest(
                self,
                project: Project,
                n: int,
                description: str | None = None,
                existing: list[dict] | None = None,
            ) -> list[dict]:
                return [{"temperature": 80.0} for _ in range(n)]

        initializer = ConcreteInitializer()
        result = initializer.suggest(sample_project, n=3)

        assert len(result) == 3
        assert all(isinstance(r, dict) for r in result)


class TestInputSpec:
    """Tests for InputSpec used by initializers."""

    def test_input_spec_continuous(
        self, sample_input_spec_continuous: InputSpec
    ) -> None:
        """Continuous input spec has correct properties."""
        spec = sample_input_spec_continuous

        assert spec.name == "temperature"
        assert spec.type == "continuous"
        assert spec.bounds == (60.0, 120.0)
        assert spec.units == "°C"
        assert spec.levels is None

    def test_input_spec_categorical(
        self, sample_input_spec_categorical: InputSpec
    ) -> None:
        """Categorical input spec has correct properties."""
        spec = sample_input_spec_categorical

        assert spec.name == "solvent"
        assert spec.type == "categorical"
        assert spec.levels == ["water", "ethanol", "dmso"]
        assert spec.bounds is None


class TestProjectForInitializer:
    """Tests for Project usage with initializers."""

    def test_project_has_inputs(self, sample_project: Project) -> None:
        """Project has input specifications."""
        assert len(sample_project.inputs) == 3
        assert sample_project.inputs[0].name == "temperature"
        assert sample_project.inputs[1].name == "pressure"
        assert sample_project.inputs[2].name == "solvent"

    def test_project_has_target_configs(self, sample_project: Project) -> None:
        """Project has target configuration."""
        assert len(sample_project.target_configs) == 1
        assert sample_project.target_configs[0].objective == "yield"
        assert sample_project.target_configs[0].objective_mode == "maximize"
