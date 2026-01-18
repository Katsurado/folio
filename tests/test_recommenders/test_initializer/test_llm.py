"""Tests for LLM-based initializer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from folio.core.project import Project
from folio.exceptions import CostLimitError, LLMResponseError
from folio.recommenders.initializer.llm import (
    LLMBackend,
    LLMInitializer,
    OpenAIBackend,
    _estimate_tokens,
)

from .conftest import MockLLMBackend


class TestLLMBackend:
    """Tests for LLMBackend ABC."""

    def test_llm_backend_is_abstract(self) -> None:
        """LLMBackend cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract|instantiate"):
            LLMBackend()


class TestOpenAIBackend:
    """Tests for OpenAI backend."""

    def test_openai_backend_requires_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAIBackend raises if no API key available."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key"):
            OpenAIBackend()

    def test_openai_backend_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OpenAIBackend reads from OPENAI_API_KEY env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        backend = OpenAIBackend()
        assert backend.api_key == "sk-test-key"
        assert backend.model == "gpt-5.2"

    def test_openai_backend_uses_provided_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAIBackend uses provided api_key over env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

        backend = OpenAIBackend(api_key="sk-provided-key")
        assert backend.api_key == "sk-provided-key"

    def test_openai_backend_custom_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OpenAIBackend accepts custom model."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        backend = OpenAIBackend(model="gpt-5.2-mini")
        assert backend.model == "gpt-5.2-mini"

    def test_openai_backend_estimate_cost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAIBackend.estimate_cost returns reasonable estimate."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        backend = OpenAIBackend()
        cost = backend.estimate_cost("Short prompt", max_output_tokens=1000)

        assert cost > 0
        assert cost < 1.0


class TestLLMInitializer:
    """Tests for LLMInitializer."""

    def test_llm_initializer_uses_backend(
        self,
        mock_llm_backend: MockLLMBackend,
        sample_project: Project,
    ) -> None:
        """LLMInitializer calls backend and returns suggestions."""
        initializer = LLMInitializer(mock_llm_backend)
        result = initializer.suggest(sample_project, n=3)

        assert len(result) == 3
        assert all(isinstance(r, dict) for r in result)

    def test_llm_initializer_validates_bounds_clamps(
        self,
        sample_project: Project,
    ) -> None:
        """Out-of-bounds values are clamped with warning."""
        response = json.dumps(
            [
                {"temperature": 200.0, "pressure": 2.0, "solvent": "water"},
            ]
        )
        backend = MockLLMBackend(response)
        initializer = LLMInitializer(backend)

        result = initializer.suggest(sample_project, n=1)

        assert result[0]["temperature"] == 120.0

    def test_llm_initializer_clamps_low_values(
        self,
        sample_project: Project,
    ) -> None:
        """Values below lower bound are clamped."""
        response = json.dumps(
            [
                {"temperature": 10.0, "pressure": 0.1, "solvent": "ethanol"},
            ]
        )
        backend = MockLLMBackend(response)
        initializer = LLMInitializer(backend)

        result = initializer.suggest(sample_project, n=1)

        assert result[0]["temperature"] == 60.0
        assert result[0]["pressure"] == 1.0

    def test_llm_initializer_coerces_types(
        self,
        sample_project: Project,
    ) -> None:
        """String numbers are coerced to float."""
        response = json.dumps(
            [
                {"temperature": "85.5", "pressure": "2.5", "solvent": "water"},
            ]
        )
        backend = MockLLMBackend(response)
        initializer = LLMInitializer(backend)

        result = initializer.suggest(sample_project, n=1)

        assert result[0]["temperature"] == 85.5
        assert isinstance(result[0]["temperature"], float)

    def test_build_prompt_includes_all_parameters(
        self,
        mock_llm_backend: MockLLMBackend,
        sample_project: Project,
    ) -> None:
        """Built prompt includes all parameter specs."""
        initializer = LLMInitializer(mock_llm_backend)
        prompt = initializer._build_prompt(
            sample_project, n=5, description="Test problem", existing=None
        )

        assert "temperature" in prompt
        assert "pressure" in prompt
        assert "solvent" in prompt
        assert "60.0" in prompt or "60" in prompt
        assert "120.0" in prompt or "120" in prompt
        assert "water" in prompt
        assert "ethanol" in prompt

    def test_parse_response_extracts_json(
        self,
        mock_llm_backend: MockLLMBackend,
        sample_project: Project,
    ) -> None:
        """Parser extracts JSON from plain response."""
        response = '[{"temperature": 80, "pressure": 2, "solvent": "water"}]'
        backend = MockLLMBackend(response)
        initializer = LLMInitializer(backend)

        result = initializer.suggest(sample_project, n=1)

        assert len(result) == 1
        assert result[0]["temperature"] == 80.0

    def test_parse_response_handles_markdown_codeblocks(
        self,
        sample_project: Project,
    ) -> None:
        """Parser extracts JSON from markdown code blocks."""
        response = """Here are my suggestions:

```json
[{"temperature": 90, "pressure": 3, "solvent": "ethanol"}]
```

These are based on literature."""
        backend = MockLLMBackend(response)
        initializer = LLMInitializer(backend)

        result = initializer.suggest(sample_project, n=1)

        assert len(result) == 1
        assert result[0]["temperature"] == 90.0
        assert result[0]["solvent"] == "ethanol"

    def test_parse_response_handles_codeblock_without_language(
        self,
        sample_project: Project,
    ) -> None:
        """Parser handles code blocks without json language tag."""
        response = """```
[{"temperature": 75, "pressure": 2.5, "solvent": "dmso"}]
```"""
        backend = MockLLMBackend(response)
        initializer = LLMInitializer(backend)

        result = initializer.suggest(sample_project, n=1)

        assert result[0]["solvent"] == "dmso"

    def test_llm_initializer_uses_default_prompt(
        self,
        mock_llm_backend: MockLLMBackend,
        sample_project: Project,
    ) -> None:
        """LLMInitializer loads default prompt template."""
        initializer = LLMInitializer(mock_llm_backend)

        assert "expert chemist" in initializer.prompt_template
        assert "{n}" in initializer.prompt_template
        assert "{objective}" in initializer.prompt_template

    def test_llm_initializer_uses_custom_prompt_string(
        self,
        mock_llm_backend: MockLLMBackend,
        sample_project: Project,
        custom_prompt_template: str,
    ) -> None:
        """LLMInitializer accepts custom prompt string."""
        initializer = LLMInitializer(
            mock_llm_backend,
            prompt_template=custom_prompt_template,
        )

        assert initializer.prompt_template == custom_prompt_template

        result = initializer.suggest(sample_project, n=3)
        assert len(result) == 3

    def test_llm_initializer_loads_prompt_from_file(
        self,
        mock_llm_backend: MockLLMBackend,
        sample_project: Project,
    ) -> None:
        """LLMInitializer loads prompt from Path."""
        custom_template = (
            "Custom: {n} exps for {objective}. {parameters}. {description}"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(custom_template)
            temp_path = Path(f.name)

        try:
            initializer = LLMInitializer(
                mock_llm_backend,
                prompt_template=temp_path,
            )

            assert initializer.prompt_template == custom_template
        finally:
            temp_path.unlink()

    def test_cost_estimation_under_limit(
        self,
        mock_llm_backend: MockLLMBackend,
        sample_project: Project,
    ) -> None:
        """Request proceeds when cost is under limit."""
        initializer = LLMInitializer(mock_llm_backend, max_cost_per_call=1.00)

        result = initializer.suggest(sample_project, n=3)

        assert len(result) == 3

    def test_cost_estimation_over_limit(
        self,
        sample_project: Project,
    ) -> None:
        """CostLimitError raised when estimated cost exceeds limit."""

        class ExpensiveBackend(MockLLMBackend):
            def estimate_cost(
                self, prompt: str, max_output_tokens: int = 4096
            ) -> float:
                return 5.00

        backend = ExpensiveBackend()
        initializer = LLMInitializer(backend, max_cost_per_call=0.50)

        with pytest.raises(CostLimitError, match="exceeds limit"):
            initializer.suggest(sample_project, n=3)

    def test_invalid_json_raises_llm_response_error(
        self,
        sample_project: Project,
    ) -> None:
        """LLMResponseError raised for invalid JSON."""
        backend = MockLLMBackend("This is not JSON at all")
        initializer = LLMInitializer(backend)

        with pytest.raises(LLMResponseError, match="JSON"):
            initializer.suggest(sample_project, n=1)

    def test_invalid_categorical_uses_first_level(
        self,
        sample_project: Project,
    ) -> None:
        """Invalid categorical value falls back to first level."""
        response = json.dumps(
            [
                {"temperature": 80, "pressure": 2, "solvent": "acetone"},
            ]
        )
        backend = MockLLMBackend(response)
        initializer = LLMInitializer(backend)

        result = initializer.suggest(sample_project, n=1)

        assert result[0]["solvent"] == "water"

    def test_existing_experiments_included_in_prompt(
        self,
        mock_llm_backend: MockLLMBackend,
        sample_project: Project,
    ) -> None:
        """Existing experiments are appended to prompt."""
        initializer = LLMInitializer(mock_llm_backend)
        existing = [{"temperature": 70, "pressure": 1.5, "solvent": "water"}]

        prompt = initializer._build_prompt(
            sample_project, n=3, description="Test", existing=existing
        )

        assert "Already planned/run" in prompt
        assert "70" in prompt


class TestEstimateTokens:
    """Tests for token estimation helper."""

    def test_estimate_tokens_basic(self) -> None:
        """Token estimate is roughly len/4."""
        text = "a" * 400
        tokens = _estimate_tokens(text)

        assert tokens == 100

    def test_estimate_tokens_empty(self) -> None:
        """Empty string returns 0 tokens."""
        assert _estimate_tokens("") == 0
