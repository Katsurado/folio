"""LLM-based experiment initialization."""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from folio.exceptions import CostLimitError, LLMResponseError
from folio.recommenders.initializer.base import Initializer

if TYPE_CHECKING:
    from folio.core.project import Project

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "default.txt"


class LLMBackend(ABC):
    """Abstract base class for LLM API backends.

    Provides a uniform interface for different LLM providers (OpenAI, Anthropic,
    local models, etc.) to be used with LLMInitializer.

    Examples
    --------
    >>> class AnthropicBackend(LLMBackend):
    ...     def complete(self, prompt):
    ...         # Call Claude API
    ...         ...
    ...     def estimate_cost(self, prompt, max_output_tokens):
    ...         # Estimate cost based on token count
    ...         ...
    """

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send prompt to LLM and return the response.

        Parameters
        ----------
        prompt : str
            The prompt to send to the LLM.

        Returns
        -------
        str
            The LLM's response text.

        Raises
        ------
        LLMResponseError
            If the API call fails or returns an invalid response.
        """
        ...

    @abstractmethod
    def estimate_cost(self, prompt: str, max_output_tokens: int = 4096) -> float:
        """Estimate the cost of an API call in USD.

        Parameters
        ----------
        prompt : str
            The prompt that will be sent.
        max_output_tokens : int, optional
            Maximum expected output tokens. Defaults to 4096.

        Returns
        -------
        float
            Estimated cost in USD.
        """
        ...


class OpenAIBackend(LLMBackend):
    """OpenAI API backend with web search capability.

    Uses the OpenAI chat completions API with the web_search_preview tool
    enabled for literature-informed suggestions.

    Parameters
    ----------
    model : str, optional
        Model identifier. Defaults to "gpt-5.2".
    api_key : str | None, optional
        OpenAI API key. If not provided, reads from OPENAI_API_KEY
        environment variable.

    Attributes
    ----------
    model : str
        The model being used.
    api_key : str
        The API key (masked in repr).

    Notes
    -----
    Pricing (as of 2025):
    - Input: $1.75 / 1M tokens
    - Output: $14.00 / 1M tokens
    - Web search: $0.01 / call + ~8k search context tokens

    Examples
    --------
    >>> backend = OpenAIBackend()  # Uses OPENAI_API_KEY env var
    >>> response = backend.complete("What is 2+2?")

    >>> backend = OpenAIBackend(model="gpt-5.2-mini", api_key="sk-...")
    >>> cost = backend.estimate_cost("Long prompt...", max_output_tokens=2000)
    """

    INPUT_COST_PER_MILLION = 1.75
    OUTPUT_COST_PER_MILLION = 14.00
    WEB_SEARCH_COST_PER_CALL = 0.01
    WEB_SEARCH_CONTEXT_TOKENS = 8000

    def __init__(self, model: str = "gpt-5.2", api_key: str | None = None) -> None:
        """Initialize OpenAI backend.

        Parameters
        ----------
        model : str, optional
            Model identifier. Defaults to "gpt-5.2".
        api_key : str | None, optional
            OpenAI API key. If not provided, reads from OPENAI_API_KEY
            environment variable.

        Raises
        ------
        ValueError
            If no API key is provided and OPENAI_API_KEY is not set.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Provide api_key parameter or set "
                "OPENAI_API_KEY environment variable."
            )

    def complete(self, prompt: str) -> str:
        """Send prompt to OpenAI API with web search enabled.

        Parameters
        ----------
        prompt : str
            The prompt to send.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        LLMResponseError
            If the API call fails or returns an unexpected format.
        """
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": prompt,
                    "tools": [{"type": "web_search_preview"}],
                },
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise LLMResponseError(f"OpenAI API request failed: {e}") from e

        try:
            data = response.json()
            # Find the message output (web search responses have multiple outputs)
            for output_item in data["output"]:
                if output_item.get("type") == "message":
                    for content_item in output_item.get("content", []):
                        if content_item.get("type") == "output_text":
                            return content_item["text"]
            raise LLMResponseError("No message content found in API response")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMResponseError(f"Unexpected OpenAI API response format: {e}") from e

    def estimate_cost(self, prompt: str, max_output_tokens: int = 4096) -> float:
        """Estimate cost of API call with web search.

        Parameters
        ----------
        prompt : str
            The prompt that will be sent.
        max_output_tokens : int, optional
            Maximum expected output tokens. Defaults to 4096.

        Returns
        -------
        float
            Estimated cost in USD.
        """
        input_tokens = _estimate_tokens(prompt)
        total_input_tokens = input_tokens + self.WEB_SEARCH_CONTEXT_TOKENS

        input_cost = (total_input_tokens / 1_000_000) * self.INPUT_COST_PER_MILLION
        output_cost = (max_output_tokens / 1_000_000) * self.OUTPUT_COST_PER_MILLION
        search_cost = self.WEB_SEARCH_COST_PER_CALL

        return input_cost + output_cost + search_cost


class LLMInitializer(Initializer):
    """LLM-based experiment initializer using literature search.

    Uses a large language model with web search capability to suggest
    initial experiments based on scientific literature and best practices.

    Parameters
    ----------
    backend : LLMBackend
        The LLM backend to use for generating suggestions.
    prompt_template : str | Path | None, optional
        Custom prompt template. If None, uses the default template.
        If Path, loads template from file. If str, uses directly.
    max_cost_per_call : float, optional
        Maximum allowed cost per API call in USD. Defaults to 0.50.
        Raises CostLimitError if estimated cost exceeds this limit.

    Attributes
    ----------
    backend : LLMBackend
        The LLM backend.
    prompt_template : str
        The prompt template string.
    max_cost_per_call : float
        Cost limit per call.

    Examples
    --------
    >>> backend = OpenAIBackend()
    >>> initializer = LLMInitializer(backend)
    >>> suggestions = initializer.suggest(project, n=5)

    >>> # With custom prompt
    >>> initializer = LLMInitializer(
    ...     backend,
    ...     prompt_template="Suggest {n} experiments for: {description}",
    ...     max_cost_per_call=1.00,
    ... )
    """

    def __init__(
        self,
        backend: LLMBackend,
        prompt_template: str | Path | None = None,
        max_cost_per_call: float = 0.50,
    ) -> None:
        """Initialize LLM-based initializer.

        Parameters
        ----------
        backend : LLMBackend
            The LLM backend to use.
        prompt_template : str | Path | None, optional
            Custom prompt template. None uses default, Path loads from file,
            str uses directly.
        max_cost_per_call : float, optional
            Maximum cost per call in USD. Defaults to 0.50.
        """
        self.backend = backend
        self.max_cost_per_call = max_cost_per_call

        if prompt_template is None:
            self.prompt_template = DEFAULT_PROMPT_PATH.read_text()
        elif isinstance(prompt_template, Path):
            self.prompt_template = prompt_template.read_text()
        else:
            self.prompt_template = prompt_template

    def suggest(
        self,
        project: Project,
        n: int,
        description: str | None = None,
        existing: list[dict] | None = None,
    ) -> list[dict]:
        """Suggest initial experiments using LLM with literature search.

        Parameters
        ----------
        project : Project
            The project defining inputs, outputs, and optimization targets.
        n : int
            Number of initial experiments to suggest.
        description : str | None, optional
            Natural language description of the problem for LLM context.
        existing : list[dict] | None, optional
            Previously run experiments to avoid duplicating.

        Returns
        -------
        list[dict]
            List of n suggested experiments.

        Raises
        ------
        CostLimitError
            If estimated cost exceeds max_cost_per_call.
        LLMResponseError
            If LLM response cannot be parsed.
        """
        prompt = self._build_prompt(project, n, description, existing)

        estimated_cost = self.backend.estimate_cost(prompt)
        logger.info(f"Estimated cost: ${estimated_cost:.4f}")

        if estimated_cost > self.max_cost_per_call:
            raise CostLimitError(
                f"Estimated cost ${estimated_cost:.4f} exceeds limit "
                f"${self.max_cost_per_call:.2f}. Increase max_cost_per_call or "
                "reduce prompt size."
            )

        response = self.backend.complete(prompt)
        return self._parse_response(response, project)

    def _build_prompt(
        self,
        project: Project,
        n: int,
        description: str | None,
        existing: list[dict] | None,
    ) -> str:
        """Build the prompt from template and project spec.

        Parameters
        ----------
        project : Project
            The project spec.
        n : int
            Number of experiments to suggest.
        description : str | None
            Problem description.
        existing : list[dict] | None
            Existing experiments.

        Returns
        -------
        str
            Formatted prompt.
        """
        parameters_text = self._format_parameters(project)
        objective_text = self._format_objectives(project)
        description_text = description or f"Optimization for project '{project.name}'"

        prompt = self.prompt_template.format(
            n=n,
            objective=objective_text,
            parameters=parameters_text,
            description=description_text,
        )

        if existing:
            existing_text = "\n\n**Already planned/run:**\n"
            existing_text += json.dumps(existing, indent=2)
            prompt += existing_text

        return prompt

    def _format_parameters(self, project: Project) -> str:
        """Format project inputs as text for the prompt.

        Parameters
        ----------
        project : Project
            The project.

        Returns
        -------
        str
            Formatted parameters description.
        """
        lines = []
        for inp in project.inputs:
            if inp.type == "continuous":
                units = f" ({inp.units})" if inp.units else ""
                lo, hi = inp.bounds
                lines.append(f"- {inp.name}: continuous, range [{lo}, {hi}]{units}")
            elif inp.type == "categorical":
                lines.append(f"- {inp.name}: categorical, choices {inp.levels}")
        return "\n".join(lines)

    def _format_objectives(self, project: Project) -> str:
        """Format project targets as objective text.

        Parameters
        ----------
        project : Project
            The project.

        Returns
        -------
        str
            Formatted objectives description.
        """
        parts = []
        for cfg in project.target_configs:
            name = cfg.name or cfg.objective or "target"
            parts.append(f"{cfg.objective_mode} {name}")
        return ", ".join(parts)

    def _parse_response(self, response: str, project: Project) -> list[dict]:
        """Parse LLM response and validate against project schema.

        Extracts JSON from response (handling markdown code blocks),
        validates bounds, and coerces types.

        Parameters
        ----------
        response : str
            Raw LLM response text.
        project : Project
            Project for validation.

        Returns
        -------
        list[dict]
            Parsed and validated suggestions.

        Raises
        ------
        LLMResponseError
            If JSON cannot be extracted or parsed.
        """
        json_str = self._extract_json(response)

        try:
            suggestions = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"Failed to parse JSON from LLM response: {e}"
            ) from e

        if not isinstance(suggestions, list):
            raise LLMResponseError(
                f"Expected JSON array, got {type(suggestions).__name__}"
            )

        validated = []
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                logger.warning(f"Skipping non-dict suggestion: {suggestion}")
                continue
            validated.append(self._validate_suggestion(suggestion, project))

        return validated

    def _extract_json(self, response: str) -> str:
        """Extract JSON array from response, handling markdown code blocks.

        Parameters
        ----------
        response : str
            Raw response text.

        Returns
        -------
        str
            Extracted JSON string.

        Raises
        ------
        LLMResponseError
            If no JSON array found.
        """
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if code_block_match:
            return code_block_match.group(1).strip()

        array_match = re.search(r"\[[\s\S]*\]", response)
        if array_match:
            return array_match.group(0)

        raise LLMResponseError(
            "Could not find JSON array in LLM response. "
            f"Response preview: {response[:200]}..."
        )

    def _validate_suggestion(self, suggestion: dict, project: Project) -> dict:
        """Validate and coerce a single suggestion.

        Parameters
        ----------
        suggestion : dict
            Raw suggestion from LLM.
        project : Project
            Project for validation.

        Returns
        -------
        dict
            Validated suggestion with correct types and clamped bounds.
        """
        validated = {}
        input_specs = {inp.name: inp for inp in project.inputs}

        for name, spec in input_specs.items():
            if name not in suggestion:
                logger.warning(f"Missing parameter '{name}' in suggestion, skipping")
                continue

            value = suggestion[name]

            if spec.type == "continuous":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    logger.warning(
                        f"Could not convert '{name}' value '{value}' to float, "
                        f"using midpoint"
                    )
                    value = (spec.bounds[0] + spec.bounds[1]) / 2

                lo, hi = spec.bounds
                if value < lo:
                    logger.warning(f"Clamping '{name}' from {value} to lower {lo}")
                    value = lo
                elif value > hi:
                    logger.warning(f"Clamping '{name}' from {value} to upper {hi}")
                    value = hi

            elif spec.type == "categorical":
                value = str(value)
                if value not in spec.levels:
                    logger.warning(
                        f"Invalid level '{value}' for '{name}', using first level"
                    )
                    value = spec.levels[0]

            validated[name] = value

        return validated


def _estimate_tokens(text: str) -> int:
    """Rough estimate of token count.

    Uses simple heuristic of ~4 characters per token.

    Parameters
    ----------
    text : str
        Text to estimate.

    Returns
    -------
    int
        Estimated token count.
    """
    return len(text) // 4
