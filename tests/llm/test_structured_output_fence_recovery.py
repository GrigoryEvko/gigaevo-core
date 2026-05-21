"""Regression tests for markdown-fenced JSON recovery in structured output.

Sonnet / Gemini frequently wrap structured replies in ``` ```json … ``` ```
fences even under a ``response_format={"type":"json_object"}`` contract.
The ``_StructuredOutputRouter._process`` path retries
``model_validate_json`` on a fence-stripped payload before treating the
call as a genuine ``parsing_error``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel, ValidationError
import pytest

from gigaevo.llm.models import (
    MultiModelRouter,
    _StructuredOutputRouter,
    _strip_markdown_fences,
)
from tests.conftest import NullWriter


class _Schema(BaseModel):
    name: str
    score: float


def _mock_model(name: str) -> MagicMock:
    m = MagicMock()
    m.model_name = name
    m.with_structured_output = MagicMock(return_value=MagicMock())
    return m


class TestStripMarkdownFences:
    def test_strips_json_language_tag(self) -> None:
        text = '```json\n{"name": "a", "score": 0.5}\n```'
        assert _strip_markdown_fences(text) == '{"name": "a", "score": 0.5}'

    def test_strips_bare_fences(self) -> None:
        text = '```\n{"name": "a", "score": 0.5}\n```'
        assert _strip_markdown_fences(text) == '{"name": "a", "score": 0.5}'

    def test_strips_with_surrounding_whitespace(self) -> None:
        text = '   ```json\n{"x": 1}\n```   '
        assert _strip_markdown_fences(text) == '{"x": 1}'

    def test_passes_through_unfenced(self) -> None:
        text = '{"name": "a", "score": 0.5}'
        assert _strip_markdown_fences(text) == text

    def test_non_string_passthrough(self) -> None:
        assert _strip_markdown_fences(None) is None  # type: ignore[arg-type]
        assert _strip_markdown_fences(42) == 42  # type: ignore[arg-type]

    def test_only_strips_outermost_fence(self) -> None:
        # A JSON string literal containing triple-backticks survives.
        text = '```json\n{"note": "see ```inline``` block"}\n```'
        stripped = _strip_markdown_fences(text)
        assert stripped == '{"note": "see ```inline``` block"}'


class TestStructuredOutputFenceRecovery:
    """``_process`` re-validates a fence-stripped payload before raising."""

    def _router(self, schema: Any = _Schema) -> _StructuredOutputRouter:
        return _StructuredOutputRouter(
            models=[],
            model_names=[],
            probabilities=[],
            langfuse=None,
            tracker=MagicMock(),
            schema=schema,
        )

    def test_recovers_fenced_pydantic_payload(self) -> None:
        raw = MagicMock()
        raw.content = '```json\n{"name": "spec", "score": 0.7}\n```'
        # Simulate langchain's include_raw=True ValidationError surface.
        try:
            _Schema.model_validate_json(raw.content)
        except ValidationError as exc:
            parse_err = exc
        else:  # pragma: no cover — defensive
            pytest.fail("test setup: fenced payload unexpectedly parsed")

        router = self._router()
        result = router._process(
            {"raw": raw, "parsed": None, "parsing_error": parse_err},
            "test-model",
        )
        assert isinstance(result, _Schema)
        assert result.name == "spec"
        assert result.score == pytest.approx(0.7)

    def test_raises_when_no_fence_and_payload_invalid(self) -> None:
        raw = MagicMock()
        raw.content = '{"name": 42, "score": "not a number"}'
        try:
            _Schema.model_validate_json(raw.content)
        except ValidationError as exc:
            parse_err = exc
        else:  # pragma: no cover
            pytest.fail("test setup: invalid payload unexpectedly parsed")

        router = self._router()
        with pytest.raises(ValidationError):
            router._process(
                {"raw": raw, "parsed": None, "parsing_error": parse_err},
                "test-model",
            )

    def test_no_schema_skips_recovery(self) -> None:
        raw = MagicMock()
        raw.content = '```json\n{"name": "spec", "score": 0.7}\n```'
        parse_err = ValueError("nope")
        router = self._router(schema=None)
        with pytest.raises(ValueError):
            router._process(
                {"raw": raw, "parsed": None, "parsing_error": parse_err},
                "test-model",
            )

    def test_non_pydantic_schema_skips_recovery(self) -> None:
        raw = MagicMock()
        raw.content = '```json\n{"x": 1}\n```'
        parse_err = ValueError("nope")
        # dict schema (TypedDict in real langchain usage) is not a
        # ``BaseModel`` subclass; recovery should be a no-op.
        router = self._router(schema=dict)
        with pytest.raises(ValueError):
            router._process(
                {"raw": raw, "parsed": None, "parsing_error": parse_err},
                "test-model",
            )

    def test_with_structured_output_threads_schema_through_router(self) -> None:
        models = [_mock_model("a")]
        router = MultiModelRouter(models, [1.0], writer=NullWriter(), name="t")
        structured = router.with_structured_output(_Schema)
        assert structured._schema is _Schema
