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

    def test_recovers_when_parsing_error_absent(self) -> None:
        """Some langchain versions return ``{parsed: None, parsing_error: None}``
        when the inner parser swallows the error. The recovery path must
        still fire on a fenced payload."""
        raw = MagicMock()
        raw.content = '```json\n{"name": "spec", "score": 0.5}\n```'
        router = self._router()
        result = router._process(
            {"raw": raw, "parsed": None, "parsing_error": None},
            "test-model",
        )
        assert isinstance(result, _Schema)
        assert result.score == pytest.approx(0.5)

    def test_recovers_unfenced_payload_with_whitespace(self) -> None:
        """Plain JSON with stray leading whitespace also recovers — the
        original parser may fail for non-fence reasons but the JSON body
        itself is valid."""
        raw = MagicMock()
        raw.content = '\n\n   {"name": "spec", "score": 0.25}   \n'
        router = self._router()
        result = router._process(
            {"raw": raw, "parsed": None, "parsing_error": None},
            "test-model",
        )
        assert isinstance(result, _Schema)
        assert result.score == pytest.approx(0.25)

    def test_raises_typed_value_error_when_recovery_fails(self) -> None:
        """When ``parsing_error`` is missing AND recovery returns ``None``,
        ``_process`` raises a typed ``ValueError`` so the call site routes
        through the standard failure path (instead of silently returning
        ``None`` which would skip the failure_hook)."""
        raw = MagicMock()
        raw.content = "this is not JSON at all"
        router = self._router()
        with pytest.raises(ValueError, match="parsed=None"):
            router._process(
                {"raw": raw, "parsed": None, "parsing_error": None},
                "test-model",
            )

    def test_recover_from_invoke_exception_uses_llm_output(self) -> None:
        """When langchain raises an exception that carries the raw model
        output on ``llm_output`` (``OutputParserException`` style), the
        invoke-side recovery path retries the schema against the
        fence-stripped payload."""
        router = self._router()

        class _ParserExc(Exception):
            pass

        exc = _ParserExc("parse failed")
        exc.llm_output = '```json\n{"name": "spec", "score": 0.9}\n```'
        recovered = router._recover_from_invoke_exception(exc)
        assert isinstance(recovered, _Schema)
        assert recovered.score == pytest.approx(0.9)

    def test_recover_from_invoke_exception_uses_validation_error_input(self) -> None:
        """Pydantic v2 ``ValidationError`` carries the offending payload on
        ``input``; the invoke-side recovery path looks there too."""
        router = self._router()
        try:
            _Schema.model_validate({"name": "spec", "score": "not a number"})
        except ValidationError as e:
            ve = e
        else:  # pragma: no cover
            pytest.fail("test setup: malformed payload unexpectedly parsed")
        ve.input = '```json\n{"name": "spec", "score": 1.0}\n```'  # type: ignore[attr-defined]
        recovered = router._recover_from_invoke_exception(ve)
        assert isinstance(recovered, _Schema)
        assert recovered.score == pytest.approx(1.0)

    def test_recover_from_invoke_exception_returns_none_without_payload(self) -> None:
        """A bare exception with no usable payload yields ``None`` so the
        original error propagates."""
        router = self._router()
        assert router._recover_from_invoke_exception(RuntimeError("network")) is None
