from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


class FrozenStrictModel(BaseModel):
    """Base class for every config schema.

    ``extra='forbid'`` turns a typo into a load-time ``ValidationError``;
    ``frozen=True`` makes the resolved config tree immutable after the
    CLI hands it to ``run_experiment``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


def reject_empty_or_cwd_path(field_name: str, value: Path | None) -> Path | None:
    """Reject the two ``Path`` values that are almost certainly a bug:
    the empty string (Pydantic coerces ``""`` to ``Path(".")``) and the
    bare current-working-directory placeholder.

    Field-level validators across the schema package share this exact
    test; centralising it keeps the rejection message uniform across
    ``log_dir``, ``prompts_dir``, ``output_dir``, ``fallback_prompts_dir``
    and ``problem_dir``.
    """
    if value is None:
        return value
    if str(value) in ("", "."):
        raise ValueError(
            f"{field_name}: path must be real and non-empty; "
            f"got {value!r} which resolves to the current working directory"
        )
    return value


# ---------------------------------------------------------------------------
# Float aliases
#
# Every float field across the schema package must opt out of inf/nan
# acceptance — without ``allow_inf_nan=False`` Pydantic happily binds
# ``float('inf')`` to a ``gt=0.0`` constraint and the downstream consumer
# multiplies/compares against an infinity it has no defence against.
# Centralising the opt-out as an ``Annotated`` alias keeps the field
# declarations short and ensures the lint test in ``tests/config/`` can
# walk every float field and assert finiteness uniformly.
# ---------------------------------------------------------------------------

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
"""Plain finite float; use when neither sign nor zero needs constraining."""

FinitePositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
"""Strictly positive finite float (``> 0``)."""

FiniteNonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
"""Non-negative finite float (``>= 0``)."""


# ---------------------------------------------------------------------------
# String aliases
#
# ``min_length=1`` is the conventional guard against empty strings but it
# does not reject whitespace-only values or strings containing ASCII
# control bytes (NUL, ESC, ...). Control characters in host names, Redis
# key prefixes and log-rotation specifiers either break the consumer
# library or open NUL-injection routes; whitespace-only values pass an
# emptiness probe while still misbehaving everywhere they are used as an
# identifier or substring.
# ---------------------------------------------------------------------------


def _reject_control_chars(value: str) -> str:
    """Reject ASCII control characters except ``\\t``, ``\\n``, ``\\r``.

    NUL (``\\x00``) is the highest-impact case — many C-backed network
    libraries truncate at NUL while their Python wrappers do not, which
    turns a stringified hostname into a covert protocol-confusion vector.
    """
    for ch in value:
        codepoint = ord(ch)
        if codepoint < 0x20 and ch not in ("\t", "\n", "\r"):
            raise ValueError(
                f"value contains control character U+{codepoint:04X} "
                "which is rejected by the schema"
            )
        if codepoint == 0x7F:
            raise ValueError(
                "value contains DEL (U+007F) which is rejected by the schema"
            )
    return value


def _reject_blank(value: str) -> str:
    """Reject strings that are non-empty but consist only of whitespace."""
    if value.strip() == "":
        raise ValueError(
            "value must contain at least one non-whitespace character"
        )
    return value


NoControlCharsStr = Annotated[str, AfterValidator(_reject_control_chars)]
"""String that rejects ASCII control characters (NUL, ESC, ...)."""

NonBlankStr = Annotated[
    str,
    AfterValidator(_reject_control_chars),
    AfterValidator(_reject_blank),
]
"""Non-blank, control-character-free string.

Use for every identifier-shaped field (hostnames, key prefixes, fitness
keys, log specifiers) where ``min_length=1`` was the previous guard.
"""
