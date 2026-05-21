from __future__ import annotations

import math
import os
import unicodedata
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from gigaevo.config.schemas._base import FrozenStrictModel

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from langchain_openai import ChatOpenAI

    from gigaevo.utils.trackers.base import LogWriter


_OPENAI_API_KEY_ENV: str = "OPENAI_API_KEY"


def _sanitize_secret(value: str | None, source: str) -> str:
    """Return the secret stripped of surrounding whitespace, rejecting
    NUL bytes and any unicode general-category-Cc control codepoint.

    OpenAI-style API keys are pure printable ASCII; a NUL slipped in
    through an ``.env`` file truncates the key at the C boundary on
    some HTTP stacks, a wrapping ESC sequence ends up in log lines,
    and stray whitespace silently authenticates as the wrong tenant.
    Rejecting them at schema-load time turns a runtime auth failure
    into a typed error frame.
    """
    if value is None:
        raise ValueError(
            f"{source} must be set; got no value from explicit api_key or "
            f"the {_OPENAI_API_KEY_ENV} environment variable"
        )
    stripped = value.strip()
    if not stripped:
        raise ValueError(
            f"{source} must contain at least one non-whitespace character"
        )
    for ch in stripped:
        if ch == "\x00":
            raise ValueError(f"{source} contains a NUL byte and is rejected")
        if unicodedata.category(ch) == "Cc":
            raise ValueError(
                f"{source} contains control character U+{ord(ch):04X} "
                "and is rejected"
            )
    return stripped


class ChatOpenAIConfig(FrozenStrictModel):
    """Single LLM endpoint configured for OpenAI-compatible servers.

    Field names mirror the ``ChatOpenAI`` constructor surface validated
    by :func:`strict_chat_openai`. The API token is **not** a schema
    field: it is resolved inside :meth:`build` from an explicit
    ``api_key`` argument or from ``OPENAI_API_KEY``. Keeping the token
    off the schema means it cannot leak through tyro's ``--help``
    rendering, the dumped ``config.json``, ``__repr__``, or the
    ``experiment_id`` hash, and ``model_validate_json`` can re-hydrate
    a dumped config without the runtime environment.
    """

    model: str = Field(
        min_length=1,
        description="Model identifier accepted by the OpenAI-compatible endpoint.",
    )
    base_url: str | None = Field(
        default=None,
        description="Override the OpenAI HTTP base URL; required for self-hosted or proxy endpoints.",
    )
    temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Sampling temperature passed to the chat completion call.",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Upper bound on generated tokens per completion.",
    )
    request_timeout: float = Field(
        default=60.0,
        gt=0.0,
        description="HTTP request timeout in seconds.",
    )

    def build(self, *, api_key: str | None = None) -> ChatOpenAI:
        """Resolve the API token then construct the runtime client.

        Resolution order: explicit ``api_key`` argument, then
        ``OPENAI_API_KEY``. The resolved value is sanitised by
        :func:`_sanitize_secret` — whitespace stripped, NUL and control
        characters rejected — before reaching the HTTP layer.
        """
        from gigaevo.llm.strict_chat_openai import strict_chat_openai

        if api_key is not None:
            resolved = _sanitize_secret(api_key, source="api_key argument")
        else:
            env_value = os.environ.get(_OPENAI_API_KEY_ENV)
            resolved = _sanitize_secret(
                env_value, source=f"{_OPENAI_API_KEY_ENV} environment variable"
            )
        return strict_chat_openai(
            model=self.model,
            api_key=resolved,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout=self.request_timeout,
        )


class BanditRouterConfig(FrozenStrictModel):
    """UCB1 bandit-driven router over a static model pool.

    ``fitness_key`` and ``higher_is_better`` are typically supplied at
    construction by the experiment's ``ProblemContext`` via the
    cross-field validator on the root config; declared here as Optional
    so the schema can validate in isolation but ``build()`` requires
    them to be resolved to non-None values.
    """

    kind: Literal["bandit"] = "bandit"
    # ``default_factory=list`` lets tyro render help text for the
    # inactive ``LLMConfig`` union branch. ``validate_default=True``
    # promotes ``min_length=1`` to fire when a caller constructs
    # ``BanditRouterConfig()`` directly without a ``models`` argument;
    # the constructor exits with a typed error instead of silently
    # accepting a zero-arm bandit.
    models: list[ChatOpenAIConfig] = Field(
        default_factory=list,
        min_length=1,
        validate_default=True,
        description="Pool of endpoints the bandit picks among.",
    )
    skip_reward_on_acceptor_reject: bool = Field(
        default=False,
        description="When true, programs the acceptor rejects do not contribute reward updates.",
    )
    # ``1.41`` ~= sqrt(2), the canonical UCB1 exploration constant
    # balancing exploration vs exploitation for bounded reward signals.
    exploration_constant: float = Field(
        default=1.41,
        gt=0.0,
        description="UCB1 exploration coefficient; sqrt(2) is the canonical default.",
    )
    window_size: int = Field(
        default=100,
        ge=1,
        description="Length of the sliding reward window per arm.",
    )
    name: str = Field(
        default="default",
        min_length=1,
        description="Router label used in tracker metric paths.",
    )

    def build(
        self,
        *,
        fitness_key: str,
        higher_is_better: bool = True,
        writer: LogWriter | None = None,
    ) -> Runnable:
        from gigaevo.llm.bandit import BanditModelRouter

        endpoints = [m.build() for m in self.models]
        uniform = [1.0 / len(endpoints)] * len(endpoints)
        return BanditModelRouter(
            endpoints,
            uniform,
            writer=writer,
            name=self.name,
            exploration_constant=self.exploration_constant,
            window_size=self.window_size,
            fitness_key=fitness_key,
            higher_is_better=higher_is_better,
        )


class EnsembleRouterConfig(FrozenStrictModel):
    """Probability-weighted router (the runtime ``MultiModelRouter``).

    When ``probabilities`` is ``None`` the runtime applies a uniform
    distribution over ``models``. When provided, the after-validator
    enforces length parity, finiteness and positivity; the runtime
    normalises the weights to a probability distribution.
    """

    kind: Literal["ensemble"] = "ensemble"
    # ``default_factory=list`` lets tyro render help text for the
    # inactive ``LLMConfig`` union branch. ``validate_default=True``
    # promotes ``min_length=1`` to fire when a caller constructs
    # ``EnsembleRouterConfig()`` directly without a ``models``
    # argument; the constructor exits with a typed error instead of
    # silently accepting a zero-model ensemble.
    models: list[ChatOpenAIConfig] = Field(
        default_factory=list,
        min_length=1,
        validate_default=True,
        description="Endpoints the router samples from.",
    )
    probabilities: list[float] | None = Field(
        default=None,
        description="Per-endpoint weights parallel to models; None applies a uniform distribution.",
    )
    name: str = Field(
        default="default",
        min_length=1,
        description="Router label used in tracker metric paths.",
    )

    @model_validator(mode="after")
    def probabilities_aligned(self) -> EnsembleRouterConfig:
        if self.probabilities is None:
            return self
        if len(self.probabilities) != len(self.models):
            raise ValueError(
                f"probabilities length ({len(self.probabilities)}) "
                f"must equal models length ({len(self.models)})"
            )
        # ``p > 0`` alone admits ``float('inf')`` and ``float('nan')``;
        # the downstream normaliser would divide by ``sum(...)`` and
        # propagate the non-finite value into the runtime sampler.
        if any(not math.isfinite(p) or p <= 0 for p in self.probabilities):
            raise ValueError(
                "all probabilities must be finite and strictly positive"
            )
        return self

    def build(self, *, writer: LogWriter | None = None) -> Runnable:
        from gigaevo.llm.models import MultiModelRouter

        endpoints = [m.build() for m in self.models]
        weights = self.probabilities or [1.0 / len(endpoints)] * len(endpoints)
        return MultiModelRouter(endpoints, weights, writer=writer, name=self.name)


LLMConfig = Annotated[
    BanditRouterConfig | EnsembleRouterConfig,
    Field(discriminator="kind"),
]
