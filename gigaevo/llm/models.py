from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import ContextVar
import os
import random
import re
import threading
import time
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse, urlunparse

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from loguru import logger
from pydantic import BaseModel, ValidationError

from gigaevo.llm.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitOpenError,
    LLMCircuitBreaker,
)
from gigaevo.llm.token_tracking import TokenTracker
from gigaevo.utils.text_sanitize import clean_identifier, sanitize_for_log
from gigaevo.utils.trackers.base import LogWriter

if TYPE_CHECKING:
    from gigaevo.programs.program import Program


_selected_model_var: ContextVar[str | None] = ContextVar("selected_model", default=None)


# Match an outer markdown code fence — optional language tag (``json``,
# ``JSON``, ``yaml`` …), surrounding newlines / whitespace, and a trailing
# closing fence. Sonnet / Gemini routinely wrap structured replies in
# fences even under ``response_format=json_object``; stripping them lets a
# subsequent ``model_validate_json`` succeed without re-issuing the call.
_MARKDOWN_FENCE_PATTERN = re.compile(
    r"^\s*```(?:[a-zA-Z0-9_-]+)?\s*\n?(?P<body>.*?)\n?```\s*$",
    re.DOTALL,
)


def _strip_markdown_fences(text: str) -> str:
    """Return ``text`` with one outer markdown code fence removed.

    Idempotent: input without a fence pair is returned verbatim. Only the
    outermost fence is stripped — nested fences inside the JSON body (a
    string literal containing triple-backticks, for example) survive.
    """
    if not isinstance(text, str):
        return text
    match = _MARKDOWN_FENCE_PATTERN.match(text)
    if match is None:
        return text
    return match.group("body")


def _redact_url(url: str) -> str:
    """Strip userinfo (user:password@) from a URL before logging. Other
    URL components are preserved verbatim. Returns the input unchanged
    on parse failure."""
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    if not parsed.hostname:
        return url
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _safe_model_name(raw: object) -> str:
    """Validate a model name read off a ChatOpenAI instance. Strips control
    characters and ANSI; logs a one-shot WARNING if the cleaning changes
    the input so operators notice a misconfigured identifier."""
    raw_str = str(raw) if raw is not None else ""
    cleaned = clean_identifier(raw_str, max_len=128)
    if cleaned != raw_str:
        logger.warning(
            "[MultiModelRouter] model_name sanitized: {!r} -> {!r}",
            sanitize_for_log(raw_str),
            cleaned,
        )
    return cleaned


def get_selected_model() -> str | None:
    """Return the last selected model name for the current async context."""
    return _selected_model_var.get()


def _remember_selected_model(model_name: str) -> None:
    _selected_model_var.set(model_name)


# ---------------------------------------------------------------------------
# Startup model-verification helpers
# ---------------------------------------------------------------------------
#
# Multiple ``MultiModelRouter`` instances and multiple driver processes
# all call ``_verify_models`` from ``__init__``.  Without coordination
# that synchronous burst against ``GET {base_url}/models`` is a textbook
# thundering herd at boot.  Within a single process the cache below
# collapses repeat probes; across processes the jitter spreads them.

_VERIFY_CACHE_TTL_SUCCESS_S = 300.0
"""Cache successful probes for 5 minutes.  Re-probing more often than
this adds no value — the upstream's model list changes on the timescale
of deployments."""

_VERIFY_CACHE_TTL_FAILURE_S = 30.0
"""Initial failure TTL.  Subsequent consecutive failures back off as
``_VERIFY_CACHE_TTL_FAILURE_S * 2**(n-1)``, capped at
``_VERIFY_CACHE_TTL_FAILURE_MAX_S`` so a permanently broken endpoint
stops re-burning the jitter sleep on every router instantiation."""

_VERIFY_CACHE_TTL_FAILURE_MAX_S = 3600.0
"""Upper bound on the failure-cache TTL — after enough consecutive
failures the cache holds the result for at most one hour, then probes
again. A successful probe resets the counter so the next failure
starts over at the base TTL."""

_VERIFY_JITTER_MAX_S = 2.0
"""Each probe sleeps a uniform [0, max] seconds before issuing the GET so
concurrent router constructions land staggered, not synchronously."""

# Maps ``base_url`` -> (cached_at_monotonic, available_models_or_None).
_verify_cache: dict[str, tuple[float, frozenset[str] | None]] = {}
# Per-``base_url`` consecutive-failure count; reset on first success.
_verify_failure_counts: dict[str, int] = {}
_verify_cache_lock = threading.Lock()


def _failure_ttl_for(consecutive_failures: int) -> float:
    """Return the failure-cache TTL for ``consecutive_failures`` strikes.

    Exponential backoff: ``base * 2**(n-1)`` for ``n>=1``, clamped at
    the configured ceiling. ``n=0`` (no recorded failure) collapses to
    the base TTL so a fresh entry never spends zero seconds cached.
    """
    if consecutive_failures <= 0:
        return _VERIFY_CACHE_TTL_FAILURE_S
    scaled = _VERIFY_CACHE_TTL_FAILURE_S * (2 ** (consecutive_failures - 1))
    return min(scaled, _VERIFY_CACHE_TTL_FAILURE_MAX_S)


def _model_base_url(model: ChatOpenAI) -> str | None:
    """Pick the OpenAI-compatible base URL off a langchain model object.

    Different langchain-openai versions surface this as ``base_url`` or
    ``openai_api_base``; check both."""
    return getattr(model, "base_url", None) or getattr(model, "openai_api_base", None)


def _model_api_key(model: ChatOpenAI) -> str | None:
    """Return the API key as plaintext for the ``Authorization`` header.

    ``ChatOpenAI.openai_api_key`` is a ``pydantic.SecretStr``;
    ``.get_secret_value()`` unwraps it.  ``None`` if the model has no
    key configured (in-cluster vLLM / SGLang typically need none)."""
    secret = getattr(model, "openai_api_key", None)
    if secret is None:
        return None
    try:
        return secret.get_secret_value()
    except AttributeError:
        return str(secret) or None


def _fetch_available_models_at(
    base_url: str, api_key: str | None
) -> frozenset[str] | None:
    """Return the set of model ids advertised by an OpenAI-compatible server.

    Process-wide TTL-cached + jittered.  Sends ``Authorization: Bearer``
    when an API key is configured so authenticated providers
    (openai.com, OpenRouter) return real results instead of 401.

    Returns ``None`` on probe failure; the caller is expected to skip
    verification logging for that base URL rather than treat ``None``
    as "no models available".
    """
    now = time.monotonic()
    with _verify_cache_lock:
        cached = _verify_cache.get(base_url)
        if cached is not None:
            cached_at, cached_value = cached
            if cached_value is not None:
                ttl = _VERIFY_CACHE_TTL_SUCCESS_S
            else:
                ttl = _failure_ttl_for(_verify_failure_counts.get(base_url, 0))
            if (now - cached_at) < ttl:
                return cached_value

    # Stagger probes across coexisting routers / drivers.
    time.sleep(random.uniform(0, _VERIFY_JITTER_MAX_S))

    from gigaevo.infra.requests_factory import make_requests_session

    headers: dict[str, str] | None = None
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}
    session = make_requests_session("model_verify", timeout=(5.0, 10.0))
    available: frozenset[str] | None
    try:
        response = session.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()
        data = response.json()
        available = frozenset(
            d["id"] for d in data.get("data", []) if isinstance(d, dict) and "id" in d
        )
    except Exception as exc:
        # Log severity scales down on repeat failures: a permanently
        # broken endpoint should not spam WARNING on every router
        # instantiation. The first failure stays WARNING; subsequent
        # failures within the failure-TTL window get one DEBUG line so
        # the failure mode is still discoverable but does not drown the
        # log.
        with _verify_cache_lock:
            prior_failures = _verify_failure_counts.get(base_url, 0)
        if prior_failures == 0:
            logger.warning(
                "[MultiModelRouter] Cannot verify models at {}: {}", base_url, exc
            )
        else:
            logger.debug(
                "[MultiModelRouter] Cannot verify models at {} (failure #{}): {}",
                base_url,
                prior_failures + 1,
                exc,
            )
        available = None
    finally:
        session.close()

    with _verify_cache_lock:
        _verify_cache[base_url] = (time.monotonic(), available)
        if available is None:
            _verify_failure_counts[base_url] = (
                _verify_failure_counts.get(base_url, 0) + 1
            )
        else:
            # First success after a failure run resets the counter so
            # the next failure starts over at the base TTL rather than
            # immediately jumping back to the prior backoff.
            _verify_failure_counts.pop(base_url, None)
    return available


def _create_langfuse_handler() -> CallbackHandler | None:
    """Create Langfuse handler if credentials are configured."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None

    handler = CallbackHandler()
    handler.client.flush_at = 1  # type: ignore[attr-defined]
    handler.client.flush_interval = 1  # type: ignore[attr-defined]
    logger.info("[MultiModelRouter] Langfuse tracing enabled")
    return handler


def _with_langfuse(
    config: RunnableConfig | None,
    handler: CallbackHandler | None,
    model_name: str | None = None,
) -> RunnableConfig | None:
    """Add Langfuse handler and metadata to config."""
    if handler is None:
        return config

    # dict() is a shallow copy: cfg["callbacks"] and cfg["metadata"] would
    # alias the caller's mutable list/dict. Copy both defensively up-front
    # so the function never mutates its input, regardless of which branch
    # below fires (or whether downstream code mutates the returned config).
    cfg: dict[str, Any] = dict(config or {})
    cfg["callbacks"] = list(cfg.get("callbacks") or [])
    if "metadata" in cfg:
        cfg["metadata"] = dict(cfg["metadata"] or {})

    if handler not in cfg["callbacks"]:
        cfg["callbacks"].append(handler)

    if model_name:
        cfg.setdefault("metadata", {})["selected_model"] = model_name

    return cast(RunnableConfig, cfg)


class MultiModelRouter(Runnable):
    """Probabilistic model router with token tracking and Langfuse tracing.

    Example:
        >>> router = MultiModelRouter(
        ...     [ChatOpenAI(model="gpt-4"), ChatOpenAI(model="gpt-3.5-turbo")],
        ...     [0.8, 0.2],
        ...     writer=metrics_writer,
        ...     name="mutation",  # metrics go to llm/tokens/mutation/...
        ... )
        >>> response = await router.ainvoke("Hello!")
        >>> structured = router.with_structured_output(MySchema)
    """

    def __init__(
        self,
        models: list[ChatOpenAI],
        probabilities: list[float],
        writer: LogWriter | None = None,
        name: str = "default",
        *,
        circuit_breaker: LLMCircuitBreaker | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        if len(models) != len(probabilities):
            raise ValueError(
                f"Length mismatch: {len(models)} models, {len(probabilities)} probabilities"
            )
        if any(p <= 0 for p in probabilities):
            raise ValueError("All probabilities must be positive")

        # Inject a pre-built breaker (production code threads the
        # config-derived instance through here), or fall back to a
        # default-configured per-router breaker.  ``None`` for both is
        # the legacy path; callers that have not opted in keep the
        # pre-breaker behaviour.
        if circuit_breaker is not None:
            self._circuit_breaker = circuit_breaker
        elif circuit_breaker_config is not None:
            self._circuit_breaker = LLMCircuitBreaker(
                name=name, config=circuit_breaker_config
            )
        else:
            self._circuit_breaker = LLMCircuitBreaker(name=name)
        self.models = models
        # ChatOpenAI.model_name comes from operator config / env interpolation
        # / occasionally LLM-generated overrides; control characters there
        # would propagate into every loguru ``{}`` substitution in this
        # module and into langfuse trace identifiers. Validate once at
        # construction.
        self.model_names = [_safe_model_name(m.model_name) for m in models]
        self.probabilities = [p / sum(probabilities) for p in probabilities]
        self._task_model_map: dict[int, str] = {}
        self._name = name

        self._tracker = TokenTracker(
            name=name,
            writer=writer.bind(path=["llm", "tokens"]) if writer else None,
        )
        self._langfuse = _create_langfuse_handler()

        model_desc = ", ".join(
            f"{n} ({p:.0%})" for n, p in zip(self.model_names, self.probabilities)
        )
        logger.info(
            "[MultiModelRouter:{}] Initialized with {} models: {}",
            name,
            len(models),
            model_desc,
        )
        # Log base URLs for debugging server connectivity
        for m, safe_name in zip(models, self.model_names):
            # ChatOpenAI exposes base_url as a property (langchain 0.1+)
            base_url = getattr(m, "base_url", None)
            if base_url:
                logger.info(
                    "[MultiModelRouter:{}] Model {} at {}",
                    name,
                    safe_name,
                    _redact_url(sanitize_for_log(str(base_url))),
                )

        self._verify_models()

    def _verify_models(self) -> None:
        """Best-effort startup probe — verify configured models exist on servers.

        Routed through :func:`_fetch_available_models_at`, which adds a
        process-wide TTL cache + random jitter before each probe so
        multiple routers and multiple drivers don't synchronously
        hammer the inference server's ``/models`` endpoint at boot.
        Sends the model's API key as ``Authorization: Bearer …`` so the
        probe works against authenticated providers (OpenAI, OpenRouter)
        instead of silently logging 401s and continuing.
        """
        checked: set[str] = set()
        for model in self.models:
            base_url = _model_base_url(model)
            # ``isinstance(str)`` short-circuits mock objects that return
            # truthy ``MagicMock`` attributes — those models never have a
            # real URL to probe, and exercising the network path with a
            # mock-stringified URL just produces noise in test logs.
            if not isinstance(base_url, str) or base_url in checked:
                continue
            checked.add(base_url)
            # Union: #17's restructured loop (TTL cache + jitter + auth via
            # ``_fetch_available_models_at``) plus #10's sanitisation +
            # base_url redaction for log lines.
            api_key = _model_api_key(model)
            available = _fetch_available_models_at(base_url, api_key)
            if available is None:
                continue  # probe failure already logged at the helper layer
            safe_base_url = _redact_url(sanitize_for_log(str(base_url)))
            for m in self.models:
                if _model_base_url(m) != base_url:
                    continue
                safe_name = sanitize_for_log(str(m.model_name))
                if m.model_name in available:
                    logger.info(
                        "[MultiModelRouter:{}] Model {} verified on {}",
                        self._name,
                        safe_name,
                        safe_base_url,
                    )
                else:
                    logger.warning(
                        "[MultiModelRouter:{}] Model {} NOT FOUND on {}. "
                        "Available: {}",
                        self._name,
                        safe_name,
                        safe_base_url,
                        sorted(available),
                    )

    @staticmethod
    def _current_task_id() -> int | None:
        """Return ``id(asyncio.current_task())`` or *None* outside an event loop."""
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return None
        return id(task) if task is not None else None

    def _select(self) -> tuple[ChatOpenAI, str]:
        """Select a model based on probabilities."""
        idx = random.choices(range(len(self.models)), weights=self.probabilities)[0]
        model, name = self.models[idx], self.model_names[idx]
        _remember_selected_model(name)
        tid = self._current_task_id()
        if tid is not None:
            self._task_model_map[tid] = name
        return model, name

    def get_last_model(self) -> str | None:
        """Return the model name selected in the most recent ``_select()`` call for the current async task."""
        tid = self._current_task_id()
        if tid is not None:
            return self._task_model_map.pop(tid, None)
        return None

    def on_mutation_outcome(
        self,
        program: Program,
        parents: list[Program],
        outcome: Any = None,
    ) -> None:
        """Callback when a mutated program completes evaluation. Override for feedback."""

    def _config(
        self, config: RunnableConfig | None, model_name: str
    ) -> RunnableConfig | None:
        return _with_langfuse(config, self._langfuse, model_name)

    @property
    def circuit_breaker(self) -> LLMCircuitBreaker:
        return self._circuit_breaker

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> BaseMessage:
        self._circuit_breaker.guard()
        model, name = self._select()
        try:
            response = model.invoke(input, self._config(config, name), **kwargs)
        except CircuitOpenError:
            # Already accounted by ``guard``; re-raise so upstream sees
            # the canonical open-breaker error.
            raise
        except Exception:
            self._circuit_breaker.record_failure()
            raise
        self._circuit_breaker.record_success()
        self._tracker.track(response, name)
        return response

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> BaseMessage:
        self._circuit_breaker.guard()
        model, name = self._select()
        try:
            response = await model.ainvoke(
                input, self._config(config, name), **kwargs
            )
        except CircuitOpenError:
            raise
        except Exception:
            self._circuit_breaker.record_failure()
            raise
        self._circuit_breaker.record_success()
        self._tracker.track(response, name)
        return response

    def stream(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> Iterator[BaseMessage]:
        model, name = self._select()
        last = None
        for chunk in model.stream(input, self._config(config, name), **kwargs):
            last = chunk
            yield chunk
        if last:
            self._tracker.track(last, name)

    async def astream(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> AsyncIterator[BaseMessage]:
        model, name = self._select()
        last = None
        async for chunk in model.astream(input, self._config(config, name), **kwargs):
            last = chunk
            yield chunk
        if last:
            self._tracker.track(last, name)

    def with_structured_output(self, schema: Any, **kwargs) -> _StructuredOutputRouter:
        """Create a router that returns parsed Pydantic models with token tracking.

        ``schema`` is forwarded to the underlying ``ChatOpenAI`` wrappers
        *and* retained on the resulting router so the fallback parser can
        re-validate a fence-stripped payload on a ``parsing_error``
        without re-issuing the LLM call. The shared circuit breaker is
        also threaded through so the structured-output path participates
        in the same open / half-open lifecycle.
        """
        wrapped = [
            m.with_structured_output(schema, include_raw=True, **kwargs)
            for m in self.models
        ]
        return _StructuredOutputRouter(
            wrapped,
            self.model_names,
            self.probabilities,
            self._langfuse,
            self._tracker,
            task_model_map=self._task_model_map,
            schema=schema,
            circuit_breaker=self._circuit_breaker,
        )


class _StructuredOutputRouter(Runnable):
    """Router for structured output with token tracking from raw responses."""

    def __init__(
        self,
        models: list,
        model_names: list[str],
        probabilities: list[float],
        langfuse: CallbackHandler | None,
        tracker: TokenTracker,
        task_model_map: dict[int, str] | None = None,
        select_override: Callable[[], tuple[Any, str]] | None = None,
        failure_hook: Callable[[BaseException, str], None] | None = None,
        schema: Any | None = None,
        circuit_breaker: LLMCircuitBreaker | None = None,
    ):
        self._models = models
        self._names = model_names
        self._probs = probabilities
        self._langfuse = langfuse
        self._tracker = tracker
        self._task_model_map = task_model_map
        self._select_override = select_override
        # Called when ``model.{,a}invoke`` raises. ``BanditModelRouter`` uses
        # it to inject a zero reward into the ledger so a failed pull does
        # not silently inflate ``total_pulls`` without a matching window
        # entry. The hook receives the exception and the selected arm name;
        # it must not re-raise (the original exception still propagates).
        self._failure_hook = failure_hook
        # Retained so ``_process`` can re-validate a fence-stripped payload
        # against the original Pydantic schema when langchain surfaces a
        # ``parsing_error``. ``None`` for dict / TypedDict schemas — the
        # fallback parser is skipped in that case.
        self._schema = schema
        # Optional breaker reference; ``None`` means "structured path
        # operates breakerless" (test paths that built the router
        # directly). Production wiring threads in the parent router's
        # breaker so the structured-output path participates in the
        # same lifecycle.
        self._circuit_breaker = circuit_breaker

    def _select(self) -> tuple[Any, str]:
        if self._select_override is not None:
            return self._select_override()
        idx = random.choices(range(len(self._models)), weights=self._probs)[0]
        model, name = self._models[idx], self._names[idx]
        _remember_selected_model(name)
        if self._task_model_map is not None:
            tid = MultiModelRouter._current_task_id()
            if tid is not None:
                self._task_model_map[tid] = name
        return model, name

    def _config(
        self, config: RunnableConfig | None, model_name: str
    ) -> RunnableConfig | None:
        return _with_langfuse(config, self._langfuse, model_name)

    def _process(self, response: dict, name: str) -> Any:
        if raw := response.get("raw"):
            self._tracker.track(raw, name)
        parsing_error = response.get("parsing_error")
        parsed = response.get("parsed")
        # langchain's ``include_raw=True`` typically surfaces schema-
        # validation failures as ``response['parsing_error']`` with
        # ``parsed=None``. Sonnet / Gemini frequently wrap structured
        # replies in markdown fences (``` ```json …``` ```) even under
        # ``response_format`` contracts; the inner parser then chokes on
        # the fence chars. A second class of failure has the inner
        # parser swallow the error and return ``{parsed: None,
        # parsing_error: None}`` — both shapes need the same tolerant
        # re-parse against the schema before we treat the call as a
        # genuine failure.
        if parsed is None:
            recovered = self._recover_from_parsing_error(raw)
            if recovered is not None:
                return recovered
            # Returning ``None`` here would silently bypass the caller's
            # ``try / except`` and the bandit's failure_hook would never
            # fire — the pull was recorded by ``_select`` but the reward
            # window would never get a matching entry. Raise the
            # parsing_error when langchain surfaced one, otherwise raise
            # a typed ValueError so the failure path is uniform.
            if parsing_error is not None:
                raise parsing_error
            raise ValueError(
                "_StructuredOutputRouter: structured response had "
                "parsed=None and no parsing_error; tolerant re-parse "
                "also failed."
            )
        return parsed

    def _recover_from_parsing_error(self, raw: Any) -> Any:
        """Best-effort re-validate ``raw`` content against ``self._schema``.

        Returns the parsed model on success, ``None`` when the schema is
        unknown, the raw payload is not a string-bearing message, or
        re-validation fails. The caller treats ``None`` as "no recovery
        possible" and raises the original ``parsing_error``.

        Attempts two payloads: the fence-stripped body (if a fence was
        present) and the original text. The second attempt covers
        responses where the inner parser failed for non-fence reasons
        (extra whitespace / preamble) but the JSON body still validates.
        """
        if self._schema is None or not isinstance(self._schema, type):
            return None
        if not issubclass(self._schema, BaseModel):
            return None
        text = self._extract_text(raw)
        if not text:
            return None
        candidates = []
        stripped = _strip_markdown_fences(text)
        if stripped != text:
            candidates.append(stripped)
        candidates.append(text.strip())
        for candidate in candidates:
            try:
                return self._schema.model_validate_json(candidate)
            except ValidationError:
                continue
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_text(raw: Any) -> str | None:
        """Return the string content of a langchain message-like ``raw``.

        Handles the common shapes: ``BaseMessage`` (``.content`` attr),
        ``dict`` envelopes (``content`` / ``text`` keys), and bare strings.
        """
        if raw is None:
            return None
        if isinstance(raw, str):
            return raw
        content = getattr(raw, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(raw, dict):
            for key in ("content", "text"):
                value = raw.get(key)
                if isinstance(value, str):
                    return value
        return None

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> Any:
        if self._circuit_breaker is not None:
            self._circuit_breaker.guard()
        model, name = self._select()
        try:
            response = model.invoke(input, self._config(config, name), **kwargs)
            # ``_process`` runs the token tracker and unwraps the parsed
            # Pydantic object. Either step can raise (telemetry-side bug,
            # malformed structured response, missing parsed field). Treat
            # those failures as call failures for ledger-symmetry purposes
            # so the failure_hook fires.
            result = self._process(response, name)
        except BaseException as exc:
            recovered = self._recover_from_invoke_exception(exc)
            if recovered is not None:
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()
                return recovered
            if not isinstance(exc, CircuitOpenError) and self._circuit_breaker is not None:
                self._circuit_breaker.record_failure()
            self._maybe_fire_failure_hook(exc, name)
            raise
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_success()
        return result

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> Any:
        if self._circuit_breaker is not None:
            self._circuit_breaker.guard()
        model, name = self._select()
        try:
            response = await model.ainvoke(input, self._config(config, name), **kwargs)
            result = self._process(response, name)
        except BaseException as exc:
            recovered = self._recover_from_invoke_exception(exc)
            if recovered is not None:
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()
                return recovered
            if not isinstance(exc, CircuitOpenError) and self._circuit_breaker is not None:
                self._circuit_breaker.record_failure()
            self._maybe_fire_failure_hook(exc, name)
            raise
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_success()
        return result

    def _recover_from_invoke_exception(self, exc: BaseException) -> Any:
        """When the langchain wrapper raises a parse/validation error
        directly (instead of routing it through ``parsing_error`` on the
        ``include_raw=True`` envelope), try to recover by re-parsing
        text reachable from the exception. ``langchain_core``'s
        ``OutputParserException`` carries the raw model output on
        ``llm_output``; pydantic's ``ValidationError`` may carry it via
        ``input`` on Pydantic v2.

        Returns the parsed schema on success, ``None`` otherwise. The
        caller treats ``None`` as "no recovery" and lets the original
        exception propagate.
        """
        if self._schema is None or not isinstance(self._schema, type):
            return None
        if not issubclass(self._schema, BaseModel):
            return None
        for attr in ("llm_output", "input"):
            payload = getattr(exc, attr, None)
            recovered = self._recover_from_parsing_error(payload)
            if recovered is not None:
                return recovered
        return None

    def _maybe_fire_failure_hook(self, exc: BaseException, name: str) -> None:
        if self._failure_hook is None:
            return
        try:
            self._failure_hook(exc, name)
        except Exception as hook_exc:  # noqa: BLE001 — observability-only
            # The hook is observability-only; it must never swallow or
            # mutate the original exception. Suppress any hook-side error
            # so the caller still sees the real LLM failure — but emit a
            # warning so a buggy hook does not silently lose telemetry.
            logger.warning(
                "[_StructuredOutputRouter] failure_hook for arm {!r} raised "
                "{!r}; original LLM exception preserved.",
                name,
                hook_exc,
            )
