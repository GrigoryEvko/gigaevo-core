"""Unit tests for the typed configuration schemas."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from gigaevo.config.schemas import (
    BanditRouterConfig,
    ChatOpenAIConfig,
    DataPlaneSettings,
    EnsembleRouterConfig,
    LLMConfig,
    RedisConfig,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class TestRedisConfig:
    def test_default_url(self) -> None:
        cfg = RedisConfig()
        assert cfg.url == "redis://localhost:6379/0"

    def test_custom_url(self) -> None:
        cfg = RedisConfig(host="redis.internal", port=6380, db=2)
        assert cfg.url == "redis://redis.internal:6380/2"

    def test_port_range(self) -> None:
        with pytest.raises(ValidationError):
            RedisConfig(port=0)
        with pytest.raises(ValidationError):
            RedisConfig(port=70_000)

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            RedisConfig(host="x", typo="oops")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        cfg = RedisConfig()
        with pytest.raises(ValidationError):
            cfg.host = "x"  # type: ignore[misc]

    def test_empty_host_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RedisConfig(host="")

    def test_to_storage_config_round_trip(self) -> None:
        redis = RedisConfig(host="r", port=6379, db=5)
        storage = redis.to_storage_config(key_prefix="my_problem")
        assert storage.redis_url == "redis://r:6379/5"
        assert storage.key_prefix == "my_problem"

    def test_json_round_trip_preserves_computed_url(self) -> None:
        original = RedisConfig(host="r", port=6380, db=2)
        as_json = original.model_dump_json()
        parsed = RedisConfig.model_validate_json(as_json)
        assert parsed == original
        assert parsed.url == "redis://r:6380/2"


class TestDataPlaneSettings:
    def test_compose_with_redis(self) -> None:
        cfg = DataPlaneSettings(
            redis=RedisConfig(),
            key_prefix="gigaevo:test",
        )
        assert cfg.redis.url == "redis://localhost:6379/0"
        assert cfg.startup_timeout_s > 0

    def test_startup_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DataPlaneSettings(
                redis=RedisConfig(),
                key_prefix="p",
                startup_timeout_s=0.0,
            )

    def test_empty_key_prefix_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataPlaneSettings(redis=RedisConfig(), key_prefix="")

    def test_nested_frozen_propagates(self) -> None:
        cfg = DataPlaneSettings(redis=RedisConfig(), key_prefix="gigaevo:test")
        with pytest.raises(ValidationError):
            cfg.redis.host = "x"  # type: ignore[misc]


class TestChatOpenAIConfig:
    def test_typo_rejected_at_construction(self) -> None:
        with pytest.raises(ValidationError, match="temprature"):
            ChatOpenAIConfig(model="x", temprature=0.5)  # type: ignore[call-arg]

    def test_temperature_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ChatOpenAIConfig(model="x", temperature=-0.1)
        with pytest.raises(ValidationError):
            ChatOpenAIConfig(model="x", temperature=2.5)

    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatOpenAIConfig(model="")

    def test_api_key_not_a_schema_field(self) -> None:
        """The API token must not be a schema field at all: it cannot
        leak through tyro's ``--help`` rendering, the dumped
        ``config.json``, ``__repr__``, or the experiment-id hash, and
        the dumped config remains self-contained across env changes."""
        assert "api_key" not in ChatOpenAIConfig.model_fields
        with pytest.raises(ValidationError, match="extra_forbidden"):
            ChatOpenAIConfig(model="x", api_key="explicit")  # type: ignore[call-arg]

    def test_kind_field_dropped(self) -> None:
        """``kind`` was dead weight on this leaf schema: the
        ``LLMConfig`` discriminated union dispatches on the
        ``BanditRouterConfig`` / ``EnsembleRouterConfig`` ``kind``,
        never on a nested ``ChatOpenAIConfig``."""
        assert "kind" not in ChatOpenAIConfig.model_fields

    def test_dump_carries_no_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dumped JSON must not contain any credential material —
        the value of OPENAI_API_KEY at construction time must not be
        observable from the serialised form."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-DO-NOT-LEAK")
        cfg = ChatOpenAIConfig(model="x")
        for surface in (cfg.model_dump(), cfg.model_dump_json(), repr(cfg)):
            text = surface if isinstance(surface, str) else str(surface)
            assert "sk-secret" not in text
            assert "api_key" not in text

    def test_dump_self_contained_across_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``dump -> unset env -> validate -> dump`` must round-trip
        byte-identically. The dumped ``config.json`` is the
        reproducibility record; making it depend on the runtime
        environment defeats the reproducibility contract."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-original")
        cfg = ChatOpenAIConfig(model="x")
        dumped = cfg.model_dump_json()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        parsed = ChatOpenAIConfig.model_validate_json(dumped)
        assert parsed.model_dump_json() == dumped


class TestChatOpenAIBuildSecretHandling:
    """Secret resolution moved into ``ChatOpenAIConfig.build``; the
    sanitiser there is the single line of defence between the env and
    the HTTP layer."""

    def _build_or_raise(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_value: str | None,
        explicit: str | None = None,
    ) -> str:
        """Capture the ``api_key`` argument passed to the strict
        ``ChatOpenAI`` constructor without actually instantiating it."""
        import gigaevo.llm.strict_chat_openai as scoa

        captured: dict[str, str] = {}

        def _fake(*_args: object, **kwargs: object) -> object:
            captured["api_key"] = kwargs["api_key"]  # type: ignore[assignment]
            return object()

        monkeypatch.setattr(scoa, "strict_chat_openai", _fake)
        if env_value is None:
            monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        else:
            monkeypatch.setenv("OPENAI_API_KEY", env_value)
        cfg = ChatOpenAIConfig(model="x")
        cfg.build(api_key=explicit)
        return captured["api_key"]

    def test_env_key_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._build_or_raise(monkeypatch, "sk-from-env") == "sk-from-env"

    def test_explicit_argument_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resolved = self._build_or_raise(
            monkeypatch, "sk-from-env", explicit="sk-explicit"
        )
        assert resolved == "sk-explicit"

    def test_missing_env_rejected_at_build(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = ChatOpenAIConfig(model="x")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            cfg.build()

    def test_whitespace_only_env_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        cfg = ChatOpenAIConfig(model="x")
        with pytest.raises(
            ValueError, match="at least one non-whitespace character"
        ):
            cfg.build()

    def test_ansi_escape_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-\x1b[31mred\x1b[0m")
        cfg = ChatOpenAIConfig(model="x")
        with pytest.raises(ValueError, match="control character"):
            cfg.build()

    def test_nul_byte_rejected_via_explicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = ChatOpenAIConfig(model="x")
        with pytest.raises(ValueError, match="NUL"):
            cfg.build(api_key="sk-with-\x00-nul")

    def test_surrounding_whitespace_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resolved = self._build_or_raise(monkeypatch, "  sk-padded  ")
        assert resolved == "sk-padded"


class TestBanditRouterConfig:
    def test_minimal(self) -> None:
        cfg = BanditRouterConfig(models=[ChatOpenAIConfig(model="gpt-4o-mini")])
        assert cfg.kind == "bandit"
        assert cfg.exploration_constant > 0

    def test_requires_at_least_one_model(self) -> None:
        with pytest.raises(ValidationError):
            BanditRouterConfig(models=[])


class TestEnsembleRouterConfig:
    def test_probabilities_length_must_match(self) -> None:
        m = ChatOpenAIConfig(model="x")
        with pytest.raises(ValidationError, match="length"):
            EnsembleRouterConfig(models=[m, m], probabilities=[1.0])

    def test_negative_probability_rejected(self) -> None:
        m = ChatOpenAIConfig(model="x")
        with pytest.raises(ValidationError, match="positive"):
            EnsembleRouterConfig(models=[m, m], probabilities=[1.0, -0.1])

    def test_nan_probability_rejected(self) -> None:
        m = ChatOpenAIConfig(model="x")
        with pytest.raises(ValidationError, match="finite"):
            EnsembleRouterConfig(
                models=[m, m], probabilities=[float("nan"), 0.5]
            )

    def test_inf_probability_rejected(self) -> None:
        m = ChatOpenAIConfig(model="x")
        with pytest.raises(ValidationError, match="finite"):
            EnsembleRouterConfig(
                models=[m, m], probabilities=[float("inf"), 0.5]
            )

    def test_no_probabilities_means_uniform_at_runtime(self) -> None:
        m = ChatOpenAIConfig(model="x")
        cfg = EnsembleRouterConfig(models=[m, m])
        assert cfg.probabilities is None


class TestLLMDiscriminatedUnion:
    def test_round_trip_bandit(self) -> None:
        m = ChatOpenAIConfig(model="x")
        cfg = BanditRouterConfig(models=[m])
        ta: TypeAdapter[LLMConfig] = TypeAdapter(LLMConfig)
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, BanditRouterConfig)

    def test_round_trip_ensemble(self) -> None:
        m = ChatOpenAIConfig(model="x")
        cfg = EnsembleRouterConfig(models=[m])
        ta: TypeAdapter[LLMConfig] = TypeAdapter(LLMConfig)
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, EnsembleRouterConfig)

    def test_unknown_kind_rejected(self) -> None:
        ta: TypeAdapter[LLMConfig] = TypeAdapter(LLMConfig)
        with pytest.raises(ValidationError):
            ta.validate_python({"kind": "unknown", "models": []})
