"""Tests for :class:`LLMCircuitBreaker`.

Validates the closed/open/half-open FSM, the consecutive-failure
threshold, the cooldown timer, and the half-open probe budget.
"""

from __future__ import annotations

import pytest

from gigaevo.llm.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    LLMCircuitBreaker,
)


class TestCircuitBreakerConfig:
    def test_defaults(self) -> None:
        cfg = CircuitBreakerConfig()
        assert cfg.failure_threshold == 5
        assert cfg.cooldown_s == 30.0
        assert cfg.half_open_max_probes == 1

    def test_invalid_threshold_rejected(self) -> None:
        with pytest.raises(ValueError):
            CircuitBreakerConfig(failure_threshold=0)

    def test_invalid_cooldown_rejected(self) -> None:
        with pytest.raises(ValueError):
            CircuitBreakerConfig(cooldown_s=0)

    def test_invalid_probe_budget_rejected(self) -> None:
        with pytest.raises(ValueError):
            CircuitBreakerConfig(half_open_max_probes=0)


class TestCircuitBreakerFSM:
    def _breaker(self, **cfg_kwargs) -> LLMCircuitBreaker:
        return LLMCircuitBreaker(
            name="test", config=CircuitBreakerConfig(**cfg_kwargs)
        )

    def test_starts_closed(self) -> None:
        b = self._breaker()
        assert b.state is CircuitState.CLOSED
        assert b.allow_request() is True

    def test_opens_after_threshold_consecutive_failures(self) -> None:
        b = self._breaker(failure_threshold=3)
        b.record_failure()
        b.record_failure()
        assert b.state is CircuitState.CLOSED
        b.record_failure()
        assert b.state is CircuitState.OPEN

    def test_success_resets_failure_counter(self) -> None:
        b = self._breaker(failure_threshold=3)
        b.record_failure()
        b.record_failure()
        b.record_success()
        # Counter reset — another two failures must not trip the breaker.
        b.record_failure()
        b.record_failure()
        assert b.state is CircuitState.CLOSED

    def test_open_short_circuits_until_cooldown(self) -> None:
        b = self._breaker(failure_threshold=1, cooldown_s=10.0)
        now = 1000.0
        b.record_failure(now=now)
        assert b.state is CircuitState.OPEN
        assert b.allow_request(now=now + 5.0) is False
        # Cooldown expired — first allow_request transitions to HALF_OPEN
        # and admits the probe.
        assert b.allow_request(now=now + 11.0) is True
        assert b.state is CircuitState.HALF_OPEN

    def test_half_open_success_closes_breaker(self) -> None:
        b = self._breaker(failure_threshold=1, cooldown_s=1.0)
        b.record_failure(now=0.0)
        assert b.allow_request(now=2.0) is True  # → HALF_OPEN
        b.record_success()
        assert b.state is CircuitState.CLOSED

    def test_half_open_failure_reopens_with_fresh_cooldown(self) -> None:
        b = self._breaker(failure_threshold=1, cooldown_s=10.0)
        b.record_failure(now=0.0)
        assert b.allow_request(now=15.0) is True  # → HALF_OPEN
        b.record_failure(now=15.0)
        assert b.state is CircuitState.OPEN
        # Cooldown clock restarts from the half-open failure.
        assert b.allow_request(now=20.0) is False
        assert b.allow_request(now=26.0) is True

    def test_half_open_probe_budget(self) -> None:
        b = self._breaker(
            failure_threshold=1, cooldown_s=1.0, half_open_max_probes=1
        )
        b.record_failure(now=0.0)
        # First request after cooldown is the single allowed probe.
        assert b.allow_request(now=2.0) is True
        # Second concurrent request must be rejected.
        assert b.allow_request(now=2.0) is False

    def test_guard_raises_when_open(self) -> None:
        b = self._breaker(failure_threshold=1, cooldown_s=10.0)
        b.record_failure(now=0.0)
        with pytest.raises(CircuitOpenError) as exc:
            b.guard(now=2.0)
        assert exc.value.name == "test"
        assert exc.value.remaining_cooldown_s == pytest.approx(8.0)

    def test_guard_admits_when_closed(self) -> None:
        b = self._breaker()
        b.guard()  # must not raise

    def test_snapshot_includes_state(self) -> None:
        b = self._breaker(failure_threshold=2, cooldown_s=300.0)
        b.record_failure()
        snap = b.snapshot()
        assert snap["state"] == "closed"
        assert snap["consecutive_failures"] == 1
        b.record_failure()
        snap = b.snapshot()
        assert snap["state"] == "open"
        # Real-time cooldown — value is roughly the full 300s.
        assert snap["remaining_cooldown_s"] > 0.0

    def test_remaining_cooldown_zero_when_closed(self) -> None:
        b = self._breaker()
        assert b.remaining_cooldown_s() == 0.0


class TestMultiModelRouterIntegration:
    """``MultiModelRouter`` short-circuits invokes when the breaker opens."""

    def _router(self, *, threshold: int = 1):
        from unittest.mock import MagicMock

        from gigaevo.llm.circuit_breaker import CircuitBreakerConfig
        from gigaevo.llm.models import MultiModelRouter
        from tests.conftest import NullWriter

        model = MagicMock()
        model.model_name = "m"
        model.with_structured_output = MagicMock(return_value=MagicMock())
        return MultiModelRouter(
            [model],
            [1.0],
            writer=NullWriter(),
            name="t",
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=threshold, cooldown_s=300.0
            ),
        )

    def test_failure_eventually_opens_breaker(self) -> None:
        router = self._router(threshold=2)
        # Two failures cross the threshold.
        router.models[0].invoke.side_effect = RuntimeError("503")
        with pytest.raises(RuntimeError):
            router.invoke("x")
        with pytest.raises(RuntimeError):
            router.invoke("x")
        # Third call short-circuits at the breaker, not the model.
        from gigaevo.llm.circuit_breaker import CircuitOpenError

        with pytest.raises(CircuitOpenError):
            router.invoke("x")
        # Underlying model was only called twice.
        assert router.models[0].invoke.call_count == 2

    def test_success_keeps_breaker_closed(self) -> None:
        router = self._router(threshold=1)
        router.models[0].invoke.return_value = "ok"
        for _ in range(5):
            assert router.invoke("x") == "ok"
        # No failures recorded; counter stays at zero.
        assert router.circuit_breaker.snapshot()["consecutive_failures"] == 0
