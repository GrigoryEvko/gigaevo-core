"""Tests for the startup model-verification helpers in
:mod:`gigaevo.llm.models`.

Covers the thundering-herd / auth-header improvements added on top of
the original ``_verify_models``:

- Process-wide TTL cache so multiple routers in the same process don't
  re-probe the same base URL.
- Failures cached briefly (30 s) so they don't burn a fresh jitter
  sleep on every router instantiation.
- ``Authorization: Bearer ...`` sent when the model has an API key.
- Mock-shaped (non-string) base URLs short-circuit without touching the
  network or the cache.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _no_jitter():
    """Tests must not actually sleep 0–2 s per call."""
    from gigaevo.llm import models as models_mod

    with patch.object(models_mod, "_VERIFY_JITTER_MAX_S", 0):
        yield


@pytest.fixture(autouse=True)
def _clean_cache():
    """Reset the module-level cache so tests start from a known state."""
    from gigaevo.llm import models as models_mod

    with models_mod._verify_cache_lock:
        models_mod._verify_cache.clear()
        models_mod._verify_failure_counts.clear()
    yield
    with models_mod._verify_cache_lock:
        models_mod._verify_cache.clear()
        models_mod._verify_failure_counts.clear()


class TestFetchAvailableModelsAt:
    def _ok_session(self, models: list[str]) -> MagicMock:
        session = MagicMock()
        response = MagicMock()
        response.json.return_value = {"data": [{"id": m} for m in models]}
        response.raise_for_status = MagicMock()
        session.get.return_value = response
        session.close = MagicMock()
        return session

    def test_returns_frozen_set_of_model_ids(self, monkeypatch) -> None:
        from gigaevo.llm import models as models_mod

        session = self._ok_session(["gpt-4o-mini", "gpt-4o"])
        monkeypatch.setattr(
            models_mod,
            "make_requests_session",
            lambda *a, **kw: session,
            raising=False,
        )
        # Stub the import path inside the function — it imports inside the body.
        import gigaevo.infra.requests_factory as rf

        monkeypatch.setattr(
            rf, "make_requests_session", lambda *a, **kw: session
        )
        result = models_mod._fetch_available_models_at(
            "http://x.invalid/v1", api_key=None
        )
        assert result == frozenset({"gpt-4o-mini", "gpt-4o"})
        # Cached for the next call.
        result2 = models_mod._fetch_available_models_at(
            "http://x.invalid/v1", api_key=None
        )
        assert result2 is result
        # Network probe only fired once.
        assert session.get.call_count == 1

    def test_sends_authorization_header_when_api_key_present(
        self, monkeypatch
    ) -> None:
        import gigaevo.infra.requests_factory as rf
        from gigaevo.llm import models as models_mod

        session = self._ok_session(["m"])
        monkeypatch.setattr(rf, "make_requests_session", lambda *a, **kw: session)

        models_mod._fetch_available_models_at(
            "http://x.invalid/v2", api_key="sk-secret"
        )
        _, kwargs = session.get.call_args
        assert kwargs.get("headers", {}).get("Authorization") == "Bearer sk-secret"

    def test_no_auth_header_when_api_key_absent(self, monkeypatch) -> None:
        import gigaevo.infra.requests_factory as rf
        from gigaevo.llm import models as models_mod

        session = self._ok_session(["m"])
        monkeypatch.setattr(rf, "make_requests_session", lambda *a, **kw: session)

        models_mod._fetch_available_models_at(
            "http://x.invalid/v3", api_key=None
        )
        _, kwargs = session.get.call_args
        assert kwargs.get("headers") in (None, {})

    def test_failure_returns_none_and_caches_briefly(self, monkeypatch) -> None:
        import gigaevo.infra.requests_factory as rf
        from gigaevo.llm import models as models_mod

        session = MagicMock()
        session.get.side_effect = RuntimeError("connection refused")
        session.close = MagicMock()
        monkeypatch.setattr(rf, "make_requests_session", lambda *a, **kw: session)

        result = models_mod._fetch_available_models_at(
            "http://broken.invalid", api_key=None
        )
        assert result is None
        # Cached at failure TTL — a second call within the failure window
        # must not re-issue the network probe.
        result2 = models_mod._fetch_available_models_at(
            "http://broken.invalid", api_key=None
        )
        assert result2 is None
        assert session.get.call_count == 1

    def test_filters_non_dict_data_entries(self, monkeypatch) -> None:
        """A misbehaving server might return mixed-type entries — the
        helper must not crash on them."""
        import gigaevo.infra.requests_factory as rf
        from gigaevo.llm import models as models_mod

        session = MagicMock()
        response = MagicMock()
        response.json.return_value = {
            "data": [
                {"id": "ok-1"},
                "not-a-dict",
                {"no_id_field": True},
                {"id": "ok-2"},
            ]
        }
        response.raise_for_status = MagicMock()
        session.get.return_value = response
        session.close = MagicMock()
        monkeypatch.setattr(rf, "make_requests_session", lambda *a, **kw: session)

        result = models_mod._fetch_available_models_at(
            "http://x.invalid/v4", api_key=None
        )
        assert result == frozenset({"ok-1", "ok-2"})


class TestFailureBackoff:
    """The failure TTL doubles per consecutive failure (capped at 1h);
    a success resets the counter."""

    def _failing_session(self) -> MagicMock:
        session = MagicMock()
        session.get.side_effect = RuntimeError("nope")
        session.close = MagicMock()
        return session

    def test_failure_ttl_doubles_per_strike(self) -> None:
        from gigaevo.llm import models as models_mod

        base = models_mod._VERIFY_CACHE_TTL_FAILURE_S
        cap = models_mod._VERIFY_CACHE_TTL_FAILURE_MAX_S
        assert models_mod._failure_ttl_for(0) == base
        assert models_mod._failure_ttl_for(1) == base
        assert models_mod._failure_ttl_for(2) == base * 2
        assert models_mod._failure_ttl_for(3) == base * 4
        # Cap kicks in once the doubling exceeds the ceiling.
        for n in range(50, 60):
            assert models_mod._failure_ttl_for(n) == cap

    def test_consecutive_failures_increment_counter(self, monkeypatch) -> None:
        import gigaevo.infra.requests_factory as rf
        from gigaevo.llm import models as models_mod

        session = self._failing_session()
        monkeypatch.setattr(rf, "make_requests_session", lambda *a, **kw: session)
        url = "http://flaky.invalid"

        # First failure.
        models_mod._fetch_available_models_at(url, api_key=None)
        assert models_mod._verify_failure_counts[url] == 1

        # Bypass the failure-cache TTL so the next call re-probes.
        with models_mod._verify_cache_lock:
            models_mod._verify_cache.pop(url, None)
        models_mod._fetch_available_models_at(url, api_key=None)
        assert models_mod._verify_failure_counts[url] == 2

    def test_success_resets_failure_counter(self, monkeypatch) -> None:
        import gigaevo.infra.requests_factory as rf
        from gigaevo.llm import models as models_mod

        url = "http://recover.invalid"
        failing = self._failing_session()
        monkeypatch.setattr(
            rf, "make_requests_session", lambda *a, **kw: failing
        )
        models_mod._fetch_available_models_at(url, api_key=None)
        models_mod._fetch_available_models_at(url, api_key=None)  # cached, no probe
        assert models_mod._verify_failure_counts[url] == 1

        # Swap in a recovering session and clear the cache so the next
        # call re-probes.
        ok_session = MagicMock()
        ok_response = MagicMock()
        ok_response.json.return_value = {"data": [{"id": "m1"}]}
        ok_response.raise_for_status = MagicMock()
        ok_session.get.return_value = ok_response
        ok_session.close = MagicMock()
        monkeypatch.setattr(
            rf, "make_requests_session", lambda *a, **kw: ok_session
        )
        with models_mod._verify_cache_lock:
            models_mod._verify_cache.pop(url, None)

        result = models_mod._fetch_available_models_at(url, api_key=None)
        assert result == frozenset({"m1"})
        assert url not in models_mod._verify_failure_counts


class TestVerifyModelsMockTolerance:
    """``MultiModelRouter.__init__`` is called with ``MagicMock`` models
    in many existing tests.  The verification path must short-circuit
    on non-string base URLs without touching the network."""

    def test_mock_base_url_short_circuits(self, monkeypatch) -> None:
        from gigaevo.llm import models as models_mod

        called = {"n": 0}

        def fail_if_called(*_a, **_kw):
            called["n"] += 1
            raise AssertionError("network probe should not run on mock URL")

        monkeypatch.setattr(
            models_mod, "_fetch_available_models_at", fail_if_called
        )

        mock_model = MagicMock()
        mock_model.model_name = "fake"
        mock_model.with_structured_output = MagicMock()
        # MagicMock attribute access returns a MagicMock — exactly the
        # state the original code was tripping over.
        assert not isinstance(
            getattr(mock_model, "base_url", None), str
        )
        models_mod.MultiModelRouter([mock_model], [1.0], name="t")
        assert called["n"] == 0
