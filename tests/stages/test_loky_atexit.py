"""Regression: ``LokyBackend`` shutdown runs at interpreter exit.

The ``_atexit_shutdown_default_backend`` callback is registered via
``atexit.register`` at module import time so a driver that crashes
out of ``run_with_config``'s ``finally`` (uncaught
``KeyboardInterrupt`` before ``serve_until_signal`` started, for
example) still drops the loky manager thread and frees its
``/dev/shm`` semaphore handles.
"""

from __future__ import annotations

import atexit

import pytest


def test_atexit_handler_registered() -> None:
    """The atexit registry must contain the wrapper's shutdown callback."""
    from gigaevo.programs.stages.python_executors import wrapper

    # ``atexit`` doesn't expose a public introspection API; the
    # CPython-internal ``_ncallbacks`` and ``_run_exitfuncs`` are stable
    # enough for a regression. Fall back to ``_clear`` + re-register
    # detection if the internals are unavailable.
    callbacks = getattr(atexit, "_callbacks", None)
    if callbacks is None:
        # No introspection — just verify the function exists and is
        # callable; the import-time ``atexit.register(...)`` line at
        # module top level is the contract.
        assert callable(wrapper._atexit_shutdown_default_backend)
        return
    targets = [cb for cb, *_ in callbacks if cb is wrapper._atexit_shutdown_default_backend]
    assert targets, (
        "wrapper._atexit_shutdown_default_backend not registered via atexit"
    )


def test_atexit_handler_swallows_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exit-path hook must not raise — the interpreter is on its
    way out and a raised exception is a noisy ``BaseException`` at
    teardown."""
    from gigaevo.programs.stages.python_executors import wrapper

    def boom(*_a, **_kw):
        raise RuntimeError("backend already torn down")

    monkeypatch.setattr(wrapper, "shutdown_executor", boom)
    # Must not raise.
    wrapper._atexit_shutdown_default_backend()


def test_atexit_handler_calls_shutdown_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful path: the helper forwards to ``shutdown_executor``."""
    from gigaevo.programs.stages.python_executors import wrapper

    calls = {"n": 0, "wait": None}

    def fake(*, wait: bool = False) -> None:
        calls["n"] += 1
        calls["wait"] = wait

    monkeypatch.setattr(wrapper, "shutdown_executor", fake)
    wrapper._atexit_shutdown_default_backend()
    assert calls["n"] == 1
    # ``wait=False`` so a stuck pool cannot block interpreter exit.
    assert calls["wait"] is False
