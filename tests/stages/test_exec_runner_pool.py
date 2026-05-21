"""Regression tests for ``WorkerPool`` and the ambient-pool ContextVar.

The factory builds a fresh ``WorkerPool`` per call. Lifecycle owners that
want to amortize subprocess startup across many calls bind a single pool
via ``set_ambient_exec_runner_pool`` for the duration of an
``asyncio.run`` and tear it down before the loop closes.

Each ``WorkerPool`` binds its ``asyncio.Queue`` and ``asyncio.Lock`` to
the event loop it is first used on, so an instance must not survive
across distinct ``asyncio.run`` invocations.
"""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.programs.stages.python_executors.wrapper import (
    WorkerPool,
    default_exec_runner_pool,
    get_ambient_exec_runner_pool,
    reset_ambient_exec_runner_pool,
    set_ambient_exec_runner_pool,
)


def test_factory_returns_fresh_instance_per_call():
    """Two consecutive calls produce distinct ``WorkerPool`` objects."""
    a = default_exec_runner_pool()
    b = default_exec_runner_pool()
    assert isinstance(a, WorkerPool)
    assert isinstance(b, WorkerPool)
    assert a is not b


def test_factory_has_no_lru_cache_attribute():
    """The factory must not be wrapped by ``functools.lru_cache``.

    A wrapped function would expose ``cache_clear`` / ``cache_info``; the
    plain function does not. The check fails fast if the cache decorator
    re-appears.
    """
    assert not hasattr(default_exec_runner_pool, "cache_clear")
    assert not hasattr(default_exec_runner_pool, "cache_info")


def test_each_pool_owns_its_asyncio_primitives():
    """Each pool gets its own queue/lock; no shared state survives across pools."""
    a = default_exec_runner_pool()
    b = default_exec_runner_pool()
    assert a._queue is not b._queue
    assert a._lock is not b._lock
    assert a._count == 0
    assert b._count == 0


def test_fresh_pool_is_safe_across_sequential_event_loops():
    """A pool built in one event loop is not reused in a second.

    Simulates the multirun pattern: a hypothetical caller that requests the
    default factory in one ``asyncio.run`` and again in another receives two
    pools with independent ``asyncio`` primitives bound to the respective
    loops.
    """
    loop_a_pool: WorkerPool | None = None

    async def in_loop_a() -> WorkerPool:
        pool = default_exec_runner_pool()
        # Touch the lock so it binds to the current loop.
        async with pool._lock:
            pass
        return pool

    loop_a_pool = asyncio.run(in_loop_a())

    async def in_loop_b() -> WorkerPool:
        pool = default_exec_runner_pool()
        async with pool._lock:
            pass
        return pool

    loop_b_pool = asyncio.run(in_loop_b())
    assert loop_b_pool is not loop_a_pool
    assert loop_b_pool._lock is not loop_a_pool._lock
    assert loop_b_pool._queue is not loop_a_pool._queue


def test_ambient_pool_defaults_to_none():
    """No ambient pool is bound at import time."""
    assert get_ambient_exec_runner_pool() is None


def test_ambient_pool_round_trip():
    """``set`` then ``reset`` restores the previous binding."""
    pool = WorkerPool()
    token = set_ambient_exec_runner_pool(pool)
    try:
        assert get_ambient_exec_runner_pool() is pool
    finally:
        reset_ambient_exec_runner_pool(token)
    assert get_ambient_exec_runner_pool() is None


def test_ambient_pool_resolves_to_same_object_within_scope():
    """A bound ambient pool is returned identically by every getter call.

    The lifecycle owner binds one pool for the scope of an experiment;
    every consumer that resolves the ambient binding sees that exact
    object, never a copy or a fresh instance.
    """
    seen: list[WorkerPool] = []

    async def scenario() -> WorkerPool:
        ambient = WorkerPool()
        token = set_ambient_exec_runner_pool(ambient)
        try:
            seen.append(get_ambient_exec_runner_pool())  # type: ignore[arg-type]
            seen.append(get_ambient_exec_runner_pool())  # type: ignore[arg-type]
            return ambient
        finally:
            reset_ambient_exec_runner_pool(token)

    ambient = asyncio.run(scenario())
    assert len(seen) == 2
    assert seen[0] is ambient
    assert seen[1] is ambient


def test_ambient_pool_rebinding_returns_new_pool_after_reset():
    """A second ``set`` after ``reset`` exposes a different pool object."""
    a = WorkerPool()
    b = WorkerPool()

    token_a = set_ambient_exec_runner_pool(a)
    assert get_ambient_exec_runner_pool() is a
    reset_ambient_exec_runner_pool(token_a)
    assert get_ambient_exec_runner_pool() is None

    token_b = set_ambient_exec_runner_pool(b)
    try:
        assert get_ambient_exec_runner_pool() is b
        assert get_ambient_exec_runner_pool() is not a
    finally:
        reset_ambient_exec_runner_pool(token_b)


def test_worker_pool_shutdown_is_idempotent():
    """Calling ``shutdown`` on an empty pool is a safe no-op.

    The pool may be torn down by the lifecycle owner even when no worker
    was ever checked out; ``shutdown`` must not blow up on an empty
    queue, and it must remain callable a second time.
    """

    async def scenario() -> None:
        pool = WorkerPool()
        await pool.shutdown()
        await pool.shutdown()
        assert pool._count == 0
        assert pool._queue.empty()

    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_no_pool_none_call_sites_remain_unguarded():
    """Audit: production callers in stages must not rebuild a pool per call.

    The audited stages must resolve any ambient pool through the
    lifecycle owner; an explicit ``default_exec_runner_pool()`` call in
    those modules would bypass the experiment-scoped binding and
    fragment subprocess startup costs.
    """
    import inspect

    from gigaevo.programs.stages import runtime_metrics
    from gigaevo.programs.stages.optimization import utils as opt_utils
    from gigaevo.programs.stages.optimization.optuna import stage as optuna_stage
    from gigaevo.programs.stages.python_executors import execution

    for module in (runtime_metrics, opt_utils, optuna_stage, execution):
        source = inspect.getsource(module)
        assert "default_exec_runner_pool(" not in source, (
            f"{module.__name__} constructs an ad-hoc pool; resolve the "
            f"ambient pool bound by the experiment driver instead."
        )
