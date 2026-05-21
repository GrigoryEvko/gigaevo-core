"""Lifecycle and leak-prevention tests for ``run_with_config``.

Covers: (1) ``serve_until_signal`` raising after ``start()`` still
fires both ``stop()`` calls plus dataplane shutdown and writer close;
(2) ``build_object_graph`` failing mid-tree still runs the outer
``finally`` (ambient pool unbound, no thread leaks); (3) ``stop()`` is
idempotent on both EvolutionEngine and DagRunner.
"""

from __future__ import annotations

import asyncio
import threading
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _fake_cfg() -> types.SimpleNamespace:
    """Minimal stand-in for an ``ExperimentConfig`` that exposes every
    attribute :func:`run_with_config` touches."""
    cfg = types.SimpleNamespace(
        name="lifecycle-test",
        problem=types.SimpleNamespace(
            name="lifecycle-problem",
            problem_dir="/nonexistent",
        ),
        pipeline=types.SimpleNamespace(prompts_dir=None),
        redis=types.SimpleNamespace(
            url="redis://localhost:6379/0",
            host="localhost",
            port=6379,
            db=0,
            resume=False,
        ),
        dataplane=types.SimpleNamespace(
            key_prefix="lifecycle-test",
            max_connections=4,
        ),
    )
    return cfg


@pytest.mark.asyncio
async def test_start_then_exception_still_stops_runner_and_engine():
    """If ``serve_until_signal`` raises after the two tasks started,
    both ``stop()`` calls plus dataplane shutdown and writer close
    must still fire before the exception escapes."""
    import gigaevo.config.object_graph as og

    fake_dag_runner = MagicMock()
    fake_dag_runner.start = MagicMock()
    fake_dag_runner.stop = AsyncMock()
    fake_dag_runner.task = None
    fake_dag_runner._dataplane = None
    fake_dag_runner._engine_root = None

    fake_engine = MagicMock()
    fake_engine.start = MagicMock()
    fake_engine.stop = AsyncMock()
    fake_engine.task = None
    fake_engine._dataplane = None
    fake_engine._engine_root = None
    fake_engine.restore_state = AsyncMock()
    fake_engine.strategy = MagicMock(restore_state=AsyncMock())

    fake_storage = MagicMock()
    fake_storage.config = MagicMock(
        redis_url="redis://localhost:6379/0", key_prefix="lifecycle-test"
    )
    fake_storage.acquire_instance_lock = AsyncMock()
    fake_storage.has_data = AsyncMock(return_value=False)
    fake_storage.close = AsyncMock()
    fake_storage.size = AsyncMock(return_value=0)
    fake_storage.recover_stranded_programs = AsyncMock(return_value=0)

    fake_loader = MagicMock()
    fake_loader.load = AsyncMock(return_value=[])

    fake_writer = MagicMock()
    fake_writer.close = MagicMock()

    fake_dataplane = MagicMock()
    fake_dataplane.shutdown = AsyncMock()

    fake_problem_ctx = MagicMock()
    fake_problem_ctx.metrics_context = MagicMock()

    fake_graph = {
        "redis_storage": fake_storage,
        "problem_context": fake_problem_ctx,
        "llm": MagicMock(),
        "strategy": MagicMock(islands=None),
        "runtime_engine_config": MagicMock(
            metrics_collection_interval=1.0, max_generations=None
        ),
        "evolution_context": MagicMock(prompt_fetcher=None),
        "pipeline_builder": MagicMock(),
        "dag_blueprint": MagicMock(),
        "runtime_runner_config": MagicMock(),
        "primary_metric": "fitness",
        "higher_is_better": True,
        "required_behavior_keys": ["fitness"],
    }

    boom = RuntimeError("serve_until_signal failed")

    async def fake_serve_until_signal(*, stop_coros=(), on_stop=()):
        for coro in stop_coros:
            coro.close()
        raise boom

    cfg = _fake_cfg()

    with (
        patch.object(og, "build_object_graph", return_value=fake_graph),
        patch.object(og, "_build_default_writer", return_value=fake_writer),
        patch(
            "gigaevo.evolution.mutation.mutation_operator.LLMMutationOperator",
            return_value=MagicMock(),
        ),
        patch("gigaevo.utils.metrics_tracker.MetricsTracker", return_value=MagicMock()),
        patch(
            "gigaevo.runner.dag_runner.DagRunner",
            return_value=fake_dag_runner,
        ),
        patch.object(
            og, "_build_evolution_engine", return_value=fake_engine
        ),
        patch(
            "gigaevo.problems.initial_loaders.DirectoryProgramLoader",
            return_value=fake_loader,
        ),
        patch(
            "gigaevo.dataplane.build_dataplane",
            AsyncMock(return_value=fake_dataplane),
        ),
        patch("gigaevo.dataplane.build_engine_root", MagicMock(return_value=object())),
        patch(
            "gigaevo.dataplane.build_actor_identity",
            MagicMock(return_value=object()),
        ),
        patch("gigaevo.dataplane.wire_storage", MagicMock()),
        patch("gigaevo.dataplane.wire_dag_runner", MagicMock()),
        patch("gigaevo.dataplane.wire_evolution_engine", MagicMock()),
        patch("gigaevo.dataplane.wire_bandit_router", MagicMock(return_value=False)),
        patch("gigaevo.utils.serve.serve_until_signal", fake_serve_until_signal),
    ):
        with pytest.raises(RuntimeError, match="serve_until_signal failed"):
            await og.run_with_config(cfg)

    # Both stops must have fired before the exception escaped.
    fake_engine.stop.assert_awaited()
    fake_dag_runner.stop.assert_awaited()
    fake_storage.close.assert_awaited()
    fake_dataplane.shutdown.assert_awaited()
    fake_writer.close.assert_called()


@pytest.mark.asyncio
async def test_build_object_graph_failure_does_not_leak_pool():
    """``build_object_graph`` raising leaves no component reachable; the
    outer ``finally`` must tolerate every ``X is None`` check, shut the
    pool down, and reset the ambient pool token."""
    from gigaevo.programs.stages.python_executors.wrapper import (
        get_ambient_exec_runner_pool,
    )
    import gigaevo.config.object_graph as og

    boom = RuntimeError("build_object_graph failed mid-tree")
    cfg = _fake_cfg()

    before_ambient = get_ambient_exec_runner_pool()
    threads_before = {t.ident for t in threading.enumerate()}

    with patch.object(og, "build_object_graph", side_effect=boom):
        with pytest.raises(RuntimeError, match="build_object_graph failed mid-tree"):
            await og.run_with_config(cfg)

    # Ambient pool was bound for the call and reset in finally.
    assert get_ambient_exec_runner_pool() is before_ambient

    # No extra threads leaked. Allow a small window for daemon threads
    # spawned by unrelated test machinery to settle.
    await asyncio.sleep(0)
    threads_after = {t.ident for t in threading.enumerate()}
    leaked = threads_after - threads_before
    assert not leaked, f"Leaked {len(leaked)} thread(s) past run_with_config"


@pytest.mark.asyncio
async def test_engine_stop_is_idempotent_after_serve_already_stopped_it():
    """``stop()`` called twice on EvolutionEngine and DagRunner is a no-op."""
    engine = object.__new__(
        __import__(
            "gigaevo.evolution.engine.core", fromlist=["EvolutionEngine"]
        ).EvolutionEngine
    )
    engine._running = False
    engine._task = None
    engine._metrics_collector_task = None
    engine._metrics_tracker = None
    engine.storage = MagicMock(close=AsyncMock())

    await engine.stop()
    await engine.stop()
    assert engine._task is None

    runner = object.__new__(
        __import__("gigaevo.runner.dag_runner", fromlist=["DagRunner"]).DagRunner
    )
    runner._stopping = False
    runner._task = None
    runner._active = {}
    runner._done_queue = []
    runner._metrics_collector_task = None
    runner._storage = MagicMock(close=AsyncMock())
    runner._state_manager = MagicMock()

    async def _flush() -> None:
        return None

    runner._flush_done_queue = _flush

    await runner.stop()
    await runner.stop()
    assert runner._task is None
