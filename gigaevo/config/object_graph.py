"""Typed-config to runtime-object adapter and end-to-end runner.

:func:`build_object_graph` takes a validated :class:`ExperimentConfig`
and constructs the runtime object tree: the Redis program storage, the
problem context, the LLM router with bandit parameters resolved from
the metrics context, the evolution strategy with the storage threaded
through, the runtime engine config with required behavior keys
resolved, the evolution context composing those, the pipeline builder,
the DAG blueprint, and the runtime DAG-runner config. The function is
pure construction with no I/O — safe to call from unit tests.

:func:`run_with_config` is the lifecycle owner. It extends the object
graph with the I/O-bound components (writer, metrics tracker, mutation
operator, DAG-runner instance, program loader, evolution-engine
instance), constructs the :class:`DataPlane` coordinator, threads the
per-subspace permission roots through every coordinator-aware object
via the ``wire_*`` helpers, and drives the engine until it stops or a
shutdown signal arrives.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.config.schemas.algorithm import (
    MultiIslandConfig,
    SingleIslandConfig,
)
from gigaevo.config.schemas.engine import (
    BusedEngineConfig,
    SteadyStateEngineConfig as SteadyStateEngineConfigSchema,
)
from gigaevo.config.schemas.experiment import ExperimentConfig
from gigaevo.config.schemas.llm import BanditRouterConfig

if TYPE_CHECKING:
    from gigaevo.dataplane import DataPlane
    from gigaevo.evolution.engine.core import EvolutionEngine
    from gigaevo.problems.context import ProblemContext
    from gigaevo.runner.dag_runner import DagRunner
    from gigaevo.utils.trackers.composite import CompositeLogger


def _resolve_bandit_keys(
    cfg: ExperimentConfig,
    problem_ctx: "ProblemContext | None" = None,
) -> tuple[str, bool]:
    """The bandit router needs a fitness key + direction; we resolve
    them from the problem's MetricsContext, falling back to the
    problem-side overrides on :class:`ProblemConfig` when set.

    ``problem_ctx`` may be passed by callers that already constructed
    one so the underlying ``metrics.yaml`` is read at most once per
    graph build.
    """
    if problem_ctx is None:
        problem_ctx = cfg.problem.build()
    metrics_ctx = problem_ctx.metrics_context
    primary_key = cfg.problem.primary_metric or metrics_ctx.get_primary_key()
    if cfg.problem.higher_is_better is not None:
        higher = cfg.problem.higher_is_better
    else:
        higher = metrics_ctx.is_higher_better(primary_key)
    return primary_key, higher


def _required_behavior_keys(cfg: ExperimentConfig) -> list[str]:
    """Collect the union of behavior keys across every island in the
    algorithm subtree. ``StandardEvolutionAcceptor`` requires every
    program have all of these present in its metrics before accepting."""
    if isinstance(cfg.algorithm, SingleIslandConfig):
        return list(cfg.algorithm.island.behavior_space.behavior_keys)
    if isinstance(cfg.algorithm, MultiIslandConfig):
        keys: set[str] = set()
        for island in cfg.algorithm.islands:
            keys.update(island.behavior_space.behavior_keys)
        return sorted(keys)
    raise NotImplementedError(
        f"behavior key extraction not implemented for "
        f"{type(cfg.algorithm).__name__}"
    )


def build_object_graph(cfg: ExperimentConfig) -> dict[str, Any]:
    """Construct the runtime object tree from a validated config.

    Returns a dict with the constructed objects. The keys form the
    canonical surface that consumers (CLI, parity harness, sweep
    utility) read against.

    Top-level keys:
        - ``redis_storage``: RedisProgramStorage
        - ``problem_context``: ProblemContext (lazy MetricsContext)
        - ``llm``: MultiModelRouter or BanditModelRouter
        - ``strategy``: MapElitesMultiIsland
        - ``runtime_engine_config``: EngineConfig / SteadyStateEngineConfig
        - ``evolution_context``: EvolutionContext
        - ``pipeline_builder``: PipelineBuilder subclass
        - ``dag_blueprint``: DAGBlueprint
        - ``runtime_runner_config``: DagRunnerConfig
        - ``primary_metric``: str (resolved from problem)
        - ``higher_is_better``: bool (resolved from problem)
        - ``required_behavior_keys``: list[str]

    The writer, mutation operator, metrics tracker, DAG-runner
    instance, program loader and evolution-engine instance are
    deliberately deferred to :func:`run_with_config` — those construct
    Redis pools, threads, or background tasks and must not run in the
    pure-construction context unit tests use.
    """
    from gigaevo.database.redis_program_storage import RedisProgramStorage
    from gigaevo.entrypoint.evolution_context import EvolutionContext
    from gigaevo.memory.provider import NullMemoryProvider

    redis_storage_config = cfg.redis.to_storage_config(
        key_prefix=cfg.dataplane.key_prefix,
        max_connections=cfg.dataplane.max_connections,
    )
    redis_storage = RedisProgramStorage(redis_storage_config)

    problem_context = cfg.problem.build()
    primary_metric, higher_is_better = _resolve_bandit_keys(
        cfg, problem_ctx=problem_context
    )
    behavior_keys = _required_behavior_keys(cfg)

    if isinstance(cfg.llm, BanditRouterConfig):
        llm = cfg.llm.build(
            fitness_key=primary_metric,
            higher_is_better=higher_is_better,
        )
    else:
        llm = cfg.llm.build()

    strategy = cfg.algorithm.build(program_storage=redis_storage)
    runtime_engine_config = cfg.engine.build_runtime_config(
        required_behavior_keys=behavior_keys
    )

    prompt_fetcher_runtime = (
        cfg.prompt_fetcher.build() if cfg.prompt_fetcher is not None else None
    )
    evolution_context = EvolutionContext(
        problem_ctx=problem_context,
        llm_wrapper=llm,
        storage=redis_storage,
        prompts_dir=cfg.pipeline.prompts_dir,
        prompt_fetcher=prompt_fetcher_runtime,
        memory_provider=NullMemoryProvider(),
    )

    pipeline_builder = cfg.pipeline.builder.build(ctx=evolution_context)
    dag_blueprint = pipeline_builder.build_blueprint()

    runtime_runner_config = cfg.runner.build()

    return {
        "redis_storage": redis_storage,
        "problem_context": problem_context,
        "llm": llm,
        "strategy": strategy,
        "runtime_engine_config": runtime_engine_config,
        "evolution_context": evolution_context,
        "pipeline_builder": pipeline_builder,
        "dag_blueprint": dag_blueprint,
        "runtime_runner_config": runtime_runner_config,
        "primary_metric": primary_metric,
        "higher_is_better": higher_is_better,
        "required_behavior_keys": behavior_keys,
    }


def _build_default_writer(cfg: ExperimentConfig) -> "CompositeLogger":
    """Build a single-backend Redis composite writer.

    The current :class:`ExperimentConfig` schema does not carry a
    ``logging`` subtree, so the writer is derived from the experiment's
    Redis coordinates: same instance, dedicated key prefix
    ``{dataplane.key_prefix}:metrics``. This mirrors the production
    deployment pattern of co-locating metric history with program
    storage. When a typed ``LoggingConfig`` lands on the experiment
    schema this helper collapses to ``cfg.logging.build_writer()``.
    """
    from gigaevo.utils.trackers import init_composite, init_redis
    from gigaevo.utils.trackers.configs import RedisMetricsConfig

    redis_cfg = RedisMetricsConfig(
        redis_url=cfg.redis.url,
        key_prefix=f"{cfg.dataplane.key_prefix}:metrics",
    )
    return init_composite(init_redis(redis_cfg))


def _build_evolution_engine(
    cfg: ExperimentConfig,
    *,
    storage,
    strategy,
    mutation_operator,
    runtime_engine_config,
    writer,
    metrics_tracker,
) -> "EvolutionEngine":
    """Pick the engine variant matching ``cfg.engine.kind`` and return
    a constructed instance with the dataplane slots left unattached
    (``run_with_config`` wires them after engine construction)."""
    from gigaevo.evolution.engine.core import EvolutionEngine
    from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine

    if isinstance(cfg.engine, SteadyStateEngineConfigSchema):
        return SteadyStateEvolutionEngine(
            storage=storage,
            strategy=strategy,
            mutation_operator=mutation_operator,
            config=runtime_engine_config,
            writer=writer,
            metrics_tracker=metrics_tracker,
        )
    if isinstance(cfg.engine, BusedEngineConfig):
        from gigaevo.evolution.bus.engine import BusedEvolutionEngine

        migration_node = cfg.engine.migration_bus.build()
        return BusedEvolutionEngine(
            migration_node=migration_node,
            max_imports_per_generation=cfg.engine.max_imports_per_generation,
            storage=storage,
            strategy=strategy,
            mutation_operator=mutation_operator,
            config=runtime_engine_config,
            writer=writer,
            metrics_tracker=metrics_tracker,
        )
    return EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=mutation_operator,
        config=runtime_engine_config,
        writer=writer,
        metrics_tracker=metrics_tracker,
    )


def _wire_archive_storages(strategy, dataplane: "DataPlane", engine_root) -> None:
    """Walk the strategy's island archives and attach the coordinator
    handles. A strategy without ``.islands`` (e.g. a single-storage
    surrogate) skips the loop cleanly; an island without an
    ``archive_storage`` attribute also skips."""
    from gigaevo.dataplane import wire_archive_storage

    islands = getattr(strategy, "islands", None)
    if not islands:
        return
    for island in islands.values():
        archive = getattr(island, "archive_storage", None)
        if archive is not None:
            wire_archive_storage(archive, dataplane, engine_root)


async def _maybe_build_prompt_dataplane(
    mutation_operator, main_dataplane: "DataPlane", actor
) -> "DataPlane | None":
    """If the mutation operator carries a GigaEvoArchivePromptFetcher,
    build a second DataPlane dialled at the fetcher's URL and attach
    both DataPlanes via :func:`wire_prompt_fetcher`. Returns the
    prompt-side DataPlane so the caller can shut it down in its
    ``finally`` block.

    Returns ``None`` when no co-evolved prompt fetcher is present —
    static fetchers need no coordinator and own no Redis pool of
    their own.
    """
    from gigaevo.dataplane import build_dataplane, wire_prompt_fetcher
    from gigaevo.prompts.fetcher import GigaEvoArchivePromptFetcher

    fetcher = getattr(mutation_operator, "_prompt_fetcher", None)
    if not isinstance(fetcher, GigaEvoArchivePromptFetcher):
        return None
    prompt_url = (
        f"redis://{fetcher._host}:{fetcher._port}/{fetcher._prompt_redis_db}"
    )
    prompt_dp = await build_dataplane(prompt_url, key_prefix=fetcher._prompt_prefix)
    wire_prompt_fetcher(fetcher, main_dataplane, prompt_dp, actor)
    return prompt_dp


def _resolve_exit_code(
    dag_runner: "DagRunner | None",
    evolution_engine: "EvolutionEngine | None",
    archive_size_before: int,
    archive_size_after: int,
) -> int:
    """Return ``0`` for a clean run, ``2`` when the run finished with at
    least one silent failure path triggered *and* the archive failed to
    grow.

    The CLI's contract: an exit of zero means "the run accomplished
    forward progress (the archive grew) or terminated cleanly without
    swallowing batch-transition / program-not-found errors". Any time
    a batch-transition rejection or a stale-id event landed and the
    archive made no net progress, the wrapper script needs to see the
    failure — otherwise a Kubernetes scheduler treats a no-op crash as
    a healthy completion and never restarts the run.
    """
    runner_silent = (
        dag_runner is not None
        and getattr(dag_runner, "_metrics", None) is not None
        and dag_runner._metrics.has_silent_failures()
    )
    engine_silent = False
    if evolution_engine is not None and getattr(
        evolution_engine, "metrics", None
    ) is not None:
        engine_silent = (
            getattr(evolution_engine.metrics, "batch_transition_failures", 0) > 0
        )
    archive_grew = archive_size_after > archive_size_before
    if (runner_silent or engine_silent) and not archive_grew:
        logger.warning(
            "[run] non-zero exit: archive did not grow "
            "(before={}, after={}) and at least one silent failure "
            "counter fired (runner={}, engine={})",
            archive_size_before,
            archive_size_after,
            runner_silent,
            engine_silent,
        )
        return 2
    return 0


async def run_with_config(cfg: ExperimentConfig) -> int:
    """End-to-end runner the CLI invokes when not in dry-run mode.

    Sequence:

    1. Build the pure object graph.
    2. Open an ambient :class:`WorkerPool` so stage executors share a
       single subprocess pool for the lifetime of the run.
    3. Construct the writer, mutation operator, metrics tracker,
       program loader, DAG-runner instance and evolution-engine
       instance from the graph.
    4. Build the :class:`DataPlane` coordinator, mint the per-subspace
       :class:`EngineRoot`, derive an :class:`ActorIdentity`, and
       attach them to every coordinator-aware object.
    5. Acquire the instance lock, honour ``redis.resume`` (recover
       stranded RUNNING programs and restore engine/strategy state)
       or load the initial population from ``problem.problem_dir``.
    6. Start both background tasks and block in
       :func:`serve_until_signal` until SIGINT/SIGTERM or natural
       completion.
    7. Idempotent shutdown in a single ``finally`` block: stop the
       engine and runner, drain the worker pool, close storage,
       shut both DataPlanes down, close the writer.
    """
    from gigaevo.dataplane import (
        build_actor_identity,
        build_dataplane,
        build_engine_root,
        wire_bandit_router,
        wire_dag_runner,
        wire_evolution_engine,
        wire_storage,
    )
    from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator
    from gigaevo.problems.initial_loaders import DirectoryProgramLoader
    from gigaevo.programs.stages.python_executors.wrapper import (
        default_exec_runner_pool,
        reset_ambient_exec_runner_pool,
        set_ambient_exec_runner_pool,
    )
    from gigaevo.runner.dag_runner import DagRunner
    from gigaevo.utils.metrics_tracker import MetricsTracker
    from gigaevo.utils.serve import serve_until_signal

    start_time = time.time()
    logger.info(
        "GigaEvo run starting | experiment={} problem={}",
        cfg.name,
        cfg.problem.name,
    )

    graph = build_object_graph(cfg)
    redis_storage = graph["redis_storage"]
    problem_context = graph["problem_context"]
    llm = graph["llm"]
    strategy = graph["strategy"]
    runtime_engine_config = graph["runtime_engine_config"]
    evolution_context = graph["evolution_context"]
    dag_blueprint = graph["dag_blueprint"]
    runtime_runner_config = graph["runtime_runner_config"]
    prompt_fetcher_runtime = evolution_context.prompt_fetcher

    exec_runner_pool = default_exec_runner_pool()
    pool_token = set_ambient_exec_runner_pool(exec_runner_pool)

    writer: CompositeLogger | None = None
    metrics_tracker: MetricsTracker | None = None
    dag_runner: DagRunner | None = None
    evolution_engine: EvolutionEngine | None = None
    dataplane: DataPlane | None = None
    prompt_dataplane: DataPlane | None = None

    archive_size_before = 0
    archive_size_after = 0
    try:
        writer = _build_default_writer(cfg)

        mutation_operator = LLMMutationOperator(
            llm_wrapper=llm,
            problem_context=problem_context,
            prompts_dir=cfg.pipeline.prompts_dir,
            prompt_fetcher=prompt_fetcher_runtime,
        )

        metrics_tracker = MetricsTracker(
            storage=redis_storage,
            metrics_context=problem_context.metrics_context,
            writer=writer,
            interval=runtime_engine_config.metrics_collection_interval,
        )

        dag_runner = DagRunner(
            storage=redis_storage,
            dag_blueprint=dag_blueprint,
            config=runtime_runner_config,
            writer=writer,
        )

        evolution_engine = _build_evolution_engine(
            cfg,
            storage=redis_storage,
            strategy=strategy,
            mutation_operator=mutation_operator,
            runtime_engine_config=runtime_engine_config,
            writer=writer,
            metrics_tracker=metrics_tracker,
        )

        program_loader = DirectoryProgramLoader(cfg.problem.problem_dir)

        dataplane = await build_dataplane(
            cfg.redis.url,
            key_prefix=cfg.dataplane.key_prefix,
            max_connections=cfg.dataplane.max_connections,
        )
        engine_root = build_engine_root()
        actor = build_actor_identity()

        wire_storage(redis_storage, dataplane, engine_root)
        wire_dag_runner(dag_runner, dataplane, engine_root)
        wire_evolution_engine(evolution_engine, dataplane, engine_root)
        _wire_archive_storages(strategy, dataplane, engine_root)
        wire_bandit_router(llm, dataplane, actor, engine_root)
        prompt_dataplane = await _maybe_build_prompt_dataplane(
            mutation_operator, dataplane, actor
        )

        await redis_storage.acquire_instance_lock()

        has_data = await redis_storage.has_data()
        resume = cfg.redis.resume

        if has_data and not resume:
            raise RuntimeError(
                f"Redis database {cfg.redis.db} at {cfg.redis.host}:{cfg.redis.port} "
                f"is not empty under prefix {cfg.dataplane.key_prefix!r}. "
                f"Flush with: redis-cli -h {cfg.redis.host} -p {cfg.redis.port} "
                f"-n {cfg.redis.db} FLUSHDB  — or set redis.resume=True."
            )

        if has_data and resume:
            recovered = await redis_storage.recover_stranded_programs()
            if recovered:
                logger.info("Recovered {} stranded RUNNING program(s)", recovered)
            await evolution_engine.restore_state()
            await evolution_engine.strategy.restore_state()
            logger.info(
                "Resumed with {} existing programs",
                await redis_storage.size(),
            )
        else:
            programs = await program_loader.load(redis_storage)
            logger.info("Loaded {} initial program(s)", len(programs))

        try:
            archive_size_before = len(
                await evolution_engine.strategy.get_program_ids()
            )
        except Exception:
            # Strategy archive read can fail on a partially-initialised
            # data plane; treat the baseline as zero so a successful
            # later read still produces a positive delta.
            archive_size_before = 0

        try:
            dag_runner.start()
            evolution_engine.start()
            logger.info(
                "Evolution running (max_generations={})",
                runtime_engine_config.max_generations or "unlimited",
            )
            await serve_until_signal(
                stop_coros=(evolution_engine.stop(), dag_runner.stop()),
                on_stop=(evolution_engine.task, dag_runner.task),
            )
        finally:
            # Idempotent stops: covers the path where something between
            # start() and serve_until_signal raises and leaves the
            # background tasks alive. stop() on an already-stopped
            # component is a no-op. ``stop()`` returning normally is the
            # common path; a real exception there is informative, so log
            # it (don't swallow with ``contextlib.suppress``).
            try:
                await evolution_engine.stop()
            except Exception:
                logger.exception("[run] evolution_engine.stop raised")
            try:
                await dag_runner.stop()
            except Exception:
                logger.exception("[run] dag_runner.stop raised")
            try:
                archive_size_after = len(
                    await evolution_engine.strategy.get_program_ids()
                )
            except Exception:
                archive_size_after = archive_size_before

        return _resolve_exit_code(
            dag_runner,
            evolution_engine,
            archive_size_before,
            archive_size_after,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return _resolve_exit_code(
            dag_runner,
            evolution_engine,
            archive_size_before,
            archive_size_after,
        )
    except Exception:
        logger.exception("Run failed")
        raise
    finally:
        # Drain pool workers before unbinding the contextvar so late
        # exec calls during shutdown still resolve to the shared pool.
        # Each ``close`` / ``shutdown`` below logs at exception level so
        # a real backend bug (Redis client double-close, transport stuck
        # at exit, …) is discoverable, but the next teardown step still
        # runs — order matters more than success on the shutdown path.
        try:
            await exec_runner_pool.shutdown()
        except Exception:
            logger.exception("[run] exec_runner_pool.shutdown raised")
        reset_ambient_exec_runner_pool(pool_token)
        # Tear the LokyBackend down here too — the python_executors path
        # uses a separate process pool from the subprocess-script
        # WorkerPool, and loky's manager thread holds sem_open file
        # descriptors under /dev/shm that survive the parent unless the
        # backend's own shutdown sequence runs. atexit catches the
        # uncaught-exception path, but driving the same call here means
        # a clean exit reclaims the shared-memory budget before the
        # next run begins.
        from gigaevo.programs.stages.python_executors.wrapper import (
            shutdown_executor,
        )

        try:
            shutdown_executor(wait=False)
        except Exception:
            logger.exception("[run] loky shutdown_executor raised")
        if redis_storage is not None:
            try:
                await redis_storage.close()
            except Exception:
                logger.exception("[run] redis_storage.close raised")
        # Shutdown the coordinator after the storage so tail writes
        # storage performs during close() still see a live pool.
        if dataplane is not None:
            try:
                await dataplane.shutdown()
            except Exception:
                logger.exception("[run] dataplane.shutdown raised")
        if prompt_dataplane is not None:
            try:
                await prompt_dataplane.shutdown()
            except Exception:
                logger.exception("[run] prompt_dataplane.shutdown raised")
        if writer is not None:
            try:
                writer.close()
            except Exception:
                logger.exception("[run] writer.close raised")
        duration = time.time() - start_time
        logger.info("Duration: {:.1f}s ({:.2f}h)", duration, duration / 3600)
