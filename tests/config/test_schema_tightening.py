"""Per-bug regression tests for the schema-validation tightening
pass. Each ``Test*`` group locks in one previously-silent invariant.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from gigaevo.config.schemas import (
    BehaviorSpaceConfig,
    BusedEngineConfig,
    BusTopologyConfig,
    ChatOpenAIConfig,
    DataPlaneSettings,
    DefaultPipelineBuilderConfig,
    AutoPipelineBuilderConfig,
    EnsembleRouterConfig,
    ExperimentConfig,
    FitnessArchiveRemoverConfig,
    FitnessProportionalEliteSelectorConfig,
    GenerationalEngineConfig,
    IslandConfig,
    LoggingSettings,
    MigrationBusConfig,
    PipelineConfig,
    ProblemConfig,
    RedisConfig,
    RedisMetricsTrackerConfig,
    RedisStreamTransportConfig,
    RingTopologyConfig,
    SingleIslandConfig,
    StandardAcceptorConfig,
    SteadyStateEngineConfig,
    SumArchiveSelectorConfig,
    TBTrackerConfig,
    TopFitnessMigrantSelectorConfig,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _bspace() -> BehaviorSpaceConfig:
    return BehaviorSpaceConfig(
        keys=["fitness"],
        bounds=[(0.0, 1.0)],
        resolutions=[100],
        binning_types=["linear"],
    )


def _island(island_id: str = "main") -> IslandConfig:
    return IslandConfig(
        island_id=island_id,
        behavior_space=_bspace(),
        archive_selector=SumArchiveSelectorConfig(
            fitness_keys=["fitness"],
            fitness_key_higher_is_better=[True],
        ),
        elite_selector=FitnessProportionalEliteSelectorConfig(fitness_key="fitness"),
        migrant_selector=TopFitnessMigrantSelectorConfig(fitness_key="fitness"),
    )


def _experiment_kwargs(name: str = "tight", **overrides: object) -> dict:  # type: ignore[no-untyped-def]
    redis = RedisConfig()
    base: dict = {
        "name": name,
        "redis": redis,
        "dataplane": DataPlaneSettings(redis=redis, key_prefix=f"gigaevo:{name}"),
        "problem": ProblemConfig(
            name=name,
            problem_dir=Path("/srv/gigaevo/problems/x"),
        ),
        "algorithm": SingleIslandConfig(island=_island()),
        "engine": SteadyStateEngineConfig(
            max_in_flight=5, max_mutations_per_generation=50
        ),
        "pipeline": PipelineConfig(builder=DefaultPipelineBuilderConfig()),
        "llm": EnsembleRouterConfig(models=[ChatOpenAIConfig(model="gpt-4o-mini")]),
    }
    base.update(overrides)
    return base


class TestBehaviorSpaceFiniteBounds:
    """``BehaviorSpaceConfig.bounds`` previously accepted NaN, ±inf,
    and zero-width intervals — each leads to silent garbage bins at
    runtime."""

    def test_inf_upper_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorSpaceConfig(
                keys=["a"],
                bounds=[(0.0, math.inf)],
                resolutions=[10],
                binning_types=["linear"],
            )

    def test_nan_lower_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorSpaceConfig(
                keys=["a"],
                bounds=[(math.nan, 1.0)],
                resolutions=[10],
                binning_types=["linear"],
            )

    def test_zero_width_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorSpaceConfig(
                keys=["a"],
                bounds=[(1.0, 1.0)],
                resolutions=[10],
                binning_types=["linear"],
            )

    def test_inverted_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorSpaceConfig(
                keys=["a"],
                bounds=[(1.0, 0.0)],
                resolutions=[10],
                binning_types=["linear"],
            )

    def test_normal_pair_accepted(self) -> None:
        cfg = BehaviorSpaceConfig(
            keys=["a"],
            bounds=[(0.0, 1.0)],
            resolutions=[10],
            binning_types=["linear"],
        )
        assert cfg.bounds == [(0.0, 1.0)]


class TestBehaviorSpaceUniqueKeys:
    """Duplicate keys produced an unreachable second axis at runtime."""

    def test_dup_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorSpaceConfig(
                keys=["a", "a"],
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                resolutions=[10, 10],
                binning_types=["linear", "linear"],
            )

    def test_unique_keys_accepted(self) -> None:
        cfg = BehaviorSpaceConfig(
            keys=["a", "b"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            resolutions=[10, 10],
            binning_types=["linear", "linear"],
        )
        assert cfg.keys == ["a", "b"]


class TestSumArchiveUniqueFitnessKeys:
    """``SumArchiveSelectorConfig.fitness_keys`` summing the same key
    twice was a config-time bug masquerading as a tuning choice."""

    def test_duplicate_fitness_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SumArchiveSelectorConfig(
                fitness_keys=["score", "score"],
                fitness_key_higher_is_better=[True, True],
            )


class TestRedisDbCeiling:
    """The stock Redis ships sixteen logical databases (0-15);
    accepting db=999 only deferred the failure to connect time."""

    def test_db_above_15_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RedisConfig(db=16)

    def test_db_15_accepted(self) -> None:
        assert RedisConfig(db=15).db == 15


class TestRedisResumeExcludedFromExperimentId:
    """``experiment_id`` hashes the dumped tree; flipping ``resume``
    must not move the run into a fresh output directory."""

    def test_resume_does_not_change_experiment_id(self) -> None:
        cfg_a = ExperimentConfig(**_experiment_kwargs())
        # Build a second ExperimentConfig with redis.resume=True.
        kwargs_b = _experiment_kwargs()
        redis_b = RedisConfig(resume=True)
        kwargs_b["redis"] = redis_b
        kwargs_b["dataplane"] = DataPlaneSettings(
            redis=redis_b, key_prefix=kwargs_b["dataplane"].key_prefix
        )
        cfg_b = ExperimentConfig(**kwargs_b)
        assert cfg_a.experiment_id == cfg_b.experiment_id


class TestPipelineStageBelowDagTimeout:
    """``stage_timeout > dag_timeout`` is structurally unreachable —
    the per-stage budget cannot fire before the whole-DAG budget
    expires, so accepting the inversion silently disabled the
    per-stage guard."""

    def test_default_builder_stage_above_dag_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DefaultPipelineBuilderConfig(dag_timeout=10.0, stage_timeout=100.0)

    def test_auto_builder_stage_above_dag_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AutoPipelineBuilderConfig(dag_timeout=10.0, stage_timeout=100.0)

    def test_equal_timeouts_accepted(self) -> None:
        cfg = DefaultPipelineBuilderConfig(dag_timeout=100.0, stage_timeout=100.0)
        assert cfg.stage_timeout == cfg.dag_timeout


class TestSeedBounded:
    """``seed`` is reserved for forwards compatibility; bounding it to
    the 32-bit unsigned range shields downstream RNG calls from
    overflow when the field is wired up."""

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(**_experiment_kwargs(seed=-1))

    def test_seed_above_u32_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(**_experiment_kwargs(seed=2**32))

    def test_seed_at_ceiling_accepted(self) -> None:
        cfg = ExperimentConfig(**_experiment_kwargs(seed=2**32 - 1))
        assert cfg.seed == 2**32 - 1


class TestIntCeilings:
    """Unbounded ints turned typo runaways into silent OOMs."""

    def test_max_in_flight_above_ceiling_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SteadyStateEngineConfig(max_in_flight=10**7)

    def test_max_generations_above_ceiling_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GenerationalEngineConfig(max_generations=10**12)

    def test_tb_queue_size_ceiling(self) -> None:
        with pytest.raises(ValidationError):
            TBTrackerConfig(logdir=Path("/tmp/x"), queue_size=10**9)

    def test_redis_metrics_history_ceiling(self) -> None:
        with pytest.raises(ValidationError):
            RedisMetricsTrackerConfig(max_history_per_metric=10**10)

    def test_max_stream_len_ceiling(self) -> None:
        with pytest.raises(ValidationError):
            RedisStreamTransportConfig(
                run_id="r",
                stream_key="s",
                max_stream_len=10**10,
            )


class TestStandardAcceptorValidityKey:
    """``validity_key`` controls which metric flags a program as
    valid; an all-whitespace key silently misroutes every program."""

    def test_blank_validity_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StandardAcceptorConfig(validity_key="   ")


class TestMigrationBusRunIdsNonBlank:
    """Ring topologies derive predecessor relationships by string
    equality on ``run_ids``; a NUL byte hidden in one ID silently
    breaks the ring."""

    def test_run_ids_with_nul_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RingTopologyConfig(run_ids=["good", "bad\x00id"])

    def test_run_ids_all_non_blank_accepted(self) -> None:
        cfg = RingTopologyConfig(run_ids=["a", "b"])
        assert cfg.run_ids == ["a", "b"]


class TestMigrationBusTransportDbCeiling:
    """Same Redis-DB ceiling applied to the migration-bus transport."""

    def test_db_above_15_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RedisStreamTransportConfig(
                run_id="r",
                stream_key="s",
                db=99,
            )

    def test_block_ms_above_ceiling_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RedisStreamTransportConfig(
                run_id="r",
                stream_key="s",
                block_ms=10**9,
            )


class TestExperimentDescriptionsLanded:
    """Spot-check that the ``ExperimentConfig`` field descriptions
    targeted by the description audit are non-empty after the fix."""

    @pytest.mark.parametrize(
        "field",
        [
            "redis",
            "dataplane",
            "problem",
            "algorithm",
            "engine",
            "pipeline",
            "llm",
            "runner",
            "prompt_fetcher",
        ],
    )
    def test_field_has_description(self, field: str) -> None:
        info = ExperimentConfig.model_fields[field]
        assert info.description and info.description.strip()
