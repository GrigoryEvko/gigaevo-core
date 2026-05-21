from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from gigaevo.config.schemas._base import (
    FinitePositiveFloat,
    FrozenStrictModel,
    NonBlankStr,
)

if TYPE_CHECKING:
    from gigaevo.evolution.bus.node import MigrationNode
    from gigaevo.evolution.bus.topology import Topology
    from gigaevo.evolution.bus.transport import Transport


class BusTopologyConfig(FrozenStrictModel):
    """Fully-connected bus: every run accepts migrants from every
    other run (and rejects its own). The runtime ``BusTopology`` is
    parameterless — this schema is a marker for the discriminated
    union."""

    kind: Literal["bus"] = "bus"

    def build(self) -> Topology:
        from gigaevo.evolution.bus.topology import BusTopology

        return BusTopology()


class RingTopologyConfig(FrozenStrictModel):
    """Ring topology: each run accepts migrants only from its
    predecessor in ``run_ids``. The list defines the ring order and
    wraps around so the first run accepts from the last."""

    kind: Literal["ring"] = "ring"
    run_ids: list[NonBlankStr] = Field(
        min_length=2,
        description="Ring order; each run accepts migrants only from its predecessor, wrapping around.",
    )

    @model_validator(mode="after")
    def _run_ids_unique(self) -> RingTopologyConfig:
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError(
                f"RingTopologyConfig.run_ids must be unique; got {self.run_ids}"
            )
        return self

    def build(self) -> Topology:
        from gigaevo.evolution.bus.topology import RingTopology

        return RingTopology(run_ids=list(self.run_ids))


TopologyConfig = Annotated[
    BusTopologyConfig | RingTopologyConfig,
    Field(discriminator="kind"),
]


class RedisStreamTransportConfig(FrozenStrictModel):
    """Redis Streams transport with SETNX exclusive claiming. The
    schema mirrors the runtime ``RedisStreamTransport`` constructor
    surface; ``run_id`` and ``stream_key`` are required because they
    encode the experiment identity and the cross-run namespace."""

    run_id: NonBlankStr = Field(
        min_length=1,
        description="Local run identity stamped on every published envelope.",
    )
    stream_key: NonBlankStr = Field(
        min_length=1,
        description="Redis Stream name carrying the cross-run migration traffic.",
    )
    host: NonBlankStr = Field(
        default="localhost",
        min_length=1,
        description="Redis hostname for the migration-bus DB.",
    )
    port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis TCP port for the migration-bus DB.",
    )
    # Standard Redis ships sixteen logical databases (0-15); see RedisConfig.
    db: int = Field(
        default=15,
        ge=0,
        le=15,
        description="Redis DB index reserved for the migration stream (0-15).",
    )
    # 10 million stream entries already exceeds Redis memory on standard
    # configurations; the ceiling rejects typo-driven runaway streams.
    max_stream_len: int = Field(
        default=1000,
        ge=1,
        le=10_000_000,
        description="Maximum entries retained on the Redis Stream before trimming.",
    )
    # 24-hour ceiling: any longer SETNX lease is effectively "no lease".
    claim_ttl: int = Field(
        default=120,
        ge=1,
        le=86_400,
        description="SETNX claim lease in seconds before an unconsumed envelope becomes available again.",
    )
    # 1-hour ceiling on XREAD blocking: longer blocks defeat shutdown.
    block_ms: int = Field(
        default=5000,
        ge=0,
        le=3_600_000,
        description="XREAD BLOCK timeout in milliseconds when polling the stream.",
    )

    def build(self) -> Transport:
        from gigaevo.evolution.bus.transport import RedisStreamTransport

        return RedisStreamTransport(
            run_id=self.run_id,
            stream_key=self.stream_key,
            host=self.host,
            port=self.port,
            db=self.db,
            max_stream_len=self.max_stream_len,
            claim_ttl=self.claim_ttl,
            block_ms=self.block_ms,
        )


class MigrationBusConfig(FrozenStrictModel):
    """Cross-run migration coordinator. Composes the transport, the
    topology filter, and the per-node tuning knobs.

    ``run_id`` is the local run's identity — by convention
    f"{problem.name}@db{redis.db}" so concurrent runs of the same
    experiment land in distinct namespaces. The ring topology's
    ``run_ids`` list must contain the local ``run_id`` (the runtime
    ``RingTopology`` silently rejects everything otherwise); the
    cross-field validator below catches that misconfiguration at
    load time.

    ``max_imports_per_generation`` is NOT carried here — it is a
    parameter of ``BusedEvolutionEngine`` (not ``MigrationNode``) and
    lives on :class:`BusedEngineConfig`."""

    run_id: NonBlankStr = Field(
        min_length=1,
        description="Local run identity; must match transport.run_id.",
    )
    transport: RedisStreamTransportConfig = Field(
        description="Wire-level transport (Redis Streams) that ships migration envelopes.",
    )
    topology: TopologyConfig = Field(
        description="Acceptance filter (bus / ring) that decides which peer runs are upstream of this one.",
    )
    max_buffer_size: int = Field(
        default=50,
        ge=1,
        description="Capacity of the in-process buffer holding consumed but not yet ingested migrants.",
    )
    consume_interval: FinitePositiveFloat = Field(
        default=5.0,
        description="Seconds between background polls of the migration stream.",
    )
    max_consume_per_poll: int = Field(
        default=20,
        ge=1,
        description="Upper bound on envelopes claimed per poll.",
    )

    @model_validator(mode="after")
    def _run_id_matches_transport(self) -> MigrationBusConfig:
        if self.transport.run_id != self.run_id:
            raise ValueError(
                f"MigrationBusConfig.run_id ({self.run_id!r}) must equal "
                f"transport.run_id ({self.transport.run_id!r}) — the bus "
                "and the transport must agree on the local identity"
            )
        return self

    @model_validator(mode="after")
    def _local_run_in_ring(self) -> MigrationBusConfig:
        topology = self.topology
        if isinstance(topology, RingTopologyConfig):
            if self.run_id not in topology.run_ids:
                raise ValueError(
                    f"MigrationBusConfig.run_id {self.run_id!r} must appear "
                    f"in the ring topology run_ids {topology.run_ids}"
                )
        return self

    def build(self) -> MigrationNode:
        from gigaevo.evolution.bus.node import MigrationNode

        return MigrationNode(
            run_id=self.run_id,
            transport=self.transport.build(),
            topology=self.topology.build(),
            max_buffer_size=self.max_buffer_size,
            consume_interval=self.consume_interval,
            max_consume_per_poll=self.max_consume_per_poll,
        )
