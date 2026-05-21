from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
import random
from typing import Final, TypeVar

from loguru import logger

from gigaevo.database.redis.config import RedisConnectionConfig
from gigaevo.exceptions import StorageError
from redis import asyncio as aioredis

T = TypeVar("T")

# Upper bound on the exponential backoff per attempt. The retry path
# multiplies the configured ``retry_delay`` by powers of two; without a
# ceiling a long-lived disconnect would push individual sleeps into the
# minute range. 30s matches typical TCP reconnect SLOs.
_MAX_BACKOFF_S: Final[float] = 30.0

# Background reconciler cadence. Polls ``PING`` and drops the cached
# pool on failure so the next op rebuilds against a healthy backend.
_RECONCILE_INTERVAL_S: Final[float] = 5.0


def _jittered_backoff(delay: float) -> float:
    """Return ``delay`` capped at ``_MAX_BACKOFF_S`` and scaled by a
    uniform random factor in ``[0.5, 1.0)``. The randomisation breaks
    the thundering-herd reconnect pattern when many workers retry
    against the same outage; the half-floor keeps the worst case at
    half the requested delay rather than collapsing toward zero.
    """
    return min(delay, _MAX_BACKOFF_S) * (0.5 + random.random() * 0.5)


class RedisConnection:
    """Manages Redis connection with retry logic and graceful shutdown."""

    def __init__(self, config: RedisConnectionConfig):
        self.config = config
        self._redis: aioredis.Redis | None = None
        self._lock = asyncio.Lock()
        self._closing = False
        self._reconcile_task: asyncio.Task[None] | None = None

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    @property
    def is_closing(self) -> bool:
        return self._closing

    async def _drop_pool(self) -> None:
        """Close and clear the cached pool so the next :meth:`get` rebuilds.

        Errors from ``aclose`` / ``disconnect`` are swallowed because the
        caller has already classified the connection as unusable; raising
        here would shadow the original failure that triggered the drop.
        """
        r, self._redis = self._redis, None
        if r is None:
            return
        with contextlib.suppress(Exception):
            await r.aclose()  # type: ignore[attr-defined]
        with contextlib.suppress(Exception):
            await r.connection_pool.disconnect(inuse_connections=True)

    async def get(self) -> aioredis.Redis:
        """Get Redis connection, creating one if needed."""
        if self._closing:
            raise StorageError("RedisConnection is closing; cannot get connection.")
        if self._redis is not None:
            return self._redis

        async with self._lock:
            if self._redis is None:
                if self._closing:
                    raise StorageError(
                        "RedisConnection is closing; cannot get connection."
                    )
                r = aioredis.from_url(
                    str(self.config.redis_url),
                    decode_responses=True,
                    max_connections=self.config.max_connections,
                    health_check_interval=self.config.health_check_interval,
                    socket_connect_timeout=self.config.connection_pool_timeout,
                    socket_timeout=self.config.connection_pool_timeout,
                    retry_on_timeout=True,
                )
                await r.ping()
                logger.debug("[RedisConnection] Connected to {}", self.config.redis_url)
                self._redis = r

        return self._redis

    def start_reconciler(self) -> None:
        """Start the background reconciler if not already running.

        The reconciler PINGs the cached pool every
        ``_RECONCILE_INTERVAL_S`` seconds and drops the pool on failure
        so the next op rebuilds. Opt-in so unit tests that drive
        ``execute`` directly do not race with a background sleep loop.
        Idempotent: calling twice or after :meth:`close` is a no-op.
        """
        if self._closing or self._reconcile_task is not None:
            return
        self._reconcile_task = asyncio.create_task(self._reconcile_loop())

    async def execute(
        self, name: str, fn: Callable[[aioredis.Redis], Awaitable[T]]
    ) -> T:
        """Execute a Redis operation with retry logic.

        On final failure, the cached pool is dropped so the next call
        reconstructs against whatever state the backend recovered to.
        """
        if self._closing:
            raise StorageError(f"Redis op {name} refused: connection is closing.")

        delay = self.config.retry_delay
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return await fn(await self.get())
            except Exception as e:
                if attempt == self.config.max_retries or self._closing:
                    # Final failure: drop the cached pool so the next
                    # call rebuilds against whatever state the backend
                    # recovered to. Without this, a half-open pool
                    # would surface the same corruption on every op.
                    await self._drop_pool()
                    logger.warning(
                        "[RedisConnection] {} failed after {} attempts: {}",
                        name,
                        self.config.max_retries,
                        e,
                    )
                    raise StorageError(f"Redis op {name} failed: {e}") from e
                logger.warning(
                    "[RedisConnection] {} retry {}/{}: {}",
                    name,
                    attempt,
                    self.config.max_retries,
                    e,
                )
                await asyncio.sleep(_jittered_backoff(delay))
                delay *= 2

        raise StorageError(
            f"Redis op {name} failed after {self.config.max_retries} attempts"
        )

    async def _reconcile_loop(self) -> None:
        """Periodically PING the pool; on failure, drop the cached pool.

        Quiet by design: the next :meth:`get` rebuilds the pool. Errors
        are logged at DEBUG so a steady-state outage does not flood
        the log with the same line every cycle.
        """
        while not self._closing:
            try:
                await asyncio.sleep(_RECONCILE_INTERVAL_S)
            except asyncio.CancelledError:
                return
            if self._closing:
                return
            r = self._redis
            if r is None:
                continue
            try:
                await r.ping()  # type: ignore[misc]
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001 - reconciler boundary
                logger.debug(
                    "[RedisConnection] reconcile PING failed; dropping pool: {}", e
                )
                await self._drop_pool()

    async def close(self) -> None:
        """Close the Redis connection."""
        self._closing = True

        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reconcile_task
            self._reconcile_task = None

        await self._drop_pool()
        await asyncio.sleep(0)  # Yield to event loop
