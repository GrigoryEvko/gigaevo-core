from __future__ import annotations

import os
from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic import Field, SecretStr

from gigaevo.config.schemas._base import (
    FinitePositiveFloat,
    FrozenStrictModel,
    NonBlankStr,
)

if TYPE_CHECKING:
    from gigaevo.database.redis.config import RedisProgramStorageConfig


_REDIS_PASSWORD_ENV: str = "REDIS_PASSWORD"


class RedisConfig(FrozenStrictModel):
    """Connection coordinates that resolve to a single Redis logical DB.

    The runtime construction site (``RedisProgramStorage``) accepts a
    pre-formed ``redis_url``; this schema exposes the host/port/db
    tuple plus optional credentials and TLS, and derives the URL via a
    computed field.
    """

    host: NonBlankStr = Field(
        default="localhost",
        min_length=1,
        description="Redis server hostname or IP address.",
    )
    port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis server TCP port.",
    )
    # The standard Redis server ships sixteen logical databases (0-15);
    # values outside that range are rejected by the server at connect
    # time. Surfacing the ceiling here turns a connect-time refusal into
    # a config-load-time error.
    db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis logical database index (0-15) for this run's program storage.",
    )
    username: NonBlankStr | None = Field(
        default=None,
        description="ACL username; None for the default user. Embedded in the URL when set.",
    )
    # The secret never lands in the serialised representation: tyro
    # cannot reflect it into ``--help``, ``config.json`` does not carry
    # it, and ``experiment_id`` hashes nothing about it. The runtime
    # ``url`` property fetches ``REDIS_PASSWORD`` lazily and embeds
    # the URL-encoded value.
    password: SecretStr | None = Field(
        default=None,
        repr=False,
        exclude=True,
        description="Optional auth password; defaults to REDIS_PASSWORD at URL-build time, omitted from dumps.",
    )
    tls: bool = Field(
        default=False,
        description="When true, emit a rediss:// URL so the client negotiates TLS.",
    )
    # ``resume`` is excluded from the serialized representation because
    # ``ExperimentConfig.experiment_id`` hashes ``model_dump_json()`` and
    # two runs that differ only in their resume intent are conceptually
    # the same experiment — flipping ``resume`` must not move the run
    # into a fresh output directory.
    resume: bool = Field(
        default=False,
        exclude=True,
        description="When true, reuse existing keys under the prefix instead of clearing them at startup.",
    )

    def _resolved_password(self) -> str | None:
        """Resolve the password from the explicit ``SecretStr`` field
        or from ``REDIS_PASSWORD``. Returns ``None`` when neither
        source is populated, so an unauthenticated server keeps a
        bare ``redis://host:port/db`` URL."""
        if self.password is not None:
            return self.password.get_secret_value()
        return os.environ.get(_REDIS_PASSWORD_ENV) or None

    @property
    def url(self) -> str:
        """Compose the Redis URL with credentials and scheme.

        Credentials are URL-quoted so a password containing ``@``,
        ``:``, ``/`` or other reserved bytes cannot inject into the
        URL grammar; the scheme switches to ``rediss://`` when
        :attr:`tls` is set so the client negotiates TLS.
        """
        scheme = "rediss" if self.tls else "redis"
        password = self._resolved_password()
        if self.username is not None or password is not None:
            user_part = quote(self.username, safe="") if self.username else ""
            pwd_part = f":{quote(password, safe='')}" if password is not None else ""
            credentials = f"{user_part}{pwd_part}@"
        else:
            credentials = ""
        return f"{scheme}://{credentials}{self.host}:{self.port}/{self.db}"

    def to_storage_config(
        self,
        *,
        key_prefix: str,
        max_connections: int = 100,
        connection_pool_timeout: float = 60.0,
        health_check_interval: int = 180,
        max_retries: int = 5,
        retry_delay: float = 0.2,
    ) -> RedisProgramStorageConfig:
        from gigaevo.database.redis.config import RedisProgramStorageConfig

        return RedisProgramStorageConfig(
            redis_url=self.url,
            key_prefix=key_prefix,
            max_connections=max_connections,
            connection_pool_timeout=connection_pool_timeout,
            health_check_interval=health_check_interval,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )


class DataPlaneSettings(FrozenStrictModel):
    """Tunables for the DataPlane coordinator.

    ``RedisConfig`` is referenced by composition so the dataplane and the
    program storage share a single Redis backing without a second URL
    declaration in the experiment file.
    """

    redis: RedisConfig = Field(
        description="Redis connection coordinates shared with the program storage backend.",
    )
    key_prefix: NonBlankStr = Field(
        min_length=1,
        description="Redis key prefix that namespaces this experiment; must equal gigaevo:{experiment.name}.",
    )
    max_connections: int = Field(
        default=16,
        ge=1,
        description="Upper bound on the connection pool used by the dataplane coordinator.",
    )
    startup_timeout_s: FinitePositiveFloat = Field(
        default=10.0,
        description="Seconds to wait for the dataplane to become ready before failing startup.",
    )
