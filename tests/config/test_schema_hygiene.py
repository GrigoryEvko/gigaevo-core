"""Hygiene tests that walk every concrete schema model and assert
class-wide invariants on field metadata.

These complement the per-field unit tests by closing the loop on
"someone adds a new ``float`` field and forgets to opt out of
``inf``" or "someone adds a new ``Field()`` without a description".
"""

from __future__ import annotations

import inspect
import types
import typing

import pytest
from pydantic import BaseModel, ValidationError

import gigaevo.config.schemas as schemas


# LLM schemas live behind their own typed surface and are excluded
# here so the hygiene tests stay scoped to the structural schema
# package; the LLM module ships its own targeted hygiene checks.
_EXEMPT_MODULES: frozenset[str] = frozenset({"gigaevo.config.schemas.llm"})


def _all_concrete_models() -> list[type[BaseModel]]:
    """Collect every ``BaseModel`` subclass that the schema package
    exports as a concrete (non-``Annotated`` / non-union) name."""
    seen: dict[str, type[BaseModel]] = {}
    for name in schemas.__all__:
        obj = getattr(schemas, name)
        if inspect.isclass(obj) and issubclass(obj, BaseModel):
            if obj.__module__ in _EXEMPT_MODULES:
                continue
            seen[obj.__module__ + "." + obj.__name__] = obj
    return list(seen.values())


def _union_arms(annotation: object) -> list[object]:
    """Return the arms of a ``Union`` annotation, or ``[annotation]``
    for non-union types."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        return list(typing.get_args(annotation))
    return [annotation]


def _is_float_field(annotation: object) -> bool:
    """``True`` when the annotation reduces to ``float`` after peeling
    away ``Optional`` / ``Union[float, None]`` wrappers."""
    for arm in _union_arms(annotation):
        if arm is type(None):
            continue
        # Annotated[...] would already be unwrapped by Pydantic into
        # ``annotation=float`` plus metadata, so the bare check is
        # enough.
        if arm is float:
            return True
    return False


class TestNoFloatAcceptsInfOrNaN:
    """Every ``float`` field across the typed schema package must opt
    out of ``inf`` and ``NaN`` via ``allow_inf_nan=False``. Without
    the opt-out Pydantic happily binds ``float('inf')`` even when a
    ``gt=0.0`` bound is declared, since the bound holds as a partial
    order on the extended reals — the downstream consumer then
    silently propagates the infinity.
    """

    def test_every_float_field_rejects_inf_nan(self) -> None:
        offenders: list[str] = []
        for model in _all_concrete_models():
            for field_name, field_info in model.model_fields.items():
                if not _is_float_field(field_info.annotation):
                    continue
                # ``_PydanticGeneralMetadata(allow_inf_nan=False)``
                # surfaces as a metadata entry with the named attribute.
                has_opt_out = any(
                    getattr(m, "allow_inf_nan", None) is False
                    for m in field_info.metadata
                )
                if not has_opt_out:
                    offenders.append(f"{model.__name__}.{field_name}")
        assert not offenders, (
            "every float field must declare allow_inf_nan=False (use the "
            "FinitePositiveFloat / FiniteNonNegativeFloat / FiniteFloat "
            "aliases from gigaevo.config.schemas._base); offenders: "
            + ", ".join(offenders)
        )


class TestEveryFieldHasDescription:
    """Every user-facing field must carry a ``description`` so the
    schema doubles as self-documentation for experiment authors.
    Discriminator literals (``kind: Literal["..."] = "..."``) are
    exempt — they encode the variant tag, not a tunable knob.
    """

    def test_every_field_has_non_empty_description(self) -> None:
        offenders: list[str] = []
        for model in _all_concrete_models():
            for field_name, field_info in model.model_fields.items():
                if field_name == "kind":
                    # Discriminator literals carry the variant marker,
                    # not a configuration concept.
                    continue
                desc = field_info.description
                if not desc or not desc.strip():
                    offenders.append(f"{model.__name__}.{field_name}")
        assert not offenders, (
            "every schema field needs a Field(description=...) entry; "
            "offenders: " + ", ".join(offenders)
        )


class TestFiniteFloatAliases:
    """Direct exercise of the shared aliases."""

    def test_finite_positive_rejects_inf(self) -> None:
        from gigaevo.config.schemas.redis import DataPlaneSettings, RedisConfig

        with pytest.raises(ValidationError):
            DataPlaneSettings(
                redis=RedisConfig(),
                key_prefix="gigaevo:x",
                startup_timeout_s=float("inf"),
            )

    def test_finite_positive_rejects_nan(self) -> None:
        from gigaevo.config.schemas.redis import DataPlaneSettings, RedisConfig

        with pytest.raises(ValidationError):
            DataPlaneSettings(
                redis=RedisConfig(),
                key_prefix="gigaevo:x",
                startup_timeout_s=float("nan"),
            )

    def test_finite_positive_rejects_zero(self) -> None:
        from gigaevo.config.schemas.redis import DataPlaneSettings, RedisConfig

        with pytest.raises(ValidationError):
            DataPlaneSettings(
                redis=RedisConfig(),
                key_prefix="gigaevo:x",
                startup_timeout_s=0.0,
            )

    def test_finite_non_negative_accepts_zero(self) -> None:
        from gigaevo.config.schemas.algorithm import BehaviorSpaceConfig

        cfg = BehaviorSpaceConfig(
            keys=["a"],
            bounds=[(0.0, 1.0)],
            resolutions=[10],
            binning_types=["linear"],
            expansion_buffer_ratio=0.0,
        )
        assert cfg.expansion_buffer_ratio == 0.0

    def test_finite_non_negative_rejects_inf(self) -> None:
        from gigaevo.config.schemas.algorithm import BehaviorSpaceConfig

        with pytest.raises(ValidationError):
            BehaviorSpaceConfig(
                keys=["a"],
                bounds=[(0.0, 1.0)],
                resolutions=[10],
                binning_types=["linear"],
                expansion_buffer_ratio=float("inf"),
            )


class TestNonBlankStrAlias:
    """The ``NonBlankStr`` alias must reject whitespace-only and
    control-character-bearing strings."""

    def test_redis_host_rejects_nul(self) -> None:
        from gigaevo.config.schemas.redis import RedisConfig

        with pytest.raises(ValidationError):
            RedisConfig(host="evil\x00.com")

    def test_redis_host_rejects_blank(self) -> None:
        from gigaevo.config.schemas.redis import RedisConfig

        with pytest.raises(ValidationError):
            RedisConfig(host="   \t  ")

    def test_redis_host_rejects_esc_control(self) -> None:
        from gigaevo.config.schemas.redis import RedisConfig

        with pytest.raises(ValidationError):
            RedisConfig(host="bad\x1bhost")

    def test_redis_host_accepts_normal(self) -> None:
        from gigaevo.config.schemas.redis import RedisConfig

        cfg = RedisConfig(host="redis.internal")
        assert cfg.host == "redis.internal"

    def test_key_prefix_rejects_nul(self) -> None:
        from gigaevo.config.schemas.redis import DataPlaneSettings, RedisConfig

        with pytest.raises(ValidationError):
            DataPlaneSettings(redis=RedisConfig(), key_prefix="gigaevo\x00x")

    def test_logging_rotation_rejects_blank(self) -> None:
        from gigaevo.config.schemas.logging import LoggingSettings

        with pytest.raises(ValidationError):
            LoggingSettings(rotation="   ")

    def test_logging_retention_rejects_nul(self) -> None:
        from gigaevo.config.schemas.logging import LoggingSettings

        with pytest.raises(ValidationError):
            LoggingSettings(retention="30 days\x00")

    def test_fitness_keys_rejects_blank(self) -> None:
        from gigaevo.config.schemas.algorithm import SumArchiveSelectorConfig

        with pytest.raises(ValidationError):
            SumArchiveSelectorConfig(
                fitness_keys=["   "],
                fitness_key_higher_is_better=[True],
            )

    def test_migration_bus_run_ids_rejects_nul(self) -> None:
        from gigaevo.config.schemas.migration_bus import RingTopologyConfig

        with pytest.raises(ValidationError):
            RingTopologyConfig(run_ids=["run1\x00", "run2"])

    def test_validity_key_rejects_blank(self) -> None:
        from gigaevo.config.schemas.engine import StandardAcceptorConfig

        with pytest.raises(ValidationError):
            StandardAcceptorConfig(validity_key="   ")
