from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from gigaevo.config.schemas._base import (
    FrozenStrictModel,
    NonBlankStr,
    reject_empty_or_cwd_path,
)

if TYPE_CHECKING:
    from gigaevo.problems.context import ProblemContext


class ProblemConfig(FrozenStrictModel):
    """Pointer to a problem directory plus optional metric overrides.

    The experiment-root layer builds the runtime ``ProblemContext`` via
    :meth:`build`, which materialises the ``MetricsContext`` from the
    directory's ``metrics.yaml``.

    ``primary_metric`` and ``higher_is_better`` are optional overrides
    for the values that ``MetricsContext`` derives from ``metrics.yaml``;
    leaving them as ``None`` defers to the on-disk declaration. Setting
    them explicitly is useful for experiments that want to optimise a
    secondary metric without editing the shared ``metrics.yaml``.
    """

    name: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="Logical problem name used in metric paths and log lines.",
    )
    problem_dir: Path = Field(
        description="Directory containing the problem's metrics.yaml, task description, and evaluator.",
    )
    primary_metric: NonBlankStr | None = Field(
        default=None,
        min_length=1,
        description="Override the on-disk primary metric; leave None to use metrics.yaml.",
    )
    higher_is_better: bool | None = Field(
        default=None,
        description="Override the on-disk fitness direction; leave None to use metrics.yaml.",
    )

    @field_validator("problem_dir")
    @classmethod
    def _problem_dir_not_empty(cls, value: Path) -> Path:
        # ``problem_dir`` is non-Optional so the helper's None branch is
        # never reached; the cast keeps the type system honest.
        return reject_empty_or_cwd_path("problem_dir", value)  # type: ignore[return-value]

    def build(self) -> ProblemContext:
        """Materialise the runtime ``ProblemContext`` after asserting
        the directory exists and carries a ``metrics.yaml``.

        The disk-shape check lives here rather than on the schema
        field so that ``model_validate_json`` can round-trip a dumped
        config without the original problem layout being mounted; the
        check fires at runtime construction, which is when the
        directory is actually required. A non-directory ``problem_dir``
        (``/etc/passwd`` and similar) fails with a clear typed error
        instead of crashing deep inside ``MetricsContext`` parsing."""
        from gigaevo.problems.context import ProblemContext
        from gigaevo.problems.layout import ProblemLayout as PL

        if not self.problem_dir.is_dir():
            raise ValueError(
                f"problem_dir {self.problem_dir!s} is not a directory"
            )
        metrics_path = self.problem_dir / PL.METRICS_FILE
        if not metrics_path.is_file():
            raise ValueError(
                f"problem_dir {self.problem_dir!s} is missing "
                f"{PL.METRICS_FILE!s}; this file is required to build the "
                "MetricsContext"
            )
        return ProblemContext(self.problem_dir)
