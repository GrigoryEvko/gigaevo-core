"""Typed entry point for the evolutionary search runtime.

The CLI is intentionally thin: explicit construction with no decorator
magic, no chdir, no module singletons. Configuration loads through
:func:`build_experiment` (Pydantic-validated), CLI overrides apply via
tyro (auto-generated from the model field tree), the resolved config
dumps to JSON for reproducibility, and the typed object graph is
handed to :func:`gigaevo.config.object_graph.run_with_config`.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
import contextlib
import os
from pathlib import Path
import sys
import tempfile

from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError

from gigaevo.config.experiment_loader import build_experiment
from gigaevo.config.schemas.experiment import ExperimentConfig


def _build_initial_parser() -> argparse.ArgumentParser:
    """Build the argparse layer that owns the experiment-path positional
    and the ``--dry-run`` switch. ``add_help`` stays off so a trailing
    ``--help`` after the experiment path reaches the tyro layer and
    prints the typed-override field tree."""
    parser = argparse.ArgumentParser(
        prog="gigaevo",
        description="Evolutionary search runtime — typed entry point",
        add_help=False,
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        type=Path,
        help="Path to an experiment Python file that exports build() -> ExperimentConfig",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, validate, and dump the resolved config without invoking the engine",
    )
    parser.add_argument(
        "-h",
        "--help",
        dest="help",
        action="store_true",
        help="Print this help; with an experiment argument, also print the tyro field tree",
    )
    return parser


def _parse_initial_args(
    argv: list[str],
) -> tuple[Path | None, bool, bool, list[str]]:
    """Parse the experiment-path + dry-run prefix; forward the remainder
    to tyro for nested field overrides.

    Returns ``(experiment_path, dry_run, help_requested, overrides)``.
    """
    parser = _build_initial_parser()
    parsed, overrides = parser.parse_known_args(argv)
    return parsed.experiment, parsed.dry_run, parsed.help, overrides


def _apply_tyro_overrides(
    baseline: ExperimentConfig, override_args: list[str]
) -> ExperimentConfig:
    """Apply ``--key value`` overrides via tyro, re-running every Pydantic
    validator against the merged configuration."""
    if not override_args:
        return baseline

    import tyro

    return tyro.cli(
        ExperimentConfig,
        default=baseline,
        args=override_args,
        prog="gigaevo overrides",
    )


def _format_validation_loc(loc: tuple[object, ...]) -> str:
    """Map a pydantic ``loc`` tuple to a CLI-style ``--a.b.c`` flag name.

    Numeric indices stay numeric (``--items.0.name``); string segments
    have underscores rewritten to hyphens to match tyro's convention.
    """
    if not loc:
        return "<root>"
    parts: list[str] = []
    for seg in loc:
        if isinstance(seg, int):
            parts.append(str(seg))
        else:
            parts.append(str(seg).replace("_", "-"))
    return "--" + ".".join(parts)


def _format_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic ``ValidationError`` as a tyro-style framed block.

    The shape matches ``tyro``'s own argparse error frames so operators
    see a consistent error surface regardless of whether the failure was
    a CLI parse error or a cross-field validator firing.
    """
    title = "Configuration validation failed"
    lines = [
        "╭─ " + title + " " + "─" * max(0, 70 - len(title) - 4) + "╮",
    ]
    errors = exc.errors()
    if not errors:
        lines.append("│  (no error details available)")
    for err in errors:
        flag = _format_validation_loc(err.get("loc", ()))
        msg = err.get("msg", "invalid value")
        lines.append(f"│  {flag}")
        lines.append(f"│    {msg}")
        ctx = err.get("ctx")
        if isinstance(ctx, dict):
            for k, v in ctx.items():
                lines.append(f"│    ({k}: {v})")
    lines.append("╰" + "─" * 72 + "╯")
    return "\n".join(lines)


def _warn_on_repeated_flags(override_args: list[str]) -> None:
    """Emit a WARNING for each repeated ``--flag`` in the override list.

    Tyro / argparse silently last-wins on repeats. Operators chaining
    multiple sweeps or template fragments occasionally end up with two
    conflicting overrides for the same field; surfacing the collision
    lets them notice before the resolved-config dump bakes in the wrong
    value.
    """
    seen: OrderedDict[str, list[str]] = OrderedDict()
    i = 0
    while i < len(override_args):
        token = override_args[i]
        if token.startswith("--"):
            key, _, inline_value = token.partition("=")
            if inline_value:
                value = inline_value
            elif i + 1 < len(override_args) and not override_args[i + 1].startswith(
                "--"
            ):
                value = override_args[i + 1]
                i += 1
            else:
                value = ""
            seen.setdefault(key, []).append(value)
        i += 1
    for flag, values in seen.items():
        if len(values) <= 1:
            continue
        winning = values[-1]
        discarded = values[:-1]
        logger.warning(
            "Override {} appeared {} times; last-wins resolved to {!r}, "
            "discarding {!r}",
            flag,
            len(values),
            winning,
            discarded,
        )


def _dump_resolved_config(cfg: ExperimentConfig) -> Path:
    """Write ``config.json`` under ``output_dir/experiment_id`` and
    return the absolute path. The dump is the reproducibility record:
    given identical inputs, two runs share an output directory.

    Two concurrent sweep workers can resolve to the same
    ``experiment_id`` (overrides that don't affect the hashed fields).
    The write is performed via ``tempfile.NamedTemporaryFile`` +
    ``os.replace`` so a half-written ``config.json`` is never visible
    to a peer reader and last-writer-wins semantics hold without data
    corruption.
    """
    out_dir = (cfg.output_dir / cfg.experiment_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    payload = cfg.model_dump_json(indent=2)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out_dir,
            prefix=".config.",
            suffix=".json.tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, config_path)
        tmp_path = None
    finally:
        # On any failure between tempfile creation and replace, the
        # ``.config.*.tmp`` entry would otherwise accumulate in the
        # output directory across retries.
        if tmp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
    logger.info(
        "Resolved config dumped to {} (experiment_id={})",
        config_path,
        cfg.experiment_id,
    )
    return config_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code so the function is
    usable from both a script entry and from in-process integration
    tests that want to assert exit semantics."""
    if argv is None:
        argv = sys.argv[1:]

    experiment_path, dry_run, help_requested, override_args = _parse_initial_args(
        argv
    )

    if help_requested and experiment_path is None:
        _build_initial_parser().print_help()
        return 0

    if experiment_path is None:
        _build_initial_parser().print_usage(sys.stderr)
        print(
            "gigaevo: error: the following arguments are required: experiment",
            file=sys.stderr,
        )
        return 2

    load_dotenv()

    try:
        baseline = build_experiment(experiment_path)
    except ValidationError as exc:
        # The experiment module itself produced a config that fails
        # Pydantic validation. Surface a friendly framed block instead
        # of the raw stack trace; the operator wants to know which CLI
        # flag would fix it, not how the Python interpreter walked here.
        print(_format_validation_error(exc), file=sys.stderr)
        return 2

    _warn_on_repeated_flags(override_args)

    if help_requested:
        _build_initial_parser().print_help()
        print()
        import tyro

        # Forward the full override list so a discriminated-union choice
        # (e.g. ``--llm.kind heterogeneous``) reshapes the schema *before*
        # tyro materialises the field tree. Without the forward, the
        # printed help shows only the baseline-discriminator's fields and
        # the operator can't see what flags the chosen variant exposes.
        try:
            tyro.cli(
                ExperimentConfig,
                default=baseline,
                args=[*override_args, "--help"],
                prog="gigaevo overrides",
            )
        except SystemExit as exit_exc:
            # tyro raises SystemExit(0) after printing --help; preserve
            # the exit code so wrapper scripts read a successful help.
            return int(exit_exc.code or 0)
        return 0

    try:
        cfg = _apply_tyro_overrides(baseline, override_args)
    except ValidationError as exc:
        print(_format_validation_error(exc), file=sys.stderr)
        return 2

    config_path = _dump_resolved_config(cfg)

    if dry_run:
        logger.info(
            "Dry run complete. Validated config at {}. Engine invocation skipped.",
            config_path,
        )
        return 0

    from gigaevo.config.object_graph import run_with_config

    return asyncio.run(run_with_config(cfg))


if __name__ == "__main__":
    sys.exit(main())
