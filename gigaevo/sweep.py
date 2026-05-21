"""Subprocess-based sweep runner.

Spawns one ``python run.py <experiment> <overrides...>`` process per
combination produced by a sweep-definition module. Each subprocess
gets a fresh Python interpreter, so module-level state never leaks
between runs.

A sweep file is a Python module exporting ``define_sweep() ->
list[list[str]]``: each inner list is the argv slice to forward to
``run.py`` for one run. See ``sweeps/`` for shipped examples.

Invocation::

    python -m gigaevo.sweep experiments/base.py sweeps/seeds.py
    python -m gigaevo.sweep experiments/base.py sweeps/seeds.py --parallel 4

Cells run with auto-injected per-cell ``--name`` and
``--dataplane.key-prefix`` overrides so each cell takes a distinct
Redis namespace; the experiment's ``_key_prefix_follows_convention``
validator otherwise rejects every cell after the first when only
non-namespace fields vary across the sweep. Users that need a
custom per-cell name can override ``--name`` (and the matching
``--dataplane.key-prefix``) in their sweep entry; explicit
user-supplied values take precedence over the injected pair.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

_NAME_FLAGS: frozenset[str] = frozenset({"--name"})
_PREFIX_FLAGS: frozenset[str] = frozenset(
    {"--dataplane.key-prefix", "--dataplane.key_prefix"}
)


def _has_flag(overrides: list[str], flags: frozenset[str]) -> bool:
    """Return ``True`` iff one of ``flags`` appears as a token in ``overrides``.

    Bare-token comparison (not ``startswith``) so a user-supplied
    ``--name-suffix`` does not mask the canonical ``--name`` injection.
    """
    return any(token in flags for token in overrides)


def _load_experiment_name(path: Path) -> str | None:
    """Return the experiment's ``name`` field by calling ``build()``.

    Failures fall through to ``None`` so the sweep still runs (with the
    cell-collision risk) when the experiment file cannot be imported
    in the parent process — the per-cell subprocess will surface the
    real error.
    """
    try:
        from gigaevo.config.experiment_loader import build_experiment
    except Exception:
        return None
    try:
        cfg = build_experiment(path)
    except Exception:
        return None
    name = getattr(cfg, "name", None)
    return str(name) if isinstance(name, str) and name else None


def _inject_namespace_overrides(
    overrides: list[str], base_name: str, cell_idx: int
) -> list[str]:
    """Prepend per-cell ``--name`` and ``--dataplane.key-prefix`` overrides.

    No-op when the user already pinned either flag. Tyro resolves a
    repeated flag by last-wins, so prepending keeps an explicit user
    choice authoritative even when only one of the pair is provided.
    """
    cell_name = f"{base_name}_{cell_idx}"
    injected: list[str] = []
    if not _has_flag(overrides, _NAME_FLAGS):
        injected.extend(["--name", cell_name])
    if not _has_flag(overrides, _PREFIX_FLAGS):
        injected.extend(["--dataplane.key-prefix", f"gigaevo:{cell_name}"])
    return [*injected, *overrides]


class SweepLoadError(Exception):
    """Raised when the user-supplied sweep file cannot be resolved into a
    list of override-argv slices. Distinct from generic ``ImportError``
    so callers can surface a single, actionable message."""


def _run_one(args: tuple[Path, list[str]]) -> int:
    experiment, overrides = args
    cmd = [sys.executable, str(REPO_ROOT / "run.py"), str(experiment), *overrides]
    try:
        return subprocess.run(cmd).returncode
    except OSError as exc:
        # A single failed spawn (E2BIG, EMFILE, ENOMEM, ...) must not
        # abort sibling runs in the parallel path: ProcessPoolExecutor
        # re-raises the worker exception out of pool.map and tears the
        # whole executor down. Convert to a non-zero return code so the
        # outer loop counts it as a failure and continues.
        print(
            f"Sweep: failed to spawn run for {experiment} {overrides}: {exc}",
            file=sys.stderr,
        )
        return 1


def _load_sweep(path: Path) -> list[list[str]]:
    resolved = path.expanduser()
    if not resolved.exists():
        raise SweepLoadError(f"sweep file not found: {resolved}")
    if not resolved.is_file():
        raise SweepLoadError(f"sweep path is not a file: {resolved}")
    if resolved.suffix != ".py":
        raise SweepLoadError(
            f"sweep file must have a .py extension: {resolved}"
        )

    spec = importlib.util.spec_from_file_location("_gigaevo_sweep", resolved)
    if spec is None or spec.loader is None:
        raise SweepLoadError(f"cannot import sweep module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SweepLoadError(
            f"sweep module {resolved} raised during import: {exc}"
        ) from exc
    if not hasattr(module, "define_sweep"):
        raise SweepLoadError(
            f"{resolved} must export define_sweep() -> list[list[str]]"
        )
    runs = module.define_sweep()
    if not isinstance(runs, list) or not all(
        isinstance(r, list) and all(isinstance(x, str) for x in r) for r in runs
    ):
        raise SweepLoadError(
            f"{resolved}.define_sweep() must return list[list[str]]; got "
            f"{type(runs).__name__}"
        )
    return runs


def _validate_experiment(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"experiment file not found: {resolved}")
    if not resolved.is_file():
        raise IsADirectoryError(f"experiment path is not a file: {resolved}")
    if resolved.suffix != ".py":
        raise ValueError(
            f"experiment file must have a .py extension: {resolved}"
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gigaevo.sweep",
        description=(
            "Run an experiment over a parameter sweep, "
            "one subprocess per combination."
        ),
    )
    parser.add_argument(
        "experiment",
        type=Path,
        help="Path to the experiment file (exports build() -> ExperimentConfig)",
    )
    parser.add_argument(
        "sweep",
        type=Path,
        help="Path to a Python module exporting define_sweep() -> list[list[str]]",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum concurrent subprocesses (default: 1, sequential)",
    )
    parsed = parser.parse_args(argv)

    if parsed.parallel < 1:
        print(
            f"--parallel must be >= 1, got {parsed.parallel}",
            file=sys.stderr,
        )
        return 2

    try:
        experiment = _validate_experiment(parsed.experiment)
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        runs = _load_sweep(parsed.sweep)
    except SweepLoadError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    total = len(runs)
    if total == 0:
        print("Sweep finished: 0/0 runs OK (empty sweep)")
        return 0

    base_name = _load_experiment_name(experiment)
    if base_name is None:
        # The experiment file failed to import in the parent process.
        # Skip per-cell namespace injection and surface the real failure
        # on the first subprocess; users that don't trip the cross-field
        # validator (e.g. by overriding ``--name`` per cell) still run.
        work = [(experiment, ovs) for ovs in runs]
    else:
        work = [
            (experiment, _inject_namespace_overrides(ovs, base_name, idx))
            for idx, ovs in enumerate(runs)
        ]

    try:
        if parsed.parallel <= 1:
            results = [_run_one(item) for item in work]
        else:
            with ProcessPoolExecutor(max_workers=parsed.parallel) as pool:
                results = list(pool.map(_run_one, work))
    except KeyboardInterrupt:
        print("Sweep aborted by SIGINT", file=sys.stderr)
        return 130

    failures = sum(1 for rc in results if rc != 0)
    if failures:
        print(
            f"Sweep finished: {failures}/{total} runs failed",
            file=sys.stderr,
        )
        return 1
    print(f"Sweep finished: {total}/{total} runs OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
