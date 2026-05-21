"""Regression tests for :func:`_resolve_exit_code` in ``object_graph``.

The CLI exits non-zero when the run finished with silent-failure
counters set on either the runner or the engine *and* the strategy
archive failed to grow. A run that grew the archive returns 0 even
if some transient batch transitions misfired."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from gigaevo.config.object_graph import _resolve_exit_code
from gigaevo.runner.dag_runner import DagRunnerMetrics


def _fake_runner(*, batch_fail: int = 0, not_found: int = 0):
    m = DagRunnerMetrics()
    if batch_fail:
        m.record_batch_transition_failure(batch_fail)
    for _ in range(not_found):
        m.record_program_not_found()
    return SimpleNamespace(_metrics=m)


def _fake_engine(*, batch_fail: int = 0):
    metrics = MagicMock()
    metrics.batch_transition_failures = batch_fail
    return SimpleNamespace(metrics=metrics)


class TestResolveExitCode:
    def test_clean_run_returns_zero(self):
        assert (
            _resolve_exit_code(
                _fake_runner(),
                _fake_engine(),
                archive_size_before=0,
                archive_size_after=5,
            )
            == 0
        )

    def test_silent_runner_failure_with_no_growth_returns_two(self):
        assert (
            _resolve_exit_code(
                _fake_runner(batch_fail=2),
                _fake_engine(),
                archive_size_before=10,
                archive_size_after=10,
            )
            == 2
        )

    def test_silent_engine_failure_with_no_growth_returns_two(self):
        assert (
            _resolve_exit_code(
                _fake_runner(),
                _fake_engine(batch_fail=1),
                archive_size_before=10,
                archive_size_after=10,
            )
            == 2
        )

    def test_silent_failure_but_archive_grew_returns_zero(self):
        """Forward progress wins: a transient batch failure that did not
        cost the run its archive growth is not worth a non-zero exit."""
        assert (
            _resolve_exit_code(
                _fake_runner(batch_fail=1),
                _fake_engine(),
                archive_size_before=5,
                archive_size_after=20,
            )
            == 0
        )

    def test_program_not_found_counts_as_silent_failure(self):
        assert (
            _resolve_exit_code(
                _fake_runner(not_found=1),
                _fake_engine(),
                archive_size_before=3,
                archive_size_after=3,
            )
            == 2
        )

    def test_none_objects_safe(self):
        """Construction can fail before either object materialises; the
        resolver must not crash in that path."""
        assert _resolve_exit_code(None, None, 0, 0) == 0
