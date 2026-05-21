"""Tests for ``ProblemConfig.build()`` disk-shape validation.

The schema layer accepts any ``Path`` so dumped configs can round-trip
without the original problem layout mounted. At ``build()`` time the
runtime context requires the directory to exist and to carry a
``metrics.yaml``; the typed error here turns a deep-down ``MetricsContext``
crash into an actionable failure at runtime construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gigaevo.config.schemas import ProblemConfig


class TestProblemBuildValidation:
    def test_nonexistent_path_rejected(self, tmp_path: Path) -> None:
        cfg = ProblemConfig(
            name="ghost", problem_dir=tmp_path / "does-not-exist"
        )
        with pytest.raises(ValueError, match="not a directory"):
            cfg.build()

    def test_regular_file_rejected(self, tmp_path: Path) -> None:
        """``/etc/passwd``-style hostile path; the directory check is
        the credential-leak boundary."""
        f = tmp_path / "regular_file"
        f.write_text("not a directory")
        cfg = ProblemConfig(name="ghost", problem_dir=f)
        with pytest.raises(ValueError, match="not a directory"):
            cfg.build()

    def test_directory_without_metrics_rejected(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_problem"
        empty.mkdir()
        cfg = ProblemConfig(name="ghost", problem_dir=empty)
        with pytest.raises(ValueError, match="metrics.yaml"):
            cfg.build()

    def test_directory_with_metrics_accepted(self, tmp_path: Path) -> None:
        layout = tmp_path / "ok_problem"
        layout.mkdir()
        (layout / "metrics.yaml").write_text(
            "primary_metric:\n"
            "  description: x\n"
            "  is_primary: true\n"
            "  higher_is_better: true\n"
            "  lower_bound: 0\n"
            "  upper_bound: 1\n"
            "is_valid:\n"
            "  description: y\n"
            "  is_primary: false\n"
            "  higher_is_better: true\n"
            "  lower_bound: 0\n"
            "  upper_bound: 1\n"
        )
        cfg = ProblemConfig(name="ok", problem_dir=layout)
        # The build call asserts the disk shape; the returned context
        # exposes the resolved path for downstream callers.
        ctx = cfg.build()
        assert ctx.problem_dir == layout.resolve()
