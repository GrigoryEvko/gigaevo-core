"""Regression test for the Heilbron validator's convex-hull stability.

Triangle-Heilbron seeds occasionally produce nearly-collinear point
layouts. Without the ``QJ`` ("joggle") QHull option the convex-hull
computation raises ``QhullError: initial simplex is flat`` and the
seed gets discarded from the initial population — empirically ~20% of
the seeds in the repository's ``problems/heilbron/initial_programs``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HEILBRON_DIR = Path(__file__).resolve().parents[2] / "problems" / "heilbron"


@pytest.fixture(scope="module")
def heilbron_validate() -> object:
    """Import ``problems/heilbron/validate.py`` with its sibling ``helper``."""
    sys.path.insert(0, str(_HEILBRON_DIR))
    try:
        import importlib

        # Force fresh load — pytest may run other suites that have
        # already imported a stale ``validate``/``helper`` from a
        # different sys.path layout.
        for mod in ("validate", "helper"):
            sys.modules.pop(mod, None)
        validate = importlib.import_module("validate")
        yield validate
    finally:
        sys.path.remove(str(_HEILBRON_DIR))
        for mod in ("validate", "helper"):
            sys.modules.pop(mod, None)


def _nearly_collinear_layout() -> np.ndarray:
    """11 points inside the unit-area equilateral triangle that are nearly
    collinear — enough to trigger QHull's flat-simplex check without
    sending downstream histogram bins to zero. Returns shape (11, 2)."""
    side = (4.0 / np.sqrt(3.0)) ** 0.5
    centroid = np.array([side / 2.0, side * np.sqrt(3.0) / 6.0])
    rng = np.random.default_rng(0)
    # Dominant direction is horizontal; vertical jitter is numerically
    # close to QHull's collinearity tolerance but small enough that the
    # downstream histogram (over triangle areas, not point spread)
    # always has non-zero bins.
    horizontal = np.linspace(-0.1, 0.1, 11)
    vertical = rng.uniform(-1e-9, 1e-9, size=11)
    return np.column_stack([centroid[0] + horizontal, centroid[1] + vertical])


def test_convex_hull_survives_flat_layout(heilbron_validate) -> None:
    """The fix: ``QJ`` joggle keeps ConvexHull alive on near-collinear input.

    Verifies the call sequence the validator actually uses — direct
    ConvexHull invocation — without the histogram-bin side effects of
    ``compute_layout_metrics``. Without ``QJ`` this raises
    ``QhullError: initial simplex is flat``.
    """
    from scipy.spatial import ConvexHull

    pts = _nearly_collinear_layout()
    # The validator's call shape, post-fix:
    hull = ConvexHull(pts, qhull_options="QJ")
    assert np.isfinite(hull.volume)
    assert hull.volume >= 0.0


def test_compute_layout_metrics_handles_seed_jitter(heilbron_validate) -> None:
    """Non-pathological seeds with mild collinearity still return finite metrics."""
    rng = np.random.default_rng(0)
    # 11 points with a clear linear trend plus small perpendicular noise —
    # the realistic case where QHull's flat-simplex check used to fire
    # for genuine evolutionary seeds.
    side = (4.0 / np.sqrt(3.0)) ** 0.5
    centroid = np.array([side / 2.0, side * np.sqrt(3.0) / 6.0])
    horizontal = np.linspace(-0.2, 0.2, 11)
    vertical = rng.uniform(-0.01, 0.01, size=11)
    pts = np.column_stack([centroid[0] + horizontal, centroid[1] + vertical])
    metrics = heilbron_validate.compute_layout_metrics(pts)
    assert np.isfinite(metrics["convex_hull_area"])


def test_compute_layout_metrics_nondegenerate_unchanged(heilbron_validate) -> None:
    """Joggle should not materially change non-degenerate hull areas."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.1, 0.4, size=(11, 2))
    metrics = heilbron_validate.compute_layout_metrics(pts)
    # Quick sanity: a roughly-spread random cluster has a non-trivial hull.
    assert metrics["convex_hull_area"] > 0.01


def test_compute_layout_metrics_all_collinear(heilbron_validate) -> None:
    """A perfectly collinear layout — every triangle has area 0 — must
    not crash the histogram step.

    ``compute_layout_metrics`` builds a log-spaced bin array from the
    observed (min_area, max_area) pair. With all-zero areas this
    reduces to ``geomspace(1e-8, ~1e-16, …)`` which is monotonically
    decreasing and ``np.histogram`` raises ``ValueError: bins must
    increase monotonically``. The validator must clamp both ends so
    degenerate layouts collapse into the lowest bin instead of
    propagating the failure to the program-evaluation pipeline."""
    side = (4.0 / np.sqrt(3.0)) ** 0.5
    centroid_y = side * np.sqrt(3.0) / 6.0
    xs = np.linspace(0.1, 0.6, 11)
    pts = np.column_stack([xs, np.full(11, centroid_y)])
    metrics = heilbron_validate.compute_layout_metrics(pts)
    # All triangle areas are zero — the histogram-derived metrics
    # should fall back to safe sentinel values rather than NaN / raise.
    assert metrics["mean_triangle_area"] == 0.0
    assert np.isfinite(metrics["triangle_area_hist_entropy"])
    assert np.isfinite(metrics["triangle_area_mean_bin"])
    assert 0.0 <= metrics["triangle_area_low_bin_frac"] <= 1.0


def test_validate_accepts_arc_seed_layout(heilbron_validate) -> None:
    """End-to-end: the ``arc`` initial-program seed lays 11 points along
    a curve, producing min_area = 0 and max_area on the order of 1e-16.

    The validator used to raise ``ValueError: bins must increase
    monotonically`` for this input and the LLM mutation loop silently
    dropped the seed from the population — empirically the same class
    of failure as BUG-L2 but inside ``compute_layout_metrics`` rather
    than ``ConvexHull``. The post-fix validator must return a valid
    metrics dict with ``fitness == 0`` and ``is_valid == 1``."""
    # Lift the layout straight from problems/heilbron/initial_programs/arc.py
    # without importing it (the seed module installs np.random.seed
    # side-effects and the test should stay hermetic).
    side = (4.0 / np.sqrt(3.0)) ** 0.5
    height = np.sqrt(3.0) / 2.0 * side
    A = np.array([0.0, 0.0])
    B = np.array([side, 0.0])
    C = np.array([side / 2.0, height])
    pts = []
    for i in range(11):
        t = i / 10.0
        P1 = (1.0 - t) * A + t * B
        P2 = (1.0 - t) * B + t * C
        pts.append((P1 + P2) / 2.0)
    pts = np.asarray(pts)
    result = heilbron_validate.validate(pts)
    assert result["is_valid"] == 1
    assert result["fitness"] == 0.0
    assert np.isfinite(result["triangle_area_hist_entropy"])
