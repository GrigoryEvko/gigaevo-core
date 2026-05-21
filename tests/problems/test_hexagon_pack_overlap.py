"""Regression: hexagon_pack helper must admit edge-touching and corner-
touching hexagons. Closed tessellations and dense packings depend on
adjacent hexagons sharing an entire edge or a single vertex; the
overlap check has to treat those contacts as non-overlapping.

Loaded by file path because ``problems/hexagon_pack/helper.py`` lives
outside the ``gigaevo`` package and the ``tests/problems/__init__.py``
shadow blocks normal ``problems.*`` imports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def _load_helper() -> ModuleType:
    name = "_hexagon_pack_helper_under_test"
    if name in sys.modules:
        return sys.modules[name]
    path = _ROOT / "problems" / "hexagon_pack" / "helper.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helper = _load_helper()


def test_disjoint_hexagons_do_not_overlap() -> None:
    """Two far-apart hexagons must not be flagged as overlapping."""
    assert (
        helper.check_hexagon_overlap_two(
            np.array([0.0, 0.0]),
            0.0,
            np.array([10.0, 0.0]),
            0.0,
        )
        is False
    )


def test_concentric_hexagons_do_overlap() -> None:
    """Two coincident hexagons must be flagged as overlapping."""
    assert (
        helper.check_hexagon_overlap_two(
            np.array([0.0, 0.0]),
            0.0,
            np.array([0.0, 0.0]),
            0.0,
        )
        is True
    )


def test_edge_sharing_tessellation_is_not_overlap() -> None:
    """In a regular hex tessellation, adjacent hexagons share an entire
    edge. With the helper's pointy-right vertex convention, the
    edge-sharing direction is perpendicular to an edge — angles 30°,
    90°, 150° from horizontal. Place the partner at the 30° apothem
    direction so the two hexagons share exactly one edge."""
    side = 1.0
    center_distance = np.sqrt(3.0) * side
    theta = np.pi / 6.0
    partner = np.array([center_distance * np.cos(theta), center_distance * np.sin(theta)])
    assert (
        helper.check_hexagon_overlap_two(
            np.array([0.0, 0.0]),
            0.0,
            partner,
            0.0,
        )
        is False
    )


def test_corner_sharing_hexagons_are_not_overlap() -> None:
    """Two hexagons that meet at a single vertex must not be flagged as
    overlapping. Pre-fix, the strict-intersection branch fired with one
    orientation = 0 and falsely returned True."""
    # Hex A vertices at angles k*60°. Vertex at angle 0 is (1, 0).
    # Place hex B so its angle-180° vertex (-1, 0) lands on A's (1, 0):
    # B's center then sits at (2, 0). At that distance the bodies clear
    # one another and share only the corner (1, 0).
    assert (
        helper.check_hexagon_overlap_two(
            np.array([0.0, 0.0]),
            0.0,
            np.array([2.0, 0.0]),
            0.0,
        )
        is False
    )


def test_offset_hexagons_have_genuine_overlap() -> None:
    """Two hexagons sharing more than a corner / edge — i.e. real
    interior overlap — must still be flagged as overlapping."""
    # Center distance < 2 * apothem means interiors overlap.
    apothem = np.sqrt(3.0) / 2.0
    overlap_distance = 2.0 * apothem - 0.1
    assert (
        helper.check_hexagon_overlap_two(
            np.array([0.0, 0.0]),
            0.0,
            np.array([overlap_distance, 0.0]),
            0.0,
        )
        is True
    )


def test_many_hex_tessellation_seven_cluster_not_overlap() -> None:
    """A 7-hexagon flower-of-life cluster (1 center + 6 ring) is the
    canonical dense packing that the overlap check must admit."""
    side = 1.0
    center_distance = np.sqrt(3.0) * side
    centers = [np.array([0.0, 0.0])]
    angles = [0.0]
    for k in range(6):
        theta = k * np.pi / 3.0 + np.pi / 6.0
        centers.append(
            np.array([center_distance * np.cos(theta), center_distance * np.sin(theta)])
        )
        angles.append(0.0)
    centers_arr = np.stack(centers)
    angles_arr = np.array(angles)
    assert helper.check_hexagon_overlap_many(centers_arr, angles_arr) is False
