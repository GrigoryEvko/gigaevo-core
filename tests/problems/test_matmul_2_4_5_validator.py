"""Regression tests for the matrix-multiplication (2,4,5) validator.

Covers two contract repairs:

* The validator no longer requires bit-exact equality with the target
  tensor — sub-ULP einsum rounding had previously rejected algebraically
  correct decompositions. Tolerance is now ``_TENSOR_ATOL = 1e-6`` as
  the problem description advertises.
* The validator bounds ``rank`` at the trivial decomposition bound
  ``n * m * p = 40`` so adversarial inputs cannot inflate the einsum
  cost arbitrarily.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_validate() -> ModuleType:
    name = "_matmul_2_4_5_validate_under_test"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        _ROOT
        / "problems"
        / "alphaevolve"
        / "matrix_multiplication"
        / "2_4_5"
        / "validate.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_mod = _load_validate()
validate = _mod.validate


def _trivial_decomposition() -> dict:
    """The textbook rank-40 outer-product decomposition for the (2,4,5)
    matmul tensor. Every triple ``(i, j, k)`` contributes one rank-1
    term whose ``u`` / ``v`` / ``w`` are one-hot at the appropriate
    flattened index."""
    n, m, p = 2, 4, 5
    rank = n * m * p
    u = np.zeros((rank, n * m), dtype=float)
    v = np.zeros((rank, m * p), dtype=float)
    w = np.zeros((rank, n * p), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u[r, i * m + j] = 1.0
                v[r, j * p + k] = 1.0
                w[r, k * n + i] = 1.0
                r += 1
    return {
        "rank": rank,
        "u_vectors": u.tolist(),
        "v_vectors": v.tolist(),
        "w_vectors": w.tolist(),
    }


def test_trivial_decomposition_validates() -> None:
    """Sanity: the textbook construction must validate at the trivial bound."""
    result = validate(_trivial_decomposition())
    assert result["is_valid"] == 1
    # fitness = BENCHMARK / rank = 32 / 40 = 0.8
    assert result["fitness"] == pytest.approx(0.8)


def test_validator_tolerates_subulp_einsum_noise() -> None:
    """The reconstruction goes through BLAS; perturbing the trivial
    decomposition with rounding-scale noise must still validate. Bit-
    exact equality (``np.array_equal``) would have rejected this."""
    candidate = _trivial_decomposition()
    rng = np.random.default_rng(0)
    candidate["u_vectors"] = (
        np.asarray(candidate["u_vectors"], dtype=float)
        + rng.normal(scale=1e-9, size=np.asarray(candidate["u_vectors"]).shape)
    ).tolist()
    result = validate(candidate)
    assert result["is_valid"] == 1


def test_validator_rejects_macroscopic_tensor_mismatch() -> None:
    """A decomposition whose reconstruction differs by ~1 must still
    fail validation — the tolerance widening must not admit truly
    wrong decompositions."""
    candidate = _trivial_decomposition()
    u = np.asarray(candidate["u_vectors"], dtype=float)
    u[0, 0] = 5.0
    candidate["u_vectors"] = u.tolist()
    with pytest.raises(ValueError, match="does not match the target tensor"):
        validate(candidate)


def test_validator_rejects_rank_above_trivial_bound() -> None:
    """Rank above ``n * m * p`` is structurally redundant and must be
    rejected so adversarial inputs cannot inflate the validator's
    einsum cost."""
    n, m, p = 2, 4, 5
    over_rank = n * m * p + 1
    candidate = {
        "rank": over_rank,
        "u_vectors": np.zeros((over_rank, n * m), dtype=float).tolist(),
        "v_vectors": np.zeros((over_rank, m * p), dtype=float).tolist(),
        "w_vectors": np.zeros((over_rank, n * p), dtype=float).tolist(),
    }
    with pytest.raises(ValueError, match="exceeds trivial upper bound"):
        validate(candidate)


def test_validator_rejects_zero_or_negative_rank() -> None:
    """Pre-existing positive-integer guard must still fire."""
    candidate = _trivial_decomposition()
    candidate["rank"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        validate(candidate)
