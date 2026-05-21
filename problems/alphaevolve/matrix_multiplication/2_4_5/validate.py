import numpy as np

# Trivial decomposition bound for the (n, m, p) = (2, 4, 5) matmul tensor:
# the standard rank-n*m*p = 40 outer-product expansion always works, so
# any candidate that claims more rank-1 terms than that is strictly
# worse than the textbook construction and reserves no novel signal.
# Bounding the rank also keeps the einsum below at a fixed cost so
# adversarial inputs cannot inflate the validator's wallclock.
_MAX_RANK = 40

# Tensor-equality tolerance. The reference tensor has 0/1 entries and
# the einsum reconstruction goes through BLAS, so bit-exact equality
# (``np.array_equal``) rejects algebraically valid decompositions
# whose only error is sub-ULP rounding. The problem description
# documents a 1e-6 tolerance for matmul tensor reconstruction.
_TENSOR_ATOL = 1e-6


def validate(result):
    if not isinstance(result, dict):
        raise ValueError("Result must be a dictionary")

    required_keys = ["rank", "u_vectors", "v_vectors", "w_vectors"]
    for key in required_keys:
        if key not in result:
            raise ValueError(f"Missing required key: {key}")

    rank = result["rank"]
    u_vectors = np.asarray(result["u_vectors"], dtype=float)
    v_vectors = np.asarray(result["v_vectors"], dtype=float)
    w_vectors = np.asarray(result["w_vectors"], dtype=float)

    n, m, p = 2, 4, 5

    # bool is a subclass of int in Python, so isinstance(True, int) is
    # True. Reject it explicitly: a Python boolean as the rank is a
    # type-confusion signal from the caller and should never reach the
    # tensor reconstruction loop.
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank <= 0:
        raise ValueError(f"Rank must be a positive integer, got {rank!r}")
    if rank > _MAX_RANK:
        raise ValueError(
            f"Rank {rank} exceeds trivial upper bound {_MAX_RANK} "
            f"(n*m*p = {n * m * p}); the textbook decomposition is "
            "already at most that rank, so higher ranks are not novel "
            "and would let adversarial inputs inflate validator cost."
        )

    if u_vectors.ndim != 2 or u_vectors.shape[1] != n * m:
        raise ValueError(
            f"u_vectors must have shape (rank, {n * m}), got {u_vectors.shape}"
        )

    if v_vectors.ndim != 2 or v_vectors.shape[1] != m * p:
        raise ValueError(
            f"v_vectors must have shape (rank, {m * p}), got {v_vectors.shape}"
        )

    if w_vectors.ndim != 2 or w_vectors.shape[1] != n * p:
        raise ValueError(
            f"w_vectors must have shape (rank, {n * p}), got {w_vectors.shape}"
        )

    if (
        u_vectors.shape[0] != rank
        or v_vectors.shape[0] != rank
        or w_vectors.shape[0] != rank
    ):
        raise ValueError(f"All vectors must have first dimension equal to rank {rank}")

    if (
        not np.all(np.isfinite(u_vectors))
        or not np.all(np.isfinite(v_vectors))
        or not np.all(np.isfinite(w_vectors))
    ):
        raise ValueError("All vectors must contain finite values")

    matmul_tensor = np.zeros((n * m, m * p, n * p), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                matmul_tensor[i * m + j, j * p + k, k * n + i] = 1

    u_reshaped = u_vectors.T
    v_reshaped = v_vectors.T
    w_reshaped = w_vectors.T

    constructed_tensor = np.einsum(
        "ir,jr,kr -> ijk", u_reshaped, v_reshaped, w_reshaped
    )

    diff = float(np.max(np.abs(constructed_tensor - matmul_tensor)))
    if diff > _TENSOR_ATOL:
        raise ValueError(
            f"Tensor constructed by decomposition does not match the "
            f"target tensor within atol={_TENSOR_ATOL:g}. "
            f"Maximum difference is {diff:.6e}."
        )

    BENCHMARK = 32
    fitness = BENCHMARK / float(rank)
    is_valid = 1

    return {
        "fitness": fitness,
        "is_valid": is_valid,
    }
