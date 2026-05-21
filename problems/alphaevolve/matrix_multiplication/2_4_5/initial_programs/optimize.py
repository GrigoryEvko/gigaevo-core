from helper import get_matrix_multiplication_tensor
import jax
import jax.numpy as jnp
import numpy as np
import optax

n, m, p = 2, 4, 5


@jax.custom_vjp
def round_to_half_ste(x):
    return jnp.round(x * 2) / 2


def round_ste_fwd(x):
    return round_to_half_ste(x), None


def round_ste_bwd(res, g):
    return (g,)


round_to_half_ste.defvjp(round_ste_fwd, round_ste_bwd)


def weighted_l2_loss(reconstructed: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    error = reconstructed - target
    weights = jnp.where(target != 0, 100.0, 1.0)
    return jnp.mean(weights * (error**2))


def l2_loss_real(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((x - y) ** 2)


def get_constrained_decomposition(
    latent_decomposition: tuple, clamp_range: float
) -> tuple:
    return jax.tree_util.tree_map(
        lambda x: clamp_range * jnp.tanh(x), latent_decomposition
    )


def _make_train_step(optimizer, loss_fn):
    @jax.jit
    def _step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return _step


def _trivial_decomposition(n: int, m: int, p: int) -> dict:
    """Standard rank-n*m*p outer-product expansion of the matmul tensor.

    The (i*m + j, j*p + k, k*n + i) entries are 1; each is a rank-1
    outer product of three one-hot basis vectors. Returns the
    decomposition in the validator's expected shapes (rank, n*m) /
    (rank, m*p) / (rank, n*p) and is guaranteed to exactly reconstruct
    the matmul tensor under any tolerance.
    """
    rank = n * m * p
    u = np.zeros((rank, n * m), dtype=np.float32)
    v = np.zeros((rank, m * p), dtype=np.float32)
    w = np.zeros((rank, n * p), dtype=np.float32)
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
        "u_vectors": u,
        "v_vectors": v,
        "w_vectors": w,
    }


def entrypoint() -> dict:
    """Continuous-optimisation seed that falls back to the trivial
    rank-n*m*p decomposition when the optimiser fails to produce a
    discrete reconstruction within tolerance.

    The continuous + half-integer-STE pipeline almost never lands an
    exact integer/half-integer match under random init, so leaving the
    seed program at its un-rounded output makes every run start with a
    rejected program and the LLM never gets a parent to mutate. The
    fallback guarantees that the seed always returns a valid
    decomposition at the trivial rank, giving the mutator something
    concrete to evolve toward lower ranks.
    """
    rank = 40
    num_restarts = 2
    phase1_steps = 4000
    phase1_lr = 0.01
    init_scale = 0.1
    l1_strength = 1e-6
    clamp_range = 4.0
    phase2_steps = 1000
    phase2_lr = 1e-4
    atol = 1e-6

    target_tensor = get_matrix_multiplication_tensor(n, m, p)
    main_key = jax.random.PRNGKey(42)

    def phase1_loss_fn(latent_decomposition: tuple) -> jnp.ndarray:
        constrained = get_constrained_decomposition(latent_decomposition, clamp_range)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *constrained)
        recon_loss = weighted_l2_loss(reconstructed, target_tensor)
        l1_penalty = sum(jnp.mean(jnp.abs(arr)) for arr in constrained)
        return recon_loss + l1_strength * l1_penalty

    def phase2_loss_fn(continuous_decomposition: tuple) -> jnp.ndarray:
        discrete_decomposition = jax.tree_util.tree_map(
            round_to_half_ste, continuous_decomposition
        )
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *discrete_decomposition)
        return l2_loss_real(reconstructed, target_tensor)

    best_loss_phase1 = float("inf")
    best_latent_decomp = None
    phase1_optimizer = optax.adam(phase1_lr)
    phase1_step = _make_train_step(phase1_optimizer, phase1_loss_fn)

    for i in range(num_restarts):
        main_key, restart_key = jax.random.split(main_key)
        init_fn = jax.nn.initializers.normal(stddev=init_scale)
        latent_decomp = (
            init_fn(restart_key, (n * m, rank)),
            init_fn(restart_key, (m * p, rank)),
            init_fn(restart_key, (n * p, rank)),
        )
        opt_state = phase1_optimizer.init(latent_decomp)

        for _ in range(phase1_steps):
            latent_decomp, opt_state, loss = phase1_step(latent_decomp, opt_state)

        final_loss = l2_loss_real(
            target_tensor,
            jnp.einsum(
                "ir,jr,kr->ijk",
                *get_constrained_decomposition(latent_decomp, clamp_range),
            ),
        )

        if final_loss < best_loss_phase1:
            best_loss_phase1 = final_loss
            best_latent_decomp = latent_decomp

    continuous_params = get_constrained_decomposition(best_latent_decomp, clamp_range)
    phase2_optimizer = optax.adam(phase2_lr)
    opt_state = phase2_optimizer.init(continuous_params)
    phase2_step = _make_train_step(phase2_optimizer, phase2_loss_fn)

    for step in range(phase2_steps):
        continuous_params, opt_state, loss = phase2_step(continuous_params, opt_state)
        if loss < 1e-7:
            break

    final_discrete_decomposition = jax.tree_util.tree_map(
        round_to_half_ste, continuous_params
    )
    final_decomposition_np = jax.tree_util.tree_map(
        np.array, final_discrete_decomposition
    )
    u_reshaped, v_reshaped, w_reshaped = final_decomposition_np

    u_vectors = u_reshaped.T
    v_vectors = v_reshaped.T
    w_vectors = w_reshaped.T

    reconstructed = np.einsum("ir,jr,kr->ijk", u_reshaped, v_reshaped, w_reshaped)
    diff = float(np.max(np.abs(reconstructed - np.array(target_tensor))))
    if diff > atol or not np.all(np.isfinite(u_vectors)) \
        or not np.all(np.isfinite(v_vectors)) \
        or not np.all(np.isfinite(w_vectors)):
        return _trivial_decomposition(n, m, p)

    return {
        "rank": rank,
        "u_vectors": u_vectors,
        "v_vectors": v_vectors,
        "w_vectors": w_vectors,
    }
