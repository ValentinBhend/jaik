import pytest
import numpy as np
import jax.numpy as jnp
import jaik

_MODELS = jaik.available_robots()

@pytest.mark.parametrize("name", _MODELS)
def test_compare(name):
    for seed in range(500):
        compare_single(name, seed)

def compare_single(name, seed, threshold=1e-10):
    fk_np, _, _ = jaik.make_robot(name, solver="numpy")
    fk_jax, _, _ = jaik.make_robot(name, solver="jax")
    fk_jax_sc, _, _ = jaik.make_robot(name, solver="jax", sincos=True)

    q_np = np.random.default_rng(seed).random(6) * 2 * np.pi
    q_jax = jnp.array(q_np)
    q_jax_sin, q_jax_cos = jnp.sin(q_jax), jnp.cos(q_jax)

    R_np, p_np = fk_np(q_np)
    R_jax, p_jax = fk_jax(q_jax)
    R_jax_sc, p_jax_sc = fk_jax_sc(q_jax_sin, q_jax_cos)

    assert np.allclose(R_np, R_jax, atol=threshold), \
        f"R mismatch (numpy vs jax) at seed {seed}"
    assert np.allclose(p_np, p_jax, atol=threshold), \
        f"p mismatch (numpy vs jax) at seed {seed}"
    assert np.allclose(R_np, R_jax_sc, atol=threshold), \
        f"R mismatch (numpy vs jax sincos) at seed {seed}"
    assert np.allclose(p_np, p_jax_sc, atol=threshold), \
        f"p mismatch (numpy vs jax sincos) at seed {seed}"