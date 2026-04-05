import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jaik


_MODELS = jaik.available_robots()


@pytest.fixture(scope="module", params=_MODELS)
def compare(request):
    name = request.param
    # fk_np, ik_full_np, ik_closest_np = jaik.make_robot(name, solver="numpy")
    # fk_jax, ik_full_jax, ik_closest_jax = jaik.make_robot(name, solver="jax")
    # fk_jax_sc, ik_full_jax_sc, ik_closest_jax_sc = jaik.make_robot(name, solver="jax_sincos")


def compare_fk(name, seed):
    fk_np, _, _ = jaik.make_robot(name, solver="numpy")
    fk_jax, _, _ = jaik.make_robot(name, solver="jax")
    fk_jax_sc, _, _ = jaik.make_robot(name, solver="jax_sincos")
    q_np = np.random.rand(seed) * 2*np.pi
    q_jax = jnp.array(q_np)