# src/jaik/_jax/utils.py
import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype


# @jaxtyped(typechecker=beartype)
def _rot(
    k: Float[Array, "3"],
    theta: Float[Array, ""],
) -> Float[Array, "3 3"]:
    """
    Rodrigues rotation matrix: rotate by theta around unit axis k.

    R = I + sin(theta) [k]× + (1 - cos(theta)) [k]×²
    """
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    kx, ky, kz = k[0], k[1], k[2]

    return jnp.array([
        [ct + kx*kx*(1-ct),      kx*ky*(1-ct) - kz*st,  kx*kz*(1-ct) + ky*st],
        [ky*kx*(1-ct) + kz*st,   ct + ky*ky*(1-ct),      ky*kz*(1-ct) - kx*st],
        [kz*kx*(1-ct) - ky*st,   kz*ky*(1-ct) + kx*st,   ct + kz*kz*(1-ct)  ],
    ])