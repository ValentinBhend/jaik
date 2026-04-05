# src/jaik/_jax/fk.py
import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype
from .utils import _rot


@jaxtyped(typechecker=beartype)
def _fk_sincos_Rp(
    sq: Float[Array, "6"],
    cq: Float[Array, "6"],
    H: Float[Array, "3 6"],
    P: Float[Array, "3 7"],
) -> tuple[Float[Array, "3 3"], Float[Array, "3"]]:
    """
    Forward kinematics via Product of Exponentials.
    Returns joint-frame rotation and position (no tool frame applied).

    Args:
        sq: (6,) sin joint angles
        cq: (6,) cos joint angles
        H: (3,6) joint axes
        P: (3,7) link displacements

    Returns:
        R: (3,3) rotation matrix of joint 6 frame in base frame
        p: (3,)  position of joint 6 frame origin in base frame
    """
    R = jnp.eye(3)
    p = P[:, 0]
    for i in range(6):
        R = R @ _rot(H[:, i], sq[i], cq[i])
        p = p + R @ P[:, i + 1]
    return R, p

@jaxtyped(typechecker=beartype)
def _fk_Rp(
    q: Float[Array, "6"],
    H: Float[Array, "3 6"],
    P: Float[Array, "3 7"],
) -> tuple[Float[Array, "3 3"], Float[Array, "3"]]:
    sq = jnp.sin(q)
    cq = jnp.cos(q)
    return _fk_sincos_Rp(sq, cq, H, P)

@jaxtyped(typechecker=beartype)
def _fk_sincos_T(
    sq: Float[Array, "6"],
    cq: Float[Array, "6"],
    H: Float[Array, "3 6"],
    P: Float[Array, "3 7"],
) -> tuple[Float[Array, "3 3"], Float[Array, "3"]]:
    R,p = _fk_sincos_Rp(sq, cq, H, P)
    T = jnp.eye((4,4))
    T[:3,:3] = R
    T[:3,3] = p
    return T

@jaxtyped(typechecker=beartype)
def _fk_T(
    q: Float[Array, "6"],
    H: Float[Array, "3 6"],
    P: Float[Array, "3 7"],
) -> tuple[Float[Array, "3 3"], Float[Array, "3"]]:
    sq = jnp.sin(q)
    cq = jnp.cos(q)
    return _fk_sincos_T(sq, cq, H, P)