# src/jaik/_jax/fk.py
import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype
from .utils import _rot


@jaxtyped(typechecker=beartype)
def _fk(
    q: Float[Array, "6"],
    H: Float[Array, "3 6"],
    P: Float[Array, "3 7"],
) -> tuple[Float[Array, "3 3"], Float[Array, "3"]]:
    """
    Forward kinematics via Product of Exponentials.
    Returns joint-frame rotation and position (no tool frame applied).

    Args:
        q: (6,) joint angles
        H: (3,6) joint axes
        P: (3,7) link displacements

    Returns:
        R: (3,3) rotation matrix of joint 6 frame in base frame
        p: (3,)  position of joint 6 frame origin in base frame
    """
    R = jnp.eye(3)
    p = P[:, 0]
    for i in range(6):
        R = R @ _rot(H[:, i], q[i])
        p = p + R @ P[:, i + 1]
    return R, p