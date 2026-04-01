import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype


@jaxtyped(typechecker=beartype)
def sp1(
    p1: Float[Array, "3"],
    p2: Float[Array, "3"],
    k:  Float[Array, "3"],
) -> Float[Array, ""]:
    """
    Subproblem 1: find theta such that R(k, theta) @ p1 = p2.
    Returns NaN if no solution exists.
    """
    KxP = jnp.cross(k, p1)
    x0  = jnp.dot(KxP, p2)
    x1  = jnp.dot(-jnp.cross(k, KxP), p2)
    return jnp.atan2(x0, x1)


@jaxtyped(typechecker=beartype)
def sp3(
    p1: Float[Array, "3"],
    p2: Float[Array, "3"],
    k:  Float[Array, "3"],
    d:  Float[Array, ""],
) -> Float[Array, "2"]:
    """
    Subproblem 3: find theta such that |R(k, theta) @ p1 - p2| = d.
    Returns [theta1, theta2]. NaN in both slots if unreachable.
    """
    KxP      = jnp.cross(k, p1)
    A1       = jnp.stack((KxP, -jnp.cross(k, KxP)))    # (2, 3)
    A        = -2 * p2 @ A1.T                            # (2,)
    norm_A_sq = jnp.dot(A, A)
    norm_A    = jnp.sqrt(norm_A_sq)
    b         = (d**2
                 - jnp.linalg.norm(p2 - jnp.dot(k, p1) * k)**2
                 - jnp.linalg.norm(KxP)**2)
    x_ls  = A1 @ (-2 * p2 * b / norm_A_sq)
    xi    = jnp.sqrt(1 - b**2 / norm_A_sq)              # NaN if unreachable
    A_perp = jnp.array([A[1], -A[0]]) / norm_A
    sc_1  = x_ls + xi * A_perp
    sc_2  = x_ls - xi * A_perp
    return jnp.array([jnp.atan2(sc_1[0], sc_1[1]),
                      jnp.atan2(sc_2[0], sc_2[1])])


@jaxtyped(typechecker=beartype)
def sp4(
    p: Float[Array, "3"],
    k: Float[Array, "3"],
    h: Float[Array, "3"],
    d: Float[Array, ""],
) -> Float[Array, "2"]:
    """
    Subproblem 4: find theta such that h · R(k, theta) · p = d.
    Returns [theta1, theta2]. NaN in both slots if unreachable.
    """
    A11          = jnp.cross(k, p)
    A1           = jnp.stack((A11, -jnp.cross(k, A11)))  # (2, 3)
    A            = h @ A1.T                                # (2,)
    b            = d - jnp.dot(h, k) * jnp.dot(k, p)
    norm_A2      = A @ A
    x_ls         = A1 @ (h * b)
    xi           = jnp.sqrt(norm_A2 - b**2)               # NaN if unreachable
    A_perp_tilde = jnp.array([A[1], -A[0]])
    sc_1         = x_ls + xi * A_perp_tilde
    sc_2         = x_ls - xi * A_perp_tilde
    return jnp.array([jnp.atan2(sc_1[0], sc_1[1]),
                      jnp.atan2(sc_2[0], sc_2[1])])