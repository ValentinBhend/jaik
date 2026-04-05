import jax
import jax.numpy as jnp
from jaxtyping import Float, Bool, Array, jaxtyped
from beartype import beartype


# @jaxtyped(typechecker=beartype)
def sp1(
    p1: Float[Array, "3"],
    p2: Float[Array, "3"],
    k:  Float[Array, "3"],
) -> tuple[Float[Array, ""], Bool[Array, ""]]:
    KxP = jnp.cross(k, p1)
    x0 = jnp.dot(KxP, p2)
    x1 = jnp.dot(-jnp.cross(k, KxP), p2)
    theta = jnp.atan2(x0, x1)
    is_LS = (jnp.abs(jnp.linalg.norm(p1) - jnp.linalg.norm(p2)) > 1e-8) | \
            (jnp.abs(jnp.dot(k, p1) - jnp.dot(k, p2)) > 1e-8)
    return theta, is_LS


# @jaxtyped(typechecker=beartype)
def sp3(
    p1: Float[Array, "3"],
    p2: Float[Array, "3"],
    k:  Float[Array, "3"],
    d:  Float[Array, ""],
) -> tuple[Float[Array, "2"], Bool[Array, ""]]:
    KxP = jnp.cross(k, p1)
    A1 = jnp.stack((KxP, -jnp.cross(k, KxP)))   # (2, 3)
    A = -2 * p2 @ A1.T                             # (2,)
    norm_A_sq = jnp.dot(A, A)
    norm_A = jnp.sqrt(norm_A_sq)
    b = d**2 - jnp.linalg.norm(p2 - jnp.dot(k, p1) * k)**2 - jnp.linalg.norm(KxP)**2
    x_ls = A1 @ (-2 * p2 * b / norm_A_sq)         # (2,)
    is_LS = x_ls @ x_ls > 1

    xi = jnp.sqrt(jnp.clip(1 - b**2 / norm_A_sq, 0.0))
    A_perp = jnp.array([A[1], -A[0]]) / norm_A

    def _ls(_):
        return jnp.array([jnp.atan2(x_ls[0], x_ls[1]), jnp.nan])

    def _exact(_):
        sc_1 = x_ls + xi * A_perp
        sc_2 = x_ls - xi * A_perp
        return jnp.array([jnp.atan2(sc_1[0], sc_1[1]),
                           jnp.atan2(sc_2[0], sc_2[1])])

    theta = jax.lax.cond(is_LS, _ls, _exact, None)
    return theta, is_LS

# @jaxtyped(typechecker=beartype)
def sp4(
    p: Float[Array, "3"],
    k: Float[Array, "3"],
    h: Float[Array, "3"],
    d: Float[Array, ""],
) -> tuple[Float[Array, "2"], Bool[Array, ""]]:
    A11 = jnp.cross(k, p)
    A1 = jnp.stack((A11, -jnp.cross(k, A11)))      # (2, 3)
    A = h @ A1.T                                     # (2,)
    b = d - jnp.dot(h, k) * jnp.dot(k, p)
    norm_A2 = A @ A
    x_ls = A1 @ (h * b)
    is_LS = norm_A2 <= b**2

    xi = jnp.sqrt(jnp.clip(norm_A2 - b**2, 0.0))   # clip before sqrt
    A_perp_tilde = jnp.array([A[1], -A[0]])          # (2,)

    def _ls(_):
        return jnp.array([jnp.atan2(x_ls[0], x_ls[1]), jnp.nan])

    def _exact(_):
        sc_1 = x_ls + xi * A_perp_tilde
        sc_2 = x_ls - xi * A_perp_tilde
        return jnp.array([jnp.atan2(sc_1[0], sc_1[1]),
                           jnp.atan2(sc_2[0], sc_2[1])])

    theta = jax.lax.cond(is_LS, _ls, _exact, None)
    return theta, is_LS