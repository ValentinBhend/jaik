# src/jaik/_jax/ik_3p2i.py
import jax.numpy as jnp
from jaxtyping import Float, Bool, Array, jaxtyped
from beartype import beartype
from .subproblems import sp1, sp3, sp4
from .utils import _rot


# @jaxtyped(typechecker=beartype)
def _wrap_to_pi(
    angle: Float[Array, ""],
) -> Float[Array, ""]:
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

def ik_3_parallel_2_intersecting(
    R_06: Float[Array, "3 3"],
    p_0T: Float[Array, "3"],
    H:    Float[Array, "3 6"],
    P:    Float[Array, "3 7"],
) -> tuple[Float[Array, "6 8"], Bool[Array, "6 8"]]:
    p_06 = p_0T - P[:, 0] - R_06 @ P[:, 6]
    d1   = H[:, 1] @ P[:, 1:5].sum(axis=1)
    t1   = sp4(p_06, -H[:, 0], H[:, 1], d1)   # (2,)

    def _branch(q_1, q_5_idx, q_3_idx):
        R_01     = _rot(H[:, 0], q_1)
        d5       = H[:, 1] @ R_01.T @ R_06 @ H[:, 5]
        t5       = sp4(H[:, 5], H[:, 4], H[:, 1], d5)     # (2,)
        q_5      = t5[q_5_idx]
        R_45     = _rot(H[:, 4], q_5)
        theta_14 = sp1(R_45 @ H[:, 5], R_01.T @ R_06 @ H[:, 5], H[:, 1])
        q_6      = sp1(R_45.T @ H[:, 1], R_06.T @ R_01 @ H[:, 1], -H[:, 5])
        d_inner  = R_01.T @ p_06 - P[:, 1] - _rot(H[:, 1], theta_14) @ P[:, 4]
        t3       = sp3(-P[:, 3], P[:, 2], H[:, 1], jnp.linalg.norm(d_inner))  # (2,)
        q_3      = t3[q_3_idx]
        q_2      = sp1(P[:, 2] + _rot(H[:, 1], q_3) @ P[:, 3], d_inner, H[:, 1])
        q_4      = _wrap_to_pi(theta_14 - q_2 - q_3)
        return jnp.array([q_1, q_2, q_3, q_4, q_5, q_6])

    # all 8 branches explicitly — (q1_idx, q5_idx, q3_idx)
    branches = [
        (t1[0], 0, 0),
        (t1[0], 0, 1),
        (t1[0], 1, 0),
        (t1[0], 1, 1),
        (t1[1], 0, 0),
        (t1[1], 0, 1),
        (t1[1], 1, 0),
        (t1[1], 1, 1),
    ]

    Q = jnp.stack([_branch(q_1, i5, i3) for q_1, i5, i3 in branches], axis=1)
    valid = ~jnp.isnan(Q).any(axis=0)
    return Q, valid


# @jaxtyped(typechecker=beartype)
def ik_3_parallel_2_intersecting_nested(
    R_06: Float[Array, "3 3"],
    p_0T: Float[Array, "3"],
    H:    Float[Array, "3 6"],
    P:    Float[Array, "3 7"],
) -> tuple[Float[Array, "6 8"], Bool[Array, "6 8"]]:
    """
    IK for robots with h2=h3=h4 (3 parallel) and p_56=0 (2 intersecting).
    e.g. UR family.

    Always returns exactly 8 branches. NaN joint angles indicate an
    infeasible branch (either geometrically impossible or out of workspace).

    Args:
        R_06:  (3,3) rotation of joint 6 frame in base frame
        p_0T:  (3,)  tool position in base frame
        H:     (3,6) joint axes
        P:     (3,7) link displacements

    Returns:
        Q:     (6, 8) joint angles, NaN for infeasible branches
        is_LS: (6, 8) all False — kept for API compatibility with numpy version
    """
    p_06 = p_0T - P[:, 0] - R_06 @ P[:, 6]
    d1   = H[:, 1] @ P[:, 1:5].sum(axis=1)

    # SP4 for q1 — (2,) angles, NaN if unreachable
    theta1s = sp4(p_06, -H[:, 0], H[:, 1], d1)

    def _solve_from_q1(q_1):
        R_01 = _rot(H[:, 0], q_1)

        tmp1 = R_01.T @ R_06      # should be (3,3)
        tmp2 = tmp1 @ H[:, 5]     # should be (3,)
        d5   = H[:, 1] @ tmp2     # should be ()

        # SP4 for q5
        d5      = H[:, 1] @ R_01.T @ R_06 @ H[:, 5]
        theta5s = sp4(H[:, 5], H[:, 4], H[:, 1], d5)

        def _solve_from_q5(q_5):
            R_45 = _rot(H[:, 4], q_5)

            # SP1 for theta_14 and q6 — scalars
            theta_14 = sp1(R_45 @ H[:, 5], R_01.T @ R_06 @ H[:, 5], H[:, 1])
            q_6      = sp1(R_45.T @ H[:, 1], R_06.T @ R_01 @ H[:, 1], -H[:, 5])

            # SP3 for q3 — (2,) angles
            d_inner = R_01.T @ p_06 - P[:, 1] - _rot(H[:, 1], theta_14) @ P[:, 4]
            d3      = jnp.linalg.norm(d_inner)
            theta3s = sp3(-P[:, 3], P[:, 2], H[:, 1], d3)

            def _solve_from_q3(q_3):
                q_2 = sp1(
                    P[:, 2] + _rot(H[:, 1], q_3) @ P[:, 3],
                    d_inner,
                    H[:, 1],
                )
                q_4 = _wrap_to_pi(theta_14 - q_2 - q_3)
                return jnp.array([q_1, q_2, q_3, q_4, q_5, q_6])

            # 2 q3 branches
            q_i0 = _solve_from_q3(theta3s[0])
            q_i1 = _solve_from_q3(theta3s[1])
            return jnp.stack([q_i0, q_i1], axis=1)   # (6, 2)

        # 2 q5 branches → (6, 4)
        Q0 = _solve_from_q5(theta5s[0])
        Q1 = _solve_from_q5(theta5s[1])
        return jnp.concatenate([Q0, Q1], axis=1)      # (6, 4)

    # 2 q1 branches → (6, 8)
    Qa = _solve_from_q1(theta1s[0])
    Qb = _solve_from_q1(theta1s[1])
    Q  = jnp.concatenate([Qa, Qb], axis=1)            # (6, 8)

    # is_LS is all False — NaN encodes infeasibility in this JAX version
    is_LS = jnp.zeros((6, 8), dtype=bool)

    return Q, is_LS