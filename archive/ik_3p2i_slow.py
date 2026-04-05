# src/jaik/_jax/ik_3p2i.py
import jax.numpy as jnp
from jaxtyping import Float, Bool, Array, jaxtyped
from beartype import beartype
from .subproblems import sp1, sp3, sp4
from .utils import _rot


@jaxtyped(typechecker=beartype)
def _wrap_to_pi(
    angle: Float[Array, ""],
) -> Float[Array, ""]:
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


@jaxtyped(typechecker=beartype)
def ik_3_parallel_2_intersecting(
    R_06: Float[Array, "3 3"],
    p_0T: Float[Array, "3"],
    H:    Float[Array, "3 6"],
    P:    Float[Array, "3 7"],
) -> tuple[Float[Array, "6 8"], Bool[Array, "6 8"]]:
    """
    IK for robots with h2=h3=h4 (3 parallel) and p_56=0 (2 intersecting).
    e.g. UR family.

    Always returns exactly 8 branches — infeasible branches have NaN joint
    angles. is_LS flags indicate least-squares (approximate) solutions.

    Args:
        R_06:  (3,3) rotation of joint 6 frame in base frame
        p_0T:  (3,)  tool position in base frame
        kin:   dict with H (3,6), P (3,7) — adjusted kin from make_robot

    Returns:
        Q:     (6, 8) joint angles, NaN for infeasible branches
        is_LS: (6, 8) per-joint LS flags
    """

    p_06 = p_0T - P[:, 0] - R_06 @ P[:, 6]
    d1 = H[:, 1] @ P[:, 1:5].sum(axis=1)

    # SP4 for q1 — always returns (2,) angles, single is_LS flag
    theta1s, t1_ls = sp4(p_06, -H[:, 0], H[:, 1], d1)

    def _solve_from_q1(q_1):
        R_01 = _rot(H[:, 0], q_1)

        # SP4 for q5
        d5 = H[:, 1] @ R_01.T @ R_06 @ H[:, 5]
        theta5s, t5_ls = sp4(H[:, 5], H[:, 4], H[:, 1], d5)

        def _solve_from_q5(q_5):
            R_45 = _rot(H[:, 4], q_5)

            # SP1 for theta_14 and q6
            theta_14, t14_ls = sp1(R_45 @ H[:, 5], R_01.T @ R_06 @ H[:, 5], H[:, 1])
            q_6,      q6_ls  = sp1(R_45.T @ H[:, 1], R_06.T @ R_01 @ H[:, 1], -H[:, 5])

            # SP3 for q3
            d_inner = R_01.T @ p_06 - P[:, 1] - _rot(H[:, 1], theta_14) @ P[:, 4]
            d3 = jnp.linalg.norm(d_inner)
            theta3s, t3_ls = sp3(-P[:, 3], P[:, 2], H[:, 1], d3)

            def _solve_from_q3(q_3):
                q_2, q2_ls = sp1(
                    P[:, 2] + _rot(H[:, 1], q_3) @ P[:, 3],
                    d_inner,
                    H[:, 1],
                )
                q_4 = _wrap_to_pi(theta_14 - q_2 - q_3)
                q_i  = jnp.array([q_1, q_2, q_3, q_4, q_5,    q_6])
                ls_i = jnp.array([t1_ls, t5_ls, t14_ls, t3_ls, q2_ls, q6_ls])
                return q_i, ls_i

            # 2 q3 branches — NaN propagates if theta3s[1] is NaN
            q_i0, ls_i0 = _solve_from_q3(theta3s[0])
            q_i1, ls_i1 = _solve_from_q3(theta3s[1])
            return jnp.stack([q_i0, q_i1], axis=1), jnp.stack([ls_i0, ls_i1], axis=1)

        # 2 q5 branches
        Q0, L0 = _solve_from_q5(theta5s[0])
        Q1, L1 = _solve_from_q5(theta5s[1])
        return jnp.concatenate([Q0, Q1], axis=1), jnp.concatenate([L0, L1], axis=1)

    # 2 q1 branches → 8 total solutions
    Qa, La = _solve_from_q1(theta1s[0])
    Qb, Lb = _solve_from_q1(theta1s[1])

    Q     = jnp.concatenate([Qa, Qb], axis=1)   # (6, 8)
    is_LS = jnp.concatenate([La, Lb], axis=1)    # (6, 8)

    return Q, is_LS