import numpy as np
from math import atan2
from .subproblems import sp1, sp3, sp4
from .utils import _rot

def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def _ensure_list(val, flag):
    """Normalize SP return values to always be lists."""
    if isinstance(val, (list, np.ndarray)):
        # val has multiple solutions but flag is a single bool — replicate it
        n = len(val)
        return list(val), [flag] * n
    return [val], [flag]


def ik_3_parallel_2_intersecting(R_06, p_0T, kin):
    P = kin['P']
    H = kin['H']
    Q = []
    is_LS_vec = []
    p_06 = p_0T - P[:, 0] - R_06 @ P[:, 6]
    d1 = H[:, 1] @ P[:, 1:5].sum(axis=1)
    theta1_list, theta1_is_ls = _ensure_list(*sp4(p_06, -H[:, 0], H[:, 1], d1))

    for q_1, t1_ls in zip(theta1_list, theta1_is_ls):
        R_01 = _rot(H[:, 0], q_1)

        # --- q5 via SP4 ---
        # h2' R01' R06 h6 = h2' R45 h6
        d5 = H[:, 1] @ R_01.T @ R_06 @ H[:, 5]
        theta5_list, theta5_is_ls = _ensure_list(*sp4(H[:, 5], H[:, 4], H[:, 1], d5))

        for q_5, t5_ls in zip(theta5_list, theta5_is_ls):
            R_45 = _rot(H[:, 4], q_5)

            # --- theta_14 (q2+q3+q4) via SP1 ---
            theta_14, t14_ls = sp1(R_45 @ H[:, 5], R_01.T @ R_06 @ H[:, 5], H[:, 1])

            # --- q6 via SP1 ---
            q_6, q6_ls = sp1(R_45.T @ H[:, 1], R_06.T @ R_01 @ H[:, 1], -H[:, 5])

            # --- q3 via SP3 ---
            # d_inner = R01' p06 - p12 - rot(h2, theta_14) p45
            # note p_56 = 0 so P[:,4] is p_45
            d_inner = R_01.T @ p_06 - P[:, 1] - _rot(H[:, 1], theta_14) @ P[:, 4]
            d = np.linalg.norm(d_inner)
            theta3_list, t3_ls = _ensure_list(*sp3(-P[:, 3], P[:, 2], H[:, 1], d))

            for q_3, q3_ls in zip(theta3_list, t3_ls):

                q_2, q2_ls = sp1(P[:, 2] + _rot(H[:, 1], q_3) @ P[:, 3], d_inner, H[:, 1])

                # --- q4 by subtraction ---
                q_4 = _wrap_to_pi(theta_14 - q_2 - q_3)

                q_i = np.array([q_1, q_2, q_3, q_4, q_5, q_6])
                ls_i = np.array([t1_ls, t5_ls, t14_ls, q3_ls, q2_ls, q6_ls])

                Q.append(q_i)
                is_LS_vec.append(ls_i)

    if len(Q) == 0:
        return np.zeros((6, 0)), np.zeros((6, 0), dtype=bool)

    return np.stack(Q, axis=1), np.stack(is_LS_vec, axis=1)