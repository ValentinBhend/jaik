import numpy as np
from math import atan2
from .subproblems import sp1, sp2, sp3, sp4
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
    """
    IK for robots with h2=h3=h4 (3 parallel) and p_56=0 (2 intersecting).
    e.g. UR family.

    Args:
        R_06:  (3,3) rotation matrix of end effector in base frame
        p_0T:  (3,)  position of tool in base frame
        kin:   dict with H (3,6), P (3,7), joint_type (6,)

    Returns:
        Q:          (6, N_solutions) joint angles
        is_LS_vec:  (6, N_solutions) least-squares flags per solution
    """
    P = kin['P']
    H = kin['H']

    Q = []
    is_LS_vec = []

    # p_06: position of joint 6 origin in base frame
    p_06 = p_0T - P[:, 0] - R_06 @ P[:, 6]

    # --- q1 via SP4 ---
    # h2' R01' p16 = h2' (p12 + p23 + p34 + p45)
    d1 = H[:, 1] @ P[:, 1:5].sum(axis=1)
    theta1_list, theta1_is_ls = _ensure_list(*sp4(H[:, 1], -H[:, 0], p_06, d1))

    for q_1, t1_ls in zip(theta1_list, theta1_is_ls):
        R_01 = _rot(H[:, 0], q_1)

        # --- q5 via SP4 ---
        # h2' R01' R06 h6 = h2' R45 h6
        d5 = H[:, 1] @ R_01.T @ R_06 @ H[:, 5]
        theta5_list, theta5_is_ls = _ensure_list(*sp4(H[:, 1], H[:, 4], H[:, 5], d5))

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

                # --- q2 via SP1 ---
                
                # --- q2 via SP1 ---
                p1_debug = P[:, 2] + _rot(H[:, 1], q_3) @ P[:, 3]
                p2_debug = d_inner
                print(f"  SP1(q2) ||p1||={np.linalg.norm(p1_debug):.10f}  "
                    f"||p2||={np.linalg.norm(p2_debug):.10f}  "
                    f"diff={abs(np.linalg.norm(p1_debug)-np.linalg.norm(p2_debug)):.2e}  "
                    f"k.p1={np.dot(H[:,1],p1_debug):.2e}  "
                    f"k.p2={np.dot(H[:,1],p2_debug):.2e}")
                q_2, q2_ls = sp1(p1_debug, p2_debug, H[:, 1])

                # q_2, q2_ls = sp1(P[:, 2] + _rot(H[:, 1], q_3) @ P[:, 3], d_inner, H[:, 1])

                # --- q4 by subtraction ---
                q_4 = _wrap_to_pi(theta_14 - q_2 - q_3)

                q_i = np.array([q_1, q_2, q_3, q_4, q_5, q_6])
                ls_i = np.array([t1_ls, t5_ls, t14_ls, q3_ls, q2_ls, q6_ls])

                Q.append(q_i)
                is_LS_vec.append(ls_i)

    if len(Q) == 0:
        return np.zeros((6, 0)), np.zeros((6, 0), dtype=bool)

    return np.stack(Q, axis=1), np.stack(is_LS_vec, axis=1)


def adjust_kin_for_3p2i(kin):
    """
    Adjust kin so that p_12=0 and p_56=0, required by ik_3_parallel_2_intersecting.
    Slides joint 1 origin to coincide with joint 2, and joint 5 to coincide with 6.
    Compensates in p_01 and p_45 respectively.

    This is a deliberate PoE convention choice, not a hotfix — the IK solver
    is derived assuming this canonical form. IK-Geo does this manually per robot.
    """
    kin = {k: v.copy() for k, v in kin.items()}
    P = kin['P']
    H = kin['H']

    # Make p_12 = 0: project p_12 onto h1, absorb into p_01
    h1 = H[:, 0]
    shift1 = np.dot(P[:, 1], h1) * h1
    P[:, 0] += shift1
    P[:, 1] -= shift1

    # Make p_56 = 0: project p_56 onto h5, absorb into p_45
    h5 = H[:, 4]
    shift5 = np.dot(P[:, 5], h5) * h5
    P[:, 4] += shift5
    P[:, 5] -= shift5

    kin['P'] = P
    return kin