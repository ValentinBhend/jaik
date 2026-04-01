import numpy as np
from .model_utils import acos_clamp, compute_ah_ik


def model_inversek(T_desired, params):
    T_desired = np.asarray(T_desired)
    a = params['a']
    d = params['d']
    alpha = params['alpha']
    a2 = params['a2']
    a3 = params['a3']
    d4 = params['d4']
    d6 = params['d6']
    eps_sing = params['eps_sing']

    th = np.zeros((6, 8))
    P_05 = T_desired @ np.array([0, 0, -d6, 1]) - np.array([0, 0, 0, 1])
    psi = np.arctan2(P_05[1], P_05[0])
    r = np.hypot(P_05[0], P_05[1])
    phi = np.arccos(d4 / max(1e-12, r))
    th[0, 0:4] = np.pi/2 + psi + phi
    th[0, 4:8] = np.pi/2 + psi - phi

    for c in [0, 4]:
        T_10 = np.linalg.inv(compute_ah_ik(1, th, c+1, a, d, alpha))
        T_16 = T_10 @ T_desired
        arg = (T_16[2,3] - d4)/d6
        arg = np.clip(arg, -1+1e-12, 1-1e-12)
        val = np.arccos(arg)
        th[4, c:(c+2)] = val
        th[4, (c+2):(c+4)] = -val

    for c in [0,2,4,6]:
        T_10 = np.linalg.inv(compute_ah_ik(1, th, c+1, a, d, alpha))
        T_16 = np.linalg.inv(T_10 @ T_desired)
        s5 = np.sin(th[4,c])
        if abs(s5) < eps_sing:
            th[5, c:(c+2)] = 0
        else:
            th[5, c:(c+2)] = np.arctan2(-T_16[1,2]/s5, T_16[0,2]/s5)

    for c in [0,2,4,6]:
        T_10 = np.linalg.inv(compute_ah_ik(1, th, c+1, a, d, alpha))
        T_65 = compute_ah_ik(6, th, c+1, a, d, alpha)
        T_54 = compute_ah_ik(5, th, c+1, a, d, alpha)
        T_14 = (T_10 @ T_desired) @ np.linalg.inv(T_54 @ T_65)
        P_13 = T_14 @ np.array([0, -d4, 0, 1]) - np.array([0,0,0,1])
        cos_t3 = (np.linalg.norm(P_13)**2 - a2**2 - a3**2) / (2*a2*a3)
        cos_t3 = np.clip(cos_t3, -1+1e-12, 1-1e-12)
        t3 = np.arccos(cos_t3)
        th[2, c] = np.real(t3)
        th[2, c+1] = -np.real(t3)

    for c in range(8):
        T_10 = np.linalg.inv(compute_ah_ik(1, th, c+1, a, d, alpha))
        T_65 = np.linalg.inv(compute_ah_ik(6, th, c+1, a, d, alpha))
        T_54 = np.linalg.inv(compute_ah_ik(5, th, c+1, a, d, alpha))
        T_14 = (T_10 @ T_desired) @ T_65 @ T_54
        P_13 = T_14 @ np.array([0, -d4, 0, 1]) - np.array([0,0,0,1])
        denom = max(np.linalg.norm(P_13), 1e-12)
        arg = np.clip(a3 * np.sin(th[2,c]) / denom, -1+1e-12, 1-1e-12)
        th[1,c] = -np.arctan2(P_13[1], -P_13[0]) + np.arcsin(arg)
        T_32 = np.linalg.inv(compute_ah_ik(3, th, c+1, a, d, alpha))
        T_21 = np.linalg.inv(compute_ah_ik(2, th, c+1, a, d, alpha))
        T_34 = T_32 @ T_21 @ T_14
        th[3,c] = np.arctan2(T_34[1,0], T_34[0,0])

    q_all = np.arctan2(np.sin(th), np.cos(th))
    return q_all
