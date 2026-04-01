import numpy as np
from .utils import _rot

def fk(q, kin):
    H = kin['H']
    P = kin['P']
    R_6T = kin['RT']

    R = np.eye(3)
    p = P[:, 0].copy()  # p_01

    for i in range(6):
        R = R @ _rot(H[:, i], q[i])
        p = p + R @ P[:, i + 1]

    R_0T = R @ R_6T
    p_0T = p              # already includes R_06 @ p_6T from last iteration

    return R_0T, p_0T