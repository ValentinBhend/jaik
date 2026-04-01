import numpy as np
from .utils import _rot

def _fk(q, kin):
    H = kin['H']
    P = kin['P']
    R = np.eye(3)
    p = P[:, 0].copy()
    for i in range(6):
        R = R @ _rot(H[:, i], q[i])
        p = p + R @ P[:, i + 1]
    return R, p