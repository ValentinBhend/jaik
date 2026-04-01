import numpy as np
from .utils import _rot

def dh_to_kin(alpha_vec, a_vec, d_vec):
    """
    Convert DH parameters to Product of Exponentials.
    
    Args:
        alpha_vec: (N,) link twists
        a_vec:     (N,) link lengths
        d_vec:     (N,) link offsets
    
    Returns dict with:
        H:          (3, N)   joint axes in base frame at home config
        P:          (3, N+1) displacement vectors between joint origins
        joint_type: (N,)     0 = revolute
    """
    N = len(alpha_vec)
    
    H = np.full((3, N), np.nan)
    P = np.full((3, N + 1), np.nan)
    
    P[:, 0] = [0, 0, 0]
    H[:, 0] = [0, 0, 1]
    
    R = np.eye(3)
    
    for i in range(N):
        # Translate d_i along z_{i-1}, a_i along x_{i-1}
        P[:, i + 1] = d_vec[i] * R[:, 2] + a_vec[i] * R[:, 0]
        # Rotate by alpha around x_{i-1}
        R = _rot(R[:, 0], alpha_vec[i]) @ R
        
        if i == N - 1: # TODO not sure why it does it twice..
            RT = _rot(R[:, 0], alpha_vec[i])
        else:
            H[:, i + 1] = R[:, 2]  # joint axis is z after rotation
    
    return {
        'H': H,
        'P': P,
        'joint_type': np.zeros(N),
        'RT': RT,
    }