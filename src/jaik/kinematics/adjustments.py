import numpy as np


def adjust_kin_for_3p2i(kin):
    """
    Adjust kin so that p_12=0 and p_56=0, required by ik_3_parallel_2_intersecting.
    Slides joint 1 origin to coincide with joint 2, and joint 5 to coincide with 6.
    Compensates in p_01 and p_45 respectively.

    This is a deliberate PoE convention choice — the IK solver is derived
    assuming this canonical form. IK-Geo does this manually per robot.
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