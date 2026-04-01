import numpy as np
from .model_utils import compute_fk_frames, dh_transform


def model_forwardk(q, params, use_calibration=False, Xtcp=None, return_list=False):
    q = np.asarray(q).flatten()
    if Xtcp is None:
        Xtcp = np.eye(4)
    if use_calibration:
        params = params.copy()
        params['a'] = params['a'] + params['delta_a']
        params['d'] = params['d'] + params['delta_d']
        params['alpha'] = params['alpha'] + params['delta_alpha']
        q = q + params['delta_theta']
    Tlist, _, _ = compute_fk_frames(q, params)
    T_flange = Tlist[5]
    T_EE = T_flange @ Xtcp
    if return_list:
        return Tlist
    else:
        return T_EE, T_flange
