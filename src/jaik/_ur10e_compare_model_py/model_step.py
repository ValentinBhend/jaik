import numpy as np
from .model_utils import compute_tau_dot
from .model_forwardk import model_forwardk


def model_step(x, v, Ts, params):
    x = np.asarray(x).flatten()
    v = np.asarray(v).flatten()
    Ts = float(Ts)
    q = x[0:6]
    tau = x[6:12]
    f_taudot = lambda qq, tt: compute_tau_dot(qq, v, tt, params)
    # RK4
    k1_q = v
    k1_tau = f_taudot(q, tau)
    k2_q = v
    k2_tau = f_taudot(q + 0.5*Ts*k1_q, tau + 0.5*Ts*k1_tau)
    k3_q = v
    k3_tau = f_taudot(q + 0.5*Ts*k2_q, tau + 0.5*Ts*k2_tau)
    k4_q = v
    k4_tau = f_taudot(q + Ts*k3_q, tau + Ts*k3_tau)
    q_next = q + (Ts/6.0)*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    tau_next = tau + (Ts/6.0)*(k1_tau + 2*k2_tau + 2*k3_tau + k4_tau)
    x_next = np.hstack((q_next, tau_next))
    T_EE, _ = model_forwardk(q_next, params, False, params['Xtcp'])
    R = T_EE[0:3, 0:3]
    p = T_EE[0:3, 3]
    rvec = None
    try:
        from .model_utils import rotm2rvec
        rvec = rotm2rvec(R, params.get('oriPolicy', 'principal'))
    except Exception:
        rvec = np.zeros(3)
    out = {}
    out['q'] = q_next
    out['tau'] = tau_next
    out['T_EE'] = T_EE
    out['p'] = p
    out['R'] = R
    out['Rx'] = R[:,0]
    out['Ry'] = R[:,1]
    out['Rz'] = R[:,2]
    out['rvec'] = rvec
    return x_next, out
