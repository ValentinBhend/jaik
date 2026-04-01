import numpy as np


# Utility functions ported from MATLAB model_utils.m


def clamp_original(x):
    return np.maximum(-1 + 1e-12, np.minimum(1 - 1e-12, x))


def acos_clamp(x):
    return np.arccos(clamp_original(x))


def skew_symmetric(v):
    v = np.asarray(v).flatten()
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotm_to_axisangle(R):
    R = np.asarray(R)
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / (2 * np.sin(angle))
        axis = v / max(1e-12, np.linalg.norm(v))
    return axis, angle


def rvec2rotm(rvec):
    rvec = np.asarray(rvec).flatten()
    angle = np.linalg.norm(rvec)
    if angle < 1e-12:
        return np.eye(3)
    axis = rvec / angle
    K = skew_symmetric(axis)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def rotm2rvec(R, policy='principal'):
    axis, angle = rotm_to_axisangle(R)
    r1 = axis * angle
    r2 = -axis * (2 * np.pi - angle)
    if policy == 'principal':
        return r1
    elif policy == 'tp_long':
        if angle < 1e-12:
            return np.zeros(3)
        elif abs(angle - np.pi) < 1e-6:
            return r1
        else:
            return r2
    elif policy == 'continuous':
        if not hasattr(rotm2rvec, '_rprev'):
            rotm2rvec._rprev = np.zeros(3)
        rprev = rotm2rvec._rprev
        if np.linalg.norm(r1 - rprev) <= np.linalg.norm(r2 - rprev):
            rvec = r1
        else:
            rvec = r2
        rotm2rvec._rprev = rvec
        return rvec
    else:
        raise ValueError(f'Unknown orientation policy: {policy}')


def dh_transform(theta, a, d, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    A = np.array([[ct, -st*ca, st*sa, a*ct],
                  [st, ct*ca, -ct*sa, a*st],
                  [0, sa, ca, d],
                  [0, 0, 0, 1]])
    return A


def compute_fk_frames(q, params):
    q = np.asarray(q).flatten()
    o = np.zeros((3, 6))
    z = np.zeros((3, 6))
    T = np.eye(4)
    Tlist = [None] * 6
    o[:, 0] = np.array([0, 0, 0])
    z[:, 0] = np.array([0, 0, 1])
    for i in range(6):
        A_i = dh_transform(q[i], params['a'][i], params['d'][i], params['alpha'][i])
        T = T @ A_i
        Tlist[i] = T.copy()
        if i < 5:
            o[:, i+1] = T[0:3, 3]
            z[:, i+1] = T[0:3, 2]
    return Tlist, o, z


def compute_jacobians_all(q, params):
    Tlist, o, z = compute_fk_frames(q, params)
    Jv_all = np.zeros((3, 6, 6))
    Jw_all = np.zeros((3, 6, 6))
    R_all = np.zeros((3, 3, 6))
    for i in range(6):
        Ti = Tlist[i]
        R = Ti[0:3, 0:3]
        p_com_local = np.hstack((params['rc'][:, i], 1.0))
        p_com_world = Ti @ p_com_local
        p_com = p_com_world[0:3]
        Jv = np.zeros((3, 6))
        Jw = np.zeros((3, 6))
        for j in range(6):
            if j <= i:
                Jw[:, j] = z[:, j]
                Jv[:, j] = np.cross(z[:, j], (p_com - o[:, j]))
        Jv_all[:, :, i] = Jv
        Jw_all[:, :, i] = Jw
        R_all[:, :, i] = R
    return Jv_all, Jw_all, R_all


def compute_mass_matrix(q, params):
    Jv_all, Jw_all, R_all = compute_jacobians_all(q, params)
    M = np.zeros((6, 6))
    for i in range(6):
        Jv = Jv_all[:, :, i]
        Jw = Jw_all[:, :, i]
        R = R_all[:, :, i]
        M += params['m'][i] * (Jv.T @ Jv)
        I_world = R @ params['I'][:, :, i] @ R.T
        M += Jw.T @ I_world @ Jw
    return M


def compute_gravity_vector(q, params):
    Jv_all, _, _ = compute_jacobians_all(q, params)
    G = np.zeros(6)
    for i in range(6):
        Jv = Jv_all[:, :, i]
        G += Jv.T @ (params['m'][i] * params['G_BASE'])
    return G


def compute_coriolis_matrix(q, qd, params):
    h = 1e-6
    dM = np.zeros((6, 6, 6))
    q = np.asarray(q).flatten()
    for k in range(6):
        dq = np.zeros(6)
        dq[k] = h
        M_plus = compute_mass_matrix(q + dq, params)
        M_minus = compute_mass_matrix(q - dq, params)
        dM[:, :, k] = (M_plus - M_minus) / (2 * h)
    C = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            cij = 0.0
            for k in range(6):
                christoffel = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
                cij += christoffel * qd[k]
            C[i, j] = cij
    return C


def compute_jacobian(q, params, point=None):
    if point is None:
        point = np.zeros(3)
    Tlist, o, z = compute_fk_frames(q, params)
    T_ee = Tlist[5] @ params['Xtcp']
    p_world = T_ee @ np.hstack((point, 1.0))
    p_point = p_world[0:3]
    Jv = np.zeros((3, 6))
    Jw = np.zeros((3, 6))
    for i in range(6):
        if i == 0:
            z_i = np.array([0, 0, 1])
            o_i = np.array([0, 0, 0])
        else:
            z_i = z[:, i]
            o_i = o[:, i]
        Jw[:, i] = z_i
        Jv[:, i] = np.cross(z_i, (p_point - o_i))
    J = np.vstack((Jv, Jw))
    return J, Jv, Jw


def compute_tau_dot(q, v, tau, params):
    M0 = compute_mass_matrix(q, params)
    h = 1e-6
    dM = np.zeros((6,6,6))
    for k in range(6):
        dq = np.zeros(6)
        dq[k] = h
        Mp = compute_mass_matrix(q + dq, params)
        Mm = compute_mass_matrix(q - dq, params)
        dM[:,:,k] = (Mp - Mm)/(2*h)
    C = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            cij = 0.0
            for k in range(6):
                cijk = 0.5*(dM[i,j,k] + dM[i,k,j] - dM[j,k,i])
                cij += cijk * v[k]
            C[i,j] = cij
    Cqd = C @ v
    G = compute_gravity_vector(q, params)
    tau_friction = params['Bv'] * v + params['Fc'] * np.sign(v)
    tau_required = Cqd + G + tau_friction
    tau_dot = (tau_required - tau) / params['Ttau']
    return tau_dot


def compute_ah_ik(n, th, c, a, d, alpha):
    theta = th[n-1, c-1]
    T_a = np.eye(4)
    T_a[0,3] = a[n-1]
    T_d = np.eye(4)
    T_d[2,3] = d[n-1]
    Rzt = np.array([[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0,0,1,0], [0,0,0,1]])
    Rxa = np.array([[1,0,0,0], [0, np.cos(alpha[n-1]), -np.sin(alpha[n-1]), 0], [0, np.sin(alpha[n-1]), np.cos(alpha[n-1]), 0], [0,0,0,1]])
    A_i = T_d @ Rzt @ T_a @ Rxa
    return A_i


def is_valid_pose(T):
    T = np.asarray(T)
    if T.shape != (4,4):
        return False
    if not np.allclose(T[3,:], np.array([0,0,0,1])):
        return False
    R = T[0:3,0:3]
    is_orthogonal = np.linalg.norm(R @ R.T - np.eye(3), 'fro') < 1e-10
    is_proper = abs(np.linalg.det(R) - 1) < 1e-10
    return is_orthogonal and is_proper


def pose_error(T1, T2):
    T1 = np.asarray(T1); T2 = np.asarray(T2)
    pos_error = np.linalg.norm(T1[0:3,3] - T2[0:3,3])
    R_error = T1[0:3,0:3].T @ T2[0:3,0:3]
    rvec_error = rotm2rvec(R_error, 'principal')
    rot_error = np.linalg.norm(rvec_error)
    total_error = pos_error + rot_error
    return pos_error, rot_error, total_error


def wrap_angles(angles):
    angles = np.asarray(angles)
    return np.arctan2(np.sin(angles), np.cos(angles))


def compute_manipulability(J):
    J = np.asarray(J)
    if J.shape[0] == 6 and J.shape[1] == 6:
        return abs(np.linalg.det(J))
    s = np.linalg.svd(J, compute_uv=False)
    return np.prod(s)


def find_singularities(q_traj, params, threshold=1e6):
    q_traj = np.asarray(q_traj)
    if q_traj.shape[0] != 6:
        raise ValueError('Joint trajectory must have 6 rows')
    N = q_traj.shape[1]
    condition_numbers = np.zeros(N)
    for i in range(N):
        J, _, _ = compute_jacobian(q_traj[:, i], params)
        condition_numbers[i] = np.linalg.cond(J)
    singular_indices = np.where(condition_numbers > threshold)[0]
    return singular_indices, condition_numbers


def joint_limits_check(q, q_min=None, q_max=None):
    q = np.asarray(q).flatten()
    if q_min is None:
        q_min = -2*np.pi * np.ones_like(q)
    if q_max is None:
        q_max = 2*np.pi * np.ones_like(q)
    in_limits = np.all((q >= q_min) & (q <= q_max))
    violations = {'min_violations': np.where(q < q_min)[0], 'max_violations': np.where(q > q_max)[0]}
    return in_limits, violations
