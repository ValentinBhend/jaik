import numpy as np
from model_py import model_params, model_forwardk, model_inversek, model_dynamics, model_step


def test_params_loading():
    params = model_params()
    required_fields = ['a', 'd', 'alpha', 'm', 'rc', 'I', 'G_BASE', 'Ttau']
    for f in required_fields:
        assert f in params
    assert len(params['a']) == 6
    assert len(params['d']) == 6
    assert len(params['alpha']) == 6
    assert len(params['m']) == 6
    rc = params['rc']
    assert rc.shape == (3, 6)


def test_forward_kinematics():
    params = model_params()
    q_home = np.zeros(6)
    T_EE, T_flange = model_forwardk(q_home, params)
    R = T_EE[0:3, 0:3]
    assert R.shape == (3, 3)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-10)
    assert np.linalg.norm(R @ R.T - np.eye(3)) < 1e-10

    q_rand = (np.random.rand(6) - 0.5) * 2 * np.pi
    T_EE_rand, _ = model_forwardk(q_rand, params)
    Rr = T_EE_rand[0:3, 0:3]
    assert np.allclose(np.linalg.det(Rr), 1.0, atol=1e-8)


def test_inverse_kinematics_home():
    params = model_params()
    q_home = np.zeros(6)
    T_target, _ = model_forwardk(q_home, params)
    q_solutions = model_inversek(T_target, params)
    # Check shape and that at least one solution is close to home
    assert q_solutions.shape[0] == 6
    found = False
    for i in range(q_solutions.shape[1]):
        q_sol = q_solutions[:, i]
        # allow a slightly looser tolerance to account for small numerical differences
        if np.linalg.norm(np.arctan2(np.sin(q_sol - q_home), np.cos(q_sol - q_home))) < 1e-5:
            found = True
            break
    assert found


def test_fk_ik_consistency_random():
    params = model_params()
    for _ in range(5):
        q_orig = (np.random.rand(6) - 0.5) * 2 * np.pi
        T_target, _ = model_forwardk(q_orig, params)
        q_solutions = model_inversek(T_target, params)
        valid_solutions = 0
        for i in range(q_solutions.shape[1]):
            q_sol = q_solutions[:, i]
            if np.all(np.isfinite(q_sol)):
                T_check, _ = model_forwardk(q_sol, params)
                pos_error = np.linalg.norm(T_check[0:3, 3] - T_target[0:3, 3])
                rot_error = np.linalg.norm(T_check[0:3, 0:3] - T_target[0:3, 0:3], ord='fro')
                if pos_error < 1e-6 and rot_error < 1e-6:
                    valid_solutions += 1
        assert valid_solutions > 0


def test_dynamics_properties():
    params = model_params()
    q = np.array([0.1, -0.5, 0.3, -1.2, 0.8, 0.2])
    qd = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.1])
    M, C, G = model_dynamics(q, qd, params)
    # Mass matrix symmetric and positive definite
    assert np.allclose(M, M.T, atol=1e-8)
    eigs = np.linalg.eigvals(M)
    assert np.all(np.real(eigs) > 0)
    Cqd = C @ qd
    assert np.all(np.isfinite(Cqd))
    assert np.all(np.isfinite(G))


def test_simulation_step_and_trajectory():
    params = model_params()
    q0 = np.zeros(6)
    tau0 = np.zeros(6)
    x0 = np.hstack((q0, tau0))
    v = np.array([0.1, -0.05, 0.02, 0.08, -0.03, 0.01])
    Ts = 0.01
    x_next, out = model_step(x0, v, Ts, params)
    assert x_next.shape[0] == 12
    assert 'q' in out and len(out['q']) == 6
    assert 'tau' in out and len(out['tau']) == 6
    assert 'rvec' in out and len(out['rvec']) == 3
    q_change = out['q'] - q0
    expected_change = v * Ts
    relative_error = np.linalg.norm(q_change - expected_change) / (np.linalg.norm(expected_change) + 1e-12)
    assert relative_error < 0.5

    # trajectory
    T_sim = 0.2
    Ts = 0.01
    N = int(round(T_sim / Ts))
    x = np.hstack((np.zeros(6), np.zeros(6)))
    q_traj = np.zeros((6, N))
    tau_traj = np.zeros((6, N))
    pos_traj = np.zeros((3, N))
    t = np.arange(N) * Ts
    freq = 0.5
    amp = 0.2
    for k in range(N):
        v_cmd = amp * np.sin(2*np.pi*freq*t[k]) * np.array([1,0.5,0.3,0.8,0.4,0.2])
        x, out = model_step(x, v_cmd, Ts, params)
        q_traj[:, k] = out['q']
        tau_traj[:, k] = out['tau']
        pos_traj[:, k] = out['p']
    assert np.all(np.isfinite(q_traj))
    assert np.all(np.isfinite(tau_traj))
    assert np.all(np.isfinite(pos_traj))
    q_diff = np.diff(q_traj, axis=1)
    max_q_jump = np.max(np.abs(q_diff))
    assert max_q_jump < 0.2
