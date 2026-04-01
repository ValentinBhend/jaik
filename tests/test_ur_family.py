# tests/test_ur_family.py
"""
FK/IK tests for all UR robot presets, both numpy and JAX backends.
Replaces test_numpy_ur_family.py.
"""
import numpy as np
import pytest
import jaik
from jaik.kinematics.robots import available_robots, _UR_PARAMS


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_np(x):
    """Convert JAX or numpy array to numpy."""
    return np.asarray(x)


def _to_q(q_np, backend):
    """Convert numpy joint array to the right type for the backend."""
    if backend == "jax":
        import jax.numpy as jnp
        return jnp.array(q_np)
    return q_np


def _call_fk(fk, q_np, backend):
    """Call fk with the right array type for the backend."""
    return fk(_to_q(q_np, backend))


def _exact(Q, is_LS):
    """Filter to exact (non-LS, non-NaN) solutions."""
    Q     = _to_np(Q)
    is_LS = _to_np(is_LS)
    mask  = ~is_LS.any(axis=0) & ~np.isnan(Q).any(axis=0)
    return Q[:, mask]


# ── parametrize over all robots × both backends ───────────────────────────────

@pytest.fixture(
    params=[
        pytest.param((name, backend), id=f"{name}-{backend}")
        for name in available_robots()
        for backend in ("numpy", "jax")
    ]
)
def fk_ik(request):
    name, backend = request.param
    fk, ik = jaik.make_robot(name, backend=backend)
    return fk, ik, name, backend


# ── FK tests ──────────────────────────────────────────────────────────────────

def test_fk_shape(fk_ik):
    fk, _, name, backend = fk_ik
    R, p = _call_fk(fk, np.zeros(6), backend)
    assert _to_np(R).shape == (3, 3)
    assert _to_np(p).shape == (3,)


def test_fk_rotation_valid(fk_ik):
    """R must be a valid rotation matrix for random configs."""
    fk, _, name, backend = fk_ik
    rng = np.random.default_rng(42)
    for _ in range(10):
        q = rng.uniform(-np.pi, np.pi, 6)
        R, _ = _call_fk(fk, q, backend)
        R = _to_np(R)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_fk_reach(fk_ik):
    """At q=0 the end effector should be within plausible reach."""
    fk, _, name, backend = fk_ik
    p = _UR_PARAMS[name]
    max_reach = (abs(p["a2"]) + abs(p["a3"]) +
                 p["d1"] + p["d4"] + p["d5"] + p["d6"])
    _, pos = _call_fk(fk, np.zeros(6), backend)
    pos_norm = np.linalg.norm(_to_np(pos))
    assert pos_norm < max_reach * 1.5
    assert pos_norm > 0.05


# ── IK tests ──────────────────────────────────────────────────────────────────

def test_ik_known_config(fk_ik):
    """A canonical config should round-trip exactly."""
    fk, ik, name, backend = fk_ik
    q_ref = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
    R_target, p_target = _call_fk(fk, q_ref, backend)
    Q, is_LS = ik(R_target, p_target)
    Q_exact = _exact(Q, is_LS)

    assert Q_exact.shape[1] > 0, \
        f"IK returned no exact solutions for {name} ({backend})"

    found = any(
        np.allclose(_to_np(_call_fk(fk, Q_exact[:, i], backend)[0]),
                    _to_np(R_target), atol=1e-5) and
        np.allclose(_to_np(_call_fk(fk, Q_exact[:, i], backend)[1]),
                    _to_np(p_target), atol=1e-5)
        for i in range(Q_exact.shape[1])
    )
    assert found, f"No IK solution matched known config for {name} ({backend})"


def test_ik_output_shape(fk_ik):
    """IK output shape: JAX always (6,8), numpy at most (6,8)."""
    fk, ik, name, backend = fk_ik
    q_ref = np.array([0.1, -0.5, 1.2, -0.3, 0.8, -1.1])
    R_target, p_target = _call_fk(fk, q_ref, backend)
    Q, is_LS = ik(R_target, p_target)
    Q     = _to_np(Q)
    is_LS = _to_np(is_LS)

    assert Q.shape[0]     == 6
    assert is_LS.shape[0] == 6
    assert Q.shape[1]     == is_LS.shape[1]

    if backend == "jax":
        assert Q.shape[1] == 8, "JAX must always return exactly 8 branches"
    else:
        assert Q.shape[1] <= 8, "numpy must return at most 8 solutions"


def test_ik_roundtrip(fk_ik):
    """FK(IK(T)) ≈ T for random reachable poses."""
    fk, ik, name, backend = fk_ik
    rng = np.random.default_rng(0)
    n_tests = 20
    n_success = 0

    for _ in range(n_tests):
        q_ref = rng.uniform(-np.pi, np.pi, 6)
        R_target, p_target = _call_fk(fk, q_ref, backend)
        Q, is_LS = ik(R_target, p_target)
        Q_exact = _exact(Q, is_LS)

        if Q_exact.shape[1] == 0:
            continue

        for i in range(Q_exact.shape[1]):
            R_check, p_check = _call_fk(fk, Q_exact[:, i], backend)
            if (np.allclose(_to_np(R_check), _to_np(R_target), atol=1e-5) and
                    np.allclose(_to_np(p_check), _to_np(p_target), atol=1e-5)):
                n_success += 1
                break

    assert n_success >= int(0.8 * n_tests), (
        f"Only {n_success}/{n_tests} roundtrips succeeded for {name} ({backend})"
    )