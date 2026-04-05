# tests/test_numpy_ur_family.py
"""
Tests for all UR-type robots (3 parallel axes + 2 intersecting pairs).
All e-series robots share identical kinematic structure — only link
lengths (a and d values) differ.

DH parameter sources:
  UR3e, UR5e, UR10e, UR16e: Williams (2024) "Universal Robot URe-Series
    Cobot Kinematics & Dynamics" + Universal Robots support articles
  UR20e: Universal Robots support articles (official)

alpha and structure is identical for all UR-type robots:
  alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
  a     = [0, -a2, -a3, 0, 0, 0]
  d     = [d1, 0, 0, d4, d5, d6]
"""
import numpy as np
import pytest
from jaik.kinematics.convention_conversions import dh_to_kin
from jaik.kinematics.adjustments import adjust_kin_for_3p2i
from jaik._numpy.fk import _fk
from jaik._numpy.ik_3p2i import ik_3_parallel_2_intersecting


# (a2, a3, d1, d4, d5, d6) in meters
# Source: https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
_ALPHA = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]) # alpha is identical for all UR robots
_UR_PARAMS = {
    "UR8long":      dict(a2=-0.8989,    a3=-0.7149,     d1=0.2186,      d4=0.1824,      d5=0.1361,  d6=0.1434),
    "UR15":         dict(a2=-0.6475,    a3=-0.5164,     d1=0.2186,      d4=0.1824,      d5=0.1361,  d6=0.1434),
    "UR18":         dict(a2=-0.475,     a3=-0.3389,     d1=0.2186,      d4=0.1824,      d5=0.1361,  d6=0.1434),
    "UR20":         dict(a2=-0.8620,    a3=-0.7287,     d1=0.2363,      d4=0.2010,      d5=0.1593,  d6=0.1543),
    "UR30":         dict(a2=-0.6370,    a3=-0.5037,     d1=0.2363,      d4=0.2010,      d5=0.1593,  d6=0.1543),
    "UR3e":         dict(a2=-0.24355,   a3=-0.2132,     d1=0.15185,     d4=0.13105,     d5=0.08535, d6=0.0921),
    "UR5e_UR7e":    dict(a2=-0.425,     a3=-0.3922,     d1=0.1625,      d4=0.1333,      d5=0.0997,  d6=0.0996),
    "UR10e_UR12e":  dict(a2=-0.6127,    a3=-0.57155,    d1=0.1807,      d4=0.17415,     d5=0.11985, d6=0.11655),
    "UR16e":        dict(a2=-0.4784,    a3=-0.36,       d1=0.1807,      d4=0.17415,     d5=0.11985, d6=0.11655),
    "UR3":          dict(a2=-0.24365,   a3=-0.21325,    d1=0.1519,      d4=0.11235,     d5=0.08535, d6=0.0819),
    "UR5":          dict(a2=-0.425,     a3=-0.39225,    d1=0.089159,    d4=0.10915,     d5=0.09465, d6=0.0823),
    "UR10":         dict(a2=-0.612,     a3=-0.5723,     d1=0.1273,      d4=0.163941,    d5=0.1157,  d6=0.0922),
}


def _make_dh(p):
    a = np.array([0, p["a2"], p["a3"], 0, 0, 0])
    d = np.array([p["d1"], 0, 0, p["d4"], p["d5"], p["d6"]])
    return _ALPHA, a, d


# ── parametrize over all robots ──────────────────────────────────────────────

@pytest.fixture(params=list(_UR_PARAMS.keys()))
def robot_name(request):
    return request.param


@pytest.fixture
def kin_raw(robot_name):
    alpha, a, d = _make_dh(_UR_PARAMS[robot_name])
    return dh_to_kin(alpha, a, d)


@pytest.fixture
def kin_adj(kin_raw):
    return adjust_kin_for_3p2i(kin_raw)


# ── FK tests ──────────────────────────────────────────────────────────────────

def test_fk_shape(kin_raw):
    q = np.zeros(6)
    R, p = _fk(q, kin_raw)
    assert R.shape == (3, 3)
    assert p.shape == (3,)


def test_fk_rotation_valid(kin_raw):
    """R must be a valid rotation matrix for random configs."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        q = rng.uniform(-np.pi, np.pi, 6)
        R, _ = _fk(q, kin_raw)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_fk_reach(kin_raw, robot_name):
    p = _UR_PARAMS[robot_name]
    max_reach = (abs(p["a2"]) + abs(p["a3"]) + 
                 p["d4"] + p["d5"] + p["d6"] + p["d1"])
    q = np.zeros(6)
    _, pos = _fk(q, kin_raw)
    assert np.linalg.norm(pos) < max_reach * 1.5
    assert np.linalg.norm(pos) > 0.05


# ── Adjustment tests ──────────────────────────────────────────────────────────

def test_adjust_zeros_p12_p56(kin_adj):
    np.testing.assert_allclose(kin_adj['P'][:, 1], np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(kin_adj['P'][:, 5], np.zeros(3), atol=1e-10)


def test_adjust_preserves_fk(kin_raw, kin_adj):
    """Adjustment must not change FK — it is purely a frame choice."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        q = rng.uniform(-np.pi, np.pi, 6)
        R1, p1 = _fk(q, kin_raw)
        R2, p2 = _fk(q, kin_adj)
        np.testing.assert_allclose(R1, R2, atol=1e-10)
        np.testing.assert_allclose(p1, p2, atol=1e-10)


# ── IK tests ──────────────────────────────────────────────────────────────────

def test_ik_known_config(kin_adj):
    """A canonical config should round-trip exactly."""
    q_ref = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
    R_target, p_target = _fk(q_ref, kin_adj)
    Q, is_LS = ik_3_parallel_2_intersecting(R_target, p_target, kin_adj)

    assert Q.shape[1] > 0, "IK returned no solutions"

    found = any(
        np.allclose(_fk(Q[:, i], kin_adj)[0], R_target, atol=1e-6) and
        np.allclose(_fk(Q[:, i], kin_adj)[1], p_target, atol=1e-6)
        for i in range(Q.shape[1])
    )
    assert found, "No IK solution matched via FK"


def test_ik_returns_up_to_8_solutions(kin_adj):
    q_ref = np.array([0.1, -0.5, 1.2, -0.3, 0.8, -1.1])
    R_target, p_target = _fk(q_ref, kin_adj)
    Q, _ = ik_3_parallel_2_intersecting(R_target, p_target, kin_adj)
    assert Q.shape[0] == 6
    assert Q.shape[1] <= 8


def test_ik_roundtrip(kin_adj):
    """FK(IK(T)) ≈ T for random reachable poses."""
    rng = np.random.default_rng(0)
    n_tests = 20
    n_success = 0

    for _ in range(n_tests):
        q_ref = rng.uniform(-np.pi, np.pi, 6)
        R_target, p_target = _fk(q_ref, kin_adj)
        Q, is_LS = ik_3_parallel_2_intersecting(R_target, p_target, kin_adj)

        if Q.shape[1] == 0:
            continue

        for i in range(Q.shape[1]):
            if is_LS[:, i].any():
                continue
            R_check, p_check = _fk(Q[:, i], kin_adj)
            if (np.allclose(R_check, R_target, atol=1e-6) and
                    np.allclose(p_check, p_target, atol=1e-6)):
                n_success += 1
                break

    assert n_success >= int(0.8 * n_tests), (
        f"Only {n_success}/{n_tests} poses had a valid IK roundtrip"
    )