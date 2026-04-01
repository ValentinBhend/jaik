# tests/test_numpy_ur10e.py
import numpy as np
import pytest
from jaik.kinematics.convention_conversions import dh_to_kin
from jaik._numpy.fk import _fk
from jaik._numpy.ik_3p2i import ik_3_parallel_2_intersecting
from jaik.kinematics.adjustments import adjust_kin_for_3p2i

# UR10e nominal DH parameters (e-series)
# Source: Universal Robots support articles
# alpha [rad], a [m], d [m]
UR10E_ALPHA = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
UR10E_A     = np.array([0, -0.6127, -0.57155, 0, 0, 0])
UR10E_D     = np.array([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])


@pytest.fixture
def ur10e_kin():
    return dh_to_kin(UR10E_ALPHA, UR10E_A, UR10E_D)


@pytest.fixture
def ur10e_kin_adj(ur10e_kin):
    return adjust_kin_for_3p2i(ur10e_kin)


# ─── DH → PoE conversion tests ──────────────────────────────────────────────

def test_dh_to_kin_shape(ur10e_kin):
    assert ur10e_kin['H'].shape == (3, 6)
    assert ur10e_kin['P'].shape == (3, 7)


def test_dh_to_kin_first_axis(ur10e_kin):
    """First joint axis should be [0, 0, 1] (z-axis)."""
    np.testing.assert_allclose(ur10e_kin['H'][:, 0], [0, 0, 1], atol=1e-10)


# ─── FK tests (use raw kin) ──────────────────────────────────────────────────

def test_fk_zero_config(ur10e_kin):
    """At q=0 the position should be within reach of a 1.3m arm."""
    q = np.zeros(6)
    R, p = _fk(q, ur10e_kin)
    assert R.shape == (3, 3)
    assert p.shape == (3,)
    assert np.linalg.norm(p) < 1.5
    assert np.linalg.norm(p) > 0.1


def test_fk_rotation_orthogonal(ur10e_kin):
    """R must always be a valid rotation matrix."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, 6)
        R, _ = _fk(q, ur10e_kin)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


# ─── Adjustment tests ────────────────────────────────────────────────────────

def test_adjust_p12_p56_zero(ur10e_kin_adj):
    """After adjustment, p_12 and p_56 must be zero."""
    np.testing.assert_allclose(ur10e_kin_adj['P'][:, 1], np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(ur10e_kin_adj['P'][:, 5], np.zeros(3), atol=1e-10)


def test_adjust_preserves_fk(ur10e_kin, ur10e_kin_adj):
    """Adjusting kin must not change FK output — it is purely a frame choice."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        q = rng.uniform(-np.pi, np.pi, 6)
        R1, p1 = _fk(q, ur10e_kin)
        R2, p2 = _fk(q, ur10e_kin_adj)
        np.testing.assert_allclose(R1, R2, atol=1e-10)
        np.testing.assert_allclose(p1, p2, atol=1e-10)


# ─── IK tests (use adjusted kin) ─────────────────────────────────────────────

def test_ik_known_config(ur10e_kin_adj):
    """A specific known joint config should round-trip exactly."""
    q_ref = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
    R_target, p_target = _fk(q_ref, ur10e_kin_adj)
    Q, is_LS = ik_3_parallel_2_intersecting(R_target, p_target, ur10e_kin_adj)

    assert Q.shape[1] > 0, "IK returned no solutions for a known config"

    found = False
    for i in range(Q.shape[1]):
        R_check, p_check = _fk(Q[:, i], ur10e_kin_adj)
        if (np.allclose(R_check, R_target, atol=1e-6) and
                np.allclose(p_check, p_target, atol=1e-6)):
            found = True
            break

    assert found, "None of the IK solutions matched the known config via FK"


def test_ik_returns_up_to_8_solutions(ur10e_kin_adj):
    """IK should return at most 8 solutions."""
    q_ref = np.array([0.1, -0.5, 1.2, -0.3, 0.8, -1.1])
    R_target, p_target = _fk(q_ref, ur10e_kin_adj)
    Q, _ = ik_3_parallel_2_intersecting(R_target, p_target, ur10e_kin_adj)
    assert Q.shape[0] == 6
    assert Q.shape[1] <= 8


def test_ik_roundtrip(ur10e_kin_adj):
    """FK(IK(T)) ≈ T for random reachable poses."""
    rng = np.random.default_rng(0)
    n_tests = 20
    n_success = 0

    for _ in range(n_tests):
        q_ref = rng.uniform(-np.pi, np.pi, 6)
        R_target, p_target = _fk(q_ref, ur10e_kin_adj)
        Q, is_LS = ik_3_parallel_2_intersecting(R_target, p_target, ur10e_kin_adj)

        if Q.shape[1] == 0:
            continue

        for i in range(Q.shape[1]):
            if is_LS[:, i].any():
                continue
            R_check, p_check = _fk(Q[:, i], ur10e_kin_adj)
            if (np.allclose(R_check, R_target, atol=1e-6) and
                    np.allclose(p_check, p_target, atol=1e-6)):
                n_success += 1
                break

    assert n_success >= int(0.8 * n_tests), (
        f"Only {n_success}/{n_tests} poses had a valid IK roundtrip"
    )


def test_ik_roundtrip_debug(ur10e_kin_adj):
    rng = np.random.default_rng(0)
    for trial in range(5):
        q_ref = rng.uniform(-np.pi, np.pi, 6)
        R_target, p_target = _fk(q_ref, ur10e_kin_adj)
        Q, is_LS = ik_3_parallel_2_intersecting(R_target, p_target, ur10e_kin_adj)

        print(f"\nTrial {trial}: {Q.shape[1]} solutions")
        for i in range(Q.shape[1]):
            R_check, p_check = _fk(Q[:, i], ur10e_kin_adj)
            p_err = np.linalg.norm(p_check - p_target)
            R_err = np.linalg.norm(R_check - R_target)
            print(f"  sol {i}: p_err={p_err:.4f} R_err={R_err:.4f} "
                  f"is_LS={is_LS[:,i]} any_ls={is_LS[:,i].any()}")