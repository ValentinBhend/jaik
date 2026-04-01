# tests/test_numpy_compare_ur10e_model.py
"""
End-to-end cross-validation: compare jaik FK/IK against the legacy UR10e
reference model. Uses only the public jaik API — make_robot returns (fk, ik).
Uses numpy backend so results are directly comparable with the numpy reference.
"""
import numpy as np
import pytest
import jaik
from jaik._ur10e_compare_model_py import model_params, model_forwardk, model_inversek

PARAMS = model_params()


@pytest.fixture(scope="module")
def fk_ik():
    # numpy backend — legacy reference model is also numpy
    return jaik.make_robot("UR10e", backend="numpy")


# ── helpers ───────────────────────────────────────────────────────────────────

def _exact(Q, is_LS):
    """Filter to exact (non-LS) solutions."""
    return np.asarray(Q)[:, ~np.asarray(is_LS).any(axis=0)]


def _ref_exact_valid(Q_ref, p_target):
    """Filter reference solutions — drop NaN and self-inconsistent ones."""
    if Q_ref.ndim == 1:
        Q_ref = Q_ref[:, None]
    mask = ~np.isnan(Q_ref).any(axis=0)
    Q_ref = Q_ref[:, mask]
    if Q_ref.shape[1] == 0:
        return Q_ref
    valid = np.array([_ref_is_valid(Q_ref[:, i], p_target)
                      for i in range(Q_ref.shape[1])])
    return Q_ref[:, valid]


def _ref_is_valid(q, p_target):
    """Check reference solution is self-consistent via its own FK."""
    T_check, _ = model_forwardk(q, PARAMS)
    return np.allclose(T_check[:3, 3], p_target, atol=1e-4)


def _make_T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _angle_match(q1, q2, tol=1e-4):
    """Check if two joint configs match up to 2pi wrapping."""
    diff = np.arctan2(np.sin(q1 - q2), np.cos(q1 - q2))
    return np.linalg.norm(diff) < tol


# ── FK comparison ─────────────────────────────────────────────────────────────

class TestFKComparison:

    def test_fk_position_matches_reference(self, fk_ik):
        """jaik FK position should match reference for random configs."""
        fk, _ = fk_ik
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = rng.uniform(-np.pi, np.pi, 6)
            _, p_jaik = fk(q)
            T_ref, _ = model_forwardk(q, PARAMS)
            np.testing.assert_allclose(
                p_jaik, T_ref[:3, 3], atol=1e-6,
                err_msg=f"FK position mismatch at q={np.round(q, 3)}"
            )

    def test_fk_rotation_matches_reference(self, fk_ik):
        """jaik FK rotation should match reference (tool frame applied)."""
        fk, _ = fk_ik
        rng = np.random.default_rng(1)
        for _ in range(50):
            q = rng.uniform(-np.pi, np.pi, 6)
            R_jaik, _ = fk(q)
            T_ref, _ = model_forwardk(q, PARAMS)
            np.testing.assert_allclose(
                R_jaik, T_ref[:3, :3], atol=1e-6,
                err_msg=f"FK rotation mismatch at q={np.round(q, 3)}"
            )

    def test_fk_known_configs(self, fk_ik):
        """jaik FK matches reference on named configurations."""
        fk, _ = fk_ik
        configs = {
            "zero":    np.zeros(6),
            "home":    np.array([0.0, -np.pi/2,  np.pi/2, -np.pi/2, -np.pi/2, 0.0]),
            "all_pi4": np.full(6, np.pi/4),
            "all_neg": np.full(6, -np.pi/3),
        }
        for name, q in configs.items():
            R_jaik, p_jaik = fk(q)
            T_ref, _ = model_forwardk(q, PARAMS)
            np.testing.assert_allclose(
                p_jaik, T_ref[:3, 3], atol=1e-6,
                err_msg=f"FK position mismatch for config '{name}'"
            )
            np.testing.assert_allclose(
                R_jaik, T_ref[:3, :3], atol=1e-6,
                err_msg=f"FK rotation mismatch for config '{name}'"
            )


# ── IK comparison ─────────────────────────────────────────────────────────────

class TestIKComparison:

    def test_ik_exact_solutions_verify_via_fk(self, fk_ik):
        """All exact jaik IK solutions should round-trip via fk."""
        fk, ik = fk_ik
        rng = np.random.default_rng(3)
        for _ in range(30):
            q_ref = rng.uniform(-np.pi, np.pi, 6)
            R_target, p_target = fk(q_ref)
            Q, is_LS = ik(R_target, p_target)
            Q_exact = _exact(Q, is_LS)
            for i in range(Q_exact.shape[1]):
                R_check, p_check = fk(Q_exact[:, i])
                np.testing.assert_allclose(p_check, p_target, atol=1e-6)
                np.testing.assert_allclose(R_check, R_target, atol=1e-6)

    def test_ik_reference_solutions_verify_via_jaik_fk(self, fk_ik):
        """Valid reference IK solutions should verify via jaik fk."""
        fk, _ = fk_ik
        rng = np.random.default_rng(4)
        for _ in range(30):
            q_ref = rng.uniform(-np.pi, np.pi, 6)
            R_target, p_target = fk(q_ref)
            T = _make_T(R_target, p_target)
            # _ref_exact_valid already filters invalid solutions — no extra guard needed
            Q_ref = _ref_exact_valid(model_inversek(T, PARAMS), p_target)
            for i in range(Q_ref.shape[1]):
                R_check, p_check = fk(Q_ref[:, i])
                np.testing.assert_allclose(
                    p_check, p_target, atol=1e-6,
                    err_msg=f"Reference solution {i} fails jaik FK (position)"
                )
                np.testing.assert_allclose(
                    R_check, R_target, atol=1e-6,
                    err_msg=f"Reference solution {i} fails jaik FK (rotation)"
                )

    def test_ik_solution_count_matches(self, fk_ik):
        """jaik and reference should return the same number of exact solutions."""
        fk, ik = fk_ik
        rng = np.random.default_rng(2)
        mismatches = 0
        n_tests = 30
        for _ in range(n_tests):
            q_ref = rng.uniform(-np.pi, np.pi, 6)
            R_target, p_target = fk(q_ref)
            T = _make_T(R_target, p_target)
            Q, is_LS = ik(R_target, p_target)
            n_jaik = _exact(Q, is_LS).shape[1]
            n_ref  = _ref_exact_valid(model_inversek(T, PARAMS), p_target).shape[1]
            if n_jaik != n_ref:
                mismatches += 1
        assert mismatches <= 5, (
            f"{mismatches}/{n_tests} poses had different solution counts"
        )

    def test_ik_jaik_finds_reference_solution(self, fk_ik):
        """At least one jaik exact solution should match a reference solution."""
        fk, ik = fk_ik
        rng = np.random.default_rng(5)
        n_tests = 30
        n_found = 0
        for _ in range(n_tests):
            q_ref = rng.uniform(-np.pi, np.pi, 6)
            R_target, p_target = fk(q_ref)
            T = _make_T(R_target, p_target)
            Q, is_LS = ik(R_target, p_target)
            Q_exact = _exact(Q, is_LS)
            Q_ref   = _ref_exact_valid(model_inversek(T, PARAMS), p_target)
            if Q_exact.shape[1] == 0 or Q_ref.shape[1] == 0:
                continue
            found = any(
                _angle_match(Q_exact[:, i], Q_ref[:, j])
                for i in range(Q_exact.shape[1])
                for j in range(Q_ref.shape[1])
            )
            if found:
                n_found += 1
        assert n_found >= int(0.9 * n_tests), (
            f"Only {n_found}/{n_tests} poses had a matching solution"
        )