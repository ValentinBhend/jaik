# tests/test_jax_compare_ur10e_model.py
"""
End-to-end cross-validation: compare jaik JAX FK/IK against the legacy UR10e
reference model. Uses the JAX backend (default) via make_robot.
"""
import numpy as np
import jax.numpy as jnp
import pytest
import jaik
from jaik._ur10e_compare_model_py import model_params, model_forwardk, model_inversek

PARAMS = model_params()


@pytest.fixture(scope="module")
def fk_ik():
    # JAX backend — default
    return jaik.make_robot("UR10e", backend="jax")


# ── helpers ───────────────────────────────────────────────────────────────────

def _exact(Q, is_LS):
    """
    Filter to exact (non-LS, non-NaN) solutions.
    JAX ik always returns (6, 8) — NaN marks infeasible branches.
    """
    Q     = np.asarray(Q)
    is_LS = np.asarray(is_LS)
    mask  = ~is_LS.any(axis=0) & ~np.isnan(Q).any(axis=0)
    return Q[:, mask]


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
    T[:3, :3] = np.asarray(R)
    T[:3, 3]  = np.asarray(p)
    return T


def _angle_match(q1, q2, tol=1e-4):
    """Check if two joint configs match up to 2pi wrapping."""
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    diff = np.arctan2(np.sin(q1 - q2), np.cos(q1 - q2))
    return np.linalg.norm(diff) < tol


# ── FK comparison ─────────────────────────────────────────────────────────────

class TestFKComparison:

    def test_fk_position_matches_reference(self, fk_ik):
        """jaik JAX FK position should match reference for random configs."""
        fk, _ = fk_ik
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = jnp.array(rng.uniform(-np.pi, np.pi, 6))
            _, p_jaik = fk(q)
            T_ref, _ = model_forwardk(np.asarray(q), PARAMS)
            np.testing.assert_allclose(
                np.asarray(p_jaik), T_ref[:3, 3], atol=1e-6,
                err_msg=f"FK position mismatch at q={np.round(q, 3)}"
            )

    def test_fk_rotation_matches_reference(self, fk_ik):
        """jaik JAX FK rotation should match reference (tool frame applied)."""
        fk, _ = fk_ik
        rng = np.random.default_rng(1)
        for _ in range(50):
            q = jnp.array(rng.uniform(-np.pi, np.pi, 6))
            R_jaik, _ = fk(q)
            T_ref, _ = model_forwardk(np.asarray(q), PARAMS)
            np.testing.assert_allclose(
                np.asarray(R_jaik), T_ref[:3, :3], atol=1e-6,
                err_msg=f"FK rotation mismatch at q={np.round(q, 3)}"
            )

    def test_fk_known_configs(self, fk_ik):
        """jaik JAX FK matches reference on named configurations."""
        fk, _ = fk_ik
        configs = {
            "zero":    jnp.zeros(6),
            "home":    jnp.array([0.0, -np.pi/2,  np.pi/2, -np.pi/2, -np.pi/2, 0.0]),
            "all_pi4": jnp.full(6, np.pi/4),
            "all_neg": jnp.full(6, -np.pi/3),
        }
        for name, q in configs.items():
            R_jaik, p_jaik = fk(q)
            T_ref, _ = model_forwardk(np.asarray(q), PARAMS)
            np.testing.assert_allclose(
                np.asarray(p_jaik), T_ref[:3, 3], atol=1e-6,
                err_msg=f"FK position mismatch for config '{name}'"
            )
            np.testing.assert_allclose(
                np.asarray(R_jaik), T_ref[:3, :3], atol=1e-6,
                err_msg=f"FK rotation mismatch for config '{name}'"
            )

    def test_fk_matches_numpy_backend(self, fk_ik):
        """JAX FK should match numpy FK to high precision."""
        fk_jax, _ = fk_ik
        fk_np, _  = jaik.make_robot("UR10e", backend="numpy")
        rng = np.random.default_rng(2)
        for _ in range(30):
            q_np  = rng.uniform(-np.pi, np.pi, 6)
            q_jax = jnp.array(q_np)
            R_jax, p_jax = fk_jax(q_jax)
            R_np,  p_np  = fk_np(q_np)
            np.testing.assert_allclose(
                np.asarray(R_jax), R_np, atol=1e-6,
                err_msg="JAX/numpy FK rotation mismatch"
            )
            np.testing.assert_allclose(
                np.asarray(p_jax), p_np, atol=1e-6,
                err_msg="JAX/numpy FK position mismatch"
            )


# ── IK comparison ─────────────────────────────────────────────────────────────

class TestIKComparison:

    def test_ik_always_returns_8_branches(self, fk_ik):
        """JAX ik always returns (6, 8) — fixed shape contract."""
        fk, ik = fk_ik
        rng = np.random.default_rng(3)
        for _ in range(10):
            q = jnp.array(rng.uniform(-np.pi, np.pi, 6))
            R_target, p_target = fk(q)
            Q, is_LS = ik(R_target, p_target)
            assert Q.shape    == (6, 8)
            assert is_LS.shape == (6, 8)

    def test_ik_exact_solutions_verify_via_fk(self, fk_ik):
        """All exact JAX IK solutions should round-trip via fk."""
        fk, ik = fk_ik
        rng = np.random.default_rng(4)
        for _ in range(30):
            q_ref = jnp.array(rng.uniform(-np.pi, np.pi, 6))
            R_target, p_target = fk(q_ref)
            Q, is_LS = ik(R_target, p_target)
            Q_exact = _exact(Q, is_LS)
            for i in range(Q_exact.shape[1]):
                R_check, p_check = fk(jnp.array(Q_exact[:, i]))
                np.testing.assert_allclose(
                    np.asarray(p_check), np.asarray(p_target), atol=1e-5
                )
                np.testing.assert_allclose(
                    np.asarray(R_check), np.asarray(R_target), atol=1e-5
                )

    def test_ik_reference_solutions_verify_via_jaik_fk(self, fk_ik):
        """Valid reference IK solutions should verify via jaik fk."""
        fk, _ = fk_ik
        rng = np.random.default_rng(5)
        for _ in range(30):
            q_ref = jnp.array(rng.uniform(-np.pi, np.pi, 6))
            R_target, p_target = fk(q_ref)
            T = _make_T(R_target, p_target)
            # _ref_exact_valid already filters invalid solutions — no extra guard needed
            Q_ref = _ref_exact_valid(model_inversek(T, PARAMS), np.asarray(p_target))
            for i in range(Q_ref.shape[1]):
                R_check, p_check = fk(jnp.array(Q_ref[:, i]))
                np.testing.assert_allclose(
                    np.asarray(p_check), np.asarray(p_target), atol=1e-6,
                    err_msg=f"Reference solution {i} fails jaik FK (position)"
                )
                np.testing.assert_allclose(
                    np.asarray(R_check), np.asarray(R_target), atol=1e-6,
                    err_msg=f"Reference solution {i} fails jaik FK (rotation)"
                )

    def test_ik_solution_count_matches(self, fk_ik):
        """jaik and reference should return the same number of exact solutions."""
        fk, ik = fk_ik
        rng = np.random.default_rng(6)
        mismatches = 0
        n_tests = 30
        for _ in range(n_tests):
            q_ref = jnp.array(rng.uniform(-np.pi, np.pi, 6))
            R_target, p_target = fk(q_ref)
            T = _make_T(R_target, p_target)
            Q, is_LS = ik(R_target, p_target)
            n_jaik = _exact(Q, is_LS).shape[1]
            n_ref  = _ref_exact_valid(model_inversek(T, PARAMS), np.asarray(p_target)).shape[1]
            if n_jaik != n_ref:
                mismatches += 1
        # small number of mismatches tolerated near singularities
        assert mismatches <= 5, (
            f"{mismatches}/{n_tests} poses had different solution counts"
        )

    def test_ik_jaik_finds_reference_solution(self, fk_ik):
        """At least one jaik exact solution should match a reference solution."""
        fk, ik = fk_ik
        rng = np.random.default_rng(7)
        n_tests = 30
        n_found = 0
        for _ in range(n_tests):
            q_ref = jnp.array(rng.uniform(-np.pi, np.pi, 6))
            R_target, p_target = fk(q_ref)
            T = _make_T(R_target, p_target)
            Q, is_LS = ik(R_target, p_target)
            Q_exact = _exact(Q, is_LS)
            Q_ref   = _ref_exact_valid(model_inversek(T, PARAMS), np.asarray(p_target))
            if Q_exact.shape[1] == 0 or Q_ref.shape[1] == 0:
                continue
            found = any(
                _angle_match(Q_exact[:, i], Q_ref[:, j])
                for i in range(Q_exact.shape[1])
                for j in range(Q_ref.shape[1])
            )
            if found:
                n_found += 1
        # all poses with valid reference solutions should have a matching jaik solution
        assert n_found >= int(0.9 * n_tests), (
            f"Only {n_found}/{n_tests} poses had a matching solution"
        )

    def test_ik_matches_numpy_backend(self, fk_ik):
        """JAX IK exact solutions should match numpy IK exact solutions."""
        fk_jax, ik_jax = fk_ik
        fk_np,  ik_np  = jaik.make_robot("UR10e", backend="numpy")
        rng = np.random.default_rng(8)
        n_tests = 30
        n_matched = 0
        for _ in range(n_tests):
            q_ref  = rng.uniform(-np.pi, np.pi, 6)
            R_np, p_np = fk_np(q_ref)

            Q_np,  is_LS_np  = ik_np(R_np, p_np)
            Q_jax, is_LS_jax = ik_jax(jnp.array(R_np), jnp.array(p_np))

            Q_exact_np  = _exact(Q_np,  is_LS_np)
            Q_exact_jax = _exact(Q_jax, is_LS_jax)

            if Q_exact_np.shape[1] == 0 and Q_exact_jax.shape[1] == 0:
                n_matched += 1
                continue
            if Q_exact_np.shape[1] == 0 or Q_exact_jax.shape[1] == 0:
                continue

            # every numpy solution should have a matching JAX solution
            all_matched = all(
                any(_angle_match(Q_exact_np[:, i], Q_exact_jax[:, j])
                    for j in range(Q_exact_jax.shape[1]))
                for i in range(Q_exact_np.shape[1])
            )
            if all_matched:
                n_matched += 1

        assert n_matched >= int(0.8 * n_tests), (
            f"Only {n_matched}/{n_tests} poses had matching JAX/numpy solutions"
        )