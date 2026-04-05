# tests/test_jax_subproblems.py
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jaik._jax.subproblems import sp1, sp3, sp4
from jaik._jax.utils import _rot

# ── helpers ───────────────────────────────────────────────────────────────────

def rand_unit(rng):
    v = rng.standard_normal(3)
    return jnp.array(v / np.linalg.norm(v))


def rand_vec(rng):
    return jnp.array(rng.standard_normal(3))


def rand_scalar(rng, low=-np.pi, high=np.pi):
    return jnp.array(rng.uniform(low, high))


# ── SP1 ───────────────────────────────────────────────────────────────────────

class TestSP1:

    def test_exact_solution(self):
        """sp1 should recover the angle used to generate p2."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            k = rand_unit(rng)
            p1 = rand_vec(rng)
            theta_true = rand_scalar(rng)
            p2 = _rot(k, theta_true) @ p1

            theta, is_LS = sp1(p1, p2, k)

            assert not is_LS
            np.testing.assert_allclose(
                _rot(k, theta) @ p1, p2, atol=1e-6
            )

    def test_identity(self):
        """p1 == p2 should give a valid theta."""
        rng = np.random.default_rng(1)
        k = rand_unit(rng)
        p1 = rand_vec(rng)
        p1 = p1 - jnp.dot(p1, k) * k
        theta, is_LS = sp1(p1, p1, k)
        np.testing.assert_allclose(_rot(k, theta) @ p1, p1, atol=1e-6)

    def test_ls_flag_when_norms_differ(self):
        rng = np.random.default_rng(2)
        k = rand_unit(rng)
        p1 = rand_vec(rng)
        p2 = 2.0 * p1
        _, is_LS = sp1(p1, p2, k)
        assert is_LS

    def test_return_shapes(self):
        """sp1 always returns scalar theta and scalar bool."""
        rng = np.random.default_rng(3)
        k = rand_unit(rng)
        p1 = rand_vec(rng)
        p2 = rand_vec(rng)
        theta, is_LS = sp1(p1, p2, k)
        assert theta.shape == ()
        assert is_LS.shape == ()

    def test_jit_compatible(self):
        """sp1 should work under jit."""
        sp1_jit = jax.jit(sp1)
        rng = np.random.default_rng(4)
        k = rand_unit(rng)
        p1 = rand_vec(rng)
        theta_true = rand_scalar(rng)
        p2 = _rot(k, theta_true) @ p1
        theta, is_LS = sp1_jit(p1, p2, k)
        np.testing.assert_allclose(_rot(k, theta) @ p1, p2, atol=1e-6)

    def test_vmap_compatible(self):
        """sp1 should work under vmap over a batch of problems."""
        rng = np.random.default_rng(5)
        n = 10
        ks   = jnp.array([rand_unit(rng) for _ in range(n)])
        p1s  = jnp.array([rand_vec(rng) for _ in range(n)])
        ts   = jnp.array([float(rand_scalar(rng)) for _ in range(n)])
        p2s  = jax.vmap(lambda k, p, t: _rot(k, t) @ p)(ks, p1s, ts)

        thetas, is_LSs = jax.vmap(sp1)(p1s, p2s, ks)

        assert thetas.shape == (n,)
        assert is_LSs.shape == (n,)
        assert not is_LSs.any()


# ── SP3 ───────────────────────────────────────────────────────────────────────

class TestSP3:

    def test_exact_solution(self):
        """sp3 should find theta such that |R(k,theta)·p - q| = d."""
        rng = np.random.default_rng(6)
        for _ in range(20):
            k = rand_unit(rng)
            p = rand_vec(rng)
            q = rand_vec(rng)
            theta_true = rand_scalar(rng)
            d = jnp.linalg.norm(_rot(k, theta_true) @ p - q)

            thetas, is_LS = sp3(p, q, k, d)

            assert not is_LS
            found = any(
                abs(float(jnp.linalg.norm(_rot(k, t) @ p - q)) - float(d)) < 1e-6
                for t in thetas
                if not jnp.isnan(t)
            )
            assert found, f"sp3: no solution satisfies |R(k,t)·p - q| = d"

    def test_ls_when_unreachable(self):
        rng = np.random.default_rng(7)
        k = rand_unit(rng)
        p = rand_vec(rng)
        q = rand_vec(rng)
        d = jnp.array(float(jnp.linalg.norm(p) + jnp.linalg.norm(q)) + 10.0)
        _, is_LS = sp3(p, q, k, d)
        assert is_LS

    def test_return_shape_always_2(self):
        """sp3 always returns (2,) regardless of exact or LS."""
        rng = np.random.default_rng(8)
        k = rand_unit(rng)
        p = rand_vec(rng)
        q = rand_vec(rng)

        # exact case
        theta_true = rand_scalar(rng)
        d_exact = jnp.linalg.norm(_rot(k, theta_true) @ p - q)
        thetas_exact, _ = sp3(p, q, k, d_exact)
        assert thetas_exact.shape == (2,)

        # LS case
        d_ls = jnp.array(float(jnp.linalg.norm(p) + jnp.linalg.norm(q)) + 10.0)
        thetas_ls, _ = sp3(p, q, k, d_ls)
        assert thetas_ls.shape == (2,)

    def test_ls_branch_has_nan(self):
        """LS branch should have NaN in second slot."""
        rng = np.random.default_rng(9)
        k = rand_unit(rng)
        p = rand_vec(rng)
        q = rand_vec(rng)
        d = jnp.array(float(jnp.linalg.norm(p) + jnp.linalg.norm(q)) + 10.0)
        thetas, is_LS = sp3(p, q, k, d)
        assert is_LS
        assert not jnp.isnan(thetas[0]), "First slot should be valid LS angle"
        assert jnp.isnan(thetas[1]),     "Second slot should be NaN for LS"

    def test_jit_compatible(self):
        sp3_jit = jax.jit(sp3)
        rng = np.random.default_rng(10)
        k = rand_unit(rng)
        p = rand_vec(rng)
        q = rand_vec(rng)
        theta_true = rand_scalar(rng)
        d = jnp.linalg.norm(_rot(k, theta_true) @ p - q)
        thetas, is_LS = sp3_jit(p, q, k, d)
        assert thetas.shape == (2,)
        assert not is_LS

    def test_vmap_compatible(self):
        rng = np.random.default_rng(11)
        n = 10
        ks  = jnp.array([rand_unit(rng) for _ in range(n)])
        ps  = jnp.array([rand_vec(rng) for _ in range(n)])
        qs  = jnp.array([rand_vec(rng) for _ in range(n)])
        ts  = jnp.array([float(rand_scalar(rng)) for _ in range(n)])
        ds  = jax.vmap(lambda k, p, q, t:
                       jnp.linalg.norm(_rot(k, t) @ p - q))(ks, ps, qs, ts)

        thetas, is_LSs = jax.vmap(sp3)(ps, qs, ks, ds)
        assert thetas.shape == (n, 2)
        assert is_LSs.shape == (n,)
        assert not is_LSs.any()


# ── SP4 ───────────────────────────────────────────────────────────────────────

class TestSP4:

    def test_exact_solution(self):
        """sp4 should recover angle used to generate d."""
        rng = np.random.default_rng(12)
        for _ in range(20):
            h = rand_unit(rng)
            k = rand_unit(rng)
            p = rand_vec(rng)
            theta_true = rand_scalar(rng)
            d = h @ _rot(k, theta_true) @ p

            thetas, is_LS = sp4(p, k, h, d)

            assert not is_LS
            found = any(
                abs(float(h @ _rot(k, t) @ p) - float(d)) < 1e-6
                for t in thetas
                if not jnp.isnan(t)
            )
            assert found, f"sp4: no solution satisfies h·R(k,t)·p = d"

    def test_ls_when_d_out_of_range(self):
        rng = np.random.default_rng(13)
        h = rand_unit(rng)
        k = rand_unit(rng)
        p = rand_vec(rng)
        d = jnp.array(1e6)
        _, is_LS = sp4(p, k, h, d)
        assert is_LS

    def test_return_shape_always_2(self):
        """sp4 always returns (2,) regardless of exact or LS."""
        rng = np.random.default_rng(14)
        h = rand_unit(rng)
        k = rand_unit(rng)
        p = rand_vec(rng)

        # exact case
        theta_true = rand_scalar(rng)
        d_exact = h @ _rot(k, theta_true) @ p
        thetas_exact, _ = sp4(p, k, h, d_exact)
        assert thetas_exact.shape == (2,)

        # LS case
        thetas_ls, _ = sp4(p, k, h, jnp.array(1e6))
        assert thetas_ls.shape == (2,)

    def test_ls_branch_has_nan(self):
        """LS branch should have NaN in second slot."""
        rng = np.random.default_rng(15)
        h = rand_unit(rng)
        k = rand_unit(rng)
        p = rand_vec(rng)
        thetas, is_LS = sp4(p, k, h, jnp.array(1e6))
        assert is_LS
        assert not jnp.isnan(thetas[0]), "First slot should be valid LS angle"
        assert jnp.isnan(thetas[1]),     "Second slot should be NaN for LS"

    def test_jit_compatible(self):
        sp4_jit = jax.jit(sp4)
        rng = np.random.default_rng(16)
        h = rand_unit(rng)
        k = rand_unit(rng)
        p = rand_vec(rng)
        theta_true = rand_scalar(rng)
        d = h @ _rot(k, theta_true) @ p
        thetas, is_LS = sp4_jit(p, k, h, d)
        assert thetas.shape == (2,)
        assert not is_LS

    def test_vmap_compatible(self):
        rng = np.random.default_rng(17)
        n = 10
        hs  = jnp.array([rand_unit(rng) for _ in range(n)])
        ks  = jnp.array([rand_unit(rng) for _ in range(n)])
        ps  = jnp.array([rand_vec(rng) for _ in range(n)])
        ts  = jnp.array([float(rand_scalar(rng)) for _ in range(n)])
        ds  = jax.vmap(lambda h, k, p, t: h @ _rot(k, t) @ p)(hs, ks, ps, ts)

        thetas, is_LSs = jax.vmap(sp4)(ps, ks, hs, ds)
        assert thetas.shape == (n, 2)
        assert is_LSs.shape == (n,)
        assert not is_LSs.any()

    def test_matches_numpy_sp4(self):
        """JAX sp4 should agree with numpy sp4 on exact solutions."""
        import numpy as np_cpu
        from jaik._numpy.subproblems import sp4 as sp4_np
        from jaik._numpy.utils import _rot as _rot_np

        rng = np.random.default_rng(18)
        for _ in range(20):
            h_np = rng.standard_normal(3)
            h_np /= np.linalg.norm(h_np)
            k_np = rng.standard_normal(3)
            k_np /= np.linalg.norm(k_np)
            p_np = rng.standard_normal(3)
            t_np = rng.uniform(-np.pi, np.pi)
            d_np = h_np @ _rot_np(k_np, t_np) @ p_np

            thetas_np, is_LS_np = sp4_np(p_np, k_np, h_np, float(d_np))
            thetas_jx, is_LS_jx = sp4(
                jnp.array(p_np), jnp.array(k_np),
                jnp.array(h_np), jnp.array(d_np)
            )

            assert bool(is_LS_np) == bool(is_LS_jx)
            np.testing.assert_allclose(
                np.sort(np.array(thetas_np)),
                np.sort(np.array(thetas_jx[~jnp.isnan(thetas_jx)])),
                atol=1e-6
            )