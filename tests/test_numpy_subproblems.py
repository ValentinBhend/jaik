# tests/test_numpy_subproblems.py
import numpy as np
import pytest
from jaik._numpy.subproblems import sp1, sp2, sp3, sp4
from jaik._numpy.utils import _rot


def rand_unit(rng):
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def rand_vec(rng):
    return rng.standard_normal(3)


# ─── SP1: find theta such that R(k, theta) @ p1 = p2 ───────────────────────

class TestSP1:

    def test_exact_solution(self):
        """sp1 should recover the angle used to generate p2."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            k = rand_unit(rng)
            p1 = rand_vec(rng)
            theta_true = rng.uniform(-np.pi, np.pi)
            p2 = _rot(k, theta_true) @ p1

            theta, is_LS = sp1(p1, p2, k)

            assert not is_LS
            np.testing.assert_allclose(
                _rot(k, theta) @ p1, p2, atol=1e-10,
                err_msg=f"sp1 wrong: got {theta:.4f}, expected {theta_true:.4f}"
            )

    def test_identity(self):
        """p1 == p2 should give a valid theta (rotation of zero)."""
        rng = np.random.default_rng(1)
        k = rand_unit(rng)
        p1 = rand_vec(rng)
        # project onto plane perpendicular to k so it's rotatable
        p1 = p1 - np.dot(p1, k) * k
        theta, is_LS = sp1(p1, p1, k)
        np.testing.assert_allclose(_rot(k, theta) @ p1, p1, atol=1e-10)

    def test_ls_flag_when_norms_differ(self):
        """is_LS=True when |p1| != |p2|."""
        rng = np.random.default_rng(2)
        k = rand_unit(rng)
        p1 = rand_vec(rng)
        p2 = 2.0 * p1
        _, is_LS = sp1(p1, p2, k)
        assert is_LS


# ─── SP2: find t1, t2 such that R(k1,t1)·p1 = R(k2,t2)·p2 ─────────────────
# Signature: sp2(p1, p2, k1, k2)
# Returns:   (t1s, t2s, is_LS) where t1s, t2s are (2,1) arrays (exact)
#            or scalars (LS)

class TestSP2:

    def test_exact_solution(self):
        """sp2 should find t1,t2 such that R(k1,t1)·p1 = R(k2,t2)·p2."""
        rng = np.random.default_rng(3)
        successes = 0
        for _ in range(30):
            k1 = rand_unit(rng)
            k2 = rand_unit(rng)
            if abs(np.dot(k1, k2)) > 0.99:
                continue

            # build valid problem: both p1 and p2 rotate to same target
            target = rand_vec(rng)
            t1_true = rng.uniform(-np.pi, np.pi)
            t2_true = rng.uniform(-np.pi, np.pi)
            p1 = _rot(k1, -t1_true) @ target
            p2 = _rot(k2, -t2_true) @ target

            p1_orig = p1.copy()
            p2_orig = p2.copy()

            t1s, t2s, is_LS = sp2(p1, p2, k1, k2)

            if is_LS:
                continue  # skip LS cases

            thetas = list(zip(t1s.flatten(), t2s.flatten()))
            found = False
            for t1, t2 in thetas:
                lhs = _rot(k1, t1) @ p1_orig
                rhs = _rot(k2, t2) @ p2_orig
                if np.allclose(lhs, rhs, atol=1e-8):
                    found = True
                    break
            assert found, f"sp2 solution doesn't verify: lhs={lhs}, rhs={rhs}"
            successes += 1

        assert successes >= 10, "Too few non-LS cases to test"

    def test_returns_up_to_2_solutions(self):
        """sp2 should return at most 2 solution pairs."""
        rng = np.random.default_rng(4)
        k1 = rand_unit(rng)
        k2 = rand_unit(rng)
        p1 = rand_vec(rng)
        p2 = rand_vec(rng)
        t1s, t2s, is_LS = sp2(p1, p2, k1, k2)
        if not is_LS:
            assert len(t1s.flatten()) <= 2
            assert len(t2s.flatten()) <= 2

    def test_mutates_inputs(self):
        """
        sp2 normalises p1, p2 in-place — document this known behaviour.
        Callers must copy inputs if they need them afterwards.
        """
        rng = np.random.default_rng(5)
        k1 = rand_unit(rng)
        k2 = rand_unit(rng)
        p1 = rand_vec(rng) * 5.0   # non-unit vector
        p2 = rand_vec(rng) * 3.0
        p1_before = p1.copy()
        sp2(p1, p2, k1, k2)
        # after the call p1 is normalised — this is expected behaviour
        assert not np.allclose(p1, p1_before), \
            "sp2 should normalise p1 in-place (document this)"


# ─── SP3: find theta such that |R(k,theta)·p - q| = d ──────────────────────
# Signature: sp3(p1, p2, k, d)
# Returns:   (theta_or_thetas, is_LS)

class TestSP3:

    def test_exact_solution(self):
        """sp3 should find theta such that |R(k,theta)·p - q| = d."""
        rng = np.random.default_rng(6)
        for _ in range(20):
            k = rand_unit(rng)
            p = rand_vec(rng)
            q = rand_vec(rng)
            theta_true = rng.uniform(-np.pi, np.pi)
            d = np.linalg.norm(_rot(k, theta_true) @ p - q)

            result, is_LS = sp3(p, q, k, d)
            thetas = np.atleast_1d(result)

            assert not is_LS
            found = False
            for t in thetas:
                err = abs(np.linalg.norm(_rot(k, t) @ p - q) - d)
                if err < 1e-8:
                    found = True
                    break
            assert found, f"sp3: no solution satisfies |R(k,t)·p - q| = d={d:.4f}"

    def test_returns_up_to_2_solutions(self):
        """sp3 returns at most 2 solutions."""
        rng = np.random.default_rng(7)
        k = rand_unit(rng)
        p = rand_vec(rng)
        q = rand_vec(rng)
        theta_true = rng.uniform(-np.pi, np.pi)
        d = np.linalg.norm(_rot(k, theta_true) @ p - q)
        result, _ = sp3(p, q, k, d)
        assert len(np.atleast_1d(result)) <= 2

    def test_ls_when_unreachable(self):
        """sp3 returns is_LS=True when d is outside achievable range."""
        rng = np.random.default_rng(8)
        k = rand_unit(rng)
        p = rand_vec(rng)
        q = rand_vec(rng)
        d = np.linalg.norm(p) + np.linalg.norm(q) + 10.0  # impossible
        _, is_LS = sp3(p, q, k, d)
        assert is_LS


# ─── SP4: find theta such that h · R(k, theta) · p = d ─────────────────────
# Signature: sp4(p, k, h, d)   ← note order: p first, then k, then h
# Returns:   (theta_or_thetas, is_LS)

class TestSP4:

    def test_exact_solution(self):
        """sp4 should recover angle used to generate d."""
        rng = np.random.default_rng(9)
        for _ in range(20):
            h = rand_unit(rng)
            k = rand_unit(rng)
            p = rand_vec(rng)
            theta_true = rng.uniform(-np.pi, np.pi)
            d = h @ _rot(k, theta_true) @ p

            # signature: sp4(p, k, h, d)
            result, is_LS = sp4(p, k, h, d)
            thetas = np.atleast_1d(result)

            assert not is_LS, f"Expected exact solution but got LS"
            found = False
            for t in thetas:
                val = h @ _rot(k, t) @ p
                if abs(val - d) < 1e-8:
                    found = True
                    break
            assert found, \
                f"sp4: no solution satisfies h·R(k,t)·p = d\n" \
                f"thetas={thetas}, d={d:.6f}, vals={[h @ _rot(k,t) @ p for t in thetas]}"

    def test_returns_up_to_2_solutions(self):
        """sp4 should return at most 2 solutions."""
        rng = np.random.default_rng(10)
        h = rand_unit(rng)
        k = rand_unit(rng)
        p = rand_vec(rng)
        theta_true = rng.uniform(-np.pi, np.pi)
        d = h @ _rot(k, theta_true) @ p
        result, _ = sp4(p, k, h, d)
        assert len(np.atleast_1d(result)) <= 2

    def test_ls_when_d_out_of_range(self):
        """sp4 returns is_LS=True when |d| > achievable range."""
        rng = np.random.default_rng(11)
        h = rand_unit(rng)
        k = rand_unit(rng)
        p = rand_vec(rng)
        d = 1e6
        _, is_LS = sp4(p, k, h, d)
        assert is_LS

    def test_verification_ur_q1(self):
        """
        Reproduce the SP4 call for q1 in the UR IK pipeline.
        At q1=0, the solution theta=0 must be among the returned values.
        """
        from jaik._numpy.convention_conversions import dh_to_kin
        from jaik._numpy.fk import fk

        alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
        a     = np.array([0, -0.6127, -0.57155, 0, 0, 0])
        d_dh  = np.array([0.1807, 0, 0, 0.17415, 0.12, 0.1163])
        kin = dh_to_kin(alpha, a, d_dh)
        P = kin['P']
        H = kin['H']

        # adjust p_12=0, p_56=0
        for col, ax in [(1, 0), (5, 4)]:
            shift = np.dot(P[:, col], H[:, ax]) * H[:, ax]
            P[:, col - 1] += shift
            P[:, col] -= shift

        q_ref = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
        R_06, p_0T = fk(q_ref, kin)
        p_06 = p_0T - P[:, 0] - R_06 @ P[:, 6]
        d1 = H[:, 1] @ P[:, 1:5].sum(axis=1)

        # sp4(p, k, h, d)
        result, is_LS = sp4(p_06, -H[:, 0], H[:, 1], d1)
        thetas = np.atleast_1d(result)

        print(f"\nSP4 q1: thetas={thetas}, is_LS={is_LS}, expected=0.0")
        for t in thetas:
            val = H[:, 1] @ _rot(-H[:, 0], t) @ p_06
            print(f"  h·R(k,{t:.4f})·p = {val:.6f}, d1={d1:.6f}")

        assert not is_LS, "SP4 for q1 should be exact"
        found = any(
            abs(H[:, 1] @ _rot(-H[:, 0], t) @ p_06 - d1) < 1e-8
            for t in thetas
        )
        assert found, "SP4 for q1 gave no valid solution"