"""
Side-by-side trace: JAX general solver vs sympy (no CSE, fully numeric).
Both use the same test pose. Every intermediate value is printed.

Run:
    uv run python trace_jax_vs_sympy.py
"""
import numpy as np
import jax.numpy as jnp
import sympy as sp

ROBOT = "UR10e"
SEED  = 42

# ── robot setup ───────────────────────────────────────────────────────────────
from jaik.kinematics.robots import make_robot, _UR_PARAMS, _UR_R6T
from jaik.kinematics.convention_conversions import dh_to_kin
from jaik.kinematics.adjustments import adjust_kin_for_3p2i
from jaik._jax.subproblems import sp1 as jax_sp1
from jaik._jax.subproblems import sp3 as jax_sp3
from jaik._jax.subproblems import sp4 as jax_sp4
from jaik._jax.utils import _rot as jax_rot

p = _UR_PARAMS[ROBOT]
a = np.array([0, p["a2"], p["a3"], 0, 0, 0])
d = np.array([p["d1"], 0, 0, p["d4"], p["d5"], p["d6"]])
alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
kin = adjust_kin_for_3p2i(dh_to_kin(alpha, a, d))

H_np = kin['H']
P_np = kin['P']
H = jnp.array(H_np)
P = jnp.array(P_np)
RT = jnp.array(_UR_R6T)

fk_gen, _ = make_robot(ROBOT, solver="general")

# ── test pose ─────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
q_ref = jnp.array(rng.uniform(-np.pi, np.pi, 6))
R_tcp, p_tcp = fk_gen(q_ref)
R_06_jax = R_tcp @ RT.T
p_0T_jax = p_tcp

R_06_np = np.asarray(R_06_jax)
p_0T_np = np.asarray(p_0T_jax)

print(f"Test pose (seed={SEED}):")
print(f"  R_06 =\n{np.round(R_06_np, 4)}")
print(f"  p_0T = {np.round(p_0T_np, 4)}")
print()

# ── sympy setup — fully numeric (substitute pose values immediately) ───────────
def _fv(x):
    return sp.Float(float(x))

def _vec(arr):
    return sp.Matrix([_fv(arr[i]) for i in range(3)])

def _mat(arr):
    return sp.Matrix([[_fv(arr[i,j]) for j in range(arr.shape[1])]
                      for i in range(arr.shape[0])])

def _sym_rot(k_vec, theta):
    kx, ky, kz = k_vec[0], k_vec[1], k_vec[2]
    c = sp.cos(theta)
    s = sp.sin(theta)
    t = 1 - c
    return sp.Matrix([
        [t*kx*kx + c,    t*kx*ky - s*kz, t*kx*kz + s*ky],
        [t*kx*ky + s*kz, t*ky*ky + c,    t*ky*kz - s*kx],
        [t*kx*kz - s*ky, t*ky*kz + s*kx, t*kz*kz + c   ],
    ])

def _sp1_sym(p1, p2, k):
    KxP = k.cross(p1)
    x0  = KxP.dot(p2)
    x1  = (-k.cross(KxP)).dot(p2)
    return sp.atan2(x0, x1)

def _sp3_sym(p1, p2, k, d):
    KxP       = k.cross(p1)
    A1_r0     = KxP
    A1_r1     = -k.cross(KxP)
    A0        = -2 * p2.dot(A1_r0)
    A1v       = -2 * p2.dot(A1_r1)
    norm_A_sq = A0**2 + A1v**2
    norm_A    = sp.sqrt(norm_A_sq)
    p2_proj   = p2 - k * k.dot(p1)
    p2_proj_sq = p2_proj.dot(p2_proj)
    KxP_sq    = KxP.dot(KxP)
    b         = d**2 - p2_proj_sq - KxP_sq
    x_ls0     = A1_r0.dot(-2 * p2) * b / norm_A_sq
    x_ls1     = A1_r1.dot(-2 * p2) * b / norm_A_sq
    xi        = sp.sqrt(1 - b**2 / norm_A_sq)
    A_perp0   = A1v / norm_A
    A_perp1   = -A0 / norm_A
    sc1 = sp.Matrix([x_ls0 + xi * A_perp0, x_ls1 + xi * A_perp1])
    sc2 = sp.Matrix([x_ls0 - xi * A_perp0, x_ls1 - xi * A_perp1])
    return sp.atan2(sc1[0], sc1[1]), sp.atan2(sc2[0], sc2[1])

def _sp4_sym(p, k, h, d):
    A11    = k.cross(p)
    A1_r0  = A11
    A1_r1  = -k.cross(A11)
    A0     = h.dot(A1_r0)
    A1v    = h.dot(A1_r1)
    b      = d - h.dot(k) * k.dot(p)
    norm_A2 = A0**2 + A1v**2
    x_ls0  = A1_r0.dot(h) * b
    x_ls1  = A1_r1.dot(h) * b
    xi     = sp.sqrt(norm_A2 - b**2)
    sc1 = sp.Matrix([x_ls0 + xi * A1v,  x_ls1 + xi * (-A0)])
    sc2 = sp.Matrix([x_ls0 - xi * A1v,  x_ls1 - xi * (-A0)])
    return sp.atan2(sc1[0], sc1[1]), sp.atan2(sc2[0], sc2[1])

def _evalf(x):
    return float(x.evalf())

def _evalf_vec(v):
    return np.array([float(v[i].evalf()) for i in range(len(v))])

# sympy numeric matrices
R_06_sym = _mat(R_06_np)
p_0T_sym = _vec(p_0T_np)
H_sym = [_vec(H_np[:, j]) for j in range(6)]
P_sym = [_vec(P_np[:, j]) for j in range(7)]


# ── trace both solvers branch by branch ───────────────────────────────────────

def fmt(arr, d=4):
    return np.round(np.asarray(arr, dtype=float), d)

branch = 0
for i_q1 in range(2):
    print(f"\n{'='*70}")
    print(f"q1 branch {i_q1}")
    print(f"{'='*70}")

    # ── JAX ───────────────────────────────────────────────────────────────────
    p_06_jax = p_0T_jax - P[:, 0] - R_06_jax @ P[:, 6]
    d1_jax   = H[:, 1] @ P[:, 1:5].sum(axis=1)
    t1s_jax  = jax_sp4(p_06_jax, -H[:, 0], H[:, 1], d1_jax)
    q1_jax   = t1s_jax[i_q1]

    # ── SYMPY ─────────────────────────────────────────────────────────────────
    p_06_sym = p_0T_sym - P_sym[0] - R_06_sym * P_sym[6]
    d1_sym   = H_sym[1].dot(P_sym[1] + P_sym[2] + P_sym[3] + P_sym[4])
    t1s_sym  = _sp4_sym(p_06_sym, -H_sym[0], H_sym[1], d1_sym)
    q1_sym   = t1s_sym[i_q1]

    print(f"  p_06: JAX={fmt(p_06_jax)}  SYM={fmt(_evalf_vec(p_06_sym))}")
    print(f"  d1:   JAX={float(d1_jax):.6f}  SYM={_evalf(d1_sym):.6f}")
    print(f"  t1s:  JAX={fmt(t1s_jax)}  SYM=[{_evalf(t1s_sym[0]):.4f}, {_evalf(t1s_sym[1]):.4f}]")
    print(f"  q1:   JAX={float(q1_jax):.4f}  SYM={_evalf(q1_sym):.4f}")

    R_01_jax = jax_rot(H[:, 0], q1_jax)
    R_01_sym = _sym_rot(H_sym[0], q1_sym)

    d5_jax   = H[:, 1] @ R_01_jax.T @ R_06_jax @ H[:, 5]
    d5_sym   = H_sym[1].dot(R_01_sym.T * R_06_sym * H_sym[5])
    t5s_jax  = jax_sp4(H[:, 5], H[:, 4], H[:, 1], d5_jax)
    t5s_sym  = _sp4_sym(H_sym[5], H_sym[4], H_sym[1], d5_sym)

    print(f"  d5:   JAX={float(d5_jax):.6f}  SYM={_evalf(d5_sym):.6f}")
    print(f"  t5s:  JAX={fmt(t5s_jax)}  SYM=[{_evalf(t5s_sym[0]):.4f}, {_evalf(t5s_sym[1]):.4f}]")

    for i_q5 in range(2):
        print(f"\n  ── q5 branch {i_q5} ──")

        q5_jax = t5s_jax[i_q5]
        q5_sym = t5s_sym[i_q5]
        print(f"  q5:    JAX={float(q5_jax):.4f}  SYM={_evalf(q5_sym):.4f}")

        R_45_jax = jax_rot(H[:, 4], q5_jax)
        R_45_sym = _sym_rot(H_sym[4], q5_sym)

        th14_jax = jax_sp1(R_45_jax @ H[:, 5], R_01_jax.T @ R_06_jax @ H[:, 5], H[:, 1])
        th14_sym = _sp1_sym(R_45_sym * H_sym[5], R_01_sym.T * R_06_sym * H_sym[5], H_sym[1])

        q6_jax = jax_sp1(R_45_jax.T @ H[:, 1], R_06_jax.T @ R_01_jax @ H[:, 1], -H[:, 5])
        q6_sym = _sp1_sym(R_45_sym.T * H_sym[1], R_06_sym.T * R_01_sym * H_sym[1], -H_sym[5])

        print(f"  th14:  JAX={float(th14_jax):.4f}  SYM={_evalf(th14_sym):.4f}")
        print(f"  q6:    JAX={float(q6_jax):.4f}  SYM={_evalf(q6_sym):.4f}")

        d_inner_jax = R_01_jax.T @ p_06_jax - P[:, 1] - jax_rot(H[:, 1], th14_jax) @ P[:, 4]
        d_inner_sym = R_01_sym.T * p_06_sym - P_sym[1] - _sym_rot(H_sym[1], th14_sym) * P_sym[4]
        d3_jax = float(jnp.linalg.norm(d_inner_jax))
        d3_sym = float(sp.sqrt(d_inner_sym.dot(d_inner_sym)).evalf())

        print(f"  d_inner: JAX={fmt(d_inner_jax)}  SYM={fmt(_evalf_vec(d_inner_sym))}")
        print(f"  d3:    JAX={d3_jax:.6f}  SYM={d3_sym:.6f}")

        t3s_jax = jax_sp3(-P[:, 3], P[:, 2], H[:, 1], jnp.array(d3_jax))
        t3s_sym = _sp3_sym(-P_sym[3], P_sym[2], H_sym[1], sp.Float(d3_sym))

        print(f"  t3s:   JAX={fmt(t3s_jax)}  SYM=[{_evalf(t3s_sym[0]):.4f}, {_evalf(t3s_sym[1]):.4f}]")

        for i_q3 in range(2):
            q3_jax = float(t3s_jax[i_q3])
            q3_sym = _evalf(t3s_sym[i_q3])

            p1_q2_jax = P[:, 2] + jax_rot(H[:, 1], jnp.array(q3_jax)) @ P[:, 3]
            p1_q2_sym = P_sym[2] + _sym_rot(H_sym[1], sp.Float(q3_sym)) * P_sym[3]

            q2_jax = float(jax_sp1(p1_q2_jax, d_inner_jax, H[:, 1]))
            q2_sym = _evalf(_sp1_sym(p1_q2_sym, d_inner_sym, H_sym[1]))

            q4_jax = float((th14_jax - q2_jax - q3_jax + jnp.pi) % (2*jnp.pi) - jnp.pi)
            q4_sym = ((_evalf(th14_sym) - q2_sym - q3_sym + np.pi) % (2*np.pi)) - np.pi

            print(f"\n    ── q3 branch {i_q3} ──")
            print(f"    q3:  JAX={q3_jax:.4f}  SYM={q3_sym:.4f}  {'✓' if abs(q3_jax-q3_sym)<1e-3 else '✗'}")
            print(f"    q2:  JAX={q2_jax:.4f}  SYM={q2_sym:.4f}  {'✓' if abs(q2_jax-q2_sym)<1e-3 else '✗'}")
            print(f"    q4:  JAX={q4_jax:.4f}  SYM={q4_sym:.4f}  {'✓' if abs(q4_jax-q4_sym)<1e-3 else '✗'}")
            print(f"    branch {branch}: JAX=[{q1_jax:.3f},{q2_jax:.3f},{q3_jax:.3f},{q4_jax:.3f},{float(q5_jax):.3f},{float(q6_jax):.3f}]")
            print(f"                    SYM=[{_evalf(q1_sym):.3f},{q2_sym:.3f},{q3_sym:.3f},{q4_sym:.3f},{_evalf(q5_sym):.3f},{_evalf(q6_sym):.3f}]")

            branch += 1