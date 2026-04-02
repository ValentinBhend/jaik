# src/jaik/kinematics/codegen.py
"""
Code generator: derives 3p2i IK symbolically, applies CSE per stage,
and emits a fast JAX file for a specific robot.

Run:
    uv run python -m jaik.kinematics.codegen UR10e
    uv run python -m jaik.kinematics.codegen UR5e UR3e

Generated files:
    src/jaik/_jax/_generated/ik_<name_lower>.py

Design:
    - H and P entries are substituted as exact sympy rationals immediately,
      so zero entries become exactly 0 and are eliminated algebraically.
    - Intermediate symbols are introduced at each stage boundary
      (t1, t5, theta_14, q6, d_inner, d3, t3) to prevent expression blowup.
    - CSE is applied per stage. The emitted code is a flat sequence of
      assignments with no branches — NaN propagates naturally for
      infeasible branches.
    - No trigsimp or simplify — only sp.cse(optimizations='basic').
"""
import sys
import time
import textwrap
from pathlib import Path
from datetime import date
import importlib

import numpy as np
import sympy as sp
from sympy.printing.pycode import PythonCodePrinter
import jax.numpy as jnp

from jaik.kinematics.convention_conversions import dh_to_kin
from jaik.kinematics.adjustments import adjust_kin_for_3p2i
from jaik.kinematics.robots import make_robot, _UR_PARAMS, _UR_R6T

# ── sympy helpers ─────────────────────────────────────────────────────────────

def _rat(x):
    """Convert float to sympy Float — keeps reasonable precision without huge rationals."""
    return sp.Float(x)


def _vec(arr):
    """numpy 3-vector → sympy Matrix column, entries as rationals."""
    return sp.Matrix([_rat(arr[i]) for i in range(3)])


def _mat(arr):
    """numpy 3x3 array → sympy Matrix, entries as rationals."""
    return sp.Matrix([[_rat(arr[i, j]) for j in range(arr.shape[1])]
                      for i in range(arr.shape[0])])

def _clean_hp(arr, tol=1e-10):
    """
    Round near-zero and near-unit entries to exact values.
    UR joint axes are exactly aligned with coordinate axes — any deviation
    from 0/±1 is purely floating point noise from DH→PoE conversion.
    """
    result = arr.copy().astype(float)
    result[np.abs(result) < tol] = 0.0
    result[np.abs(result - 1.0) < tol] = 1.0
    result[np.abs(result + 1.0) < tol] = -1.0
    return result

def _sym_rot(k_vec, theta):
    """Rodrigues rotation: k_vec is sympy column vector, theta is sympy expr."""
    kx, ky, kz = k_vec
    c = sp.cos(theta)
    s = sp.sin(theta)
    t = 1 - c
    return sp.Matrix([
        [t*kx*kx + c,    t*kx*ky - s*kz, t*kx*kz + s*ky],
        [t*kx*ky + s*kz, t*ky*ky + c,    t*ky*kz - s*kx],
        [t*kx*kz - s*ky, t*ky*kz + s*kx, t*kz*kz + c   ],
    ])


def _sp1(p1, p2, k):
    """Direct translation of JAX sp1."""
    KxP = k.cross(p1)
    x0  = KxP.dot(p2)
    x1  = (-k.cross(KxP)).dot(p2)
    return sp.atan2(x0, x1)


def _sp3(p1, p2, k, d):
    """Direct translation of JAX sp3 — no branching, NaN propagates."""
    KxP      = k.cross(p1)
    A1_r0    = KxP
    A1_r1    = -k.cross(KxP)
    A0       = -2 * p2.dot(A1_r0)
    A1v      = -2 * p2.dot(A1_r1)
    norm_A_sq = A0**2 + A1v**2
    norm_A   = sp.sqrt(norm_A_sq)
    # avoid .norm()**2 — compute squared norms via .dot() to stay polynomial
    p2_proj  = p2 - k * k.dot(p1)
    p2_proj_sq = p2_proj.dot(p2_proj)
    KxP_sq   = KxP.dot(KxP)
    b        = d**2 - p2_proj_sq - KxP_sq
    x_ls0    = A1_r0.dot(-2 * p2) * b / norm_A_sq
    x_ls1    = A1_r1.dot(-2 * p2) * b / norm_A_sq
    # xi       = sp.sqrt(1 - b**2 / norm_A_sq)
    xi = sp.sqrt(sp.expand(1 - b**2 / norm_A_sq))
    A_perp0  = A1v / norm_A
    A_perp1  = -A0 / norm_A
    sc1 = sp.Matrix([x_ls0 + xi * A_perp0, x_ls1 + xi * A_perp1])
    sc2 = sp.Matrix([x_ls0 - xi * A_perp0, x_ls1 - xi * A_perp1])
    return sp.atan2(sc1[0], sc1[1]), sp.atan2(sc2[0], sc2[1])


def _sp4(p, k, h, d):
    """Direct translation of JAX sp4 — no branching, NaN propagates."""
    A11       = k.cross(p)
    A1_r0     = A11
    A1_r1     = -k.cross(A11)
    # A = h @ A1.T  →  (2,)
    A0        = h.dot(A1_r0)
    A1v       = h.dot(A1_r1)
    b         = d - h.dot(k) * k.dot(p)
    norm_A2   = A0**2 + A1v**2
    # x_ls = A1 @ (h * b)
    x_ls0     = A1_r0.dot(h) * b
    x_ls1     = A1_r1.dot(h) * b
    # xi        = sp.sqrt(norm_A2 - b**2)
    xi = sp.sqrt(sp.expand(norm_A2 - b**2))
    # A_perp_tilde = [A[1], -A[0]]
    sc1 = sp.Matrix([x_ls0 + xi * A1v,  x_ls1 + xi * (-A0)])
    sc2 = sp.Matrix([x_ls0 - xi * A1v,  x_ls1 - xi * (-A0)])
    return sp.atan2(sc1[0], sc1[1]), sp.atan2(sc2[0], sc2[1])


# ── staged derivation ─────────────────────────────────────────────────────────

def _derive_staged(H_num, P_num):
    """
    Derive 3p2i IK symbolically in stages, introducing fresh intermediate
    symbols at each branch boundary to prevent expression blowup.

    Returns a list of (stage_name, input_syms, replacements, reduced_exprs,
                       output_sym_names, output_exprs)
    which are emitted sequentially into the generated file.
    """
    # convert H and P to exact rational sympy matrices
    H = [_vec(H_num[:, j]) for j in range(6)]   # H[:,j]
    P = [_vec(P_num[:, j]) for j in range(7)]   # P[:,j]

    # ── input symbols ─────────────────────────────────────────────────────────
    R_syms = [[sp.Symbol(f'r{i+1}{j+1}') for j in range(3)] for i in range(3)]
    p_syms = [sp.Symbol(f'p{i+1}') for i in range(3)]
    R_06   = sp.Matrix(R_syms)
    p_0T   = sp.Matrix(p_syms)
    input_syms = [R_syms[i][j] for i in range(3) for j in range(3)] + p_syms

    stages = []

    # ── stage 1: p_06, d1, SP4 → t1_0, t1_1 ─────────────────────────────────
    print("  Stage 1: q1 (SP4)...")
    p_06 = p_0T - P[0] - R_06 * P[6]
    d1   = H[1].dot(P[1] + P[2] + P[3] + P[4])  # H[:,1]·sum(P[:,1:5])
    t1_0_expr, t1_1_expr = _sp4(p_06, -H[0], H[1], d1)

    t1_0, t1_1 = sp.symbols('t1_0 t1_1')
    stage1_exprs   = [t1_0_expr, t1_1_expr]
    stage1_outputs = [t1_0, t1_1]
    repl, red = sp.cse(stage1_exprs, optimizations='basic')
    stages.append(('stage1_q1', input_syms, repl, red, stage1_outputs))
    print(f"    {len(repl)} CSE subexpressions")

    # ── stage 2: for each q1 branch, R_01, d5, SP4 → t5 ─────────────────────
    print("  Stage 2: q5 (SP4) for each q1 branch...")
    t5_syms = []
    for i_q1, t1 in enumerate([t1_0, t1_1]):
        R_01      = _sym_rot(H[0], t1)
        d5_expr   = H[1].dot(R_01.T * R_06 * H[5])
        t5_0_expr, t5_1_expr = _sp4(H[5], H[4], H[1], d5_expr)

        t5_0 = sp.Symbol(f't5_0_q1{i_q1}')
        t5_1 = sp.Symbol(f't5_1_q1{i_q1}')
        t5_syms.append((t5_0, t5_1))

        repl, red = sp.cse([t5_0_expr, t5_1_expr], optimizations='basic')
        stages.append((f'stage2_q5_q1{i_q1}', [t1], repl, red, [t5_0, t5_1]))
        print(f"    q1 branch {i_q1}: {len(repl)} CSE subexpressions")

    # ── stage 3: theta_14, q6, d_inner, d3 ───────────────────────────────────
    print("  Stage 3: theta_14, q6, d_inner, d3...")
    mid_syms = {}  # keyed by (i_q1, i_q5)

    for i_q1, t1 in enumerate([t1_0, t1_1]):
        R_01 = _sym_rot(H[0], t1)
        for i_q5, t5 in enumerate(t5_syms[i_q1]):
            R_45 = _sym_rot(H[4], t5)

            th14_expr = _sp1(R_45 * H[5],     R_01.T * R_06 * H[5], H[1])
            q6_expr   = _sp1(R_45.T * H[1],   R_06.T * R_01 * H[1], -H[5])

            # introduce fresh symbols for theta_14 and q6
            th14 = sp.Symbol(f'th14_q1{i_q1}_q5{i_q5}')
            q6   = sp.Symbol(f'q6_q1{i_q1}_q5{i_q5}')

            repl, red = sp.cse([th14_expr, q6_expr], optimizations='basic')
            stages.append((f'stage3a_th14q6_q1{i_q1}_q5{i_q5}',
                           [t1, t5], repl, red, [th14, q6]))

            # d_inner using fresh th14 symbol
            d_inner_expr = R_01.T * p_06 - P[1] - _sym_rot(H[1], th14) * P[4]
            d_inner_exprs = [d_inner_expr[0], d_inner_expr[1], d_inner_expr[2]]
            d3_expr = sp.sqrt(sum(e**2 for e in d_inner_exprs))

            di0 = sp.Symbol(f'di0_q1{i_q1}_q5{i_q5}')
            di1 = sp.Symbol(f'di1_q1{i_q1}_q5{i_q5}')
            di2 = sp.Symbol(f'di2_q1{i_q1}_q5{i_q5}')
            d3  = sp.Symbol(f'd3_q1{i_q1}_q5{i_q5}')

            repl, red = sp.cse(d_inner_exprs + [d3_expr], optimizations='basic')
            stages.append((f'stage3b_dinner_q1{i_q1}_q5{i_q5}',
                           [t1, t5, th14], repl, red, [di0, di1, di2, d3]))

            mid_syms[(i_q1, i_q5)] = (th14, q6, di0, di1, di2, d3)
            print(f"    q1={i_q1} q5={i_q5}: done")

    # ── stage 4: SP3 → t3_0, t3_1 ────────────────────────────────────────────
    print("  Stage 4: q3 (SP3)...")
    t3_syms = {}

    for i_q1 in range(2):
        for i_q5 in range(2):
            th14, q6, di0, di1, di2, d3 = mid_syms[(i_q1, i_q5)]
            d_inner_sym = sp.Matrix([di0, di1, di2])

            t3_0_expr, t3_1_expr = _sp3(-P[3], P[2], H[1], d3)

            t3_0 = sp.Symbol(f't3_0_q1{i_q1}_q5{i_q5}')
            t3_1 = sp.Symbol(f't3_1_q1{i_q1}_q5{i_q5}')
            t3_syms[(i_q1, i_q5)] = (t3_0, t3_1)

            repl, red = sp.cse([t3_0_expr, t3_1_expr], optimizations='basic')
            stages.append((f'stage4_q3_q1{i_q1}_q5{i_q5}',
                           [d3], repl, red, [t3_0, t3_1]))
            print(f"    q1={i_q1} q5={i_q5}: {len(repl)} CSE subexpressions")

    # ── stage 5: q2, q4 per full branch ──────────────────────────────────────
    print("  Stage 5: q2, q4...")
    # branch order: (q1=0,q5=0,q3=0), (q1=0,q5=0,q3=1),
    #               (q1=0,q5=1,q3=0), ..., (q1=1,q5=1,q3=1)
    branch_joints = []  # list of (q1, q2, q3, q4, q5, q6) sympy exprs

    for i_q1, t1 in enumerate([t1_0, t1_1]):
        for i_q5, t5 in enumerate(t5_syms[i_q1]):
            th14, q6, di0, di1, di2, d3 = mid_syms[(i_q1, i_q5)]
            d_inner_sym = sp.Matrix([di0, di1, di2])

            for i_q3, t3 in enumerate(t3_syms[(i_q1, i_q5)]):
                q2_expr = _sp1(
                    P[2] + _sym_rot(H[1], t3) * P[3],
                    d_inner_sym,
                    H[1],
                )
                q4_expr = th14 - q2_expr - t3

                deps = [t1, t5, th14, di0, di1, di2, t3]
                repl, red = sp.cse([q2_expr, q4_expr], optimizations='basic')
                name = f'stage5_q2q4_q1{i_q1}_q5{i_q5}_q3{i_q3}'

                q2_sym = sp.Symbol(f'q2_{name}')
                q4_sym = sp.Symbol(f'q4_{name}')

                stages.append((name, deps, repl, red, [q2_sym, q4_sym]))

                branch_joints.append((t1, q2_sym, t3, q4_sym, t5, q6))
                print(f"    q1={i_q1} q5={i_q5} q3={i_q3}: done")

    return input_syms, stages, branch_joints


# ── code emitter ──────────────────────────────────────────────────────────────

class _JnpPrinter(PythonCodePrinter):
    def _print_sin(self, expr):
        return f"jnp.sin({self._print(expr.args[0])})"
    def _print_cos(self, expr):
        return f"jnp.cos({self._print(expr.args[0])})"
    def _print_sqrt(self, expr):
        return f"jnp.sqrt({self._print(expr.args[0])})"
    def _print_Pow(self, expr):
        b, e = expr.args
        if e == sp.Rational(1, 2):
            return f"jnp.sqrt({self._print(b)})"
        if e == -1:
            return f"(1.0 / ({self._print(b)}))"
        return f"({self._print(b)} ** {self._print(e)})"
    def _print_atan2(self, expr):
        return f"jnp.arctan2({self._print(expr.args[0])}, {self._print(expr.args[1])})"
    def _print_acos(self, expr):
        return f"jnp.arccos({self._print(expr.args[0])})"
    def _print_asin(self, expr):
        return f"jnp.arcsin({self._print(expr.args[0])})"
    def _print_Float(self, expr):
        return repr(float(expr))
    def _print_Integer(self, expr):
        return repr(int(expr))
    def _print_Rational(self, expr):
        return repr(float(expr))


def _emit_file(robot_name, input_syms, stages, branch_joints, out_path):
    printer = _JnpPrinter()

    def emit(expr):
        return printer.doprint(expr)

    lines = []
    lines += [
        f'# AUTO-GENERATED by jaik.kinematics.codegen — do not edit',
        f'# Robot: {robot_name}  |  Family: 3p2i  |  Generated: {date.today()}',
        'import jax.numpy as jnp',
        'from jaxtyping import Float, Bool, Array, jaxtyped',
        'from beartype import beartype',
        '',
        '',
        '@jaxtyped(typechecker=beartype)',
        f'def ik_{robot_name.lower()}(',
        '    R_06: Float[Array, "3 3"],',
        '    p_0T: Float[Array, "3"],',
        ') -> tuple[Float[Array, "6 8"], Bool[Array, "8"]]:',
        f'    """CSE-generated IK for {robot_name} (3p2i).',
        f'    Returns (Q, valid): Q is (6,8), valid is (8,) bool.',
        f'    NaN entries mark infeasible branches."""',
        '',
    ]

    # unpack input symbols
    sym_names = [str(s) for s in input_syms]
    for i in range(3):
        for j in range(3):
            lines.append(f'    {sym_names[i*3+j]} = R_06[{i}, {j}]')
    for i in range(3):
        lines.append(f'    {sym_names[9+i]} = p_0T[{i}]')
    lines.append('')

    # # emit each stage's CSE assignments
    # for (stage_name, dep_syms, repl, red, out_syms) in stages:
    #     lines.append(f'    # ── {stage_name} ──')
    #     for sym, expr in repl:
    #         lines.append(f'    {sym} = {emit(expr)}')
    #     for out_sym, r_expr in zip(out_syms, red):
    #         lines.append(f'    {out_sym} = {emit(r_expr)}')
    #     lines.append('')
    
    # in _emit_file
    cse_counter = [0]  # mutable counter shared across stages

    for (stage_name, dep_syms, repl, red, out_syms) in stages:
        lines.append(f'    # ── {stage_name} ──')
        # build rename map for this stage's temporaries
        rename = {}
        for sym, expr in repl:
            new_name = f'_x{cse_counter[0]}'
            cse_counter[0] += 1
            rename[sym] = sp.Symbol(new_name)

        for sym, expr in repl:
            new_sym = rename[sym]
            renamed_expr = expr.subs(rename)
            lines.append(f'    {new_sym} = {emit(renamed_expr)}')

        for out_sym, r_expr in zip(out_syms, red):
            renamed_expr = r_expr.subs(rename)
            lines.append(f'    {out_sym} = {emit(renamed_expr)}')

    # stack branches into (6, 8)
    assert len(branch_joints) == 8
    for b, joints in enumerate(branch_joints):
        joint_strs = [emit(j) for j in joints]
        lines.append(f'    _b{b} = jnp.array([{", ".join(joint_strs)}])')

    lines += [
        '',
        '    Q = jnp.stack([',
    ]
    for b in range(8):
        lines.append(f'        _b{b},')
    lines += [
        '    ], axis=1)  # (6, 8)',
        '',
        '    valid = ~jnp.isnan(Q).any(axis=0)',
        '    return Q, valid',
        '',
    ]

    out_path.write_text('\n'.join(lines))
    print(f"  Written: {out_path} ({len(lines)} lines)")


# ── validation ────────────────────────────────────────────────────────────────
def _validate_intermediates(robot_name):
    import jax
    import jax.numpy as jnp
    from jaik.kinematics.robots import make_robot, _UR_PARAMS, _UR_R6T
    from jaik._jax.ik_3p2i import ik_3_parallel_2_intersecting
    from jaik._jax.subproblems import sp1, sp3, sp4
    from jaik._jax.utils import _rot
    import importlib

    fk_gen, _ = make_robot(robot_name, solver="general")
    mod = importlib.import_module(f"jaik._jax._generated.ik_{robot_name.lower()}")
    ik_cse = getattr(mod, f"ik_{robot_name.lower()}")

    from jaik.kinematics.robots import _UR_PARAMS, _UR_R6T
    from jaik.kinematics.adjustments import adjust_kin_for_3p2i
    p = _UR_PARAMS[robot_name]
    a = np.array([0, p["a2"], p["a3"], 0, 0, 0])
    d = np.array([p["d1"], 0, 0, p["d4"], p["d5"], p["d6"]])
    alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
    kin = adjust_kin_for_3p2i(dh_to_kin(alpha, a, d))
    H = jnp.array(kin['H'])
    P = jnp.array(kin['P'])
    RT = jnp.array(_UR_R6T)

    rng = np.random.default_rng(0)
    q = jnp.array(rng.uniform(-np.pi, np.pi, 6))
    R_tcp, p_tcp = fk_gen(q)
    R_06 = R_tcp @ RT.T

    print(f"\n  R_06 =\n{np.round(np.asarray(R_06), 4)}")
    print(f"  p_0T = {np.round(np.asarray(p_tcp), 4)}")

    # JAX general solver intermediates
    p_06 = p_tcp - P[:, 0] - R_06 @ P[:, 6]
    d1   = H[:, 1] @ P[:, 1:5].sum(axis=1)
    t1s  = sp4(p_06, -H[:, 0], H[:, 1], d1)
    print(f"\n  [JAX] t1s = {np.round(np.asarray(t1s), 4)}")

    # for branch 0
    q1 = t1s[0]
    R_01 = _rot(H[:, 0], q1)
    d5   = H[:, 1] @ R_01.T @ R_06 @ H[:, 5]
    t5s  = sp4(H[:, 5], H[:, 4], H[:, 1], d5)
    print(f"  [JAX] t5s (q1 branch 0) = {np.round(np.asarray(t5s), 4)}")

    q5 = t5s[0]
    R_45 = _rot(H[:, 4], q5)
    th14 = sp1(R_45 @ H[:, 5], R_01.T @ R_06 @ H[:, 5], H[:, 1])
    q6   = sp1(R_45.T @ H[:, 1], R_06.T @ R_01 @ H[:, 1], -H[:, 5])
    print(f"  [JAX] th14={float(th14):.4f}  q6={float(q6):.4f}")

    d_inner = R_01.T @ p_06 - P[:, 1] - _rot(H[:, 1], th14) @ P[:, 4]
    d3 = jnp.linalg.norm(d_inner)
    t3s = sp3(-P[:, 3], P[:, 2], H[:, 1], d3)
    print(f"  [JAX] d3={float(d3):.4f}  t3s={np.round(np.asarray(t3s), 4)}")

    q3 = t3s[0]
    q2 = sp1(P[:, 2] + _rot(H[:, 1], q3) @ P[:, 3], d_inner, H[:, 1])
    q4 = float((th14 - q2 - q3 + jnp.pi) % (2*jnp.pi) - jnp.pi)
    print(f"  [JAX] q2={float(q2):.4f}  q3={float(q3):.4f}  q4={q4:.4f}")
    print(f"  [JAX] full branch 0: {np.round([float(q1),float(q2),float(q3),q4,float(q5),float(q6)], 4)}")

    # CSE output
    Q_cse, valid_cse = ik_cse(R_06, p_tcp)
    print(f"\n  [CSE] branch 0: {np.round(np.asarray(Q_cse[:, 0]), 4)}")
    print(f"  [CSE] valid: {np.asarray(valid_cse)}")

    # check SP3 numerically
    p1_sp3 = -P[:, 3]
    p2_sp3 = P[:, 2]
    k_sp3  = H[:, 1]
    print(f"\n  [JAX] SP3 inputs:")
    print(f"    p1 = {np.round(np.asarray(p1_sp3), 4)}")
    print(f"    p2 = {np.round(np.asarray(p2_sp3), 4)}")
    print(f"    k  = {np.round(np.asarray(k_sp3), 4)}")
    print(f"    d  = {float(d3):.6f}")
    t3s_check = sp3(p1_sp3, p2_sp3, k_sp3, d3)
    print(f"  [JAX] SP3 output: {np.round(np.asarray(t3s_check), 4)}")

    # now check what the CSE stage4 computes
    # stage4 uses _sp3(-P[3], P[2], H[1], d3) with sympy H/P values
    # verify the numeric H/P values match
    print(f"\n  H[:,1] = {np.round(np.asarray(H[:,1]), 6)}")
    print(f"  P[:,2] = {np.round(np.asarray(P[:,2]), 6)}")
    print(f"  P[:,3] = {np.round(np.asarray(P[:,3]), 6)}")


def _validate(robot_name, out_path, n_tests=50):
    import importlib
    import jax.numpy as jnp
    from jaik.kinematics.robots import make_robot, _UR_PARAMS, _UR_R6T

    # _validate_intermediates(robot_name)
    # input("wwwwwwwwwwwwwww")

    print(f"  Validating CSE output via FK roundtrip ({n_tests} poses)...")

    fk_gen, ik_gen = make_robot(robot_name, solver="general")

    mod_name = f"jaik._jax._generated.ik_{robot_name.lower()}"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)
    ik_cse = getattr(mod, f"ik_{robot_name.lower()}")

    RT = jnp.array(_UR_R6T)
    rng = np.random.default_rng(43)

    n_no_valid   = 0
    n_fk_fail    = 0
    n_ok         = 0
    p_errs_all   = []
    R_errs_all   = []
    n_valid_per_pose = []

    for trial in range(n_tests):
        q = jnp.array(rng.uniform(-np.pi, np.pi, 6))
        R_tcp, p_tcp = fk_gen(q)
        R_06 = R_tcp @ RT.T

        Q_cse, valid_cse = ik_cse(R_06, p_tcp)
        Q_gen, valid_gen = ik_gen(R_tcp, p_tcp)

        Q_cse_np    = np.asarray(Q_cse)
        valid_cse_np = np.asarray(valid_cse)
        Q_gen_np    = np.asarray(Q_gen)
        valid_gen_np = np.asarray(valid_gen)

        n_valid = int(valid_cse_np.sum())
        n_valid_per_pose.append(n_valid)

        if n_valid == 0:
            n_no_valid += 1
            if trial < 3:
                print(f"\n  [trial {trial}] NO VALID solutions from CSE")
                print(f"    general valid: {valid_gen_np}")
                print(f"    CSE Q col 0: {Q_cse_np[:, 0]}")
                print(f"    CSE has NaN: {np.isnan(Q_cse_np).any(axis=0)}")
            continue

        pose_ok = True
        for i in range(8):
            if not valid_cse_np[i]:
                continue
            q_sol = jnp.array(Q_cse_np[:, i])
            R_check, p_check = fk_gen(q_sol)
            p_err = float(np.linalg.norm(np.asarray(p_check) - np.asarray(p_tcp)))
            R_err = float(np.linalg.norm(np.asarray(R_check) - np.asarray(R_tcp)))
            p_errs_all.append(p_err)
            R_errs_all.append(R_err)

            if p_err > 1e-5 or R_err > 1e-5:
                pose_ok = False
                if n_fk_fail < 3:
                    print(f"\n  [trial {trial}] FK MISMATCH on branch {i}:")
                    print(f"    q_sol   = {np.round(Q_cse_np[:, i], 4)}")
                    print(f"    p_err   = {p_err:.2e}  R_err = {R_err:.2e}")
                    print(f"    p_target= {np.round(np.asarray(p_tcp), 4)}")
                    print(f"    p_check = {np.round(np.asarray(p_check), 4)}")
                    # compare with general solver
                    print(f"    general valid branches: {np.where(valid_gen_np)[0]}")
                    for j in range(8):
                        if valid_gen_np[j]:
                            R_g, p_g = fk_gen(jnp.array(Q_gen_np[:, j]))
                            pe = float(np.linalg.norm(np.asarray(p_g) - np.asarray(p_tcp)))
                            print(f"    general branch {j}: q={np.round(Q_gen_np[:, j], 3)}  p_err={pe:.2e}")
                break

        if pose_ok:
            n_ok += 1
        else:
            n_fk_fail += 1

    print(f"\n  ── Summary ──────────────────────────────────────────")
    print(f"  Poses tested:          {n_tests}")
    print(f"  OK (FK roundtrip):     {n_ok}")
    print(f"  No valid solutions:    {n_no_valid}")
    print(f"  FK mismatch:           {n_fk_fail}")
    print(f"  Valid solutions/pose:  min={min(n_valid_per_pose)} "
          f"max={max(n_valid_per_pose)} "
          f"mean={np.mean(n_valid_per_pose):.1f}")
    if p_errs_all:
        print(f"  p_err (valid branches): "
              f"mean={np.mean(p_errs_all):.2e} "
              f"max={np.max(p_errs_all):.2e} "
              f"min={np.min(p_errs_all):.2e}")
        print(f"  R_err (valid branches): "
              f"mean={np.mean(R_errs_all):.2e} "
              f"max={np.max(R_errs_all):.2e} "
              f"min={np.min(R_errs_all):.2e}")
    print(f"  ─────────────────────────────────────────────────────")

    if n_fk_fail > 3:
        raise RuntimeError(
            f"Validation FAILED: {n_fk_fail}/{n_tests} poses had solutions "
            f"that don't round-trip via FK."
        )
    print(f"  Validation passed.")



# ── entry point ───────────────────────────────────────────────────────────────

def generate(robot_name, out_dir=None):

    if robot_name not in _UR_PARAMS:
        raise ValueError(f"Unknown robot '{robot_name}'. "
                         f"Available: {sorted(_UR_PARAMS.keys())}")

    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "_jax" / "_generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    init = out_dir / "__init__.py"
    if not init.exists():
        init.write_text("# auto-generated\n")

    p = _UR_PARAMS[robot_name]
    a = np.array([0, p["a2"], p["a3"], 0, 0, 0])
    d = np.array([p["d1"], 0, 0, p["d4"], p["d5"], p["d6"]])
    alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

    kin   = adjust_kin_for_3p2i(dh_to_kin(alpha, a, d))
    H_num = _clean_hp(kin['H'])   # ← clean before symbolic derivation
    P_num = _clean_hp(kin['P'])   # ← clean before symbolic derivation

    out_path = out_dir / f"ik_{robot_name.lower()}.py"

    print(f"\n=== Generating CSE IK for {robot_name} ===")
    print(f"  H[:,0] = {H_num[:,0]}  (should be [0,0,1])")
    print(f"  P[:,1] = {P_num[:,1]}  (should be [0,0,0])")
    print(f"  P[:,5] = {P_num[:,5]}  (should be [0,0,0])")

    t0 = time.perf_counter()

    print("  Deriving symbolic IK (staged)...")
    input_syms, stages, branch_joints = _derive_staged(H_num, P_num)
    print(f"  Derivation done in {time.perf_counter()-t0:.1f}s")

    _emit_file(robot_name, input_syms, stages, branch_joints, out_path)
    _validate(robot_name, out_path)

    print(f"  Total: {time.perf_counter()-t0:.1f}s")
    print(f"=== Done: {out_path} ===\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m jaik.kinematics.codegen <Robot> [<Robot2> ...]")
        sys.exit(1)
    for name in sys.argv[1:]:
        generate(name)