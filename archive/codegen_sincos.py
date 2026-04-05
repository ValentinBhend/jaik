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

def _vec(arr):
    """numpy 3-vector → sympy Matrix column, entries as rationals."""
    return sp.Matrix([sp.Float(arr[i]) for i in range(3)])


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

def _sym_rot(k_vec, s_theta, c_theta):
    """Rodrigues rotation: k_vec is sympy column vector, theta is sympy expr."""
    kx, ky, kz = k_vec
    c = c_theta
    s = s_theta
    t = 1 - c
    return sp.Matrix([
        [t*kx*kx + c,    t*kx*ky - s*kz, t*kx*kz + s*ky],
        [t*kx*ky + s*kz, t*ky*ky + c,    t*ky*kz - s*kx],
        [t*kx*kz - s*ky, t*ky*kz + s*kx, t*kz*kz + c   ],
    ])


# def _sp1_sincos(p1, p2, k):
#     """
#     Returns (sin_theta, cos_theta) instead of atan2.
#     Eliminates transcendental calls in the hot path.
#     """
#     # KxP is the basis vector for the sine component
#     KxP = k.cross(p1)
    
#     # y is the sine component, x is the cosine component
#     y = KxP.dot(p2)
#     x = (-k.cross(KxP)).dot(p2) # TODO should be the same, but check by replacing: x = (p1 - k * k.dot(p1)).dot(p2)
    
#     # Normalization factor
#     # In JAX, we'll want to ensure this isn't zero to avoid NaN
#     mag = sp.sqrt(y**2 + x**2)
    
#     return y / mag, x / mag

# def _sp1_sincos(p1, p2, k):
#     # 1. Project p1 and p2 into the plane perpendicular to k
#     # This is critical for UR robots where axes are often offset
#     p1_p = p1 - k * k.dot(p1)
#     p2_p = p2 - k * k.dot(p2)
    
#     # 2. Define the orthonormal basis for the plane
#     # u is the 0-angle direction (cosine axis)
#     # v is the 90-degree direction (sine axis)
#     u = p1_p
#     v = k.cross(u)
    
#     # 3. Sine is projection on v, Cosine is projection on u
#     # Note: we need the dot product of p2_p with these basis vectors
#     # But since u and v aren't unit, we normalize at the end anyway
#     y = p2_p.dot(v)
#     x = p2_p.dot(u)
    
#     mag = sp.sqrt(y**2 + x**2)
#     return y / mag, x / mag

def _sp1_sincos(p1, p2, k):
    """
    Subproblem 1: Find rotation about axis k that takes p1 to p2.
    Matches Rodrigues: R(k, theta) * p1 = p2
    """
    # 1. Ensure projections are perpendicular to the axis
    p1_p = p1 - k * k.dot(p1)
    p2_p = p2 - k * k.dot(p2)
    
    # 2. Basis vectors
    # u (cosine axis) is the direction of p1_p
    # v (sine axis) must be k cross u to follow right-hand rule
    u = p1_p
    v = k.cross(u)
    
    # 3. Component projections
    # p2_p = cos(theta)*u + sin(theta)*v
    y = p2_p.dot(v)
    x = p2_p.dot(u)
    
    mag = sp.sqrt(y**2 + x**2)
    return y / mag, x / mag


def _sp3_sincos(p1, p2, k, d):
    """Mechanical translation of SP3 to Sincos."""
    KxP = k.cross(p1)
    A1_r0 = KxP
    A1_r1 = -k.cross(KxP)
    
    A0 = -2 * p2.dot(A1_r0)
    A1v = -2 * p2.dot(A1_r1)
    norm_A_sq = A0**2 + A1v**2
    norm_A = sp.sqrt(norm_A_sq)
    
    p2_proj = p2 - k * k.dot(p1)
    p2_proj_sq = p2_proj.dot(p2_proj)
    KxP_sq = KxP.dot(KxP)
    
    b = d**2 - p2_proj_sq - KxP_sq
    
    # Least squares components
    x_ls0 = A1_r0.dot(-2 * p2) * b / norm_A_sq
    x_ls1 = A1_r1.dot(-2 * p2) * b / norm_A_sq
    
    # Discriminant for the two solutions
    xi = sp.sqrt(sp.expand(1 - b**2 / norm_A_sq))
    
    A_perp0 = A1v / norm_A
    A_perp1 = -A0 / norm_A
    
    # Raw sine and cosine components
    s1_raw, c1_raw = x_ls0 + xi * A_perp0, x_ls1 + xi * A_perp1
    s2_raw, c2_raw = x_ls0 - xi * A_perp0, x_ls1 - xi * A_perp1
    
    # Normalize to ensure unit vectors (sin^2 + cos^2 = 1)
    mag1 = sp.sqrt(s1_raw**2 + c1_raw**2)
    mag2 = sp.sqrt(s2_raw**2 + c2_raw**2)
    
    return (s1_raw/mag1, c1_raw/mag1), (s2_raw/mag2, c2_raw/mag2)


# def _sp4_sincos(p, k, h, d):
#     """Mechanical translation of SP4 to Sincos."""
#     A11 = k.cross(p)
#     A1_r0 = A11
#     A1_r1 = -k.cross(A11)
    
#     A0 = h.dot(A1_r0)
#     A1v = h.dot(A1_r1)
#     b = d - h.dot(k) * k.dot(p)
#     norm_A2 = A0**2 + A1v**2
    
#     x_ls0 = A1_r0.dot(h) * b
#     x_ls1 = A1_r1.dot(h) * b
    
#     xi = sp.sqrt(sp.expand(norm_A2 - b**2))
    
#     # Raw sine and cosine components
#     # Note: division by norm_A2 is built into the ls components 
#     # but we normalize explicitly at the end anyway.
#     s1_raw, c1_raw = (x_ls0 + xi * A1v) / norm_A2, (x_ls1 + xi * (-A0)) / norm_A2
#     s2_raw, c2_raw = (x_ls0 - xi * A1v) / norm_A2, (x_ls1 - xi * (-A0)) / norm_A2
    
#     mag1 = sp.sqrt(s1_raw**2 + c1_raw**2)
#     mag2 = sp.sqrt(s2_raw**2 + c2_raw**2)
    
#     return (s1_raw/mag1, c1_raw/mag1), (s2_raw/mag2, c2_raw/mag2)

# def _sp4_sincos(p, k, h, d):
#     A11 = k.cross(p)
#     A1_r0 = A11
#     A1_r1 = -k.cross(A11)
    
#     A0 = h.dot(A1_r0)
#     A1v = h.dot(A1_r1)
#     b = d - h.dot(k) * k.dot(p)
#     norm_A2 = A0**2 + A1v**2
    
#     # Do NOT divide by norm_A2 yet. 
#     # Just replicate the sc1/sc2 logic from your old code.
#     x_ls0 = A1_r0.dot(h) * b
#     x_ls1 = A1_r1.dot(h) * b
    
#     xi = sp.sqrt(sp.expand(norm_A2 - b**2))
    
#     # These match your old sc1/sc2 Matrix entries exactly
#     s1_raw, c1_raw = x_ls0 + xi * A1v, x_ls1 + xi * (-A0)
#     s2_raw, c2_raw = x_ls0 - xi * A1v, x_ls1 - xi * (-A0)
    
#     mag1 = sp.sqrt(s1_raw**2 + c1_raw**2)
#     mag2 = sp.sqrt(s2_raw**2 + c2_raw**2)
    
#     return (s1_raw/mag1, c1_raw/mag1), (s2_raw/mag2, c2_raw/mag2)

def _sp4_sincos(p, k, h, d):
    """
    Subproblem 4: Find rotation about axis k such that h.dot(R*p) = d
    Matches the sc1/sc2 logic from the working atan2 version.
    """
    A11 = k.cross(p)
    A1_r0 = A11          # This is the 'sine' direction basis
    A1_r1 = -k.cross(A11) # This is the 'cosine' direction basis
    
    A0 = h.dot(A1_r0)
    A1v = h.dot(A1_r1)
    b = d - h.dot(k) * k.dot(p)
    norm_A2 = A0**2 + A1v**2
    
    # These represent the 'Least Squares' center point in the (s, c) plane
    x_ls0 = A0 * b
    x_ls1 = A1v * b
    
    # xi is the 'distance' from the LS center to the valid circle intersections
    xi = sp.sqrt(sp.expand(norm_A2 - b**2))
    
    # Combine to get two solutions for (sin, cos)
    # Solution 1
    s1_raw = x_ls0 + xi * A1v
    c1_raw = x_ls1 - xi * A0
    # Solution 2
    s2_raw = x_ls0 - xi * A1v
    c2_raw = x_ls1 + xi * A0
    
    mag1 = sp.sqrt(s1_raw**2 + c1_raw**2)
    mag2 = sp.sqrt(s2_raw**2 + c2_raw**2)
    
    return (s1_raw/mag1, c1_raw/mag1), (s2_raw/mag2, c2_raw/mag2)

def generate_param_symbols(H_num, P_num):
    param_map = {}
    
    def _sym_vec(arr, name):
        vals = []
        for i in range(3):
            # Check if it's a significant constant (not 0 or 1)
            val = arr[i]
            if abs(val) > 1e-10 and abs(val - 1.0) > 1e-10 and abs(val + 1.0) > 1e-10:
                if val > 0:
                    symbol = sp.Symbol(f"{name}{i+1}", real=True, nonzero=True, positive=True)
                else:
                    symbol = sp.Symbol(f"{name}{i+1}", real=True, nonzero=True, negative=True)
                param_map[symbol] = float(val)
                vals.append(symbol)
            else:
                vals.append(sp.Integer(int(round(val))))
        return sp.Matrix(vals)

    H_sym = [_sym_vec(H_num[:, j], f"H{j}") for j in range(6)]
    P_sym = [_sym_vec(P_num[:, j], f"P{j}") for j in range(7)]
    
    return H_sym, P_sym, param_map

def _derive_global(H_num, P_num):
    """
    Collect all raw (sym, expr) pairs in dependency order,
    then run ONE global sp.cse.  Boundary symbols (t1_*, t5_*, th14_*,
    d3_*, t3_*) remain explicit atoms that prevent expression blow-up
    while enabling cross-stage CSE to factor sin/cos(t1_0) etc. once.
    """
    # H = [_vec(H_num[:, j]) for j in range(6)]
    # P = [_vec(P_num[:, j]) for j in range(7)]

    H, P, param_map = generate_param_symbols(H_num, P_num)

    R_syms = [[sp.Symbol(f'r{i+1}{j+1}', real=True) for j in range(3)] for i in range(3)]
    p_syms = [sp.Symbol(f'p{i+1}', real=True) for i in range(3)]
    R_06   = sp.Matrix(R_syms)
    p_0T   = sp.Matrix(p_syms)
    input_syms = [R_syms[i][j] for i in range(3) for j in range(3)] + p_syms

    raw = []  # (output_sym, raw_sympy_expr) in strict dependency order

    # ── q1 ────────────────────────────────────────────────────────────────────
    p_06 = p_0T - P[0] - R_06 * P[6]
    d1   = H[1].dot(P[1] + P[2] + P[3] + P[4])
    (s_t1_0_expr, c_t1_0_expr), (s_t1_1_expr, c_t1_1_expr) = _sp4_sincos(p_06, -H[0], H[1], d1)
    s_t1_0, c_t1_0, s_t1_1, c_t1_1 = sp.symbols('s_t1_0 c_t1_0 s_t1_1 c_t1_1')
    raw += [(s_t1_0, s_t1_0_expr), (c_t1_0, c_t1_0_expr), (s_t1_1, s_t1_1_expr), (c_t1_1, c_t1_1_expr)]

    # ── q5 per q1 branch ──────────────────────────────────────────────────────
    t5_syms = []
    for i_q1, (s_t1, c_t1) in enumerate([(s_t1_0, c_t1_0), (s_t1_1, c_t1_1)]):
        R_01    = _sym_rot(H[0], s_t1, c_t1)               # cos(t1)/sin(t1) as atoms
        d5_expr = H[1].dot(R_01.T * R_06 * H[5])
        (s_t5_0_expr, c_t5_0_expr), (s_t5_1_expr, c_t5_1_expr) = _sp4_sincos(H[5], H[4], H[1], d5_expr)
        s_t5_0 = sp.Symbol(f's_t5_0_q1{i_q1}', real=True)
        c_t5_0 = sp.Symbol(f'c_t5_0_q1{i_q1}', real=True)
        s_t5_1 = sp.Symbol(f's_t5_1_q1{i_q1}', real=True)
        c_t5_1 = sp.Symbol(f'c_t5_1_q1{i_q1}', real=True)
        t5_syms.append(((s_t5_0, c_t5_0), (s_t5_1, c_t5_1)))
        raw += [(s_t5_0, s_t5_0_expr), (c_t5_0, c_t5_0_expr), (s_t5_1, s_t5_1_expr), (c_t5_1, c_t5_1_expr)]

    # ── th14, q6, d_inner, d3 ────────────────────────────────────────────────
    # Key: R_01.T @ R_06 @ H[5] is shared between th14 and the q5=0,q5=1
    # branches for the same q1 — global CSE will factor it out automatically.
    mid_syms = {}
    lazy_sub_map = {}
    for i_q1, (s_t1, c_t1) in enumerate([(s_t1_0, c_t1_0), (s_t1_1, c_t1_1)]):
        R_01 = _sym_rot(H[0], s_t1, c_t1)
        for i_q5, (s_t5, c_t5) in enumerate(t5_syms[i_q1]):
            R_45 = _sym_rot(H[4], s_t5, c_t5)
            s_th14_expr, c_th14_expr = _sp1_sincos(R_45 * H[5],   R_01.T * R_06 * H[5], H[1])
            s_q6_expr, c_q6_expr   = _sp1_sincos(R_45.T * H[1], R_06.T * R_01 * H[1], -H[5])
            s_th14 = sp.Symbol(f's_th14_q1{i_q1}_q5{i_q5}', real=True)
            c_th14 = sp.Symbol(f'c_th14_q1{i_q1}_q5{i_q5}', real=True)
            s_q6   = sp.Symbol(f's_q6_q1{i_q1}_q5{i_q5}', real=True)
            c_q6   = sp.Symbol(f'c_q6_q1{i_q1}_q5{i_q5}', real=True)
            raw += [(s_th14, s_th14_expr), (c_th14, c_th14_expr), (s_q6, s_q6_expr), (c_q6, c_q6_expr)]

            # d_inner uses th14 as atom — no blowup
            d_inner = R_01.T * p_06 - P[1] - _sym_rot(H[1], s_th14, c_th14) * P[4]
            di0 = sp.Symbol(f'di0_q1{i_q1}_q5{i_q5}', real=True)
            di1 = sp.Symbol(f'di1_q1{i_q1}_q5{i_q5}', real=True)
            di2 = sp.Symbol(f'di2_q1{i_q1}_q5{i_q5}', real=True)
            d3  = sp.Symbol(f'd3_q1{i_q1}_q5{i_q5}', real=True)
            d3_expr = sp.sqrt(d_inner[0]**2 + d_inner[1]**2 + d_inner[2]**2)
            # raw += [(di0, d_inner[0]), (di1, d_inner[1]),
            #         (di2, d_inner[2]), (d3, d3_expr)]
            
            mid_syms[(i_q1, i_q5)] = ((s_th14, c_th14), (s_q6, c_q6), di0, di1, di2, d3)
            
            lazy_sub_map[di0] = d_inner[0]
            lazy_sub_map[di1] = d_inner[1]
            lazy_sub_map[di2] = d_inner[2]
            lazy_sub_map[d3] = d3_expr
            
            # d_inner = R_01.T * p_06 - P[1] - _sym_rot(H[1], s_th14, c_th14) * P[4]
            # di0 = d_inner[0]
            # di1 = d_inner[1]
            # di2 = d_inner[2]
            # d3_expr = sp.sqrt(d_inner[0]**2 + d_inner[1]**2 + d_inner[2]**2)
            # d3  = d3_expr
            
            # mid_syms[(i_q1, i_q5)] = ((s_th14, c_th14), (s_q6, c_q6), di0, di1, di2, d3)

    # ── q3 ────────────────────────────────────────────────────────────────────
    # _sp3 args are all rational + the d3 atom → polynomial in d3 after expand
    t3_syms = {}
    for i_q1 in range(2):
        for i_q5 in range(2):
            *_, d3 = mid_syms[(i_q1, i_q5)]
            (s_t3_0_expr, c_t3_0_expr), (s_t3_1_expr, c_t3_1_expr) = _sp3_sincos(-P[3], P[2], H[1], d3)
            s_t3_0 = sp.Symbol(f's_t3_0_q1{i_q1}_q5{i_q5}', real=True)
            c_t3_0 = sp.Symbol(f'c_t3_0_q1{i_q1}_q5{i_q5}', real=True)
            s_t3_1 = sp.Symbol(f's_t3_1_q1{i_q1}_q5{i_q5}', real=True)
            c_t3_1 = sp.Symbol(f'c_t3_1_q1{i_q1}_q5{i_q5}', real=True)
            t3_syms[(i_q1, i_q5)] = ((s_t3_0, c_t3_0), (s_t3_1, c_t3_1))
            raw += [(s_t3_0, s_t3_0_expr), (c_t3_0, c_t3_0_expr), (s_t3_1, s_t3_1_expr), (c_t3_1, c_t3_1_expr)]

    # ── q2, q4 per branch ─────────────────────────────────────────────────────
    branch_joints = []
    for i_q1, (s_t1, c_t1) in enumerate([(s_t1_0, c_t1_0), (s_t1_1, c_t1_1)]):
        for i_q5, (s_t5, c_t5) in enumerate(t5_syms[i_q1]):
            (s_th14, c_th14), (s_q6, c_q6), di0, di1, di2, _ = mid_syms[(i_q1, i_q5)]
            d_inner_sym = sp.Matrix([di0, di1, di2])
            for i_q3, (s_t3, c_t3) in enumerate(t3_syms[(i_q1, i_q5)]):
                s_q2_expr, c_q2_expr = _sp1_sincos(P[2] + _sym_rot(H[1], s_t3, c_t3) * P[3], d_inner_sym, H[1])

                # s_q4_expr = s_th14 - s_q2_expr - s_t3
                # c_q4_expr = c_th14 - c_q2_expr - c_t3
                # First find sin/cos of (th14 - q2)
                s_14_2 = s_th14 * c_q2_expr - c_th14 * s_q2_expr
                c_14_2 = c_th14 * c_q2_expr + s_th14 * s_q2_expr
                # Then find sin/cos of ((th14 - q2) - q3)
                s_q4_expr = s_14_2 * c_t3 - c_14_2 * s_t3
                c_q4_expr = c_14_2 * c_t3 + s_14_2 * s_t3

                s_q2 = sp.Symbol(f's_q2_b{i_q1}{i_q5}{i_q3}', real=True)
                c_q2 = sp.Symbol(f'c_q2_b{i_q1}{i_q5}{i_q3}', real=True)
                s_q4 = sp.Symbol(f's_q4_b{i_q1}{i_q5}{i_q3}', real=True)
                c_q4 = sp.Symbol(f'c_q4_b{i_q1}{i_q5}{i_q3}', real=True)
                raw += [(s_q2, s_q2_expr), (c_q2, c_q2_expr), (s_q4, s_q4_expr), (c_q4, c_q4_expr)]
                branch_joints.append((s_t1, c_t1, s_q2, c_q2, s_t3, c_t3, s_q4, c_q4, s_t5, c_t5, s_q6, c_q6))

    # ── ONE global CSE pass ───────────────────────────────────────────────────
    all_syms  = [s for s, _ in raw]
    all_exprs = [e for _, e in raw]

    all_exprs = [sp.trigsimp(e) for e in all_exprs]
    all_exprs = [sp.simplify(e) for e in all_exprs]
    all_exprs = [sp.trigsimp(e) for e in all_exprs]

    all_exprs = [e.subs(lazy_sub_map, simultaneous=True) for e in all_exprs]

    # all_exprs = [sp.trigsimp(e) for e in all_exprs]
    # all_exprs = [sp.simplify(e) for e in all_exprs]
    # all_exprs = [sp.trigsimp(e) for e in all_exprs]

    blacklist = set(all_syms) | set(input_syms)

    # Define your inputs that should be "ignored" for constant-only folding
    print(f"  Running global CSE on {len(all_exprs)} expressions...")
    print(f"{sum(sp.count_ops(e) for e in all_exprs)} total operations...")
    const_repl, const_red = sp.cse(all_exprs, ignore=blacklist, optimizations='basic', symbols=sp.numbered_symbols('c'))
    global_repl, global_red = sp.cse(const_red, optimizations='basic', symbols=sp.numbered_symbols('x'))
    print(f"  Global CSE: {len(const_repl)} constant subexpressions extracted")
    print(f"  Global CSE: {len(global_repl)} shared subexpressions extracted")

    def finalize_subs(subs_list, red_list):
        # hypot = sp.Function('hypot')
        
        a = sp.Wild('a', properties=[lambda x: x.is_Symbol])
        b = sp.Wild('b', properties=[lambda x: x.is_Symbol])

        def _replace(expr):
            # # sqrt(a**2 + b**2) → hypot(a, b)
            # expr = expr.replace(
            #     sp.sqrt(a**2 + b**2),
            #     lambda a, b: hypot(a, b)
            # )
            # # 1/sqrt(a**2 + b**2) → 1/hypot(a, b)
            # expr = expr.replace(
            #     (a**2 + b**2) ** sp.Rational(-1, 2),
            #     lambda a, b: 1 / hypot(a, b)
            # )
            # a / Abs(a) → sign(a)
            expr = expr.replace(a / sp.Abs(a), sp.sign(a))
            return expr

        new_repl = [(sym, _replace(expr)) for sym, expr in subs_list]
        new_red  = [_replace(expr) for expr in red_list]
        return new_repl, new_red

    # global_repl, global_red = finalize_subs(global_repl, global_red)

    # global_repl, global_red = sp.cse(all_exprs, optimizations='basic')

    return input_syms, all_syms, const_repl, global_repl, global_red, branch_joints, param_map


class _JnpPrinter(PythonCodePrinter):    
    def __init__(self, settings=None):
        super().__init__(settings)
        # Force these to be handled by our _print_ methods
        self.known_functions.pop('hypot', None)
        self.known_functions.pop('sign', None)
    
    def _print_Function(self, expr):
        # This handles the sp.Function('hypot') objects
        if expr.func.__name__ == 'hypot':
            return f"jnp.hypot({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        if expr.func.__name__ == 'sign':
            return f"jnp.sign({self._print(expr.args[0])})"
        
        # Fallback for sin, cos, etc.
        return super()._print_Function(expr)

    def _print_hypot(self, expr):
        arg0 = self._print(expr.args[0])
        arg1 = self._print(expr.args[1])
        return f"np.hypot({arg0}, {arg1})"
    # 2. Handle SymPy's symbolic sign function
    def _print_sign(self, expr):
        arg0 = self._print(expr.args[0])
        return f"jnp.sign({arg0})"
    
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
        if e == sp.Rational(-1, 2) or e == -0.5:
            return f"(1.0 / jnp.sqrt({self._print(b)}))"
        if e == -1:
            return f"(1.0 / ({self._print(b)}))"
        return f"(({self._print(b)}) ** {self._print(e)})" # Wrap the base in its own parentheses!
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

def _emit_file(robot_name, input_syms, all_syms, const_repl, global_repl, global_red,
               branch_joints, param_map, out_path):
    printer = _JnpPrinter()

    def emit(expr):
        return printer.doprint(expr)

    lines = [
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
        ') -> tuple[Float[Array, "12 8"], Bool[Array, "8"]]:',
        f'    """CSE-generated IK for {robot_name} (3p2i).',
        f'    Returns (Q, valid): Q is (12,8), valid is (8,) bool.',
        f'    NaN entries mark infeasible branches."""',
        '',
    ]

    # ── Robot Parameters (Constants) ──
    for symbol, value in param_map.items():
        lines.append(f"    {symbol} = {value}")
    lines.append('')

    # -- Constant replacements --
    for sym, expr in const_repl:
        lines.append(f'    {sym} = {emit(expr)}')
    lines.append('')

    # -- Actual runtime calculations --
    sym_names = [str(s) for s in input_syms]
    for i in range(3):
        for j in range(3):
            lines.append(f'    {sym_names[i*3+j]} = R_06[{i}, {j}]')
    for i in range(3):
        lines.append(f'    {sym_names[9+i]} = p_0T[{i}]')
    lines.append('')

    # # All globally shared subexpressions first (sin/cos of boundary angles, etc.)
    # lines.append('    # ── global CSE: shared subexpressions ──')
    # for sym, expr in global_repl:
    #     lines.append(f'    {sym} = {emit(expr)}')
    # lines.append('')

    # # Stage outputs, each reduced to reference the shared subexpressions above
    # lines.append('    # ── stage outputs ──')
    # for sym, red_expr in zip(all_syms, global_red):
    #     lines.append(f'    {sym} = {emit(red_expr)}')
    # lines.append('')

    def get_max_x(sym) -> int:
        """
        Finds the maximum integer 'n' for all symbols following the pattern 'xn'.
        Works for both single symbols and complex expressions.
        """
        max_x = -1
        
        # .atoms(sp.Symbol) retrieves all unique variables in the expression
        for atom in sym.atoms(sp.Symbol):
            name = atom.name
            # Check if the name starts with 'x' followed by digits
            if name.startswith('x') and name[1:].isdigit():
                index = int(name[1:])
                if index > max_x:
                    max_x = index
                    
        return max_x

    # All globally shared subexpressions first (sin/cos of boundary angles, etc.)
    # TODO: Bit of a janky way to get the order right, but it works for now..
    lines.append('    # ── global CSE: shared subexpressions + stage outputs ──')
    for sym, expr in global_repl:
        x_ind_normal = get_max_x(sym)
        lines.append(f'    {sym} = {emit(expr)}')
        for sym, red_expr in zip(all_syms, global_red):
            x_ind_staged = get_max_x(red_expr)
            if x_ind_staged == x_ind_normal:
                lines.append(f'    {sym} = {emit(red_expr)}')
    lines.append('')

    # for sym, red_expr in zip(all_syms, global_red):
    #     lines.append(f'    {sym} = {emit(red_expr)}')
    # lines.append('')

    assert len(branch_joints) == 8
    for b, joints in enumerate(branch_joints):
        lines.append(f'    _b{b} = jnp.array([{", ".join(emit(j) for j in joints)}])')

    lines += [
        '',
        '    Q = jnp.stack([',
        *[f'        _b{b},' for b in range(8)],
        '    ], axis=1)  # (12, 8)',
        '',
        '    valid = ~jnp.isnan(Q).any(axis=0)',
        '    return Q, valid',
        '',
    ]

    out_path.write_text('\n'.join(lines))
    print(f"  Written: {out_path} ({len(lines)} lines)")



# ── validation ────────────────────────────────────────────────────────────────

def _validate(robot_name, out_path, n_tests=100):
    print(f"  Validating CSE output via FK roundtrip ({n_tests} poses)...")

    fk_gen, ik_gen, _ = make_robot(robot_name, solver="general")

    mod_name = f"jaik._jax._generated.ik_{robot_name.lower()}_sincos"
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

        sc_Q_cse, valid_cse = ik_cse(R_06, p_tcp)
        Q_gen, valid_gen = ik_gen(R_tcp, p_tcp)

        sc_Q_cse_np    = np.asarray(sc_Q_cse)
        sines = sc_Q_cse_np[0::2, :]
        cosines = sc_Q_cse_np[1::2, :]
        Q_cse_np = np.arctan2(sines, cosines)

        valid_cse_np = np.asarray(valid_cse)
        Q_gen_np    = np.asarray(Q_gen)
        valid_gen_np = np.asarray(valid_gen)

        n_valid = int(valid_cse_np.sum())
        n_valid_per_pose.append(n_valid)

        if n_valid == 0:
            n_no_valid += 1
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

    out_path = out_dir / f"ik_{robot_name.lower()}_sincos.py"

    print(f"\n=== Generating CSE IK for {robot_name} ===")
    # print(f"  H[:,0] = {H_num[:,0]}  (should be [0,0,1])")
    # print(f"  P[:,1] = {P_num[:,1]}  (should be [0,0,0])")
    # print(f"  P[:,5] = {P_num[:,5]}  (should be [0,0,0])")

    t0 = time.perf_counter()

    print("  Deriving symbolic IK (staged)...")
    input_syms, all_syms, const_repl, global_repl, global_red, branch_joints, param_map = _derive_global(H_num, P_num)
    print(f"  Derivation done in {time.perf_counter()-t0:.1f}s")

    _emit_file(robot_name, input_syms, all_syms, const_repl, global_repl, global_red, branch_joints, param_map, out_path)
    _validate(robot_name, out_path)

    print(f"  Total: {time.perf_counter()-t0:.1f}s")
    print(f"=== Done: {out_path} ===\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m jaik.kinematics.codegen <Robot> [<Robot2> ...]")
        sys.exit(1)
    for name in sys.argv[1:]:
        generate(name)