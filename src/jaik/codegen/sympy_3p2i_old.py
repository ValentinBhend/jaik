import sys
import time
from pathlib import Path
from datetime import date
import importlib

import numpy as np
import sympy as sp

import jax.numpy as jnp

from jaik.kinematics.convention_conversions import dh_to_kin
from jaik.kinematics.adjustments import adjust_kin_for_3p2i
from jaik.kinematics.robots import make_robot

from .utils import _vec, _sym_rot, _clean_hp, _generate_param_symbols
from .subproblems import _sp1, _sp3, _sp4
from .sympy_helpers import _get_apply_pythagorean, _replace_sign, _replace_hypot, _replace_after_cse

def sympy_3p2i_fk(kin):
    H_num = kin["H"]
    P_num = kin["P"]
    H, P, param_map = _generate_param_symbols(H_num, P_num)
    sq = [sp.Symbol(f'sq{i+1}', real=True) for i in range(6)]
    cq = [sp.Symbol(f'cq{i+1}', real=True) for i in range(6)]

    input_syms = [sq[i] for i in range(6)] + [cq[i] for i in range(6)]

    R = sp.eye(3)
    p = P[0]
    for i in range(6):
        R = R @ _sym_rot(H[i], sq[i], cq[i])
        p = p + R @ P[i + 1]
    
    R_syms = [[sp.Symbol(f'r{i+1}{j+1}', real=True) for j in range(3)] for i in range(3)]
    p_syms = [sp.Symbol(f'p{i+1}', real=True) for i in range(3)]
    all_syms  = [R_syms[i][j] for i in range(3) for j in range(3)] + p_syms
    all_exprs = [R[i,j] for i in range(3) for j in range(3)] + [p[i] for i in range(3)]

    blacklist = set(all_syms) | set(input_syms)

    # Define your inputs that should be "ignored" for constant-only folding
    print(f"  Running global CSE on {len(all_exprs)} expressions...")
    print(f"{sum(sp.count_ops(e) for e in all_exprs)} total operations...")
    const_repl, const_red = sp.cse(all_exprs, ignore=blacklist, optimizations='basic', symbols=sp.numbered_symbols('c'))
    global_repl, global_red = sp.cse(const_red, optimizations='basic', symbols=sp.numbered_symbols('x'))
    print(f"  Global CSE: {len(const_repl)} constant subexpressions extracted")
    print(f"  Global CSE: {len(global_repl)} shared subexpressions extracted")

    return input_syms, all_syms, const_repl, global_repl, global_red, param_map
    
    

def sympy_3p2i_ik(kin):
    """
    Collect all raw (sym, expr) pairs in dependency order,
    then run ONE global sp.cse.  Boundary symbols (t1_*, t5_*, th14_*,
    d3_*, t3_*) remain explicit atoms that prevent expression blow-up
    while enabling cross-stage CSE to factor sin/cos(t1_0) etc. once.
    """
    H_num = kin["H"]
    P_num = kin["P"]

    H, P, param_map = _generate_param_symbols(H_num, P_num)

    R_syms = [[sp.Symbol(f'r{i+1}{j+1}', real=True) for j in range(3)] for i in range(3)]
    p_syms = [sp.Symbol(f'p{i+1}', real=True) for i in range(3)]
    R_06   = sp.Matrix(R_syms)
    p_0T   = sp.Matrix(p_syms)
    input_syms = [R_syms[i][j] for i in range(3) for j in range(3)] + p_syms

    raw = []  # (output_sym, raw_sympy_expr) in strict dependency order

    # ── q1 ────────────────────────────────────────────────────────────────────
    p_06 = p_0T - P[0] - R_06 * P[6]
    d1   = H[1].dot(P[1] + P[2] + P[3] + P[4])
    (s_t1_0_expr, c_t1_0_expr), (s_t1_1_expr, c_t1_1_expr) = _sp4(p_06, -H[0], H[1], d1)
    s_t1_0, c_t1_0, s_t1_1, c_t1_1 = sp.symbols('s_t1_0 c_t1_0 s_t1_1 c_t1_1')
    raw += [(s_t1_0, s_t1_0_expr), (c_t1_0, c_t1_0_expr), (s_t1_1, s_t1_1_expr), (c_t1_1, c_t1_1_expr)]

    # ── q5 per q1 branch ──────────────────────────────────────────────────────
    t5_syms = []
    for i_q1, (s_t1, c_t1) in enumerate([(s_t1_0, c_t1_0), (s_t1_1, c_t1_1)]):
        R_01    = _sym_rot(H[0], s_t1, c_t1)               # cos(t1)/sin(t1) as atoms
        d5_expr = H[1].dot(R_01.T * R_06 * H[5])
        (s_t5_0_expr, c_t5_0_expr), (s_t5_1_expr, c_t5_1_expr) = _sp4(H[5], H[4], H[1], d5_expr)
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
            s_th14_expr, c_th14_expr = _sp1(R_45 * H[5],   R_01.T * R_06 * H[5], H[1])
            s_q6_expr, c_q6_expr   = _sp1(R_45.T * H[1], R_06.T * R_01 * H[1], -H[5])
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
    
    # ── q3 ────────────────────────────────────────────────────────────────────
    # _sp3 args are all rational + the d3 atom → polynomial in d3 after expand
    t3_syms = {}
    for i_q1 in range(2):
        for i_q5 in range(2):
            *_, d3 = mid_syms[(i_q1, i_q5)]
            (s_t3_0_expr, c_t3_0_expr), (s_t3_1_expr, c_t3_1_expr) = _sp3(-P[3], P[2], H[1], d3)
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
                s_q2_expr, c_q2_expr = _sp1(P[2] + _sym_rot(H[1], s_t3, c_t3) * P[3], d_inner_sym, H[1])

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
    
    th14_pairs = [
        (mid_syms[(i_q1, i_q5)][0][0], mid_syms[(i_q1, i_q5)][0][1])
        for i_q1 in range(2) for i_q5 in range(2)
    ]
    all_angle_pairs = list(dict.fromkeys(
        [(entry[i], entry[i+1]) for entry in branch_joints for i in range(0, len(entry), 2)]
        + th14_pairs
    ))
    _apply_pythagorean = _get_apply_pythagorean(all_angle_pairs)

    all_exprs = [sp.trigsimp(e) for e in all_exprs]
    all_exprs = [sp.simplify(e) for e in all_exprs]
    all_exprs = [sp.trigsimp(e) for e in all_exprs]


    all_exprs = [_apply_pythagorean(e) for e in all_exprs]

    all_exprs = [e.subs(lazy_sub_map, simultaneous=True) for e in all_exprs]
    
    all_exprs = [_apply_pythagorean(e) for e in all_exprs]
    all_exprs = [sp.trigsimp(e) for e in all_exprs]
    all_exprs = [sp.simplify(e) for e in all_exprs]
    all_exprs = [sp.trigsimp(e) for e in all_exprs]
    all_exprs = [_apply_pythagorean(e) for e in all_exprs]
    
    # all_exprs = [_replace_sign(e) for e in all_exprs]

    # all_exprs = [sp.trigsimp(e) for e in all_exprs]
    # all_exprs = [sp.simplify(e) for e in all_exprs]
    # all_exprs = [sp.trigsimp(e) for e in all_exprs]

    all_exprs = [_replace_sign(e) for e in all_exprs]
    all_exprs = [_replace_hypot(e) for e in all_exprs]

    blacklist = set(all_syms) | set(input_syms)

    # Define your inputs that should be "ignored" for constant-only folding
    print(f"  Running global CSE on {len(all_exprs)} expressions...")
    print(f"{sum(sp.count_ops(e) for e in all_exprs)} total operations...")
    const_repl, const_red = sp.cse(all_exprs, ignore=blacklist, optimizations='basic', symbols=sp.numbered_symbols('c'))
    global_repl, global_red = sp.cse(const_red, optimizations='basic', symbols=sp.numbered_symbols('x'))
    print(f"  Global CSE: {len(const_repl)} constant subexpressions extracted")
    print(f"  Global CSE: {len(global_repl)} shared subexpressions extracted")

    global_repl, global_red = _replace_after_cse(global_repl, global_red, _replace_sign)
    global_repl, global_red = _replace_after_cse(global_repl, global_red, _replace_hypot)

    return input_syms, all_syms, const_repl, global_repl, global_red, branch_joints, param_map