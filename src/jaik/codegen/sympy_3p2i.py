import sys
import time
from pathlib import Path
from datetime import date
import importlib

import numpy as np
import sympy as sp
from sympy.core.sorting import default_sort_key

import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import NamedTuple

from jaik.kinematics.convention_conversions import dh_to_kin
from jaik.kinematics.adjustments import adjust_kin_for_3p2i
from jaik.kinematics.robots import make_robot

from .utils import _vec, _sym_rot, _clean_hp, _generate_param_symbols
from .subproblems import _sp1, _sp3, _sp4
from .sympy_helpers import _get_apply_pythagorean, _replace_sign, _replace_hypot, _replace_after_cse


def sympy_3p2i_fk(kin):
    H_num = kin["H"]
    P_num = kin["P"]
    RT_num = kin["R_6T"]
    H, P, RT, param_map = _generate_param_symbols(H_num, P_num, RT_num)

    sq = [sp.Symbol(f'sq{i+1}', real=True) for i in range(6)]
    cq = [sp.Symbol(f'cq{i+1}', real=True) for i in range(6)]
    input_syms = sq + cq

    R = sp.eye(3)
    p = P[0]
    for i in range(6):
        R = R @ _sym_rot(H[i], sq[i], cq[i])
        p = p + R @ P[i + 1]
    R = R @ RT

    R_syms = [[sp.Symbol(f'r{i+1}{j+1}', real=True) for j in range(3)] for i in range(3)]
    p_syms = [sp.Symbol(f'p{i+1}', real=True) for i in range(3)]
    all_syms  = [R_syms[i][j] for i in range(3) for j in range(3)] + p_syms
    all_exprs = [R[i, j] for i in range(3) for j in range(3)] + [p[i] for i in range(3)]

    blacklist = set(all_syms) | set(input_syms)

    print(f"  Running FK CSE on {len(all_exprs)} expressions...")
    print(f"  {sum(sp.count_ops(e) for e in all_exprs)} total operations...")

    const_repl, const_red   = sp.cse(all_exprs, ignore=blacklist, optimizations='basic', symbols=sp.numbered_symbols('c'))
    global_repl, global_red = sp.cse(const_red,  optimizations='basic', symbols=sp.numbered_symbols('x'))

    global_repl, global_red = _replace_after_cse(global_repl, global_red, _replace_sign)
    global_repl, global_red = _replace_after_cse(global_repl, global_red, _replace_hypot)

    return input_syms, all_syms, const_repl, global_repl, global_red, [], param_map, {} 


# ── Data types ────────────────────────────────────────────────────────────────

class AnglePair(NamedTuple):
    s: sp.Symbol
    c: sp.Symbol

@dataclass
class RawAssignment:
    """One (output_sym, raw_expr) pair in dependency order."""
    sym: sp.Symbol
    expr: sp.Expr

@dataclass
class BranchCtx:
    i_q1: int
    i_q5: int
    i_q3: int
    q1:   AnglePair
    q5:   AnglePair
    q3:   AnglePair        # ← add this
    th14: AnglePair
    q6:   AnglePair
    d_inner: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    d3:  sp.Symbol

# ── Stage functions ───────────────────────────────────────────────────────────

def _stage_q1(p_06, H, P, raw) -> list[AnglePair]:
    d1 = H[1].dot(P[1] + P[2] + P[3] + P[4])
    (s0, c0), (s1, c1) = _sp4(p_06, -H[0], H[1], d1)
    pairs = []
    for i, (se, ce) in enumerate([(s0, c0), (s1, c1)]):
        s = sp.Symbol(f's_t1_{i}', real=True)
        c = sp.Symbol(f'c_t1_{i}', real=True)
        raw += [RawAssignment(s, se), RawAssignment(c, ce)]
        pairs.append(AnglePair(s, c))
    return pairs


def _stage_q5(q1: AnglePair, i_q1: int, H, R_06, raw) -> list[AnglePair]:
    R_01 = _sym_rot(H[0], q1.s, q1.c)
    d5   = H[1].dot(R_01.T * R_06 * H[5])
    (s0, c0), (s1, c1) = _sp4(H[5], H[4], H[1], d5)
    pairs = []
    for i, (se, ce) in enumerate([(s0, c0), (s1, c1)]):
        s = sp.Symbol(f's_t5_{i}_q1{i_q1}', real=True)
        c = sp.Symbol(f'c_t5_{i}_q1{i_q1}', real=True)
        raw += [RawAssignment(s, se), RawAssignment(c, ce)]
        pairs.append(AnglePair(s, c))
    return pairs


def _stage_th14_q6_d(
    q1: AnglePair, q5: AnglePair,
    i_q1: int, i_q5: int,
    p_06, H, P, R_06, raw,
) -> tuple[AnglePair, AnglePair, tuple, sp.Symbol, dict]:
    R_01 = _sym_rot(H[0], q1.s, q1.c)
    R_45 = _sym_rot(H[4], q5.s, q5.c)

    tag = f'q1{i_q1}_q5{i_q5}'

    s_th14_e, c_th14_e = _sp1(R_45 * H[5],   R_01.T * R_06 * H[5], H[1])
    s_q6_e,   c_q6_e   = _sp1(R_45.T * H[1], R_06.T * R_01 * H[1], -H[5])
    s_th14 = sp.Symbol(f's_th14_{tag}', real=True)
    c_th14 = sp.Symbol(f'c_th14_{tag}', real=True)
    s_q6   = sp.Symbol(f's_q6_{tag}',   real=True)
    c_q6   = sp.Symbol(f'c_q6_{tag}',   real=True)
    raw += [
        RawAssignment(s_th14, s_th14_e), RawAssignment(c_th14, c_th14_e),
        RawAssignment(s_q6,   s_q6_e),   RawAssignment(c_q6,   c_q6_e),
    ]
    th14 = AnglePair(s_th14, c_th14)
    q6   = AnglePair(s_q6,   c_q6)

    # Lazy d_inner: defer until after simplification
    d_inner_exprs = R_01.T * p_06 - P[1] - _sym_rot(H[1], s_th14, c_th14) * P[4]
    di0 = sp.Symbol(f'di0_{tag}', real=True)
    di1 = sp.Symbol(f'di1_{tag}', real=True)
    di2 = sp.Symbol(f'di2_{tag}', real=True)
    d3  = sp.Symbol(f'd3_{tag}',  real=True)
    lazy = {
        di0: d_inner_exprs[0],
        di1: d_inner_exprs[1],
        di2: d_inner_exprs[2],
        d3:  sp.sqrt(d_inner_exprs[0]**2 + d_inner_exprs[1]**2 + d_inner_exprs[2]**2),
    }
    return th14, q6, (di0, di1, di2), d3, lazy


def _stage_q3(d3: sp.Symbol, i_q1: int, i_q5: int, H, P, raw) -> list[AnglePair]:
    tag = f'q1{i_q1}_q5{i_q5}'
    (s0, c0), (s1, c1) = _sp3(-P[3], P[2], H[1], d3)
    pairs = []
    for i, (se, ce) in enumerate([(s0, c0), (s1, c1)]):
        s = sp.Symbol(f's_t3_{i}_{tag}', real=True)
        c = sp.Symbol(f'c_t3_{i}_{tag}', real=True)
        raw += [RawAssignment(s, se), RawAssignment(c, ce)]
        pairs.append(AnglePair(s, c))
    return pairs


def _stage_q2_q4(ctx: BranchCtx, H, P, raw) -> tuple[AnglePair, AnglePair]:
    tag = f'b{ctx.i_q1}{ctx.i_q5}{ctx.i_q3}'
    d_inner_sym = sp.Matrix(list(ctx.d_inner))

    s_q2_e, c_q2_e = _sp1(
        P[2] + _sym_rot(H[1], ctx.q3.s, ctx.q3.c) * P[3],
        d_inner_sym, H[1]
    )
    s_14_2 = ctx.th14.s * c_q2_e - ctx.th14.c * s_q2_e
    c_14_2 = ctx.th14.c * c_q2_e + ctx.th14.s * s_q2_e
    s_q4_e = s_14_2 * ctx.q3.c - c_14_2 * ctx.q3.s
    c_q4_e = c_14_2 * ctx.q3.c + s_14_2 * ctx.q3.s

    s_q2 = sp.Symbol(f's_q2_{tag}', real=True)
    c_q2 = sp.Symbol(f'c_q2_{tag}', real=True)
    s_q4 = sp.Symbol(f's_q4_{tag}', real=True)
    c_q4 = sp.Symbol(f'c_q4_{tag}', real=True)
    raw += [
        RawAssignment(s_q2, s_q2_e), RawAssignment(c_q2, c_q2_e),
        RawAssignment(s_q4, s_q4_e), RawAssignment(c_q4, c_q4_e),
    ]
    return AnglePair(s_q2, c_q2), AnglePair(s_q4, c_q4)


def _deduplicate_outputs(all_syms, all_exprs, constant_syms: set = frozenset()):
    """
    Detects aliases of the form: sym = k * ref_sym
    where k is any expression involving only constant_syms (or a bare number).
    Defaults to ±1 detection only (constant_syms=frozenset()).
    """
    def canonical_key(expr):
        return -expr if expr.could_extract_minus_sign() else expr

    def extract_constant_ratio(expr, ref_expr):
        ratio = sp.cancel(expr / ref_expr)
        if ratio.free_symbols <= constant_syms:
            return ratio
        return None

    canon_to_kept: dict = {}
    alias_map = {}
    kept_syms  = []
    kept_exprs = []

    for sym, expr in zip(all_syms, all_exprs):
        key = canonical_key(expr)
        if key in canon_to_kept:
            ref_sym, ref_expr = canon_to_kept[key]
            ratio = extract_constant_ratio(expr, ref_expr)
            if ratio is not None:
                alias_map[sym] = (ratio, ref_sym)
                continue

        canon_to_kept[key] = (sym, expr)
        kept_syms.append(sym)
        kept_exprs.append(expr)

    return kept_syms, kept_exprs, alias_map

def _simplify_pipeline(exprs):
    exprs = [sp.trigsimp(e) for e in exprs]
    exprs = [sp.simplify(e) for e in exprs]
    exprs = [sp.trigsimp(e) for e in exprs]
    return exprs

def sympy_3p2i_ik(kin):
    H_num = kin["H"]
    P_num = kin["P"]
    RT_num = kin["R_6T"]
    H, P, RT, param_map = _generate_param_symbols(H_num, P_num, RT_num)

    R_syms = [[sp.Symbol(f'r{i+1}{j+1}', real=True) for j in range(3)] for i in range(3)]
    p_syms = [sp.Symbol(f'p{i+1}', real=True) for i in range(3)]
    R_06   = sp.Matrix(R_syms) @ RT.T
    p_0T   = sp.Matrix(p_syms)
    input_syms = [R_syms[i][j] for i in range(3) for j in range(3)] + p_syms
    p_06   = p_0T - P[0] - R_06 * P[6]

    raw: list[RawAssignment] = []
    lazy_sub_map = {}
    branches: list[BranchCtx] = []

    # Stage 1: q1
    q1_pairs = _stage_q1(p_06, H, P, raw)

    # Stage 2-4: q5, th14/q6/d, q3 — one pass per (q1, q5) combo
    q5_per_q1   = {}
    th14_q6_per = {}
    q3_per      = {}

    for i_q1, q1 in enumerate(q1_pairs):
        q5_per_q1[i_q1] = _stage_q5(q1, i_q1, H, R_06, raw)

        for i_q5, q5 in enumerate(q5_per_q1[i_q1]):
            th14, q6, d_inner, d3, lazy = _stage_th14_q6_d(
                q1, q5, i_q1, i_q5, p_06, H, P, R_06, raw
            )
            lazy_sub_map.update(lazy)
            th14_q6_per[(i_q1, i_q5)] = (th14, q6, d_inner, d3)
            q3_per[(i_q1, i_q5)] = _stage_q3(d3, i_q1, i_q5, H, P, raw)

    # Stage 5: q2, q4 — flat branch list
    for i_q1, q1 in enumerate(q1_pairs):
        for i_q5, q5 in enumerate(q5_per_q1[i_q1]):
            th14, q6, d_inner, d3 = th14_q6_per[(i_q1, i_q5)]
            for i_q3, q3 in enumerate(q3_per[(i_q1, i_q5)]):
                ctx = BranchCtx(i_q1, i_q5, i_q3, q1, q5, q3, th14, q6, d_inner, d3)
                # ctx.q3 = q3  # attach q3 for use in q2/q4 stage
                q2, q4 = _stage_q2_q4(ctx, H, P, raw)
                branches.append((q1, q2, q3, q4, q5, q6))  # full joint tuple

    # ── Simplification + CSE ──────────────────────────────────────────────────
    all_syms_raw  = [a.sym  for a in raw]
    all_exprs = [a.expr for a in raw]

    all_exprs = _simplify_pipeline(all_exprs)
    all_exprs = [e.subs(lazy_sub_map, simultaneous=True) for e in all_exprs]
    all_exprs = _simplify_pipeline(all_exprs)

    param_syms = set(param_map.keys())
    all_syms, all_exprs, alias_map = _deduplicate_outputs(all_syms_raw, all_exprs, constant_syms=param_syms)

    all_exprs = _simplify_pipeline(all_exprs)
    alias_subs = {sym: coeff * ref for sym, (coeff, ref) in alias_map.items()}
    all_exprs = [e.subs(alias_subs) for e in all_exprs]
    all_exprs = _simplify_pipeline(all_exprs)

    all_exprs = [_replace_sign(e) for e in all_exprs]
    all_exprs = [_replace_hypot(e) for e in all_exprs]

    output_syms = set(all_syms_raw)
    blacklist = output_syms | set(input_syms)
    const_repl, const_red   = sp.cse(all_exprs, ignore=blacklist, optimizations='basic', symbols=sp.numbered_symbols('c'))
    global_repl, global_red = sp.cse(const_red,  optimizations='basic', symbols=sp.numbered_symbols('x'))
    
    global_repl, global_red = _replace_after_cse(global_repl, global_red, _replace_sign)
    global_repl, global_red = _replace_after_cse(global_repl, global_red, _replace_hypot)

    return input_syms, all_syms, const_repl, global_repl, global_red, branches, param_map, alias_map