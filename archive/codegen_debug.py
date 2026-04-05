"""
Debug script: trace all 8 IK branches for both JAX general solver and CSE,
compare intermediate values side by side.

Run:
    uv run python debug_cse.py
"""
import numpy as np
import jax.numpy as jnp
import jax
import importlib

ROBOT = "UR10e"
SEED  = 0

# ── setup ─────────────────────────────────────────────────────────────────────
from jaik.kinematics.robots import make_robot, _UR_PARAMS, _UR_R6T
from jaik.kinematics.convention_conversions import dh_to_kin
from jaik.kinematics.adjustments import adjust_kin_for_3p2i
from jaik._jax.subproblems import sp1, sp3, sp4
from jaik._jax.utils import _rot

p = _UR_PARAMS[ROBOT]
a = np.array([0, p["a2"], p["a3"], 0, 0, 0])
d = np.array([p["d1"], 0, 0, p["d4"], p["d5"], p["d6"]])
alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
kin = adjust_kin_for_3p2i(dh_to_kin(alpha, a, d))
H = jnp.array(kin['H'])
P = jnp.array(kin['P'])
RT = jnp.array(_UR_R6T)

fk_gen, _ = make_robot(ROBOT, solver="general")
mod = importlib.import_module(f"jaik._jax._generated.ik_{ROBOT.lower()}")
ik_cse = getattr(mod, f"ik_{ROBOT.lower()}")

# ── test pose ─────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
q_ref = jnp.array(rng.uniform(-np.pi, np.pi, 6))
R_tcp, p_tcp = fk_gen(q_ref)
R_06 = R_tcp @ RT.T

print(f"Test pose (seed={SEED}):")
print(f"  R_06 =\n{np.round(np.asarray(R_06), 4)}")
print(f"  p_0T = {np.round(np.asarray(p_tcp), 4)}")
print("=" * 70)


# ── JAX general solver — all branches ─────────────────────────────────────────

def jax_trace_all():
    p_06 = p_tcp - P[:, 0] - R_06 @ P[:, 6]
    d1   = H[:, 1] @ P[:, 1:5].sum(axis=1)
    t1s  = sp4(p_06, -H[:, 0], H[:, 1], d1)

    results = {}
    branch = 0
    for i_q1, q1 in enumerate(t1s):
        R_01 = _rot(H[:, 0], q1)
        d5   = H[:, 1] @ R_01.T @ R_06 @ H[:, 5]
        t5s  = sp4(H[:, 5], H[:, 4], H[:, 1], d5)

        for i_q5, q5 in enumerate(t5s):
            R_45 = _rot(H[:, 4], q5)
            th14 = sp1(R_45 @ H[:, 5], R_01.T @ R_06 @ H[:, 5], H[:, 1])
            q6   = sp1(R_45.T @ H[:, 1], R_06.T @ R_01 @ H[:, 1], -H[:, 5])

            d_inner = R_01.T @ p_06 - P[:, 1] - _rot(H[:, 1], th14) @ P[:, 4]
            d3      = jnp.linalg.norm(d_inner)
            t3s     = sp3(-P[:, 3], P[:, 2], H[:, 1], d3)

            for i_q3, q3 in enumerate(t3s):
                q2 = sp1(P[:, 2] + _rot(H[:, 1], q3) @ P[:, 3], d_inner, H[:, 1])
                q4 = float((th14 - q2 - q3 + jnp.pi) % (2*jnp.pi) - jnp.pi)
                results[branch] = {
                    'q1': float(q1), 'q2': float(q2), 'q3': float(q3),
                    'q4': q4,        'q5': float(q5), 'q6': float(q6),
                    'th14': float(th14), 'd3': float(d3),
                    'd_inner': np.asarray(d_inner),
                    'i_q1': i_q1, 'i_q5': i_q5, 'i_q3': i_q3,
                }
                branch += 1
    return results


# ── sympy/CSE trace — evaluate generated code symbolically ────────────────────

def cse_trace_all():
    """Evaluate CSE by running ik_cse and also tracing intermediates."""
    Q_cse, valid_cse = ik_cse(R_06, p_tcp)
    results = {}
    for b in range(8):
        q = np.asarray(Q_cse[:, b])
        results[b] = {
            'q1': q[0], 'q2': q[1], 'q3': q[2],
            'q4': q[3], 'q5': q[4], 'q6': q[5],
            'valid': bool(np.asarray(valid_cse)[b]),
        }
    return results, Q_cse, valid_cse


# ── compare ───────────────────────────────────────────────────────────────────

print("JAX general solver — all 8 branches:")
jax_res = jax_trace_all()
for b, r in jax_res.items():
    q = [r['q1'], r['q2'], r['q3'], r['q4'], r['q5'], r['q6']]
    print(f"  branch {b} (q1={r['i_q1']} q5={r['i_q5']} q3={r['i_q3']}): "
          f"{np.round(q, 4)}")

print()
print("CSE — all 8 branches:")
cse_res, Q_cse, valid_cse = cse_trace_all()
for b, r in cse_res.items():
    q = [r['q1'], r['q2'], r['q3'], r['q4'], r['q5'], r['q6']]
    print(f"  branch {b} (valid={r['valid']}): {np.round(q, 4)}")

print()
print("Comparison (JAX vs CSE) per branch:")
print(f"  {'br':>2}  {'joint':>5}  {'JAX':>10}  {'CSE':>10}  {'diff':>10}  match")
print(f"  {'-'*2}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  -----")
joint_names = ['q1','q2','q3','q4','q5','q6']
for b in range(8):
    jr = jax_res[b]
    cr = cse_res[b]
    for jname in joint_names:
        jval = jr[jname]
        cval = cr[jname]
        diff = abs(float(np.arctan2(np.sin(jval - cval), np.cos(jval - cval))))
        match = "✓" if diff < 1e-3 else "✗"
        if diff > 1e-3:
            print(f"  {b:>2}  {jname:>5}  {jval:>10.4f}  {cval:>10.4f}  {diff:>10.4f}  {match}")

print()
print("FK roundtrip check (CSE solutions):")
for b in range(8):
    if not cse_res[b]['valid']:
        print(f"  branch {b}: invalid (NaN)")
        continue
    q_sol = jnp.array([cse_res[b][j] for j in joint_names])
    R_chk, p_chk = fk_gen(q_sol)
    p_err = float(jnp.linalg.norm(p_chk - p_tcp))
    R_err = float(jnp.linalg.norm(R_chk - R_tcp))
    ok = "✓" if p_err < 1e-4 and R_err < 1e-4 else "✗"
    print(f"  branch {b}: p_err={p_err:.2e}  R_err={R_err:.2e}  {ok}")

print()
print("FK roundtrip check (JAX general solutions):")
for b in range(8):
    r = jax_res[b]
    q_sol = jnp.array([r[j] for j in joint_names])
    R_chk, p_chk = fk_gen(q_sol)
    p_err = float(jnp.linalg.norm(p_chk - p_tcp))
    R_err = float(jnp.linalg.norm(R_chk - R_tcp))
    ok = "✓" if p_err < 1e-4 and R_err < 1e-4 else "✗"
    print(f"  branch {b}: p_err={p_err:.2e}  R_err={R_err:.2e}  {ok}")