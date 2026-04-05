"""
Debug script: generated IK code embedded directly with print statements,
compared against JAX general solver intermediates.

Run:
    uv run python debug_cse.py [SEED]
"""
import sys
import numpy as np
import jax.numpy as jnp

ROBOT = "UR10e"
SEED  = int(sys.argv[1]) if len(sys.argv) > 1 else 0

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

rng = np.random.default_rng(SEED)
q_ref = jnp.array(rng.uniform(-np.pi, np.pi, 6))
R_tcp, p_tcp = fk_gen(q_ref)
R_06 = R_tcp @ RT.T

print(f"Test pose (seed={SEED}):")
print(f"  R_06 =\n{np.round(np.asarray(R_06), 4)}")
print(f"  p_0T = {np.round(np.asarray(p_tcp), 4)}")
print("=" * 70)

# ── JAX general solver trace ──────────────────────────────────────────────────
print("JAX general solver — all 8 branches:")
p_06_jax = p_tcp - P[:, 0] - R_06 @ P[:, 6]
d1_jax   = H[:, 1] @ P[:, 1:5].sum(axis=1)
t1s_jax  = sp4(p_06_jax, -H[:, 0], H[:, 1], d1_jax)
print(f"  p_06={np.round(np.asarray(p_06_jax),4)}  d1={float(d1_jax):.6f}")
print(f"  t1s={np.round(np.asarray(t1s_jax),4)}")

jax_branches = {}
b = 0
for i_q1, q1 in enumerate(t1s_jax):
    R_01 = _rot(H[:, 0], q1)
    d5   = H[:, 1] @ R_01.T @ R_06 @ H[:, 5]
    t5s  = sp4(H[:, 5], H[:, 4], H[:, 1], d5)
    for i_q5, q5 in enumerate(t5s):
        R_45 = _rot(H[:, 4], q5)
        th14 = sp1(R_45 @ H[:, 5], R_01.T @ R_06 @ H[:, 5], H[:, 1])
        q6   = sp1(R_45.T @ H[:, 1], R_06.T @ R_01 @ H[:, 1], -H[:, 5])
        d_inner = R_01.T @ p_06_jax - P[:, 1] - _rot(H[:, 1], th14) @ P[:, 4]
        d3 = jnp.linalg.norm(d_inner)
        t3s = sp3(-P[:, 3], P[:, 2], H[:, 1], d3)
        for i_q3, q3 in enumerate(t3s):
            q2 = sp1(P[:, 2] + _rot(H[:, 1], q3) @ P[:, 3], d_inner, H[:, 1])
            q4 = float((th14 - q2 - q3 + jnp.pi) % (2*jnp.pi) - jnp.pi)
            jax_branches[b] = [float(q1),float(q2),float(q3),q4,float(q5),float(q6)]
            print(f"  b{b}(q1={i_q1},q5={i_q5},q3={i_q3}): "
                  f"th14={float(th14):.4f} d3={float(d3):.4f} "
                  f"d_inner={np.round(np.asarray(d_inner),4)} "
                  f"q={np.round(jax_branches[b],4)}")
            b += 1

# ── CSE embedded with prints ──────────────────────────────────────────────────
print()
print("=" * 70)
print("CSE trace (generated code with prints):")

def ik_ur10e_debug(R_06, p_0T):
    r11 = R_06[0, 0]; r12 = R_06[0, 1]; r13 = R_06[0, 2]
    r21 = R_06[1, 0]; r22 = R_06[1, 1]; r23 = R_06[1, 2]
    r31 = R_06[2, 0]; r32 = R_06[2, 1]; r33 = R_06[2, 2]
    p1 = p_0T[0]; p2 = p_0T[1]; p3 = p_0T[2]

    # ── stage1_q1 ──
    _x0 = 1.0*p2 + 0.11655*r22
    _x1 = 1.0*p1 + 0.11655*r12
    _x2 = jnp.sqrt((_x0 ** 2) + (_x1 ** 2) - 0.03032822250000001)
    _x3 = _x0*_x2
    _x4 = 0.17415000000000003*p1 + 0.020297182500000004*r12
    _x5 = 0.17415000000000003*p2 + 0.020297182500000004*r22
    t1_0 = jnp.arctan2(-_x3 + _x4, -_x1*_x2 - _x5)
    t1_1 = jnp.arctan2(_x3 + _x4, _x1*_x2 - _x5)
    print(f"  t1_0={float(t1_0):.4f}  t1_1={float(t1_1):.4f}")

    # ── stage2_q5 ──
    _x6  = -r12*jnp.sin(t1_0) + r22*jnp.cos(t1_0)
    _x7  = 1.0*jnp.sqrt(1 - (_x6 ** 2))
    _x8  = 1.0*_x6
    t5_0_q10 = jnp.arctan2(_x7, _x8)
    t5_1_q10 = jnp.arctan2(-_x7, _x8)
    _x9  = -r12*jnp.sin(t1_1) + r22*jnp.cos(t1_1)
    _x10 = 1.0*jnp.sqrt(1 - (_x9 ** 2))
    _x11 = 1.0*_x9
    t5_0_q11 = jnp.arctan2(_x10, _x11)
    t5_1_q11 = jnp.arctan2(-_x10, _x11)
    print(f"  t5_0_q10={float(t5_0_q10):.4f}  t5_1_q10={float(t5_1_q10):.4f}")
    print(f"  t5_0_q11={float(t5_0_q11):.4f}  t5_1_q11={float(t5_1_q11):.4f}")

    # ── stage3a th14/q6 ──
    _x12 = 1.0*jnp.sin(t5_0_q10); _x13 = jnp.cos(t1_0); _x14 = jnp.sin(t1_0)
    th14_q10_q50 = jnp.arctan2(_x12*r32, _x12*(_x13*r12 + _x14*r22))
    q6_q10_q50   = jnp.arctan2(-_x12*(-_x13*r23 + _x14*r13), _x12*(-_x13*r21 + _x14*r11))

    _x22 = 1.0*jnp.sin(t5_1_q10); _x23 = jnp.cos(t1_0); _x24 = jnp.sin(t1_0)
    th14_q10_q51 = jnp.arctan2(_x22*r32, _x22*(_x23*r12 + _x24*r22))
    q6_q10_q51   = jnp.arctan2(-_x22*(-_x23*r23 + _x24*r13), _x22*(-_x23*r21 + _x24*r11))

    _x32 = 1.0*jnp.sin(t5_0_q11); _x33 = jnp.cos(t1_1); _x34 = jnp.sin(t1_1)
    th14_q11_q50 = jnp.arctan2(_x32*r32, _x32*(_x33*r12 + _x34*r22))
    q6_q11_q50   = jnp.arctan2(-_x32*(-_x33*r23 + _x34*r13), _x32*(-_x33*r21 + _x34*r11))

    _x42 = 1.0*jnp.sin(t5_1_q11); _x43 = jnp.cos(t1_1); _x44 = jnp.sin(t1_1)
    th14_q11_q51 = jnp.arctan2(_x42*r32, _x42*(_x43*r12 + _x44*r22))
    q6_q11_q51   = jnp.arctan2(-_x42*(-_x43*r23 + _x44*r13), _x42*(-_x43*r21 + _x44*r11))

    print(f"  th14_q10_q50={float(th14_q10_q50):.4f}  q6_q10_q50={float(q6_q10_q50):.4f}")
    print(f"  th14_q10_q51={float(th14_q10_q51):.4f}  q6_q10_q51={float(q6_q10_q51):.4f}")
    print(f"  th14_q11_q50={float(th14_q11_q50):.4f}  q6_q11_q50={float(q6_q11_q50):.4f}")
    print(f"  th14_q11_q51={float(th14_q11_q51):.4f}  q6_q11_q51={float(q6_q11_q51):.4f}")

    # ── stage3b d_inner ──
    _x15=jnp.cos(t1_0); _x16=p1+0.11655*r12; _x17=p2+0.11655*r22; _x18=1.0*jnp.sin(t1_0)
    _x19=_x15*_x16+_x17*_x18-0.11984999999999998*jnp.sin(th14_q10_q50)
    _x20=_x15*_x17-_x16*_x18+0.17415000000000003
    _x21=1.0*p3+0.11655*r32+0.11984999999999998*jnp.cos(th14_q10_q50)-0.1807
    di0_q10_q50=_x19; di1_q10_q50=_x20; di2_q10_q50=_x21
    d3_q10_q50=jnp.sqrt((_x19**2)+(_x20**2)+(_x21**2))

    _x25=jnp.cos(t1_0); _x26=p1+0.11655*r12; _x27=p2+0.11655*r22; _x28=1.0*jnp.sin(t1_0)
    _x29=_x25*_x26+_x27*_x28-0.11984999999999998*jnp.sin(th14_q10_q51)
    _x30=_x25*_x27-_x26*_x28+0.17415000000000003
    _x31=1.0*p3+0.11655*r32+0.11984999999999998*jnp.cos(th14_q10_q51)-0.1807
    di0_q10_q51=_x29; di1_q10_q51=_x30; di2_q10_q51=_x31
    d3_q10_q51=jnp.sqrt((_x29**2)+(_x30**2)+(_x31**2))

    _x35=jnp.cos(t1_1); _x36=p1+0.11655*r12; _x37=p2+0.11655*r22; _x38=1.0*jnp.sin(t1_1)
    _x39=_x35*_x36+_x37*_x38-0.11984999999999998*jnp.sin(th14_q11_q50)
    _x40=_x35*_x37-_x36*_x38+0.17415000000000003
    _x41=1.0*p3+0.11655*r32+0.11984999999999998*jnp.cos(th14_q11_q50)-0.1807
    di0_q11_q50=_x39; di1_q11_q50=_x40; di2_q11_q50=_x41
    d3_q11_q50=jnp.sqrt((_x39**2)+(_x40**2)+(_x41**2))

    _x45=jnp.cos(t1_1); _x46=p1+0.11655*r12; _x47=p2+0.11655*r22; _x48=1.0*jnp.sin(t1_1)
    _x49=_x45*_x46+_x47*_x48-0.11984999999999998*jnp.sin(th14_q11_q51)
    _x50=_x45*_x47-_x46*_x48+0.17415000000000003
    _x51=1.0*p3+0.11655*r32+0.11984999999999998*jnp.cos(th14_q11_q51)-0.1807
    di0_q11_q51=_x49; di1_q11_q51=_x50; di2_q11_q51=_x51
    d3_q11_q51=jnp.sqrt((_x49**2)+(_x50**2)+(_x51**2))

    print(f"  d_inner_q10_q50=[{float(di0_q10_q50):.4f},{float(di1_q10_q50):.4f},{float(di2_q10_q50):.4f}]  d3={float(d3_q10_q50):.6f}")
    print(f"  d_inner_q10_q51=[{float(di0_q10_q51):.4f},{float(di1_q10_q51):.4f},{float(di2_q10_q51):.4f}]  d3={float(d3_q10_q51):.6f}")
    print(f"  d_inner_q11_q50=[{float(di0_q11_q50):.4f},{float(di1_q11_q50):.4f},{float(di2_q11_q50):.4f}]  d3={float(d3_q11_q50):.6f}")
    print(f"  d_inner_q11_q51=[{float(di0_q11_q51):.4f},{float(di1_q11_q51):.4f},{float(di2_q11_q51):.4f}]  d3={float(d3_q11_q51):.6f}")

    # ── stage4 q3 ──
    _x52=(d3_q10_q50**2); _x53=1.0*jnp.sqrt(1-2.0386176964492657*(_x52-0.7020706925**2)); _x54=1.427801700674595*_x52-1.0024177287452907
    t3_0_q10_q50=jnp.arctan2(_x53,_x54); t3_1_q10_q50=jnp.arctan2(-_x53,_x54)
    _x55=(d3_q10_q51**2); _x56=1.0*jnp.sqrt(1-2.0386176964492657*(_x55-0.7020706925**2)); _x57=1.427801700674595*_x55-1.0024177287452907
    t3_0_q10_q51=jnp.arctan2(_x56,_x57); t3_1_q10_q51=jnp.arctan2(-_x56,_x57)
    _x58=(d3_q11_q50**2); _x59=1.0*jnp.sqrt(1-2.0386176964492657*(_x58-0.7020706925**2)); _x60=1.427801700674595*_x58-1.0024177287452907
    t3_0_q11_q50=jnp.arctan2(_x59,_x60); t3_1_q11_q50=jnp.arctan2(-_x59,_x60)
    _x61=(d3_q11_q51**2); _x62=1.0*jnp.sqrt(1-2.0386176964492657*(_x61-0.7020706925**2)); _x63=1.427801700674595*_x61-1.0024177287452907
    t3_0_q11_q51=jnp.arctan2(_x62,_x63); t3_1_q11_q51=jnp.arctan2(-_x62,_x63)

    print(f"  t3_q10_q50=[{float(t3_0_q10_q50):.4f},{float(t3_1_q10_q50):.4f}]")
    print(f"  t3_q10_q51=[{float(t3_0_q10_q51):.4f},{float(t3_1_q10_q51):.4f}]")
    print(f"  t3_q11_q50=[{float(t3_0_q11_q50):.4f},{float(t3_1_q11_q50):.4f}]")
    print(f"  t3_q11_q51=[{float(t3_0_q11_q51):.4f},{float(t3_1_q11_q51):.4f}]")

    # ── stage5 q2/q4 ──
    _x64=0.57155*jnp.sin(t3_0_q10_q50); _x65=0.57155*jnp.cos(t3_0_q10_q50)+0.6127
    _x66=jnp.arctan2(_x64*di0_q10_q50-_x65*di2_q10_q50,-_x64*di2_q10_q50-_x65*di0_q10_q50)
    q2_b0=_x66; q4_b0=-_x66-t3_0_q10_q50+th14_q10_q50

    _x67=0.57155*jnp.sin(t3_1_q10_q50); _x68=0.57155*jnp.cos(t3_1_q10_q50)+0.6127
    _x69=jnp.arctan2(_x67*di0_q10_q50-_x68*di2_q10_q50,-_x67*di2_q10_q50-_x68*di0_q10_q50)
    q2_b1=_x69; q4_b1=-_x69-t3_1_q10_q50+th14_q10_q50

    _x70=0.57155*jnp.sin(t3_0_q10_q51); _x71=0.57155*jnp.cos(t3_0_q10_q51)+0.6127
    _x72=jnp.arctan2(_x70*di0_q10_q51-_x71*di2_q10_q51,-_x70*di2_q10_q51-_x71*di0_q10_q51)
    q2_b2=_x72; q4_b2=-_x72-t3_0_q10_q51+th14_q10_q51

    _x73=0.57155*jnp.sin(t3_1_q10_q51); _x74=0.57155*jnp.cos(t3_1_q10_q51)+0.6127
    _x75=jnp.arctan2(_x73*di0_q10_q51-_x74*di2_q10_q51,-_x73*di2_q10_q51-_x74*di0_q10_q51)
    q2_b3=_x75; q4_b3=-_x75-t3_1_q10_q51+th14_q10_q51

    _x76=0.57155*jnp.sin(t3_0_q11_q50); _x77=0.57155*jnp.cos(t3_0_q11_q50)+0.6127
    _x78=jnp.arctan2(_x76*di0_q11_q50-_x77*di2_q11_q50,-_x76*di2_q11_q50-_x77*di0_q11_q50)
    q2_b4=_x78; q4_b4=-_x78-t3_0_q11_q50+th14_q11_q50

    _x79=0.57155*jnp.sin(t3_1_q11_q50); _x80=0.57155*jnp.cos(t3_1_q11_q50)+0.6127
    _x81=jnp.arctan2(_x79*di0_q11_q50-_x80*di2_q11_q50,-_x79*di2_q11_q50-_x80*di0_q11_q50)
    q2_b5=_x81; q4_b5=-_x81-t3_1_q11_q50+th14_q11_q50

    _x82=0.57155*jnp.sin(t3_0_q11_q51); _x83=0.57155*jnp.cos(t3_0_q11_q51)+0.6127
    _x84=jnp.arctan2(_x82*di0_q11_q51-_x83*di2_q11_q51,-_x82*di2_q11_q51-_x83*di0_q11_q51)
    q2_b6=_x84; q4_b6=-_x84-t3_0_q11_q51+th14_q11_q51

    _x85=0.57155*jnp.sin(t3_1_q11_q51); _x86=0.57155*jnp.cos(t3_1_q11_q51)+0.6127
    _x87=jnp.arctan2(_x85*di0_q11_q51-_x86*di2_q11_q51,-_x85*di2_q11_q51-_x86*di0_q11_q51)
    q2_b7=_x87; q4_b7=-_x87-t3_1_q11_q51+th14_q11_q51

    _b0=jnp.array([t1_0,q2_b0,t3_0_q10_q50,q4_b0,t5_0_q10,q6_q10_q50])
    _b1=jnp.array([t1_0,q2_b1,t3_1_q10_q50,q4_b1,t5_0_q10,q6_q10_q50])
    _b2=jnp.array([t1_0,q2_b2,t3_0_q10_q51,q4_b2,t5_1_q10,q6_q10_q51])
    _b3=jnp.array([t1_0,q2_b3,t3_1_q10_q51,q4_b3,t5_1_q10,q6_q10_q51])
    _b4=jnp.array([t1_1,q2_b4,t3_0_q11_q50,q4_b4,t5_0_q11,q6_q11_q50])
    _b5=jnp.array([t1_1,q2_b5,t3_1_q11_q50,q4_b5,t5_0_q11,q6_q11_q50])
    _b6=jnp.array([t1_1,q2_b6,t3_0_q11_q51,q4_b6,t5_1_q11,q6_q11_q51])
    _b7=jnp.array([t1_1,q2_b7,t3_1_q11_q51,q4_b7,t5_1_q11,q6_q11_q51])

    Q=jnp.stack([_b0,_b1,_b2,_b3,_b4,_b5,_b6,_b7],axis=1)
    valid=~jnp.isnan(Q).any(axis=0)

    print()
    print("  CSE branches:")
    for bi in range(8):
        print(f"  b{bi}: {np.round(np.asarray(Q[:,bi]),4)}")

    return Q, valid

Q_cse, valid_cse = ik_ur10e_debug(R_06, p_tcp)

# ── comparison ────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Diff (JAX vs CSE, only mismatches):")
joint_names = ['q1','q2','q3','q4','q5','q6']
for b in range(8):
    jq = jax_branches[b]
    cq = np.asarray(Q_cse[:, b])
    for ji, jname in enumerate(joint_names):
        diff = abs(float(np.arctan2(np.sin(jq[ji]-cq[ji]), np.cos(jq[ji]-cq[ji]))))
        if diff > 1e-3:
            print(f"  b{b} {jname}: JAX={jq[ji]:.4f}  CSE={cq[ji]:.4f}  diff={diff:.4f}")

print()
print("FK roundtrip (CSE):")
for b in range(8):
    if not bool(np.asarray(valid_cse)[b]):
        print(f"  b{b}: invalid"); continue
    q_sol = jnp.array(np.asarray(Q_cse[:, b]))
    R_chk, p_chk = fk_gen(q_sol)
    p_err = float(jnp.linalg.norm(p_chk - p_tcp))
    R_err = float(jnp.linalg.norm(R_chk - R_tcp))
    ok = "✓" if p_err < 1e-4 and R_err < 1e-4 else "✗"
    print(f"  b{b}: p_err={p_err:.2e}  R_err={R_err:.2e}  {ok}")