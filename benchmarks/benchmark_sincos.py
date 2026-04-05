"""
jaik benchmark — FK and IK with sincos=True, single and batched, JAX and numba.

Usage:
    python benchmark_sincos.py [--robot ur10e] [--n-warmup 50]

Measurements are in microseconds (µs) per call.
Results are printed as a table and optionally saved as benchmark_sincos_results.json.
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────

BATCH_SIZES_JAX   = [4**i for i in range(12)]
BATCH_SIZES_NUMBA = [4**i for i in range(8)]
N_REPEAT  = 200
N_WARMUP  = 50

rng = np.random.default_rng(42)

def _rand_q(n=1):
    return rng.uniform(-np.pi, np.pi, (n, 6)).astype(np.float64)

def _rand_sincos(n=1):
    q = _rand_q(n)
    return np.sin(q), np.cos(q)

def _rand_pose(n=1):
    """Random valid (R, p) pairs via numba FK."""
    import jaik
    fk_np, _, _ = jaik.make_robot("ur10e", solver="numba", sincos=True)
    Rs, ps = [], []
    sq_all, cq_all = _rand_sincos(n)
    for sq, cq in zip(sq_all, cq_all):
        R, p = fk_np(sq, cq)
        Rs.append(R); ps.append(p)
    return np.stack(Rs), np.stack(ps)

def _sink(*args):
    total = 0.0
    for a in args:
        a = np.asarray(a)
        total += float(a.ravel()[0])
    return total

def _fmt(µs):
    if µs < 1e3:
        return f"{µs:8.2f} µs"
    elif µs < 1e6:
        return f"{µs/1e3:8.2f} ms"
    else:
        return f"{µs/1e6:8.2f}  s"

def _timer(fn, n_repeat):
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e6

# ── JAX benchmarks ────────────────────────────────────────────────────────────

def bench_jax(robot, results, n_warmup=N_WARMUP):
    import jax
    import jax.numpy as jnp

    devices = {"cpu": jax.devices("cpu")[0]}
    try:
        gpu = jax.devices("gpu")
        if gpu:
            devices["gpu"] = gpu[0]
    except Exception:
        pass

    for dev_name, device in devices.items():
        print(f"\n  JAX / {dev_name.upper()}")
        fk, ik_full, _ = robot

        # ── single call ───────────────────────────────────────────────────────
        sq_s, cq_s = _rand_sincos(1)
        sq_s = jnp.array(sq_s[0], device=device)
        cq_s = jnp.array(cq_s[0], device=device)
        R_s  = jnp.array(_rand_pose(1)[0][0], device=device)
        p_s  = jnp.array(_rand_pose(1)[1][0], device=device)

        fk_jit  = jax.jit(fk,      backend=dev_name)
        ik_jit  = jax.jit(ik_full, backend=dev_name)

        for _ in range(n_warmup):
            [x.block_until_ready() for x in jax.tree_util.tree_leaves(fk_jit(sq_s, cq_s))]
            [x.block_until_ready() for x in jax.tree_util.tree_leaves(ik_jit(R_s, p_s))]

        def _fk_single():
            R, p = fk_jit(sq_s, cq_s)
            R.block_until_ready()

        def _ik_single():
            sQ, cQ, v = ik_jit(R_s, p_s)
            sQ.block_until_ready()

        key = f"jax_{dev_name}"
        t_fk = _timer(_fk_single, N_REPEAT)
        t_ik = _timer(_ik_single, N_REPEAT)
        results[key]["fk"][1] = t_fk
        results[key]["ik"][1] = t_ik
        print(f"    fk  single : {_fmt(t_fk)}")
        print(f"    ik  single : {_fmt(t_ik)}")

        # ── batched via vmap ──────────────────────────────────────────────────
        fk_vmap = jax.jit(jax.vmap(fk),      backend=dev_name)
        ik_vmap = jax.jit(jax.vmap(ik_full), backend=dev_name)

        for B in BATCH_SIZES_JAX:
            sq_b, cq_b = _rand_sincos(B)
            sq_b = jnp.array(sq_b, device=device)
            cq_b = jnp.array(cq_b, device=device)
            Rs_b = jnp.array(_rand_pose(B)[0], device=device)
            ps_b = jnp.array(_rand_pose(B)[1], device=device)

            for _ in range(5):
                [x.block_until_ready() for x in jax.tree_util.tree_leaves(fk_vmap(sq_b, cq_b))]
                [x.block_until_ready() for x in jax.tree_util.tree_leaves(ik_vmap(Rs_b, ps_b))]

            def _fk_batch():
                R, p = fk_vmap(sq_b, cq_b)
                R.block_until_ready()

            def _ik_batch():
                sQ, cQ, v = ik_vmap(Rs_b, ps_b)
                sQ.block_until_ready()

            t_fk = _timer(_fk_batch, N_REPEAT) / B
            t_ik = _timer(_ik_batch, N_REPEAT) / B
            results[key]["fk"][B] = t_fk
            results[key]["ik"][B] = t_ik
            print(f"    fk  B={B:<5}: {_fmt(t_fk)}/call")
            print(f"    ik  B={B:<5}: {_fmt(t_ik)}/call")

# ── numba benchmarks ──────────────────────────────────────────────────────────

def bench_numba(robot, results, n_warmup=N_WARMUP):
    from numba import njit, prange

    fk, ik_full, _ = robot

    print("\n  Numba / CPU")
    sq_s, cq_s = _rand_sincos(1)
    sq_s, cq_s = sq_s[0], cq_s[0]
    Rs, ps = _rand_pose(1)
    R_s, p_s = Rs[0], ps[0]

    for _ in range(n_warmup):
        _sink(*fk(sq_s, cq_s))
        _sink(*ik_full(R_s, p_s))

    t_fk = _timer(lambda: _sink(*fk(sq_s, cq_s)),          N_REPEAT)
    t_ik = _timer(lambda: _sink(*ik_full(R_s, p_s)),        N_REPEAT)
    results["numba_serial"]["fk"][1] = t_fk
    results["numba_serial"]["ik"][1] = t_ik
    results["numba_prange"]["fk"][1] = t_fk
    results["numba_prange"]["ik"][1] = t_ik
    print(f"    fk  single : {_fmt(t_fk)}")
    print(f"    ik  single : {_fmt(t_ik)}")

    @njit
    def _fk_serial(sqs, cqs):
        N = sqs.shape[0]
        Rs = np.empty((N, 3, 3))
        ps = np.empty((N, 3))
        for i in range(N):
            R, p = fk(sqs[i], cqs[i])
            Rs[i] = R
            ps[i] = p
        return Rs, ps

    @njit
    def _ik_serial(Rs, ps):
        N = Rs.shape[0]
        sQs   = np.empty((N, 6, 8))
        cQs   = np.empty((N, 6, 8))
        valids = np.empty((N, 8), dtype=np.bool_)
        for i in range(N):
            sQ, cQ, v = ik_full(Rs[i], ps[i])
            sQs[i]    = sQ
            cQs[i]    = cQ
            valids[i] = v
        return sQs, cQs, valids

    @njit(parallel=True)
    def _fk_prange(sqs, cqs):
        N = sqs.shape[0]
        Rs = np.empty((N, 3, 3))
        ps = np.empty((N, 3))
        for i in prange(N):
            R, p = fk(sqs[i], cqs[i])
            Rs[i] = R
            ps[i] = p
        return Rs, ps

    @njit(parallel=True)
    def _ik_prange(Rs, ps):
        N = Rs.shape[0]
        sQs   = np.empty((N, 6, 8))
        cQs   = np.empty((N, 6, 8))
        valids = np.empty((N, 8), dtype=np.bool_)
        for i in prange(N):
            sQ, cQ, v = ik_full(Rs[i], ps[i])
            sQs[i]    = sQ
            cQs[i]    = cQ
            valids[i] = v
        return sQs, cQs, valids

    for B in BATCH_SIZES_NUMBA:
        sq_b, cq_b = _rand_sincos(B)
        Rs_b, ps_b = _rand_pose(B)

        for _ in range(5):
            _sink(*_fk_serial(sq_b, cq_b))
            _sink(*_ik_serial(Rs_b, ps_b))
            _sink(*_fk_prange(sq_b, cq_b))
            _sink(*_ik_prange(Rs_b, ps_b))

        t_fk_s = _timer(lambda: _sink(*_fk_serial(sq_b, cq_b)),        N_REPEAT) / B
        t_ik_s = _timer(lambda: _sink(*_ik_serial(Rs_b, ps_b)),         N_REPEAT) / B
        t_fk_p = _timer(lambda: _sink(*_fk_prange(sq_b, cq_b)),        N_REPEAT) / B
        t_ik_p = _timer(lambda: _sink(*_ik_prange(Rs_b, ps_b)),         N_REPEAT) / B

        results["numba_serial"]["fk"][B] = t_fk_s
        results["numba_serial"]["ik"][B] = t_ik_s
        results["numba_prange"]["fk"][B] = t_fk_p
        results["numba_prange"]["ik"][B] = t_ik_p

        print(f"    fk  B={B:<5}: serial {_fmt(t_fk_s)}/call   prange {_fmt(t_fk_p)}/call")
        print(f"    ik  B={B:<5}: serial {_fmt(t_ik_s)}/call   prange {_fmt(t_ik_p)}/call")

# ── summary table ─────────────────────────────────────────────────────────────

def _print_table(results):
    solvers = list(results.keys())
    sizes   = [1] + BATCH_SIZES_JAX
    print("\n" + "═" * 80)
    print("  SUMMARY — µs per call (median)")
    print("═" * 80)
    for op in ("fk", "ik"):
        print(f"\n  {op.upper()}")
        header = f"  {'B':>6}  " + "  ".join(f"{s:>16}" for s in solvers)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for B in sizes:
            row = f"  {B:>6}  "
            for s in solvers:
                val = results[s][op].get(B)
                row += f"  {_fmt(val) if val else '         ---':>16}"
            print(row)
    print("\n" + "═" * 80)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot",    default="ur10e")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--save",     action="store_true",
                        help="Save results to benchmark_sincos_results.json")
    args = parser.parse_args()

    import jaik
    print(f"\njaik sincos benchmark — {args.robot}")
    print(f"Batch sizes: JAX: {BATCH_SIZES_JAX}  Numba: {BATCH_SIZES_NUMBA}  Repeats: {N_REPEAT}  Warmup: {args.n_warmup}\n")

    results = defaultdict(lambda: {"fk": {}, "ik": {}})

    print("── JAX ─────────────────────────────────────────────────────────────")
    jax_robot = jaik.make_robot(args.robot, solver="jax", sincos=True)
    bench_jax(jax_robot, results, n_warmup=args.n_warmup)

    print("\n── Numba ───────────────────────────────────────────────────────────")
    numba_robot = jaik.make_robot(args.robot, solver="numba", sincos=True)
    bench_numba(numba_robot, results, n_warmup=args.n_warmup)

    _print_table(results)

    if args.save:
        out = {k: {"fk": dict(v["fk"]), "ik": dict(v["ik"])} for k, v in results.items()}
        with open("benchmark_sincos_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print("  Saved benchmark_sincos_results.json")

if __name__ == "__main__":
    main()