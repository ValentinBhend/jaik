"""
jaik benchmark — FK and IK, single and batched, JAX and numba.

Usage:
    python benchmark.py [--robot ur10e] [--n-warmup 50]

Measurements are in microseconds (µs) per call.
Results are printed as a table and optionally saved as benchmark_results.json.
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────

BATCH_SIZES_JAX = [4**i for i in range(12)]
BATCH_SIZES_NUMBA = [4**i for i in range(8)]
N_REPEAT    = 200       # outer timing loops for single-call benchmarks
N_WARMUP    = 50        # calls to trigger JIT before timing

rng = np.random.default_rng(42)

def _rand_q(n=1):
    return rng.uniform(-np.pi, np.pi, (n, 6)).astype(np.float64)

def _rand_pose(n=1):
    """Random valid (R, p) pairs via numba FK."""
    import jaik
    fk_np, _, _ = jaik.make_robot("ur10e", solver="numba")
    Rs, ps = [], []
    for q in _rand_q(n):
        R, p = fk_np(q)
        Rs.append(R); ps.append(p)
    return np.stack(Rs), np.stack(ps)

def _sink(*args):
    """Prevent the compiler from optimizing away results."""
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
    """Return median wall time in µs over n_repeat calls."""
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
        q_s = jnp.array(_rand_q(1)[0], device=device)
        R_s, p_s = jnp.array(_rand_pose(1)[0][0], device=device), \
                   jnp.array(_rand_pose(1)[1][0], device=device)

        fk_jit  = jax.jit(fk,      backend=dev_name)
        ik_jit  = jax.jit(ik_full, backend=dev_name)

        # warmup (triggers compilation)
        for _ in range(n_warmup):
            fk_jit(q_s).R.block_until_ready() if hasattr(fk_jit(q_s), 'R') \
                else [x.block_until_ready() for x in jax.tree_util.tree_leaves(fk_jit(q_s))]
            [x.block_until_ready() for x in jax.tree_util.tree_leaves(ik_jit(R_s, p_s))]

        q_pool = jnp.array(_rand_q(N_REPEAT), device=device)

        def _fk_single_fair():
            times = []
            for i in range(N_REPEAT):
                # 2. Grab a unique row each time
                q_input = q_pool[i] 
                
                t0 = time.perf_counter()
                R, p = fk_jit(q_input)
                R.block_until_ready()
                times.append(time.perf_counter() - t0)
                
            return float(np.median(times)) * 1e6

        # def _fk_single():
        #     R, p = fk_jit(q_s)
        #     R.block_until_ready()

        # def _ik_single():
        #     Q, v = ik_jit(R_s, p_s)
        #     Q.block_until_ready()

        # 1. Pre-allocate pools for the Rotation matrices and Position vectors
        Rs_pool, ps_pool = _rand_pose(N_REPEAT)
        Rs_pool = jnp.array(Rs_pool, device=device)
        ps_pool = jnp.array(ps_pool, device=device)

        def _ik_single_fair():
            times = []
            for i in range(N_REPEAT):
                # 2. Grab the unique pose for this iteration
                R_input = Rs_pool[i]
                p_input = ps_pool[i]
                
                t0 = time.perf_counter()
                # 3. Execute and force synchronization
                Q, v = ik_jit(R_input, p_input)
                Q.block_until_ready()
                
                times.append(time.perf_counter() - t0)
                
            return float(np.median(times)) * 1e6

        t_fk = _fk_single_fair()
        t_ik = _ik_single_fair()
        key = f"jax_{dev_name}"
        results[key]["fk"][1]  = t_fk
        results[key]["ik"][1]  = t_ik
        print(f"    fk  single : {_fmt(t_fk)}")
        print(f"    ik  single : {_fmt(t_ik)}")

        # ── batched via vmap ──────────────────────────────────────────────────
        fk_vmap = jax.jit(jax.vmap(fk),      backend=dev_name)
        ik_vmap = jax.jit(jax.vmap(ik_full), backend=dev_name)

        for B in BATCH_SIZES_JAX:
            # 1. Create a buffer that is B + N_REPEAT long
            # This allows us to take N_REPEAT different slices of size B
            q_pool = jnp.array(_rand_q(B + N_REPEAT), device=device)
            Rs_raw, ps_raw = _rand_pose(B + N_REPEAT)
            Rs_pool = jnp.array(Rs_raw, device=device)
            ps_pool = jnp.array(ps_raw, device=device)

            # warmup (using the first slice)
            for _ in range(5):
                q_warm = q_pool[:B]
                [x.block_until_ready() for x in jax.tree_util.tree_leaves(fk_vmap(q_warm))]

            # 2. Custom timing loop with sliding window
            def _run_batched_fk():
                times = []
                for i in range(N_REPEAT):
                    # Grab a unique slice: every iteration is numerically different
                    q_slice = q_pool[i : i + B]
                    
                    t0 = time.perf_counter()
                    R, p = fk_vmap(q_slice)
                    # Synchronize on the output to ensure math is done
                    R.block_until_ready()
                    times.append(time.perf_counter() - t0)
                return (float(np.median(times)) * 1e6) / B

            def _run_batched_ik():
                times = []
                for i in range(N_REPEAT):
                    R_slice = Rs_pool[i : i + B]
                    p_slice = ps_pool[i : i + B]
                    
                    t0 = time.perf_counter()
                    Q, v = ik_vmap(R_slice, p_slice)
                    Q.block_until_ready()
                    times.append(time.perf_counter() - t0)
                return (float(np.median(times)) * 1e6) / B

            t_fk = _run_batched_fk()
            t_ik = _run_batched_ik()
            results[key]["fk"][B] = t_fk
            results[key]["ik"][B] = t_ik
            print(f"    fk  B={B:<5}: {_fmt(t_fk)}/call")
            print(f"    ik  B={B:<5}: {_fmt(t_ik)}/call")

# ── numba benchmarks ──────────────────────────────────────────────────────────

def bench_numba(robot, results, n_warmup=N_WARMUP):
    from numba import njit, prange

    fk, ik_full, _ = robot

    # ── single call ───────────────────────────────────────────────────────────
    print("\n  Numba / CPU")
    q_s  = _rand_q(1)[0]
    Rs, ps = _rand_pose(1)
    R_s, p_s = Rs[0], ps[0]

    # warmup
    for _ in range(n_warmup):
        _sink(*fk(q_s))
        _sink(*ik_full(R_s, p_s))

    t_fk = _timer(lambda: _sink(*fk(q_s)),         N_REPEAT)
    t_ik = _timer(lambda: _sink(*ik_full(R_s, p_s)), N_REPEAT)
    results["numba_serial"]["fk"][1] = t_fk
    results["numba_serial"]["ik"][1] = t_ik
    results["numba_prange"]["fk"][1] = t_fk   # prange has no effect on B=1
    results["numba_prange"]["ik"][1] = t_ik
    print(f"    fk  single : {_fmt(t_fk)}")
    print(f"    ik  single : {_fmt(t_ik)}")

    # ── batched ───────────────────────────────────────────────────────────────
    # Build serial and prange batch wrappers dynamically after warmup
    # (so numba sees already-compiled fk/ik_full)

    @njit
    def _fk_serial(qs):
        N = qs.shape[0]
        # pre-allocate — call once to get shapes
        R0, p0 = fk(qs[0])
        Rs = np.empty((N, 3, 3))
        ps = np.empty((N, 3))
        for i in range(N):
            R, p = fk(qs[i])
            Rs[i] = R
            ps[i] = p
        return Rs, ps

    @njit
    def _ik_serial(Rs, ps):
        N = Rs.shape[0]
        Qs    = np.empty((N, 6, 8))
        valids = np.empty((N, 8), dtype=np.bool_)
        for i in range(N):
            Q, v = ik_full(Rs[i], ps[i])
            Qs[i]     = Q
            valids[i] = v
        return Qs, valids

    @njit(parallel=True)
    def _fk_prange(qs):
        N = qs.shape[0]
        Rs = np.empty((N, 3, 3))
        ps = np.empty((N, 3))
        for i in prange(N):
            R, p = fk(qs[i])
            Rs[i] = R
            ps[i] = p
        return Rs, ps

    @njit(parallel=True)
    def _ik_prange(Rs, ps):
        N = Rs.shape[0]
        Qs    = np.empty((N, 6, 8))
        valids = np.empty((N, 8), dtype=np.bool_)
        for i in prange(N):
            Q, v = ik_full(Rs[i], ps[i])
            Qs[i]     = Q
            valids[i] = v
        return Qs, valids

    for B in BATCH_SIZES_NUMBA:
        q_b  = _rand_q(B)
        Rs_b, ps_b = _rand_pose(B)

        # warmup batch wrappers
        for _ in range(5):
            _sink(*_fk_serial(q_b))
            _sink(*_ik_serial(Rs_b, ps_b))
            _sink(*_fk_prange(q_b))
            _sink(*_ik_prange(Rs_b, ps_b))

        t_fk_s = _timer(lambda: _sink(*_fk_serial(q_b)),          N_REPEAT) / B
        t_ik_s = _timer(lambda: _sink(*_ik_serial(Rs_b, ps_b)),   N_REPEAT) / B
        t_fk_p = _timer(lambda: _sink(*_fk_prange(q_b)),          N_REPEAT) / B
        t_ik_p = _timer(lambda: _sink(*_ik_prange(Rs_b, ps_b)),   N_REPEAT) / B

        results["numba_serial"]["fk"][B] = t_fk_s
        results["numba_serial"]["ik"][B] = t_ik_s
        results["numba_prange"]["fk"][B] = t_fk_p
        results["numba_prange"]["ik"][B] = t_ik_p

        print(f"    fk  B={B:<5}: serial {_fmt(t_fk_s)}/call   prange {_fmt(t_fk_p)}/call")
        print(f"    ik  B={B:<5}: serial {_fmt(t_ik_s)}/call   prange {_fmt(t_ik_p)}/call")

# ── summary table ──────────────────────────────────────────────────────────────

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
                        help="Save results to benchmark_results.json")
    args = parser.parse_args()

    import jaik
    print(f"\njaik benchmark — {args.robot}")
    print(f"Batch sizes: Jax: {BATCH_SIZES_JAX} Numba {BATCH_SIZES_NUMBA}   Repeats: {N_REPEAT}   Warmup: {args.n_warmup}\n")

    results = defaultdict(lambda: {"fk": {}, "ik": {}})

    print("── JAX ─────────────────────────────────────────────────────────────")
    jax_robot = jaik.make_robot(args.robot, solver="jax")
    bench_jax(jax_robot, results, n_warmup=args.n_warmup)

    print("\n── Numba ───────────────────────────────────────────────────────────")
    numba_robot = jaik.make_robot(args.robot, solver="numba")
    bench_numba(numba_robot, results, n_warmup=args.n_warmup)

    # _print_table(results)

    if args.save:
        # convert defaultdict for json serialization
        out = {k: {"fk": dict(v["fk"]), "ik": dict(v["ik"])} for k, v in results.items()}
        with open("benchmark_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print("  Saved benchmark_results.json")

if __name__ == "__main__":
    main()