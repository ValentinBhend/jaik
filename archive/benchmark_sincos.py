"""
jaik performance benchmark — FK and IK, numpy and JAX backends.

Measures:
  - Single call (Python-level, JIT-compiled JAX)
  - Looped single calls (inside jax.lax.scan to eliminate Python overhead)
  - Batched calls via vmap for batch sizes up to 2048

All timings reported in microseconds (µs).

Run:
    python benchmark.py           # CPU only
    python benchmark.py --fast    # CPU only with fast-math XLA flags
    python benchmark.py --gpu     # CPU + GPU
    python benchmark.py --fast --gpu

GPU note:
    Data is explicitly placed on GPU via jax.device_put so JAX actually
    runs on the GPU (computation follows data).
"""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

import jaik

# ── optional fast-math flags (must be set before JAX initialises) ─────────────
if "--fast" in sys.argv:
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_enable_fast_math=true "
        # "--xla_cpu_fast_math_honor_nans=false "
        # "--xla_cpu_fast_math_honor_infs=false "
        "--xla_cpu_fast_math_honor_division=false "
        "--xla_cpu_fast_math_honor_functions=false "
        "--xla_cpu_enable_fast_min_max=true"
    )

# jax.config.update("jax_enable_x64", True)

# ── configuration ─────────────────────────────────────────────────────────────

ROBOT       = "UR10e"
N_WARMUP    = 10
N_SINGLE    = 1000
N_LOOP      = 1000
N_VMAP_REPS = 20
BATCH_SIZES = [1, 8, 32, 128, 256, 512, 1024, 2048, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19]

# ── setup ─────────────────────────────────────────────────────────────────────

fk_jax, ik_jax, _ = jaik.make_robot(ROBOT, solver="cse_sincos")
fk_np,  ik_np, _  = jaik.make_robot(ROBOT, solver="numpy")

rng   = np.random.default_rng(0)
q_np  = rng.uniform(-np.pi, np.pi, 6)
q_jax = jnp.array(q_np)

R_np, p_np   = fk_np(q_np)
R_jax, p_jax = fk_jax(q_jax)

# compile once — reuse everywhere
fk_jit  = jax.jit(fk_jax)
ik_jit  = jax.jit(ik_jax)
fk_vmap = jax.jit(jax.vmap(fk_jax))
ik_vmap = jax.jit(jax.vmap(ik_jax))

# warmup JIT compilations
for _ in range(N_WARMUP):
    jax.block_until_ready(fk_jit(q_jax))
    jax.block_until_ready(ik_jit(R_jax, p_jax))

# ── helpers ───────────────────────────────────────────────────────────────────

def block(x):
    jax.block_until_ready(x)


def time_fn(fn, n):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.array(times) * 1e6  # return in µs


def print_result(label, times_us):
    m, s = times_us.mean(), times_us.std()
    print(f"  {label:<45s}  {m:9.2f} µs  ±{s:7.2f} µs")
    sys.stdout.flush()


def print_vmap_row(label, n, times_us):
    total = times_us.mean()
    print(f"  {label:<20s}  {n:>6}  {total:>10.1f}  {total/n:>12.3f}")
    sys.stdout.flush()


def build_Rs_ps(qs_batch):
    """Build (Rs, ps) for a batch using fk_vmap — no Python loop."""
    Rs, ps = fk_vmap(qs_batch)
    block((Rs, ps))
    return Rs, ps


# ── numpy ─────────────────────────────────────────────────────────────────────

def bench_numpy_single():
    print("\n── numpy single call ────────────────────────────────────────────")
    sys.stdout.flush()
    print_result("fk", time_fn(lambda: fk_np(q_np), N_SINGLE))
    print_result("ik", time_fn(lambda: ik_np(R_np, p_np), N_SINGLE))


def bench_numpy_loop():
    print("\n── numpy loop (N={}) ────────────────────────────────────────────".format(N_LOOP))
    sys.stdout.flush()

    t0 = time.perf_counter()
    for _ in range(N_LOOP):
        fk_np(q_np)
    print(f"  {'fk per call in loop':<45s}  {(time.perf_counter()-t0)/N_LOOP*1e6:9.2f} µs")
    sys.stdout.flush()

    t0 = time.perf_counter()
    for _ in range(N_LOOP):
        ik_np(R_np, p_np)
    print(f"  {'ik per call in loop':<45s}  {(time.perf_counter()-t0)/N_LOOP*1e6:9.2f} µs")
    sys.stdout.flush()


# ── JAX single + scan ─────────────────────────────────────────────────────────

def bench_jax_single():
    print("\n── JAX single call (JIT, CPU) ───────────────────────────────────")
    sys.stdout.flush()
    print_result("fk (jit)", time_fn(lambda: block(fk_jit(q_jax)), N_SINGLE))
    print_result("ik (jit)", time_fn(lambda: block(ik_jit(R_jax, p_jax)), N_SINGLE))


def bench_jax_scan():
    print("\n── JAX lax.scan loop (N={}, CPU) ────────────────────────────────".format(N_LOOP))
    sys.stdout.flush()

    qs = jnp.array(rng.uniform(-np.pi, np.pi, (N_LOOP, 6)))
    Rs, ps = build_Rs_ps(qs)

    @jax.jit
    def scan_fk(qs):
        def step(_, q):
            return None, fk_jax(q)
        _, out = jax.lax.scan(step, None, qs)
        return out

    @jax.jit
    def scan_ik(Rs, ps):
        def step(_, Rp):
            R, p = Rp
            Q, valid = ik_jax(R, p)
            return None, (Q, valid)
        _, out = jax.lax.scan(step, None, (Rs, ps))
        return out

    for _ in range(N_WARMUP):
        block(scan_fk(qs))
        block(scan_ik(Rs, ps))

    t0 = time.perf_counter()
    block(scan_fk(qs))
    print(f"  {'fk per call in scan':<45s}  {(time.perf_counter()-t0)/N_LOOP*1e6:9.2f} µs")
    sys.stdout.flush()

    t0 = time.perf_counter()
    block(scan_ik(Rs, ps))
    print(f"  {'ik per call in scan':<45s}  {(time.perf_counter()-t0)/N_LOOP*1e6:9.2f} µs")
    sys.stdout.flush()


# ── vmap (generic, works for CPU and GPU) ─────────────────────────────────────

def bench_vmap(label, fk_fn, ik_fn, device=None):
    """
    Benchmark vmap FK and IK for all batch sizes.
    Each batch size is compiled and timed separately — results print as they go.
    Data is placed on `device` if provided, so computation actually runs there.
    """
    suffix = f" ({label})" if label else ""
    print(f"\n── JAX vmap batch{suffix} ──────────────────────────────────────────")
    print(f"  {'operation':<20s}  {'batch':>6}  {'total µs':>10}  {'per call µs':>12}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*10}  {'-'*12}")
    sys.stdout.flush()

    def put(x):
        return jax.device_put(x, device) if device is not None else x

    # fk — compile and time each batch size as we go
    for n in BATCH_SIZES:
        qs = put(jnp.array(rng.uniform(-np.pi, np.pi, (n, 6))))
        for _ in range(N_WARMUP):
            block(fk_fn(qs))
        times = time_fn(lambda: block(fk_fn(qs)), N_VMAP_REPS)
        print_vmap_row(f"fk (vmap{suffix})", n, times)

    print()

    # ik — compile and time each batch size as we go
    for n in BATCH_SIZES:
        qs = jnp.array(rng.uniform(-np.pi, np.pi, (n, 6)))
        Rs, ps = build_Rs_ps(qs)   # build on CPU first
        Rs, ps = put(Rs), put(ps)  # then move to target device
        for _ in range(N_WARMUP):
            block(ik_fn(Rs, ps))
        times = time_fn(lambda: block(ik_fn(Rs, ps)), N_VMAP_REPS)
        print_vmap_row(f"ik (vmap{suffix})", n, times)


def bench_jax_vmap_cpu():
    bench_vmap("CPU", fk_vmap, ik_vmap, device=None)


def bench_gpu():
    print("\n── GPU ──────────────────────────────────────────────────────────")
    sys.stdout.flush()

    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        print("  Skipped — no CUDA-enabled JAX available.")
        print("  Install jax[cuda] and re-run to benchmark GPU.")
        sys.stdout.flush()
        return

    gpu = gpus[0]
    print(f"  GPU: {gpu}")
    print(f"  Verifying data placement...", end=" ")
    sys.stdout.flush()

    # verify GPU placement actually works
    test_q = jax.device_put(jnp.array(rng.uniform(-np.pi, np.pi, (4, 6))), gpu)
    test_R, test_p = fk_vmap(test_q)
    devices_used = test_R.devices()
    print(f"result on: {devices_used}")
    sys.stdout.flush()

    if not any("gpu" in str(d).lower() or "cuda" in str(d).lower()
               for d in devices_used):
        print("  WARNING: result is not on GPU — computation may have fallen back to CPU.")
        sys.stdout.flush()

    bench_vmap("GPU", fk_vmap, ik_vmap, device=gpu)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"jaik benchmark — {ROBOT}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    if "--fast" in sys.argv:
        print("fast-math enabled (--fast)")
    if "--gpu" in sys.argv:
        print("GPU benchmark enabled (--gpu)")
    sys.stdout.flush()

    bench_numpy_single()
    bench_numpy_loop()
    bench_jax_single()
    bench_jax_scan()
    bench_jax_vmap_cpu()

    if "--gpu" in sys.argv:
        bench_gpu()

    print("\nDone.")