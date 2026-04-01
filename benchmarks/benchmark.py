"""
jaik performance benchmark — FK and IK, numpy and JAX backends.

Measures:
  - Single call (Python-level, JIT-compiled JAX)
  - Looped single calls (inside jax.lax.scan to eliminate Python overhead)
  - Batched calls via vmap for batch sizes up to 2048

Run:
    python benchmark.py

GPU:
    Skipped automatically if no CUDA-enabled JAX is available.
    To run on GPU: install jax[cuda] and re-run.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

import jaik

# ── configuration ─────────────────────────────────────────────────────────────

ROBOT        = "UR10e"
N_WARMUP     = 5        # JAX JIT warmup calls
N_SINGLE     = 1000     # single-call timing repetitions
N_LOOP       = 1000     # iterations inside lax.scan loop
BATCH_SIZES  = [1, 8, 32, 128, 256, 512, 1024, 2048]

# ── setup ─────────────────────────────────────────────────────────────────────

fk_jax, ik_jax = jaik.make_robot(ROBOT, backend="jax")
fk_np,  ik_np  = jaik.make_robot(ROBOT, backend="numpy")

rng   = np.random.default_rng(0)
q_np  = rng.uniform(-np.pi, np.pi, 6)
q_jax = jnp.array(q_np)

R_np, p_np   = fk_np(q_np)
R_jax, p_jax = fk_jax(q_jax)

# ── helpers ───────────────────────────────────────────────────────────────────

def mean_std_ms(times_s):
    t = np.array(times_s) * 1000
    return t.mean(), t.std()


def print_result(label, mean_ms, std_ms):
    print(f"  {label:<45s}  {mean_ms:8.3f} ms  ±{std_ms:6.3f} ms")


def time_fn(fn, n):
    """Time a zero-argument callable n times, return list of elapsed seconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def block(x):
    """Force JAX to finish computation (block_until_ready)."""
    jax.block_until_ready(x)


# ── numpy single call ─────────────────────────────────────────────────────────

def bench_numpy_single():
    print("\n── numpy single call ────────────────────────────────────────────")

    times_fk = time_fn(lambda: fk_np(q_np), N_SINGLE)
    m, s = mean_std_ms(times_fk)
    print_result("fk", m, s)

    times_ik = time_fn(lambda: ik_np(R_np, p_np), N_SINGLE)
    m, s = mean_std_ms(times_ik)
    print_result("ik", m, s)


# ── numpy loop ────────────────────────────────────────────────────────────────

def bench_numpy_loop():
    print("\n── numpy loop (N={}) ────────────────────────────────────────────".format(N_LOOP))

    def fk_loop():
        for _ in range(N_LOOP):
            fk_np(q_np)

    def ik_loop():
        for _ in range(N_LOOP):
            ik_np(R_np, p_np)

    t0 = time.perf_counter()
    fk_loop()
    t1 = time.perf_counter()
    per_call = (t1 - t0) / N_LOOP * 1000
    print(f"  {'fk per call in loop':<45s}  {per_call:8.3f} ms")

    t0 = time.perf_counter()
    ik_loop()
    t1 = time.perf_counter()
    per_call = (t1 - t0) / N_LOOP * 1000
    print(f"  {'ik per call in loop':<45s}  {per_call:8.3f} ms")


# ── JAX single call (JIT) ─────────────────────────────────────────────────────

def bench_jax_single():
    print("\n── JAX single call (JIT, CPU) ───────────────────────────────────")

    fk_jit = jax.jit(fk_jax)
    ik_jit = jax.jit(ik_jax)

    # warmup
    for _ in range(N_WARMUP):
        block(fk_jit(q_jax))
        block(ik_jit(R_jax, p_jax))

    times_fk = time_fn(lambda: block(fk_jit(q_jax)), N_SINGLE)
    m, s = mean_std_ms(times_fk)
    print_result("fk (jit)", m, s)

    times_ik = time_fn(lambda: block(ik_jit(R_jax, p_jax)), N_SINGLE)
    m, s = mean_std_ms(times_ik)
    print_result("ik (jit)", m, s)


# ── JAX lax.scan loop (eliminates Python overhead) ───────────────────────────

def bench_jax_scan():
    print("\n── JAX lax.scan loop (N={}, CPU) ────────────────────────────────".format(N_LOOP))

    qs = jnp.array(rng.uniform(-np.pi, np.pi, (N_LOOP, 6)))

    # build Rs and ps by unpacking fk output properly
    Rs_list, ps_list = [], []
    for q in np.array(qs):
        R, p = fk_jax(jnp.array(q))
        Rs_list.append(np.asarray(R))
        ps_list.append(np.asarray(p))
    Rs = jnp.array(np.stack(Rs_list))  # (N_LOOP, 3, 3)
    ps = jnp.array(np.stack(ps_list))  # (N_LOOP, 3)

    @jax.jit
    def scan_fk(qs):
        def step(_, q):
            R, p = fk_jax(q)
            return None, (R, p)
        _, out = jax.lax.scan(step, None, qs)
        return out

    @jax.jit
    def scan_ik(Rs, ps):
        def step(_, Rp):
            R, p = Rp
            Q, is_LS = ik_jax(R, p)
            return None, (Q, is_LS)
        _, out = jax.lax.scan(step, None, (Rs, ps))
        return out

    # warmup
    for _ in range(N_WARMUP):
        block(scan_fk(qs))
        block(scan_ik(Rs, ps))

    t0 = time.perf_counter()
    block(scan_fk(qs))
    t1 = time.perf_counter()
    per_call = (t1 - t0) / N_LOOP * 1000
    print(f"  {'fk per call in scan':<45s}  {per_call:8.3f} ms")

    t0 = time.perf_counter()
    block(scan_ik(Rs, ps))
    t1 = time.perf_counter()
    per_call = (t1 - t0) / N_LOOP * 1000
    print(f"  {'ik per call in scan':<45s}  {per_call:8.3f} ms")


# ── JAX vmap batch ────────────────────────────────────────────────────────────

def bench_jax_vmap():
    print("\n── JAX vmap batch (CPU) ─────────────────────────────────────────")
    print(f"  {'operation':<45s}  {'batch':>6}  {'total ms':>10}  {'per call ms':>12}")
    print(f"  {'-'*45}  {'-'*6}  {'-'*10}  {'-'*12}")

    fk_batch = jax.jit(jax.vmap(fk_jax))

    for n in BATCH_SIZES:
        qs_batch = jnp.array(rng.uniform(-np.pi, np.pi, (n, 6)))

        for _ in range(N_WARMUP):
            block(fk_batch(qs_batch))
            
        times = time_fn(lambda: block(fk_batch(qs_batch)), 20)
        total_ms = np.mean(times) * 1000
        per_ms   = total_ms / n
        print(f"  {'fk (vmap)':<45s}  {n:>6}  {total_ms:>10.3f}  {per_ms:>12.4f}")

    
    ik_batch = jax.jit(jax.vmap(ik_jax))

    for n in BATCH_SIZES:
        Rs_list, ps_list = [], []
        for q in np.array(qs_batch):
            R, p = fk_jax(jnp.array(q))
            Rs_list.append(np.asarray(R))
            ps_list.append(np.asarray(p))
        Rs_batch = jnp.array(np.stack(Rs_list))  # (n, 3, 3)
        ps_batch = jnp.array(np.stack(ps_list))  # (n, 3)

        for _ in range(N_WARMUP):
            block(ik_batch(Rs_batch, ps_batch))

        times = time_fn(lambda: block(ik_batch(Rs_batch, ps_batch)), 20)
        total_ms = np.mean(times) * 1000
        per_ms   = total_ms / n
        print(f"  {'ik (vmap)':<45s}  {n:>6}  {total_ms:>10.3f}  {per_ms:>12.4f}")


# ── GPU (skipped if unavailable) ──────────────────────────────────────────────

def bench_gpu():
    print("\n── GPU ──────────────────────────────────────────────────────────")
    try:
        gpu = jax.devices("gpu")
    except RuntimeError:
        print("  Skipped — no CUDA-enabled JAX available.")
        print("  Install jax[cuda] and re-run to benchmark GPU.")
        return

    print(f"  GPU device: {gpu[0]}")
    # same structure as vmap but on GPU — just move data there
    fk_batch = jax.jit(jax.vmap(fk_jax), backend="gpu")
    ik_batch = jax.jit(jax.vmap(ik_jax), backend="gpu")

    for n in BATCH_SIZES:
        qs_batch = jax.device_put(
            jnp.array(rng.uniform(-np.pi, np.pi, (n, 6))), gpu[0]
        )
        Rs_batch = jax.device_put(
            jnp.array(np.stack([np.asarray(fk_jax(jnp.array(q)))[0]
                                 for q in np.array(qs_batch)])), gpu[0]
        )
        ps_batch = jax.device_put(
            jnp.array(np.stack([np.asarray(fk_jax(jnp.array(q)))[1]
                                 for q in np.array(qs_batch)])), gpu[0]
        )

        for _ in range(N_WARMUP):
            block(fk_batch(qs_batch))
            block(ik_batch(Rs_batch, ps_batch))

        times = time_fn(lambda: block(fk_batch(qs_batch)), 20)
        total_ms = np.mean(times) * 1000
        print(f"  {'fk (vmap, GPU)':<45s}  {n:>6}  {total_ms:>10.3f}  {total_ms/n:>12.4f}")

        times = time_fn(lambda: block(ik_batch(Rs_batch, ps_batch)), 20)
        total_ms = np.mean(times) * 1000
        print(f"  {'ik (vmap, GPU)':<45s}  {n:>6}  {total_ms:>10.3f}  {total_ms/n:>12.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"jaik benchmark — {ROBOT}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")

    bench_numpy_single()
    bench_numpy_loop()
    bench_jax_single()
    bench_jax_scan()
    bench_jax_vmap()
    bench_gpu()

    print("\nDone.")