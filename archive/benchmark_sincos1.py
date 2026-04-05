import time
import numpy as np
import jax
import jax.numpy as jnp
import jaik
import sys
import os

# --- Settings ---
N_TRIALS = 10
N_INNER_LOOPS = 1000
VMAP_BATCH = 1024
ROBOT = "UR10e"

if "--fast" in sys.argv:
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_enable_fast_math=true "
        # "--xla_cpu_fast_math_honor_nans=false "
        # "--xla_cpu_fast_math_honor_infs=false "
        "--xla_cpu_fast_math_honor_division=false "
        "--xla_cpu_fast_math_honor_functions=false "
        "--xla_cpu_enable_fast_min_max=true"
    )

fk_jax, ik_jax, _ = jaik.make_robot(ROBOT, solver="cse_sincos")

rng = np.random.default_rng(42)
q_ref_np = rng.uniform(-np.pi, np.pi, 6)


def run_bench_on_device(device):
    label = f"JAX {device.device_kind.upper()}"
    print(f"\n--- {ROBOT} Hardware Floor Benchmark | {label} | "
          f"{N_TRIALS} trials | {N_INNER_LOOPS} inner loops | {VMAP_BATCH} vmap batch ---")

    # Pin all data to this device up front
    to_dev = lambda x: jax.device_put(x, device)

    q_ref = to_dev(jnp.array(q_ref_np))

    fk_jit = jax.jit(fk_jax, device=device)
    R_target, p_target = jax.block_until_ready(fk_jit(q_ref))

    @jax.jit
    def scan_ik_kernel(Rs, ps):
        def step(_, inputs):
            return None, ik_jax(inputs[0], inputs[1])
        return jax.lax.scan(step, None, (Rs, ps))[1]

    @jax.jit
    def throughput_kernel(Rs_batch, ps_batch):
        def step(_, inputs):
            return None, jax.vmap(ik_jax)(inputs[0], inputs[1])
        return jax.lax.scan(step, None, (Rs_batch, ps_batch))[1]

    # Pre-build all data on device
    test_Rs = to_dev(jnp.repeat(R_target[None], N_INNER_LOOPS, axis=0))
    test_ps = to_dev(jnp.repeat(p_target[None], N_INNER_LOOPS, axis=0))

    v_Rs = to_dev(jnp.repeat(jnp.repeat(R_target[None, None], N_INNER_LOOPS, 0), VMAP_BATCH, 1))
    v_ps = to_dev(jnp.repeat(jnp.repeat(p_target[None, None], N_INNER_LOOPS, 0), VMAP_BATCH, 1))

    # Warmup — compile + run once on this device
    jax.block_until_ready(scan_ik_kernel(test_Rs, test_ps))
    jax.block_until_ready(throughput_kernel(v_Rs, v_ps))

    # Latency (sequential scan)
    latencies = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        jax.block_until_ready(scan_ik_kernel(test_Rs, test_ps))
        latencies.append((time.perf_counter() - t0) / N_INNER_LOOPS * 1e6)
    print(f"Min Latency (per call): {min(latencies):.3f} µs")

    # Throughput (vmap batch)
    throughputs = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        jax.block_until_ready(throughput_kernel(v_Rs, v_ps))
        throughputs.append(time.perf_counter() - t0)

    best_time = min(throughputs)
    total_solves = N_INNER_LOOPS * VMAP_BATCH
    print(f"Throughput Floor:       {best_time / total_solves * 1e9:.2f} ns/solve")
    print(f"Mega-Solves/sec:        {total_solves / best_time / 1e6:.2f} M/s")


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    
    if "--fast" in sys.argv:
        print("fast-math enabled (--fast)")

    devices = []
    for backend in ('cpu', 'gpu'):
        try:
            devices.extend(jax.devices(backend))
        except RuntimeError:
            pass  # backend not available

    print(f"Devices to benchmark: {devices}")

    for device in devices:
        try:
            run_bench_on_device(device)
        except Exception as e:
            print(f"  Skipped {device}: {e}")