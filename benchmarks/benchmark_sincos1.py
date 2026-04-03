import time
import numpy as np
import jax
import jax.numpy as jnp
import jaik

# --- Settings ---
N_TRIALS = 10
N_INNER_LOOPS = 1000
VMAP_BATCH = 1024
ROBOT = "UR10e"

# 1. Setup Robot
fk_jax, ik_jax, _ = jaik.make_robot(ROBOT, solver="cse_sincos")
fk_jit = jax.jit(fk_jax)

# Prepare Data
rng = np.random.default_rng(42)
q_ref = jnp.array(rng.uniform(-jnp.pi, jnp.pi, 6))
R_target, p_target = fk_jit(q_ref)

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

def run_bench():
    # Fix the repeat logic for Tuples
    test_Rs = jnp.repeat(R_target[None], N_INNER_LOOPS, axis=0)
    test_ps = jnp.repeat(p_target[None], N_INNER_LOOPS, axis=0)
    
    # Warmup
    _ = jax.block_until_ready(scan_ik_kernel(test_Rs, test_ps))

    print(f"--- {ROBOT} Hardware Floor Benchmark {N_TRIALS} trials {N_INNER_LOOPS} inner loops {VMAP_BATCH} vmap batch size ---")

    # Latency
    latencies = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        jax.block_until_ready(scan_ik_kernel(test_Rs, test_ps))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) / N_INNER_LOOPS * 1e6)
    
    print(f"Min Latency (per call): {min(latencies):.3f} µs")

    # Throughput
    v_Rs = jnp.repeat(jnp.repeat(R_target[None, None], N_INNER_LOOPS, 0), VMAP_BATCH, 1)
    v_ps = jnp.repeat(jnp.repeat(p_target[None, None], N_INNER_LOOPS, 0), VMAP_BATCH, 1)
    
    throughputs = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        jax.block_until_ready(throughput_kernel(v_Rs, v_ps))
        t1 = time.perf_counter()
        throughputs.append(t1 - t0)
    
    best_time = min(throughputs)
    total_solves = N_INNER_LOOPS * VMAP_BATCH
    
    print(f"Throughput Floor:       {best_time / total_solves * 1e9:.2f} ns/solve")
    print(f"Mega-Solves/sec:        {total_solves / best_time / 1e6:.2f} M/s")

if __name__ == "__main__":
    run_bench()