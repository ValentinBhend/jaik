import time
import numpy as np
from numba import njit
from jaik import make_robot

# 1. Initialize the robot with the Numba solver
# This will trigger the codegen and initial JIT compilation
fk, ik, _ = make_robot("UR10e", solver="numba")

# 2. Setup dummy data (Target TCP Pose)
# Identity rotation and a reachable position
R_target = np.eye(3)
p_target = np.array([0.5, 0.0, 0.5])

# 3. WARMUP: Essential to trigger LLVM compilation before measuring
print("Warming up LLVM kernels...")
_ = ik(R_target, p_target)

# 4. Define the NJIT-accelerated loop
# This measures "Native-to-Native" speed (zero Python overhead)
@njit
def benchmark_njit_loop(R, p, iterations):
    # We store a dummy result to prevent the compiler from 
    # optimizing the entire loop away (Dead Code Elimination)
    checksum = 0.0
    for _ in range(iterations):
        Q, valid = ik(R, p)
        checksum += Q[0, 0] 
    return checksum


def run_benchmark(iterations=100_000, repetitions=100):
    print(f"\nRunning {iterations:,} iterations across {repetitions} repetitions...")
    
    # WARMUP
    for _ in range(10):
        Q, valid = ik(R_target, p_target)
    _ = benchmark_njit_loop(R_target, p_target, 10)
    
    py_loop_times = []
    njit_loop_times = []
    
    # --- TEST 1: Python For-Loop ---
    # Measures: Python Dispatch + Numba Math
    for _ in range(repetitions):
        start = time.perf_counter()
        for _ in range(iterations):
            Q, valid = ik(R_target, p_target)
        end = time.perf_counter()
        py_loop_times.append((end - start) / iterations)
        
    # --- TEST 2: NJIT For-Loop ---
    # Measures: Pure Machine Code Math
    for _ in range(repetitions):
        start = time.perf_counter()
        cs = benchmark_njit_loop(R_target, p_target, iterations)
        end = time.perf_counter()
        njit_loop_times.append((end - start) / iterations)

    # --- Statistics Calculation ---
    py_loop_times = np.array(py_loop_times) * 1e6    # Convert to microseconds
    njit_loop_times = np.array(njit_loop_times) * 1e6
    
    py_mean = np.mean(py_loop_times)
    py_std = np.std(py_loop_times)
    
    njit_mean = np.mean(njit_loop_times)
    njit_std = np.std(njit_loop_times)
    
    # Overhead = Python Time - NJIT Time
    disp_mean = py_mean - njit_mean
    # Error propagation for the difference of two means
    disp_std = np.sqrt(py_std**2 + njit_std**2)

    # --- Results ---
    print("-" * 45)
    print(f"Python -> Numba: {py_mean:.3f} ± {py_std:.3f} µs / solve")
    print(f"Inside NJIT:     {njit_mean:.3f} ± {njit_std:.3f} µs / solve")
    print("-" * 45)
    print(f"Dispatch Overhead: {disp_mean:.3f} ± {disp_std:.3f} µs")


if __name__ == "__main__":
    run_benchmark()