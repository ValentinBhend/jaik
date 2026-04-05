import numpy as np
from scipy.optimize import linear_sum_assignment
import jax.numpy as jnp


def _angle_diff(a, b):
    """Wrapped difference a - b ∈ (-π, π], any broadcastable shape."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def match_branches_single(Q, valid, ref_solutions):
    Q = np.asarray(Q)
    valid = np.asarray(valid, dtype=bool)
    valid_idx = np.where(valid)[0]
    Q_valid = Q[:, valid_idx]
    n_jaik = Q_valid.shape[1]

    # ref returning 0 is fine (ur_ikfast filtered everything for this pose)
    # jaik returning 0 when ref has solutions is the failure case
    if len(ref_solutions) == 0:
        return dict(mapping=[], errors=np.empty((0, 6)),
                    n_jaik=n_jaik, n_ref=0, count_match=True)
    if n_jaik == 0:
        return None   # jaik failed — this is the failure

    R = np.stack(ref_solutions)
    n_ref = R.shape[0]

    diff = _angle_diff(
        R[:, :, np.newaxis],
        Q_valid[np.newaxis, :, :]
    )
    cost = np.mean(np.abs(diff), axis=1)

    row_ind, col_ind = linear_sum_assignment(cost)

    errors = np.abs(_angle_diff(R[row_ind], Q_valid[:, col_ind].T))

    return dict(
        mapping=list(zip(row_ind.tolist(), valid_idx[col_ind].tolist())),
        errors=errors,
        n_jaik=n_jaik,
        n_ref=n_ref,
        # mismatch only when ref found solutions jaik didn't cover
        count_match=(n_ref <= n_jaik),
    )


def compare_solvers(fk_ref, ik_ref, fk_jaik, ik_jaik,
                    n_poses=500, seed=42, verbose=True, 
                    noise_threshold=1e-8, solver="jax"):
    """
    Generate random poses via jaik FK, solve with both solvers, find the
    optimal branch mapping per pose, and report per-joint error statistics.

    fk_jaik        : callable(q (6,))        -> (R (3,3), p (3,))
    ik_ref         : callable(T (4,4))        -> list of (6,) arrays
    ik_jaik        : callable(R (3,3), p (3)) -> (Q (6,8), valid (8,))
    noise_threshold: max allowed error (rad) for the bool 'passed' flag
    """
    to_arr = jnp.array if solver == "jax" else np.array

    rng = np.random.default_rng(seed)
    all_errors = []
    count_mismatches = 0
    no_solution_poses = 0

    for _ in range(n_poses):
        q_np = rng.uniform(-np.pi, np.pi, 6)

        q = to_arr(q_np)
        R, p = fk_jaik(q)
        R_np, p_np = np.asarray(R), np.asarray(p)
        T_np = np.eye(4)
        T_np[:3, :3] = R_np
        T_np[:3, 3]  = p_np

        T_ref = fk_ref(*q_np)
        R_ref = to_arr(T_ref[:3, :3])
        p_ref = to_arr(T_ref[:3, 3])
        T_ref_np = np.asarray(T_ref)

        for R_in, p_in, T_in in [(R, p, T_np), (R_ref, p_ref, T_ref_np)]:
            ref_sols       = ik_ref(T_in)
            Q, valid       = ik_jaik(R_in, p_in)
            Q_np, valid_np = np.asarray(Q), np.asarray(valid)
            result = match_branches_single(Q_np, valid_np, ref_sols)
            if result is None:
                no_solution_poses += 1
                continue
            if not result["count_match"]:
                count_mismatches += 1
            all_errors.append(result["errors"])

    if not all_errors:
        print("No matched solutions found across all poses.")
        return False, dict(
            errors=np.empty((0, 6)),
            per_joint_min=np.zeros(6),
            per_joint_mean=np.zeros(6),
            per_joint_max=np.zeros(6),
            count_mismatches=count_mismatches,
            no_solution_poses=no_solution_poses,
        )

    E = np.concatenate(all_errors, axis=0)  # (total_matched_pairs, 6)

    passed = (
        no_solution_poses == 0          # jaik failed where ref succeeded
        and count_mismatches == 0       # ref found more solutions than jaik
        and float(E.max()) < noise_threshold
    )

    if verbose:
        w = 10
        print(f"\n{'─'*56}")
        print(f"  Solver comparison over {n_poses} random poses")
        print(f"  Total matched branch pairs : {len(E)}")
        print(f"  Poses with count mismatch  : {count_mismatches}")
        print(f"  Poses with no solutions    : {no_solution_poses}")
        print(f"{'─'*56}")
        print(f"  {'joint':>5}  {'min':>{w}}  {'mean':>{w}}  {'max':>{w}}")
        print(f"  {'':>5}  {'(rad)':>{w}}  {'(rad)':>{w}}  {'(rad)':>{w}}")
        print(f"{'─'*56}")
        for j in range(6):
            col = E[:, j]
            print(f"  {'q'+str(j+1):>5}  {col.min():>{w}.3e}  "
                  f"{col.mean():>{w}.3e}  {col.max():>{w}.3e}")
        print(f"{'─'*56}")
        status = "✓ PASSED — numerical noise only" if passed else "✗ FAILED"
        print(f"  Result: {status}  (threshold {noise_threshold:.0e} rad)")
        print(f"{'─'*56}\n")

    return passed, dict(
        errors=E,
        per_joint_min=E.min(axis=0),
        per_joint_mean=E.mean(axis=0),
        per_joint_max=E.max(axis=0),
        count_mismatches=count_mismatches,
        no_solution_poses=no_solution_poses,
    )