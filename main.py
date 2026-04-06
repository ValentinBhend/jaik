import jaik
import jax.numpy as jnp
import jax
import numpy as np

from ur_analytic_ik import ur10e


fk, ik_full, ik_closest = jaik.make_robot("ur10e", solver="jax")

q = np.random.rand(6)
print("q", q.shape)

R, p = fk(q)
print("R", R.shape)
print("p", p.shape)

Q, valid = ik_full(R, p)
print("Q", Q.shape)
print("valid", valid.shape)

# q = jax.random.uniform(jax.random.PRNGKey(0), (6,))
# print("q", q.shape)

# R, p = fk(q)
# print("R", R.shape)
# print("p", p.shape)

# Q, valid = ik_full(R, p)
# print("Q", Q.shape)
# print("valid", valid.shape)


from numba import njit

def ik_closest_jax(R, p, sq0, cq0, weights):
    sq, cq, valid = ik_full(R, p)
    cos_diffs = cq * cq0[:, None] + sq * sq0[:, None] # cos(delta_theta)
    weighted_score = jnp.sum(cos_diffs * weights[:, None], axis=0)
    masked_score = jnp.where(valid, weighted_score, -jnp.inf) # masking
    best_idx = jnp.argmax(masked_score)
    best_sq, best_cq = sq[:, best_idx], cq[:, best_idx]
    return best_sq, best_cq, valid[best_idx]


@njit
def ik_closest_numba(R, p, sq0, cq0, weights):
    sq, cq, valid = ik_full(R, p)
    proximity_scores = np.full(8, -np.inf)
    for i in range(8):
        if valid[i]:
            score = 0.0 # sum of cos(theta - theta0) across 6 joints
            for j in range(6):
                score += (cq[j, i] * cq0[j] + sq[j, i] * sq0[j]) * weights[j]
            proximity_scores[i] = score
    best_idx = np.argmax(proximity_scores)
    is_valid = valid[best_idx]
    return sq[:, best_idx], cq[:, best_idx], is_valid