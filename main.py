import jaik
import jax.numpy as jnp
import jax
import numpy as np


fk, ik_full, ik_closest = jaik.make_robot("ur10e", solver="numba")

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