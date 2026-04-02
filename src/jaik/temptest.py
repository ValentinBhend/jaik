import jax
import jaik

# should now say ["auto", "cse", "general", "numpy"]
print(jaik.robot_infos("UR10e"))

fk, ik = jaik.make_robot("UR10e", solver="cse")
import jax.numpy as jnp
q = jnp.zeros(6)
R, p = fk(q)
Q, valid = ik(R, p)
print(Q.shape, valid)  # (6, 8), (8,)