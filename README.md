# Jaik
JAX analytic inverse kinematics: A fast solver for robots. Currently implemented are UR-robot arms, more will follow. 

## Installation

```
pip install jaik
```

It has `jax`, `numpy`, and `sympy` as dependencies. 

## Usage

UR robots can be imported by name, which uses default DH parameters. Custom DH and PoE parameters as well as URDF as input are planned in the future. 
```
import jax
import jaik

fk, ik_full, ik_closest = jaik.make_robot("ur10e")

q = jax.random.uniform(jax.random.PRNGKey(0), (6,))
R, p = fk(q)
Qs, valid = ik_full(R,p)
q0 = q * 1.1
q, branch = ik_closest(R, p, q0)
```

Three different solvers are offered, selectable in `make_robot`: "jax", "numpy", and "numba" (default: `solver="jax"`). Jax is the standard one here. Numpy is more or less a translation of the [IK-Geo](https://github.com/rpiRobotics/ik-geo/tree/main/matlab) matlab code, useful for debugging. 
Numba is an optional jit-compiled solver that uses numpy and not jax. It beats jax in performance for small batches. Install with `pip install "jaik[numba]"`

The solvers avoid trigonometric functions where possible. If the first thing you do with the returned joint angles is `jnp.sin(q), jnp.cos(q)`, we can save us both some time by using the keyword `sincos=True` in `make_robot`. This returns two values per joint as `sin(q), cos(q)`. For UR robots, this avoids using trigonometric functions altogether. 

By default it expects a (3,3) rotation matrix and (3,) vector as input (`format="Rp"` in `make_robot`). It can be changed to a single (4,4) matrix with `format="T"`. 

## Planned

- Add more robots available by name
- Support for custom (calibrated) DH & PoE parameters
- Support for URDF files as input

## Benchmarks

The benchmarks were done on a Lenovo ThinkPad (Intel Core Ultra 7 265U, 32 GB RAM, JAX on CPU) and a cluster (...)
Im unsure how much of the measured time is overhead or if anything got "compiled away", some guidance there would be appreciated. 

...plots...

## Some examples

...