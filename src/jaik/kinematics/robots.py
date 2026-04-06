# src/jaik/kinematics/robots.py
"""
Robot definitions and the make_robot factory.

Usage:
    fk, ik = jaik.make_robot("UR10e")                    # fastest available
    fk, ik = jaik.make_robot("UR10e", solver="cse")      # explicit CSE (fast, UR-specific)
    fk, ik = jaik.make_robot("UR10e", solver="general")  # IK-Geo general solver
    fk, ik = jaik.make_robot("UR10e", solver="numpy")    # numpy backend, for debugging

    fk, ik = jaik.make_robot_from_dh(alpha, a, d)        # custom DH parameters

    jaik.available_robots()                               # list of named presets
    jaik.robot_infos("UR10e")                            # dict of robot metadata

URDF parsing is intentionally not included — use a dependency like
pyroki or yourdfpy to extract DH parameters, then pass them to
make_robot_from_dh.
"""
from typing import Callable
import importlib
import numpy as np
import jax.numpy as jnp
import os

from jaik.kinematics.convention_conversions import dh_to_kin
from jaik.kinematics.adjustments import adjust_kin_for_3p2i

from .UR_params import _UR_ALPHA, _UR_PARAMS



# ── public API ────────────────────────────────────────────────────────────────

def available_robots() -> list:
    """Return sorted list of available named robot presets."""
    return sorted(k.lower() for k in _UR_PARAMS)


def robot_infos(name: str) -> dict:
    """
    Return a dict of metadata for a named robot.

    Keys:
        available_solvers: list of solver names accepted by make_robot(name, solver=...)

    Example:
        jaik.robot_infos("UR10e")
        # {"available_solvers": ["auto", "cse", "general", "numpy"]}

        jaik.robot_infos("UR3e")
        # {"available_solvers": ["auto", "general", "numpy"]}  # if no CSE generated yet
    """
    if name not in _UR_PARAMS:
        raise ValueError(
            f"Unknown robot '{name}'. "
            f"Available: {sorted(_UR_PARAMS.keys())}."
        )
    return {
        "available_solvers": "TODO..",
    }


def make_robot(
    robot:   str | dict | os.PathLike,
    solver: str = "jax",
    format: str = "Rp",
    sincos: bool = False,
) -> tuple[Callable, Callable, Callable]:
    """
    Return (fk, ik_full, ik_closest) callables for a named UR robot.

    Args:
        name:   Robot name, e.g. "UR10e". See available_robots().
        ...

    Returns:
        fk: q -> (R, p)            forward kinematics
        ik_full: (R, p) -> (Q, valid)   IK, all 8 branches
                                   Q:     (6, 8) joint angles, NaN = infeasible
                                   valid: (8,)   bool mask of feasible branches
        ik_closest: (R, p, q0) -> (q, branch)   IK, selects cloeset branch
                                   q:     (6,) joint angles, NaN = infeasible
                                   branch: integer branch index
    """

    kin, codegen_id = _resolve_robot(robot)
    _fk, _ik_full = _load_solvers(solver, kin, codegen_id)
    fk, ik_full, ik_closest = _wrap_solvers(_fk, _ik_full, solver, format, sincos)
    return fk, ik_full, ik_closest

# ── internal ──────────────────────────────────────────────────────────────────
def _resolve_robot(robot: str | dict | os.PathLike):
    if isinstance(robot, str):
        if robot.endswith(".urdf") or os.path.exists(robot):
            kin, codegen_id = _resolve_urdf(robot)
        else:
            kin, codegen_id = _resolve_named(robot)
    elif isinstance(robot, dict):
        kin, codegen_id = _resolve_custom(robot)
    elif isinstance(robot, os.PathLike):
        kin, codegen_id = _resolve_urdf(robot)
    else:
        raise ValueError (f"Robot input {robot} with type {type(robot)} could not be resolved")
    return kin, codegen_id

def _resolve_custom(definition: dict):
    raise NotImplementedError

def _resolve_urdf(path: str):
    raise NotImplementedError

def _resolve_named(name: str):
    if name.lower() in available_robots():
        if name.lower() in (k.lower() for k in _UR_PARAMS):
            return _resolve_UR(name)
        else:
            raise NotImplementedError
    else:
        raise ValueError (f"Robot name {name} not known. Currently available robots by name are {available_robots()}")

def _resolve_UR(name):
    p = next((v for k, v in _UR_PARAMS.items() if k.lower() == name.lower()), None)

    kin = dh_to_kin(p["alpha"], p["a"], p["d"])
    kin = adjust_kin_for_3p2i(kin)
    kin["R_6T"] = p["R_6T"]

    codegen_id = name.lower()
    return kin, codegen_id

def _load_solvers(solver, kin, codegen_id):
    module_name = f"jaik._{solver}._generated.{codegen_id}"
    if importlib.util.find_spec(module_name) is None:
        print(f"Generating new sympy cse for {solver}. Might take a few minutes (only once, then saved)")
        _generate_new(solver, kin, module_name, codegen_id)
    _fk, _ik_full = _load_generated(module_name)
    return _fk, _ik_full

def _load_generated(module_name):
    module = importlib.import_module(module_name)
    _fk = getattr(module, "fk")
    _ik_full = getattr(module, "ik_full")
    return _fk, _ik_full

def _generate_new(solver, kin, module_name, codegen_id):
    from jaik.codegen.generate import generate
    if codegen_id in available_robots():
        generate(solver, kin, module_name, codegen_id)
    else:
        raise NotImplementedError

def _wrap_solvers(_fk, _ik_full, solver, format, sincos):
    if solver == "jax":
        fk, ik_full, ik_closest = _wrap_solver_jax(_fk, _ik_full, format, sincos)
    elif solver == "numba":
        fk, ik_full, ik_closest = _wrap_solver_numba(_fk, _ik_full, format, sincos)
    else:
        raise ValueError
    
    return fk, ik_full, ik_closest

def _wrap_solver_jax(_fk, _ik_full, format, sincos):
    def fk(sq, cq):
        R, p = _fk(sq, cq)
        return R, p
    def ik_full(R, p):
        sq, cq, valid = _ik_full(R, p)
        return sq, cq, valid
    
    if not sincos:
        fk_orig = fk
        ik_full_orig = ik_full
        def fk(q):
            sq, cq = jnp.sin(q), jnp.cos(q)
            return fk_orig(sq, cq)
        def ik_full(R, p):
            sq, cq, valid = ik_full_orig(R, p)
            q = jnp.atan2(sq, cq)
            return q, valid
    
    if format == "Rp":
        pass
    elif format == "T":
        fk_orig = fk
        ik_full_orig = ik_full
        def fk(*args):
            R, p = fk_orig(*args)
            T = jnp.concatenate([
                jnp.concatenate([R, p[:,None]], axis=1),
                jnp.array([[0., 0., 0., 1.]])
            ], axis=0)
            return T
        def ik_full(T):
            R = T[:3,:3]
            p = T[:3,3]
            return ik_full_orig(R, p)
    else:
        raise ValueError

    def ik_closest():
        raise NotImplementedError
    
    return fk, ik_full, ik_closest


def _wrap_solver_numba(_fk, _ik_full, format, sincos):
    try:
        from numba import njit
    except ImportError:
        raise ImportError(
            "numba is required for solver='numba'. "
            "Install it with: pip install 'jaik[numba]'"
        ) from None

    @njit
    def _fk_Rp_sincos(sq, cq):
        R, p = _fk(sq, cq)
        return R, p

    @njit
    def _fk_Rp(q):
        sq, cq = np.sin(q), np.cos(q)
        return _fk_Rp_sincos(sq, cq)

    @njit
    def _fk_T_sincos(sq, cq):
        R, p = _fk_Rp_sincos(sq, cq)
        T = np.empty((4, 4))
        T[:3, :3] = R
        T[:3, 3]  = p
        T[3, :3]  = 0.
        T[3, 3]   = 1.
        return T

    @njit
    def _fk_T(q):
        sq, cq = np.sin(q), np.cos(q)
        return _fk_T_sincos(sq, cq)

    @njit
    def _ik_full_sincos(R, p):
        sq, cq, valid = _ik_full(R, p)
        return sq, cq, valid

    @njit
    def _ik_full_q(R, p):
        sq, cq, valid = _ik_full_sincos(R, p)
        return np.arctan2(sq, cq), valid

    @njit
    def _ik_full_T_sincos(T):
        return _ik_full_sincos(T[:3, :3], T[:3, 3])

    @njit
    def _ik_full_T_q(T):
        return _ik_full_q(T[:3, :3], T[:3, 3])

    def ik_closest():
        raise NotImplementedError

    if format == "Rp" and sincos:
        return _fk_Rp_sincos, _ik_full_sincos, ik_closest
    elif format == "Rp" and not sincos:
        return _fk_Rp, _ik_full_q, ik_closest
    elif format == "T" and sincos:
        return _fk_T_sincos, _ik_full_T_sincos, ik_closest
    elif format == "T" and not sincos:
        return _fk_T, _ik_full_T_q, ik_closest
    else:
        raise ValueError(f"Unknown format='{format}'")

# def _wrap_solver_numba(_fk, _ik_full, format, sincos, R_6T):
#     try:
#         from numba import njit
#     except ImportError:
#         raise ImportError(
#             "numba is required for solver='numba'. "
#             "Install it with: pip install 'jaik[numba]'"
#         ) from None
    
#     RT = np.asarray(R_6T)

#     @njit
#     def _fk_Rp_sincos(sq, cq):
#         R, p = _fk(sq, cq)
#         return R @ RT, p

#     @njit
#     def _fk_Rp(q):
#         sq, cq = np.sin(q), np.cos(q)
#         return _fk_Rp_sincos(sq, cq)

#     @njit
#     def _fk_T_sincos(sq, cq):
#         R, p = _fk_Rp_sincos(sq, cq)
#         T = np.empty((4, 4))
#         T[:3, :3] = R
#         T[:3, 3]  = p
#         T[3, :3]  = 0.
#         T[3, 3]   = 1.
#         return T

#     @njit
#     def _fk_T(q):
#         sq, cq = np.sin(q), np.cos(q)
#         return _fk_T_sincos(sq, cq)

#     @njit
#     def _ik_full_sincos(R, p, sq, cq, valid):
#         R_06 = R @ RT.T
#         _ik_full(R_06, p, sq, cq, valid)

#     # @njit
#     # def _ik_full_q(R, p):
#     #     sq, cq, valid = _ik_full_sincos(R, p)
#     #     return np.arctan2(sq, cq), valid

#     # @njit
#     # def _ik_full_T_sincos(T):
#     #     return _ik_full_sincos(T[:3, :3], T[:3, 3])

#     # @njit
#     # def _ik_full_T_q(T):
#     #     return _ik_full_q(T[:3, :3], T[:3, 3])

#     def ik_closest():
#         raise NotImplementedError

#     if format == "Rp" and sincos:
#         return _fk_Rp_sincos, _ik_full_sincos, ik_closest
#     # elif format == "Rp" and not sincos:
#     #     return _fk_Rp, _ik_full_q, ik_closest
#     # elif format == "T" and sincos:
#     #     return _fk_T_sincos, _ik_full_T_sincos, ik_closest
#     # elif format == "T" and not sincos:
#     #     return _fk_T, _ik_full_T_q, ik_closest
#     else:
#         raise ValueError(f"Unknown format='{format}'")