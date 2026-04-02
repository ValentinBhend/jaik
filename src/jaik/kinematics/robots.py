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
from jaik.kinematics.convention_conversions import dh_to_kin
from jaik.kinematics.adjustments import adjust_kin_for_3p2i

# ── robot parameter presets ───────────────────────────────────────────────────

# All UR-type robots share the same alpha pattern
_UR_ALPHA = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

# Named presets: (a2, a3, d1, d4, d5, d6) in meters, signed per UR convention
# Source: Universal Robots support articles
#   https://www.universal-robots.com/articles/ur/application-installation/
#   dh-parameters-for-calculations-of-kinematics-and-dynamics/
_UR_PARAMS = {
    # e-series
    "UR3e":    dict(a2=-0.24355,  a3=-0.2132,   d1=0.15185,  d4=0.13105,  d5=0.08535, d6=0.0921),
    "UR5e":    dict(a2=-0.425,    a3=-0.3922,   d1=0.1625,   d4=0.1333,   d5=0.0997,  d6=0.0996),
    "UR7e":    dict(a2=-0.425,    a3=-0.3922,   d1=0.1625,   d4=0.1333,   d5=0.0997,  d6=0.0996),
    "UR10e":   dict(a2=-0.6127,   a3=-0.57155,  d1=0.1807,   d4=0.17415,  d5=0.11985, d6=0.11655),
    "UR12e":   dict(a2=-0.6127,   a3=-0.57155,  d1=0.1807,   d4=0.17415,  d5=0.11985, d6=0.11655),
    "UR16e":   dict(a2=-0.4784,   a3=-0.36,     d1=0.1807,   d4=0.17415,  d5=0.11985, d6=0.11655),
    "UR20":    dict(a2=-0.8620,   a3=-0.7287,   d1=0.2363,   d4=0.2010,   d5=0.1593,  d6=0.1543),
    "UR30":    dict(a2=-0.6370,   a3=-0.5037,   d1=0.2363,   d4=0.2010,   d5=0.1593,  d6=0.1543),
    # CB3 series
    "UR3":     dict(a2=-0.24365,  a3=-0.21325,  d1=0.1519,   d4=0.11235,  d5=0.08535, d6=0.0819),
    "UR5":     dict(a2=-0.425,    a3=-0.39225,  d1=0.089159, d4=0.10915,  d5=0.09465, d6=0.0823),
    "UR10":    dict(a2=-0.612,    a3=-0.5723,   d1=0.1273,   d4=0.163941, d5=0.1157,  d6=0.0922),
    # heavy payload
    "UR8long": dict(a2=-0.8989,   a3=-0.7149,   d1=0.2186,   d4=0.1824,   d5=0.1361,  d6=0.1434),
    "UR15":    dict(a2=-0.6475,   a3=-0.5164,   d1=0.2186,   d4=0.1824,   d5=0.1361,  d6=0.1434),
    "UR18":    dict(a2=-0.475,    a3=-0.3389,   d1=0.2186,   d4=0.1824,   d5=0.1361,  d6=0.1434),
}

# Standard UR tool frame convention — matches what the teach pendant reports.
# Rx(90°): z-axis pointing forward along the tool, y-axis up.
_UR_R6T = np.array([[1, 0,  0],
                    [0, 0, -1],
                    [0, 1,  0]], dtype=float)

# Solvers always available regardless of robot
_BASE_SOLVERS = ["auto", "general", "numpy"]


# ── public API ────────────────────────────────────────────────────────────────

def make_robot(
    name:   str,
    solver: str = "auto",
    R_6T:   np.ndarray = None,
) -> tuple[Callable, Callable]:
    """
    Return (fk, ik) callables for a named UR robot.

    Args:
        name:   Robot name, e.g. "UR10e". See available_robots().
        solver: Which IK solver to use:
                  "auto"    — fastest available (default)
                  "cse"     — CSE-generated, fast, robot-specific
                  "general" — IK-Geo general JAX solver
                  "numpy"   — numpy backend, for debugging
                See robot_infos(name)["available_solvers"] for what is
                available for a specific robot.
        R_6T:   Additional tool frame rotation applied on top of the standard
                UR convention (Rx 90°). Pass np.eye(3) to remove the default
                UR tool frame and get raw joint-frame output.

    Returns:
        fk: q -> (R, p)            forward kinematics
        ik: (R, p) -> (Q, valid)   IK, all 8 branches
                                   Q:     (6, 8) joint angles, NaN = infeasible
                                   valid: (8,)   bool mask of feasible branches

    Raises:
        ValueError: if name, solver, or combination is not supported.
    """
    if name not in _UR_PARAMS:
        raise ValueError(
            f"Unknown robot '{name}'. "
            f"Available: {sorted(_UR_PARAMS.keys())}.\n"
            f"For custom parameters use make_robot_from_dh()."
        )
    p = _UR_PARAMS[name]
    a = np.array([0, p["a2"], p["a3"], 0, 0, 0])
    d = np.array([p["d1"], 0, 0, p["d4"], p["d5"], p["d6"]])
    r6t = _UR_R6T if R_6T is None else _UR_R6T @ np.asarray(R_6T, dtype=float)

    return _build_callables(
        _UR_ALPHA, a, d, r6t,
        solver=solver,
        family="3p2i",
        robot_name=name,
    )


def make_robot_from_dh(
    alpha:  np.ndarray,
    a:      np.ndarray,
    d:      np.ndarray,
    solver: str = "auto",
    R_6T:   np.ndarray = None,
    family: str = "3p2i",
) -> tuple[Callable, Callable]:
    """
    Return (fk, ik) callables from raw DH parameters.

    Useful when DH parameters come from a URDF parser (e.g. pyroki, yourdfpy)
    or from a robot's calibration file.

    Args:
        alpha:  (6,) link twist angles [rad]
        a:      (6,) link lengths [m], signed per DH convention
        d:      (6,) link offsets [m]
        solver: "auto", "general", or "numpy". "cse" is not available for
                custom robots since it requires pre-generated code.
        R_6T:   Optional (3,3) tool frame rotation. Defaults to identity.
        family: Kinematic family. Currently only "3p2i" is supported.

    Returns:
        fk: q -> (R, p)
        ik: (R, p) -> (Q, valid)

    Raises:
        ValueError: if family or solver is not supported.
    """
    if family != "3p2i":
        raise ValueError(
            f"Unsupported kinematic family '{family}'. "
            f"Currently supported: '3p2i'."
        )
    r6t = np.asarray(R_6T, dtype=float) if R_6T is not None else np.eye(3)
    return _build_callables(alpha, a, d, r6t, solver=solver,
                            family=family, robot_name=None)


def available_robots() -> list:
    """Return sorted list of available named robot presets."""
    return sorted(_UR_PARAMS.keys())


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
    solvers = list(_BASE_SOLVERS)
    if _has_cse(name):
        solvers.insert(1, "cse")  # after "auto", before "general"
    return {
        "available_solvers": solvers,
    }


# ── internal ──────────────────────────────────────────────────────────────────

def _has_cse(name: str) -> bool:
    """Check if a CSE-generated solver exists for this robot."""
    module_name = f"jaik._jax._generated.ik_{name.lower()}"
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _resolve_solver(solver: str, robot_name: str | None) -> str:
    """Resolve 'auto' to the best available concrete solver."""
    if solver != "auto":
        return solver
    if robot_name is not None and _has_cse(robot_name):
        return "cse"
    return "general"


def _build_kin(alpha, a, d, R_6T, family):
    """Build and adjust a raw kin dict."""
    kin = dh_to_kin(alpha, a, d)
    if family == "3p2i":
        kin = adjust_kin_for_3p2i(kin)
    kin['RT'] = np.asarray(R_6T, dtype=float)
    return kin


def _build_callables(alpha, a, d, R_6T, solver, family, robot_name):
    resolved = _resolve_solver(solver, robot_name)

    if resolved == "cse":
        if robot_name is None:
            raise ValueError(
                "solver='cse' requires a named robot — use make_robot() "
                "instead of make_robot_from_dh()."
            )
        if not _has_cse(robot_name):
            raise ValueError(
                f"No CSE solver available for '{robot_name}'. "
                f"Run codegen first, or use solver='auto' or 'general'."
            )

    if resolved not in ("cse", "cse_sincos", "general", "numpy"):
        raise ValueError(
            f"Unknown solver '{solver}'. "
            f"Choose from: 'auto', 'cse', 'general', 'numpy'."
        )

    if resolved == "cse" and robot_name is None:
        raise ValueError(
            "solver='cse' requires a named robot. "
            "Use make_robot() instead of make_robot_from_dh()."
        )

    if resolved == "cse" and not _has_cse(robot_name):
        raise ValueError(
            f"No CSE solver available for '{robot_name}'. "
            f"Run codegen first, or use solver='auto' or 'general'."
        )

    kin = _build_kin(alpha, a, d, R_6T, family)

    if resolved == "numpy":
        return _make_numpy_callables(kin, family)
    elif resolved == "cse":
        return _make_cse_callables(kin, robot_name)
    elif resolved == "cse_sincos":
        return _make_cse_sincos_callables(kin, robot_name)
    else:
        return _make_jax_callables(kin, family)


def _make_jax_callables(kin, family):
    """JAX general solver (IK-Geo 3p2i)."""
    import jax.numpy as jnp
    from jaik._jax.fk import _fk as _fk_jax

    if family == "3p2i":
        from jaik._jax.ik_3p2i import ik_3_parallel_2_intersecting as _solver
    else:
        raise ValueError(f"No JAX solver for family '{family}'.")

    H  = jnp.asarray(kin['H'])
    P  = jnp.asarray(kin['P'])
    RT = jnp.asarray(kin['RT'])

    def fk(q):
        R, p = _fk_jax(q, H, P)
        return R @ RT, p

    def ik(R_target, p_target):
        R_06 = R_target @ RT.T
        Q, _ = _solver(R_06, p_target, H, P)
        valid = ~__import__('jax').numpy.isnan(Q).any(axis=0)
        return Q, valid

    return fk, ik


def _make_cse_callables(kin, robot_name):
    """CSE-generated solver for a specific named robot."""
    import jax.numpy as jnp
    from jaik._jax.fk import _fk as _fk_jax

    module_name = f"jaik._jax._generated.ik_{robot_name.lower()}"
    mod = importlib.import_module(module_name)
    fn_name = f"ik_{robot_name.lower()}"
    _solver = getattr(mod, fn_name)

    H  = jnp.asarray(kin['H'])
    P  = jnp.asarray(kin['P'])
    RT = jnp.asarray(kin['RT'])

    def fk(q):
        R, p = _fk_jax(q, H, P)
        return R @ RT, p

    def ik(R_target, p_target):
        R_06 = R_target @ RT.T
        return _solver(R_06, p_target)

    return fk, ik


def _make_cse_sincos_callables(kin, robot_name):
    """CSE-generated solver for a specific named robot."""
    import jax.numpy as jnp
    from jaik._jax.fk import _fk as _fk_jax

    module_name = f"jaik._jax._generated.ik_{robot_name.lower()}_sincos"
    mod = importlib.import_module(module_name)
    fn_name = f"ik_{robot_name.lower()}"
    _solver = getattr(mod, fn_name)

    H  = jnp.asarray(kin['H'])
    P  = jnp.asarray(kin['P'])
    RT = jnp.asarray(kin['RT'])

    def fk(q):
        R, p = _fk_jax(q, H, P)
        return R @ RT, p

    def ik(R_target, p_target):
        R_06 = R_target @ RT.T
        return _solver(R_06, p_target)

    return fk, ik


def _make_numpy_callables(kin, family):
    """Numpy solver for debugging."""
    from jaik._numpy.fk import _fk as _fk_np

    if family == "3p2i":
        from jaik._numpy.ik_3p2i import ik_3_parallel_2_intersecting as _solver
    else:
        raise ValueError(f"No numpy solver for family '{family}'.")

    RT = kin['RT']

    def fk(q):
        R, p = _fk_np(q, kin)
        return R @ RT, p

    def ik(R_target, p_target):
        R_06 = R_target @ RT.T
        Q, is_LS = _solver(R_06, p_target, kin)
        # convert is_LS to valid mask for API consistency
        valid = ~is_LS.any(axis=0)
        return Q, valid

    return fk, ik