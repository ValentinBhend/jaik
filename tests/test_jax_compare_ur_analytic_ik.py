import pytest
import numpy as np
import ur_analytic_ik
import jaik
from utils import compare_solvers

_MODELS = ["ur3", "ur3e", "ur5", "ur5e", "ur7e", "ur8long", "ur10", "ur10e", "ur12e", "ur15", "ur16e", "ur18", "ur20", "ur30"]
_SOLVERS = ["jax", "numba"]

@pytest.fixture(scope="module", params=[(model, solver) for model in _MODELS for solver in _SOLVERS])
def solver_pair(request):
    name, solver = request.param
    try:
        fk, ik_full, _ = jaik.make_robot(name, solver=solver)
    except ValueError:
        pytest.skip(f"jaik does not support {name} yet")

    if not hasattr(ur_analytic_ik, name):
        pytest.skip(f"ur_analytic_ik does not support {name}")
    ref = getattr(ur_analytic_ik, name)

    def ik_ref(T):
        sols = ref.inverse_kinematics(T)
        return [] if sols is None else list(sols)

    fk_ref = ref.forward_kinematics
    return fk_ref, ik_ref, fk, ik_full, name, solver

def test_ik_matches_reference(solver_pair):
    fk_ref, ik_ref, fk, ik_full, name, solver = solver_pair
    passed, info = compare_solvers(
        fk_ref=fk_ref,
        ik_ref=ik_ref,
        fk_jaik=fk,
        ik_jaik=ik_full,
        n_poses=500,
        verbose=False,
        solver=solver,
    )
    assert passed, (
        f"{name}: comparison failed — "
        f"count_mismatches={info['count_mismatches']}, "
        f"no_solution_poses={info['no_solution_poses']}, "
        f"max_err={info['errors'].max():.2e} rad"
    )