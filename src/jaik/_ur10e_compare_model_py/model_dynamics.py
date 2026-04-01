import numpy as np
from .model_utils import compute_mass_matrix, compute_coriolis_matrix, compute_gravity_vector


def model_dynamics(q, qd, params):
    q = np.asarray(q).flatten()
    qd = np.asarray(qd).flatten()
    M = compute_mass_matrix(q, params)
    C = compute_coriolis_matrix(q, qd, params)
    G = compute_gravity_vector(q, params)
    return M, C, G
