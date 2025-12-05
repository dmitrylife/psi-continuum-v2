# cosmology/likelihoods/bao_likelihood.py

"""
BAO likelihood for SDSS DR12 Consensus dataset.
"""

import numpy as np
from numpy.linalg import inv


def bao_vector_model(z, model_dm, model_hz):
    """
    Build 6D model vector from DM(z)/rs and H(z)*rs values.

    Arguments:
        z         – array of redshifts (3)
        model_dm  – DM(z)/rs model values (3)
        model_hz  – H(z)*rs model values (3)

    Returns:
        vector (6,)
    """
    vec = np.zeros(6)
    vec[0] = model_dm[0]
    vec[1] = model_hz[0]
    vec[2] = model_dm[1]
    vec[3] = model_hz[1]
    vec[4] = model_dm[2]
    vec[5] = model_hz[2]
    return vec


def chi2_bao(data_vec, cov, model_vec):
    """
    Compute χ² for BAO:

        χ² = (d - m)^T C^{-1} (d - m)
    """
    delta = data_vec - model_vec
    inv_cov = inv(cov)
    return float(delta.T @ inv_cov @ delta)

