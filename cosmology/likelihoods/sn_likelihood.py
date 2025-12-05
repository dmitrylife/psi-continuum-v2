# cosmology/likelihoods/sn_likelihood.py

from typing import Tuple, Callable

import numpy as np


def chi2_sn_full_cov(
    mu_obs: np.ndarray,
    mu_th: np.ndarray,
    cov: np.ndarray,
) -> float:
    """
    Full χ² taking into account the covariance matrix:
    χ² = (μ_obs - μ_th)^T C^{-1} (μ_obs - μ_th)
    """
    mu_obs = np.asarray(mu_obs, dtype=float)
    mu_th = np.asarray(mu_th, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if mu_obs.shape != mu_th.shape:
        raise ValueError("mu_obs and mu_th must have the same length")
    if cov.shape[0] != cov.shape[1] or cov.shape[0] != mu_obs.size:
        raise ValueError("The cov dimension is inconsistent with the data")

    diff = mu_obs - mu_th
    inv_cov = np.linalg.inv(cov)
    return float(diff.T @ inv_cov @ diff)


def sn_loglike_from_model(
    z: np.ndarray,
    mu_obs: np.ndarray,
    cov: np.ndarray,
    mu_model: Callable[[np.ndarray], np.ndarray],
) -> Tuple[float, float]:
    """
    Auxiliary wrapper:
    - takes the model μ(z)
    - returns (χ², -0.5 χ²) — χ² and log L with normalization without a constant.
    """
    mu_th = mu_model(z)
    chi2 = chi2_sn_full_cov(mu_obs, mu_th, cov)
    return chi2, -0.5 * chi2
