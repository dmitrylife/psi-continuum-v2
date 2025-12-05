# cosmology/data_loaders/validators.py

"""
A set of functions for checking the correctness of downloaded cosmological data.
Used in both analysis scripts and tests.
"""

from typing import Dict, Any
import numpy as np


def validate_pantheonplus_dataset(data: Dict[str, Any]) -> None:
    """
    Checks the basic integrity of Pantheon+ HF data.
        Requires the following keys:
            - 'z'
            - 'mu'
            - 'mu_err'
            - 'cov'
            - 'N'
        Checks:
            - Dimension consistency
            - z >= 0
            - Covariance matrix is ​​square and symmetric
            - Covariance is positive definite (or at least semi-definite)
    """
    required_keys = ["z", "mu", "mu_err", "cov", "N"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Pantheon+ data key missing '{k}'")

    z = np.asarray(data["z"], dtype=float)
    mu = np.asarray(data["mu"], dtype=float)
    mu_err = np.asarray(data["mu_err"], dtype=float)
    cov = np.asarray(data["cov"], dtype=float)
    N = int(data["N"])

    if not (len(z) == len(mu) == len(mu_err) == N):
        raise ValueError(
            "The dimensions z/mu/mu_err/N do not match: "
            f"len(z)={len(z)}, len(mu)={len(mu)}, len(mu_err)={len(mu_err)}, N={N}"
        )

    if cov.shape != (N, N):
        raise ValueError(
            f"Dimension of the covariance matrix {cov.shape}, expected ({N}, {N})"
        )

    # z >= 0
    if np.any(z < 0):
        raise ValueError("There are negative z's in Pantheon+ data")

    # Covariance symmetry
    if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-10):
        raise ValueError("The covariance matrix is not symmetric.")

    # Positive definiteness (or at least semi-definite)
    # Checking using eigenvalues
    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) < -1e-8:
        raise ValueError(
            "The covariance matrix has essentially negative eigenvalues"
            f"(min eigenvalue = {np.min(eigvals):.3e})"
        )


def validate_hz_dataset(data: Dict[str, Any]) -> None:
    """
    Checks the basic integrity of the H(z) compilation.
        Expects keys:
            - 'z'
            - 'Hz'
            - 'sigma_Hz'
            - 'N'

        Checks:
            - size consistency
            - z >= 0
            - sigma_Hz > 0
            - H(z) values ​​within reasonable limits (e.g., (0, 1000) km/s/Mpc)
    """
    required_keys = ["z", "Hz", "sigma_Hz", "N"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"There is no key in the H(z) data '{k}'")

    z = np.asarray(data["z"], dtype=float)
    Hz = np.asarray(data["Hz"], dtype=float)
    sigma_Hz = np.asarray(data["sigma_Hz"], dtype=float)
    N = int(data["N"])

    if not (len(z) == len(Hz) == len(sigma_Hz) == N):
        raise ValueError(
            "The dimensions z/Hz/sigma_Hz/N do not match: "
            f"len(z)={len(z)}, len(Hz)={len(Hz)}, len(sigma_Hz)={len(sigma_Hz)}, N={N}"
        )

    if np.any(z < 0):
        raise ValueError("There are negative z's in the H(z) data")

    if np.any(sigma_Hz <= 0):
        raise ValueError("There are non-positive errors sigma_Hz in the H(z) data")

    # Simple sanity-check ranges
    if np.any(Hz <= 0) or np.any(Hz > 1000):
        raise ValueError(
            "Some values ​​of H(z) are outside the reasonable range (0, 1000) km/s/Mpc"
        )
