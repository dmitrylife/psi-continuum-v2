# cosmology/data_loaders/validators.py

"""
A set of functions for checking the correctness of downloaded cosmological data.
Used in both analysis scripts and tests.
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path


def require_file(path: str | Path, instructions: str = "") -> Path:
    """
    Check that a required data file exists.
    If missing, raises FileNotFoundError with a human-readable message.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        msg = f"Required data file was not found:\n  {p}"
        if instructions:
            msg += f"\n\n{instructions}"
        raise FileNotFoundError(msg)
    return p


def validate_pantheonplus_dataset(data: Dict[str, Any]) -> None:
    """
    Checks the basic integrity of Pantheon+ HF data.

    Requires:
        - 'z', 'mu', 'mu_err', 'cov', 'N'
    """
    required_keys = ["z", "mu", "mu_err", "cov", "N"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Pantheon+ data key '{k}' is missing.")

    z = np.asarray(data["z"], dtype=float)
    mu = np.asarray(data["mu"], dtype=float)
    mu_err = np.asarray(data["mu_err"], dtype=float)
    cov = np.asarray(data["cov"], dtype=float)
    N = int(data["N"])

    if not (len(z) == len(mu) == len(mu_err) == N):
        raise ValueError(
            "Inconsistent dimensions: "
            f"len(z)={len(z)}, len(mu)={len(mu)}, len(mu_err)={len(mu_err)}, N={N}"
        )

    if cov.shape != (N, N):
        raise ValueError(
            f"Covariance matrix shape {cov.shape}, expected ({N}, {N})."
        )

    if (z < 0).any():
        raise ValueError("Pantheon+ contains negative redshifts (z < 0).")

    if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-10):
        raise ValueError("Covariance matrix is not symmetric.")

    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < -1e-8:
        raise ValueError(
            "Covariance matrix has negative eigenvalues: "
            f"min={eigvals.min():.3e}"
        )


def validate_hz_dataset(data: Dict[str, Any]) -> None:
    """
    Checks basic integrity of the H(z) compilation.
    Requires keys: 'z', 'Hz', 'sigma_Hz', 'N'.
    """
    required_keys = ["z", "Hz", "sigma_Hz", "N"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing required key '{k}' in H(z) dataset.")

    z = np.asarray(data["z"], dtype=float)
    Hz = np.asarray(data["Hz"], dtype=float)
    sigma_Hz = np.asarray(data["sigma_Hz"], dtype=float)
    N = int(data["N"])

    if not (len(z) == len(Hz) == len(sigma_Hz) == N):
        raise ValueError(
            f"Inconsistent dimensions: len(z)={len(z)}, len(Hz)={len(Hz)}, "
            f"len(sigma_Hz)={len(sigma_Hz)}, N={N}"
        )

    if (z < 0).any():
        raise ValueError("H(z) dataset contains negative redshifts (z < 0).")

    if (sigma_Hz <= 0).any():
        raise ValueError("H(z) dataset contains non-positive uncertainties sigma_Hz.")

    if (Hz <= 0).any() or (Hz > 1000).any():
        raise ValueError(
            "Some H(z) values are outside the reasonable range (0, 1000) km/s/Mpc."
        )
