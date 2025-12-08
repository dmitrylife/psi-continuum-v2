# cosmology/data_loaders/covariance_loader.py

"""
Universal loader for SN covariance matrices in JLA / Pantheon / Pantheon+ style.

File format:
    First line: N  (number of SNe)
    Remaining lines: N*N floating-point numbers
                     stored either in C (row-major) or Fortran (column-major) order.
We auto-detect which ordering is closer to a symmetric matrix.
"""

from pathlib import Path
import numpy as np

# NEW: unified file validator
from psi_continuum_v2.cosmology.data_loaders.validators import require_file


def _symmetry_error(mat: np.ndarray) -> float:
    """
    Return a relative symmetry error ||M - M^T||_F / ||M||_F.
    """
    diff = mat - mat.T
    num = np.linalg.norm(diff, ord="fro")
    den = np.linalg.norm(mat, ord="fro")
    if den == 0.0:
        return 0.0
    return num / den


def load_sn_covariance(file: Path) -> np.ndarray:
    """
    Load SN covariance matrix from a file with format:

        N
        c11
        c12 or c21
        ...

    Returns:
        cov : ndarray, shape (N, N)
    """

    # NEW: use require_file for human-readable errors
    file = require_file(
        file,
        instructions=(
            "This covariance file is required for supernova likelihoods.\n"
            "Please download it and place it into the Pantheon+ or SN directory."
        )
    )

    with open(file, "r") as f:
        first_line = f.readline().strip()
        try:
            N = int(first_line)
        except ValueError:
            raise ValueError(f"Invalid first line in covariance file: {first_line}")

        values = np.fromfile(f, dtype=float, sep="\n")

    expected = N * N
    if values.size != expected:
        raise ValueError(
            f"Covariance file has {values.size} numbers after the first line, "
            f"expected {expected} (= {N}^2)."
        )

    # Build both candidates
    cov_C = values.reshape((N, N), order="C")
    cov_F = values.reshape((N, N), order="F")

    err_C = _symmetry_error(cov_C)
    err_F = _symmetry_error(cov_F)

    # Choose the more symmetric one
    cov = cov_C if err_C <= err_F else cov_F

    # Explicit symmetrization to remove tiny numerical asymmetries
    cov = 0.5 * (cov + cov.T)

    return cov
