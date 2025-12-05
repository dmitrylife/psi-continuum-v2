# cosmology/data_loaders/pantheonplus_loader.py

"""
Pantheon+SH0ES HF loader.

We read:
    - z   = zHD
    - mu  = MU_SH0ES
    - mu_err = MU_SH0ES_ERR_DIAG
from Pantheon+SH0ES.dat

and the full STAT+SYS covariance matrix from Pantheon+SH0ES_STAT+SYS.cov
using the generic SN covariance loader.
"""

from pathlib import Path
import numpy as np

from .covariance_loader import load_sn_covariance


def load_pantheonplus_hf(base_dir: Path | str | None = None):
    """
    Load Pantheon+SH0ES HF SN sample.

    Args:
        base_dir: directory that contains:
            - Pantheon+SH0ES.dat
            - Pantheon+SH0ES_STAT+SYS.cov

    Returns:
        dict with keys:
            'z'       : ndarray, shape (N,)
            'mu'      : ndarray, shape (N,)
            'mu_err'  : ndarray, shape (N,)
            'cov'     : ndarray, shape (N, N)
            'N'       : int
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2] / "data" / "pantheon_plus"
    else:
        base_dir = Path(base_dir)

    data_file = base_dir / "Pantheon+SH0ES.dat"
    cov_file = base_dir / "Pantheon+SH0ES_STAT+SYS.cov"

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not cov_file.exists():
        raise FileNotFoundError(f"Covariance file not found: {cov_file}")

    # Read SN table with header
    data = np.genfromtxt(
        data_file,
        names=True,
        dtype=None,
        encoding=None,
    )

    try:
        z = np.asarray(data["zHD"], dtype=float)
        mu = np.asarray(data["MU_SH0ES"], dtype=float)
        mu_err = np.asarray(data["MU_SH0ES_ERR_DIAG"], dtype=float)
    except KeyError as exc:
        raise KeyError(f"Missing required column in Pantheon+SH0ES.dat: {exc}")

    N = len(z)

    # Covariance (STAT+SYS)
    cov = load_sn_covariance(cov_file)
    if cov.shape != (N, N):
        raise ValueError(
            f"Covariance shape mismatch: cov.shape={cov.shape}, N={N}"
        )

    return {
        "z": z,
        "mu": mu,
        "mu_err": mu_err,
        "cov": cov,
        "N": N,
    }
