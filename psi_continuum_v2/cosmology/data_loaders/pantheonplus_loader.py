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

from psi_continuum_v2.cosmology.data_loaders.validators import require_file
from psi_continuum_v2.cosmology.data_loaders.covariance_loader import load_sn_covariance


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

    # Use unified safe file validator
    data_file = require_file(
        base_dir / "Pantheon+SH0ES.dat",
        instructions=(
            "Pantheon+SH0ES.dat is missing.\n"
            "Download from the Pantheon+ HF dataset and place it into:\n"
            f"{base_dir}"
        )
    )

    cov_file = require_file(
        base_dir / "Pantheon+SH0ES_STAT+SYS.cov",
        instructions=(
            "Pantheon+SH0ES_STAT+SYS.cov is missing.\n"
            "Download the STAT+SYS covariance from the Pantheon+ dataset and place it into:\n"
            f"{base_dir}"
        )
    )

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
