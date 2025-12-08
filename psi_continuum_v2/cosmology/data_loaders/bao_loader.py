# cosmology/data_loaders/bao_loader.py

"""
BAO data loaders:

- SDSS DR12 consensus BAO: DM(z), H(z)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from psi_continuum_v2.cosmology.data_loaders.validators import require_file


# ----------------- SDSS DR12 BAO ----------------- #

def load_bao_dr12(data_dir: str | Path):
    """
    Load SDSS DR12 Consensus BAO measurements.

    Expected files:
        sdss_DR12Consensus_bao.dat
        BAO_consensus_covtot_dM_Hz.txt

    Returns:
        dict:
            z      – array of redshifts (3,)
            dm_rs  – array of DM/rs (3,)
            hz_rs  – array of H(z)*rs (3,)
            vec    – 6×1 BAO data vector
            cov    – 6×6 covariance matrix
    """

    data_dir = Path(data_dir)

    mean_file = require_file(
        data_dir / "sdss_DR12Consensus_bao.dat",
        instructions=(
            "Required BAO file 'sdss_DR12Consensus_bao.dat' was not found.\n"
            "Please download it and place into your BAO directory, e.g.:\n"
            "    data/bao/"
        )
    )

    cov_file = require_file(
        data_dir / "BAO_consensus_covtot_dM_Hz.txt",
        instructions=(
            "Required BAO covariance file 'BAO_consensus_covtot_dM_Hz.txt' was not found.\n"
            "Please download it and place into your BAO directory, e.g.:\n"
            "    data/bao/"
        )
    )

    # Load mean BAO table (DM/rs and H(z)*rs alternating rows)
    raw = np.loadtxt(mean_file, usecols=(0, 1))
    z = raw[::2, 0]     # 0.38, 0.51, 0.61
    dm_rs = raw[::2, 1]
    hz_rs = raw[1::2, 1]

    if not (len(z) == len(dm_rs) == len(hz_rs) == 3):
        raise ValueError(
            f"Unexpected BAO format: expected exactly 3 redshifts, got {len(z)}."
        )

    # Build the 6D data vector in DR12 standard order
    vec = np.array([
        dm_rs[0], hz_rs[0],
        dm_rs[1], hz_rs[1],
        dm_rs[2], hz_rs[2],
    ])

    # Covariance
    cov = np.loadtxt(cov_file)
    if cov.shape != (6, 6):
        raise ValueError(
            f"BAO covariance matrix must be 6×6, got shape {cov.shape}."
        )

    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("BAO covariance matrix is not symmetric.")

    return {
        "z": z,
        "dm_rs": dm_rs,
        "hz_rs": hz_rs,
        "vec": vec,
        "cov": cov,
    }
