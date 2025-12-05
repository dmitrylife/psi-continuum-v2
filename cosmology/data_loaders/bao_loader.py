# cosmology/data_loaders/bao_loader.py

"""
BAO data loaders:

- SDSS DR12 consensus BAO: DM(z), H(z)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import os


# ----------------- SDSS DR12 BAO ----------------- #

def load_bao_dr12(data_dir):
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
            vec    – 6×1 vector [dm1, hz1, dm2, hz2, dm3, hz3]
            cov    – 6×6 covariance matrix
    """

    mean_file = os.path.join(data_dir, "sdss_DR12Consensus_bao.dat")
    cov_file  = os.path.join(data_dir, "BAO_consensus_covtot_dM_Hz.txt")

    # Load mean BAO values
    raw = np.loadtxt(mean_file, usecols=(0, 1))
    z = raw[::2, 0]  # 0.38, 0.51, 0.61 (every 2 rows)

    dm_rs = raw[::2, 1]  # DM/rs at each z
    hz_rs = raw[1::2, 1]  # H(z)*rs at each z

    # Confirm shape
    if not (len(z) == len(dm_rs) == len(hz_rs) == 3):
        raise ValueError("Unexpected BAO data format")

    # Build 6D data vector
    data_vec = np.zeros(6)
    data_vec[0] = dm_rs[0]
    data_vec[1] = hz_rs[0]
    data_vec[2] = dm_rs[1]
    data_vec[3] = hz_rs[1]
    data_vec[4] = dm_rs[2]
    data_vec[5] = hz_rs[2]

    # Load covariance
    cov = np.loadtxt(cov_file)
    if cov.shape != (6, 6):
        raise ValueError("BAO covariance matrix must be 6×6.")

    # Symmetry check
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("BAO covariance matrix is not symmetric!")

    return {
        "z": z,
        "dm_rs": dm_rs,
        "hz_rs": hz_rs,
        "vec": data_vec,
        "cov": cov,
    }
