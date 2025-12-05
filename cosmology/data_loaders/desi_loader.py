# cosmology/data_loaders/desi_loader.py

"""
DESI DR2 ALL_GCcomb Gaussian BAO loader for Psi-Continuum project.

This loader handles the REAL format of your files:

mean file lines look like:
    z   value   label

labels may be:
    DM_over_rs
    DH_over_rs
    DV_over_rs

The order of rows in mean-file is the order to be used in covariance.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def load_desi_dr2(data_dir: str | Path,
                  mean_filename="desi_gaussian_bao_ALL_GCcomb_mean.txt",
                  cov_filename="desi_gaussian_bao_ALL_GCcomb_cov.txt"):
    """
    Load DESI DR2 BAO data in the exact format Dmitry provided.

    Returns:
        {
            "z": array of length N,
            "label": list of strings,
            "values": array of length N,
            "vec": array (N,),
            "cov": array (N, N)
        }

    The order of rows is preserved exactly as in the mean file.
    """

    data_dir = Path(data_dir)

    mean_file = data_dir / mean_filename
    cov_file = data_dir / cov_filename

    if not mean_file.exists():
        raise FileNotFoundError(mean_file)
    if not cov_file.exists():
        raise FileNotFoundError(cov_file)

    # --- READ MEAN FILE ---
    z_list = []
    val_list = []
    label_list = []

    with open(mean_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Unexpected DESI mean-file format: {line}")

            z = float(parts[0])
            value = float(parts[1])
            label = parts[2]

            z_list.append(z)
            val_list.append(value)
            label_list.append(label)

    z = np.array(z_list, dtype=float)
    values = np.array(val_list, dtype=float)

    # --- READ COVARIANCE MATRIX ---
    cov = np.loadtxt(cov_file)
    N = len(values)

    if cov.shape != (N, N):
        raise ValueError(
            f"DESI covariance shape {cov.shape}, expected ({N}, {N})."
        )

    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("DESI covariance matrix is not symmetric")

    return {
        "z": z,
        "labels": label_list,
        "values": values,
        "vec": values.copy(),   # direct mapping
        "cov": cov
    }
