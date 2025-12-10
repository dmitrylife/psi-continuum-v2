# analysis/check/check_bao_dr12_data.py

"""
Check SDSS DR12 consensus BAO dataset:
 - loading
 - validation (covariance, dimensionality, ranges)
 - summary
 - diagnostic plots DM/rs and H*rs
"""

import numpy as np
import matplotlib.pyplot as plt

from psi_continuum_v2.utils import get_data_path, get_results_path
from psi_continuum_v2.cosmology.data_loaders.bao_loader import load_bao_dr12


def validate_covariance(name: str, cov: np.ndarray) -> None:
    """Check symmetry and positive-definiteness of a covariance matrix."""
    if not np.isfinite(cov).all():
        raise ValueError(f"{name}: covariance contains NaN or Inf.")

    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name}: covariance must be square, got {cov.shape}.")

    # Symmetry check
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name}: covariance is not symmetric.")

    # Positive-definite check via Cholesky
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name}: covariance is not positive-definite.")

    print(f"{name}: covariance OK (symmetric, positive-definite).")


def main() -> None:
    # --- Locate datasets and results directory ---

    # data/bao/ lives under the active data root detected by get_data_path
    bao_dir = get_data_path("bao", must_exist=True)

    # results/figures/data_checks/ lives next to data/
    out_dir = get_results_path("figures", "data_checks")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load BAO DR12 data using the existing loader ---
    bao = load_bao_dr12(str(bao_dir))

    z = bao["z"]
    dm_rs = bao["dm_rs"]
    hz_rs = bao["hz_rs"]
    vec = bao["vec"]
    cov = bao["cov"]

    print("=== BAO DR12 dataset check ===")
    print(f"z values     : {z}")
    print(f"DM/rs        : {dm_rs}")
    print(f"H*rs         : {hz_rs}")
    print(f"Vector len   : {len(vec)} (expected 6)")
    print(f"Cov shape    : {cov.shape}")

    validate_covariance("BAO DR12", cov)

    # --- Diagnostic plots ---

    # DM/rs
    plt.figure(figsize=(6, 4))
    plt.plot(z, dm_rs, "o-")
    plt.xlabel("z")
    plt.ylabel(r"$D_M/r_s$")
    plt.title("SDSS DR12 BAO: DM/rs")
    plt.tight_layout()
    plt.savefig(out_dir / "bao_dr12_DM_check.png", dpi=200)
    plt.close()

    # H(z) * rs
    plt.figure(figsize=(6, 4))
    plt.plot(z, hz_rs, "o-")
    plt.xlabel("z")
    plt.ylabel(r"$H(z) r_s$ [km/s]")
    plt.title("SDSS DR12 BAO: H⋅r_s")
    plt.tight_layout()
    plt.savefig(out_dir / "bao_dr12_Hz_check.png", dpi=200)
    plt.close()

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
