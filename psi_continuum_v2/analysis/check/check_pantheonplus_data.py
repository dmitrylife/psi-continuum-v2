# analysis/check/check_pantheonplus_data.py

"""
Check Pantheon+ SH0ES HF supernova dataset:
    - Load and validate dataset
    - Print summary statistics
    - Generate basic diagnostic histograms:
        * redshift distribution
        * distance-modulus uncertainty distribution

Output directory (automatically determined):
    results/figures/data_checks/
"""

import numpy as np
import matplotlib.pyplot as plt

from psi_continuum_v2.utils import get_data_path, get_results_path
from psi_continuum_v2.cosmology.data_loaders import (
    load_pantheonplus_hf,
    validate_pantheonplus_dataset,
)


def main() -> None:
    # -------------------------------------------------------------
    # Locate input dataset and output directory
    # -------------------------------------------------------------
    data_dir = get_data_path("pantheon_plus", must_exist=True)
    fig_dir = get_results_path("figures", "data_checks")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Load and validate dataset
    # -------------------------------------------------------------
    sn = load_pantheonplus_hf(data_dir)
    validate_pantheonplus_dataset(sn)

    z = sn["z"]
    mu = sn["mu"]
    mu_err = sn["mu_err"]
    N = sn["N"]

    # Additional safety checks
    if not np.all(np.isfinite(z)):
        raise ValueError("Pantheon+: z contains non-finite values")

    if not np.all(np.isfinite(mu)):
        raise ValueError("Pantheon+: mu contains non-finite values")

    if not np.all(np.isfinite(mu_err)):
        raise ValueError("Pantheon+: mu_err contains non-finite values")

    # -------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------
    print("=== Pantheon+ HF data check ===")
    print(f"Total SNe      : {N}")
    print(f"Redshift range : [{z.min():.5f}, {z.max():.5f}]")
    print(f"μ range        : [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"Median σ_μ     : {np.median(mu_err):.3f}")
    print()

    # -------------------------------------------------------------
    # Histograms
    # -------------------------------------------------------------

    # Redshift distribution
    plt.figure(figsize=(6, 4))
    plt.hist(z, bins=40, color="tab:blue", alpha=0.8)
    plt.xlabel("Redshift z")
    plt.ylabel("Count")
    plt.title("Pantheon+ HF: Redshift distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "pantheonplus_z_hist.png", dpi=200)
    plt.close()

    # μ uncertainty distribution
    plt.figure(figsize=(6, 4))
    plt.hist(mu_err, bins=40, color="tab:green", alpha=0.8)
    plt.xlabel(r"Distance modulus uncertainty $\sigma_\mu$")
    plt.ylabel("Count")
    plt.title("Pantheon+ HF: μ uncertainty distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "pantheonplus_muerr_hist.png", dpi=200)
    plt.close()

    print("Diagnostic figures saved to:", fig_dir)


if __name__ == "__main__":
    main()
