# analysis/check/check_hz_data.py

"""
Quick H(z) compilation check:
- loading
- validation
- summary
- diagnostic plots
"""

import numpy as np
import matplotlib.pyplot as plt

from psi_continuum_v2.utils import get_data_path, get_results_path
from psi_continuum_v2.cosmology.data_loaders import (
    load_hz_compilation,
    validate_hz_dataset,
)


def main() -> None:
    # --- Locate data and results directories ---

    # H(z) compilation lives under data/hz/
    hz_dir = get_data_path("hz", must_exist=True)

    # results/figures/data_checks/ next to data/
    results_dir = get_results_path("figures", "data_checks")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and validate dataset ---
    hzdata = load_hz_compilation(hz_dir)
    validate_hz_dataset(hzdata)

    z = hzdata["z"]
    Hz = hzdata["Hz"]
    sigma_Hz = hzdata["sigma_Hz"]
    N = hzdata["N"]

    print("=== H(z) data check ===")
    print(f"N_Hz            = {N}")
    print(f"z range         = [{z.min():.4f}, {z.max():.4f}]")
    print(f"H(z) range      = [{Hz.min():.3f}, {Hz.max():.3f}] km/s/Mpc")
    print(f"median sigma_Hz = {np.median(sigma_Hz):.3f}")

    # -----------------------------
    # Plot 1: H(z) points with errors
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.errorbar(z, Hz, yerr=sigma_Hz, fmt="o", alpha=0.8)
    plt.xlabel("z")
    plt.ylabel(r"$H(z)$ [km/s/Mpc]")
    plt.title("H(z) compilation")
    plt.tight_layout()
    plt.savefig(results_dir / "hz_data_points.png", dpi=200)
    plt.close()

    # -----------------------------
    # Plot 2: histogram of relative errors
    # -----------------------------
    rel_err = sigma_Hz / Hz

    plt.figure(figsize=(6, 4))
    plt.hist(rel_err, bins=20)
    plt.xlabel(r"$\sigma_H / H$")
    plt.ylabel("N")
    plt.title("Relative H(z) errors")
    plt.tight_layout()
    plt.savefig(results_dir / "hz_relative_errors_hist.png", dpi=200)
    plt.close()

    print("The charts are saved in:", results_dir)


if __name__ == "__main__":
    main()
