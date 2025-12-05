# analysis/check_hz_data.py

"""
Quick H(z) compilation check:
- loading
- validation
- summary
- diagnostic plots
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cosmology.data_loaders import (
    load_hz_compilation,
    validate_hz_dataset,
)


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "hz"
    results_dir = project_root / "results" / "figures" / "data_checks"
    results_dir.mkdir(parents=True, exist_ok=True)

    hzdata = load_hz_compilation(data_dir)
    validate_hz_dataset(hzdata)

    z = hzdata["z"]
    Hz = hzdata["Hz"]
    sigma_Hz = hzdata["sigma_Hz"]
    N = hzdata["N"]

    print("=== H(z) data check ===")
    print(f"N_Hz           = {N}")
    print(f"z range        = [{z.min():.4f}, {z.max():.4f}]")
    print(f"H(z) range     = [{Hz.min():.3f}, {Hz.max():.3f}] km/s/Mpc")
    print(f"median sigma_Hz = {np.median(sigma_Hz):.3f}")

    # H(z) vs z
    plt.figure(figsize=(6, 4))
    plt.errorbar(z, Hz, yerr=sigma_Hz, fmt="o", alpha=0.8)
    plt.xlabel("z")
    plt.ylabel(r"$H(z)$ [km/s/Mpc]")
    plt.tight_layout()
    plt.savefig(results_dir / "hz_data_points.png", dpi=200)
    plt.close()

    # Relative errors σ_H / H
    rel_err = sigma_Hz / Hz
    plt.figure(figsize=(6, 4))
    plt.hist(rel_err, bins=20)
    plt.xlabel(r"$\sigma_H / H$")
    plt.ylabel("N")
    plt.tight_layout()
    plt.savefig(results_dir / "hz_relative_errors_hist.png", dpi=200)
    plt.close()

    print("The charts are saved in:", results_dir)


if __name__ == "__main__":
    main()
