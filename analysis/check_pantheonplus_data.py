# analysis/check_pantheonplus_data.py

"""
Quick check of Pantheon+ HF data:
- loading
- validation
- simple summary
- saving a couple of diagnostic graphs
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cosmology.data_loaders import (
    load_pantheonplus_hf,
    validate_pantheonplus_dataset,
)


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "pantheon_plus"
    results_dir = project_root / "results" / "figures" / "data_checks"
    results_dir.mkdir(parents=True, exist_ok=True)

    sn = load_pantheonplus_hf(data_dir)
    validate_pantheonplus_dataset(sn)

    z = sn["z"]
    mu = sn["mu"]
    mu_err = sn["mu_err"]
    N = sn["N"]

    print("=== Pantheon+ HF data check ===")
    print(f"N_SN       = {N}")
    print(f"z range    = [{z.min():.4f}, {z.max():.4f}]")
    print(f"mu range   = [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"mu_err med = {np.median(mu_err):.3f}")

    # Histograms z and mu_err
    plt.figure(figsize=(6, 4))
    plt.hist(z, bins=30)
    plt.xlabel("z")
    plt.ylabel("N")
    plt.tight_layout()
    plt.savefig(results_dir / "pantheonplus_z_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(mu_err, bins=30)
    plt.xlabel(r"$\sigma_\mu$")
    plt.ylabel("N")
    plt.tight_layout()
    plt.savefig(results_dir / "pantheonplus_muerr_hist.png", dpi=200)
    plt.close()

    print("The charts are saved in:", results_dir)


if __name__ == "__main__":
    main()

