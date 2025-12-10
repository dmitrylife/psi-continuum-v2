# analysis/check/check_desi_dr2_data.py

"""
Check DESI DR2 Gaussian BAO dataset:
 - loading
 - validation (covariance, labels, dimension)
 - summary
 - diagnostic plots DM/rs, DH/rs, DV/rs
"""

import numpy as np
import matplotlib.pyplot as plt

from psi_continuum_v2.utils import get_data_path, get_results_path
from psi_continuum_v2.cosmology.data_loaders.desi_loader import load_desi_dr2


def validate_covariance(name: str, cov: np.ndarray) -> None:
    """Ensure covariance is symmetric and positive definite."""
    if not np.isfinite(cov).all():
        raise ValueError(f"{name}: covariance contains NaN or Inf values.")

    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name}: covariance must be square, got {cov.shape}")

    # Symmetry check
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name}: covariance is not symmetric.")

    # Positive-definite check via Cholesky
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name}: covariance is not positive definite.")

    print(f"{name}: covariance OK (symmetric, pos-def).")


def main() -> None:
    # --- Locate datasets and results directory ---

    # DESI DR2 Gaussian BAO lives under data/desi/dr2/
    desi_dir = get_data_path("desi", "dr2", must_exist=True)

    # results/figures/data_checks/ lives next to data/
    out_dir = get_results_path("figures", "data_checks")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load DESI DR2 data using the existing loader ---
    desi = load_desi_dr2(desi_dir)

    z = desi["z"]
    labels = desi["labels"]
    vec = desi["vec"]
    cov = desi["cov"]

    n_points = len(z)
    print("=== DESI DR2 BAO dataset check ===")
    print(f"N points        = {n_points}")
    print(f"Labels          = {labels}")
    print(f"z range         = [{z.min():.3f}, {z.max():.3f}]")
    print(f"Vector length   = {len(vec)}")
    print(f"Cov shape       = {cov.shape}")

    if len(labels) != len(z) or len(vec) != len(z):
        raise ValueError("DESI DR2: mismatched lengths of z / labels / vec.")

    # Validate labels
    for lab in labels:
        if not (lab.startswith("DM") or lab.startswith("DH") or lab.startswith("DV")):
            raise ValueError(f"Bad DESI label: {lab}")

    validate_covariance("DESI DR2", cov)

    # -------- Separate DM, DH, DV --------
    DM_z, DM_val = [], []
    DH_z, DH_val = [], []
    DV_z, DV_val = [], []

    for zi, lab, val in zip(z, labels, vec):
        if lab.startswith("DM"):
            DM_z.append(zi)
            DM_val.append(val)
        elif lab.startswith("DH"):
            DH_z.append(zi)
            DH_val.append(val)
        elif lab.startswith("DV"):
            DV_z.append(zi)
            DV_val.append(val)

    # -------- Plots --------
    if DM_z:
        plt.figure(figsize=(6, 4))
        plt.plot(DM_z, DM_val, "o-")
        plt.xlabel("z")
        plt.ylabel("DM/rs")
        plt.title("DESI DR2: DM/rs")
        plt.tight_layout()
        plt.savefig(out_dir / "desi_dr2_DM_check.png", dpi=200)
        plt.close()

    if DH_z:
        plt.figure(figsize=(6, 4))
        plt.plot(DH_z, DH_val, "o-")
        plt.xlabel("z")
        plt.ylabel("DH/rs")
        plt.title("DESI DR2: DH/rs")
        plt.tight_layout()
        plt.savefig(out_dir / "desi_dr2_DH_check.png", dpi=200)
        plt.close()

    if DV_z:
        plt.figure(figsize=(6, 4))
        plt.plot(DV_z, DV_val, "o-")
        plt.xlabel("z")
        plt.ylabel("DV/rs")
        plt.title("DESI DR2: DV/rs")
        plt.tight_layout()
        plt.savefig(out_dir / "desi_dr2_DV_check.png", dpi=200)
        plt.close()

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
