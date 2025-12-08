# analysis/check_desi_dr2_data.py

"""
Check DESI DR2 Gaussian BAO dataset:
 - loading
 - validation (covariance, labels, dimension)
 - summary
 - diagnostic plots DM/rs, DH/rs, DV/rs
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('psi_continuum_v2/analysis/styles/psi_style.mplstyle')

from psi_continuum_v2.cosmology.data_loaders.desi_loader import load_desi_dr2


def validate_covariance(name: str, cov: np.ndarray):
    """Ensure covariance is symmetric and positive definite."""
    if not np.isfinite(cov).all():
        raise ValueError(f"{name}: covariance contains NaN or Inf values.")

    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name}: covariance must be square, got {cov.shape}")

    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name}: covariance is not symmetric.")

    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name}: covariance is not positive definite.")

    print(f"{name}: covariance OK (symmetric, pos-def).")


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "desi" / "dr2"
    out_dir = project_root / "results" / "figures" / "data_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    desi = load_desi_dr2(data_dir)

    z = desi["z"]
    labels = desi["labels"]
    vec = desi["vec"]
    cov = desi["cov"]

    N = len(z)
    print("=== DESI DR2 BAO dataset check ===")
    print(f"N points        = {N}")
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
