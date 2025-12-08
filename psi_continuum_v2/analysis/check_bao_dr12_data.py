# # analysis/check_bao_dr12_data.py

"""
Check SDSS DR12 consensus BAO dataset:
 - loading
 - validation (covariance, dimensionality, ranges)
 - summary
 - diagnostic plots DM/rs and H*rs
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('psi_continuum_v2/analysis/styles/psi_style.mplstyle')

from psi_continuum_v2.cosmology.data_loaders.bao_loader import load_bao_dr12


def validate_covariance(name: str, cov: np.ndarray):
    """Check symmetry and positive-definiteness."""
    if not np.isfinite(cov).all():
        raise ValueError(f"{name}: covariance contains NaN or Inf.")

    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name}: covariance must be square, got {cov.shape}.")

    # symmetry
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name}: covariance is not symmetric.")

    # positive definite?
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name}: covariance is not positive-definite.")

    print(f"{name}: covariance OK (symmetric, pos-def).")


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "bao"
    out_dir = project_root / "results" / "figures" / "data_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    bao = load_bao_dr12(str(data_dir))

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

    # --------------- Plots ---------------
    plt.figure(figsize=(6, 4))
    plt.plot(z, dm_rs, "o-")
    plt.xlabel("z")
    plt.ylabel(r"$D_M/r_s$")
    plt.title("SDSS DR12 BAO: DM/rs")
    plt.tight_layout()
    plt.savefig(out_dir / "bao_dr12_DM_check.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(z, hz_rs, "o-")
    plt.xlabel("z")
    plt.ylabel(r"$H(z) r_s$ [km/s]")
    plt.title("SDSS DR12 BAO: H*rs")
    plt.tight_layout()
    plt.savefig(out_dir / "bao_dr12_Hz_check.png", dpi=200)
    plt.close()

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
