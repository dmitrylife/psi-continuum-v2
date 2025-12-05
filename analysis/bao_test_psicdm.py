# analysis/bao_test_psicdm.py

"""
BAO DR12 test for ΛCDM and ΨCDM.

We use the SDSS DR12 consensus BAO data:
    - sdss_DR12Consensus_bao.dat
    - BAO_consensus_covtot_dM_Hz.txt

Data vector (length 6):
    [DM(z1)/rs, H(z1)*rs, DM(z2)/rs, H(z2)*rs, DM(z3)/rs, H(z3)*rs]

Units:
    DM/rs   – dimensionless
    H*rs    – km/s

For a flat universe:
    DM(z) = d_L(z) / (1 + z),  DM in Mpc
    We then use:
        DM/rs = DM / r_d
        H*rs  = H(z) * r_d

This script:
    1) loads BAO data and covariance,
    2) computes ΛCDM and ΨCDM predictions for DM/rs and H*rs,
    3) evaluates χ² for each model,
    4) produces diagnostic plots.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cosmology.data_loaders.bao_loader import load_bao_dr12
from cosmology.likelihoods.bao_likelihood import bao_vector_model, chi2_bao

from cosmology.background.lcdm import H_lcdm, dL_lcdm
from cosmology.background.psicdm import H_psicdm, dL_psicdm

from cosmology.models.lcdm_params import LCDMParams
from cosmology.models.psicdm_params import PsiCDMParams


def dm_from_dl(z: np.ndarray, dL: np.ndarray) -> np.ndarray:
    """
    Transverse comoving distance DM(z) for a flat universe:
        DM = d_L / (1 + z)
    d_L is assumed to be in Mpc, DM will be in Mpc.
    """
    z = np.asarray(z, dtype=float)
    dL = np.asarray(dL, dtype=float)
    return dL / (1.0 + z)


def main():
    # Resolve project paths
    project_root = Path(__file__).resolve().parents[1]
    bao_dir = project_root / "data" / "bao"
    fig_dir = project_root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load BAO DR12 dataset
    bao = load_bao_dr12(str(bao_dir))
    z_data = bao["z"]          # shape (3,)
    # These are DM/rs and H(z)*rs, as loaded in bao_loader
    dm_obs = bao["dm_rs"]      # DM(z)/r_s (dimensionless)
    hz_obs = bao["hz_rs"]      # H(z)*r_s [km/s]
    data_vec = bao["vec"]      # 6D data vector [DM/rs, H*rs, ...]
    cov = bao["cov"]           # 6x6 covariance for [DM/rs, H*rs, ...]

    # Extract 1D uncertainties from diagonal for plotting
    cov_diag = np.diag(cov)
    sigma_dm = np.sqrt(cov_diag[0::2])  # indices 0,2,4 → DM/rs
    sigma_hz = np.sqrt(cov_diag[1::2])  # indices 1,3,5 → H*rs

    # Baseline ΛCDM parameters
    lcdm_params = LCDMParams(H0=70.0, Om0=0.3)

    # Example ΨCDM parameters (can be tuned later)
    psicdm_params = PsiCDMParams(
        H0=70.0,
        Om0=0.3,
        eps0=0.05,  # small deviation from ΛCDM
        n=1.0,
        rd=lcdm_params.rd,  # use the same fiducial sound horizon
    )

    rd = lcdm_params.rd  # BAO sound horizon in Mpc

    # ΛCDM predictions at BAO redshifts
    dL_lcdm_val = dL_lcdm(z_data, lcdm_params)           # d_L [Mpc]
    dm_lcdm_mpc = dm_from_dl(z_data, dL_lcdm_val)        # DM [Mpc]
    dm_lcdm = dm_lcdm_mpc / rd                           # DM/rs (dimensionless)
    hz_lcdm = H_lcdm(z_data, lcdm_params) * rd           # H(z)*rs [km/s]

    # ΨCDM predictions at BAO redshifts
    dL_psi_val = dL_psicdm(z_data, psicdm_params)        # d_L [Mpc]
    dm_psi_mpc = dm_from_dl(z_data, dL_psi_val)          # DM [Mpc]
    dm_psi = dm_psi_mpc / rd                             # DM/rs
    hz_psi = H_psicdm(z_data, psicdm_params) * rd        # H(z)*rs [km/s]

    # Build model vectors for χ² (must match the structure of data_vec)
    model_vec_lcdm = bao_vector_model(z_data, dm_lcdm, hz_lcdm)
    model_vec_psi = bao_vector_model(z_data, dm_psi, hz_psi)

    # Compute χ² for both models
    chi2_lcdm = chi2_bao(data_vec, cov, model_vec_lcdm)
    chi2_psi = chi2_bao(data_vec, cov, model_vec_psi)

    dof = len(data_vec)  # 6 data points, no fit parameters here

    print("=== BAO DR12 test (DM/rs, H*rs) ===")
    print(f"LCDM parameters: H0={lcdm_params.H0}, Om0={lcdm_params.Om0}, rd={lcdm_params.rd}")
    print(f"PsiCDM parameters: H0={psicdm_params.H0}, Om0={psicdm_params.Om0}, "
          f"eps0={psicdm_params.eps0}, n={psicdm_params.n}, rd={psicdm_params.rd}")
    print()
    print(f"χ²_LCDM   = {chi2_lcdm:.3f}  for {dof} data points")
    print(f"χ²_PsiCDM = {chi2_psi:.3f}  for {dof} data points")
    print(f"Δχ² (PsiCDM - LCDM) = {chi2_psi - chi2_lcdm:+.3f}")

    # ----------- Plots -----------

    # 1) DM/rs(z) comparison
    plt.figure(figsize=(6, 4))
    plt.errorbar(z_data, dm_obs, yerr=sigma_dm, fmt="o", label="BAO DR12")
    plt.plot(z_data, dm_lcdm, "-o", label=r"$\Lambda$CDM")
    plt.plot(z_data, dm_psi, "-o", label=r"$\Psi$CDM")
    plt.xlabel("z")
    plt.ylabel(r"$D_M(z)/r_s$")
    plt.title(r"BAO DR12: $D_M(z)/r_s$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "bao_dr12_DM_psicdm.png", dpi=200)
    plt.close()

    # 2) H(z)*rs comparison
    plt.figure(figsize=(6, 4))
    plt.errorbar(z_data, hz_obs, yerr=sigma_hz, fmt="o", label="BAO DR12")
    plt.plot(z_data, hz_lcdm, "-o", label=r"$\Lambda$CDM")
    plt.plot(z_data, hz_psi, "-o", label=r"$\Psi$CDM")
    plt.xlabel("z")
    plt.ylabel(r"$H(z)\, r_s\ [\mathrm{km/s}]$")
    plt.title(r"BAO DR12: $H(z)\, r_s$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "bao_dr12_Hz_psicdm.png", dpi=200)
    plt.close()

    print("Figures saved to:", fig_dir)


if __name__ == "__main__":
    main()
