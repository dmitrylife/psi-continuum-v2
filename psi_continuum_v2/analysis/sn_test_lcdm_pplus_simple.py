# analysis/sn_test_lcdm_pplus_simple.py

"""
SN test for flat ΛCDM using Pantheon+SH0ES HF data.

Outputs:
- results/figures/sn/pantheonplus_hf_hubble_diagram.png
- results/figures/sn/pantheonplus_hf_residuals.png
- results/tables/sn/pantheonplus_hf_chi2_lcdm.txt
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('psi_continuum_v2/analysis/styles/psi_style.mplstyle')

from psi_continuum_v2.cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from psi_continuum_v2.cosmology.background.lcdm import dL_lcdm, mu_from_dL
from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
from psi_continuum_v2.cosmology.likelihoods.sn_likelihood import chi2_sn_full_cov


def main():
    project_root = Path(__file__).resolve().parents[2]

    # ------------------------------------------------------------------
    # Output folders (figures and tables)
    # ------------------------------------------------------------------
    results_dir = project_root / "results"

    fig_dir = results_dir / "figures" / "sn"
    tab_dir = results_dir / "tables" / "sn"

    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Pantheon+ HF dataset
    # ------------------------------------------------------------------
    sn = load_pantheonplus_hf(project_root / "data" / "pantheon_plus")
    z = sn["z"]
    mu_obs = sn["mu"]
    cov = sn["cov"]

    # ------------------------------------------------------------------
    # 2. ΛCDM parameters (fixed; this script does not fit them)
    # ------------------------------------------------------------------
    lcdm_params = LCDMParams(H0=70.0, Om0=0.3)

    # ------------------------------------------------------------------
    # 3. Theoretical distance modulus
    # ------------------------------------------------------------------
    dL = dL_lcdm(z, lcdm_params)
    mu_th = mu_from_dL(dL)

    # ------------------------------------------------------------------
    # 4. χ² goodness-of-fit
    # ------------------------------------------------------------------
    chi2 = chi2_sn_full_cov(mu_obs, mu_th, cov)
    dof = len(z) - 2  # approximately two free parameters (H0, Om0)
    red_chi2 = chi2 / dof

    print(f"Pantheon+ HF ΛCDM: χ² = {chi2:.2f}, χ²/dof = {red_chi2:.3f}")

    # ------------------------------------------------------------------
    # 5. Hubble diagram μ(z)
    # ------------------------------------------------------------------
    order = np.argsort(z)
    z_sorted = z[order]
    mu_obs_sorted = mu_obs[order]
    mu_th_sorted = mu_th[order]

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        z_sorted,
        mu_obs_sorted,
        yerr=np.sqrt(np.diag(cov))[order],
        fmt=".",
        alpha=0.6,
        label="Pantheon+ HF",
    )
    plt.plot(
        z_sorted,
        mu_th_sorted,
        "-",
        label=r"ΛCDM, $H_0=70$, $\Omega_m=0.3$",
    )
    plt.xlabel("z")
    plt.ylabel(r"$\mu(z)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pantheonplus_hf_hubble_diagram.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # 6. Residuals: μ_obs − μ_th
    # ------------------------------------------------------------------
    residuals = mu_obs - mu_th

    plt.figure(figsize=(7, 4))
    plt.axhline(0.0, ls="--", color="gray")
    plt.errorbar(z, residuals, yerr=np.sqrt(np.diag(cov)), fmt=".", alpha=0.6)
    plt.xlabel("z")
    plt.ylabel(r"$\mu_{\rm obs} - \mu_{\rm th}$")
    plt.tight_layout()
    plt.savefig(fig_dir / "pantheonplus_hf_residuals.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # 7. Save χ² summary table
    # ------------------------------------------------------------------
    with open(tab_dir / "pantheonplus_hf_chi2_lcdm.txt", "w", encoding="utf-8") as f:
        f.write("# Pantheon+ HF ΛCDM fit\n")
        f.write(f"N_SN = {len(z)}\n")
        f.write(f"H0 = {lcdm_params.H0}\n")
        f.write(f"Om0 = {lcdm_params.Om0}\n")
        f.write(f"chi2 = {chi2:.6f}\n")
        f.write(f"dof = {dof}\n")
        f.write(f"chi2_reduced = {red_chi2:.6f}\n")


if __name__ == "__main__":
    main()
