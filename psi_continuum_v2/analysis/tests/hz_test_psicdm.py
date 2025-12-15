# analysis/tests/hz_test_psicdm.py

"""
ΨCDM vs ΛCDM test using the compiled H(z) dataset.

Outputs:
- results/figures/hz/hz_psicdm_test.png
- results/figures/hz/hz_psicdm_chi2_eps_scan.png
- results/tables/hz/hz_psicdm_chi2.txt
"""

import numpy as np
import matplotlib.pyplot as plt

from psi_continuum_v2.utils import get_data_path, get_results_path
from psi_continuum_v2.cosmology.data_loaders.hz_loader import load_hz_compilation
from psi_continuum_v2.cosmology.background.lcdm import H_lcdm
from psi_continuum_v2.cosmology.background.psicdm import H_psicdm
from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams
from psi_continuum_v2.cosmology.likelihoods.hz_likelihood import chi2_hz


def main():

    # ------------------------------------------------------------------
    # 1. Input and output directories
    # ------------------------------------------------------------------
    hzdata = load_hz_compilation(get_data_path("hz"))

    fig_dir = get_results_path("figures", "hz")
    fig_dir.mkdir(parents=True, exist_ok=True)

    tab_dir = get_results_path("tables", "hz")
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. H(z) dataset
    # ------------------------------------------------------------------
    z = hzdata["z"]
    Hz_obs = hzdata["Hz"]
    sigma_Hz = hzdata["sigma_Hz"]

    # ------------------------------------------------------------------
    # 3. ΛCDM baseline
    # ------------------------------------------------------------------
    lcdm_params = LCDMParams(H0=70.0, Om0=0.3)

    chi2_lcdm = chi2_hz(hzdata, H_lcdm, lcdm_params)
    dof = len(z) - 2
    red_chi2_lcdm = chi2_lcdm / dof

    print(f"H(z) ΛCDM: χ² = {chi2_lcdm:.2f}, χ²/dof = {red_chi2_lcdm:.3f}")

    # ------------------------------------------------------------------
    # 4. ΨCDM ε₀ scan
    # ------------------------------------------------------------------
    eps_vals = np.linspace(-0.2, 0.2, 81)
    chi2_vals = []

    for eps in eps_vals:
        params = PsiCDMParams(
            H0=lcdm_params.H0,
            Om0=lcdm_params.Om0,
            eps0=eps,
            n=1.0,
        )
        chi2_vals.append(chi2_hz(hzdata, H_psicdm, params))

    chi2_vals = np.array(chi2_vals)

    best_idx = int(np.argmin(chi2_vals))
    best_eps = float(eps_vals[best_idx])
    best_chi2 = float(chi2_vals[best_idx])
    red_chi2_best = best_chi2 / dof

    print(
        f"H(z) ΨCDM best-fit: eps0 = {best_eps:.4g}, χ² = {best_chi2:.2f}, "
        f"χ²/dof = {red_chi2_best:.3f}"
    )
    print(f"Δχ² = {best_chi2 - chi2_lcdm:.3f}")

    # ------------------------------------------------------------------
    # 5. Figure: H(z) + model curves
    # ------------------------------------------------------------------
    z_plot = np.linspace(0.0, float(z.max()) * 1.05, 500)
    H_LCDM_plot = H_lcdm(z_plot, lcdm_params)

    psi_params = PsiCDMParams(
        H0=lcdm_params.H0,
        Om0=lcdm_params.Om0,
        eps0=best_eps,
        n=1.0,
    )
    H_PSI_plot = H_psicdm(z_plot, psi_params)

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        z, Hz_obs,
        yerr=sigma_Hz,
        fmt="o",
        alpha=0.8,
        label="H(z) data",
    )
    plt.plot(z_plot, H_LCDM_plot, "-", label=r"ΛCDM")
    plt.plot(z_plot, H_PSI_plot, "--", label=r"ΨCDM (best-fit ε₀)")
    plt.xlabel("z")
    plt.ylabel(r"$H(z)$ [km/s/Mpc]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "hz_psicdm_test.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # 6. χ²(ε₀) scan plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.axhline(chi2_lcdm, ls="--", label=r"ΛCDM")
    plt.plot(eps_vals, chi2_vals, "-o", markersize=3, label=r"ΨCDM")
    plt.xlabel(r"$\varepsilon_0$")
    plt.ylabel(r"$\chi^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "hz_psicdm_chi2_eps_scan.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # 7. Save χ² table
    # ------------------------------------------------------------------
    out_file = tab_dir / "hz_psicdm_chi2.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# H(z) ΛCDM and ΨCDM fits\n")
        f.write(f"N_Hz = {len(z)}\n")
        f.write(f"chi2_LCDM = {chi2_lcdm:.8f}\n")
        f.write(f"chi2_LCDM_reduced = {red_chi2_lcdm:.8f}\n")
        f.write(f"best_eps0 = {best_eps:.8e}\n")
        f.write(f"chi2_Psi_best = {best_chi2:.8f}\n")
        f.write(f"chi2_Psi_best_reduced = {red_chi2_best:.8f}\n")
        f.write(f"Delta_chi2 = {best_chi2 - chi2_lcdm:.8f}\n\n")
        f.write("# eps0   chi2\n")
        for e, c in zip(eps_vals, chi2_vals):
            f.write(f"{e: .6e} {c: .8f}\n")

    print("Saved:", out_file)


if __name__ == "__main__":
    main()
