# analysis/hz_test_psicdm.py

"""
ΨCDM vs ΛCDM test using H(z) compilation.

Conclusion:
- results/figures/hz_psicdm_test.png   (H(z) + models)
- results/tables/hz/hz_psicdm_chi2.txt    (χ² for ΛCDM and ΨCDM, scanned by eps0)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cosmology.data_loaders.hz_loader import load_hz_compilation
from cosmology.background.lcdm import H_lcdm
from cosmology.background.psicdm import H_psicdm
from cosmology.models.lcdm_params import LCDMParams
from cosmology.models.psicdm_params import PsiCDMParams
from cosmology.likelihoods.hz_likelihood import chi2_hz


def main():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    fig_dir = results_dir / "figures"
    tab_dir = results_dir / "tables" / "hz"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading H(z)
    hzdata = load_hz_compilation(project_root / "data" / "hz")
    z = hzdata["z"]
    Hz_obs = hzdata["Hz"]
    sigma_Hz = hzdata["sigma_Hz"]

    # 2. Basic ΛCDM
    lcdm_params = LCDMParams(H0=70.0, Om0=0.3)
    chi2_lcdm = chi2_hz(hzdata, H_lcdm, lcdm_params)
    dof = len(z) - 2  # rude
    red_chi2_lcdm = chi2_lcdm / dof
    print(f"H(z) ΛCDM: χ² = {chi2_lcdm:.2f}, χ²/dof = {red_chi2_lcdm:.3f}")

    # 3. ΨCDM: scan eps0
    eps_vals = np.linspace(-0.2, 0.2, 81)
    chi2_vals = []

    for eps in eps_vals:
        p = PsiCDMParams(
            H0=lcdm_params.H0,
            Om0=lcdm_params.Om0,
            eps0=eps,
            n=1.0,
        )
        chi2_p = chi2_hz(hzdata, H_psicdm, p)
        chi2_vals.append(chi2_p)

    chi2_vals = np.array(chi2_vals)
    best_idx = int(np.argmin(chi2_vals))
    best_eps = float(eps_vals[best_idx])
    best_chi2 = float(chi2_vals[best_idx])
    red_chi2_psi = best_chi2 / dof

    print(f"H(z) best ΨCDM: eps0 = {best_eps:.4g}, χ² = {best_chi2:.2f}, "
          f"χ²/dof = {red_chi2_psi:.3f}")
    print(f"Δχ² = χ²(ΨCDM_best) - χ²(ΛCDM) = {best_chi2 - chi2_lcdm:.3f}")

    # 4. Figure H(z) + models
    z_plot = np.linspace(0.0, float(z.max()) * 1.05, 500)
    H_l_plot = H_lcdm(z_plot, lcdm_params)
    psi_best_params = PsiCDMParams(
        H0=lcdm_params.H0,
        Om0=lcdm_params.Om0,
        eps0=best_eps,
        n=1.0,
    )
    H_psi_plot = H_psicdm(z_plot, psi_best_params)

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        z,
        Hz_obs,
        yerr=sigma_Hz,
        fmt="o",
        alpha=0.8,
        label="H(z) data",
    )
    plt.plot(z_plot, H_l_plot, "-", label=r"ΛCDM")
    plt.plot(z_plot, H_psi_plot, "--", label=r"ΨCDM (best $\varepsilon_0$)")
    plt.xlabel("z")
    plt.ylabel(r"$H(z)$ [km/s/Mpc]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "hz_psicdm_test.png", dpi=200)
    plt.close()

    # 5. Plot of χ²(eps0)
    plt.figure(figsize=(6, 4))
    plt.axhline(chi2_lcdm, ls="--", label=r"ΛCDM")
    plt.plot(eps_vals, chi2_vals, "-o", markersize=3, label=r"ΨCDM")
    plt.xlabel(r"$\varepsilon_0$")
    plt.ylabel(r"$\chi^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "hz_psicdm_chi2_eps_scan.png", dpi=200)
    plt.close()

    # 6. Table χ²
    with open(tab_dir / "hz_psicdm_chi2.txt", "w", encoding="utf-8") as f:
        f.write("# H(z) ΛCDM and ΨCDM fits\n")
        f.write(f"N_Hz = {len(z)}\n")
        f.write(f"chi2_LCDM = {chi2_lcdm:.8f}\n")
        f.write(f"chi2_LCDM_red = {red_chi2_lcdm:.8f}\n")
        f.write(f"best_eps0 = {best_eps:.8e}\n")
        f.write(f"chi2_Psi_best = {best_chi2:.8f}\n")
        f.write(f"chi2_Psi_best_red = {red_chi2_psi:.8f}\n")
        f.write(f"Delta_chi2 = {best_chi2 - chi2_lcdm:.8f}\n")
        f.write("\n# eps0   chi2\n")
        for e, c in zip(eps_vals, chi2_vals):
            f.write(f"{e: .6e} {c: .8f}\n")


if __name__ == "__main__":
    main()
