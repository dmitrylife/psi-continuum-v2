# analysis/sn_test_psicdm_pplus.py

"""
ΨCDM test using Pantheon+ HF.

Comparison of χ²(ΛCDM) and χ²(ΨCDM), scanned by eps0.
Conclusion:
- results/figures/psi_tests/pantheonplus_hf_chi2_eps_scan.png
- results/tables/psi_tests/chi2_eps_scan.txt
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from cosmology.background.psicdm import dL_psicdm
from cosmology.background.lcdm import dL_lcdm, mu_from_dL
from cosmology.models.lcdm_params import LCDMParams
from cosmology.models.psicdm_params import PsiCDMParams
from cosmology.likelihoods.sn_likelihood import chi2_sn_full_cov


def main():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    fig_dir = results_dir / "figures" / "psi_tests"
    tab_dir = results_dir / "tables" / "psi_tests"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    sn = load_pantheonplus_hf(project_root / "data" / "pantheon_plus")
    z = sn["z"]
    mu_obs = sn["mu"]
    cov = sn["cov"]

    # Basic ΛCDM parameters
    lcdm_params = LCDMParams(H0=70.0, Om0=0.3)
    dL_l = dL_lcdm(z, lcdm_params)
    mu_l = mu_from_dL(dL_l)
    chi2_l = chi2_sn_full_cov(mu_obs, mu_l, cov)
    print(f"ΛCDM baseline: χ² = {chi2_l:.2f}")

    # ΨCDM: scan by eps0 (with fixed H0, Om0, n)
    eps_vals = np.linspace(-0.1, 0.1, 81)  # exemple: [-0.1, 0.1]
    chi2_vals = []

    for eps in eps_vals:
        psicdm_params = PsiCDMParams(
            H0=lcdm_params.H0,
            Om0=lcdm_params.Om0,
            eps0=eps,
            n=1.0,
        )
        dL_p = dL_psicdm(z, psicdm_params)
        mu_p = mu_from_dL(dL_p)
        chi2_p = chi2_sn_full_cov(mu_obs, mu_p, cov)
        chi2_vals.append(chi2_p)

    chi2_vals = np.array(chi2_vals)
    best_idx = int(np.argmin(chi2_vals))
    best_eps = float(eps_vals[best_idx])
    best_chi2 = float(chi2_vals[best_idx])

    print(f"Best ΨCDM: eps0 = {best_eps:.4g}, χ² = {best_chi2:.2f}")
    print(f"Δχ² = χ²(ΨCDM_best) - χ²(ΛCDM) = {best_chi2 - chi2_l:.3f}")

    # График χ²(eps0)
    plt.figure(figsize=(6, 4))
    plt.axhline(chi2_l, ls="--", label=r"ΛCDM")
    plt.plot(eps_vals, chi2_vals, "-o", markersize=3, label=r"ΨCDM")
    plt.xlabel(r"$\varepsilon_0$")
    plt.ylabel(r"$\chi^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pantheonplus_hf_chi2_eps_scan.png", dpi=200)
    plt.close()

    # Table χ²(eps0)
    with open(tab_dir / "chi2_eps_scan.txt", "w", encoding="utf-8") as f:
        f.write("# eps0   chi2\n")
        for e, c in zip(eps_vals, chi2_vals):
            f.write(f"{e: .6e} {c: .8f}\n")
        f.write("\n")
        f.write(f"# chi2_LCDM = {chi2_l:.8f}\n")
        f.write(f"# best_eps0 = {best_eps:.8e}\n")
        f.write(f"# best_chi2 = {best_chi2:.8f}\n")
        f.write(f"# Delta_chi2 = {best_chi2 - chi2_l:.8f}\n")


if __name__ == "__main__":
    main()
