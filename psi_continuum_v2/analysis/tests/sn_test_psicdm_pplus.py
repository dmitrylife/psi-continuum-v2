# analysis/tests/sn_test_psicdm_pplus.py

"""
ΨCDM test using the Pantheon+SH0ES HF sample.

Outputs:
- results/figures/sn/pantheonplus_hf_chi2_eps_scan.png
- results/tables/sn/chi2_eps_scan.txt
"""

import numpy as np
import matplotlib.pyplot as plt

from psi_continuum_v2.utils import get_data_path, get_results_path
from psi_continuum_v2.cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from psi_continuum_v2.cosmology.background.psicdm import dL_psicdm
from psi_continuum_v2.cosmology.background.lcdm import dL_lcdm, mu_from_dL
from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams
from psi_continuum_v2.cosmology.likelihoods.sn_likelihood import chi2_sn_full_cov


def main():

    # --------------------------------------------------------------
    # Output directories (relative to active data root)
    # --------------------------------------------------------------
    fig_dir = get_results_path("figures", "sn")
    fig_dir.mkdir(parents=True, exist_ok=True)

    tab_dir = get_results_path("tables", "sn")
    tab_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # 1. Load Pantheon+ HF dataset
    # --------------------------------------------------------------
    sn = load_pantheonplus_hf(get_data_path("pantheon_plus"))
    z = sn["z"]
    mu_obs = sn["mu"]
    cov = sn["cov"]

    # --------------------------------------------------------------
    # 2. ΛCDM baseline
    # --------------------------------------------------------------
    lcdm_params = LCDMParams(H0=70.0, Om0=0.3)

    dL_l = dL_lcdm(z, lcdm_params)
    mu_l = mu_from_dL(dL_l)
    chi2_l = chi2_sn_full_cov(mu_obs, mu_l, cov)

    print(f"ΛCDM baseline: χ² = {chi2_l:.2f}")

    # --------------------------------------------------------------
    # 3. ΨCDM scan over ε₀
    # --------------------------------------------------------------
    eps_vals = np.linspace(-0.1, 0.1, 81)
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
    print(f"Δχ² = {best_chi2 - chi2_l:.3f}")

    # --------------------------------------------------------------
    # 4. Plot χ²(ε₀)
    # --------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.axhline(chi2_l, ls="--", label=r"ΛCDM")
    plt.plot(eps_vals, chi2_vals, "-o", markersize=3, label=r"ΨCDM")
    plt.xlabel(r"$\varepsilon_0$")
    plt.ylabel(r"$\chi^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pantheonplus_hf_chi2_eps_scan.png", dpi=200)
    plt.close()

    # --------------------------------------------------------------
    # 5. Save χ² table
    # --------------------------------------------------------------
    out_file = tab_dir / "chi2_eps_scan.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# eps0   chi2\n")
        for e, c in zip(eps_vals, chi2_vals):
            f.write(f"{e: .6e} {c: .8f}\n")

        f.write("\n")
        f.write(f"# chi2_LCDM = {chi2_l:.8f}\n")
        f.write(f"# best_eps0 = {best_eps:.8e}\n")
        f.write(f"# best_chi2 = {best_chi2:.8f}\n")
        f.write(f"# Delta_chi2 = {best_chi2 - chi2_l:.8f}\n")

    print("Saved:", out_file)


if __name__ == "__main__":
    main()
