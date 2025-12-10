# analysis/scans/scan_eps_psicdm.py

"""
Scan of the ΨCDM epsilon_0 parameter:

We evaluate:
    χ²_SN(ε0)
    χ²_H(z)(ε0)
    χ²_BAO_SDSS(ε0)
    χ²_DESI_DR2(ε0)
    χ²_total(ε0)

and compute Δχ²(ε0) relative to ΛCDM.

Outputs:
    results/tables/scan/eps_scan_psicdm.txt
    results/figures/scan/eps_scan_total.png
"""

import numpy as np
import matplotlib.pyplot as plt

from psi_continuum_v2.utils import get_data_path, get_results_path
from psi_continuum_v2.cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from psi_continuum_v2.cosmology.data_loaders.hz_loader import load_hz_compilation
from psi_continuum_v2.cosmology.data_loaders.bao_loader import load_bao_dr12
from psi_continuum_v2.cosmology.data_loaders.desi_loader import load_desi_dr2

from psi_continuum_v2.cosmology.background.lcdm import (
    H_lcdm, dL_lcdm, mu_from_dL, DM_lcdm, DH_lcdm
)
from psi_continuum_v2.cosmology.background.psicdm import (
    H_psicdm, dL_psicdm, DM_psicdm, DH_psicdm
)

from psi_continuum_v2.cosmology.likelihoods.bao_likelihood import bao_vector_model, chi2_bao
from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams


# ===============================================================
# Likelihood terms
# ===============================================================

def chi2_sn(sn, model, lcdm_params=None, psicdm_params=None):
    """Pantheon+ HF χ² with full covariance."""
    z = sn["z"]
    mu_obs = sn["mu"]
    cov = sn["cov"]

    if model == "lcdm":
        dL = dL_lcdm(z, lcdm_params)
    else:
        dL = dL_psicdm(z, psicdm_params)

    mu_model = mu_from_dL(dL)
    diff = mu_obs - mu_model
    invcov = np.linalg.inv(cov)
    return float(diff.T @ invcov @ diff)


def chi2_hz(hz, model, lcdm_params=None, psicdm_params=None):
    """H(z) Gaussian χ²."""
    z = hz["z"]
    Hobs = hz["Hz"]
    sig = hz["sigma_Hz"]

    if model == "lcdm":
        Hmodel = H_lcdm(z, lcdm_params)
    else:
        Hmodel = H_psicdm(z, psicdm_params)

    return float(np.sum(((Hobs - Hmodel) / sig) ** 2))


def chi2_bao_sdss(bao, model, lcdm_params=None, psicdm_params=None):
    """SDSS DR12 BAO 6-vector χ²."""
    z = bao["z"]
    data_vec = bao["vec"]
    cov = bao["cov"]

    if model == "lcdm":
        dL = dL_lcdm(z, lcdm_params)
        Hz = H_lcdm(z, lcdm_params)
    else:
        dL = dL_psicdm(z, psicdm_params)
        Hz = H_psicdm(z, psicdm_params)

    DM = dL / (1 + z)
    vec_model = bao_vector_model(z, DM, Hz)

    return float(chi2_bao(data_vec, cov, vec_model))


def chi2_desi(desi, model, lcdm_params=None, psicdm_params=None):
    """DESI DR2 Gaussian BAO χ²."""
    z = desi["z"]
    labels = desi["labels"]
    obs = desi["vec"]
    cov = desi["cov"]

    preds = []

    for zi, lab in zip(z, labels):
        if model == "lcdm":
            rd = lcdm_params.rd
            if lab.startswith("DM"):
                preds.append(DM_lcdm(zi, lcdm_params) / rd)
            elif lab.startswith("DH"):
                preds.append(DH_lcdm(zi, lcdm_params) / rd)
            elif lab.startswith("DV"):
                DMv = DM_lcdm(zi, lcdm_params)
                DHv = DH_lcdm(zi, lcdm_params)
                preds.append(((DMv * DMv * zi * DHv)**(1/3)) / rd)
            else:
                raise ValueError(lab)

        else:
            rd = psicdm_params.rd
            if lab.startswith("DM"):
                preds.append(DM_psicdm(zi, psicdm_params) / rd)
            elif lab.startswith("DH"):
                preds.append(DH_psicdm(zi, psicdm_params) / rd)
            elif lab.startswith("DV"):
                DMv = DM_psicdm(zi, psicdm_params)
                DHv = DH_psicdm(zi, psicdm_params)
                preds.append(((DMv * DMv * zi * DHv)**(1/3)) / rd)
            else:
                raise ValueError(lab)

    preds = np.array(preds)
    diff = obs - preds
    invcov = np.linalg.inv(cov)
    return float(diff.T @ invcov @ diff)


# ===============================================================
# Main scan
# ===============================================================

def main():

    # --- Load datasets ---
    sn = load_pantheonplus_hf(get_data_path("pantheon_plus"))
    hz = load_hz_compilation(get_data_path("hz"))
    bao = load_bao_dr12(get_data_path("bao"))
    desi = load_desi_dr2(get_data_path("desi", "dr2"))

    # --- Reference ΛCDM ---
    lcdm = LCDMParams(H0=70.0, Om0=0.3)

    chi2_ref = (
        chi2_sn(sn, "lcdm", lcdm_params=lcdm)
        + chi2_hz(hz, "lcdm", lcdm_params=lcdm)
        + chi2_bao_sdss(bao, "lcdm", lcdm_params=lcdm)
        + chi2_desi(desi, "lcdm", lcdm_params=lcdm)
    )

    print(f"ΛCDM reference χ² = {chi2_ref:.3f}")

    # --- Epsilon grid ---
    eps_grid = np.linspace(-0.10, +0.10, 201)
    chi2_total = np.zeros_like(eps_grid)

    print("Scanning eps0...")

    for i, eps0 in enumerate(eps_grid):
        psi = PsiCDMParams(H0=70.0, Om0=0.3, eps0=float(eps0), n=1.0)

        chi2_total[i] = (
            chi2_sn(sn, "psicdm", psicdm_params=psi)
            + chi2_hz(hz, "psicdm", psicdm_params=psi)
            + chi2_bao_sdss(bao, "psicdm", psicdm_params=psi)
            + chi2_desi(desi, "psicdm", psicdm_params=psi)
        )

    delta = chi2_total - chi2_ref

    idx_best = np.argmin(chi2_total)
    eps_best = eps_grid[idx_best]
    chi2_best = chi2_total[idx_best]

    print("\n=== Best-fit ΨCDM ===")
    print(f"eps0_best = {eps_best:.6f}")
    print(f"χ²_best   = {chi2_best:.3f}")
    print(f"Δχ²_best  = {chi2_best - chi2_ref:+.3f}")

    # ===============================================================
    # Save table
    # ===============================================================
    tab_dir = get_results_path("tables", "scan")
    tab_dir.mkdir(parents=True, exist_ok=True)
    out_table = tab_dir / "eps_scan_psicdm.txt"

    with open(out_table, "w") as f:
        f.write("# eps0    chi2_total    delta_chi2\n")
        for e, ct, dt in zip(eps_grid, chi2_total, delta):
            f.write(f"{e:+.6f}  {ct:12.6f}  {dt:12.6f}\n")

        f.write("\n# Best fit\n")
        f.write(f"eps_best = {eps_best:.8f}\n")
        f.write(f"chi2_best = {chi2_best:.8f}\n")
        f.write(f"delta_best = {chi2_best - chi2_ref:.8f}\n")

    print("Table saved to:", out_table)

    # ===============================================================
    # Save Δχ² plot
    # ===============================================================
    fig_dir = get_results_path("figures", "scan")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_file = fig_dir / "eps_scan_total.png"

    plt.figure(figsize=(7, 5))
    plt.plot(eps_grid, delta, label="Δχ²(ε₀)")
    plt.axhline(0, color="black", lw=1)
    plt.axvline(eps_best, color="red", ls="--", label=f"best ε₀={eps_best:.4f}")
    plt.xlabel(r"$\varepsilon_0$")
    plt.ylabel(r"$\Delta \chi^2$")
    plt.title("ΨCDM epsilon scan (total Δχ²)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_file, dpi=200)
    plt.close()

    print("Plot saved:", fig_file)


if __name__ == "__main__":
    main()
