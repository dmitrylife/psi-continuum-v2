# analysis/scan_eps_psicdm.py

"""
Scan of epsilon_0 parameter in ΨCDM:
    eps0 ∈ [eps_min, eps_max]

We compute:
    χ²_SN(ε₀)
    χ²_H(z)(ε₀)
    χ²_BAO_SDSS(ε₀)
    χ²_DESI(ε₀)
    χ²_total(ε₀)
and Δχ² relative to ΛCDM.

This produces the first full combined constraint on ΨCDM
using SN + H(z) + SDSS BAO + DESI DR2 BAO.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from cosmology.data_loaders.hz_loader import load_hz_compilation
from cosmology.data_loaders.bao_loader import load_bao_dr12
from cosmology.data_loaders.desi_loader import load_desi_dr2

from cosmology.background.lcdm import (
    H_lcdm,
    dL_lcdm,
    mu_from_dL,
    DM_lcdm,
    DH_lcdm,
)
from cosmology.background.psicdm import (
    H_psicdm,
    dL_psicdm,
    DM_psicdm,
    DH_psicdm,
)

from cosmology.likelihoods.bao_likelihood import bao_vector_model, chi2_bao
from cosmology.models.lcdm_params import LCDMParams
from cosmology.models.psicdm_params import PsiCDMParams



# ----------------------------------------------------------------------
#                           LIKELIHOOD PARTS
# ----------------------------------------------------------------------

def chi2_sn(sn_data, model, lcdm_params=None, psicdm_params=None):
    z = sn_data["z"]
    mu_obs = sn_data["mu"]
    cov = sn_data["cov"]

    if model == "lcdm":
        dL = dL_lcdm(z, lcdm_params)
    else:
        dL = dL_psicdm(z, psicdm_params)

    mu_model = mu_from_dL(dL)
    resid = mu_obs - mu_model
    inv_cov = np.linalg.inv(cov)

    return float(resid.T @ inv_cov @ resid)


def chi2_hz(hz_data, model, lcdm_params=None, psicdm_params=None):
    z = hz_data["z"]
    Hz_obs = hz_data["Hz"]
    sigma = hz_data["sigma_Hz"]

    if model == "lcdm":
        Hz_model = H_lcdm(z, lcdm_params)
    else:
        Hz_model = H_psicdm(z, psicdm_params)

    return float(np.sum(((Hz_obs - Hz_model) / sigma) ** 2))


def chi2_bao_dataset(bao_data, model, lcdm_params=None, psicdm_params=None):
    """
    SDSS DR12 consensus BAO:
    data vector = [DM1, H1, DM2, H2, DM3, H3]
    """
    z = bao_data["z"]
    data_vec = bao_data["vec"]
    cov = bao_data["cov"]

    if model == "lcdm":
        dL = dL_lcdm(z, lcdm_params)
        Hz_model = H_lcdm(z, lcdm_params)
    else:
        dL = dL_psicdm(z, psicdm_params)
        Hz_model = H_psicdm(z, psicdm_params)

    DM = dL / (1 + z)
    vec_model = bao_vector_model(z, DM, Hz_model)

    from cosmology.likelihoods.bao_likelihood import chi2_bao as bao_chi2
    return float(bao_chi2(data_vec, cov, vec_model))



def chi2_desi_dataset(desi_data, model, lcdm_params=None, psicdm_params=None):
    """
    DESI DR2 Gaussian BAO likelihood.
    Labels are of form: DM_over_rs, DH_over_rs, DV_over_rs.
    """

    z = desi_data["z"]
    labels = desi_data["labels"]
    vec_obs = desi_data["vec"]
    cov = desi_data["cov"]

    preds = []

    for zi, lab in zip(z, labels):

        # -------- LCDM --------
        if model == "lcdm":
            if lab.startswith("DM"):
                preds.append(DM_lcdm(zi, lcdm_params) / lcdm_params.rd)

            elif lab.startswith("DH"):
                preds.append(DH_lcdm(zi, lcdm_params) / lcdm_params.rd)

            elif lab.startswith("DV"):
                DMv = DM_lcdm(zi, lcdm_params)
                DHv = DH_lcdm(zi, lcdm_params)
                DVv = (DMv * DMv * zi * DHv) ** (1.0 / 3.0)
                preds.append(DVv / lcdm_params.rd)

            else:
                raise ValueError(f"Unknown DESI BAO label {lab}")

        # -------- ΨCDM --------
        else:
            if lab.startswith("DM"):
                preds.append(DM_psicdm(zi, psicdm_params) / psicdm_params.rd)

            elif lab.startswith("DH"):
                preds.append(DH_psicdm(zi, psicdm_params) / psicdm_params.rd)

            elif lab.startswith("DV"):
                DMv = DM_psicdm(zi, psicdm_params)
                DHv = DH_psicdm(zi, psicdm_params)
                DVv = (DMv * DMv * zi * DHv) ** (1.0 / 3.0)
                preds.append(DVv / psicdm_params.rd)

            else:
                raise ValueError(f"Unknown DESI BAO label {lab}")

    preds = np.array(preds)
    diff = vec_obs - preds
    invcov = np.linalg.inv(cov)

    return float(diff.T @ invcov @ diff)



# ----------------------------------------------------------------------
#                               MAIN SCAN
# ----------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parents[1]

    # Load data
    sn_data = load_pantheonplus_hf(project_root / "data" / "pantheon_plus")
    hz_data = load_hz_compilation(project_root / "data" / "hz")
    bao_data = load_bao_dr12(project_root / "data" / "bao")
    desi_data = load_desi_dr2(project_root / "data" / "desi" / "dr2")

    N_sn = sn_data["N"]
    N_hz = len(hz_data["z"])
    N_bao = len(bao_data["vec"])
    N_desi = len(desi_data["z"])

    # ΛCDM reference
    lcdm_params = LCDMParams(H0=70, Om0=0.3)

    chi2_sn_ref = chi2_sn(sn_data, "lcdm", lcdm_params=lcdm_params)
    chi2_hz_ref = chi2_hz(hz_data, "lcdm", lcdm_params=lcdm_params)
    chi2_bao_ref = chi2_bao_dataset(bao_data, "lcdm", lcdm_params=lcdm_params)
    chi2_desi_ref = chi2_desi_dataset(desi_data, "lcdm", lcdm_params=lcdm_params)

    chi2_ref = chi2_sn_ref + chi2_hz_ref + chi2_bao_ref + chi2_desi_ref

    print("ΛCDM reference χ² =", chi2_ref)

    # Scan eps
    eps_grid = np.linspace(-0.10, +0.10, 201)

    chi2_sn_arr = np.zeros_like(eps_grid)
    chi2_hz_arr = np.zeros_like(eps_grid)
    chi2_bao_arr = np.zeros_like(eps_grid)
    chi2_desi_arr = np.zeros_like(eps_grid)
    chi2_total_arr = np.zeros_like(eps_grid)

    print("Scanning eps0...")

    for i, eps0 in enumerate(eps_grid):
        psi_params = PsiCDMParams(
            H0=70.0,
            Om0=0.3,
            eps0=float(eps0),
            n=1.0,
        )

        chi2_sn_arr[i] = chi2_sn(sn_data, "psicdm", psicdm_params=psi_params)
        chi2_hz_arr[i] = chi2_hz(hz_data, "psicdm", psicdm_params=psi_params)
        chi2_bao_arr[i] = chi2_bao_dataset(bao_data, "psicdm", psicdm_params=psi_params)
        chi2_desi_arr[i] = chi2_desi_dataset(desi_data, "psicdm", psicdm_params=psi_params)

        chi2_total_arr[i] = (
            chi2_sn_arr[i]
            + chi2_hz_arr[i]
            + chi2_bao_arr[i]
            + chi2_desi_arr[i]
        )

    # Δχ²
    delta_total = chi2_total_arr - chi2_ref

    # Best epsilon
    idx_best = np.argmin(chi2_total_arr)
    eps_best = eps_grid[idx_best]
    chi2_best = chi2_total_arr[idx_best]

    print()
    print("=== ΨCDM best-fit ===")
    print(f"eps0_best = {eps_best:.6f}")
    print(f"χ²_best   = {chi2_best:.3f}")
    print(f"Δχ²_best  = {chi2_best - chi2_ref:+.3f}")

    # Save table
    out_dir = project_root / "results" / "tables"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / "eps_scan_psicdm.txt"

    with open(out_file, "w") as f:
        f.write("# eps0    chi2_total    delta_total\n")
        for e, ct, dt in zip(eps_grid, chi2_total_arr, delta_total):
            f.write(f"{e:+.6f}  {ct:12.6f}  {dt:12.6f}\n")

    print("Table saved to:", out_file)

    # Figures
    fig_dir = project_root / "results" / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(7, 5))
    plt.plot(eps_grid, delta_total, label="Δχ² total")
    plt.axhline(0, color="black", lw=1)
    plt.axvline(eps_best, color="red", ls="--")
    plt.xlabel("eps0")
    plt.ylabel("Δχ²")
    plt.title("ΨCDM epsilon scan — total Δχ²")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "eps_scan_total.png", dpi=200)
    plt.close()

    print("Plot saved:", fig_dir / "eps_scan_total.png")


if __name__ == "__main__":
    main()
