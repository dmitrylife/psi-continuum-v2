# analysis/tests/bao_desi_dr2_test.py

"""
DESI DR2 BAO test for ΛCDM and ΨCDM.

Outputs:
 - results/tables/bao/desi_dr2_chi2.txt
 - results/figures/bao/desi_dr2_DM.png
 - results/figures/bao/desi_dr2_DH.png
 - results/figures/bao/desi_dr2_DV.png
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from psi_continuum_v2.utils import get_data_path, get_results_path

from psi_continuum_v2.cosmology.data_loaders.desi_loader import load_desi_dr2
from psi_continuum_v2.cosmology.background.lcdm import H_lcdm, dL_lcdm
from psi_continuum_v2.cosmology.background.psicdm import H_psicdm, dL_psicdm
from psi_continuum_v2.cosmology.constants import C_LIGHT


# -------------------------------------------------------------------
# DM distances
# -------------------------------------------------------------------
def DM_lcdm(z, params):
    return dL_lcdm(z, params) / (1.0 + z)

def DM_psicdm(z, params):
    return dL_psicdm(z, params) / (1.0 + z)


# -------------------------------------------------------------------
# Build DESI BAO observable vector
# -------------------------------------------------------------------
def compute_desi_model(z, labels, params, rd, model="lcdm"):
    """
    Compute DESI BAO observables (DM/rs, DH/rs, DV/rs) for a given model.
    """

    if model == "lcdm":
        DM = DM_lcdm
        H = H_lcdm
    elif model == "psicdm":
        DM = DM_psicdm
        H = H_psicdm
    else:
        raise ValueError(f"Unknown model: {model}")

    vec = []

    for i, lbl in enumerate(labels):
        zi = z[i]

        if lbl.startswith("DM"):
            DM_i = DM(zi, params)
            vec.append(DM_i / rd)

        elif lbl.startswith("DH"):
            H_i = H(zi, params)
            DH_i = C_LIGHT / H_i
            vec.append(DH_i / rd)

        elif lbl.startswith("DV"):
            DM_i = DM(zi, params)
            H_i = H(zi, params)
            DH_i = C_LIGHT / H_i
            DV_i = (DM_i * DM_i * zi * DH_i) ** (1.0 / 3.0)
            vec.append(DV_i / rd)

        else:
            raise ValueError(f"Unknown DESI BAO label: {lbl}")

    return np.array(vec)


# -------------------------------------------------------------------
# χ²
# -------------------------------------------------------------------
def chi2(data_vec, cov, model_vec):
    diff = data_vec - model_vec
    inv = np.linalg.inv(cov)
    return float(diff.T @ inv @ diff)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    # --------------------------
    # Directories
    # --------------------------

    # Input data directory
    data_dir = get_data_path("desi", "dr2", must_exist=True)

    # Output directories (next to data/)
    fig_dir = get_results_path("figures", "bao")
    fig_dir.mkdir(parents=True, exist_ok=True)

    tab_dir = get_results_path("tables", "bao")
    tab_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # Load DESI data
    # --------------------------
    desi = load_desi_dr2(data_dir)
    z = desi["z"]
    labels = desi["labels"]
    data_vec = desi["vec"]
    cov = desi["cov"]

    # --------------------------
    # Parameters
    # --------------------------
    from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
    from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams

    lcdm_params = LCDMParams(H0=70.0, Om0=0.3)

    psicdm_params = PsiCDMParams(
        H0=70.0,
        Om0=0.3,
        eps0=0.0,
        n=1.0,
        rd=lcdm_params.rd,
    )

    rd = lcdm_params.rd

    # --------------------------
    # χ² LCDM
    # --------------------------
    vec_lcdm = compute_desi_model(z, labels, lcdm_params, rd, model="lcdm")
    chi2_lcdm = chi2(data_vec, cov, vec_lcdm)

    # --------------------------
    # χ² ΨCDM
    # --------------------------
    vec_psicdm = compute_desi_model(z, labels, psicdm_params, rd, model="psicdm")
    chi2_psicdm = chi2(data_vec, cov, vec_psicdm)

    # --------------------------
    # Print result
    # --------------------------
    print("=== DESI DR2 BAO test ===")
    print(f"χ²_LCDM   = {chi2_lcdm:.3f}")
    print(f"χ²_ΨCDM    = {chi2_psicdm:.3f}")
    print(f"Δχ² (ΨCDM - LCDM) = {chi2_psicdm - chi2_lcdm:.3f}")

    # --------------------------
    # Save χ² table
    # --------------------------
    with open(tab_dir / "desi_dr2_chi2.txt", "w", encoding="utf-8") as f:
        f.write("# DESI DR2 BAO χ² comparison\n")
        f.write(f"chi2_LCDM = {chi2_lcdm:.8f}\n")
        f.write(f"chi2_PsiCDM = {chi2_psicdm:.8f}\n")
        f.write(f"Delta_chi2 = {chi2_psicdm - chi2_lcdm:.8f}\n")

    # --------------------------
    # Split observables by type
    # --------------------------
    DM_z = []; DM_obs = []; DM_l = []; DM_p = []
    DH_z = []; DH_obs = []; DH_l = []; DH_p = []
    DV_z = []; DV_obs = []; DV_l = []; DV_p = []

    for i, lbl in enumerate(labels):
        if lbl.startswith("DM"):
            DM_z.append(z[i]); DM_obs.append(data_vec[i])
            DM_l.append(vec_lcdm[i]); DM_p.append(vec_psicdm[i])

        elif lbl.startswith("DH"):
            DH_z.append(z[i]); DH_obs.append(data_vec[i])
            DH_l.append(vec_lcdm[i]); DH_p.append(vec_psicdm[i])

        elif lbl.startswith("DV"):
            DV_z.append(z[i]); DV_obs.append(data_vec[i])
            DV_l.append(vec_lcdm[i]); DV_p.append(vec_psicdm[i])

    # --------------------------
    # Plotting helper
    # --------------------------
    def plot_block(zb, obs, lc, pc, name):
        plt.figure(figsize=(8, 5))
        plt.errorbar(zb, obs, fmt="o", label="DESI")
        plt.plot(zb, lc, "o-", label="ΛCDM")
        plt.plot(zb, pc, "o-", label="ΨCDM")
        plt.xlabel("z")
        plt.ylabel(f"{name} / r_d")
        plt.title(f"DESI DR2 BAO: {name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fig_dir / f"desi_dr2_{name}.png", dpi=200)
        plt.close()

    if DM_z: plot_block(DM_z, DM_obs, DM_l, DM_p, "DM")
    if DH_z: plot_block(DH_z, DH_obs, DH_l, DH_p, "DH")
    if DV_z: plot_block(DV_z, DV_obs, DV_l, DV_p, "DV")

    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
