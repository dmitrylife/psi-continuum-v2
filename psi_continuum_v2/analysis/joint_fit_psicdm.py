# analysis/joint_fit_psicdm.py

"""
Joint χ² comparison of ΛCDM and ΨCDM using:

 - Pantheon+SH0ES (HF) supernova sample
 - H(z) compilation
 - SDSS DR12 BAO consensus (DM/rs, H*rs)
 - DESI DR2 Gaussian BAO vector (DM/rs, DH/rs, DV/rs)

This script:
    1) loads all datasets,
    2) evaluates χ² for fixed ΛCDM and ΨCDM parameter choices,
    3) prints the full χ² breakdown,
    4) saves a summary table into results/tables/joint/.
"""

from pathlib import Path
import numpy as np

# ------------------- DATA LOADERS -------------------

from psi_continuum_v2.cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from psi_continuum_v2.cosmology.data_loaders.hz_loader import load_hz_compilation
from psi_continuum_v2.cosmology.data_loaders.bao_loader import load_bao_dr12
from psi_continuum_v2.cosmology.data_loaders.desi_loader import load_desi_dr2

# ------------------- COSMOLOGY MODELS -------------------

from psi_continuum_v2.cosmology.background.lcdm import (
    H_lcdm, dL_lcdm, mu_from_dL, DM_lcdm, DH_lcdm
)
from psi_continuum_v2.cosmology.background.psicdm import (
    H_psicdm, dL_psicdm, DM_psicdm, DH_psicdm
)

# ------------------- LIKELIHOODS -------------------

from psi_continuum_v2.cosmology.likelihoods.bao_likelihood import bao_vector_model, chi2_bao
from psi_continuum_v2.cosmology.likelihoods.joint_likelihood import build_joint_chi2

# ------------------- PARAMETER CONTAINERS -------------------

from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams


# ======================================================================
#                       χ² CONTRIBUTIONS
# ======================================================================

def chi2_sn_pantheonplus(sn_data, model, lcdm_params=None, psicdm_params=None):
    """Pantheon+ HF χ² with full covariance."""
    z = sn_data["z"]
    mu_obs = sn_data["mu"]
    cov = sn_data["cov"]

    if model == "lcdm":
        dL = dL_lcdm(z, lcdm_params)
    elif model == "psicdm":
        dL = dL_psicdm(z, psicdm_params)
    else:
        raise ValueError(model)

    mu_model = mu_from_dL(dL)
    diff = mu_obs - mu_model
    invcov = np.linalg.inv(cov)
    return float(diff.T @ invcov @ diff)


def chi2_hz_dataset(hz_data, model, lcdm_params=None, psicdm_params=None):
    """H(z) gaussian χ²."""
    z = hz_data["z"]
    Hz_obs = hz_data["Hz"]
    sigma = hz_data["sigma_Hz"]

    if model == "lcdm":
        Hz_model = H_lcdm(z, lcdm_params)
    elif model == "psicdm":
        Hz_model = H_psicdm(z, psicdm_params)
    else:
        raise ValueError(model)

    return float(np.sum(((Hz_obs - Hz_model) / sigma) ** 2))


def chi2_bao_dataset(bao_data, model, lcdm_params=None, psicdm_params=None):
    """SDSS DR12 BAO 6-vector χ²."""
    z = bao_data["z"]
    data_vec = bao_data["vec"]
    cov = bao_data["cov"]

    if model == "lcdm":
        DMm = DM_lcdm(z, lcdm_params)
        Hm = H_lcdm(z, lcdm_params)
    elif model == "psicdm":
        DMm = DM_psicdm(z, psicdm_params)
        Hm = H_psicdm(z, psicdm_params)
    else:
        raise ValueError(model)

    model_vec = bao_vector_model(z, DMm, Hm)
    return float(chi2_bao(data_vec, cov, model_vec))


def chi2_desi_dataset(desi_data, model, lcdm_params=None, psicdm_params=None):
    """DESI DR2 Gaussian BAO χ² over DM/rs, DH/rs, DV/rs."""
    z = desi_data["z"]
    labels = desi_data["labels"]
    obs = desi_data["vec"]
    cov = desi_data["cov"]

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
                DV = (DMv * DMv * zi * DHv) ** (1/3)
                preds.append(DV / rd)
            else:
                raise ValueError(lab)

        elif model == "psicdm":
            rd = psicdm_params.rd
            if lab.startswith("DM"):
                preds.append(DM_psicdm(zi, psicdm_params) / rd)
            elif lab.startswith("DH"):
                preds.append(DH_psicdm(zi, psicdm_params) / rd)
            elif lab.startswith("DV"):
                DMv = DM_psicdm(zi, psicdm_params)
                DHv = DH_psicdm(zi, psicdm_params)
                DV = (DMv * DMv * zi * DHv) ** (1/3)
                preds.append(DV / rd)
            else:
                raise ValueError(lab)

        else:
            raise ValueError(model)

    preds = np.array(preds)
    diff = obs - preds
    invcov = np.linalg.inv(cov)
    return float(diff.T @ invcov @ diff)


# ======================================================================
#                              MAIN
# ======================================================================

def main():
    root = Path(__file__).resolve().parents[2]

    # dirs for output
    tab_dir = root / "results" / "tables" / "joint"
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Data ----------------
    sn_data = load_pantheonplus_hf(root / "data" / "pantheon_plus")
    hz_data = load_hz_compilation(root / "data" / "hz")
    bao_data = load_bao_dr12(str(root / "data" / "bao"))
    desi_data = load_desi_dr2(root / "data" / "desi" / "dr2")

    N_sn = sn_data["N"]
    N_hz = len(hz_data["z"])
    N_bao = len(bao_data["vec"])
    N_desi = len(desi_data["z"])

    # ---------------- Parameters ----------------
    lcdm_params = LCDMParams(H0=70.0, Om0=0.3)
    psicdm_params = PsiCDMParams(H0=70.0, Om0=0.3, eps0=0.05, n=1.0)

    # ---------------- ΛCDM χ² ----------------
    chi2_sn_l = chi2_sn_pantheonplus(sn_data, "lcdm", lcdm_params=lcdm_params)
    chi2_hz_l = chi2_hz_dataset(hz_data, "lcdm", lcdm_params=lcdm_params)
    chi2_bao_l = chi2_bao_dataset(bao_data, "lcdm", lcdm_params=lcdm_params)
    chi2_desi_l = chi2_desi_dataset(desi_data, "lcdm", lcdm_params=lcdm_params)

    joint_l = build_joint_chi2(
        chi2_sn=chi2_sn_l, dof_sn=N_sn,
        chi2_hz=chi2_hz_l, dof_hz=N_hz,
        chi2_bao=chi2_bao_l, dof_bao=N_bao,
        chi2_desi=chi2_desi_l, dof_desi=N_desi,
    )

    # ---------------- ΨCDM χ² ----------------
    chi2_sn_p = chi2_sn_pantheonplus(sn_data, "psicdm", psicdm_params=psicdm_params)
    chi2_hz_p = chi2_hz_dataset(hz_data, "psicdm", psicdm_params=psicdm_params)
    chi2_bao_p = chi2_bao_dataset(bao_data, "psicdm", psicdm_params=psicdm_params)
    chi2_desi_p = chi2_desi_dataset(desi_data, "psicdm", psicdm_params=psicdm_params)

    joint_p = build_joint_chi2(
        chi2_sn=chi2_sn_p, dof_sn=N_sn,
        chi2_hz=chi2_hz_p, dof_hz=N_hz,
        chi2_bao=chi2_bao_p, dof_bao=N_bao,
        chi2_desi=chi2_desi_p, dof_desi=N_desi,
    )

    # ==================================================================
    #                             REPORT
    # ==================================================================
    print("\n=== Joint ΛCDM vs ΨCDM test ===\n")

    print("ΛCDM parameters:")
    print(f"  H0  = {lcdm_params.H0}")
    print(f"  Om0 = {lcdm_params.Om0}\n")

    print("ΨCDM parameters:")
    print(f"  H0   = {psicdm_params.H0}")
    print(f"  Om0  = {psicdm_params.Om0}")
    print(f"  eps0 = {psicdm_params.eps0}")
    print(f"  n    = {psicdm_params.n}\n")

    print("--- χ² breakdown (ΛCDM) ---")
    print(f"SN       : {chi2_sn_l:.3f}")
    print(f"H(z)     : {chi2_hz_l:.3f}")
    print(f"BAO DR12 : {chi2_bao_l:.3f}")
    print(f"DESI DR2 : {chi2_desi_l:.3f}")
    print(f"TOTAL    : {joint_l.chi2_total:.3f},   red = {joint_l.chi2_reduced:.3f}\n")

    print("--- χ² breakdown (ΨCDM) ---")
    print(f"SN       : {chi2_sn_p:.3f}")
    print(f"H(z)     : {chi2_hz_p:.3f}")
    print(f"BAO DR12 : {chi2_bao_p:.3f}")
    print(f"DESI DR2 : {chi2_desi_p:.3f}")
    print(f"TOTAL    : {joint_p.chi2_total:.3f},   red = {joint_p.chi2_reduced:.3f}\n")

    print("--- Δχ² (ΨCDM − ΛCDM) ---")
    print(f"SN       : {chi2_sn_p - chi2_sn_l:+8.3f}")
    print(f"H(z)     : {chi2_hz_p - chi2_hz_l:+8.3f}")
    print(f"BAO DR12 : {chi2_bao_p - chi2_bao_l:+8.3f}")
    print(f"DESI DR2 : {chi2_desi_p - chi2_desi_l:+8.3f}")
    print(f"TOTAL    : {joint_p.chi2_total - joint_l.chi2_total:+8.3f}\n")

    # ==================================================================
    #                       SAVE SUMMARY TABLE
    # ==================================================================
    out = tab_dir / "joint_fit_summary.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Joint χ² comparison of ΛCDM and ΨCDM\n")
        f.write(f"H0 = {lcdm_params.H0}, Om0 = {lcdm_params.Om0}\n")
        f.write(f"eps0 = {psicdm_params.eps0}, n = {psicdm_params.n}\n\n")

        f.write("# χ² contributions (LCDM)\n")
        f.write(f"chi2_SN_LCDM   = {chi2_sn_l:.8f}\n")
        f.write(f"chi2_Hz_LCDM   = {chi2_hz_l:.8f}\n")
        f.write(f"chi2_BAO_LCDM  = {chi2_bao_l:.8f}\n")
        f.write(f"chi2_DESI_LCDM = {chi2_desi_l:.8f}\n")
        f.write(f"chi2_total_LCDM = {joint_l.chi2_total:.8f}\n\n")

        f.write("# χ² contributions (PsiCDM)\n")
        f.write(f"chi2_SN_Psi    = {chi2_sn_p:.8f}\n")
        f.write(f"chi2_Hz_Psi    = {chi2_hz_p:.8f}\n")
        f.write(f"chi2_BAO_Psi   = {chi2_bao_p:.8f}\n")
        f.write(f"chi2_DESI_Psi  = {chi2_desi_p:.8f}\n")
        f.write(f"chi2_total_Psi = {joint_p.chi2_total:.8f}\n\n")

        f.write("# Δχ²\n")
        f.write(f"Delta_SN   = {chi2_sn_p - chi2_sn_l:+.8f}\n")
        f.write(f"Delta_Hz   = {chi2_hz_p - chi2_hz_l:+.8f}\n")
        f.write(f"Delta_BAO  = {chi2_bao_p - chi2_bao_l:+.8f}\n")
        f.write(f"Delta_DESI = {chi2_desi_p - chi2_desi_l:+.8f}\n")
        f.write(f"Delta_total = {joint_p.chi2_total - joint_l.chi2_total:+.8f}\n")

    print(f"Saved summary table: {out}")


if __name__ == "__main__":
    main()
