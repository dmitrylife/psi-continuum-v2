# analysis/tests/eps_best_joint_test.py

"""
Evaluate the ΨCDM best-fit obtained from the full epsilon scan.

Datasets included:
    • Pantheon+SH0ES HF supernovae
    • H(z) compilation
    • SDSS DR12 BAO (DM, H)
    • DESI DR2 Gaussian BAO vector (DM/rs, DH/rs, DV/rs)

This script:
    1) loads all datasets
    2) evaluates χ² for ΛCDM and ΨCDM at the chosen best-fit epsilon
    3) prints the full χ² breakdown
    4) saves a summary table

Nothing is fitted here — this is a fixed-point evaluation.
"""

import numpy as np

from psi_continuum_v2.utils import get_data_path, get_results_path

# -------------------- Data loaders --------------------

from psi_continuum_v2.cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from psi_continuum_v2.cosmology.data_loaders.hz_loader import load_hz_compilation
from psi_continuum_v2.cosmology.data_loaders.bao_loader import load_bao_dr12
from psi_continuum_v2.cosmology.data_loaders.desi_loader import load_desi_dr2

# -------------------- Cosmology models --------------------

from psi_continuum_v2.cosmology.background.lcdm import (
    H_lcdm, dL_lcdm, mu_from_dL, DM_lcdm, DH_lcdm
)
from psi_continuum_v2.cosmology.background.psicdm import (
    H_psicdm, dL_psicdm, DM_psicdm, DH_psicdm
)

# -------------------- Likelihood helpers --------------------

from psi_continuum_v2.cosmology.likelihoods.bao_likelihood import (
    bao_vector_model,
    chi2_bao as bao_chi2,
)
from psi_continuum_v2.cosmology.likelihoods.joint_likelihood import build_joint_chi2

# -------------------- Parameter containers --------------------

from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams


# ===============================================================
# χ² COMPONENTS
# ===============================================================

def chi2_sn(sn, model, lcdm_params=None, psicdm_params=None):
    """Pantheon+ HF full covariance χ²."""
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
    """SDSS DR12 consensus BAO 6-vector χ²."""
    z = bao["z"]
    vec_obs = bao["vec"]
    cov = bao["cov"]

    if model == "lcdm":
        dL = dL_lcdm(z, lcdm_params)
        H_model = H_lcdm(z, lcdm_params)
    else:
        dL = dL_psicdm(z, psicdm_params)
        H_model = H_psicdm(z, psicdm_params)

    DM = dL / (1 + z)
    vec_model = bao_vector_model(z, DM, H_model)

    return float(bao_chi2(vec_obs, cov, vec_model))


def chi2_desi(desi, model, lcdm_params=None, psicdm_params=None):
    """DESI DR2 Gaussian BAO χ² (DM/rs, DH/rs, DV/rs)."""

    z = desi["z"]
    labels = desi["labels"]
    vec_obs = desi["vec"]
    cov = desi["cov"]

    preds = []

    for zi, lab in zip(z, labels):
        if model == "lcdm":
            DMv = DM_lcdm(zi, lcdm_params)
            DHv = DH_lcdm(zi, lcdm_params)
            rd = lcdm_params.rd
        else:
            DMv = DM_psicdm(zi, psicdm_params)
            DHv = DH_psicdm(zi, psicdm_params)
            rd = psicdm_params.rd

        if lab.startswith("DM"):
            preds.append(DMv / rd)
        elif lab.startswith("DH"):
            preds.append(DHv / rd)
        elif lab.startswith("DV"):
            DVv = (DMv * DMv * zi * DHv) ** (1/3)
            preds.append(DVv / rd)
        else:
            raise ValueError(f"Unknown DESI label: {lab}")

    preds = np.array(preds)
    diff = vec_obs - preds
    invcov = np.linalg.inv(cov)

    return float(diff.T @ invcov @ diff)


# ===============================================================
# MAIN
# ===============================================================

def main():

    # ---------------- Load datasets ----------------
    sn = load_pantheonplus_hf(get_data_path("pantheon_plus"))
    hz = load_hz_compilation(get_data_path("hz"))
    bao = load_bao_dr12(get_data_path("bao"))
    desi = load_desi_dr2(get_data_path("desi", "dr2"))

    # N-points
    N_sn = sn["N"]
    N_hz = len(hz["z"])
    N_bao = len(bao["vec"])
    N_desi = len(desi["z"])

    # ---------------- Models ----------------
    lcdm = LCDMParams(H0=70.0, Om0=0.3)

    # Best-fit epsilon from scan
    eps_best = 0.031
    psicdm = PsiCDMParams(H0=70.0, Om0=0.3, eps0=eps_best, n=1.0)

    # ---------------- LCDM χ² ----------------
    chi2_lcdm_sn = chi2_sn(sn, "lcdm", lcdm_params=lcdm)
    chi2_lcdm_hz = chi2_hz(hz, "lcdm", lcdm_params=lcdm)
    chi2_lcdm_bao = chi2_bao_sdss(bao, "lcdm", lcdm_params=lcdm)
    chi2_lcdm_desi = chi2_desi(desi, "lcdm", lcdm_params=lcdm)

    joint_lcdm = build_joint_chi2(
        chi2_sn=chi2_lcdm_sn,  dof_sn=N_sn,
        chi2_hz=chi2_lcdm_hz,  dof_hz=N_hz,
        chi2_bao=chi2_lcdm_bao + chi2_lcdm_desi,
        dof_bao=N_bao + N_desi,
    )

    # ---------------- ΨCDM χ² ----------------
    chi2_psi_sn = chi2_sn(sn, "psicdm", psicdm_params=psicdm)
    chi2_psi_hz = chi2_hz(hz, "psicdm", psicdm_params=psicdm)
    chi2_psi_bao = chi2_bao_sdss(bao, "psicdm", psicdm_params=psicdm)
    chi2_psi_desi = chi2_desi(desi, "psicdm", psicdm_params=psicdm)

    joint_psi = build_joint_chi2(
        chi2_sn=chi2_psi_sn,  dof_sn=N_sn,
        chi2_hz=chi2_psi_hz,  dof_hz=N_hz,
        chi2_bao=chi2_psi_bao + chi2_psi_desi,
        dof_bao=N_bao + N_desi,
    )

    # ===============================================================
    # REPORT
    # ===============================================================

    print("\n=== Joint ΨCDM test at eps0 = 0.031 ===\n")

    print("--- ΛCDM ---")
    print(f"SN        : {chi2_lcdm_sn:.3f}")
    print(f"H(z)      : {chi2_lcdm_hz:.3f}")
    print(f"BAO SDSS  : {chi2_lcdm_bao:.3f}")
    print(f"DESI DR2  : {chi2_lcdm_desi:.3f}")
    print(f"TOTAL     : {joint_lcdm.chi2_total:.3f}\n")

    print("--- ΨCDM ---")
    print(f"SN        : {chi2_psi_sn:.3f}")
    print(f"H(z)      : {chi2_psi_hz:.3f}")
    print(f"BAO SDSS  : {chi2_psi_bao:.3f}")
    print(f"DESI DR2  : {chi2_psi_desi:.3f}")
    print(f"TOTAL     : {joint_psi.chi2_total:.3f}\n")

    print("--- Δχ² (ΨCDM − ΛCDM) ---")
    print(f"SN        : {chi2_psi_sn - chi2_lcdm_sn:+.3f}")
    print(f"H(z)      : {chi2_psi_hz - chi2_lcdm_hz:+.3f}")
    print(f"SDSS BAO  : {chi2_psi_bao - chi2_lcdm_bao:+.3f}")
    print(f"DESI      : {chi2_psi_desi - chi2_lcdm_desi:+.3f}")
    print(f"TOTAL     : {joint_psi.chi2_total - joint_lcdm.chi2_total:+.3f}\n")

    # ===============================================================
    # SAVE TABLE
    # ===============================================================

    out_dir = get_results_path("tables", "joint")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eps_best_joint.txt"

    with open(out_file, "w") as f:
        f.write("# Joint χ² comparison at eps_best = 0.031\n\n")

        f.write("[LCDM]\n")
        f.write(f"chi2_total = {joint_lcdm.chi2_total:.8f}\n\n")

        f.write("[PsiCDM]\n")
        f.write(f"chi2_total = {joint_psi.chi2_total:.8f}\n\n")

        f.write("[Delta]\n")
        f.write(f"delta_chi2 = {joint_psi.chi2_total - joint_lcdm.chi2_total:+.8f}\n")

    print("Saved:", out_file)


if __name__ == "__main__":
    main()
