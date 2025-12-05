# analysis/eps_best_joint_test.py

"""
Evaluate ΨCDM at ε0 = 0.031 (best-fit from full epsilon scan).
Datasets:
    - Pantheon+ SH0ES HF SN
    - H(z) compilation
    - SDSS DR12 BAO (DM, H)
    - DESI DR2 Gaussian BAO (DM/rs, DH/rs, DV/rs)

Produces a full χ² breakdown and Δχ² relative to ΛCDM.
"""

from pathlib import Path
import numpy as np

# --- Data loaders ---
from cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from cosmology.data_loaders.hz_loader import load_hz_compilation
from cosmology.data_loaders.bao_loader import load_bao_dr12
from cosmology.data_loaders.desi_loader import load_desi_dr2

# --- Models ---
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

# --- BAO likelihood ---
from cosmology.likelihoods.bao_likelihood import bao_vector_model, chi2_bao as bao_chi2
from cosmology.likelihoods.joint_likelihood import build_joint_chi2

# --- Parameters ---
from cosmology.models.lcdm_params import LCDMParams
from cosmology.models.psicdm_params import PsiCDMParams


# ============================================================
#   χ² components
# ============================================================

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
    z = bao_data["z"]
    data_vec = bao_data["vec"]
    cov = bao_data["cov"]

    if model == "lcdm":
        H_model = H_lcdm(z, lcdm_params)
        dL = dL_lcdm(z, lcdm_params)
    else:
        H_model = H_psicdm(z, psicdm_params)
        dL = dL_psicdm(z, psicdm_params)

    DM = dL / (1 + z)
    vec_model = bao_vector_model(z, DM, H_model)

    return float(bao_chi2(data_vec, cov, vec_model))


def chi2_desi_dataset(desi_data, model, lcdm_params=None, psicdm_params=None):
    """
    DESI DR2 Gaussian BAO likelihood.
    DV uses the correct formula:
        DV = (DM^2 * z * DH)^(1/3)
    """

    z = desi_data["z"]
    labels = desi_data["labels"]
    vec_obs = desi_data["vec"]
    cov = desi_data["cov"]

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


# ============================================================
#   MAIN
# ============================================================

def main():
    project_root = Path(__file__).resolve().parents[1]

    # Load datasets
    sn = load_pantheonplus_hf(project_root / "data" / "pantheon_plus")
    hz = load_hz_compilation(project_root / "data" / "hz")
    bao = load_bao_dr12(project_root / "data" / "bao")
    desi = load_desi_dr2(project_root / "data" / "desi" / "dr2")

    # Counts
    N_sn = sn["N"]
    N_hz = len(hz["z"])          # FIXED: not hz["N"]
    N_bao = len(bao["vec"])
    N_desi = len(desi["z"])      # FIXED: vector length ≠ number of points

    # LCDM parameters
    lcdm = LCDMParams(H0=70.0, Om0=0.3)

    # ΨCDM best-fit
    eps_best = 0.031
    psicdm = PsiCDMParams(H0=70.0, Om0=0.3, eps0=eps_best, n=1.0)

    # ---- LCDM χ² ----
    chi2_sn_lcdm = chi2_sn(sn, "lcdm", lcdm_params=lcdm)
    chi2_hz_lcdm = chi2_hz(hz, "lcdm", lcdm_params=lcdm)
    chi2_bao_lcdm = chi2_bao_dataset(bao, "lcdm", lcdm_params=lcdm)
    chi2_desi_lcdm = chi2_desi_dataset(desi, "lcdm", lcdm_params=lcdm)

    joint_lcdm = build_joint_chi2(
        chi2_sn=chi2_sn_lcdm,  dof_sn=N_sn,
        chi2_hz=chi2_hz_lcdm,  dof_hz=N_hz,
        chi2_bao=chi2_bao_lcdm + chi2_desi_lcdm,
        dof_bao=N_bao + N_desi,
    )

    # ---- ΨCDM χ² ----
    chi2_sn_psi = chi2_sn(sn, "psicdm", psicdm_params=psicdm)
    chi2_hz_psi = chi2_hz(hz, "psicdm", psicdm_params=psicdm)
    chi2_bao_psi = chi2_bao_dataset(bao, "psicdm", psicdm_params=psicdm)
    chi2_desi_psi = chi2_desi_dataset(desi, "psicdm", psicdm_params=psicdm)

    joint_psi = build_joint_chi2(
        chi2_sn=chi2_sn_psi,  dof_sn=N_sn,
        chi2_hz=chi2_hz_psi,  dof_hz=N_hz,
        chi2_bao=chi2_bao_psi + chi2_desi_psi,
        dof_bao=N_bao + N_desi,
    )

    # ---- Print report ----
    print("=== ΨCDM joint test at ε₀ = 0.031 ===\n")

    print("ΛCDM:")
    print(f"  χ²_SN   = {chi2_sn_lcdm:.3f}")
    print(f"  χ²_H(z) = {chi2_hz_lcdm:.3f}")
    print(f"  χ²_BAO  = {chi2_bao_lcdm:.3f}")
    print(f"  χ²_DESI = {chi2_desi_lcdm:.3f}")
    print(f"  TOTAL   = {joint_lcdm.chi2_total:.3f}\n")

    print("ΨCDM (ε₀ = 0.031):")
    print(f"  χ²_SN   = {chi2_sn_psi:.3f}")
    print(f"  χ²_H(z) = {chi2_hz_psi:.3f}")
    print(f"  χ²_BAO  = {chi2_bao_psi:.3f}")
    print(f"  χ²_DESI = {chi2_desi_psi:.3f}")
    print(f"  TOTAL   = {joint_psi.chi2_total:.3f}\n")

    print("Δχ² (ΨCDM − ΛCDM):")
    print(f"  SN      : {chi2_sn_psi - chi2_sn_lcdm:+.3f}")
    print(f"  H(z)    : {chi2_hz_psi - chi2_hz_lcdm:+.3f}")
    print(f"  SDSS BAO: {chi2_bao_psi - chi2_bao_lcdm:+.3f}")
    print(f"  DESI    : {chi2_desi_psi - chi2_desi_lcdm:+.3f}")
    print(f"  TOTAL   : {joint_psi.chi2_total - joint_lcdm.chi2_total:+.3f}")

    # Save results
    out_dir = project_root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eps_best_joint.txt"

    with open(out_file, "w") as f:
        f.write(f"eps0 = {eps_best}\n\n")
        f.write("--- LCDM ---\n")
        f.write(f"chi2_total = {joint_lcdm.chi2_total:.6f}\n\n")
        f.write("--- PsiCDM ---\n")
        f.write(f"chi2_total = {joint_psi.chi2_total:.6f}\n\n")
        f.write("--- Δχ² ---\n")
        f.write(f"Delta = {joint_psi.chi2_total - joint_lcdm.chi2_total:+.6f}\n")

    print("\nSaved:", out_file)


if __name__ == "__main__":
    main()
