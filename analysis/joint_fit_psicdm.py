# analysis/joint_fit_psicdm.py

"""
Joint χ² test for ΛCDM and ΨCDM using:
    - Pantheon+SH0ES (HF) supernova sample
    - H(z) compilation
    - SDSS DR12 consensus BAO (DM, H)
    - DESI DR2 Gaussian BAO vector (DM/rs, DH/rs, DV/rs)

This script is not a full parameter fit yet.
It simply:
    1) loads all datasets,
    2) evaluates χ² for a chosen ΛCDM parameter set,
    3) evaluates χ² for a chosen ΨCDM parameter set,
    4) prints a detailed χ² breakdown and totals.
"""

from pathlib import Path

import numpy as np

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
from cosmology.likelihoods.joint_likelihood import build_joint_chi2

from cosmology.models.lcdm_params import LCDMParams
from cosmology.models.psicdm_params import PsiCDMParams


def chi2_sn_pantheonplus(
    sn_data: dict,
    model: str,
    lcdm_params: LCDMParams | None = None,
    psicdm_params: PsiCDMParams | None = None,
) -> float:
    """
    Compute χ² for Pantheon+SH0ES HF sample.

    We use the full STAT+SYS covariance matrix:
        χ² = (μ_obs - μ_model)^T C^{-1} (μ_obs - μ_model)

    Args:
        sn_data: dict from load_pantheonplus_hf(...)
                 keys: 'z', 'mu', 'cov', 'N'
        model: "lcdm" or "psicdm"
        lcdm_params: LCDMParams instance (required if model="lcdm")
        psicdm_params: PsiCDMParams instance (required if model="psicdm")

    Returns:
        chi2_sn (float)
    """
    z = sn_data["z"]
    mu_obs = sn_data["mu"]
    cov = sn_data["cov"]

    if model == "lcdm":
        if lcdm_params is None:
            raise ValueError("lcdm_params must be provided for model='lcdm'.")
        dL = dL_lcdm(z, lcdm_params)
    elif model == "psicdm":
        if psicdm_params is None:
            raise ValueError("psicdm_params must be provided for model='psicdm'.")
        dL = dL_psicdm(z, psicdm_params)
    else:
        raise ValueError(f"Unknown model type: {model}")

    mu_model = mu_from_dL(dL)  # distance modulus from d_L [Mpc]

    resid = mu_obs - mu_model

    # Invert covariance once per call (for scans we can precompute C^{-1})
    inv_cov = np.linalg.inv(cov)
    chi2 = float(resid.T @ inv_cov @ resid)

    return chi2


def chi2_hz_dataset(
    hz_data: dict,
    model: str,
    lcdm_params: LCDMParams | None = None,
    psicdm_params: PsiCDMParams | None = None,
) -> float:
    """
    Compute χ² for H(z) compilation:

        χ² = sum_i [(H_obs - H_model)/σ]^2
    """
    z = hz_data["z"]
    Hz_obs = hz_data["Hz"]
    sigma = hz_data["sigma_Hz"]

    if model == "lcdm":
        if lcdm_params is None:
            raise ValueError("lcdm_params must be provided for model='lcdm'.")
        Hz_model = H_lcdm(z, lcdm_params)
    elif model == "psicdm":
        if psicdm_params is None:
            raise ValueError("psicdm_params must be provided for model='psicdm'.")
        Hz_model = H_psicdm(z, psicdm_params)
    else:
        raise ValueError(f"Unknown model type: {model}")

    chi2 = np.sum(((Hz_obs - Hz_model) / sigma) ** 2)
    return float(chi2)


def chi2_bao_dataset(
    bao_data: dict,
    model: str,
    lcdm_params: LCDMParams | None = None,
    psicdm_params: PsiCDMParams | None = None,
) -> float:
    """
    Compute χ² for SDSS DR12 consensus BAO (DM, H).

    Data vector (length 6):
        [DM(z1), H(z1), DM(z2), H(z2), DM(z3), H(z3)]
    """
    z = bao_data["z"]
    data_vec = bao_data["vec"]
    cov = bao_data["cov"]

    if model == "lcdm":
        if lcdm_params is None:
            raise ValueError("lcdm_params must be provided for model='lcdm'.")
        dL = dL_lcdm(z, lcdm_params)
        Hz_model = H_lcdm(z, lcdm_params)
    elif model == "psicdm":
        if psicdm_params is None:
            raise ValueError("psicdm_params must be provided for model='psicdm'.")
        dL = dL_psicdm(z, psicdm_params)
        Hz_model = H_psicdm(z, psicdm_params)
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Transverse comoving distance DM(z) for a flat universe
    DM_model = dL / (1.0 + z)

    model_vec = bao_vector_model(z, DM_model, Hz_model)
    chi2 = chi2_bao(data_vec, cov, model_vec)
    return float(chi2)


def chi2_desi_dataset(
    desi_data: dict,
    model: str,
    lcdm_params: LCDMParams | None = None,
    psicdm_params: PsiCDMParams | None = None,
) -> float:
    """
    Compute χ² for DESI DR2 BAO Gaussian data vector.

    The DESI vector contains values labeled as, e.g.:
        DM_over_rs, DH_over_rs, DV_over_rs

    We compute the corresponding model predictions (DM/rs, DH/rs, DV/rs)
    in the same order and with the same labels, then form

        χ² = (d - m)^T C^{-1} (d - m).
    """

    z = desi_data["z"]
    labels = desi_data["labels"]
    vec = desi_data["vec"]
    cov = desi_data["cov"]

    preds: list[float] = []

    for zi, lab in zip(z, labels):
        # --- ΛCDM ---
        if model == "lcdm":
            if lcdm_params is None:
                raise ValueError("lcdm_params required for model='lcdm'.")

            if lab.startswith("DM"):
                preds.append(DM_lcdm(zi, lcdm_params) / lcdm_params.rd)

            elif lab.startswith("DH"):
                preds.append(DH_lcdm(zi, lcdm_params) / lcdm_params.rd)

            elif lab.startswith("DV"):
                DM_val = DM_lcdm(zi, lcdm_params)
                DH_val = DH_lcdm(zi, lcdm_params)
                # Proper DV definition:
                # DV = [ DM^2 * z * DH ]^{1/3}
                DV_val = (DM_val * DM_val * zi * DH_val) ** (1.0 / 3.0)
                preds.append(DV_val / lcdm_params.rd)

            else:
                raise ValueError(f"Unknown DESI BAO label: {lab}")

        # --- ΨCDM ---
        elif model == "psicdm":
            if psicdm_params is None:
                raise ValueError("psicdm_params required for model='psicdm'.")

            if lab.startswith("DM"):
                preds.append(DM_psicdm(zi, psicdm_params) / psicdm_params.rd)

            elif lab.startswith("DH"):
                preds.append(DH_psicdm(zi, psicdm_params) / psicdm_params.rd)

            elif lab.startswith("DV"):
                DM_val = DM_psicdm(zi, psicdm_params)
                DH_val = DH_psicdm(zi, psicdm_params)
                DV_val = (DM_val * DM_val * zi * DH_val) ** (1.0 / 3.0)
                preds.append(DV_val / psicdm_params.rd)

            else:
                raise ValueError(f"Unknown DESI BAO label: {lab}")

        else:
            raise ValueError(f"Unknown model type: {model}")

    preds = np.array(preds, dtype=float)

    if preds.shape != vec.shape:
        raise ValueError(
            f"DESI model vector shape {preds.shape} does not match data shape {vec.shape}"
        )

    diff = vec - preds
    invcov = np.linalg.inv(cov)

    return float(diff.T @ invcov @ diff)


def main():
    project_root = Path(__file__).resolve().parents[1]

    # ---------- Load datasets ----------
    sn_dir = project_root / "data" / "pantheon_plus"
    hz_dir = project_root / "data" / "hz"
    bao_dir = project_root / "data" / "bao"
    desi_dir = project_root / "data" / "desi" / "dr2"

    sn_data = load_pantheonplus_hf(sn_dir)
    hz_data = load_hz_compilation(hz_dir)
    bao_data = load_bao_dr12(str(bao_dir))
    desi_data = load_desi_dr2(desi_dir)

    N_sn = sn_data["N"]
    N_hz = len(hz_data["z"])
    N_bao = len(bao_data["vec"])  # 6
    N_desi = len(desi_data["z"])

    # ---------- Define cosmological parameter sets ----------

    # Baseline ΛCDM
    lcdm_params = LCDMParams(H0=70.0, Om0=0.3)

    # Example ΨCDM (can be tuned)
    psicdm_params = PsiCDMParams(
        H0=70.0,
        Om0=0.3,
        eps0=0.05,
        n=1.0,
    )

    # ---------- Compute χ² for ΛCDM ----------

    chi2_sn_lcdm = chi2_sn_pantheonplus(
        sn_data, model="lcdm", lcdm_params=lcdm_params
    )
    chi2_hz_lcdm = chi2_hz_dataset(
        hz_data, model="lcdm", lcdm_params=lcdm_params
    )
    chi2_bao_lcdm = chi2_bao_dataset(
        bao_data, model="lcdm", lcdm_params=lcdm_params
    )
    chi2_desi_lcdm = chi2_desi_dataset(
        desi_data, model="lcdm", lcdm_params=lcdm_params
    )

    joint_lcdm = build_joint_chi2(
        chi2_sn=chi2_sn_lcdm,
        dof_sn=N_sn,        # for now we do not subtract parameter count
        chi2_hz=chi2_hz_lcdm,
        dof_hz=N_hz,
        chi2_bao=chi2_bao_lcdm,
        dof_bao=N_bao,
        chi2_desi=chi2_desi_lcdm,
        dof_desi=N_desi,
    )

    # ---------- Compute χ² for ΨCDM ----------

    chi2_sn_psi = chi2_sn_pantheonplus(
        sn_data, model="psicdm", psicdm_params=psicdm_params
    )
    chi2_hz_psi = chi2_hz_dataset(
        hz_data, model="psicdm", psicdm_params=psicdm_params
    )
    chi2_bao_psi = chi2_bao_dataset(
        bao_data, model="psicdm", psicdm_params=psicdm_params
    )
    chi2_desi_psi = chi2_desi_dataset(
        desi_data, model="psicdm", psicdm_params=psicdm_params
    )

    joint_psi = build_joint_chi2(
        chi2_sn=chi2_sn_psi,
        dof_sn=N_sn,
        chi2_hz=chi2_hz_psi,
        dof_hz=N_hz,
        chi2_bao=chi2_bao_psi,
        dof_bao=N_bao,
        chi2_desi=chi2_desi_psi,
        dof_desi=N_desi,
    )

    # ---------- Print report ----------

    print("=== Joint ΛCDM vs ΨCDM test ===")
    print()
    print("ΛCDM parameters:")
    print(f"  H0  = {lcdm_params.H0}")
    print(f"  Om0 = {lcdm_params.Om0}")
    print()
    print("ΨCDM parameters:")
    print(f"  H0   = {psicdm_params.H0}")
    print(f"  Om0  = {psicdm_params.Om0}")
    print(f"  eps0 = {psicdm_params.eps0}")
    print(f"  n    = {psicdm_params.n}")
    print()

    print("--- χ² breakdown (ΛCDM) ---")
    print(f"SN   : chi2 = {chi2_sn_lcdm:.3f}  (N = {N_sn})")
    print(f"H(z) : chi2 = {chi2_hz_lcdm:.3f}  (N = {N_hz})")
    print(f"BAO  : chi2 = {chi2_bao_lcdm:.3f}  (N = {N_bao})")
    print(f"DESI : chi2 = {chi2_desi_lcdm:.3f} (N = {N_desi})")
    print(
        f"Total: chi2 = {joint_lcdm.chi2_total:.3f}, "
        f"dof = {joint_lcdm.dof_total}, "
        f"chi2_red = {joint_lcdm.chi2_reduced:.3f}"
    )
    print()

    print("--- χ² breakdown (ΨCDM) ---")
    print(f"SN   : chi2 = {chi2_sn_psi:.3f}  (N = {N_sn})")
    print(f"H(z) : chi2 = {chi2_hz_psi:.3f}  (N = {N_hz})")
    print(f"BAO  : chi2 = {chi2_bao_psi:.3f}  (N = {N_bao})")
    print(f"DESI : chi2 = {chi2_desi_psi:.3f} (N = {N_desi})")
    print(
        f"Total: chi2 = {joint_psi.chi2_total:.3f}, "
        f"dof = {joint_psi.dof_total}, "
        f"chi2_red = {joint_psi.chi2_reduced:.3f}"
    )
    print()

    print("Δχ² (ΨCDM - ΛCDM):")
    print(f"  SN   : {chi2_sn_psi  - chi2_sn_lcdm :+8.3f}")
    print(f"  H(z) : {chi2_hz_psi  - chi2_hz_lcdm :+8.3f}")
    print(f"  BAO  : {chi2_bao_psi - chi2_bao_lcdm:+8.3f}")
    print(f"  DESI : {chi2_desi_psi - chi2_desi_lcdm:+8.3f}")
    print(f"  Total: {joint_psi.chi2_total - joint_lcdm.chi2_total:+8.3f}")


if __name__ == "__main__":
    main()
