# analysis/bao_desi_dr2_test.py

"""
DESI DR2 BAO test for ΛCDM and ΨCDM.

Produces:
 - χ² comparison
 - diagnostic plots for DM/rs, DH/rs and DV/rs
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cosmology.data_loaders.desi_loader import load_desi_dr2
from cosmology.background.lcdm import H_lcdm
from cosmology.background.psicdm import H_psicdm

# Speed of light in km/s
c_light = 299792.458


# ----- Derived transverse distance: DM = dL/(1+z) -----

def DM_lcdm(z, params):
    from cosmology.background.lcdm import dL_lcdm
    return dL_lcdm(z, params) / (1.0 + z)

def DM_psicdm(z, params):
    from cosmology.background.psicdm import dL_psicdm
    return dL_psicdm(z, params) / (1.0 + z)


# -------------------------------
# Helper: compute model observables
# -------------------------------
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
            DH_i = c_light / H_i
            vec.append(DH_i / rd)

        elif lbl.startswith("DV"):
            DM_i = DM(zi, params)
            H_i = H(zi, params)
            DH_i = c_light / H_i
            DV_i = (DM_i * DM_i * zi * DH_i) ** (1.0 / 3.0)
            vec.append(DV_i / rd)

        else:
            raise ValueError(f"Unknown DESI BAO label: {lbl}")

    return np.array(vec)


# -------------------------------
# χ² helper
# -------------------------------
def chi2(data_vec, cov, model_vec):
    diff = data_vec - model_vec
    inv = np.linalg.inv(cov)
    return float(diff.T @ inv @ diff)


# -------------------------------
# Main script
# -------------------------------
def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "desi" / "dr2"

    # Load DESI DR2 BAO
    desi = load_desi_dr2(data_dir)
    z = desi["z"]
    labels = desi["labels"]
    data_vec = desi["vec"]
    cov = desi["cov"]

    # Model parameters
    from cosmology.models.lcdm_params import LCDMParams
    from cosmology.models.psicdm_params import PsiCDMParams

    lcdm_params = LCDMParams(H0=70.0, Om0=0.3)
    psicdm_params = PsiCDMParams(
        H0=70.0,
        Om0=0.3,
        eps0=0.0,
        n=1.0,
        rd=lcdm_params.rd,
    )

    rd = lcdm_params.rd

    # LCDM
    vec_lcdm = compute_desi_model(z, labels, lcdm_params, rd, model="lcdm")
    chi2_lcdm = chi2(data_vec, cov, vec_lcdm)

    # PsiCDM
    vec_psicdm = compute_desi_model(z, labels, psicdm_params, rd, model="psicdm")
    chi2_psicdm = chi2(data_vec, cov, vec_psicdm)

    # Print results
    print("=== DESI DR2 BAO test ===")
    print(f"χ²_LCDM  = {chi2_lcdm:.3f}")
    print(f"χ²_PsiCDM = {chi2_psicdm:.3f}")
    print(f"Δχ² (Psi - LCDM) = {chi2_psicdm - chi2_lcdm:.3f}")

    # ----------------------
    # Diagnostic plots
    # ----------------------
    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot DM, DH, DV separately
    DM_z = []
    DM_obs = []
    DM_l = []
    DM_p = []

    DH_z = []
    DH_obs = []
    DH_l = []
    DH_p = []

    DV_z = []
    DV_obs = []
    DV_l = []
    DV_p = []

    for i, lbl in enumerate(labels):
        if lbl.startswith("DM"):
            DM_z.append(z[i])
            DM_obs.append(data_vec[i])
            DM_l.append(vec_lcdm[i])
            DM_p.append(vec_psicdm[i])

        elif lbl.startswith("DH"):
            DH_z.append(z[i])
            DH_obs.append(data_vec[i])
            DH_l.append(vec_lcdm[i])
            DH_p.append(vec_psicdm[i])

        elif lbl.startswith("DV"):
            DV_z.append(z[i])
            DV_obs.append(data_vec[i])
            DV_l.append(vec_lcdm[i])
            DV_p.append(vec_psicdm[i])

    # --- Plot function ---
    def plot_block(zb, obs, lc, pc, name):
        plt.figure(figsize=(8, 5))
        plt.errorbar(zb, obs, fmt="o", label="DESI", color="tab:blue")
        plt.plot(zb, lc, "o-", label="ΛCDM", color="tab:orange")
        plt.plot(zb, pc, "o-", label="ΨCDM", color="tab:green")
        plt.xlabel("z")
        plt.ylabel(f"{name} / r_d")
        plt.title(f"DESI DR2 BAO: {name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        outfile = fig_dir / f"desi_dr2_{name}.png"
        plt.savefig(outfile, dpi=150)
        plt.close()

    if DM_z:
        plot_block(DM_z, DM_obs, DM_l, DM_p, "DM")
    if DH_z:
        plot_block(DH_z, DH_obs, DH_l, DH_p, "DH")
    if DV_z:
        plot_block(DV_z, DV_obs, DV_l, DV_p, "DV")

    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
