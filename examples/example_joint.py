#!/usr/bin/env python3

"""
Example — Joint χ² computation for ΛCDM using all datasets:

 - Pantheon+ HF (SN)
 - H(z) compilation
 - SDSS DR12 BAO
 - DESI DR2 compressed BAO vector

This example demonstrates the low-level API:
loading data manually and computing chi² with model predictions.
"""

import sys
from pathlib import Path

# Find repository root (folder containing psi_continuum_v2/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

# --- Psi-Continuum imports ---
from psi_continuum_v2.utils import get_data_path
from psi_continuum_v2.cosmology.constants import C_LIGHT

from psi_continuum_v2.cosmology.background.lcdm import (
    H_lcdm,
    dL_lcdm,
    DM_lcdm,
    DH_lcdm,
)

from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams

from psi_continuum_v2.cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from psi_continuum_v2.cosmology.data_loaders.hz_loader import load_hz_compilation
from psi_continuum_v2.cosmology.data_loaders.bao_loader import load_bao_dr12
from psi_continuum_v2.cosmology.data_loaders.desi_loader import load_desi_dr2

from psi_continuum_v2.cosmology.likelihoods.sn_likelihood import chi2_sn_full_cov
from psi_continuum_v2.cosmology.likelihoods.hz_likelihood import chi2_hz
from psi_continuum_v2.cosmology.likelihoods.bao_likelihood import (
    bao_vector_model,
    chi2_bao,
)


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("\n=== JOINT χ² EXAMPLE (ΛCDM) ===\n")

    # ------------------------------------------------------------------
    # 1. PARAMETERS
    # ------------------------------------------------------------------
    lcdm = LCDMParams(H0=70.0, Om0=0.3)

    # ------------------------------------------------------------------
    # 2. LOAD ALL DATASETS THROUGH THE OFFICIAL API
    # ------------------------------------------------------------------
    sn = load_pantheonplus_hf(get_data_path("pantheon_plus"))
    hz = load_hz_compilation(get_data_path("hz"))
    bao = load_bao_dr12(get_data_path("bao"))
    desi = load_desi_dr2(get_data_path("desi", "dr2"))

    # ------------------------------------------------------------------
    # 3. SUPERNOVAE χ²
    # ------------------------------------------------------------------
    dL = dL_lcdm(sn["z"], lcdm)
    mu_th = 5 * np.log10(dL) + 25
    chi2_sn = chi2_sn_full_cov(sn["mu"], mu_th, sn["cov"])

    # ------------------------------------------------------------------
    # 4. H(z) χ² (diagonal)
    # ------------------------------------------------------------------
    chi2_hz_val = chi2_hz(hz, H_lcdm, lcdm)

    # ------------------------------------------------------------------
    # 5. SDSS DR12 BAO (6-dimensional vector)
    # ------------------------------------------------------------------
    z_dr12 = bao["z"]
    DM_dr12 = DM_lcdm(z_dr12, lcdm) / lcdm.rd
    H_dr12 = H_lcdm(z_dr12, lcdm) * lcdm.rd / C_LIGHT  # Hz * rs / c

    model_vec = bao_vector_model(z_dr12, DM_dr12, H_dr12)
    chi2_bao_dr12 = chi2_bao(bao["vec"], bao["cov"], model_vec)

    # ------------------------------------------------------------------
    # 6. DESI DR2 compressed Gaussian vector
    # ------------------------------------------------------------------
    obs = desi["vec"]
    cov = desi["cov"]

    preds = []
    for zi, lab in zip(desi["z"], desi["labels"]):
        if lab.startswith("DM"):
            preds.append(DM_lcdm(zi, lcdm) / lcdm.rd)
        elif lab.startswith("DH"):
            DH_val = DH_lcdm(zi, lcdm) / lcdm.rd
            preds.append(DH_val)
        elif lab.startswith("DV"):
            DMv = DM_lcdm(zi, lcdm)
            DHv = DH_lcdm(zi, lcdm)
            DV = (DMv * DMv * zi * DHv) ** (1/3)
            preds.append(DV / lcdm.rd)
        else:
            raise ValueError(lab)

    preds = np.array(preds)
    diff = obs - preds
    chi2_desi = float(diff.T @ np.linalg.inv(cov) @ diff)

    # ------------------------------------------------------------------
    # 7. SUMMARY
    # ------------------------------------------------------------------
    total = chi2_sn + chi2_hz_val + chi2_bao_dr12 + chi2_desi
    dof = len(sn["z"]) + len(hz["z"]) + len(bao["vec"]) + len(desi["vec"])
    reduced = total / dof

    print("=== RESULTS ===")
    print(f"χ²_SN      = {chi2_sn}")
    print(f"χ²_Hz      = {chi2_hz_val}")
    print(f"χ²_BAO     = {chi2_bao_dr12}")
    print(f"χ²_DESI    = {chi2_desi}")
    print("-" * 29)
    print(f"TOTAL χ²   = {total}")
    print(f"REDUCED χ² = {reduced}")

    print()


# ======================================================================

if __name__ == "__main__":
    main()
