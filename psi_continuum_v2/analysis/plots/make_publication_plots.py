# analysis/plots/make_publication_plots.py

"""
Publication-grade plot generator for Psi-Continuum v2.

This script does NOT participate in the basic analysis pipeline and
does NOT modify the standard results structure.

It only reads:
    - data/* (Pantheon+, H(z), BAO DR12, DESI DR2)
    - results/tables/scan/eps_scan_psicdm.txt  (for joint χ² scan)

and produces a clean publication-oriented layout:

results/figures/publication/
    main_figures/
        fig1_Ez_comparison.png
        fig2_SN_Hubble.png
        fig3_BAO_DR12_multipanel.png
        fig4_BAO_DESI_multipanel.png
        fig5_BAO_fits_LCDM_vs_PsiCDM.png
        fig6_Hz_dataset.png
        fig7_joint_chi2_eps.png
        fig8_delta_chi2_contributions.png

    appendix/
        sn_histograms.png
        hz_quality_checks.png
        model_lcdm_Ez_Hz_dL.png
        model_psicdm_Ez_scan.png
        hz_only_chi2_scan.png
        bao_dr12_raw.png
        bao_desi_raw.png

    talk_figures/
        talk_summary.png
        talk_bao.png
        talk_sn.png

    summary/
        poster_summary.png

The idea is that `main_figures/` can be copied directly into the LaTeX project.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from psi_continuum_v2.utils import get_data_path

# --- Cosmology imports ---
from psi_continuum_v2.cosmology.background.lcdm import (
    E_lcdm,
    H_lcdm,
    dL_lcdm,
    DM_lcdm,
    DH_lcdm,
)
from psi_continuum_v2.cosmology.background.psicdm import (
    E_psicdm,
    H_psicdm,
    dL_psicdm,
    DM_psicdm,
    DH_psicdm,
)

from psi_continuum_v2.cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from psi_continuum_v2.cosmology.data_loaders.hz_loader import load_hz_compilation
from psi_continuum_v2.cosmology.data_loaders.bao_loader import load_bao_dr12
from psi_continuum_v2.cosmology.data_loaders.desi_loader import load_desi_dr2

from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams
from psi_continuum_v2.cosmology.likelihoods.hz_likelihood import chi2_hz


# ======================================================================
# Utility helpers
# ======================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_eps_scan_table() -> Path:
    """
    Automatically locate epsilon scan table somewhere under results/tables.

    Preferred paths (in order):
        results/tables/scan/eps_scan_psicdm.txt
        results/tables/eps_scan_psicdm.txt
        any *eps*scan*.txt under results/tables/
    """
    root = Path.cwd()

    base = root / "results" / "tables"

    preferred = [
        base / "scan" / "eps_scan_psicdm.txt",
        base / "eps_scan_psicdm.txt",
    ]

    for p in preferred:
        if p.exists():
            print(f"[make_publication_plots] Using eps-scan table: {p}")
            return p

    # Fallback: recursive search
    candidates = []
    if base.exists():
        for p in base.rglob("*.txt"):
            name = p.name.lower()
            if "eps" in name and "scan" in name:
                candidates.append(p)

    if candidates:
        # deterministic order
        candidates = sorted(candidates)
        print(f"[make_publication_plots] Using eps-scan table: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError("No eps-scan table found under results/tables/")


def load_eps_table(path: Path):
    """
    Robust loader for eps-scan table.

    Expected "rich" format (from scan_eps_psicdm.py):

        # eps0    chi2_total    delta_total

    but we also allow:
    - arbitrary commented lines (# ...)
    - arbitrary extra columns
    - fallback to simple 2-column format: eps, chi2
    """
    eps, chi2 = [], []

    if not path.exists():
        raise FileNotFoundError(f"eps-scan table not found: {path}")

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                e = float(parts[0])

                # If line has >= 3 columns and second is total χ²,
                # use the second column as χ²_total.
                # Otherwise fall back to second column.
                if len(parts) >= 3:
                    c = float(parts[1])
                else:
                    c = float(parts[1])
            except ValueError:
                # skip malformed rows
                continue

            eps.append(e)
            chi2.append(c)

    return np.array(eps, dtype=float), np.array(chi2, dtype=float)


# ======================================================================
# MAIN FIGURES (for the article)
# ======================================================================

def fig1_Ez_comparison(figdir: Path) -> None:
    """
    Fig. 1 – E(z) comparison: ΛCDM vs ΨCDM (ε0 ≈ best-fit).
    """
    z = np.linspace(0.0, 3.0, 400)

    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    psicdm = PsiCDMParams(H0=70.0, Om0=0.3, eps0=0.031, n=1.0)

    Ez_l = E_lcdm(z, lcdm)
    Ez_p = E_psicdm(z, psicdm)

    plt.figure(figsize=(7, 5))
    plt.plot(z, Ez_l, "k-", label=r"$\Lambda$CDM")
    plt.plot(z, Ez_p, "r-", label=r"ΨCDM (best-fit $\varepsilon_0$)")
    plt.xlabel("Redshift $z$")
    plt.ylabel(r"$E(z) = H(z)/H_0$")
    plt.grid(alpha=0.3)
    plt.title("Background expansion: ΛCDM vs ΨCDM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "fig1_Ez_comparison.png", dpi=200)
    plt.close()


def fig2_SN_Hubble(figdir: Path) -> None:
    """
    Fig. 2 – Hubble diagram + residuals for Pantheon+ HF.
    """
    sn = load_pantheonplus_hf(get_data_path("pantheon_plus"))

    z = sn["z"]
    mu_obs = sn["mu"]
    mu_err = sn["mu_err"]

    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    psicdm = PsiCDMParams(H0=70.0, Om0=0.3, eps0=0.031, n=1.0)

    mu_l = 5 * np.log10(dL_lcdm(z, lcdm)) + 25
    mu_p = 5 * np.log10(dL_psicdm(z, psicdm)) + 25

    z_smooth = np.linspace(0.01, 2.5, 400)
    mu_l_s = 5 * np.log10(dL_lcdm(z_smooth, lcdm)) + 25
    mu_p_s = 5 * np.log10(dL_psicdm(z_smooth, psicdm)) + 25

    fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # Hubble diagram
    ax[0].errorbar(z, mu_obs, mu_err, fmt=".", color="grey", alpha=0.6, label="Pantheon+ HF")
    ax[0].plot(z_smooth, mu_l_s, "k-", label=r"$\Lambda$CDM")
    ax[0].plot(z_smooth, mu_p_s, "r-", label=r"ΨCDM")
    ax[0].set_ylabel(r"$\mu(z)$")
    ax[0].set_title("Pantheon+ HF Hubble Diagram")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # Residuals
    ax[1].axhline(0.0, color="black", lw=1)
    ax[1].scatter(z, mu_obs - mu_l, s=6, color="black", label=r"$\Delta\mu$ (ΛCDM)")
    ax[1].scatter(z, mu_obs - mu_p, s=6, color="red", alpha=0.8, label=r"$\Delta\mu$ (ΨCDM)")
    ax[1].set_xlabel("Redshift $z$")
    ax[1].set_ylabel(r"$\Delta\mu$")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(figdir / "fig2_SN_Hubble.png", dpi=200)
    plt.close(fig)


def fig3_BAO_DR12_multipanel(figdir: Path) -> None:
    """
    Fig. 3 – SDSS DR12 BAO distance measures: DM/rs, DH/rs, DV/rs.
    """
    rd = 147.0  # effective sound horizon used for visualisation

    bao = load_bao_dr12(get_data_path("bao"))
    z = bao["z"]
    dm_rs = bao["dm_rs"]  # D_M / r_d
    hz_rs = bao["hz_rs"]  # H * r_d / c

    # Physical values (Mpc)
    DM = dm_rs * rd
    DH = rd / hz_rs
    DV = (DM * DM * z * DH) ** (1.0 / 3.0)

    # Normalised
    DM_rs = dm_rs
    DH_rs = 1.0 / hz_rs
    DV_rs = (DM_rs * DM_rs * z * DH_rs) ** (1.0 / 3.0)

    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    # DM/rs
    ax[0].errorbar(z, DM_rs, fmt="o", ms=6)
    ax[0].set_ylabel(r"$D_M / r_s$")
    ax[0].set_title("SDSS DR12 BAO (consensus)")
    ax[0].grid(alpha=0.3)

    # DH/rs
    ax[1].errorbar(z, DH_rs, fmt="o", ms=6)
    ax[1].set_ylabel(r"$D_H / r_s$")
    ax[1].grid(alpha=0.3)

    # DV/rs
    ax[2].errorbar(z, DV_rs, fmt="o", ms=6)
    ax[2].set_ylabel(r"$D_V / r_s$")
    ax[2].set_xlabel("Redshift $z$")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figdir / "fig3_BAO_DR12_multipanel.png", dpi=200)
    plt.close(fig)


def fig4_BAO_DESI_multipanel(figdir: Path) -> None:
    """
    Fig. 4 – DESI DR2 compressed BAO vector: DM/rs, DH/rs, DV/rs.
    """
    desi = load_desi_dr2(get_data_path("desi", "dr2"))

    DMz, DMv = [], []
    DHz, DHv = [], []
    DVz, DVv = [], []

    for zi, lab, val in zip(desi["z"], desi["labels"], desi["values"]):
        if lab == "DM_over_rs":
            DMz.append(zi)
            DMv.append(val)
        elif lab == "DH_over_rs":
            DHz.append(zi)
            DHv.append(val)
        elif lab == "DV_over_rs":
            DVz.append(zi)
            DVv.append(val)

    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    if DMz:
        ax[0].errorbar(DMz, DMv, fmt="o", ms=5)
    ax[0].set_ylabel(r"$D_M / r_s$")
    ax[0].set_title("DESI DR2 BAO (compressed Gaussian vector)")
    ax[0].grid(alpha=0.3)

    if DHz:
        ax[1].errorbar(DHz, DHv, fmt="o", ms=5)
    ax[1].set_ylabel(r"$D_H / r_s$")
    ax[1].grid(alpha=0.3)

    if DVz:
        ax[2].errorbar(DVz, DVv, fmt="o", ms=5)
    ax[2].set_ylabel(r"$D_V / r_s$")
    ax[2].set_xlabel("Redshift $z$")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figdir / "fig4_BAO_DESI_multipanel.png", dpi=200)
    plt.close(fig)


def fig5_BAO_fits_LCDM_vs_PsiCDM(figdir: Path) -> None:
    """
    Fig. 5 – BAO fits: ΛCDM and ΨCDM curves over SDSS DR12 + DESI DR2.
    Plots DM/rs, DH/rs, DV/rs for both data sets and both models.
    """
    rd = 147.0  # only for visualization

    # --- SDSS DR12 data ---
    bao = load_bao_dr12(get_data_path("bao"))

    z_dr12 = bao["z"]
    dm_rs_dr12 = bao["dm_rs"]
    hz_rs_dr12 = bao["hz_rs"]

    DM_dr12_rs = dm_rs_dr12
    DH_dr12_rs = 1.0 / hz_rs_dr12
    DV_dr12_rs = (DM_dr12_rs**2 * z_dr12 * DH_dr12_rs) ** (1/3)

    # --- DESI DR2 data ---
    desi = load_desi_dr2(get_data_path("desi", "dr2"))

    DMz_desi, DMv_desi = [], []
    DHz_desi, DHv_desi = [], []
    DVz_desi, DVv_desi = [], []

    for zi, lab, val in zip(desi["z"], desi["labels"], desi["values"]):
        if lab == "DM_over_rs":
            DMz_desi.append(zi)
            DMv_desi.append(val)
        elif lab == "DH_over_rs":
            DHz_desi.append(zi)
            DHv_desi.append(val)
        elif lab == "DV_over_rs":
            DVz_desi.append(zi)
            DVv_desi.append(val)

    # --- model redshift range ---
    z_all = np.concatenate([
        z_dr12,
        np.array(DMz_desi) if DMz_desi else np.array([]),
        np.array(DHz_desi) if DHz_desi else np.array([]),
        np.array(DVz_desi) if DVz_desi else np.array([]),
    ])
    z_model = np.linspace(0, float(z_all.max()) * 1.05, 400) if len(z_all) else np.linspace(0, 2.5, 400)

    # --- model parameters ---
    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    psicdm = PsiCDMParams(H0=70.0, Om0=0.3, eps0=0.031, n=1.0)

    # --- model curves (renamed variables to avoid shadowing) ---
    DM_l = DM_lcdm(z_model, lcdm) / lcdm.rd
    DH_l = DH_lcdm(z_model, lcdm) / lcdm.rd
    DV_l = (DM_l**2 * z_model * DH_l) ** (1/3)

    DM_p = DM_psicdm(z_model, psicdm) / psicdm.rd
    DH_p = DH_psicdm(z_model, psicdm) / psicdm.rd
    DV_p = (DM_p**2 * z_model * DH_p) ** (1/3)

    # --- plotting ---
    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    # DM/rs
    ax[0].errorbar(z_dr12, DM_dr12_rs, fmt="o", label="DR12")
    if DMz_desi:
        ax[0].errorbar(DMz_desi, DMv_desi, fmt="s", label="DESI")
    ax[0].plot(z_model, DM_l, "k-", label=r"$\Lambda$CDM")
    ax[0].plot(z_model, DM_p, "r--", label=r"ΨCDM")
    ax[0].set_ylabel(r"$D_M / r_s$")
    ax[0].set_title("BAO fits: ΛCDM vs ΨCDM")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # DH/rs
    ax[1].errorbar(z_dr12, DH_dr12_rs, fmt="o")
    if DHz_desi:
        ax[1].errorbar(DHz_desi, DHv_desi, fmt="s")
    ax[1].plot(z_model, DH_l, "k-")
    ax[1].plot(z_model, DH_p, "r--")
    ax[1].set_ylabel(r"$D_H / r_s$")
    ax[1].grid(alpha=0.3)

    # DV/rs
    ax[2].errorbar(z_dr12, DV_dr12_rs, fmt="o")
    if DVz_desi:
        ax[2].errorbar(DVz_desi, DVv_desi, fmt="s")
    ax[2].plot(z_model, DV_l, "k-")
    ax[2].plot(z_model, DV_p, "r--")
    ax[2].set_ylabel(r"$D_V / r_s$")
    ax[2].set_xlabel("Redshift $z$")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figdir / "fig5_BAO_fits_LCDM_vs_PsiCDM.png", dpi=200)
    plt.close(fig)


def fig6_Hz_dataset(figdir: Path) -> None:
    """
    Fig. 6 – H(z) dataset with ΛCDM and ΨCDM curves.
    """
    hzdata = load_hz_compilation(get_data_path("hz"))

    z = hzdata["z"]
    Hz = hzdata["Hz"]
    sigma_Hz = hzdata["sigma_Hz"]

    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    psicdm = PsiCDMParams(H0=70.0, Om0=0.3, eps0=0.031, n=1.0)

    z_model = np.linspace(0.0, float(z.max()) * 1.05, 400)
    H_l = H_lcdm(z_model, lcdm)
    H_p = H_psicdm(z_model, psicdm)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(z, Hz, yerr=sigma_Hz, fmt="o", ms=4, alpha=0.85, label="H(z) data")
    ax.plot(z_model, H_l, "k-", label=r"$\Lambda$CDM")
    ax.plot(z_model, H_p, "r--", label=r"ΨCDM")
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("H(z) compilation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig6_Hz_dataset.png", dpi=200)
    plt.close(fig)


def fig7_joint_chi2_eps(figdir: Path) -> None:
    """
    Fig. 7 – Joint Δχ²(ε0) from eps-scan table (SN + H(z) + BAO DR12 + DESI).
    """
    path = find_eps_scan_table()
    eps, chi2 = load_eps_table(path)

    if len(eps) == 0:
        raise RuntimeError(f"{path} contains no usable eps-scan data")

    chi2_min = np.min(chi2)
    delta = chi2 - chi2_min

    plt.figure(figsize=(7, 5))
    plt.plot(eps, delta, "r-", lw=2)
    plt.axhline(1.0, color="k", ls="--", lw=0.8)
    plt.axhline(4.0, color="k", ls="--", lw=0.8)
    plt.xlabel(r"$\varepsilon_0$")
    plt.ylabel(r"$\Delta\chi^2_{\rm joint}$")
    plt.title(r"Joint Constraints $\Delta\chi^2(\varepsilon_0)$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "fig7_joint_chi2_eps.png", dpi=200)
    plt.close()


def fig8_delta_chi2_contributions(figdir: Path) -> None:
    """
    Fig. 8 – Δχ² contributions per dataset (fixed numbers from best-fit analysis).
    """
    delta = {
        "SN": +7.376,
        "H(z)": -0.041,
        "BAO DR12": -0.508,
        "DESI DR2": -7.585,
    }

    labels = list(delta.keys())
    values = [delta[k] for k in labels]

    plt.figure(figsize=(7, 5))
    plt.axhline(0.0, color="black", lw=1)
    plt.bar(labels, values)
    plt.ylabel(r"$\Delta\chi^2$")
    plt.title(r"$\Delta\chi^2$ contributions per dataset")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "fig8_delta_chi2_contributions.png", dpi=200)
    plt.close()


# ======================================================================
# APPENDIX FIGURES (diagnostics)
# ======================================================================

def appendix_sn_histograms(figdir: Path) -> None:
    """
    Pantheon+ HF histograms: z-distribution and σ_mu distribution.
    """
    sn = load_pantheonplus_hf(get_data_path("pantheon_plus"))
    z = sn["z"]
    mu_err = sn["mu_err"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

    ax1.hist(z, bins=40, histtype="stepfilled", alpha=0.7)
    ax1.set_xlabel("Redshift $z$")
    ax1.set_ylabel("Number of SNe")
    ax1.set_title("Pantheon+ HF: redshift distribution")
    ax1.grid(alpha=0.3)

    ax2.hist(mu_err, bins=40, histtype="stepfilled", alpha=0.7)
    ax2.set_xlabel(r"$\sigma_\mu$")
    ax2.set_ylabel("Number of SNe")
    ax2.set_title("Pantheon+ HF: distance-modulus uncertainties")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figdir / "sn_histograms.png", dpi=200)
    plt.close(fig)


def appendix_hz_quality(figdir: Path) -> None:
    """
    H(z) quality plots: data points + relative error histogram.
    """
    hzdata = load_hz_compilation(get_data_path("hz"))

    z = hzdata["z"]
    Hz = hzdata["Hz"]
    sigma_Hz = hzdata["sigma_Hz"]
    rel_err = sigma_Hz / Hz

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

    # data
    ax1.errorbar(z, Hz, yerr=sigma_Hz, fmt="o", ms=4)
    ax1.set_xlabel("Redshift $z$")
    ax1.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax1.set_title("H(z) compilation")
    ax1.grid(alpha=0.3)

    # relative errors
    ax2.hist(rel_err, bins=20, histtype="stepfilled", alpha=0.7)
    ax2.set_xlabel(r"$\sigma_H / H$")
    ax2.set_ylabel("Number of points")
    ax2.set_title("Relative errors in H(z)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figdir / "hz_quality_checks.png", dpi=200)
    plt.close(fig)


def appendix_model_checks(figdir: Path) -> None:
    """
    Diagnostic plots for ΛCDM and ΨCDM background:
        - LCDM: E(z), H(z), dL(z)
        - PsiCDM: E(z) for several eps0
    """
    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    z = np.linspace(0.0, 2.5, 400)

    E_l = E_lcdm(z, lcdm)
    H_l = H_lcdm(z, lcdm)
    dL_l = dL_lcdm(z, lcdm)

    # LCDM triple
    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    ax[0].plot(z, E_l)
    ax[0].set_ylabel(r"$E(z)$")
    ax[0].set_title("ΛCDM background checks")
    ax[0].grid(alpha=0.3)

    ax[1].plot(z, H_l)
    ax[1].set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax[1].grid(alpha=0.3)

    ax[2].plot(z, dL_l)
    ax[2].set_ylabel(r"$d_L(z)$ [Mpc]")
    ax[2].set_xlabel("Redshift $z$")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figdir / "model_lcdm_Ez_Hz_dL.png", dpi=200)
    plt.close(fig)

    # PsiCDM E(z) scan for several eps0
    eps_values = [-0.10, -0.05, 0.0, 0.05, 0.10]

    plt.figure(figsize=(7, 5))
    for eps0 in eps_values:
        psi = PsiCDMParams(H0=lcdm.H0, Om0=lcdm.Om0, eps0=eps0, n=1.0)
        E_p = E_psicdm(z, psi)
        label = rf"$\varepsilon_0={eps0:+.2f}$"
        plt.plot(z, E_p, label=label)

    plt.xlabel("Redshift $z$")
    plt.ylabel(r"$E(z)$")
    plt.title(r"ΨCDM: $E(z)$ for different $\varepsilon_0$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "model_psicdm_Ez_scan.png", dpi=200)
    plt.close()


def appendix_hz_only_chi2_scan(figdir: Path) -> None:
    """
    Δχ²(ε0) for H(z) only (ΛCDM vs ΨCDM).
    """
    hzdata = load_hz_compilation(get_data_path("hz"))

    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    chi2_l = chi2_hz(hzdata, H_lcdm, lcdm)

    eps_vals = np.linspace(-0.10, 0.10, 81)
    chi2_psi = []

    for eps0 in eps_vals:
        psi = PsiCDMParams(H0=70.0, Om0=0.3, eps0=eps0, n=1.0)
        chi2_psi.append(chi2_hz(hzdata, H_psicdm, psi))

    chi2_psi = np.array(chi2_psi)
    dchi2 = chi2_psi - chi2_l

    idx_best = np.argmin(dchi2)
    eps_best = eps_vals[idx_best]
    dchi2_best = dchi2[idx_best]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(eps_vals, dchi2, "-")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(eps_best, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$\varepsilon_0$")
    ax.set_ylabel(r"$\Delta\chi^2_{\rm H(z)}$")
    ax.set_title(r"H(z) only: $\Delta\chi^2(\varepsilon_0)$")

    txt = rf"best: $\varepsilon_0={eps_best:.3f}$" + "\n" + rf"$\Delta\chi^2={dchi2_best:.3f}$"
    ax.text(
        0.02,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(figdir / "hz_only_chi2_scan.png", dpi=200)
    plt.close(fig)


def appendix_bao_raw(figdir: Path) -> None:
    """
    Raw BAO diagnostic plots:
        - SDSS DR12 (DM/rs, DH/rs, DV/rs)
        - DESI DR2 compressed vector (DM/rs, DH/rs, DV/rs)
    """
    rd = 147.0

    # DR12
    bao = load_bao_dr12(get_data_path("bao"))
    z = bao["z"]
    dm_rs = bao["dm_rs"]
    hz_rs = bao["hz_rs"]

    DM_rs = dm_rs
    DH_rs = 1.0 / hz_rs
    DV_rs = (DM_rs**2 * z * DH_rs) ** (1.0 / 3.0)

    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    ax[0].plot(z, DM_rs, "o-")
    ax[0].set_ylabel(r"$D_M/r_s$")
    ax[0].set_title("SDSS DR12 BAO (raw)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(z, DH_rs, "o-")
    ax[1].set_ylabel(r"$D_H/r_s$")
    ax[1].grid(alpha=0.3)

    ax[2].plot(z, DV_rs, "o-")
    ax[2].set_ylabel(r"$D_V/r_s$")
    ax[2].set_xlabel("Redshift $z$")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figdir / "bao_dr12_raw.png", dpi=200)
    plt.close(fig)

    # DESI
    desi = load_desi_dr2(get_data_path("desi", "dr2"))

    DMz, DMv = [], []
    DHz, DHv = [], []
    DVz, DVv = [], []

    for zi, lab, val in zip(desi["z"], desi["labels"], desi["values"]):
        if lab == "DM_over_rs":
            DMz.append(zi)
            DMv.append(val)
        elif lab == "DH_over_rs":
            DHz.append(zi)
            DHv.append(val)
        elif lab == "DV_over_rs":
            DVz.append(zi)
            DVv.append(val)

    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    if DMz:
        ax[0].plot(DMz, DMv, "o-")
    ax[0].set_ylabel(r"$D_M/r_s$")
    ax[0].set_title("DESI DR2 BAO (raw)")
    ax[0].grid(alpha=0.3)

    if DHz:
        ax[1].plot(DHz, DHv, "o-")
    ax[1].set_ylabel(r"$D_H/r_s$")
    ax[1].grid(alpha=0.3)

    if DVz:
        ax[2].plot(DVz, DVv, "o-")
    ax[2].set_ylabel(r"$D_V/r_s$")
    ax[2].set_xlabel("Redshift $z$")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figdir / "bao_desi_raw.png", dpi=200)
    plt.close(fig)


# ======================================================================
# TALK & POSTER FIGURES
# ======================================================================

def talk_summary(figdir: Path, main_dir: Path) -> None:
    """
    Simple 16:9 summary slide combining:
        - E(z) comparison
        - joint Δχ²(ε0)
        - BAO fits panel (thumbnail)
    It reuses already generated main_figures PNGs.
    """
    import matplotlib.image as mpimg

    # Reuse existing figs
    Ez_path = main_dir / "fig1_Ez_comparison.png"
    chi2_path = main_dir / "fig7_joint_chi2_eps.png"
    bao_path = main_dir / "fig5_BAO_fits_LCDM_vs_PsiCDM.png"

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for ax, pth, title in zip(
        axes[:3],
        [Ez_path, chi2_path, bao_path],
        ["E(z)", r"$\Delta\chi^2(\varepsilon_0)$", "BAO fits"],
    ):
        if pth.exists():
            img = mpimg.imread(pth)
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.axis("off")

    # Empty bottom-right with text
    ax = axes[3]
    ax.axis("off")
    ax.text(
        0.02,
        0.95,
        r"Psi-Continuum v2" + "\n"
        + r"ΛCDM leftrightarrow ΨCDM comparison" + "\n"
        + r"Datasets: Pantheon+ HF, H(z), SDSS DR12 BAO, DESI DR2",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(figdir / "talk_summary.png", dpi=150)
    plt.close(fig)


def talk_bao(figdir: Path, main_dir: Path) -> None:
    """
    BAO-only talk figure: just reuse fig5_BAO_fits_LCDM_vs_PsiCDM.
    """
    import matplotlib.image as mpimg

    src = main_dir / "fig5_BAO_fits_LCDM_vs_PsiCDM.png"
    if not src.exists():
        return

    img = mpimg.imread(src)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figdir / "talk_bao.png", dpi=150)
    plt.close(fig)


def talk_sn(figdir: Path, main_dir: Path) -> None:
    """
    SN-only talk figure: reuse fig2_SN_Hubble.
    """
    import matplotlib.image as mpimg

    src = main_dir / "fig2_SN_Hubble.png"
    if not src.exists():
        return

    img = mpimg.imread(src)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figdir / "talk_sn.png", dpi=150)
    plt.close(fig)


def poster_summary(summary_dir: Path, main_dir: Path) -> None:
    """
    Poster-style summary figure: grid collage from key main figures.
    """
    import matplotlib.image as mpimg

    Ez_path = main_dir / "fig1_Ez_comparison.png"
    sn_path = main_dir / "fig2_SN_Hubble.png"
    bao_path = main_dir / "fig5_BAO_fits_LCDM_vs_PsiCDM.png"
    chi2_path = main_dir / "fig7_joint_chi2_eps.png"

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = axes.flatten()
    paths = [Ez_path, sn_path, bao_path, chi2_path]
    titles = ["E(z)", "Hubble Diagram", "BAO fits", r"$\Delta\chi^2(\varepsilon_0)$"]

    for ax, pth, title in zip(axes, paths, titles):
        if pth.exists():
            img = mpimg.imread(pth)
            ax.imshow(img)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.axis("off")

    fig.suptitle("Psi-Continuum v2 – Summary", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(summary_dir / "poster_summary.png", dpi=200)
    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    root = Path.cwd()
    pub_root = root / "results" / "figures" / "publication"

    main_dir = pub_root / "main_figures"
    appendix_dir = pub_root / "appendix"
    talk_dir = pub_root / "talk_figures"
    summary_dir = pub_root / "summary"

    for p in (main_dir, appendix_dir, talk_dir, summary_dir):
        ensure_dir(p)

    print("Generating publication-ready Psi-Continuum v2 figures...")
    print(f"Output root: {pub_root}")

    # --- Main figures for the article ---
    fig1_Ez_comparison(main_dir)
    fig2_SN_Hubble(main_dir)
    fig3_BAO_DR12_multipanel(main_dir)
    fig4_BAO_DESI_multipanel(main_dir)
    fig5_BAO_fits_LCDM_vs_PsiCDM(main_dir)
    fig6_Hz_dataset(main_dir)
    fig7_joint_chi2_eps(main_dir)
    fig8_delta_chi2_contributions(main_dir)

    # --- Appendix / diagnostics ---
    appendix_sn_histograms(appendix_dir)
    appendix_hz_quality(appendix_dir)
    appendix_model_checks(appendix_dir)
    appendix_hz_only_chi2_scan(appendix_dir)
    appendix_bao_raw(appendix_dir)

    # --- Talk & poster figures ---
    talk_summary(talk_dir, main_dir)
    talk_bao(talk_dir, main_dir)
    talk_sn(talk_dir, main_dir)
    poster_summary(summary_dir, main_dir)

    print("Done. Copy `results/figures/publication/main_figures/` into your LaTeX project.")


if __name__ == "__main__":
    main()
