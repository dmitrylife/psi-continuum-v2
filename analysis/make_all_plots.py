# analysis/make_all_plots.py

"""
Final publication-grade plot generator for Psi-Continuum v2.

This version is:
 - fully compatible with your real data structures
 - supports compressed DESI DR2 BAO vector
 - robust to malformed table rows
 - robust to missing DV points
 - robust to any number of columns in eps_scan
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cosmology.background.lcdm import E_lcdm, dL_lcdm, H_lcdm
from cosmology.background.psicdm import E_psicdm, dL_psicdm, H_psicdm

from cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from cosmology.data_loaders.hz_loader import load_hz_compilation
from cosmology.data_loaders.bao_loader import load_bao_dr12
from cosmology.data_loaders.desi_loader import load_desi_dr2

from cosmology.models.lcdm_params import LCDMParams
from cosmology.models.psicdm_params import PsiCDMParams

from cosmology.likelihoods.hz_likelihood import chi2_hz


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ==============================================================================
#  Utility: safely parse eps-scan table
# ==============================================================================
def load_eps_table(path: Path):
    """
    Robust loader for eps-scan table.

    Expected format (from scan_eps_psicdm.py):

        # eps0  chi2_sn  chi2_hz  chi2_bao  chi2_desi  chi2_total  delta_total

    But we allow:
    - arbitrary comments (# ...)
    - arbitrary extra columns
    - fall back to 2-column format: eps, chi2
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
                continue  # skip malformed rows

            try:
                e = float(parts[0])
                # If we have a "full" line from scan_eps_psicdm.txt,
                # use chi2_total (6-я колонка, индекс 5).
                if len(parts) >= 6:
                    c = float(parts[5])
                else:
                    # Fallback: second column is chi2
                    c = float(parts[1])
            except ValueError:
                continue  # skip bad data rows

            eps.append(e)
            chi2.append(c)

    return np.array(eps), np.array(chi2)


# ==============================================================================
# 1. E(z) comparison
# ==============================================================================
def plot_Ez_comparison(figdir: Path):
    z = np.linspace(0, 3.0, 400)

    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    psicdm = PsiCDMParams(H0=70.0, Om0=0.3, eps0=0.031, n=1.0)

    plt.figure(figsize=(8, 5))
    plt.plot(z, E_lcdm(z, lcdm), 'k-', label=r'$\Lambda$CDM')
    plt.plot(z, E_psicdm(z, psicdm), 'r-', label=r'$\Psi$CDM')
    plt.xlabel("Redshift z")
    plt.ylabel(r"$E(z)$")
    plt.grid(alpha=0.3)
    plt.title("Expansion Rate Comparison")
    plt.legend()

    plt.savefig(figdir / "Ez_comparison.png", dpi=180)
    plt.close()


# ==============================================================================
# 2. dL + SN residuals
# ==============================================================================
def plot_dL_SN(figdir: Path):
    root = Path(__file__).resolve().parents[1]
    sn = load_pantheonplus_hf(root / "data" / "pantheon_plus")
    z = sn["z"]
    mu_obs = sn["mu"]
    mu_err = sn["mu_err"]

    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    psicdm = PsiCDMParams(H0=70.0, Om0=0.3, eps0=0.031, n=1.0)

    mu_l = 5 * np.log10(dL_lcdm(z, lcdm)) + 25
    mu_p = 5 * np.log10(dL_psicdm(z, psicdm)) + 25

    z_smooth = np.linspace(0.02, 2.5, 400)
    mu_l_s = 5 * np.log10(dL_lcdm(z_smooth, lcdm)) + 25
    mu_p_s = 5 * np.log10(dL_psicdm(z_smooth, psicdm)) + 25

    fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # Top panel
    ax[0].errorbar(z, mu_obs, mu_err, fmt='.', color='gray', alpha=0.6)
    ax[0].plot(z_smooth, mu_l_s, 'k-', label=r'$\Lambda$CDM')
    ax[0].plot(z_smooth, mu_p_s, 'r-', label=r'$\Psi$CDM')
    ax[0].set_ylabel(r"$\mu(z)$")
    ax[0].set_title("Hubble Diagram (Pantheon+ HF)")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # Bottom panel
    ax[1].axhline(0, color='black', lw=1)
    ax[1].scatter(z, mu_obs - mu_l, s=5, color='black', label="ΛCDM residuals")
    ax[1].scatter(z, mu_obs - mu_p, s=5, color='red', label="ΨCDM residuals")
    ax[1].set_xlabel("Redshift z")
    ax[1].set_ylabel(r"$\Delta\mu$")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(figdir / "dL_SN_comparison.png", dpi=180)
    plt.close()


# ==============================================================================
# 3. BAO multipanel (DR12 + DESI DR2 compressed vector)
# ==============================================================================
def plot_BAO_multipanel(figdir: Path):
    root = Path(__file__).resolve().parents[1]
    rd = 147.0

    # ------------------------ SDSS DR12 ------------------------
    dr12 = load_bao_dr12(root / "data" / "bao")
    z12 = dr12["z"]

    # In your format:
    #  dm_rs ≡ D_M / r_d
    #  hz_rs ≡ H * r_d / c
    # (standard DR12 convention)

    dm_rs = dr12["dm_rs"]
    hz_rs = dr12["hz_rs"]

    # Physical values (Mpc)
    DM12 = dm_rs * rd                    # D_M = (D_M/r_d) * r_d
    DH12 = rd / hz_rs                    # DH = c/H = r_d / (H r_d / c)
    DV12 = (DM12**2 * z12 * DH12) ** (1.0 / 3.0)

    # Normalized values (DM/rs, DH/rs, DV/rs)
    DM12_rs = dm_rs
    DH12_rs = 1.0 / hz_rs                # DH / r_d = 1 / (H r_d / c)
    DV12_rs = (DM12_rs**2 * z12 * DH12_rs) ** (1.0 / 3.0)

    # ------------------------ DESI DR2 ------------------------
    desi = load_desi_dr2(root / "data" / "desi" / "dr2")

    # Physical: DM, DH, DV (in Mpc)
    DMz, DMv = [], []
    DHz, DHv = [], []
    DVz, DVv = [], []

    # Normalized: DM/rs, DH/rs, DV/rs
    DMz_rs, DMv_rs = [], []
    DHz_rs, DHv_rs = [], []
    DVz_rs, DVv_rs = [], []

    for zi, lab, val in zip(desi["z"], desi["labels"], desi["values"]):

        # DESI gives directly DM/rs, DH/rs, DV/rs
        if lab == "DM_over_rs":
            DMz_rs.append(zi)
            DMv_rs.append(val)
            DMz.append(zi)
            DMv.append(val * rd)

        elif lab == "DH_over_rs":
            DHz_rs.append(zi)
            DHv_rs.append(val)
            DHz.append(zi)
            DHv.append(val * rd)

        elif lab == "DV_over_rs":
            DVz_rs.append(zi)
            DVv_rs.append(val)
            DVz.append(zi)
            DVv.append(val * rd)

    # ------------------------ PLOTS ------------------------

    # === 1) Physical multipanel ===
    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    # Panel 1: DM in Mpc
    ax[0].scatter(z12, DM12, c='blue', s=25, label="DR12 DM")
    ax[0].scatter(DMz, DMv, c='purple', s=25, label="DESI DM")
    ax[0].set_ylabel("DM [Mpc]")
    ax[0].set_title("BAO Distance Measures (physical Mpc)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # Panel 2: DH in Mpc
    ax[1].scatter(z12, DH12, c='blue', s=25, label="DR12 DH")
    ax[1].scatter(DHz, DHv, c='purple', s=25, label="DESI DH")
    ax[1].set_ylabel("DH [Mpc]")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    # Panel 3: DV in Mpc
    ax[2].scatter(z12, DV12, c='blue', s=25, label="DR12 DV")
    if len(DVz):
        ax[2].scatter(DVz, DVv, c='purple', s=25, label="DESI DV")
    ax[2].set_ylabel("DV [Mpc]")
    ax[2].set_xlabel("Redshift z")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figdir / "BAO_multipanel.png", dpi=180)
    plt.close()

    # === 2) Normalized multipanel ===
    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    # Panel 1: DM/rs
    ax[0].scatter(z12, DM12_rs, c='blue', s=25, label="DR12 DM/rs")
    ax[0].scatter(DMz_rs, DMv_rs, c='purple', s=25, label="DESI DM/rs")
    ax[0].set_ylabel("DM / r_s")
    ax[0].set_title("BAO Distance Measures (normalized to r_s)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # Panel 2: DH/rs
    ax[1].scatter(z12, DH12_rs, c='blue', s=25, label="DR12 DH/rs")
    ax[1].scatter(DHz_rs, DHv_rs, c='purple', s=25, label="DESI DH/rs")
    ax[1].set_ylabel("DH / r_s")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    # Panel 3: DV/rs
    ax[2].scatter(z12, DV12_rs, c='blue', s=25, label="DR12 DV/rs")
    if len(DVz_rs):
        ax[2].scatter(DVz_rs, DVv_rs, c='purple', s=25, label="DESI DV/rs")
    ax[2].set_ylabel("DV / r_s")
    ax[2].set_xlabel("Redshift z")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figdir / "BAO_multipanel_rs.png", dpi=180)
    plt.close()


# ==============================================================================
# 4. Δχ² contributions (fixed numeric values from your best-fit)
# ==============================================================================
def plot_delta_chi2_bar(figdir: Path):
    delta = {
        "SN": +7.376,
        "H(z)": -0.041,
        "BAO DR12": -0.508,
        "DESI DR2": -7.585,
    }

    plt.figure(figsize=(7, 5))
    plt.axhline(0, color='black', lw=1)
    plt.bar(delta.keys(), delta.values(), color='gray')
    plt.ylabel(r"$\Delta\chi^2$")
    plt.title("Δχ² Contributions per Dataset")
    plt.grid(axis='y', alpha=0.3)

    plt.savefig(figdir / "delta_chi2_bar.png", dpi=180)
    plt.close()


# ==============================================================================
# 5. Δχ²(eps0) curve (joint SN+H+BAO+DESI)
# ==============================================================================
def plot_chi2_epsilon_curve(figdir: Path):
    root = Path(__file__).resolve().parents[1]
    path = root / "results" / "tables" / "eps_scan_psicdm.txt"

    eps, chi2 = load_eps_table(path)
    if len(eps) == 0:
        raise RuntimeError("eps_scan_psicdm.txt contains no usable data")

    chi2_min = np.min(chi2)
    delta = chi2 - chi2_min

    plt.figure(figsize=(8, 5))
    plt.plot(eps, delta, 'r-', lw=2)
    plt.axhline(1, color='k', ls='--')
    plt.axhline(4, color='k', ls='--')
    plt.xlabel(r"$\varepsilon_0$")
    plt.ylabel(r"$\Delta\chi^2$")
    plt.grid(alpha=0.3)
    plt.title("Joint Likelihood Δχ²(ε₀)")

    plt.savefig(figdir / "chi2_epsilon_curve.png", dpi=180)
    plt.close()


# ==============================================================================
# 6. H(z) dataset + curves
# ==============================================================================
def plot_hz_dataset(figdir: Path) -> None:
    """
    H(z) data + ΛCDM and ΨCDM curves.
    Output: hz_psicdm_test.png
    """
    root = Path(__file__).resolve().parents[1]
    hzdata = load_hz_compilation(root / "data" / "hz")

    z = hzdata["z"]
    Hz = hzdata["Hz"]
    sigma_Hz = hzdata["sigma_Hz"]

    # Same fiducial parameters as in other tests
    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    psicdm = PsiCDMParams(H0=70.0, Om0=0.3, eps0=0.031, n=1.0)

    z_smooth = np.linspace(0.0, z.max() * 1.05, 400)
    H_l = H_lcdm(z_smooth, lcdm)
    H_p = H_psicdm(z_smooth, psicdm)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(z, Hz, yerr=sigma_Hz, fmt="o", ms=4, label="H(z) data")
    ax.plot(z_smooth, H_l, label=r"$\Lambda$CDM")
    ax.plot(
        z_smooth,
        H_p,
        label=r"$\Psi$CDM (best-fit $\varepsilon_0$)",
        ls="--",
    )

    ax.set_xlabel("Redshift z")
    ax.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(figdir / "hz_psicdm_test.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# 7. H(z) Δχ²(ε₀) scan
# ==============================================================================
def plot_hz_chi2_scan(figdir: Path) -> None:
    """
    Δχ²(ε₀) for H(z) only (ΛCDM vs ΨCDM).
    Output: hz_psicdm_chi2_eps_scan.png
    """
    root = Path(__file__).resolve().parents[1]
    hzdata = load_hz_compilation(root / "data" / "hz")

    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    chi2_lcdm = chi2_hz(hzdata, H_lcdm, lcdm)

    # Scan ε₀ in the same range as before
    eps_values = np.linspace(-0.10, 0.10, 81)
    chi2_psi = []

    for eps0 in eps_values:
        psi = PsiCDMParams(H0=70.0, Om0=0.3, eps0=eps0, n=1.0)
        chi2_val = chi2_hz(hzdata, H_psicdm, psi)
        chi2_psi.append(chi2_val)

    chi2_psi = np.array(chi2_psi)
    dchi2 = chi2_psi - chi2_lcdm

    idx_best = np.argmin(dchi2)
    eps_best = eps_values[idx_best]
    dchi2_best = dchi2[idx_best]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eps_values, dchi2, "-")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(eps_best, color="k", ls="--", lw=0.8)

    ax.set_xlabel(r"$\varepsilon_0$")
    ax.set_ylabel(r"$\Delta\chi^2_{\mathrm{H(z)}}$")
    ax.set_title(r"H(z) only: $\Delta\chi^2(\varepsilon_0)$")

    txt = (
        rf"best: $\varepsilon_0 = {eps_best:.3f}$"
        + "\n"
        + rf"$\Delta\chi^2 = {dchi2_best:.3f}$"
    )
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
    fig.savefig(figdir / "hz_psicdm_chi2_eps_scan.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# 8. Simple SN data histograms
# ==============================================================================
def plot_sn_data_histograms(figdir: Path) -> None:
    """
    Simple Pantheon+ HF data histograms:
    - redshift distribution
    - distance-modulus error distribution

    Output: pantheonplus_data_histograms.png
    """
    root = Path(__file__).resolve().parents[1]
    sn = load_pantheonplus_hf(root / "data" / "pantheon_plus")

    z = sn["z"]
    mu_err = sn["mu_err"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

    ax1.hist(z, bins=40, histtype="stepfilled", alpha=0.7)
    ax1.set_xlabel("Redshift z")
    ax1.set_ylabel("Number of SNe")
    ax1.set_title("Pantheon+ HF: redshift distribution")
    ax1.grid(alpha=0.3)

    ax2.hist(mu_err, bins=40, histtype="stepfilled", alpha=0.7)
    ax2.set_xlabel(r"$\sigma_{\mu}$")
    ax2.set_ylabel("Number of SNe")
    ax2.set_title("Pantheon+ HF: distance-modulus uncertainties")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figdir / "pantheonplus_data_histograms.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    root = Path(__file__).resolve().parents[1]
    outdir = root / "results" / "figures" / "publication"
    ensure_dir(outdir)

    print("Generating publication-ready plots...")

    plot_Ez_comparison(outdir)
    plot_dL_SN(outdir)
    plot_BAO_multipanel(outdir)
    plot_delta_chi2_bar(outdir)
    plot_chi2_epsilon_curve(outdir)

    plot_hz_dataset(outdir)
    plot_hz_chi2_scan(outdir)
    plot_sn_data_histograms(outdir)

    print(f"All plots saved to: {outdir}")


if __name__ == "__main__":
    main()
