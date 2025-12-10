# analysis/check/check_models.py

"""
Basic sanity checks for ΛCDM and ΨCDM background models.

Performed tests:

1) ΛCDM:
   - Verify E(0) = 1 and H(0) = H0.
   - Verify d_L(0) ≈ 0.
   - Verify luminosity distance is monotonic increasing.
   - Produce diagnostic plots: E(z), H(z), d_L(z).

2) ΨCDM:
   - Verify ΨCDM → ΛCDM limit when eps0 = 0.
   - Produce diagnostic plots:
        * E(z) for several eps0 values
        * Comparison ΨCDM vs ΛCDM

Output directory:
    results/figures/model_checks/
"""

import numpy as np
import matplotlib.pyplot as plt

from psi_continuum_v2.cosmology.background.lcdm import E_lcdm, H_lcdm, dL_lcdm
from psi_continuum_v2.cosmology.background.psicdm import E_psicdm, H_psicdm, dL_psicdm
from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams
from psi_continuum_v2.utils.paths import get_results_path


# ======================================================================
# Helper: monotonicity
# ======================================================================

def check_monotonic_increasing(x: np.ndarray, y: np.ndarray, name: str) -> None:
    """
    Verify that y(x) is strictly increasing (allowing small numerical noise).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    dx = np.diff(x)
    dy = np.diff(y)

    if np.any(dx <= 0):
        raise ValueError(f"{name}: x-grid must be strictly increasing.")

    eps = 1e-10
    if np.any(dy < -eps):
        idx = np.where(dy < -eps)[0][0]
        raise ValueError(
            f"{name}: Non-monotonic at x[{idx}]={x[idx]} → x[{idx+1}]={x[idx+1]}, dy={dy[idx]}"
        )


# ======================================================================
# LCDM checks
# ======================================================================

def check_lcdm_basic(params: LCDMParams) -> None:
    """
    Perform core consistency checks for ΛCDM.
    """
    E0 = float(E_lcdm([0.0], params)[0])
    H0_val = float(H_lcdm([0.0], params)[0])

    if not np.allclose(E0, 1.0, rtol=1e-10):
        raise ValueError(f"LCDM: Expected E(0)=1, got {E0}")

    if not np.allclose(H0_val, params.H0, rtol=1e-10):
        raise ValueError(f"LCDM: Expected H(0)=H0, got {H0_val}, H0={params.H0}")

    # Check monotonicity of d_L(z)
    z_grid = np.linspace(0.0, 2.5, 400)
    dL = dL_lcdm(z_grid, params)

    if abs(dL[0]) > 1e-6:
        raise ValueError(f"LCDM: d_L(0) ≠ 0 within tolerance, got {dL[0]}")

    check_monotonic_increasing(z_grid, dL, "LCDM d_L(z)")

    if not np.all(np.isfinite(dL)):
        raise ValueError("LCDM: Non-finite values in d_L(z)")

    print("LCDM basic checks passed.")


# ======================================================================
# ΨCDM → ΛCDM limit
# ======================================================================

def check_psicdm_limit_to_lcdm(params_lcdm: LCDMParams) -> None:
    """
    Verify ΨCDM reduces to ΛCDM when eps0 = 0.
    """
    psi0 = PsiCDMParams(
        H0=params_lcdm.H0,
        Om0=params_lcdm.Om0,
        eps0=0.0,
        n=1.0,
    )

    z = np.linspace(0.0, 2.5, 400)

    E_l = E_lcdm(z, params_lcdm)
    E_p = E_psicdm(z, psi0)

    H_l = H_lcdm(z, params_lcdm)
    H_p = H_psicdm(z, psi0)

    dL_l = dL_lcdm(z, params_lcdm)
    dL_p = dL_psicdm(z, psi0)

    if not np.allclose(E_l, E_p, rtol=1e-10, atol=1e-10):
        raise ValueError("E(z) mismatch: ΨCDM(eps0=0) != ΛCDM")

    if not np.allclose(H_l, H_p, rtol=1e-10):
        raise ValueError("H(z) mismatch: ΨCDM(eps0=0) != ΛCDM")

    if not np.allclose(dL_l, dL_p, rtol=1e-8):
        raise ValueError("d_L mismatch: ΨCDM(eps0=0) != ΛCDM")

    print("ΨCDM → ΛCDM limit OK (eps0 = 0).")


# ======================================================================
# Plot generators
# ======================================================================

def make_lcdm_plots(params: LCDMParams, outdir) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.linspace(0.0, 2.5, 400)

    E_vals = E_lcdm(z, params)
    H_vals = H_lcdm(z, params)
    dL_vals = dL_lcdm(z, params)

    # E(z)
    plt.figure(figsize=(6, 4))
    plt.plot(z, E_vals)
    plt.xlabel("z")
    plt.ylabel("E(z) = H(z)/H0")
    plt.title("ΛCDM: E(z)")
    plt.tight_layout()
    plt.savefig(outdir / "lcdm_Ez.png", dpi=200)
    plt.close()

    # H(z)
    plt.figure(figsize=(6, 4))
    plt.plot(z, H_vals)
    plt.xlabel("z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.title("ΛCDM: H(z)")
    plt.tight_layout()
    plt.savefig(outdir / "lcdm_Hz.png", dpi=200)
    plt.close()

    # d_L(z)
    plt.figure(figsize=(6, 4))
    plt.plot(z, dL_vals)
    plt.xlabel("z")
    plt.ylabel("d_L(z) [Mpc]")
    plt.title("ΛCDM: luminosity distance")
    plt.tight_layout()
    plt.savefig(outdir / "lcdm_dL.png", dpi=200)
    plt.close()


def make_psicdm_plots(params_lcdm: LCDMParams, outdir) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.linspace(0.0, 2.5, 400)
    E_l = E_lcdm(z, params_lcdm)

    eps_values = [-0.1, -0.05, 0.0, 0.05, 0.1]

    plt.figure(figsize=(6, 4))
    for eps in eps_values:
        p = PsiCDMParams(
            H0=params_lcdm.H0,
            Om0=params_lcdm.Om0,
            eps0=eps,
            n=1.0,
        )
        E_p = E_psicdm(z, p)
        plt.plot(z, E_p, label=rf"$\varepsilon_0={eps:+.2f}$")

    plt.xlabel("z")
    plt.ylabel("E(z)")
    plt.title("ΨCDM: E(z) for multiple ε₀")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "psicdm_eps_scan_Ez.png", dpi=200)
    plt.close()

    # Comparison with LCDM
    eps_test = 0.05
    p_test = PsiCDMParams(
        H0=params_lcdm.H0,
        Om0=params_lcdm.Om0,
        eps0=eps_test,
        n=1.0,
    )
    E_p_test = E_psicdm(z, p_test)

    plt.figure(figsize=(6, 4))
    plt.plot(z, E_l, label="ΛCDM")
    plt.plot(z, E_p_test, label=rf"ΨCDM, $\varepsilon_0={eps_test:+.2f}$")
    plt.xlabel("z")
    plt.ylabel("E(z)")
    plt.title("ΛCDM vs ΨCDM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "psicdm_vs_lcdm_Ez.png", dpi=200)
    plt.close()


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    # results/figures/model_checks/ next to data/
    outdir = get_results_path("figures", "model_checks")

    lcdm = LCDMParams(H0=70.0, Om0=0.3)

    print("=== Checking ΛCDM ===")
    check_lcdm_basic(lcdm)
    make_lcdm_plots(lcdm, outdir)

    print("=== Checking ΨCDM limit ===")
    check_psicdm_limit_to_lcdm(lcdm)

    print("=== Generating ΨCDM plots ===")
    make_psicdm_plots(lcdm, outdir)

    print("All model checks complete.")
    print("Figures saved to:", outdir)


if __name__ == "__main__":
    main()
