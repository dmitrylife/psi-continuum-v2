# analysis/check_models.py

"""
Basic sanity checks for ΛCDM and ΨCDM background models:

- Verify E(0) = 1 and H(0) = H0.
- Verify d_L(0) ≈ 0 and d_L(z) is monotonic increasing.
- Check that ΨCDM reduces to ΛCDM when eps0 = 0.
- Produce a few diagnostic plots for visual inspection.

Output:
- results/figures/model_checks/lcdm_Ez.png
- results/figures/model_checks/lcdm_Hz.png
- results/figures/model_checks/lcdm_dL.png
- results/figures/model_checks/psicdm_vs_lcdm_Ez.png
- results/figures/model_checks/psicdm_eps_scan_Ez.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cosmology.background.lcdm import E_lcdm, H_lcdm, dL_lcdm
from cosmology.background.psicdm import E_psicdm, H_psicdm, dL_psicdm
from cosmology.models.lcdm_params import LCDMParams
from cosmology.models.psicdm_params import PsiCDMParams


def check_monotonic_increasing(x: np.ndarray, y: np.ndarray, name: str) -> None:
    """
    Check that y(x) is strictly increasing (or at least non-decreasing).
    Raises ValueError if the monotonicity is violated.
    """
    dx = np.diff(x)
    dy = np.diff(y)

    if np.any(dx <= 0):
        raise ValueError(f"{name}: x is not strictly increasing.")

    # allow tiny numerical noise: require dy >= -eps
    eps = 1e-10
    if np.any(dy < -eps):
        idx = np.where(dy < -eps)[0][0]
        raise ValueError(
            f"{name}: function is not monotonic increasing at x[{idx}]={x[idx]} → "
            f"x[{idx+1}]={x[idx+1]}, dy={dy[idx]}"
        )


def check_lcdm_basic(params: LCDMParams) -> None:
    """
    Perform basic consistency checks for flat ΛCDM model.
    """
    z0 = np.array([0.0])
    E0 = E_lcdm(z0, params)[0]
    H0_val = H_lcdm(z0, params)[0]

    # E(0) = 1
    if not np.allclose(E0, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(f"LCDM: E(0) != 1, got {E0}")

    # H(0) = H0
    if not np.allclose(H0_val, params.H0, rtol=1e-10, atol=1e-10):
        raise ValueError(f"LCDM: H(0) != H0, got {H0_val}, H0={params.H0}")

    # d_L(0) ≈ 0 and monotonic in z
    z_grid = np.linspace(0.0, 2.5, 400)
    dL = dL_lcdm(z_grid, params)

    if np.abs(dL[0]) > 1e-6:
        raise ValueError(f"LCDM: d_L(0) is not ~0, got {dL[0]}")

    check_monotonic_increasing(z_grid, dL, "LCDM d_L(z)")

    if not np.all(np.isfinite(dL)):
        raise ValueError("LCDM: d_L(z) contains non-finite values.")

    print("LCDM basic checks passed.")


def check_psicdm_limit_to_lcdm(params_lcdm: LCDMParams) -> None:
    """
    Check that ΨCDM reduces to ΛCDM when eps0 = 0.
    """
    # Match LCDM parameters
    p_psi = PsiCDMParams(
        H0=params_lcdm.H0,
        Om0=params_lcdm.Om0,
        eps0=0.0,
        n=1.0,
    )

    z_grid = np.linspace(0.0, 2.5, 400)

    E_l = E_lcdm(z_grid, params_lcdm)
    E_p = E_psicdm(z_grid, p_psi)

    H_l = H_lcdm(z_grid, params_lcdm)
    H_p = H_psicdm(z_grid, p_psi)

    dL_l = dL_lcdm(z_grid, params_lcdm)
    dL_p = dL_psicdm(z_grid, p_psi)

    # All of these should match to high precision
    if not np.allclose(E_l, E_p, rtol=1e-10, atol=1e-10):
        raise ValueError("PsiCDM limit check failed: E_lcdm != E_psicdm(eps0=0).")

    if not np.allclose(H_l, H_p, rtol=1e-10, atol=1e-10):
        raise ValueError("PsiCDM limit check failed: H_lcdm != H_psicdm(eps0=0).")

    if not np.allclose(dL_l, dL_p, rtol=1e-8, atol=1e-8):
        raise ValueError("PsiCDM limit check failed: dL_lcdm != dL_psicdm(eps0=0).")

    print("PsiCDM → LCDM limit checks passed (eps0 = 0).")


def make_lcdm_plots(params: LCDMParams, outdir: Path) -> None:
    """
    Produce diagnostic plots for ΛCDM:
    - E(z)
    - H(z)
    - d_L(z)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    z_grid = np.linspace(0.0, 2.5, 400)
    E_vals = E_lcdm(z_grid, params)
    H_vals = H_lcdm(z_grid, params)
    dL_vals = dL_lcdm(z_grid, params)

    # E(z)
    plt.figure(figsize=(6, 4))
    plt.plot(z_grid, E_vals)
    plt.xlabel("z")
    plt.ylabel("E(z) = H(z)/H0")
    plt.title("ΛCDM: E(z)")
    plt.tight_layout()
    plt.savefig(outdir / "lcdm_Ez.png", dpi=200)
    plt.close()

    # H(z)
    plt.figure(figsize=(6, 4))
    plt.plot(z_grid, H_vals)
    plt.xlabel("z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.title("ΛCDM: H(z)")
    plt.tight_layout()
    plt.savefig(outdir / "lcdm_Hz.png", dpi=200)
    plt.close()

    # d_L(z)
    plt.figure(figsize=(6, 4))
    plt.plot(z_grid, dL_vals)
    plt.xlabel("z")
    plt.ylabel("d_L(z) [Mpc]")
    plt.title("ΛCDM: luminosity distance")
    plt.tight_layout()
    plt.savefig(outdir / "lcdm_dL.png", dpi=200)
    plt.close()


def make_psicdm_plots(params_lcdm: LCDMParams, outdir: Path) -> None:
    """
    Produce diagnostic plots for ΨCDM:
    - E(z) for several eps0 values
    - Comparison ΨCDM vs ΛCDM at best "small" eps0
    """
    outdir.mkdir(parents=True, exist_ok=True)

    z_grid = np.linspace(0.0, 2.5, 400)

    # Baseline ΛCDM
    E_l = E_lcdm(z_grid, params_lcdm)

    # Several eps0 values (phenomenological deviations)
    eps_values = [-0.1, -0.05, 0.0, 0.05, 0.1]

    plt.figure(figsize=(6, 4))
    for eps in eps_values:
        p_psi = PsiCDMParams(
            H0=params_lcdm.H0,
            Om0=params_lcdm.Om0,
            eps0=eps,
            n=1.0,
        )
        E_p = E_psicdm(z_grid, p_psi)
        label = rf"$\varepsilon_0 = {eps:+.2f}$"
        plt.plot(z_grid, E_p, label=label)

    plt.xlabel("z")
    plt.ylabel("E(z) = H(z)/H0")
    plt.title("ΨCDM: E(z) for different ε₀")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "psicdm_eps_scan_Ez.png", dpi=200)
    plt.close()

    # ΨCDM vs ΛCDM comparison at a small non-zero eps0
    eps_test = 0.05
    p_psi_test = PsiCDMParams(
        H0=params_lcdm.H0,
        Om0=params_lcdm.Om0,
        eps0=eps_test,
        n=1.0,
    )
    E_p_test = E_psicdm(z_grid, p_psi_test)

    plt.figure(figsize=(6, 4))
    plt.plot(z_grid, E_l, label="ΛCDM")
    plt.plot(z_grid, E_p_test, label=rf"ΨCDM, $\varepsilon_0={eps_test:+.2f}$")
    plt.xlabel("z")
    plt.ylabel("E(z)")
    plt.title("ΛCDM vs ΨCDM (E(z))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "psicdm_vs_lcdm_Ez.png", dpi=200)
    plt.close()


def main():
    project_root = Path(__file__).resolve().parents[1]
    fig_dir = project_root / "results" / "figures" / "model_checks"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Baseline LCDM parameters (can be adjusted later if needed)
    lcdm_params = LCDMParams(H0=70.0, Om0=0.3)

    print("=== Checking ΛCDM model ===")
    check_lcdm_basic(lcdm_params)
    make_lcdm_plots(lcdm_params, fig_dir)

    print("=== Checking ΨCDM limit to ΛCDM (eps0 = 0) ===")
    check_psicdm_limit_to_lcdm(lcdm_params)

    print("=== Producing ΨCDM diagnostic plots ===")
    make_psicdm_plots(lcdm_params, fig_dir)

    print("All model checks finished successfully.")
    print("Figures saved to:", fig_dir)


if __name__ == "__main__":
    main()
