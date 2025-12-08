# cosmology/background/psicdm.py

import numpy as np
from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams
from psi_continuum_v2.cosmology.constants import C_LIGHT


def E_psicdm(z: np.ndarray, params: PsiCDMParams) -> np.ndarray:
    """
    The simplest phenomenological Ψ-deformation of ΛCDM:

    E^2(z) = [ Ω_m (1+z)^3 + (1-Ω_m) + (1-Ω_m)*eps0*(1+z)^n ] /
             [ 1 + (1-Ω_m)*eps0 ]

    Normalized so that E(0) = 1, and reduces to ΛCDM when eps0 = 0.
    """
    z = np.asarray(z, dtype=float)
    Om0 = params.Om0
    eps0 = params.eps0
    n = params.n

    base = Om0 * (1.0 + z) ** 3 + (1.0 - Om0)
    psi_term = (1.0 - Om0) * eps0 * (1.0 + z) ** n

    norm0 = 1.0 + (1.0 - Om0) * eps0  # ensures E(0)=1

    E2 = (base + psi_term) / norm0
    return np.sqrt(E2)


def H_psicdm(z: np.ndarray, params: PsiCDMParams) -> np.ndarray:
    """
    H(z) in km/s/Mpc for ΨCDM.
    """
    return params.H0 * E_psicdm(z, params)


def dL_psicdm(z: np.ndarray, params: PsiCDMParams, nz: int = 4000) -> np.ndarray:
    """
    Luminosity distance d_L(z) in Mpc for ΨCDM via numerical integral:

        d_c(z) = c/H0 ∫ dz'/E(z')
        d_L = (1 + z) * d_c
    """
    z = np.asarray(z, dtype=float)
    z_max = float(np.max(z))
    if z_max <= 0.0:
        return np.zeros_like(z)

    z_grid = np.linspace(0.0, z_max, nz)
    dz = z_grid[1] - z_grid[0]

    inv_E = 1.0 / E_psicdm(z_grid, params)

    # Cumulative trapezoidal integration
    I = np.zeros_like(z_grid)
    for i in range(1, len(z_grid)):
        I[i] = I[i - 1] + 0.5 * (inv_E[i - 1] + inv_E[i]) * dz

    # d_c(z)
    d_c_grid = (C_LIGHT / params.H0) * I
    d_c = np.interp(z, z_grid, d_c_grid)

    return (1.0 + z) * d_c


# --- BAO distances for ΨCDM ---

def DM_psicdm(z: np.ndarray, params: PsiCDMParams) -> np.ndarray:
    """
    Transverse comoving distance for ΨCDM:
        DM(z) = d_L(z) / (1 + z)
    """
    z = np.asarray(z, dtype=float)
    return dL_psicdm(z, params) / (1.0 + z)


def DH_psicdm(z: np.ndarray, params: PsiCDMParams) -> np.ndarray:
    """
    Radial BAO distance:
        DH(z) = c / H(z)
    """
    z = np.asarray(z, dtype=float)
    return C_LIGHT / H_psicdm(z, params)
