# cosmology/background/lcdm.py

import numpy as np
from cosmology.models.lcdm_params import LCDMParams

_c = 299792.458  # speed of light, km/s


def E_lcdm(z: np.ndarray, params: LCDMParams) -> np.ndarray:
    """
    Dimensionless H(z)/H0 for flat ΛCDM.
    """
    z = np.asarray(z, dtype=float)
    Om0 = params.Om0
    return np.sqrt(Om0 * (1.0 + z) ** 3 + (1.0 - Om0))


def H_lcdm(z: np.ndarray, params: LCDMParams) -> np.ndarray:
    """
    Physical H(z) in km/s/Mpc.
    """
    return params.H0 * E_lcdm(z, params)


def dL_lcdm(z: np.ndarray, params: LCDMParams, nz: int = 4000) -> np.ndarray:
    """
    Purely numerical calculation of the luminosity d_L(z) in Mpc
    via the integral d_c(z) = c/H0 ∫ dz'/E(z') and d_L = (1+z)*d_c.

    Here we create a single z-grid and construct an interpolation
    (rough, but fast and self-contained, without SciPy).
    """
    z = np.asarray(z, dtype=float)
    z_max = float(np.max(z))
    if z_max <= 0.0:
        return np.zeros_like(z)

    # Grid for integration
    z_grid = np.linspace(0.0, z_max, nz)
    dz = z_grid[1] - z_grid[0]

    inv_E = 1.0 / E_lcdm(z_grid, params)

    # Cumulative integral over trapezoids: I(z_i) ~ ∫0^z_i dz'/E(z')
    # I_0 = 0; I_i = I_{i-1} + 0.5 * (f_{i-1} + f_i) * dz
    I = np.zeros_like(z_grid)
    for i in range(1, len(z_grid)):
        I[i] = I[i - 1] + 0.5 * (inv_E[i - 1] + inv_E[i]) * dz

    # d_c in Mpc
    d_c_grid = (_c / params.H0) * I  # c/H0 * I(z)

    # Interpolation of d_c(z) for arbitrary z
    d_c = np.interp(z, z_grid, d_c_grid)

    # d_L = (1+z)*d_c
    return (1.0 + z) * d_c


def mu_from_dL(dL_mpc: np.ndarray) -> np.ndarray:
    """
    The distance modulus μ = 5 log10(d_L / 10 pc), d_L – in Mpc.
    """
    dL_pc = np.asarray(dL_mpc) * 1.0e6  # Мпк → пк
    return 5.0 * (np.log10(dL_pc) - 1.0)


# --- BAO distances for ΛCDM ---

def DM_lcdm(z: np.ndarray, params: LCDMParams) -> np.ndarray:
    """
    Transverse comoving distance:
        DM(z) = d_L(z) / (1 + z)
    """
    z = np.asarray(z, dtype=float)
    return dL_lcdm(z, params) / (1.0 + z)


def DH_lcdm(z: np.ndarray, params: LCDMParams) -> np.ndarray:
    """
    Radial BAO distance:
        DH(z) = c / H(z)
    """
    z = np.asarray(z, dtype=float)
    return _c / H_lcdm(z, params)
