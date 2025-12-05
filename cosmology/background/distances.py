# cosmology/background/distances.py

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Any, Union

ArrayLike = Union[float, np.ndarray]

# Speed of light in km/s
C_LIGHT: float = 299792.458


def dL_flat_from_H(
    z_array: ArrayLike,
    H_of_z: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
    params: Dict[str, Any],
) -> np.ndarray:
    """
    Luminosity distance d_L(z) in Mpc for a flat universe:

        d_L(z) = (1 + z) * c ∫_0^z dz' / H(z').

    Numerical integration is performed on a sorted fine grid in z.
    H(z) must be in units of km/s/Mpc, and d_L will be returned in Mpc.

    Parameters
    ----------
    z_array : float or array
        Redshift(s) where d_L should be evaluated.
    H_of_z : callable
        Function H(z, params) → array in km/s/Mpc.
    params : dict
        Dictionary of cosmological parameters to pass to H_of_z.

    Returns
    -------
    dL : ndarray
        Luminosity distance in Mpc at the requested redshifts.
    """
    z_array = np.asarray(z_array, dtype=float)
    z_work = np.clip(z_array, 0.0, None)

    # Handle trivial z ≈ 0 case
    if np.allclose(z_work, 0.0):
        return np.zeros_like(z_array)

    # Sort redshifts to integrate monotonically
    sort_idx = np.argsort(z_work)
    z_sorted = z_work[sort_idx]

    # Evaluate H(z) on the sorted grid
    H_grid = H_of_z(z_sorted, params)  # km/s/Mpc
    H_grid = np.asarray(H_grid, dtype=float)

    if np.any(H_grid <= 0.0) or np.any(~np.isfinite(H_grid)):
        raise ValueError("H(z) contains non-positive or non-finite values.")

    # Trapezoidal integration of ∫ dz / H(z)
    integrand = 1.0 / H_grid
    dz = np.diff(z_sorted)

    local_int = 0.5 * (integrand[:-1] + integrand[1:]) * dz
    integral = np.concatenate([[0.0], np.cumsum(local_int)])  # same length as z_sorted

    # d_L(z) = (1+z) * c * ∫ dz'/H(z')
    dL_sorted = (1.0 + z_sorted) * C_LIGHT * integral  # Mpc

    # Map back to original order
    dL = np.empty_like(dL_sorted)
    dL[sort_idx] = dL_sorted

    return dL


def mu_from_dL(dL_mpc: ArrayLike) -> np.ndarray:
    """
    Distance modulus:

        μ = 5 log10(d_L / Mpc) + 25.

    Parameters
    ----------
    dL_mpc : float or array
        Luminosity distance in Mpc.

    Returns
    -------
    μ : ndarray
        Distance modulus.
    """
    dL_mpc = np.asarray(dL_mpc, dtype=float)
    # Avoid log10(0)
    dL_safe = np.clip(dL_mpc, 1e-6, None)
    return 5.0 * np.log10(dL_safe) + 25.0
