# cosmology/background/__init__.py

"""
Background expansion functions for ΛCDM and ΨCDM models.

Contains:
- E(z), H(z) for ΛCDM and ΨCDM
- d_L(z) (luminosity distance) via numerical integration
"""

from psi_continuum_v2.cosmology.constants import C_LIGHT

from psi_continuum_v2.cosmology.background.lcdm import (
    E_lcdm,
    H_lcdm,
    dL_lcdm,
    mu_from_dL,
)

from psi_continuum_v2.cosmology.background.psicdm import (
    E_psicdm,
    H_psicdm,
    dL_psicdm,
)

__all__ = [
    "C_LIGHT",

    # LCDM
    "E_lcdm",
    "H_lcdm",
    "dL_lcdm",
    "mu_from_dL",

    # ΨCDM
    "E_psicdm",
    "H_psicdm",
    "dL_psicdm",
]
