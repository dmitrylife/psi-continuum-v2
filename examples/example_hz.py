#!/usr/bin/env python3

"""
Example: compute H(z) for ΛCDM and ΨCDM.
"""

from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams
from psi_continuum_v2.cosmology.background.lcdm import H_lcdm
from psi_continuum_v2.cosmology.background.psicdm import H_psicdm
import numpy as np


def main():
    z = np.linspace(0, 2, 5)

    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    psi = PsiCDMParams(H0=70.0, Om0=0.3, eps0=0.05, n=1.0)

    print("\n=== H(z) comparison ===")
    print("z =", z)

    print("\nΛCDM:", H_lcdm(z, lcdm))
    print("ΨCDM:", H_psicdm(z, psi))


if __name__ == "__main__":
    main()
