#!/usr/bin/env python3

"""
Example: compute SN distance moduli for ΛCDM and ΨCDM.
"""

from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams
from psi_continuum_v2.cosmology.background.lcdm import dL_lcdm
from psi_continuum_v2.cosmology.background.psicdm import dL_psicdm
import numpy as np


def main():
    z = np.array([0.01, 0.1, 0.5, 1.0])

    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    psi = PsiCDMParams(H0=70.0, Om0=0.3, eps0=0.05, n=1.0)

    print("\n=== Example SN luminosity distances ===")
    print("z =", z)

    print("\nΛCDM dL =", dL_lcdm(z, lcdm))
    print("ΨCDM  dL =", dL_psicdm(z, psi))


if __name__ == "__main__":
    main()
