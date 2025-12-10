"""
Example: compute BAO distance measures for ΛCDM and ΨCDM.
"""

from psi_continuum_v2.cosmology.models.lcdm_params import LCDMParams
from psi_continuum_v2.cosmology.models.psicdm_params import PsiCDMParams
from psi_continuum_v2.cosmology.background.lcdm import DM_lcdm, DH_lcdm
from psi_continuum_v2.cosmology.background.psicdm import DM_psicdm, DH_psicdm

import numpy as np


def main():
    z = np.array([0.38, 0.51, 0.61])

    lcdm = LCDMParams(H0=70.0, Om0=0.3)
    psi  = PsiCDMParams(H0=70.0, Om0=0.3, eps0=0.05, n=1.0)

    print("\n=== BAO distance measures ===")
    print("z =", z)

    print("\nΛCDM:")
    print("DM =", DM_lcdm(z, lcdm))
    print("DH =", DH_lcdm(z, lcdm))

    print("\nΨCDM:")
    print("DM =", DM_psicdm(z, psi))
    print("DH =", DH_psicdm(z, psi))


if __name__ == "__main__":
    main()
