# cosmology/models/psicdm_params.py

from dataclasses import dataclass

@dataclass
class PsiCDMParams:
    """
    The simplest phenomenological parameterization of ΨCDM:
    H^2(z) = H0^2 * [Ω_m (1+z)^3 + (1-Ω_m) + (1-Ω_m)*ε0 (1+z)^n] / [1 + (1-Ω_m)*ε0]
    When ε0 = 0 → ΛCDM.
    """
    H0: float = 70.0
    Om0: float = 0.3
    eps0: float = 0.0
    n: float = 1.0
    rd: float = 147.0   # use same fiducial BAO sound horizon
