# cosmology/models/lcdm_params.py

from dataclasses import dataclass

@dataclass
class LCDMParams:
    """
    Minimum set of parameters for flat ΛCDM.
    H0 in km/s/Mpc, Om0 = Ω_m0.
    """
    H0: float = 70.0
    Om0: float = 0.3
    rd: float = 147.0   # sound horizon at drag epoch [Mpc]
