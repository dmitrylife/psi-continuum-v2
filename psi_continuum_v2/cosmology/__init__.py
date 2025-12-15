# cosmology/__init__.py

"""
Cosmology-level public API for Psi-Continuum.

Exports:
- C_LIGHT constant
- (in the future: H0 units, fiducial values, etc.)
"""

from psi_continuum_v2.cosmology.constants import C_LIGHT

__all__ = [
    "C_LIGHT",
]
