# cosmology/data_loaders/__init__.py

"""
Data loading utilities for Pantheon+, H(z), BAO, and covariance matrices.
"""

from psi_continuum_v2.cosmology.data_loaders.validators import require_file

from psi_continuum_v2.cosmology.data_loaders.pantheonplus_loader import load_pantheonplus_hf
from psi_continuum_v2.cosmology.data_loaders.hz_loader import load_hz_compilation
from psi_continuum_v2.cosmology.data_loaders.covariance_loader import load_sn_covariance

# Optional validators
try:
    from psi_continuum_v2.cosmology.data_loaders.validators import (
        validate_pantheonplus_dataset,
        validate_hz_dataset,
    )
except ImportError:
    validate_pantheonplus_dataset = None
    validate_hz_dataset = None


__all__ = [
    "load_pantheonplus_hf",
    "load_hz_compilation",
    "load_sn_covariance",
    "require_file",
    "validate_pantheonplus_dataset",
    "validate_hz_dataset",
]
