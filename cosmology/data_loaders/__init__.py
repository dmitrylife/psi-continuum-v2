# cosmology/data_loaders/__init__.py

"""
Data loading utilities for Pantheon+, H(z), and general SN covariance matrices.
"""

# Direct imports — explicit and stable
from .pantheonplus_loader import load_pantheonplus_hf
from .hz_loader import load_hz_compilation
from .covariance_loader import load_sn_covariance

# Optional: validators (if they exist)
try:
    from .validators import (
        validate_pantheonplus_dataset,
        validate_hz_dataset,
    )
except ImportError:
    # allow missing validators gracefully
    pass

__all__ = [
    "load_pantheonplus_hf",
    "load_hz_compilation",
    "load_sn_covariance",
    "validate_pantheonplus_dataset",
    "validate_hz_dataset",
]
