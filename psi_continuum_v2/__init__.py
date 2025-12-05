"""
Top-level Python package for the Psi-Continuum v2 project.

Usage:

    import psi_continuum_v2 as psi
    psi.cosmology.background.lcdm.E_lcdm(...)

Internally, the core scientific code lives in the separate
`cosmology` package. Here we just provide a convenient façade
and project-level metadata.
"""

from __future__ import annotations

from importlib import import_module

# Public handle to the underlying cosmology package
cosmology = import_module("cosmology")

# Basic metadata
__all__ = [
    "cosmology",
    "__version__",
]

# Bump this manually when you make significant changes to the public API
__version__ = "0.2.0"

