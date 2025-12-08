# cosmology/likelihoods/joint_likelihood.py

"""
Joint likelihood utilities for combining SN, H(z), BAO χ², and DESI χ².

This module does NOT compute individual χ² values itself.
Instead, it takes precomputed χ² contributions from:
    - supernovae (SN)
    - expansion-rate data H(z)
    - BAO measurements (SDSS)
    - DESI DR2 BAO

and returns a structured result with:
    - per-dataset χ² and degrees of freedom (dof)
    - total χ² and total dof

The goal is to keep this module as a simple, extensible "glue layer"
between separate likelihood components.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class JointChi2Result:
    """
    Container for joint χ² values from several datasets.

    Attributes:
        chi2_sn     : χ² from supernovae
        dof_sn      : dof for SN

        chi2_hz     : χ² from H(z)
        dof_hz      : dof for H(z)

        chi2_bao    : χ² from SDSS BAO
        dof_bao     : dof for SDSS BAO

        chi2_desi   : χ² from DESI DR2 BAO
        dof_desi    : dof for DESI DR2
    """

    chi2_sn: Optional[float] = None
    dof_sn: Optional[int] = None

    chi2_hz: Optional[float] = None
    dof_hz: Optional[int] = None

    chi2_bao: Optional[float] = None
    dof_bao: Optional[int] = None

    # --- NEW: DESI DR2 BAO support ---
    chi2_desi: Optional[float] = None
    dof_desi: Optional[int] = None

    @property
    def chi2_total(self) -> float:
        """
        Total χ² from all provided datasets.
        Datasets with chi2 = None are ignored.
        """
        total = 0.0
        if self.chi2_sn is not None:
            total += self.chi2_sn
        if self.chi2_hz is not None:
            total += self.chi2_hz
        if self.chi2_bao is not None:
            total += self.chi2_bao
        if self.chi2_desi is not None:
            total += self.chi2_desi
        return total

    @property
    def dof_total(self) -> int:
        """
        Total degrees of freedom (sum over all datasets with non-None dof).
        """
        total = 0
        if self.dof_sn is not None:
            total += self.dof_sn
        if self.dof_hz is not None:
            total += self.dof_hz
        if self.dof_bao is not None:
            total += self.dof_bao
        if self.dof_desi is not None:
            total += self.dof_desi
        return total

    @property
    def chi2_reduced(self) -> Optional[float]:
        """
        Reduced χ² = χ²_total / dof_total, if dof_total > 0.
        Returns None if dof_total == 0.
        """
        dof = self.dof_total
        if dof <= 0:
            return None
        return self.chi2_total / dof


def build_joint_chi2(
    chi2_sn: Optional[float] = None,
    dof_sn: Optional[int] = None,

    chi2_hz: Optional[float] = None,
    dof_hz: Optional[int] = None,

    chi2_bao: Optional[float] = None,
    dof_bao: Optional[int] = None,

    # --- NEW DESI fields ---
    chi2_desi: Optional[float] = None,
    dof_desi: Optional[int] = None,
) -> JointChi2Result:
    """
    Helper function to create a JointChi2Result in one line.
    Supports SN + H(z) + SDSS BAO + DESI DR2 BAO.
    """

    return JointChi2Result(
        chi2_sn=chi2_sn,
        dof_sn=dof_sn,

        chi2_hz=chi2_hz,
        dof_hz=dof_hz,

        chi2_bao=chi2_bao,
        dof_bao=dof_bao,

        # NEW
        chi2_desi=chi2_desi,
        dof_desi=dof_desi,
    )
