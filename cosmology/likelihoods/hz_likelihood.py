# cosmology/likelihoods/hz_likelihood.py

from typing import Callable, Any

import numpy as np


def chi2_hz(
    hzdata: dict,
    H_model: Callable[[np.ndarray, Any], np.ndarray],
    params: Any,
) -> float:
    """
    Simple diagonal χ² for H(z):

    hzdata: dictionary with keys 'z', 'Hz', 'sigma_Hz'
    H_model: function H_model(z, params) → H(z)
    params: parameters object (LCDMParams, PsiCDMParams, ...)

    χ² = Σ [(H_obs - H_th)^2 / σ_H^2]
    """
    z = np.asarray(hzdata["z"], dtype=float)
    H_obs = np.asarray(hzdata["Hz"], dtype=float)
    sigma_H = np.asarray(hzdata["sigma_Hz"], dtype=float)

    H_th = H_model(z, params)

    if H_th.shape != H_obs.shape:
        raise ValueError("The H_th form does not match H_obs")

    return float(np.sum((H_obs - H_th) ** 2 / sigma_H**2))
