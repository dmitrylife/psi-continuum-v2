# utils/style.py

from pathlib import Path
from importlib import resources
import matplotlib.pyplot as plt


# Module path where the Matplotlib style file lives.
# This corresponds to: psi_continuum_v2/analysis/styles/psi_style.mplstyle
STYLE_PACKAGE = "psi_continuum_v2.analysis.styles"
STYLE_FILENAME = "psi_style.mplstyle"


def get_psi_style_path() -> Path:
    """
    Return the absolute path to the Psi-Continuum Matplotlib style file.

    The style is resolved using importlib.resources so that it works both:
      - in a source checkout (editable install)
      - in a wheel/sdist installed via pip
    """
    style_path = resources.files(STYLE_PACKAGE) / STYLE_FILENAME
    return Path(style_path)


def use_psi_style() -> None:
    """
    Activate the Psi-Continuum Matplotlib style globally.

    This is a convenience wrapper around plt.style.use(get_psi_style_path()).
    """
    style_path = get_psi_style_path()
    plt.style.use(style_path)
