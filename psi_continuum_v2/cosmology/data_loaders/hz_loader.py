# cosmology/data_loaders/hz_loader.py

from pathlib import Path
from typing import Dict, Any
import numpy as np

from psi_continuum_v2.cosmology.data_loaders.validators import require_file


def load_hz_compilation(
    base_dir: Path | str | None = None,
    filename: str = "HZ_compilation.csv",
) -> Dict[str, Any]:
    """
    Load the H(z) compilation.

    Expected CSV format:
        z, Hz, sigma_Hz, ...

    If headers are missing or columns have unexpected names,
    the loader falls back to using the first three columns.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2] / "data" / "hz"
    else:
        base_dir = Path(base_dir)

    # Human-readable error if file missing
    data_file = require_file(
        base_dir / filename,
        instructions=(
            f"H(z) file '{filename}' was not found.\n"
            "Please download the H(z) compilation and place it into:\n"
            f"{base_dir}"
        ),
    )

    # Detect header
    with open(data_file, "r", encoding="utf-8") as f:
        header_line = f.readline()
    has_header = any(c.isalpha() for c in header_line)

    if has_header:
        # Read named columns
        data = np.genfromtxt(
            data_file,
            delimiter=",",
            names=True,
            dtype=None,
            encoding=None,
        )
        names = [n.lower() for n in data.dtype.names]

        # Helper finder
        def _find(candidates):
            for cand in candidates:
                if cand.lower() in names:
                    idx = names.index(cand.lower())
                    return data[data.dtype.names[idx]]
            raise KeyError

        try:
            z = _find(["z", "z_hd", "z_hubble", "redshift"])
            Hz = _find(["Hz", "H", "H_z"])
            sigma_Hz = _find(["sigma_Hz", "err_Hz", "sigmaH", "sigma_h"])
        except KeyError:
            # fallback: first 3 columns
            arr = np.loadtxt(data_file, delimiter=",", skiprows=1, unpack=True)
            if arr.shape[0] < 3:
                raise ValueError("H(z) file must contain at least 3 numeric columns.")
            z, Hz, sigma_Hz = arr[:3]

    else:
        # No header → simple numeric table
        z, Hz, sigma_Hz = np.loadtxt(data_file, delimiter=",", unpack=True)

    return {
        "z": np.asarray(z, dtype=float),
        "Hz": np.asarray(Hz, dtype=float),
        "sigma_Hz": np.asarray(sigma_Hz, dtype=float),
        "N": len(z),
    }
