# cosmology/data_loaders/hz_loader.py

from pathlib import Path
from typing import Dict, Any

import numpy as np


def load_hz_compilation(
    base_dir: Path | str | None = None,
    filename: str = "HZ_compilation.csv",
) -> Dict[str, Any]:
    """
    Loading H(z) compilation.

    Expected CSV format:
    z, Hz, sigma_Hz, ...

    If there are no headers or the names are different, use the first three columns.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2] / "data" / "hz"
    else:
        base_dir = Path(base_dir)

    data_file = base_dir / filename
    if not data_file.exists():
        raise FileNotFoundError(f"File not found H(z): {data_file}")

    # We try to read with the title
    with open(data_file, "r", encoding="utf-8") as f:
        header_line = f.readline()

    has_header = any(c.isalpha() for c in header_line)

    if has_header:
        data = np.genfromtxt(
            data_file,
            delimiter=",",
            names=True,
            dtype=None,
            encoding=None,
        )
        names = [n.lower() for n in data.dtype.names]

        def _find(name_candidates):
            for cand in name_candidates:
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
                raise ValueError("The H(z) file has less than three columns.")
            z, Hz, sigma_Hz = arr[:3]
    else:
        # No title - simple table
        z, Hz, sigma_Hz = np.loadtxt(data_file, delimiter=",", unpack=True)

    return {
        "z": np.asarray(z, dtype=float),
        "Hz": np.asarray(Hz, dtype=float),
        "sigma_Hz": np.asarray(sigma_Hz, dtype=float),
        "N": len(z),
    }
