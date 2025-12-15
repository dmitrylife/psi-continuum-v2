# utils/paths.py

from pathlib import Path
from typing import List

# Package root: .../site-packages/psi_continuum_v2   or   .../repo_root/psi_continuum_v2
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Candidate locations for the "data/" directory.
# The first existing directory in this list is considered the active data root.
DATA_DIR_CANDIDATES: List[Path] = [
    Path.cwd() / "data",                               # 1) ./data/ (recommended working directory layout)
    Path.home() / ".psi_continuum" / "data",           # 2) ~/.psi_continuum/data/ (optional shared data)
    PACKAGE_ROOT.parent / "data",                      # 3) <repo_root>/data/ (useful in editable/dev installs)
]


def get_data_root(must_exist: bool = True) -> Path:
    """
    Return the root directory where cosmological datasets are stored.

    Search order:
        1) ./data/
        2) ~/.psi_continuum/data/
        3) <repo_root>/data/

    Parameters
    ----------
    must_exist : bool, optional
        If True, raises FileNotFoundError when no existing data root is found.
        If False, returns the first candidate path (even if it does not exist).

    Returns
    -------
    Path
        Path object pointing to the chosen data root.
    """
    for candidate in DATA_DIR_CANDIDATES:
        if candidate.is_dir():
            return candidate

    if not must_exist:
        # Fall back to the first candidate, even if it does not exist yet.
        return DATA_DIR_CANDIDATES[0]

    candidates_str = "\n".join(str(p) for p in DATA_DIR_CANDIDATES)
    raise FileNotFoundError(
        "Could not locate a data/ directory.\n"
        "Create one of the following directories and place the datasets there:\n\n"
        f"{candidates_str}\n"
    )


def get_data_path(*relative_parts: str, must_exist: bool = True) -> Path:
    """
    Return the full path to a file or directory inside the active data root.

    Examples
    --------
    get_data_path("pantheon_plus", "Pantheon+SH0ES.dat")
    get_data_path("bao", "BAO_consensus_covtot_dM_Hz.txt")

    Parameters
    ----------
    *relative_parts : str
        Path components relative to the data root.

    must_exist : bool, optional
        If True, raises FileNotFoundError if the final path does not exist.
        If False, returns the path even if it does not exist yet.

    Returns
    -------
    Path
        Path object pointing to the requested location.
    """
    root = get_data_root(must_exist=must_exist)
    path = root.joinpath(*relative_parts)

    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"Requested data file or directory does not exist:\n"
            f"  {path}\n\n"
            f"Active data root: {root}"
        )

    return path


def get_results_root() -> Path:
    """
    Return the root directory for analysis results.

    By design, results live next to the active data/ directory:

        <work_dir>/data/    →  <work_dir>/results/

    The directory is created automatically if it does not exist.

    Returns
    -------
    Path
        Path object pointing to the results root.
    """
    data_root = get_data_root(must_exist=True)
    results_root = data_root.parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def get_results_path(*relative_parts: str, create_parents: bool = True) -> Path:
    """
    Return a path inside the results/ directory (next to data/).

    Examples
    --------
    get_results_path("figures", "data_checks")
    get_results_path("tables")
    get_results_path("logs", "hz_test.log")

    Parameters
    ----------
    *relative_parts : str
        Path components relative to the results root.

    create_parents : bool, optional
        If True, all parent directories are created automatically.

    Returns
    -------
    Path
        Path object pointing to the requested results location.
    """
    root = get_results_root()
    path = root.joinpath(*relative_parts)

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def data_root_exists() -> bool:
    """
    Safe check whether any valid data/ directory exists.
    Does NOT raise exceptions.
    """
    for path in DATA_DIR_CANDIDATES:
        if path.is_dir():
            return True
    return False
