"""
Psi-Continuum v2 — Data Integrity Checker (CLI)

This tool checks whether required datasets are present in the *current working directory*.

IMPORTANT:
 - The directory where you run this script must contain: data/
 - We do NOT search inside site-packages to avoid confusion.
"""

from pathlib import Path

# ----------- ANSI COLORS -----------
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def status(ok: bool) -> str:
    return f"{GREEN}OK{RESET}" if ok else f"{RED}MISSING{RESET}"


# -------------------------------------------------------------
# Individual dataset checks
# -------------------------------------------------------------

def check_pantheon_plus(root: Path):
    dir_ = root / "data" / "pantheon_plus"
    required = [
        "Pantheon+SH0ES.dat",
        "Pantheon+SH0ES_STAT+SYS.cov",
    ]

    if not dir_.exists():
        return dir_, required   # entire folder missing

    missing = [f for f in required if not (dir_ / f).exists()]
    return dir_, missing


def check_hz(root: Path):
    dir_ = root / "data" / "hz"
    required = ["HZ_compilation.csv"]

    if not dir_.exists():
        return dir_, required

    missing = [f for f in required if not (dir_ / f).exists()]
    return dir_, missing


def check_bao_dr12(root: Path):
    dir_ = root / "data" / "bao"
    required = [
        "sdss_DR12Consensus_bao.dat",
        "BAO_consensus_covtot_dM_Hz.txt",
    ]

    if not dir_.exists():
        return dir_, required

    missing = [f for f in required if not (dir_ / f).exists()]
    return dir_, missing


def check_desi_dr2(root: Path):
    dir_ = root / "data" / "desi" / "dr2"
    required = [
        "desi_gaussian_bao_ALL_GCcomb_mean.txt",
        "desi_gaussian_bao_ALL_GCcomb_cov.txt",
    ]

    if not dir_.exists():
        return dir_, required

    missing = [f for f in required if not (dir_ / f).exists()]
    return dir_, missing


def data_exists(data_root):
    """Check whether all required dataset files are present."""
    required = [
        data_root / "pantheon_plus" / "Pantheon+SH0ES.dat",
        data_root / "pantheon_plus" / "Pantheon+SH0ES_STAT+SYS.cov",
        data_root / "hz" / "HZ_compilation.csv",
        data_root / "bao" / "sdss_DR12Consensus_bao.dat",
        data_root / "bao" / "BAO_consensus_covtot_dM_Hz.txt",
        data_root / "desi" / "dr2" / "desi_gaussian_bao_ALL_GCcomb_mean.txt",
        data_root / "desi" / "dr2" / "desi_gaussian_bao_ALL_GCcomb_cov.txt",
    ]
    return all(p.exists() for p in required)


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    root = Path.cwd()

    print("\nPsi-Continuum v2 — Data Check")
    print(f"Project root: {YELLOW}{root}{RESET}\n")

    checks = [
        ("Pantheon+ SH0ES HF", check_pantheon_plus),
        ("H(z) compilation",   check_hz),
        ("SDSS DR12 BAO",      check_bao_dr12),
        ("DESI DR2 BAO",       check_desi_dr2),
    ]

    total_missing = 0

    for name, func in checks:
        dir_, missing = func(root)
        ok = (len(missing) == 0)

        print(f"{name:<22}: {status(ok)}")

        if not ok:
            total_missing += len(missing)
            print(f"  expected at: {YELLOW}{dir_}{RESET}")

            if not dir_.exists():
                print(f"   {RED}Folder missing entirely.{RESET}")
                print(f"   Please create: {dir_}\n")
                continue

            for f in missing:
                print(f"   - {RED}{f}{RESET}")
            print()

    print("--------------------------------")

    if total_missing == 0:
        print(f"{GREEN}All required datasets are present!{RESET}\n")
    else:
        print(f"{RED}{total_missing} missing file(s).{RESET}")
        print("Download datasets and place them in the indicated directories.\n")


if __name__ == "__main__":
    main()
