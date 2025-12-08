# analysis/check_data.py

"""
Psi-Continuum v2 — Data Integrity Checker (CLI)

This tool checks whether required datasets are present in the *current working directory*.

IMPORTANT:
 - When installed via pip, data/ must be placed next to where you run commands.
 - We intentionally DO NOT search inside site-packages.
"""

from pathlib import Path

# ----------- ANSI COLORS -----------
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def status(ok: bool) -> str:
    return f"{GREEN}OK{RESET}" if ok else f"{RED}MISSING{RESET}"


def check_pantheon_plus(root: Path):
    dir_ = root / "data" / "pantheon_plus"
    required = [
        "Pantheon+SH0ES.dat",
        "Pantheon+SH0ES_STAT+SYS.cov",
    ]
    missing = [f for f in required if not (dir_ / f).exists()]
    return dir_, missing


def check_hz(root: Path):
    dir_ = root / "data" / "hz"
    required = ["HZ_compilation.csv"]
    missing = [f for f in required if not (dir_ / f).exists()]
    return dir_, missing


def check_bao_dr12(root: Path):
    dir_ = root / "data" / "bao"
    required = [
        "sdss_DR12Consensus_bao.dat",
        "BAO_consensus_covtot_dM_Hz.txt",
    ]
    missing = [f for f in required if not (dir_ / f).exists()]
    return dir_, missing


def check_desi_dr2(root: Path):
    dir_ = root / "data" / "desi" / "dr2"
    required = [
        "desi_gaussian_bao_ALL_GCcomb_mean.txt",
        "desi_gaussian_bao_ALL_GCcomb_cov.txt",
    ]
    missing = [f for f in required if not (dir_ / f).exists()]
    return dir_, missing


def main():
    # 🔥 КЛЮЧЕВАЯ СТРОКА: ИСПОЛЬЗУЕМ ТОЛЬКО ТЕКУЩУЮ ПАПКУ, НЕ site-packages!
    root = Path.cwd()

    print("\nPsi-Continuum v2 — Data Check")

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
            print(f"  expected files in: {YELLOW}{dir_}{RESET}")
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
