#!/usr/bin/env python3

"""
Automated pipeline runner for Psi-Continuum v2.
Runs all analysis scripts in correct scientific order.

Creates:
    results/logs/run_all.log  — full output of all steps
    results/logs/<script>.log — per-script logs

Stops on critical errors and prints a summary.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ======================================================
# Helper functions
# ======================================================

def run_step(name: str, cmd: list[str], log_dir: Path):
    """
    Runs a single analysis step.
    Saves stdout+stderr into results/logs/<name>.log
    """
    print(f"\n=== Running: {name} ===")

    logfile = log_dir / f"{name}.log"
    with logfile.open("w") as log:

        log.write(f"=== {name} ===\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write(f"Started: {datetime.now()}\n\n")

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            log.write(result.stdout)
            print(result.stdout)

            if result.returncode != 0:
                print(f"ERROR: {name} failed with exit code {result.returncode}")
                print(f"See log: {logfile}")
                sys.exit(result.returncode)

        except Exception as e:
            print(f"EXCEPTION while running {name}: {e}")
            print(f"See log: {logfile}")
            sys.exit(1)

        log.write(f"\nFinished: {datetime.now()}\n")
        log.write("====================================\n")

    print(f"✓ Done: {name}\nLog saved to: {logfile}\n")


# ======================================================
# Main pipeline
# ======================================================

def main():
    root = Path(__file__).resolve().parent
    analysis = root / "analysis"

    # <-- FIXED: save logs here
    log_dir = root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================")
    print("   Psi-Continuum v2 — FULL PIPELINE RUN   ")
    print("=========================================")

    steps = [
        ("check_models",               ["python3", str(analysis / "check_models.py")]),
        ("sn_test_lcdm_pplus",         ["python3", str(analysis / "sn_test_lcdm_pplus_simple.py")]),
        ("sn_test_psicdm_pplus",       ["python3", str(analysis / "sn_test_psicdm_pplus.py")]),
        ("hz_test_psicdm",             ["python3", str(analysis / "hz_test_psicdm.py")]),
        ("bao_desi_test",              ["python3", str(analysis / "bao_desi_dr2_test.py")]),
        ("joint_fit_psicdm",           ["python3", str(analysis / "joint_fit_psicdm.py")]),
        ("scan_eps_psicdm",            ["python3", str(analysis / "scan_eps_psicdm.py")]),
        ("eps_best_joint_test",        ["python3", str(analysis / "eps_best_joint_test.py")]),
        ("make_all_plots",             ["python3", str(analysis / "make_all_plots.py")]),
    ]

    for name, cmd in steps:
        run_step(name, cmd, log_dir)

    print("\n=========================================")
    print("     ALL ANALYSIS SCRIPTS COMPLETED       ")
    print("=========================================")
    print(f"Logs saved in: {log_dir}\n")


if __name__ == "__main__":
    main()
