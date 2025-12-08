#!/usr/bin/env python3

"""
Automated pipeline runner for Psi-Continuum v2.
Runs all analysis scripts in correct scientific order.

Creates:
    results/logs/run_all.log   — full output of all steps
    results/logs/<script>.log  — per-script logs

Stops on critical errors and prints a summary.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ======================================================
# Helper functions
# ======================================================

def run_step(name: str, cmd: list[str], log_dir: Path, master_log):
    """
    Run a single analysis step.
    Save stdout+stderr into results/logs/<name>.log
    Also mirror output into run_all.log
    """
    print(f"\n=== Running: {name} ===")
    master_log.write(f"\n=== Running: {name} ===\n")

    logfile = log_dir / f"{name}.log"
    with logfile.open("w") as log:

        header = (
            f"=== {name} ===\n"
            f"Command: {' '.join(cmd)}\n"
            f"Started: {datetime.now()}\n\n"
        )
        log.write(header)
        master_log.write(header)

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            log.write(result.stdout)
            master_log.write(result.stdout)
            print(result.stdout)

            if result.returncode != 0:
                err_msg = f"ERROR: {name} failed with exit code {result.returncode}"
                print(err_msg)
                master_log.write(err_msg + "\n")
                print(f"See log: {logfile}")
                sys.exit(result.returncode)

        except Exception as e:
            msg = f"EXCEPTION while running {name}: {e}"
            print(msg)
            master_log.write(msg + "\n")
            print(f"See log: {logfile}")
            sys.exit(1)

        footer = f"\nFinished: {datetime.now()}\n{'='*40}\n"
        log.write(footer)
        master_log.write(footer)

    print(f"✓ Done: {name}\nLog saved to: {logfile}\n")


# ======================================================
# Main pipeline
# ======================================================

def main():
    root = Path(__file__).resolve().parent
    analysis = root / "psi_continuum_v2" / "analysis"
    pkg_root = root / "psi_continuum_v2"

    # logs directory
    log_dir = root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    master_log_file = log_dir / "run_all.log"
    master_log = master_log_file.open("w")

    master_log.write("Psi-Continuum v2 — FULL PIPELINE RUN\n")
    master_log.write(f"Started: {datetime.now()}\n")
    master_log.write("=" * 60 + "\n")

    print("=========================================")
    print("   Psi-Continuum v2 — FULL PIPELINE RUN   ")
    print("=========================================")

    steps = [
        ("check_data",                ["python3", str(pkg_root / "check_data.py")]),
        ("check_models",               ["python3", str(analysis / "check_models.py")]),

        # New data validation steps
        ("check_bao_dr12_data",        ["python3", str(analysis / "check_bao_dr12_data.py")]),
        ("check_desi_dr2_data",        ["python3", str(analysis / "check_desi_dr2_data.py")]),

        # SN
        ("sn_test_lcdm_pplus",         ["python3", str(analysis / "sn_test_lcdm_pplus_simple.py")]),
        ("sn_test_psicdm_pplus",       ["python3", str(analysis / "sn_test_psicdm_pplus.py")]),

        # H(z)
        ("hz_test_psicdm",             ["python3", str(analysis / "hz_test_psicdm.py")]),

        # BAO
        ("bao_desi_test",              ["python3", str(analysis / "bao_desi_dr2_test.py")]),

        # Combined likelihood
        ("joint_fit_psicdm",           ["python3", str(analysis / "joint_fit_psicdm.py")]),
        ("scan_eps_psicdm",            ["python3", str(analysis / "scan_eps_psicdm.py")]),
        ("eps_best_joint_test",        ["python3", str(analysis / "eps_best_joint_test.py")]),

        # Final publication-ready figures
        ("make_publication_plots", ["python3", str(analysis / "make_publication_plots.py")]),
    ]

    for name, cmd in steps:
        run_step(name, cmd, log_dir, master_log)

    summary = (
        "\n=========================================\n"
        "     ALL ANALYSIS SCRIPTS COMPLETED       \n"
        "=========================================\n"
        f"Logs saved in: {log_dir}\n"
    )

    print(summary)
    master_log.write(summary)
    master_log.close()


if __name__ == "__main__":
    main()
