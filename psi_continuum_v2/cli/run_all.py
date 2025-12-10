#!/usr/bin/env python3

"""
Automated pipeline runner for Psi-Continuum v2.
Runs all analysis scripts in correct scientific order.

Creates:
    results/logs/run_all.log
    results/logs/<script>.log

Results directory is always created NEXT TO data/, not inside cli/.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

from psi_continuum_v2.utils import (
    get_data_root,
    get_results_root,
)


# ======================================================
# Helper functions
# ======================================================

def run_step(name: str, cmd: list[str], log_dir: Path, master_log):
    """Run a single analysis step and store logs."""
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
                err = f"ERROR: {name} failed with exit code {result.returncode}"
                print(err)
                master_log.write(err + "\n")
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

    # Determine real project root (2 levels above cli/)
    root = Path(__file__).resolve().parents[2]

    analysis = root / "psi_continuum_v2" / "analysis"
    pkg_root = root / "psi_continuum_v2"

    # Compute correct locations
    data_root = get_data_root()
    results_root = get_results_root()

    log_dir = results_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    master_log_file = log_dir / "run_all.log"
    master_log = master_log_file.open("w")

    master_log.write("Psi-Continuum v2 — FULL PIPELINE RUN\n")
    master_log.write(f"Started: {datetime.now()}\n")
    master_log.write("=" * 60 + "\n")

    print("=========================================")
    print("   Psi-Continuum v2 — FULL PIPELINE RUN   ")
    print("=========================================")
    print(f"Data root   : {data_root}")
    print(f"Results root: {results_root}\n")

    steps = [
        ("check_data",                ["python3", "-m", "psi_continuum_v2.check_data"]),
        ("check_models",              ["python3", "-m", "psi_continuum_v2.analysis.check.check_models"]),
        ("check_bao_dr12_data",       ["python3", "-m", "psi_continuum_v2.analysis.check.check_bao_dr12_data"]),
        ("check_desi_dr2_data",       ["python3", "-m", "psi_continuum_v2.analysis.check.check_desi_dr2_data"]),

        ("sn_test_lcdm_pplus",        ["python3", "-m", "psi_continuum_v2.analysis.tests.sn_test_lcdm_pplus_simple"]),
        ("sn_test_psicdm_pplus",      ["python3", "-m", "psi_continuum_v2.analysis.tests.sn_test_psicdm_pplus"]),

        ("hz_test_psicdm",            ["python3", "-m", "psi_continuum_v2.analysis.tests.hz_test_psicdm"]),
        ("bao_desi_test",             ["python3", "-m", "psi_continuum_v2.analysis.tests.bao_desi_dr2_test"]),

        ("joint_fit_psicdm",          ["python3", "-m", "psi_continuum_v2.analysis.pipeline.joint_fit_psicdm"]),
        ("scan_eps_psicdm",           ["python3", "-m", "psi_continuum_v2.analysis.scans.scan_eps_psicdm"]),
        ("eps_best_joint_test",       ["python3", "-m", "psi_continuum_v2.analysis.tests.eps_best_joint_test"]),

        ("make_publication_plots",    ["python3", "-m", "psi_continuum_v2.analysis.plots.make_publication_plots"]),
    ]

    for name, cmd in steps:
        run_step(name, cmd, log_dir, master_log)

    summary = (
        "\n=========================================\n"
        "     ALL ANALYSIS SCRIPTS COMPLETED       \n"
        "=========================================\n"
        f"Results are stored in: {results_root}\n"
        f"Logs are in: {log_dir}\n"
    )

    print(summary)
    master_log.write(summary)
    master_log.close()


if __name__ == "__main__":
    main()
