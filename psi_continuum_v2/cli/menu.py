#!/usr/bin/env python3

from psi_continuum_v2.utils.paths import (
    data_root_exists,
    get_data_root,
)
import subprocess
import webbrowser


def menu():
    print("=========================================")
    print("       Psi-Continuum v2 — CLI Menu        ")
    print("=========================================\n")

    has_data = data_root_exists()

    print(f"Data status: {'OK' if has_data else 'MISSING'}\n")

    print("  1) Download required datasets")

    if has_data:
        print("  2) Check datasets")
        print("  3) Run full analysis pipeline")
    else:
        print("  2) Check datasets   [unavailable: no data]")
        print("  3) Run full analysis pipeline   [unavailable: no data]")

    print("  4) Open documentation")
    print("  5) Show project paths")
    print("  6) Exit\n")

    choice = input("Enter choice (1–6): ").strip()

    # -------------------------------------------------
    # 1 — DOWNLOAD DATA
    # -------------------------------------------------
    if choice == "1":
        subprocess.run(["psi-download-data"])
        return

    # -------------------------------------------------
    # 2 — CHECK DATA
    # -------------------------------------------------
    if choice == "2":
        if not has_data:
            print("\nERROR: No data found. Run option 1 first.\n")
            return
        subprocess.run(["psi-check-data"])
        return

    # -------------------------------------------------
    # 3 — RUN PIPELINE
    # -------------------------------------------------
    if choice == "3":
        if not has_data:
            print("\nERROR: No data found. Run option 1 first.\n")
            return
        subprocess.run(["psi-run-all"])
        return

    # -------------------------------------------------
    # 4 — DOCUMENTATION
    # -------------------------------------------------
    if choice == "4":
        webbrowser.open("https://psi-continuum.org/docs/v2/index.html")
        return

    # -------------------------------------------------
    # 5 — SHOW PATHS
    # -------------------------------------------------
    if choice == "5":
        print("\n--- Project paths ---")
        print("Data root:")
        if has_data:
            print(f"  {get_data_root()}")
        else:
            print("  <not available: no data>")
        print("----------------------\n")
        return

    # -------------------------------------------------
    # 6 — EXIT
    # -------------------------------------------------
    if choice == "6":
        return

    print("Invalid selection.")
