# data_manager/download.py

import argparse
import urllib.request
from pathlib import Path
from psi_continuum_v2.utils import get_data_root
from psi_continuum_v2.data_manager.sources import (
    GITHUB_SOURCES,
    OFFICIAL_SOURCES,
)


def download_file(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  → Downloading:\n      {url}")
        print(f"    → Saving to: {target}")
        urllib.request.urlretrieve(url, target)
        return True
    except Exception as e:
        print(f" !!! ERROR downloading {url}\n     {e}")
        return False


def download_dataset(source_dict: dict, data_root: Path) -> bool:
    all_ok = True
    for folder, files in source_dict.items():
        folder_path = data_root / folder
        for fname, url in files.items():
            ok = download_file(url, folder_path / fname)
            if not ok:
                all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download Psi-Continuum v2 datasets."
    )
    parser.add_argument(
        "--source",
        choices=["github", "official"],
        default="github",
        help="Select data source (default: github).",
    )
    args = parser.parse_args()

    print("Psi-Continuum v2 — Data Downloader")
    print("----------------------------------")

    # allow creation of new data directory
    data_root = get_data_root(must_exist=False)

    # ensure directory exists
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"Source: {args.source.capitalize()}")
    print(f"\nData directory: {data_root}\n")

    source_dict = GITHUB_SOURCES if args.source == "github" else OFFICIAL_SOURCES

    ok = download_dataset(source_dict, data_root)

    if ok:
        print("\nAll datasets downloaded successfully!")
        print("You may now run:")
        print("   psi-check-data")
        print("   psi-run-all")
    else:
        print("\nSome files FAILED to download.")
