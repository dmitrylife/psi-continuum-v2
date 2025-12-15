# tests/test_check_data.py

import subprocess
import sys
from pathlib import Path


def test_check_data_runs(tmp_path):
    """
    Smoke-test for psi-check-data:
    Ensures CLI runs without crashing and prints expected structure.
    Works with or without real datasets.
    """

    # Run inside a temporary directory to avoid interference
    cwd = tmp_path

    result = subprocess.run(
        [sys.executable, "-m", "psi_continuum_v2.check_data"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    out = result.stdout
    err = result.stderr

    # 1. CLI should exit normally (0)
    assert result.returncode == 0, f"Non-zero exit code: {result.returncode}"

    # 2. Basic header should be present
    assert "Psi-Continuum v2 — Data Check" in out

    # 3. Script should report entries for datasets
    assert "Pantheon+" in out
    assert "H(z)" in out
    assert "BAO" in out

    # 4. Status indicators must appear
    assert ("OK" in out) or ("MISSING" in out)

    # 5. Script should NOT crash
    assert "Traceback" not in out
    assert err.strip() == ""  # no errors printed to stderr
