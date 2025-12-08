# tests/test_check_data.py

import subprocess
import sys


def test_check_data_runs():
    """
    Ensure that the CLI tool 'psi-check-data' runs correctly in any environment:
    - with real data (OK)
    - without data (MISSING)
    """

    result = subprocess.run(
        [sys.executable, "-m", "psi_continuum_v2.check_data"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    out = result.stdout + result.stderr

    # Fundamental markers
    assert "Psi-Continuum v2 — Data Check" in out
    assert "Pantheon+" in out
    assert "BAO" in out

    # Script must show at least one status (either OK or MISSING)
    assert ("OK" in out) or ("MISSING" in out)

    # Script must not crash
    assert "Traceback" not in out
