# data_manager/sources.py

"""
Defines URLs for downloading required datasets for Psi-Continuum v2.
"""

# --------------------------------------------------------------
# GitHub mirror (recommended)
# --------------------------------------------------------------
GITHUB_BASE = (
    "https://raw.githubusercontent.com/dmitrylife/psi-continuum-v2/main/data"
)

GITHUB_SOURCES = {
    "pantheon_plus": {
        "Pantheon+SH0ES.dat": f"{GITHUB_BASE}/pantheon_plus/Pantheon+SH0ES.dat",
        "Pantheon+SH0ES_STAT+SYS.cov": f"{GITHUB_BASE}/pantheon_plus/Pantheon+SH0ES_STAT+SYS.cov",
    },
    "hz": {
        "HZ_compilation.csv": f"{GITHUB_BASE}/hz/HZ_compilation.csv",
    },
    "bao": {
        "sdss_DR12Consensus_bao.dat": f"{GITHUB_BASE}/bao/sdss_DR12Consensus_bao.dat",
        "BAO_consensus_covtot_dM_Hz.txt": f"{GITHUB_BASE}/bao/BAO_consensus_covtot_dM_Hz.txt",
    },
    "desi/dr2": {
        "desi_gaussian_bao_ALL_GCcomb_mean.txt":
            f"{GITHUB_BASE}/desi/dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt",
        "desi_gaussian_bao_ALL_GCcomb_cov.txt":
            f"{GITHUB_BASE}/desi/dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt",
    },
}

# --------------------------------------------------------------
# Official sources (not implemented)
# --------------------------------------------------------------
OFFICIAL_SOURCES = {
    "pantheon_plus": {},
    "hz": {},
    "bao": {},
    "desi/dr2": {},
}

# Default
DEFAULT_SOURCE = GITHUB_SOURCES
