# Psi-Continuum v2  

### A Minimal One-Parameter Extension of ΛCDM  
**Author:** Dmitry Vasilevich Klimov
**Status:** Research release (final archived version v0.2.3)

📘 **Documentation (PyPI Package):** https://psi-continuum.org/docs/v2

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Reproducible](https://img.shields.io/badge/Reproducible-Build-blueviolet.svg)
![Status](https://img.shields.io/badge/Status-Final%20Archived-brightgreen.svg)


## Overview

Psi-Continuum (ΨCDM) is a minimal phenomenological extension of ΛCDM that
introduces a single deformation parameter ε₀, modifying the late-time expansion
rate while retaining ΛCDM exactly as ε₀ → 0.
It is designed as a strictly background-level model for late-time expansion, 
without modifying perturbations or early-Universe physics.

The modified Hubble function is:

$$
H_{\Psi}(z) = H_{\Lambda}(z)\left(1 + \frac{\varepsilon_{0}}{1+z}\right)
$$

This parametrisation ensures strict ΛCDM recovery at ε₀ = 0 and suppresses deviations as z → ∞.

A smooth percent-level correction at low redshift:

- reduces to ΛCDM when ε₀ = 0,
- preserves all early-Universe observables (CMB, primordial BAO),
- can be interpreted phenomenologically as an effective macroscopic response term.

---

## Included in This Repository

This repository provides a fully reproducible late-time cosmology pipeline, including:

### Cosmological Models
- ΛCDM background expansion
- ΨCDM background with deformation parameter ε₀
- Distance measures:
  - H(z)
  - E(z)
  - d_L(z)
  - D_M / r_d
  - D_H / r_d
  - D_V / r_d

### Data Loaders
- Pantheon+ High-Fidelity SNe Ia (full covariance χ²)
- 32 cosmic-chronometer H(z) measurements
- SDSS BOSS DR12 consensus BAO
- DESI DR2 Gaussian BAO likelihood

### Likelihoods & Analysis
- Full joint χ²: SN + CC + SDSS DR12 + DESI DR2
- ε₀ scans and Δχ² profiling
- Best-fit extraction and ΛCDM comparison
- Statistical χ² breakdown per dataset
- Publication-ready figure generator:
  - make_publication_plots.py

### Tools
- psi-cli for quick computations and model comparison
- Data-validation utilities
- Publication-ready figures used in the 2025 paper

---

## Main Scientific Results (v2)

Using the combined late-time dataset (Pantheon+ HF, cosmic chronometers,
SDSS DR12 BAO, and DESI DR2 BAO), the ΨCDM model was confronted with
current background-expansion observations.

Individual datasets show different sensitivities to the deformation parameter:
- Pantheon+ HF supernovae provide only weak constraints on ε₀,
- DESI DR2 BAO mildly prefer positive ε₀ values,
- SDSS DR12 BAO and cosmic chronometers remain broadly consistent with ΛCDM.

The combined late-time likelihood exhibits a shallow minimum around:

$$
\varepsilon_{0} \simeq 0.03
$$

corresponding to a smooth, percent-level enhancement of the expansion rate
at low redshift.

The overall improvement relative to ΛCDM is not statistically significant,
indicating that ΨCDM and ΛCDM remain observationally equivalent at current
precision, while allowing controlled percent-level deviations in the
late-time Hubble flow.

---

## Repository Structure

```text
psi-continuum-v2/
├── .github
│   └── workflows
│       └── tests.yml
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── MANIFEST.in
├── paper
│   └── v2_2025-12-10.pdf
├── README.md
├── data
│   ├── bao
│   │   ├── BAO_consensus_covtot_dM_Hz.txt
│   │   └── sdss_DR12Consensus_bao.dat
│   ├── desi
│   │   └── dr2
│   │       ├── desi_gaussian_bao_ALL_GCcomb_cov.txt
│   │       └── desi_gaussian_bao_ALL_GCcomb_mean.txt
│   ├── hz
│   │   └── HZ_compilation.csv
│   └── pantheon_plus
│       ├── Pantheon+SH0ES.dat
│       └── Pantheon+SH0ES_STAT+SYS.cov
├── psi_continuum_v2
│   ├── analysis
│   │   ├── check
│   │   │   ├── check_bao_dr12_data.py
│   │   │   ├── check_desi_dr2_data.py
│   │   │   ├── check_hz_data.py
│   │   │   ├── check_models.py
│   │   │   └── check_pantheonplus_data.py
│   │   ├── pipeline
│   │   │   └── joint_fit_psicdm.py
│   │   ├── plots
│   │   │   └── make_publication_plots.py
│   │   ├── scans
│   │   │   └── scan_eps_psicdm.py
│   │   ├── styles
│   │   │   └── psi_style.mplstyle
│   │   └── tests
│   │       ├── bao_desi_dr2_test.py
│   │       ├── eps_best_joint_test.py
│   │       ├── hz_test_psicdm.py
│   │       ├── sn_test_lcdm_pplus_simple.py
│   │       └── sn_test_psicdm_pplus.py
│   ├── check_data.py
│   ├── cli
│   │   ├── menu.py
│   │   └── run_all.py
│   ├── cosmology
│   │   ├── background
│   │   │   ├── distances.py
│   │   │   ├── lcdm.py
│   │   │   └── psicdm.py
│   │   ├── constants.py
│   │   ├── data_loaders
│   │   │   ├── bao_loader.py
│   │   │   ├── covariance_loader.py
│   │   │   ├── desi_loader.py
│   │   │   ├── hz_loader.py
│   │   │   ├── pantheonplus_loader.py
│   │   │   └── validators.py
│   │   ├── likelihoods
│   │   │   ├── bao_likelihood.py
│   │   │   ├── hz_likelihood.py
│   │   │   ├── joint_likelihood.py
│   │   │   └── sn_likelihood.py
│   │   └── models
│   │       ├── lcdm_params.py
│   │       └── psicdm_params.py
│   ├── data_manager
│   │   ├── download.py
│   │   └── sources.py
│   ├── tests
│   │   └── test_check_data.py
│   └── utils
│       ├── paths.py
│       └── style.py
└──pyproject.toml
```

These directories are included in the Zenodo software archive as a
frozen snapshot of the results used in the publication.

- `results/figures/publication/main_figures/`
- `results/figures/publication/appendix/`
- `results/figures/bao/`

---

## Installation

Installation options:

- Installation via PyPI (recommended)
- Installation from source (clone repository)

### Installation via PyPI (recommended)

You can install Psi-Continuum v2 directly from PyPI:

```bash
pip install psi-continuum-v2
```

This only installs the package code.
All scientific datasets must be downloaded manually and placed in the local `data/` directory 
(see the section "Preparing the data/ directory").

---

## Command-Line Interface (CLI)

### **Interactive CLI**

```bash
psi-cli
```

Menu options:

> 1. Download datasets
> 2. Check datasets
> 3. Run full analysis pipeline
> 4. Open documentation
> 5. Show project paths

This is the recommended entry point for new users.

### Automatic dataset download

Psi-Continuum v2 includes a built-in downloader:

```bash
psi-download-data
```

This command creates data/ automatically and fetches all required datasets from the official GitHub mirror.

---

### Installation from source (clone repository)

```bash
git clone https://github.com/dmitrylife/psi-continuum-v2.git
```

```bash
cd psi-continuum-v2
```

### Create virtual environment

```bash
python3 -m venv sci_venv
source sci_venv/bin/activate
```

### (Optional) Install the package as editable

```bash
pip install .
```

---

### Preparing the `data/` Directory

The `data/` directory is intentionally **excluded** from the PyPI package.  
Users must download all datasets manually and place them in the directory structure shown below.

The expected directory layout is:

```text
data/
├── bao
│   ├── BAO_consensus_covtot_dM_Hz.txt
│   └── sdss_DR12Consensus_bao.dat
├── desi
│   └── dr2
│       ├── desi_gaussian_bao_ALL_GCcomb_cov.txt
│       └── desi_gaussian_bao_ALL_GCcomb_mean.txt
├── hz
│   └── HZ_compilation.csv
└── pantheon_plus
    ├── Pantheon+SH0ES.dat
    └── Pantheon+SH0ES_STAT+SYS.cov
```

You may download these datasets from their original public sources 
or download using the (`psi-download-data`) command:

 - Pantheon+ HF supernova sample
 - Cosmic chronometer H(z) compilation
 - SDSS DR12 BAO consensus
 - DESI DR2 Gaussian BAO data 

Place all files exactly under the paths shown above.

### Quick data status check

After creating the `data/` directory, you can quickly check that all
required files are visible to the package:

```bash
psi-check-data
```

This command will report, for each dataset, whether the expected files
are present, for example:

```text
Pantheon+ SH0ES   OK
H(z) compilation  MISSING   → please place HZ_compilation.csv into ./data/hz/
SDSS DR12 BAO     OK
DESI DR2 BAO      MISSING   → please place DESI DR2 Gaussian files into ./data/desi/dr2/
```

When installed via pip, the tool looks for the `data/` directory in your current working folder.

If some files are missing, the analysis scripts will raise a clear
FileNotFoundError with instructions on where to place the data.

---

## Examples

Данная `examples/` директория исключена из пакета PyPI

Minimal demonstration scripts live in:

```text
examples/
```

Run example:

```bash
python3 examples/example_bao.py
python3 examples/example_hz.py
python3 examples/example_joint.py
python3 examples/example_sn.py

```

---

## Output Directory Structure


```text
results/
├── figures
│   ├── bao
│   │   ├── desi_dr2_DH.png
│   │   ├── desi_dr2_DM.png
│   │   └── desi_dr2_DV.png
│   ├── data_checks
│   │   ├── bao_dr12_DM_check.png
│   │   ├── bao_dr12_Hz_check.png
│   │   ├── desi_dr2_DH_check.png
│   │   ├── desi_dr2_DM_check.png
│   │   └── desi_dr2_DV_check.png
│   ├── hz
│   │   ├── hz_psicdm_chi2_eps_scan.png
│   │   └── hz_psicdm_test.png
│   ├── model_checks
│   │   ├── lcdm_Ez.png
│   │   ├── lcdm_Hz.png
│   │   ├── lcdm_dL.png
│   │   ├── psicdm_eps_scan_Ez.png
│   │   └── psicdm_vs_lcdm_Ez.png
│   ├── publication
│   │   ├── appendix
│   │   │   ├── bao_desi_raw.png
│   │   │   ├── bao_dr12_raw.png
│   │   │   ├── hz_only_chi2_scan.png
│   │   │   ├── hz_quality_checks.png
│   │   │   ├── model_lcdm_Ez_Hz_dL.png
│   │   │   ├── model_psicdm_Ez_scan.png
│   │   │   └── sn_histograms.png
│   │   ├── main_figures
│   │   │   ├── fig1_Ez_comparison.png
│   │   │   ├── fig2_SN_Hubble.png
│   │   │   ├── fig3_BAO_DR12_multipanel.png
│   │   │   ├── fig4_BAO_DESI_multipanel.png
│   │   │   ├── fig5_BAO_fits_LCDM_vs_PsiCDM.png
│   │   │   ├── fig6_Hz_dataset.png
│   │   │   ├── fig7_joint_chi2_eps.png
│   │   │   └── fig8_delta_chi2_contributions.png
│   │   ├── summary
│   │   │   └── poster_summary.png
│   │   └── talk_figures
│   │       ├── talk_bao.png
│   │       ├── talk_sn.png
│   │       └── talk_summary.png
│   ├── scan
│   │   └── eps_scan_total.png
│   └── sn
│       ├── pantheonplus_hf_chi2_eps_scan.png
│       ├── pantheonplus_hf_hubble_diagram.png
│       └── pantheonplus_hf_residuals.png
└── tables
    ├── bao
    │   └── desi_dr2_chi2.txt
    ├── hz
    │   └── hz_psicdm_chi2.txt
    ├── joint
    │   ├── eps_best_joint.txt
    │   └── joint_fit_summary.txt
    ├── scan
    │   └── eps_scan_psicdm.txt
    └── sn
        ├── chi2_eps_scan.txt
        └── pantheonplus_hf_chi2_lcdm.txt
```

---

## Plotting Style

All publication-ready figures use the custom Matplotlib style:

- `psi_continuum_v2/analysis/styles/`

contains the file:

- `psi_style.mplstyle`

This style enforces consistent fonts, line widths, grids, color palette, and
overall layout across all figures generated by the analysis pipeline
(`make_publication_plots.py`).

---

## License

This project is licensed under the terms of the MIT License.
See the **LICENSE** file for details.

---

## Citation

If you use Psi-Continuum v2 in academic work:

```
Dmitry Vasilevich Klimov (2025).
Psi–Continuum Cosmology v2: A Minimal One–Parameter Extension of ΛCDM.
Zenodo preprint. DOI: 10.5281/zenodo.17879744
Zenodo software. DOI: 10.5281/zenodo.17928837
```

Machine-readable citation is provided in CITATION.cff.

---

## Contact

 - Email: d.klimov.psi@gmail.com
 - Website: https://psi-continuum.org
