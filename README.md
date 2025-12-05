# Psi-Continuum v2  
### Cosmological Framework and Joint Likelihood Analysis  
**Author:** Dmitry V. Klimov
**Status:** Working research prototype

---

## Overview

Psi-Continuum (Ψ-CDM) is a phenomenological extension of ΛCDM that introduces a single additional parameter **ε₀**, modifying the late-time expansion rate:

E²(z) = Ωₘ (1+z)³ + (1 − Ωₘ) · (1 + ε₀ f(z)).

In the limit **ε₀ → 0**, ΨCDM reduces exactly to ΛCDM.

This repository provides:

- Background cosmology modules (ΛCDM and ΨCDM)
- Full cosmological data pipeline:
  - Pantheon+ HF supernovae
  - H(z) cosmic chronometers
  - SDSS DR12 BAO consensus
  - DESI DR2 Gaussian BAO
- χ² likelihoods for all datasets
- Joint likelihood combination
- ε₀ scanning and best-fit extraction
- Publication-quality plots
- Automated pipeline runner (`run_all.py`)

---

## Main Scientific Results (v2)

- Supernovae favour **ε₀ < 0**
- DESI BAO strongly favours **ε₀ > 0**
- Combined likelihood gives:

ε₀_best ≈ 0.03–0.08

- Joint fit shows improvement over ΛCDM:

Δχ² ≈ −6.9

This indicates ΨCDM fits current background-expansion data slightly better, while keeping ΛCDM as a limiting case.

---

## Repository Structure

```
psi-continuum-v2/
├── analysis/
│   ├── check_models.py
│   ├── sn_test_lcdm_pplus_simple.py
│   ├── sn_test_psicdm_pplus.py
│   ├── hz_test_psicdm.py
│   ├── bao_desi_dr2_test.py
│   ├── joint_fit_psicdm.py
│   ├── scan_eps_psicdm.py
│   ├── eps_best_joint_test.py
│   └── make_all_plots.py
│  
├── cosmology/
│   ├── background/
│   │   ├── lcdm.py
│   │   ├── psicdm.py
│   │   └── distances.py
│   ├── likelihoods/
│   │   ├── sn_likelihood.py
│   │   ├── hz_likelihood.py
│   │   ├── bao_likelihood.py
│   │   └── joint_likelihood.py
│   ├── data_loaders/
│   │   ├── pantheonplus_loader.py
│   │   ├── hz_loader.py
│   │   ├── bao_loader.py
│   │   └── desi_loader.py
│   ├── models/
│   │   ├── lcdm_params.py
│   │   └── psicdm_params.py
│   └── utils/
│  
├── data/
│   ├── pantheon_plus/
│   ├── hz/
│   ├── bao/
│   └── desi/dr2/
│  
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
│  
├── run_all.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/dmitrylife/psi-continuum-v2.git
cd psi-continuum-v2
```

### 2. Create virtual environment

```bash
python3 -m venv sci_venv
source sci_venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
---

## Running the Full Pipeline

To execute all scientific steps:

```bash
./run_all.py
```

Python ≥ 3.10 recommended.

This performs:

1. check_models.py
2. sn_test_lcdm_pplus_simple.py
3. sn_test_psicdm_pplus.py
4. hz_test_psicdm.py
5. bao_desi_dr2_test.py
6. joint_fit_psicdm.py
7. scan_eps_psicdm.py
8. eps_best_joint_test.py
9. make_all_plots.py

All logs are stored in:

results/logs/

---

## Output and Results

### Tables

results/tables/eps_scan_psicdm.txt – χ²(ε₀) scan
results/tables/eps_best_joint.txt – joint test at ε₀ = 0.031
Dataset-specific χ² tables

### Publication Figures

Stored in:

results/figures/publication/

Includes:

- Ez_comparison.png
- dL_SN_comparison.png
- BAO_multipanel.png
- BAO_multipanel_rs.png
- chi2_epsilon_curve.png
- hz_psicdm_test.png
- delta_chi2_bar.png

---

## Scientific Notes

- ΨCDM is extendable to perturbation theory and CMB
- Planned upgrades:
  - MCMC with emcee
  - Full DESI non-Gaussian likelihood
  - fσ₈ growth analysis
  - CMB distance priors

---

## License

MIT License © 2025 Dmitry V. Klimov

---

## 📚 Citation

If you use this repository or figures in your research:

**Dmitry Vasilevich Klimov (2025).
*Psi–Continuum Cosmology v2: A Minimal One–Parameter Extension of ΛCDM.***

Zenodo. https://doi.org/...

---

## 📮 Contact

📧 Email: **d.klimov.psi@gmail.com**
🌐 Website: **https://psi-continuum.org**
