# Analysis Scripts — Psi-Continuum v2

This directory contains the fully reproducible analysis scripts used in the
ΨCDM cosmological study.
Each script performs a well-defined statistical or diagnostic task and stores
its outputs under `results/`.

All scripts may be executed individually or collectively via:


```bash
./run_all.py

```

---

## 1. Supernovae: Pantheon+ SH0ES (HF)

`sn_test_lcdm_pplus_simple.py`

Computes the baseline ΛCDM χ² for the Pantheon+ SH0ES (HF) supernova sample.

`sn_test_psicdm_pplus.py`

Computes ΨCDM χ², performs an ε₀-scan, and produces:

 - Hubble diagram

 - residuals plot

 - χ²(ε₀) scan

Outputs are saved in:


```text
results/figures/sn/
results/tables/sn/
```
 
 ---
 
## 2. H(z) Expansion-Rate Compilation

`hz_test_psicdm.py`

Loads the full H(z) compilation and computes:

 - ΛCDM vs ΨCDM predictions

 - χ²(ε₀) scan

 - diagnostic plots

Outputs:

```text
results/figures/hz/
results/tables/hz/
```

---

## 3. BAO Likelihoods

`check_bao_dr12_data.py`

Validates SDSS DR12 BAO consensus vector & covariance, generates diagnostic plots.

`bao_desi_dr2_test.py`

Processes DESI DR2 Gaussian BAO data, producing:

 - model vs data plots

 - χ² breakdown

Outputs stored in:

```text
results/figures/bao/
results/tables/bao/
```

---

## 4. Joint Likelihood Analysis

`joint_fit_psicdm.py`

Combines four datasets:

 - Pantheon+ SH0ES (HF)

 - H(z) compilation

 - SDSS DR12 BAO

 - DESI DR2 Gaussian BAO

Produces:

 - χ² per dataset

 - total χ² and reduced χ²

 - Δχ² (ΨCDM − ΛCDM)

 - summary tables

Stored in:

```text
results/tables/joint/
results/figures/
```

---

## 5. Parameter Scans and Best-Fit Evaluations

`scan_eps_psicdm.py`

Scans ε₀, determines best-fit value, and generates χ²(ε₀) plots.

`eps_best_joint_test.py`

Runs the joint likelihood at a fixed chosen ε₀ (e.g., ε₀ = 0.031).

---

## 6. Validation & Utility Scripts

`check_models.py`

Validates:

 - ΛCDM and ΨCDM background functions

 - numerical stability of E(z), H(z), dL(z)

 - ΨCDM → ΛCDM limit as ε₀ → 0

`check_hz_data.py`, `check_pantheonplus_data.py`, `check_desi_dr2_data.py`

Dataset sanity checks:

 - covariance shapes

 - ranges

 - diagnostic distributions

---

## 7. Output Directories

All generated outputs are stored under:

```text
results/figures/
results/tables/
results/logs/
```

The entire analysis pipeline is deterministic and fully reproducible.
