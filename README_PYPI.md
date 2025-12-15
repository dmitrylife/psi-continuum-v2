# Psi-Continuum v2

**A minimal one-parameter phenomenological extension of the ΛCDM cosmological model**

**Author:** Dmitry V. Klimov  
**Status:** Final archived research release (v0.2.3)

---

## Overview

Psi-Continuum v2 (ΨCDM) is a minimal background-level extension of the standard
ΛCDM cosmological model.  
It introduces a single dimensionless late-time deformation parameter **ε₀**
while not introducing any modifications to early-Universe physics
or standard perturbation dynamics.

The model is designed for transparent, reproducible testing of late-time
cosmological expansion using modern observational datasets.

This repository represents the **final archived software release** associated
with the Psi-Continuum v2 publication.

---

## Model Definition

The modified Hubble expansion rate is defined as:

```text
H_Ψ(z) = H_Λ(z) · (1 + ε₀ / (1 + z))
```


Properties of the parametrisation:

- Exact recovery of ΛCDM when ε₀ = 0
- Percent-level deviations only at low redshift
- Suppressed modifications as z → ∞
- No impact on early-Universe observables (CMB, primordial BAO)

---

## Installation

Install the package from PyPI:

```bash
pip install psi-continuum-v2
```

### Important notes

- Scientific datasets are **not included** in the PyPI package.
- Generated results (`results/`) are **not included**.
- All datasets must be downloaded separately using the built-in tools.

---

## Command-Line Interface (CLI)

Psi-Continuum v2 provides an interactive and scriptable CLI.

### Interactive menu

```bash
psi-cli
```

Menu options include:

- Download datasets
- Check datasets
- Run full analysis pipeline
- Open documentation
- Show project paths

---

## Built-in commands

Download all required datasets:

```bash
psi-download-data
```

Verify dataset integrity and paths:

```bash
psi-check-data
```

Run the complete analysis pipeline:

```bash
psi-run-all
```

These commands reproduce all numerical results and figures reported in the
Psi-Continuum v2 publication.

---

## Documentation and Source Code

- **Documentation**: https://psi-continuum.org/docs/v2
- **Source code**: https://github.com/dmitrylife/psi-continuum-v2

---

## Publications and DOIs

- Psi-Continuum v2 preprint:  
  DOI: https://doi.org/10.5281/zenodo.17879744

- Psi-Continuum v2 software archive:  
  DOI: https://doi.org/10.5281/zenodo.17928837

---

## Project Status

This is a **final archived research release**.

The codebase is frozen and preserved for long-term reproducibility and citation.
Future developments of the Psi-Continuum framework will be released as separate
repositories and publications.

---

## Contact

**Dmitry V. Klimov**  
Email: d.klimov.psi@gmail.com  
Website: https://psi-continuum.org

