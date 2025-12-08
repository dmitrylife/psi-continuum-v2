# Contributing to Psi-Continuum v2

Thank you for your interest in contributing!

Psi-Continuum v2 is a scientific Python package designed for transparent and 
reproducible cosmological analysis.  
We welcome contributions that improve code quality, scientific correctness, 
documentation, and dataset support.

---

## 1. Repository Structure

The main package code lives in:

```text
psi_continuum_v2/
├── cosmology/
└── analysis/
```

All executable analysis scripts are located under:

```text
psi_continuum_v2/analysis/
```

---

## 2. Reporting Issues

When opening an issue on GitHub, please include:

- a clear description of the problem,
- steps to reproduce the issue,
- expected vs. observed behavior,
- dataset(s) used,
- Python version and operating system.

This helps maintain reproducibility and ensures fast debugging.

---

## 3. Submitting Pull Requests (PR)

1. **Fork** the repository.
2. Create a new branch:

```bash
git checkout -b feature/my-improvement
```

3. Make changes following the project style guidelines (see below).

4. Ensure all χ² values and plots remain numerically stable.

5. Run the full pipeline to verify integrity:

```bash
python run_all.py
```

6. Ensure all unit tests pass:

 - `pytest -q`

7. Submit your pull request to the **main** branch with a clear description of the changes.

---

## 4. Coding Style Guidelines

 - Follow **PEP8** for all Python files.

 - Use **type hints** wherever possible (-> float, -> np.ndarray, etc.).

 - Prefer **pure functions** for likelihoods and model evaluations.

 - Avoid hard-coded file paths — always use:

```python
from pathlib import Path
Path(__file__).resolve()
```

 - **Use NumPy-style docstrings** for all public functions.
 
---

## 5. Scientific Reproducibility Requirements

Any new dataset, model, or likelihood **must include**:

- The original data files placed in the `data/` directory.
- A corresponding loader module under `cosmology/data_loaders/`.
- Clear documentation of the format and covariance usage.
- Validation and diagnostic plots.
- At least one analysis script demonstrating correct usage.

All scientific results should be fully reproducible with:

```bash
python run_all.py
```

---

## 6. Running Tests

Before submitting a pull request, run the test suite:

```bash
pytest -q
```
If you installed the package via `pip install -e .`, tests should execute without errors.

All new contributions must:

 - include tests when appropriate,

 - keep existing tests passing,

 - avoid breaking the reproducibility pipeline (`run_all.py`).

---

## 7. Contact

For scientific discussion, questions, or collaboration inquiries:

**Dmitry V. Klimov**  
Email: d.klimov.psi@gmail.com

Please use the GitHub issue tracker:
https://github.com/dmitrylife/psi-continuum-v2/issues

Pull requests:
https://github.com/dmitrylife/psi-continuum-v2/pulls
