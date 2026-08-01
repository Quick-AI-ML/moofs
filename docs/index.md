# moofs

**Multi-Objective Optimization for Feature Selection** — a Python library
providing reference implementations of multi-objective feature-selection
algorithms behind a unified, scikit-learn compatible API.

Feature selection is inherently multi-objective: you want the **lowest
classification error** with the **fewest features**. Instead of a single
answer, `moofs` returns the whole Pareto front of optimal trade-offs and
lets you choose.

## Why moofs?

- **Scikit-learn API** — `MOFSSelector` works with `fit`/`transform`,
  pipelines and `get_feature_names_out`.
- **Faithful implementations** — each algorithm ships with the canonical
  parameters of its source paper; ambiguities are documented and exposed as
  explicit flags, never silently resolved.
- **Comparable metrics** — IGD, HV and set coverage follow the PlatEMO
  definitions used in published tables.
- **Built-in visualization** — one-liners to plot and compare Pareto fronts.

## Installation

```bash
pip install moofs
```

Head to [Getting started](getting_started.md) for a full example.
