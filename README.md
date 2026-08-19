# [moofs](https://quick-ai-ml.github.io/moofs/)

![Banner Image](https://github.com/Quick-AI-ML/moofs/raw/main/docs/assets/banner.jpeg)

[![PyPi Version](https://img.shields.io/pypi/v/moofs)](https://pypi.org/project/moofs/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://quick-ai-ml.github.io/moofs/)
[![GitHub Stars](https://img.shields.io/github/stars/Quick-AI-ML/moofs?style=social)](https://github.com/Quick-AI-ML/moofs)
[![License](https://img.shields.io/github/license/Quick-AI-ML/moofs)](https://github.com/Quick-AI-ML/moofs/blob/main/LICENSE)

**Multi-Objective Optimization for Feature Selection**

A Python library for multi-objective feature selection with a unified, scikit-learn compatible API. `moofs` searches for the best trade-offs between **classification error** and **number of selected features**, returns the full Pareto front, and lets you pick the subset that fits your needs.
**Multi-Objective Optimization for Feature Selection**

A Python library for multi-objective feature selection with a unified, scikit-learn compatible API. `moofs` searches for the best trade-offs between **classification error** and **number of selected features**, returns the full Pareto front, and lets you pick the subset that fits your needs.

## Installation

```bash
pip install moofs
```

## Quick start

```python
from moofs import MOFSSelector

selector = MOFSSelector(algorithm="mofs-rfga", max_evals=5000, random_state=0)
X_reduced = selector.fit_transform(X, y)

selector.pareto_front_   # (n_solutions, 2): [error %, subset size]
selector.support_        # boolean mask of the chosen subset
selector.get_feature_names_out()
```

It drops into any scikit-learn pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

pipe = Pipeline([
    ("fs", MOFSSelector(max_evals=5000, random_state=0)),
    ("clf", KNeighborsClassifier(n_neighbors=3)),
]).fit(X_train, y_train)
```

## Available algorithms

| Algorithm | Key | Authors | Reference |
|-----------|-----|---------|-----------|
| **MOFS-RFGA** | `mofs-rfga` | Xue, Zhu & Neri (2023) | [paper](https://doi.org/10.1016/j.asoc.2023.109987) |
| **NSGA-II** | `nsga2` | Deb, Pratap, Agarwal & Meyarivan (2002) | [paper](https://doi.org/10.1109/4235.996017) |

More algorithms from the MOFS literature (SparseEA, NSGA-II/SDR, SPEA2, MOEA/D, NSPSOFS, CMDPSOFS) are planned for upcoming releases — see the [CHANGELOG](CHANGELOG.md).

## Visualizing Pareto fronts

```python
from moofs import plot_selector, plot_fronts

plot_selector(selector)              # front + highlighted chosen subset
plot_fronts({"MOFS-RFGA": r1, "NSGA-II": r2}, reference=True)
```

## Metrics

Quality indicators follow the **PlatEMO definitions** used in the MOFS literature, so values are directly comparable with published tables: `igd`, `hv` (normalized, reference point (1,1)), `coverage` (weak dominance), `nfs`, `spacing`.

```python
from moofs import compare

table = compare({"MOFS-RFGA": r1, "NSGA-II": r2})
#   algorithm    IGD     HV   NFS  best_error_%  min_subset_size
```

## Research-style API

For experiments and full control over the search:

```python
from moofs import FeatureSelectionProblem, MOFSRFGA, NSGA2

problem = FeatureSelectionProblem(X, y)   # KNN k=3, 3-fold CV, cached
result = MOFSRFGA(problem, pop_size=100, max_evals=300_000, seed=0).run()
result.F        # objective matrix of the Pareto front
result.front    # solutions with binary masks (.x)
```

The evaluation protocol follows the reference paper: k-NN (k=3) classifier, 3-fold cross-validation, objectives = (classification error %, subset size). Evaluations are memoized; cache hits still count toward `max_evals` so budgets stay comparable.

## Faithfulness notes

Implementations are traceable to their source papers, and ambiguities are documented rather than silently resolved. Notably, the MOFS-RFGA paper's Fig. 1 and its Algorithm 3 disagree on the crossover semantics; `moofs` defaults to the Fig. 1 reading (consistent with the mutation operator) and exposes `interpretation="pseudocode"` for the literal alternative. See the documentation for details.

## License

MIT — see [LICENSE](LICENSE).

## Citing

If you use `moofs` in academic work, please cite the underlying algorithm papers (see the table above). A citable software DOI is planned.
