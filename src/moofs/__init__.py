"""moofs — Multi-Objective Optimization for Feature Selection.

A reference library of multi-objective feature-selection algorithms with a
scikit-learn compatible API, Pareto-front visualization, and
PlatEMO-compatible quality indicators.

v0.1 algorithms: MOFS-RFGA (Xue, Zhu & Neri, 2023) and NSGA-II (Deb et al.,
2002).

Quick start
-----------
>>> from moofs import MOFSSelector
>>> selector = MOFSSelector(algorithm="mofs-rfga", max_evals=2000,
...                         random_state=0)
>>> X_reduced = selector.fit_transform(X, y)

Research-style API
------------------
>>> from moofs import FeatureSelectionProblem, MOFSRFGA
>>> problem = FeatureSelectionProblem(X, y)
>>> result = MOFSRFGA(problem, pop_size=60, max_evals=2000, seed=0).run()
>>> result.F
"""

from .algorithms import MOFSRFGA, NSGA2
from .core import (
    Algorithm,
    FeatureSelectionProblem,
    Problem,
    Result,
    Solution,
)
from .filters import mutual_information, relieff
from .metrics import (
    compare,
    coverage,
    gd,
    hv,
    hv_raw,
    igd,
    merge_reference_front,
    nfs,
    spacing,
)
from .plotting import plot_fronts, plot_pareto_front, plot_selector
from .selection import MOFSSelector

__version__ = "0.1.0"

ALGORITHMS = {
    "mofs-rfga": MOFSRFGA,
    "nsga2": NSGA2,
}

__all__ = [
    "MOFSSelector",
    "MOFSRFGA", "NSGA2", "ALGORITHMS",
    "Problem", "FeatureSelectionProblem", "Solution", "Algorithm", "Result",
    "relieff", "mutual_information",
    "igd", "gd", "hv", "hv_raw", "coverage", "nfs", "spacing",
    "merge_reference_front", "compare",
    "plot_pareto_front", "plot_fronts", "plot_selector",
    "__version__",
]
