"""Scikit-learn compatible feature selectors.

``MOFSSelector`` wraps the moofs algorithms behind the standard
``fit`` / ``transform`` / ``get_support`` API, so multi-objective feature
selection drops into any scikit-learn pipeline::

    from moofs import MOFSSelector

    selector = MOFSSelector(algorithm="mofs-rfga", max_evals=2000,
                            random_state=0)
    X_reduced = selector.fit_transform(X, y)
    selector.pareto_front_        # (n_solutions, 2) objective matrix
    selector.support_             # boolean mask of the chosen subset
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from .algorithms import MOFSRFGA, NSGA2
from .core.problem import FeatureSelectionProblem

_ALGORITHMS = {
    "mofs-rfga": MOFSRFGA,
    "nsga2": NSGA2,
}


class MOFSSelector(SelectorMixin, BaseEstimator):
    """Multi-objective feature selection with a scikit-learn interface.

    Runs a multi-objective algorithm minimizing (classification error %,
    number of selected features), then picks one solution from the Pareto
    front according to ``strategy``.

    Parameters
    ----------
    algorithm : {"mofs-rfga", "nsga2"}, default="mofs-rfga"
        Search algorithm. MOFS-RFGA (Xue, Zhu & Neri, 2023) is the
        ReliefF-guided hybrid; NSGA-II is the classic baseline.
    pop_size : int, default=60
        Population size N.
    max_evals : int, default=5000
        Budget in objective-function evaluations (maxFEs).
    strategy : {"knee", "min_error", "min_features"}, default="knee"
        How to pick the final subset from the Pareto front:
        "knee" = best normalized trade-off, "min_error" = most accurate,
        "min_features" = smallest subset.
    sc : array-like, optional
        Precomputed feature scores for MOFS-RFGA (defaults to built-in
        ReliefF). Ignored by NSGA-II.
    random_state : int, optional
        Seed for reproducibility.
    verbose : bool, default=False

    Attributes
    ----------
    pareto_front_ : ndarray of shape (n_solutions, 2)
        Objective values [error %, subset size] of the final Pareto front.
    pareto_masks_ : ndarray of shape (n_solutions, n_features)
        Binary masks of the Pareto-front solutions.
    support_ : ndarray of shape (n_features,)
        Boolean mask of the selected subset (per ``strategy``).
    result_ : moofs.Result
        Full algorithm result.
    n_evals_ : int
        Evaluations actually consumed.
    """

    def __init__(self, algorithm="mofs-rfga", pop_size=60, max_evals=5000,
                 strategy="knee", sc=None, random_state=None, verbose=False):
        self.algorithm = algorithm
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.strategy = strategy
        self.sc = sc
        self.random_state = random_state
        self.verbose = verbose


    def _more_tags(self):
        return {"requires_y": True, "allow_nan": False}

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


    @staticmethod
    def _pick(F, strategy):
        if strategy == "min_error":
            return int(np.lexsort((F[:, 1], F[:, 0]))[0])
        if strategy == "min_features":
            return int(np.lexsort((F[:, 0], F[:, 1]))[0])
        if strategy == "knee":
            fmin = F.min(axis=0)
            span = F.max(axis=0) - fmin
            span[span == 0] = 1.0
            Fn = (F - fmin) / span
            return int(np.argmin(Fn.sum(axis=1)))
        raise ValueError(f"Unknown strategy: {strategy!r}")


    def fit(self, X, y):
        """Run the multi-objective search on (X, y)."""
        if self.algorithm not in _ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {sorted(_ALGORITHMS)}, "
                f"got {self.algorithm!r}")

        if isinstance(X, pd.DataFrame):
            Xdf = X.reset_index(drop=True)
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            X = np.asarray(X)
            Xdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        ydf = pd.Series(np.asarray(y)).reset_index(drop=True)
        self.n_features_in_ = Xdf.shape[1]

        problem = FeatureSelectionProblem(Xdf, ydf)
        cls = _ALGORITHMS[self.algorithm]
        kwargs = dict(pop_size=self.pop_size, max_evals=self.max_evals,
                      seed=self.random_state, verbose=self.verbose)
        if self.algorithm == "mofs-rfga" and self.sc is not None:
            kwargs["sc"] = self.sc
        algo = cls(problem, **kwargs)

        self.result_ = algo.run()
        self.n_evals_ = self.result_.n_evals
        self.pareto_front_ = self.result_.F
        self.pareto_masks_ = np.array(
            [np.asarray(s.x, dtype=int) for s in self.result_.front])

        idx = self._pick(self.pareto_front_, self.strategy)
        self.support_ = self.pareto_masks_[idx].astype(bool)
        self.selected_objectives_ = self.pareto_front_[idx]
        return self


    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        if input_features is None:
            input_features = getattr(
                self, "feature_names_in_",
                np.array([f"x{i}" for i in range(self.n_features_in_)],
                         dtype=object))
        return np.asarray(input_features, dtype=object)[self.support_]
