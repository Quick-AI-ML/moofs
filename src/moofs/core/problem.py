"""Problem definitions.

``FeatureSelectionProblem`` follows the standard protocol of the MOFS
literature (Xue et al.): minimize simultaneously

- f1: classification error (%) of a KNN classifier (k=3) with 3-fold CV,
- f2: number of selected features.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


class Problem:
    """Abstract multi-objective minimization problem.

    Parameters
    ----------
    n_var : int
        Number of decision variables.
    n_obj : int
        Number of objectives.
    encoding : {"binary", "real"}
        Native encoding of the decision vector.
    """

    def __init__(self, n_var, n_obj, encoding="binary"):
        self.n_var = n_var
        self.n_obj = n_obj
        self.encoding = encoding
        self.n_evals = 0

    def evaluate(self, x):
        """Evaluate a decision vector; increments the evaluation counter."""
        self.n_evals += 1
        return self._evaluate(x)

    def _evaluate(self, x):
        raise NotImplementedError

    def reset_counter(self):
        self.n_evals = 0


class FeatureSelectionProblem(Problem):
    """Wrapper-based multi-objective feature selection problem.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target labels.
    estimator : sklearn classifier, optional
        Defaults to ``KNeighborsClassifier(n_neighbors=3)`` as in the papers.
    n_splits : int, default=3
        Number of CV folds.
    random_state : int, default=64
        Seed of the K-Fold shuffling (fixed so that f1 is deterministic and
        cacheable).
    cache : bool, default=True
        Memoize evaluations. Cache hits still increment ``n_evals`` so that
        FE-based stopping criteria stay comparable across algorithms.
    """

    def __init__(self, X, y, estimator=None, n_splits=3, random_state=64,
                 cache=True):
        super().__init__(n_var=X.shape[1], n_obj=2, encoding="binary")
        self.X = X
        self.y = y
        self.estimator = estimator
        self.n_splits = n_splits
        self.random_state = random_state
        self._cache = {} if cache else None

    def _make_estimator(self):
        if self.estimator is None:
            return KNeighborsClassifier(n_neighbors=3)
        from sklearn.base import clone
        return clone(self.estimator)

    def _evaluate(self, mask):
        mask = np.asarray(mask).astype(int)
        n_selected = int(mask.sum())
        if n_selected == 0:
            # Empty subset: worst possible error, zero features.
            return np.array([100.0, 0.0])

        key = tuple(mask)
        if self._cache is not None and key in self._cache:
            return self._cache[key].copy()

        Xs = self.X.iloc[:, mask.astype(bool)]
        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.random_state)
        errors = []
        for tr, te in kf.split(Xs):
            clf = self._make_estimator()
            clf.fit(Xs.iloc[tr], self.y.iloc[tr])
            pred = clf.predict(Xs.iloc[te])
            errors.append(np.mean(np.asarray(self.y.iloc[te]) != pred))
        F = np.array([float(np.mean(errors)) * 100.0, float(n_selected)])

        if self._cache is not None:
            self._cache[key] = F.copy()
        return F
