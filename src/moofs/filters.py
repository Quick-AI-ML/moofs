"""Filter methods producing feature-score vectors (higher = better)."""

import numpy as np


def relieff(X, y, n_neighbors=10, sample_size=None, random_state=None):
    """ReliefF feature scoring (Kononenko, 1994).

    Parameters
    ----------
    X : pandas.DataFrame or ndarray
        Feature matrix.
    y : pandas.Series or ndarray
        Class labels.
    n_neighbors : int, default=10
        Number of nearest hits/misses per class.
    sample_size : int, optional
        Number of instances sampled to estimate the weights (default: all).
    random_state : int, optional

    Returns
    -------
    ndarray of shape (n_features,)
        Feature weights; higher means more relevant.
    """
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y)
    n, d = Xa.shape
    rng = np.random.default_rng(random_state)

    # Normalize features to [0, 1] so that diffs are comparable.
    xmin = Xa.min(axis=0)
    span = Xa.max(axis=0) - xmin
    span[span == 0] = 1.0
    Xn = (Xa - xmin) / span

    classes, counts = np.unique(ya, return_counts=True)
    priors = {c: cnt / n for c, cnt in zip(classes, counts)}

    idx = np.arange(n)
    if sample_size is not None and sample_size < n:
        idx = rng.choice(n, size=sample_size, replace=False)

    weights = np.zeros(d)
    for i in idx:
        xi = Xn[i]
        dists = np.abs(Xn - xi).sum(axis=1)  # Manhattan distance
        dists[i] = np.inf
        for c in classes:
            members = np.where(ya == c)[0]
            members = members[members != i]
            if len(members) == 0:
                continue
            k = min(n_neighbors, len(members))
            nearest = members[np.argsort(dists[members])[:k]]
            diff = np.abs(Xn[nearest] - xi).mean(axis=0)
            if c == ya[i]:
                weights -= diff / len(idx)
            else:
                w = priors[c] / (1.0 - priors[ya[i]] + 1e-12)
                weights += w * diff / len(idx)
    return weights


def mutual_information(X, y, random_state=0):
    from sklearn.feature_selection import mutual_info_classif
    return mutual_info_classif(X, y, random_state=random_state)
