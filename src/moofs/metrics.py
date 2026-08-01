"""Quality indicators for multi-objective solution sets, PlatEMO-compatible.

The IGD, HV and coverage definitions follow the PlatEMO implementations used
in the MOFS literature (Tian et al., PlatEMO, IEEE CIM 2017), so values are
directly comparable with published tables:

- ``igd``: mean of the minimum distances from each reference point (IGD.m).
- ``hv``: exact 2-D hypervolume after PlatEMO normalization (HV.m):
  objectives scaled by 1.1 * (max(PF) - fmin), reference point (1, 1).
- ``coverage``: weak-dominance set coverage (Coverage.m).

All functions accept 2-D objective arrays, lists of ``Solution``, or a
``Result``.
"""

import numpy as np
import pandas as pd

from .core.dominance import dominates
from .core.solution import Solution


def _as_F(front):
    from .core.algorithm import Result
    if isinstance(front, Result):
        return front.F
    if len(front) == 0:
        return np.empty((0, 2))
    if isinstance(front[0], Solution):
        return np.array([s.F for s in front], dtype=float)
    return np.asarray(front, dtype=float)


def igd(reference_front, front):
    """Inverted Generational Distance, PlatEMO definition (lower is better)."""
    R = _as_F(reference_front)
    A = _as_F(front)
    if len(A) == 0:
        return np.inf
    d = np.sqrt(((R[:, None, :] - A[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


def gd(reference_front, front):
    """Generational Distance (lower is better)."""
    R = _as_F(reference_front)
    A = _as_F(front)
    if len(A) == 0:
        return np.inf
    d = np.sqrt(((A[:, None, :] - R[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


def hv(front, reference_front):
    """Hypervolume, PlatEMO definition (higher is better).

    Objectives are normalized by ``fmin = min(min(front), 0)`` and
    ``fmax = max(reference_front)`` with a 1.1 scaling factor; points beyond
    the (1, 1) reference point are discarded; the exact 2-D hypervolume of
    the remaining non-dominated points is returned.
    """
    A = _as_F(front)
    R = _as_F(reference_front)
    if len(A) == 0 or len(R) == 0:
        return 0.0
    if A.shape[1] != 2:
        raise NotImplementedError("hv is implemented for 2 objectives.")
    fmin = np.minimum(A.min(axis=0), 0.0)
    fmax = R.max(axis=0)
    span = (fmax - fmin) * 1.1
    span[span == 0] = 1.0
    An = (A - fmin) / span
    An = An[~np.any(An > 1.0, axis=1)]
    if len(An) == 0:
        return 0.0
    keep = [i for i in range(len(An))
            if not any(dominates(An[j], An[i]) for j in range(len(An)) if j != i)]
    An = An[keep]
    An = An[np.argsort(An[:, 0])]
    score, prev = 0.0, 1.0
    for f1, f2 in An:
        if f2 < prev:
            score += (1.0 - f1) * (prev - f2)
            prev = f2
    return float(score)


def hv_raw(front, reference_point):

    A = _as_F(front)
    if len(A) == 0:
        return 0.0
    ref = np.asarray(reference_point, dtype=float)
    A = A[(A[:, 0] < ref[0]) & (A[:, 1] < ref[1])]
    if len(A) == 0:
        return 0.0
    keep = [i for i in range(len(A))
            if not any(dominates(A[j], A[i]) for j in range(len(A)) if j != i)]
    A = A[keep]
    A = A[np.argsort(A[:, 0])]
    score, prev = 0.0, ref[1]
    for f1, f2 in A:
        if f2 < prev:
            score += (ref[0] - f1) * (prev - f2)
            prev = f2
    return float(score)


def coverage(A, B):
    """Set coverage SC(A, B), 

    Fraction of solutions in B that are weakly dominated by (i.e. no better
    in any objective than) at least one solution in A.
    """
    FA = _as_F(A)
    FB = _as_F(B)
    if len(FB) == 0:
        return 0.0
    count = sum(
        1 for fb in FB if any(np.all(fa <= fb) for fa in FA)
    )
    return count / len(FB)


def nfs(front):
    """Number of Feature Subsets: distinct solutions in the front."""
    A = _as_F(front)
    return len(np.unique(A, axis=0))


def spacing(front):
    """Schott's spacing metric (lower = more uniform distribution)."""
    A = _as_F(front)
    if len(A) < 2:
        return 0.0
    d = np.abs(A[:, None, :] - A[None, :, :]).sum(axis=2)
    np.fill_diagonal(d, np.inf)
    di = d.min(axis=1)
    return float(np.sqrt(((di - di.mean()) ** 2).sum() / (len(A) - 1)))


def merge_reference_front(*fronts):
    """Reference ("true") Pareto front: non-dominated union of several fronts.

    This follows the protocol of the MOFS literature: the fronts of all
    algorithms are merged and non-dominated sorted; the first front is
    treated as the reference.
    """
    Fs = [_as_F(f) for f in fronts if len(_as_F(f)) > 0]
    if not Fs:
        return np.empty((0, 2))
    all_F = np.unique(np.vstack(Fs), axis=0)
    keep = [i for i in range(len(all_F))
            if not any(dominates(all_F[j], all_F[i])
                       for j in range(len(all_F)) if j != i)]
    return all_F[keep]


def compare(results, reference_front=None):
    """Metric table for a set of results.

    Parameters
    ----------
    results : dict
        Mapping ``name -> Result`` (or front).
    reference_front : array-like, optional
        Reference front; defaults to the non-dominated union of all results.

    Returns
    -------
    pandas.DataFrame
        One row per algorithm: IGD, HV, NFS, best error, smallest subset.
    """
    ref = (merge_reference_front(*results.values())
           if reference_front is None else _as_F(reference_front))
    rows = []
    for name, res in results.items():
        F = _as_F(res)
        rows.append({
            "algorithm": name,
            "IGD": igd(ref, F),
            "HV": hv(F, ref),
            "NFS": nfs(F),
            "best_error_%": float(F[:, 0].min()) if len(F) else np.nan,
            "min_subset_size": int(F[:, 1].min()) if len(F) else -1,
        })
    return (pd.DataFrame(rows)
            .sort_values("IGD")
            .reset_index(drop=True))
