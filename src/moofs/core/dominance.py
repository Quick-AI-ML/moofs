"""Dominance relations and sorting procedures (Pareto and SDR)."""

import numpy as np

from .solution import population_F


def dominates(f1, f2):
    """Pareto dominance for minimization: f1 dominates f2."""
    f1 = np.asarray(f1)
    f2 = np.asarray(f2)
    return bool(np.all(f1 <= f2) and np.any(f1 < f2))


def fast_non_dominated_sort(population):
    """Deb's fast non-dominated sort. Assigns ``rank`` (1-based) and returns fronts."""
    n = len(population)
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(population[i].F, population[j].F):
                S[i].append(j)
                n_dom[j] += 1
            elif dominates(population[j].F, population[i].F):
                S[j].append(i)
                n_dom[i] += 1
        if n_dom[i] == 0:
            population[i].rank = 1
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    population[j].rank = k + 2
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    fronts.pop()
    return [[population[i] for i in front] for front in fronts]


def crowding_distance(front):
    """Compute and assign NSGA-II crowding distances within a front."""
    n = len(front)
    if n == 0:
        return front
    F = population_F(front)
    dist = np.zeros(n)
    for m in range(F.shape[1]):
        order = np.argsort(F[:, m])
        fmin, fmax = F[order[0], m], F[order[-1], m]
        dist[order[0]] = dist[order[-1]] = np.inf
        if fmax - fmin == 0:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / (fmax - fmin)
    for s, d in zip(front, dist):
        s.crowding = d
    return front


def environmental_selection(population, n_survive):
    """Standard NSGA-II environmental selection (rank then crowding)."""
    fronts = fast_non_dominated_sort(population)
    survivors = []
    for front in fronts:
        crowding_distance(front)
        if len(survivors) + len(front) <= n_survive:
            survivors.extend(front)
        else:
            front = sorted(front, key=lambda s: s.crowding, reverse=True)
            survivors.extend(front[: n_survive - len(survivors)])
            break
    return survivors


def nondominated(population):
    """Return the non-dominated subset of a population."""
    if not population:
        return []
    fronts = fast_non_dominated_sort(population)
    return fronts[0]


# ---------------------------------------------------------------------------
# Strengthened Dominance Relation (SDR), Tian et al., IEEE TEVC 2019.
# ---------------------------------------------------------------------------

def sdr_sort(population):
    """Non-dominated sort under the Strengthened Dominance Relation.

    x SDR-dominates y iff::

        Con(x) < Con(y)               if angle(x, y) <  theta_bar
        Con(x) * angle/theta_bar < Con(y)   otherwise

    where ``Con`` is the sum of normalized objectives and ``theta_bar`` is an
    adaptive niche angle (mean angle of each solution to its nearest
    neighbour, following the adaptive estimation of the original paper).
    Assigns ``rank`` and returns the fronts.
    """
    n = len(population)
    F = population_F(population)
    fmin = F.min(axis=0)
    fmax = F.max(axis=0)
    span = np.where(fmax - fmin == 0, 1.0, fmax - fmin)
    Fn = (F - fmin) / span
    con = Fn.sum(axis=1)

    # Pairwise angles between normalized objective vectors.
    norms = np.linalg.norm(Fn, axis=1)
    norms = np.where(norms == 0, 1e-12, norms)
    cos = np.clip((Fn @ Fn.T) / np.outer(norms, norms), -1.0, 1.0)
    angle = np.arccos(cos)
    np.fill_diagonal(angle, np.inf)
    nearest = angle.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    theta_bar = float(finite.mean()) if len(finite) else np.pi / 2
    theta_bar = max(theta_bar, 1e-12)

    def sdr_dom(i, j):
        a = angle[i, j]
        if a < theta_bar:
            return con[i] < con[j]
        return con[i] * (a / theta_bar) < con[j]

    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts = [[]]
    for i in range(n):
        for j in range(i + 1, n):
            if sdr_dom(i, j):
                S[i].append(j)
                n_dom[j] += 1
            elif sdr_dom(j, i):
                S[j].append(i)
                n_dom[i] += 1
        if n_dom[i] == 0:
            population[i].rank = 1
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    population[j].rank = k + 2
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    fronts.pop()
    if not fronts:  # degenerate case: everything mutually SDR-dominated
        for s in population:
            s.rank = 1
        return [list(population)]
    return [[population[i] for i in front] for front in fronts]
