"""MOFS-RFGA: Multi-Objective Feature Selection with ReliefF and Genetic
Algorithm.

Reference: Y. Xue, H. Zhu, F. Neri, "A feature selection approach based on
NSGA-II with ReliefF", Applied Soft Computing 134 (2023) 109987.
https://doi.org/10.1016/j.asoc.2023.109987

Hybrid filter/wrapper method: a ReliefF (or any filter) score vector guides
the initialisation, a 3-parent crossover and the mutation; NSGA-II
environmental selection drives the multi-objective search.

Note on the crossover semantics: the paper's Fig. 1 and its Algorithm 3
disagree on which gene is removed from S2 / added from S1. The default here
follows Fig. 1 (remove the *worse*-scored gene from S2, add the
*better*-scored gene from S1), which is consistent with the mutation
operator's prose and the score-guided philosophy of the method. Pass
``interpretation="pseudocode"`` for the literal Algorithm 3 reading.
"""

import numpy as np

from ..core.algorithm import Algorithm
from ..core.dominance import dominates, environmental_selection
from ..core.operators import ensure_nonempty
from ..filters import relieff


class MOFSRFGA(Algorithm):
    """MOFS-RFGA in the unified API.

    Parameters
    ----------
    sc : array-like of float, optional
        Feature-score vector (higher = better). If None, ReliefF scores are
        computed automatically from ``problem.X`` and ``problem.y``.
    D_init : int, optional
        Upper bound on the number of features selected at initialisation;
        defaults to ``n_var``.
    """

    name = "MOFS-RFGA"

    def __init__(self, problem, pop_size=60, max_evals=20000, sc=None,
                 D_init=None, interpretation="figure", seed=None,
                 verbose=False):
        super().__init__(problem, pop_size, max_evals, seed, verbose)
        if sc is None:
            sc = relieff(problem.X, problem.y)
        self.sc = np.asarray(sc, dtype=float)
        self.D_init = D_init if D_init is not None else problem.n_var
        if interpretation not in ("figure", "pseudocode"):
            raise ValueError("interpretation must be 'figure' or 'pseudocode'")
        self.interpretation = interpretation


    def _initial_population(self):
        pop = []
        for _ in range(self.pop_size):
            x = np.zeros(self.problem.n_var, dtype=int)
            R = self.rng.integers(1, max(2, self.D_init))
            for _ in range(R):
                i, j = self.rng.choice(len(self.sc), size=2, replace=False)
                x[i if self.sc[i] >= self.sc[j] else j] = 1
            x = ensure_nonempty(x, self.rng)
            pop.append(self.evaluate(x))
        return pop


    def _tournament_parents(self, pop, k=3):
        parents = []
        for _ in range(k):
            i, j = self.rng.choice(len(pop), size=2, replace=False)
            a, b = pop[i], pop[j]
            if dominates(a.F, b.F):
                parents.append(a)
            elif dominates(b.F, a.F):
                parents.append(b)
            else:
                parents.append(a if self.rng.random() < 0.5 else b)
        return parents

    def _crossover_3_to_1(self, p1, p2, p3):
        L1, L2, L3 = p1 & p2, p1 & p3, p2 & p3
        O = np.logical_or(L1, np.logical_or(L2, L3)).astype(int)
        S3 = L1 & L2 & L3                       # genes selected 3 times
        S2 = O ^ S3                             # genes selected exactly twice
        S1 = np.logical_or(p1, np.logical_or(p2, p3)).astype(int) ^ S3 ^ S2
        remove_better = self.interpretation == "pseudocode"
        if self.rng.random() < 0.5:
            cand = np.where(S2 == 1)[0]
            if len(cand) > 0:
                picks = self.rng.choice(cand, size=min(2, len(cand)),
                                        replace=False)
                key = max if remove_better else min
                target = key(picks, key=lambda d: self.sc[d])
                O[target] = 0
        else:
            cand = np.where(S1 == 1)[0]
            if len(cand) > 0:
                picks = self.rng.choice(cand, size=min(2, len(cand)),
                                        replace=False)
                key = min if remove_better else max
                target = key(picks, key=lambda d: self.sc[d])
                O[target] = 1
        return O

    def _mutation(self, o):
        o = o.copy()
        if self.rng.random() < 0.5:
            cand = np.where(o == 1)[0]
            if len(cand) > 0:
                picks = self.rng.choice(cand, size=min(2, len(cand)),
                                        replace=False)
                worst = min(picks, key=lambda d: self.sc[d])
                o[worst] = 0
        else:
            cand = np.where(o == 0)[0]
            if len(cand) > 0:
                picks = self.rng.choice(cand, size=min(2, len(cand)),
                                        replace=False)
                best = max(picks, key=lambda d: self.sc[d])
                o[best] = 1
        return o


    def _run(self):
        pop = self._initial_population()
        pop = environmental_selection(pop, self.pop_size)
        gen = 0
        while self.budget_left():
            offspring = []
            while len(offspring) < self.pop_size and self.budget_left():
                p1, p2, p3 = self._tournament_parents(pop)
                c = self._crossover_3_to_1(p1.x, p2.x, p3.x)
                c = self._mutation(c)
                c = ensure_nonempty(c, self.rng)
                offspring.append(self.evaluate(c))
            pop = environmental_selection(pop + offspring, self.pop_size)
            gen += 1
            self.log(f"gen={gen} evals={self.n_evals}")
        return self._result(pop)
