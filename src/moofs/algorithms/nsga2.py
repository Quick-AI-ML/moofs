"""NSGA-II (Deb et al., 2002) with binary encoding for feature selection.

Default parameters follow the MOFS literature: Pc = 0.9, Pm = 1/D.
"""

from ..core.algorithm import Algorithm
from ..core.dominance import environmental_selection
from ..core.operators import (
    binary_tournament,
    bitflip_mutation,
    ensure_nonempty,
    single_point_crossover,
    uniform_crossover,
)


class NSGA2(Algorithm):
    """NSGA-II for binary multi-objective feature selection.

    Parameters
    ----------
    pc : float, default=0.9
        Crossover probability.
    pm : float, optional
        Per-gene mutation probability; defaults to 1/D.
    crossover : {"single_point", "uniform"}, default="single_point"
    """

    name = "NSGA-II"

    def __init__(self, problem, pop_size=60, max_evals=20000, pc=0.9, pm=None,
                 crossover="single_point", seed=None, verbose=False):
        super().__init__(problem, pop_size, max_evals, seed, verbose)
        self.pc = pc
        self.pm = pm if pm is not None else 1.0 / problem.n_var
        self._crossover = (uniform_crossover if crossover == "uniform"
                           else single_point_crossover)

    def _initial_population(self):
        pop = []
        for _ in range(self.pop_size):
            x = (self.rng.random(self.problem.n_var) < 0.5).astype(int)
            x = ensure_nonempty(x, self.rng)
            pop.append(self.evaluate(x))
        return pop

    def _sort_population(self, population):
        """Hook overridden by NSGA-II/SDR."""
        return environmental_selection(population, self.pop_size)

    def _run(self):
        pop = self._initial_population()
        pop = self._sort_population(pop)
        gen = 0
        while self.budget_left():
            offspring = []
            while len(offspring) < self.pop_size and self.budget_left():
                p1 = binary_tournament(pop, self.rng)
                p2 = binary_tournament(pop, self.rng)
                c1, c2 = self._crossover(p1.x, p2.x, self.rng, pc=self.pc)
                for c in (c1, c2):
                    if len(offspring) >= self.pop_size or not self.budget_left():
                        break
                    c = bitflip_mutation(c, self.rng, pm=self.pm)
                    c = ensure_nonempty(c, self.rng)
                    offspring.append(self.evaluate(c))
            pop = self._sort_population(pop + offspring)
            gen += 1
            self.log(f"gen={gen} evals={self.n_evals}")
        return self._result(pop)
