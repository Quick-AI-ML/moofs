"""Base class shared by every algorithm, and the Result container."""

import numpy as np

from .dominance import nondominated
from .solution import Solution, population_F


class Result:
    """Outcome of a run.

    Attributes
    ----------
    front : list of Solution
        Non-dominated solutions found.
    population : list of Solution
        Final population (or archive).
    n_evals : int
        Number of objective-function evaluations consumed.
    algorithm : str
        Name of the algorithm.
    """

    def __init__(self, front, population, n_evals, algorithm):
        self.front = front
        self.population = population
        self.n_evals = n_evals
        self.algorithm = algorithm

    @property
    def F(self):
        """Objective matrix of the front (n_solutions x n_obj)."""
        return population_F(self.front)

    def __repr__(self):
        return (f"Result(algorithm={self.algorithm!r}, "
                f"front={len(self.front)}, n_evals={self.n_evals})")


class Algorithm:
    """Base class for all algorithms.

    Parameters
    ----------
    problem : Problem
    pop_size : int, default=60
        Population (or swarm/archive) size N.
    max_evals : int, default=20000
        Maximum number of objective-function evaluations (maxFES).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
    """

    name = "algorithm"

    def __init__(self, problem, pop_size=60, max_evals=20000, seed=None,
                 verbose=False):
        self.problem = problem
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self._start_evals = 0


    def evaluate(self, x):
        """Evaluate a decision vector and wrap it in a Solution."""
        return Solution(x, self.problem.evaluate(x))

    @property
    def n_evals(self):
        return self.problem.n_evals - self._start_evals

    def budget_left(self):
        return self.n_evals < self.max_evals

    def log(self, *args):
        if self.verbose:
            print(f"[{self.name}]", *args)

    def _result(self, population):
        front = nondominated([s for s in population if s.F is not None])
        return Result(front, population, self.n_evals, self.name)

    # -- interface ---------------------------------------------------------

    def run(self):
        """Run the algorithm until the FE budget is exhausted."""
        self._start_evals = self.problem.n_evals
        return self._run()

    def _run(self):
        raise NotImplementedError
