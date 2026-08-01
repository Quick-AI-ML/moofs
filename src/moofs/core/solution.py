"""Unified solution representation shared by all algorithms."""

import numpy as np


class Solution:
    """A candidate solution.

    Attributes
    ----------
    x : np.ndarray
        Decision vector. Binary mask for feature selection, or real vector
        (e.g. PSO particle position).
    F : np.ndarray or None
        Objective values (minimization).
    rank : int
        Non-domination rank (1 = first front).
    crowding : float
        Crowding distance.
    attrs : dict
        Algorithm-specific data (velocity, pbest, SPEA2 fitness, ...).
    """

    __slots__ = ("x", "F", "rank", "crowding", "attrs")

    def __init__(self, x, F=None):
        self.x = np.asarray(x)
        self.F = None if F is None else np.asarray(F, dtype=float)
        self.rank = 0
        self.crowding = 0.0
        self.attrs = {}

    def copy(self):
        s = Solution(self.x.copy(), None if self.F is None else self.F.copy())
        s.rank = self.rank
        s.crowding = self.crowding
        s.attrs = dict(self.attrs)
        return s

    def __repr__(self):
        return f"Solution(F={self.F}, rank={self.rank})"


def population_F(population):
    """Stack the objective vectors of a population into a 2-D array."""
    return np.array([s.F for s in population], dtype=float)
