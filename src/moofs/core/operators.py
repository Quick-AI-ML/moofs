"""Variation operators: binary GA operators, SBX, polynomial mutation, repair."""

import numpy as np

from .dominance import dominates



def binary_tournament(population, rng, key=None):
    """Binary tournament selection.

    ``key(s)`` must return a sortable value where *smaller is better*.
    Default key: (rank, -crowding), i.e. NSGA-II comparison.
    """
    if key is None:
        key = lambda s: (s.rank, -s.crowding)
    i, j = rng.choice(len(population), size=2, replace=False)
    a, b = population[i], population[j]
    return a if key(a) <= key(b) else b


def dominance_tournament(population, rng):
    """Binary tournament based on raw Pareto dominance (random tie-break)."""
    i, j = rng.choice(len(population), size=2, replace=False)
    a, b = population[i], population[j]
    if dominates(a.F, b.F):
        return a
    if dominates(b.F, a.F):
        return b
    return a if rng.random() < 0.5 else b


# ---------------------------------------------------------------------------
# Binary operators
# ---------------------------------------------------------------------------

def single_point_crossover(x1, x2, rng, pc=0.9):
    """Single-point crossover on binary vectors."""
    x1 = x1.copy()
    x2 = x2.copy()
    if rng.random() < pc and len(x1) > 1:
        point = rng.integers(1, len(x1))
        c1 = np.concatenate([x1[:point], x2[point:]])
        c2 = np.concatenate([x2[:point], x1[point:]])
        return c1, c2
    return x1, x2


def uniform_crossover(x1, x2, rng, pc=0.9):
    """Uniform crossover on binary vectors."""
    x1 = x1.copy()
    x2 = x2.copy()
    if rng.random() < pc:
        mask = rng.random(len(x1)) < 0.5
        c1 = np.where(mask, x1, x2)
        c2 = np.where(mask, x2, x1)
        return c1, c2
    return x1, x2


def bitflip_mutation(x, rng, pm=None):
    """Bit-flip mutation with probability ``pm`` per gene (default 1/D)."""
    x = x.copy()
    if pm is None:
        pm = 1.0 / len(x)
    flip = rng.random(len(x)) < pm
    x[flip] = 1 - x[flip]
    return x


def ensure_nonempty(mask, rng):
    """Repair an all-zero mask by activating one random feature."""
    if mask.sum() == 0:
        mask = mask.copy()
        mask[rng.integers(len(mask))] = 1
    return mask



def sbx_crossover(x1, x2, rng, pc=1.0, eta=20, low=0.0, high=1.0):
    """Simulated Binary Crossover (distribution index ``eta``, e.g. beta_c=20)."""
    x1 = x1.astype(float).copy()
    x2 = x2.astype(float).copy()
    if rng.random() >= pc:
        return x1, x2
    for i in range(len(x1)):
        if rng.random() < 0.5 and abs(x1[i] - x2[i]) > 1e-14:
            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
            c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            x1[i] = np.clip(c1, low, high)
            x2[i] = np.clip(c2, low, high)
    return x1, x2


def polynomial_mutation(x, rng, pm=None, eta=20, low=0.0, high=1.0):
    """Polynomial mutation (distribution index ``eta``, e.g. beta_m=20)."""
    x = x.astype(float).copy()
    if pm is None:
        pm = 1.0 / len(x)
    for i in range(len(x)):
        if rng.random() < pm:
            u = rng.random()
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
            x[i] = np.clip(x[i] + delta * (high - low), low, high)
    return x
