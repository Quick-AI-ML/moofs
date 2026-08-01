# NSGA-II : Non-dominated Sorting Genetic Algorithm - II

The classic multi-objective evolutionary algorithm, used both as a
standalone baseline and as the environmental-selection engine inside
[MOFS-RFGA](mofs_rfga.md). This page describes the binary-encoded variant
used for feature selection.

## How it works

NSGA-II searches for the Pareto front by evolving a population of binary
feature masks through three mechanisms:

**Fast non-dominated sorting.** Every generation, the combined parent and
offspring population is ranked into fronts: front 1 contains the
non-dominated solutions, front 2 the solutions dominated only by front 1,
and so on. This ranking is what lets the algorithm push the whole
population toward the Pareto-optimal boundary rather than toward a single
point.

**Crowding distance.** Within a front, solutions are also ranked by how
isolated they are in objective space, the average distance to their two
neighbors along each objective. Solutions at the extremes of a front get an
infinite distance and are always kept. This is the mechanism that keeps the
final front spread out across the whole trade-off (few features / high
accuracy to many features / low error) instead of clustering in one region.

**Elitist environmental selection.** Parents and offspring are merged
before selection, so a good solution can never be lost to a worse one by
chance. The next population is filled front by front; if a front doesn't
fit entirely, it is truncated by crowding distance, keeping the least
crowded (most diverse) solutions first.

**Variation operators**, applied to binary masks: a crossover (single-point
or uniform, `pc` probability) recombines two parents selected by binary
tournament on (rank, crowding distance); a bit-flip mutation then toggles
each gene independently with probability `pm`.

## Usage

```python
from moofs import FeatureSelectionProblem, NSGA2

problem = FeatureSelectionProblem(X, y)
result = NSGA2(problem, pop_size=100, max_evals=300_000, seed=0).run()
```

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 60 | Population size N |
| `max_evals` | 20000 | Budget in evaluations |
| `pc` | 0.9 | Crossover probability |
| `pm` | 1/D | Per-gene mutation probability |
| `crossover` | `"single_point"` | `"single_point"` or `"uniform"` |

`pc = 0.9` and `pm = 1/D` are the canonical parameters used across the MOFS
literature, including as the baseline setting in the MOFS-RFGA paper.

## Reference

K. Deb, A. Pratap, S. Agarwal, T. Meyarivan. **A fast and elitist
multiobjective genetic algorithm: NSGA-II.** *IEEE Transactions on
Evolutionary Computation*, 6(2), 2002. [DOI: 10.1109/4235.996017](https://doi.org/10.1109/4235.996017)