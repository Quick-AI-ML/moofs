# NSGA-II

**A fast and elitist multiobjective genetic algorithm: NSGA-II**
K. Deb, A. Pratap, S. Agarwal, T. Meyarivan — *IEEE Transactions on
Evolutionary Computation* 6(2), 2002 —
[DOI](https://doi.org/10.1109/4235.996017)

The classic multi-objective baseline, here with binary encoding for feature
selection: fast non-dominated sorting, crowding-distance diversity, elitist
environmental selection. Canonical parameters from the MOFS literature:
Pc = 0.9, Pm = 1/D.

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
