# MOFS-RFGA

**A feature selection approach based on NSGA-II with ReliefF**
Y. Xue, H. Zhu, F. Neri — *Applied Soft Computing* 134 (2023) 109987 —
[DOI](https://doi.org/10.1016/j.asoc.2023.109987)

A hybrid filter/wrapper method. ReliefF scores every feature once; the
scores then guide the population initialization, a 3-parent ("3-to-1")
crossover and the mutation, while NSGA-II environmental selection drives the
multi-objective search. Per the paper, the algorithm requires no pre-set
parameters beyond the population size and evaluation budget.

## Usage

```python
from moofs import FeatureSelectionProblem, MOFSRFGA

problem = FeatureSelectionProblem(X, y)
result = MOFSRFGA(problem, pop_size=100, max_evals=300_000, seed=0).run()
```

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 60 | Population size N (paper: 100) |
| `max_evals` | 20000 | Budget in evaluations (paper: 300000) |
| `sc` | None | Feature scores; computed with built-in ReliefF if omitted |
| `D_init` | n_var | Upper bound on features activated at initialization |
| `interpretation` | `"figure"` | Crossover semantics (see below) |

## Documented ambiguity: the crossover semantics

The paper's Fig. 1 and its Algorithm 3 contradict each other on the 3-to-1
crossover. Fig. 1 removes the **worse**-scored gene among two candidates of
S2 and adds the **better**-scored gene from S1 — consistent with the
mutation operator's prose and the method's score-guided philosophy.
Algorithm 3, read literally, does the opposite.

`moofs` defaults to the Fig. 1 reading (`interpretation="figure"`) and
provides `interpretation="pseudocode"` for the literal Algorithm 3 variant,
so both readings can be compared explicitly.
