# Getting started

## The scikit-learn way

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from moofs import MOFSSelector, plot_selector

data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0)

selector = MOFSSelector(algorithm="mofs-rfga", pop_size=60, max_evals=5000,
                        strategy="knee", random_state=0)
X_train_r = selector.fit_transform(X_train, y_train)
X_test_r = selector.transform(X_test)

print(selector.get_feature_names_out())
print(selector.pareto_front_)      # [error %, subset size] per solution
plot_selector(selector)            # front + highlighted choice
```

The `strategy` parameter picks the final subset from the front:
`"knee"` (best normalized trade-off, default), `"min_error"`, or
`"min_features"`.

## The research way

```python
from moofs import (FeatureSelectionProblem, MOFSRFGA, NSGA2,
                   compare, plot_fronts)

problem = FeatureSelectionProblem(X_train, y_train)
r1 = MOFSRFGA(problem, pop_size=60, max_evals=5000, seed=0).run()

problem2 = FeatureSelectionProblem(X_train, y_train)
r2 = NSGA2(problem2, pop_size=60, max_evals=5000, seed=0).run()

print(compare({"MOFS-RFGA": r1, "NSGA-II": r2}))
plot_fronts({"MOFS-RFGA": r1, "NSGA-II": r2}, reference=True)
```

!!! note "Evaluation budget"
    `max_evals` counts objective-function evaluations (maxFEs), the standard
    budget unit of the MOFS literature. Evaluations are cached, and cache
    hits still count, so budgets remain comparable across algorithms.
