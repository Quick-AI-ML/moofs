
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from moofs import (
    ALGORITHMS,
    FeatureSelectionProblem,
    MOFSSelector,
    compare,
    coverage,
    hv,
    igd,
    merge_reference_front,
    nfs,
    plot_fronts,
    plot_pareto_front,
    plot_selector,
    relieff,
)
from moofs.core.dominance import dominates


@pytest.fixture(scope="module")
def toy_data():
    rng = np.random.RandomState(0)
    n, d = 80, 10
    X = pd.DataFrame(rng.rand(n, d), columns=[f"f{i}" for i in range(d)])
    y = pd.Series(((X["f0"] + X["f1"] + 0.3 * X["f2"]) > 1.15).astype(int))
    return X, y



@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_algorithm_runs(name, toy_data):
    X, y = toy_data
    problem = FeatureSelectionProblem(X, y)
    res = ALGORITHMS[name](problem, pop_size=10, max_evals=80, seed=42).run()
    F = res.F
    assert len(F) > 0 and F.shape[1] == 2
    for i in range(len(F)):
        for j in range(len(F)):
            if i != j:
                assert not dominates(F[i], F[j])


def test_reproducibility(toy_data):
    X, y = toy_data
    runs = []
    for _ in range(2):
        problem = FeatureSelectionProblem(X, y)
        r = ALGORITHMS["mofs-rfga"](problem, pop_size=8, max_evals=60,
                                    seed=7).run()
        runs.append(np.sort(r.F, axis=0))
    assert np.allclose(runs[0], runs[1])


def test_crossover_interpretations(toy_data):
    X, y = toy_data
    from moofs import MOFSRFGA
    for interp in ("figure", "pseudocode"):
        problem = FeatureSelectionProblem(X, y)
        r = MOFSRFGA(problem, pop_size=8, max_evals=40, seed=1,
                     interpretation=interp).run()
        assert len(r.front) > 0
    with pytest.raises(ValueError):
        MOFSRFGA(FeatureSelectionProblem(X, y), interpretation="nope")


def test_relieff_scores(toy_data):
    X, y = toy_data
    w = relieff(X, y, n_neighbors=5, sample_size=40, random_state=0)
    assert w.shape == (10,)
    assert w[[0, 1]].mean() > w[5:].mean()


def test_selector_fit_transform(toy_data):
    X, y = toy_data
    sel = MOFSSelector(algorithm="mofs-rfga", pop_size=10, max_evals=80,
                       random_state=0)
    Xr = sel.fit_transform(X, y)
    assert Xr.shape[0] == X.shape[0]
    assert 1 <= Xr.shape[1] <= X.shape[1]
    assert sel.support_.sum() == Xr.shape[1]
    assert sel.pareto_front_.shape[1] == 2
    names = sel.get_feature_names_out()
    assert len(names) == Xr.shape[1]
    assert set(names) <= set(X.columns)


@pytest.mark.parametrize("strategy", ["knee", "min_error", "min_features"])
def test_selector_strategies(strategy, toy_data):
    X, y = toy_data
    sel = MOFSSelector(algorithm="nsga2", pop_size=8, max_evals=48,
                       strategy=strategy, random_state=0).fit(X, y)
    F = sel.pareto_front_
    err, size = sel.selected_objectives_
    if strategy == "min_error":
        assert err == F[:, 0].min()
    if strategy == "min_features":
        assert size == F[:, 1].min()


def test_selector_sklearn_compat(toy_data):
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    X, y = toy_data
    sel = MOFSSelector(pop_size=8, max_evals=48, random_state=0)
    clone(sel)  # get_params/set_params round-trip
    pipe = Pipeline([("fs", sel), ("clf", KNeighborsClassifier(3))])
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.5


def test_selector_numpy_input(toy_data):
    X, y = toy_data
    sel = MOFSSelector(pop_size=8, max_evals=48, random_state=0)
    Xr = sel.fit_transform(X.values, y.values)
    assert Xr.shape[1] >= 1



def test_igd_platemo_mean():
    ref = np.array([[0.0, 0.0], [2.0, 2.0]])
    front = np.array([[0.0, 1.0]])
    # distances: 1.0 and sqrt(5) -> mean
    assert igd(ref, front) == pytest.approx((1.0 + np.sqrt(5)) / 2)


def test_hv_platemo_normalized():
    ref = np.array([[0.0, 10.0], [10.0, 0.0]])
    # A single point at the ideal corner: normalized to (0,0), HV = 1.0
    assert hv(np.array([[0.0, 0.0]]), ref) == pytest.approx(1.0)
    # A point outside the 1.1-scaled box is discarded
    assert hv(np.array([[20.0, 20.0]]), ref) == 0.0


def test_coverage_weak_dominance():
    A = np.array([[1.0, 1.0]])
    B = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]])
    # weak dominance: covers the equal point and (2,2), not (0.5,3)
    assert coverage(A, B) == pytest.approx(2 / 3)


def test_compare_table(toy_data):
    X, y = toy_data
    results = {}
    for name in ALGORITHMS:
        problem = FeatureSelectionProblem(X, y)
        results[name] = ALGORITHMS[name](problem, pop_size=8, max_evals=48,
                                         seed=0).run()
    table = compare(results)
    assert set(table["algorithm"]) == set(ALGORITHMS)
    assert {"IGD", "HV", "NFS"}.issubset(table.columns)
    ref = merge_reference_front(*results.values())
    assert len(ref) >= 1
    assert nfs(ref) == len(ref)



def test_plotting(toy_data):
    X, y = toy_data
    problem = FeatureSelectionProblem(X, y)
    res = ALGORITHMS["mofs-rfga"](problem, pop_size=8, max_evals=48,
                                  seed=0).run()
    ax = plot_pareto_front(res, annotate=True)
    assert ax.get_xlabel().startswith("F2")
    ax2 = plot_fronts({"MOFS-RFGA": res}, reference=True)
    assert len(ax2.get_legend().get_texts()) >= 2
    sel = MOFSSelector(pop_size=8, max_evals=48, random_state=0).fit(X, y)
    ax3 = plot_selector(sel)
    assert ax3 is not None
