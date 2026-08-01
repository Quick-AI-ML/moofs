"""moofs quickstart: sklearn API + front comparison on breast cancer."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from moofs import (FeatureSelectionProblem, MOFSRFGA, NSGA2, MOFSSelector,
                   compare, plot_fronts, plot_selector)

data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0)

# --- 1. sklearn way -------------------------------------------------------
sel = MOFSSelector(algorithm="mofs-rfga", pop_size=20, max_evals=600,
                   random_state=0)
Xr = sel.fit_transform(X_train, y_train)
print("Selected features:", list(sel.get_feature_names_out()))
print("Chosen trade-off  [error %, size]:", sel.selected_objectives_)
plot_selector(sel).figure.savefig("selected.png", dpi=150)

# --- 2. research way ------------------------------------------------------
r1 = MOFSRFGA(FeatureSelectionProblem(X_train, y_train),
              pop_size=20, max_evals=600, seed=0).run()
r2 = NSGA2(FeatureSelectionProblem(X_train, y_train),
           pop_size=20, max_evals=600, seed=0).run()
print(compare({"MOFS-RFGA": r1, "NSGA-II": r2}).round(4).to_string(index=False))
plot_fronts({"MOFS-RFGA": r1, "NSGA-II": r2},
            reference=True).figure.savefig("fronts.png", dpi=150)
