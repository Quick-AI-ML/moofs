# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-08-01

### Added
- MOFS-RFGA (Xue, Zhu & Neri, 2023) with ReliefF-guided initialization,
  3-to-1 crossover and score-guided mutation. Documented `interpretation`
  flag ("figure" default / "pseudocode") for the paper's crossover ambiguity.
- NSGA-II (Deb et al., 2002) with binary encoding for feature selection.
- `MOFSSelector`: scikit-learn compatible selector (fit/transform,
  Pipeline-ready, knee / min_error / min_features strategies).
- `FeatureSelectionProblem`: KNN(k=3) + 3-fold CV protocol with evaluation
  caching.
- Built-in ReliefF filter (`moofs.relieff`).
- PlatEMO-compatible metrics: IGD, normalized HV, set coverage (weak
  dominance), NFS, spacing, reference-front construction, `compare()` table.
- Pareto-front visualization: `plot_pareto_front`, `plot_fronts`,
  `plot_selector`.

### Planned
- SparseEA, NSGA-II/SDR, SPEA2, MOEA/D, NSPSOFS, CMDPSOFS.
- Dataset loaders for the 20 UCI benchmarks of the reference paper.
- Numerical validation campaign against published tables.
