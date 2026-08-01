# Contributing to moofs

Thanks for your interest! Contributions of new algorithms, bug fixes and
documentation are welcome.

## Development setup

```bash
git clone https://github.com/Herman-Motcheyo/moofs.git
cd moofs
pip install -e .[dev]
pytest
```

## Adding an algorithm

1. Subclass `moofs.Algorithm` in `src/moofs/algorithms/<name>.py` and
   implement `_run()` returning `self._result(population)`.
2. Use the canonical parameters of the source paper as defaults, and cite
   the paper in the module docstring (authors, venue, year, DOI).
3. If the paper is ambiguous on any point, document the ambiguity and expose
   an explicit flag rather than silently choosing.
4. Register the class in `moofs.ALGORITHMS` and, if applicable, in
   `MOFSSelector`.
5. Add smoke tests in `tests/` (runs end-to-end, front is mutually
   non-dominated, seeded runs are reproducible).
6. Add a documentation page under `docs/algorithms/`.

## Pull requests

- One algorithm or fix per PR.
- `pytest` must pass; keep new tests fast (< a few seconds each).
- Follow the existing code style (PEP 8, NumPy-style docstrings).
