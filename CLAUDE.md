# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`fselect` is a PyPI package implementing the A-RANK feature-selection algorithm
(Dash & Liu, "Feature Selection for Clustering"). It ranks the continuous features of a
dataframe by an entropy measure, so the most informative features for downstream
clustering can be selected. The package lives in `src/fselect/` and targets Python
3.10–3.14, managed with **uv** + a **hatchling** build backend (PEP 621 `pyproject.toml`).

## Commands

Uses [uv](https://docs.astral.sh/uv/) for everything:

```bash
uv venv            # create the virtualenv (pinned by .python-version, currently 3.12)
uv sync            # install runtime + dev deps (pytest) from uv.lock
uv run pytest      # run the suite
uv run pytest tests/test_fselect.py::test_rank_features   # a single test
uv build           # build sdist + wheel via hatchling
```

Test a specific interpreter (uv fetches it if missing — this is how the CI matrix runs):

```bash
uv run --python 3.14 pytest
```

CI (`.github/workflows/ci.yml`) runs the suite across 3.10–3.14 on push/PR.
Publishing is automated (`.github/workflows/python-publish.yml`): on a published GitHub
**release** it runs `uv build` and uploads via `pypa/gh-action-pypi-publish` using the
`PYPI_API_TOKEN` secret. Bump `version` in `pyproject.toml` before cutting a release.

## Architecture

The whole algorithm lives in one module; the package is a thin src-layout wrapper:

- `src/fselect/core.py` — the entire implementation (pandas-only). `rank_features`
  implements A-RANK by a **leave-one-out** strategy: for each feature it drops that feature
  (or its correlated group), computes the entropy of the *remaining* columns, and uses that
  as the feature's score; results are sorted by entropy descending and given a 1-based
  `rank` (higher entropy-without-the-feature ⇒ more important). `compute_entropy` builds a
  pairwise-distance matrix, calibrates `alpha` so the mean distance maps to similarity 0.5,
  forms `exp(-alpha*dist)`, then sums per-pair Shannon-style entropy and divides by 2 (the
  matrix is symmetric, so each pair is double counted). `get_correlated_columns` maps each
  column to those whose |correlation| exceeds the threshold (always including itself); only
  used when `remove_correlated_columns=True`.
- `src/fselect/__init__.py` — re-exports `rank_features`, `compute_entropy` and
  `get_correlated_columns` from `core`.

## Invariants and gotchas

- Input must be **continuous and normalized**; the entropy math assumes numeric data and
  the similarity calibration is scale-sensitive.
- `rank_features` returns a dataframe with columns `feature`, `entropy`, `rank`.
- `rank_features` raises `Exception("Empty Dataframe!")` if dropping a feature (or its
  correlated group) leaves zero columns — i.e. a single-feature input, or a single
  non-correlated feature when `remove_correlated_columns=True`.
- `compute_entropy` legitimately produces `log2(0)`/NaN warnings on the similarity-matrix
  diagonal; `np.nansum` is what drops those terms. The warnings are expected, not bugs.
- **scipy version fork (non-obvious):** `pyproject.toml` carries
  `scipy>=1.16; python_version >= '3.11'`. fselect doesn't use scipy directly — this exists
  because the older scipy that still supports 3.10 has no cp314 wheel, so without the marker
  uv would build scipy from source on 3.14. The marker makes uv fork the lock (older scipy
  for 3.10, wheel-backed newer scipy for ≥3.11). Don't remove it while 3.10 and 3.14 are
  both supported.
- pytest only discovers `test_*.py`; the test file is `tests/test_fselect.py` (renamed from
  the old `tests/tests.py`, which auto-discovery skipped).
