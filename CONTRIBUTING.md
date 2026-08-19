# Contributing to het-ai

Thank you for your interest in contributing to het-ai! This document covers
how to set up a development environment, run the tests, build the
documentation, and submit changes. The same content is available in the
[online documentation](https://het-ai.readthedocs.io/en/latest/contributing.html).

## Code of Conduct

Please note that this project is released with a
[Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this
project you agree to abide by its terms.

## Ways to contribute

- **Report a bug** — open an issue with a minimal reproduction.
- **Request a feature** — open an issue describing the problem you are solving.
- **Fix a bug or add a feature** — open a pull request (see below).
- **Improve documentation** — fix typos, expand guides, or add examples.
- **Add an example case** — extend the `tests/cases/` suite with a new
  end-to-end scenario.

## Development environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/HeT-FTI/het-ai.git
   cd het-ai
   ```

2. **Create a virtual environment and install in editable mode:**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate

   pip install -e ".[dev]"
   ```

3. **Install framework extras as needed for tests:**

   ```bash
   pip install -e ".[examples]"
   ```

## Running tests

The test suite uses `pytest`. All 12 example cases are parametrised through
`tests/test_simulate_all.py`, which runs `dry_run()` on each case:

```bash
# Run all tests
python -m pytest tests/test_simulate_all.py -v

# Run a specific case
python -m pytest tests/test_simulate_all.py -k "case1" -v

# With coverage
pip install pytest-cov
python -m pytest tests/test_simulate_all.py --cov=het_ai --cov-report=term
```

## Linting

We use [ruff](https://docs.astral.sh/ruff/):

```bash
pip install ruff
ruff check src/
```

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- **Docstrings are written in English** (project convention), Google-style
  formatting. `napoleon` in Sphinx renders them automatically.
- Keep imports organised: standard library → third-party → local.
- Use type hints for all public methods and functions.

## Adding a new example case

1. Create `tests/cases/caseNN_descriptive_name.py` following the naming
   convention.
2. Subclass `het_ai.studio.base.BaseTrainer` and implement:
   - `load_data()`
   - `mock_data()`
   - `train()` (decorated with `@BaseTrainer.search(...)`)
   - `export_model()` (if needed)
3. Add the new case to the parametrised test in `tests/test_simulate_all.py`.

## Building the documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/_build/html
```

## Submitting changes

1. Create a branch from `main`.
2. Make your changes and add tests.
3. Run `pytest` and `ruff` locally and make sure everything passes.
4. Open a pull request. Describe the motivation for the change and any design
   decisions you made.

## Release process

Releases are tagged with `vX.Y.Z` (e.g., `v1.0.0`). Pushing a tag triggers the
CI workflow that builds the package and publishes it to PyPI via trusted
publishing. A Zenodo integration (via the GitHub-Zenodo webhook) creates an
archived snapshot and DOI for each tagged release.
