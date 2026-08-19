# Changelog

All notable changes to het-ai are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- JOSS submission materials: `paper/paper.md`, `paper/paper.bib`,
  `CITATION.cff`, `.zenodo.json`.
- Community files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  `CHANGELOG.md`, GitHub issue and pull-request templates.
- Continuous integration now runs on every push and pull request to `main`.

### Changed

- All docstrings in `src/` translated to English (previously Chinese),
  consistent with the Sphinx-generated API documentation.

## [1.0.0] - 2026-08-19

### Added

- `het_ai.studio` core:
  - `BaseTrainer` with declarative `@BaseTrainer.search(...)` search spaces.
  - `TunableInt` / `TunableFloat` / `TunableCategorical` tunable types.
  - Single- and multi-objective HPO via Optuna, with Pareto-front selection.
  - `TrainConfig` with environment-variable-overridable fields.
  - `dry_run()` local pipeline validation with `mock_data()`.
  - `DataBundle` framework-agnostic data container and `TrainResult`
    standardised result.
  - Trial isolation, pruning via `report()`, and aggregated failure reporting.
- `het_ai.dvc`: `DVCLoader`, `TagResolver` protocol, `GitHubTagResolver`,
  `FixedTagResolver`, and SeaweedFS/S3 remote support.
- `het_ai.mlflow`: `MLflowRunLogger` with automatic params/metrics/lineage
  logging, flavor registry, and model registration.
- 12 end-to-end example cases across PyTorch, TensorFlow, scikit-learn, PyMC,
  NumPy, and black-box processes.
- Sphinx documentation with ReadTheDocs configuration.
- CI (pytest + ruff) and trusted-publishing PyPI workflow.

[Unreleased]: https://github.com/HeT-FTI/het-ai
[1.0.0]: https://github.com/HeT-FTI/het-ai/releases/tag/v1.0.0
