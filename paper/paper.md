---
title: 'het-ai: A framework-agnostic MLOps training DSL for automated hyperparameter optimisation'
tags:
  - Python
  - machine learning
  - hyperparameter optimisation
  - MLOps
  - data versioning
  - experiment tracking
  - Optuna
  - DVC
  - MLflow
authors:
  - name: Fuxin Liu
    affiliation: 1
  - name: Shixing Qi
    affiliation: 1
  - name: Guiqi Li
    affiliation: 1
  - name: Guangfeng Tang
    affiliation: 1
  - name: Weiwei Yang
    affiliation: 1
  - name: Bowen Chen
    affiliation: 1
  - name: Chen Zhang
    orcid: 0009-0007-7689-5030
    corresponding: true
    affiliation: 1
affiliations:
  - name: Shenzhen HeT Intelligent Control Co., Ltd., Shenzhen, China
    index: 1
date: 19 August 2026
bibliography: paper.bib
---

# Summary

het-ai is a framework-agnostic MLOps training DSL (domain-specific language)
for Python that provides a structured abstraction over the full machine
learning training lifecycle — *data loading, hyperparameter optimisation
(HPO), model training, export, and experiment tracking* — through a single,
consistent interface. Users subclass a `BaseTrainer`, annotate tunable
parameters directly in their `train()` method with a declarative
`@BaseTrainer.search(...)` decorator, and implement three small hooks
(`load_data`, `mock_data`, `export_model`). het-ai then drives the whole
pipeline automatically: it runs an Optuna-powered search over the declared
search space, records every trial, selects the best model, exports it in the
format of the user's choice, and — when the optional `dvc` and `mlflow`
extras are installed — resolves the exact data version that produced a run and
registers the trained model with full data lineage.

het-ai is deliberately framework-agnostic: the same interface works with
PyTorch, TensorFlow, scikit-learn, PyMC, plain NumPy, and arbitrary black-box
external processes. The bundled test suite exercises twelve end-to-end cases
covering classification, regression, clustering, anomaly detection,
time-series forecasting, text classification, Bayesian inference, and black-box
optimisation, demonstrating that the abstraction does not leak framework- or
task-specific assumptions.

# Statement of need

Modern machine learning research and industrial model development share a
common, repetitive workflow: load versioned data, search a hyperparameter
space, train candidate models, compare them, export the winner, and record
everything so the experiment can be reproduced. In practice this workflow is
glued together manually, and the glue is the source of most friction:

- **HPO boilerplate.** Optuna and similar optimisers require objective
  functions, `suggest_*` calls, pruner wiring, and study-management code that
  is duplicated across every experiment. When a research question changes,
  the boilerplate must be rewritten.
- **Broken traceability.** Without a shared data-versioning and tracking layer,
  it is difficult or impossible to answer *"which exact data version produced
  this model and this set of metrics?"* — a prerequisite for reproducible
  research and for auditing model behaviour in regulated settings.
- **Framework lock-in.** Search, tracking, and export utilities are often
  written for one framework (e.g., PyTorch) and do not transfer when a
  researcher switches to scikit-learn, a Bayesian workflow in PyMC, or a
  proprietary black-box training service.

het-ai addresses all three problems with a single abstraction. The target
audience is researchers and ML engineers who repeatedly train models across
different frameworks and need reproducible, traceable HPO without writing
per-project plumbing. het-ai complements rather than replaces existing tools:
it builds on Optuna for the search itself and, when enabled, on DVC for data
versioning and MLflow for experiment tracking, while removing the boilerplate
and unifying the workflow across frameworks.

# State of the field

Several strong tools already exist in this space, and het-ai deliberately
builds on rather than reinvents them.

**Hyperparameter optimisation libraries.** Optuna [@optuna] and related
libraries such as Hyperopt [@hyperopt] and Ray Tune [@raytune] provide
powerful samplers, pruners, and study management. However, they operate at the
level of "define an objective function and optimise it": the surrounding
lifecycle (data loading, model export, experiment logging, data-version
lineage) is left to the user, and each framework needs its own wiring.
scikit-learn's `GridSearchCV`/`RandomizedSearchCV` [@scikit-learn] are
convenient but limited to scikit-learn estimators and shallow search spaces.

**Experiment tracking platforms.** MLflow [@mlflow], Weights & Biases, and
Neptune provide tracking, model registries, and (in MLflow's case) lineage
APIs, but they are reporting back-ends rather than training abstractions:
they do not manage the search loop or the data-loading step, and the logging
code must be written and maintained per project.

**Data version control.** DVC [@dvc] provides Git-based data versioning and
pipeline orchestration but is agnostic to how training and HPO happen, leaving
the integration of "data version → experiment → model" to the user.

**Why not contribute to an existing project instead of building a new one?**
The gap that het-ai fills is not a better sampler or a better tracker but a
*language* for expressing the complete training workflow. A declarative search
space attached to the `train()` method, a universal data container, a
standardised return protocol for single- and multi-objective training, and
automatic DVC→MLflow lineage wiring constitute a distinct abstraction layer
that sits *above* Optuna, DVC, and MLflow. None of the existing tools provides
this layer, and bolting it onto any one of them would couple it to that tool's
ecosystem. het-ai's plugin-style integration points (a `TagResolver` protocol
for data versions, a flavor registry for model logging, and a `select_best_trial`
hook for Pareto-front selection) keep the core independent of any single
framework, which is precisely the property the ecosystem lacks.

# Software design

The design of het-ai is organised around four principles: *declare, don't
script*; *framework-agnostic containers*; *convention over configuration with
explicit escape hatches*; and *integration through pluggable protocols*.

**Declarative search spaces via annotation-time metadata.** Tunable parameters
are declared with `TunableInt`, `TunableFloat`, and `TunableCategorical`
objects inside the `@BaseTrainer.search(...)` decorator on the `train()`
method. These objects are subclasses of `int`, `float`, and `str` that carry
their sampling metadata in a `_meta` attribute; the decorator attaches the
search space to the function object without executing any sampling. A
`__init_subclass__` hook performs *early static validation*: at class
definition time it cross-checks the declared search space against the actual
`train()` signature and raises a `TypeError` on typos before a single trial
runs. At run time, a small sampler translates each `_meta` descriptor into the
corresponding Optuna `suggest_*` call, so users never touch Optuna's API
directly. This separation (annotation metadata vs. runtime sampling) keeps the
core free of any dependency on a particular HPO engine.

**Framework-agnostic containers.** Data flows through a `DataBundle`
container that does not presume any task type, framework, or array format: it
carries `splits`, `feature_list`, `target_list`, and a free-form `meta` dict.
Similarly, `train()` may return any of six standardised forms — a scalar, a
`(metric, artifact)` pair, a `(metric, artifact, metrics_dict)` triple, or
their multi-objective variants wrapped in a `Result` object — and a single
unpacker normalises them. This uniform contract is what allows the same
pipeline to serve PyTorch, scikit-learn, PyMC, NumPy, and black-box processes
without special-casing.

**Convention over configuration, with explicit escape hatches.** Every
`TrainConfig` field is overridable via environment variables (priority:
environment variables > explicit arguments > defaults), which makes container
and CI/CD integration seamless. Users opt into DVC and MLflow integration by
setting optional config fields, and the framework triggers them automatically.
For the decisions that are inherently project-specific — how data directories
are laid out, how versions are merged, how a model should be exported — het-ai
provides hooks (`load_data`, `mock_data`, `export_model`, `select_best_trial`,
`on_study_end`, `before_mlflow_log`, `on_model_registered`) and leaves the
logic to the user, rather than guessing.

**Integration through pluggable protocols.** Data versioning is split into a
`TagResolver` protocol ("which version to use") and a `DVCLoader` ("how to get
the data"), so users can swap in a GitLab resolver, a fixed-version resolver
for reproduction, or an environment-variable resolver for CI without touching
the DVC mechanics. MLflow model logging is dispatched through a file-extension
flavor registry (`onnx`, `pt`, `pkl`, `joblib`, `tflite`, ...) that users can
extend with `register_flavor()`, replacing what would otherwise be a hard-coded
`if/elif` chain. A `dry_run()` mode validates the entire pipeline locally with
synthetic data before any production run, so configuration errors and broken
hooks surface in seconds rather than after an expensive remote HPO run.

# Research impact statement

het-ai is developed and used within an active industrial research program at
Shenzhen HeT Intelligent Control Co., Ltd., where it supports the
development and deployment of machine learning models for edge-side vertical
applications — including predictive maintenance on microcontrollers, visual
quality inspection on embedded systems, time-series forecasting, and Bayesian
modelling.

**Open-set coverage and reproducibility.** The framework-agnostic claim is
validated by twelve representative training scenarios that ship with the
repository under `tests/cases/`, covering five ML frameworks and seven task
types (Table 1). Every scenario completes the full pipeline — mock data,
training, and model export — through the framework's `dry_run()` mechanism,
with no DVC, object-storage, or MLflow infrastructure required. The entire
suite runs end-to-end in under a minute on a standard workstation and is
therefore reproducible by any reviewer or research group.

| # | Scenario | Framework | Export format | Dry-run | Score |
|---|----------|-----------|---------------|---------|-------|
| 1 | Tabular classification | PyTorch | ONNX | 2.3 s | 0.367 (acc) |
| 2 | Multi-target classification | PyTorch | ONNX | 2.3 s | (0.333, 0.667) |
| 3 | Image segmentation | TensorFlow / YOLO | TFLite | 0.0 s† | 0.0 (mock) |
| 4 | Gradient boosting | scikit-learn | joblib | 0.2 s | 0.301 (f1) |
| 5 | Unsupervised clustering (Pareto) | scikit-learn | joblib | 5.1 s | (0.621, 219.6) |
| 6 | Bayesian inference | PyMC | NetCDF | 5.1 s | −158.9 (lp) |
| 7 | Logistic regression (pure NumPy) | NumPy | JSON | 0.0 s | 0.233 (acc) |
| 8 | Black-box external process | subprocess | — | 0.0 s† | 0.750 (mock) |
| 9 | Regression (Pareto) | scikit-learn | joblib | 1.0 s | (0.755, −0.560) |
| 10 | Time-series forecasting | PyTorch (LSTM) | ONNX | 4.0 s | −0.402 (mse) |
| 11 | Anomaly detection (Pareto) | scikit-learn | joblib | 0.6 s | (0.400, 1.000) |
| 12 | Text classification | PyTorch (CNN) | ONNX | 3.5 s | 0.300 (acc) |

† Cases 3 and 8 deliberately short-circuit in `dry_run()`: they exercise the
pipeline with synthetic results rather than launching heavyweight YOLO training
or an external binary during validation. Scores are deterministic — each case
fixes the global random seed via `config.random_state`. Run times were measured
on a Windows 11 workstation (Python 3.12, conda environment) and vary with
hardware.

**Integration boilerplate reduction.** Compared with a "vanilla" baseline that
manually wires Optuna, DVC, and MLflow (study management, logging
orchestration, DVC coordination, trial directory handling, and model export),
het-ai reduces integration code by an average of 59.6% across the twelve
scenarios (Table 2). The saved lines are precisely the per-project plumbing
that would otherwise be copy-pasted into every new training task.

| # | Scenario | het-ai (LOC) | Vanilla (LOC) | Reduction |
|---|----------|--------------|---------------|-----------|
| 1 | PyTorch tabular | 118 | 287 | 58.9% |
| 2 | PyTorch multi-target | 145 | 338 | 57.1% |
| 3 | TensorFlow image | 180 | 402 | 55.2% |
| 4 | sklearn GBT | 98 | 256 | 61.7% |
| 5 | Clustering (Pareto) | 112 | 310 | 63.9% |
| 6 | PyMC Bayesian | 120 | 298 | 59.7% |
| 7 | NumPy logistic | 108 | 275 | 60.7% |
| 8 | Black-box process | 102 | 264 | 61.4% |
| 9 | sklearn regression | 110 | 290 | 62.1% |
| 10 | PyTorch LSTM | 152 | 356 | 57.3% |
| 11 | sklearn anomaly | 118 | 305 | 61.3% |
| 12 | PyTorch TextCNN | 168 | 385 | 56.4% |
| **Average** | | **128** | **314** | **59.6%** |

*The "vanilla" baselines in Table 2 are reference implementations written by
the authors that manually wire Optuna, DVC, and MLflow; the comparison is
indicative of the boilerplate removed by the DSL rather than a formal
benchmark, and absolute figures depend on implementation style.*

The framework is actively used in the company's research workflows across the
edge-AI domains listed above and is released openly with a public development
history, tagged releases, automated tests, and documentation, so that other
research groups can adopt it for reproducible, framework-agnostic
hyperparameter optimisation.

# AI usage disclosure

Generative AI tools (e.g., GitHub Copilot) were used during the development of
this software to assist with code generation, refactoring, documentation
writing, and test scaffolding. All AI-assisted outputs were reviewed, edited,
validated, and verified by the human authors, who made the core design
decisions (including the declarative search-space abstraction, the
standardised training return protocol, the pluggable `TagResolver` protocol,
and the flavor registry for model logging) and are fully responsible for the
correctness, originality, licensing, and compliance of all submitted
materials.

# Acknowledgements

We acknowledge the open-source communities behind Optuna, DVC, MLflow,
scikit-learn, PyTorch, and PyMC, whose ecosystems make this work possible.

# References
