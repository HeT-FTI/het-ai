het-ai
======

.. image:: https://img.shields.io/badge/python-3.10%2B-blue
   :alt: Python 3.10+
.. image:: https://img.shields.io/badge/license-Apache%202.0-green
   :alt: Apache 2.0
.. image:: https://img.shields.io/badge/HPO-Optuna-orange
   :alt: Optuna

**het-ai** is a framework-agnostic MLOps training DSL that provides a structured
abstraction for the full training lifecycle, powered by `Optuna <https://optuna.org/>`_-driven
automated hyperparameter optimisation (HPO).

Whether you work with PyTorch, TensorFlow, scikit-learn, PyMC, plain NumPy, or any
black-box external process, het-ai lets you express the complete
*data loading → hyperparameter search → model training → export* pipeline through
a single, consistent interface.

Developed and maintained by
`Shenzhen HeT Intelligent Control Co., Ltd. <https://github.com/HeT-FTI>`_

----

Key Features
------------

- **Framework-agnostic** — optional dependency groups bring first-class support for
  PyTorch, TensorFlow, scikit-learn, PyMC, and arbitrary black-box processes.
- **Declarative search spaces** — annotate tunable parameters directly in the
  ``train()`` signature with ``@BaseTrainer.search(...)``. No Optuna boilerplate required.
- **Single- and multi-objective HPO** — native Pareto-front support for multi-metric
  optimisation; override ``select_best_trial()`` for custom selection logic.
- **Environment-variable-driven config** — every ``TrainConfig`` field can be
  overridden by an environment variable, making container and CI/CD integration seamless.
- **Dry-run validation** — verify the entire pipeline locally without production data
  or a remote training server using a single ``trainer.dry_run()`` call.
- **DVC integration** — the ``load_data`` hook accepts a DVC data root directory out
  of the box.
- **ONNX model export** — exports to ONNX by default; easily extended to any custom
  format via ``export_model()``.

----

Installation
------------

Core package (minimal dependencies):

.. code-block:: bash

   pip install het-ai

Install optional framework extras as needed:

.. code-block:: bash

   # PyTorch + ONNX export
   pip install "het-ai[torch]"

   # TensorFlow
   pip install "het-ai[tensorflow]"

   # scikit-learn
   pip install "het-ai[sklearn]"

   # Bayesian modelling with PyMC
   pip install "het-ai[bayes]"

   # All dependencies required to run the bundled examples
   pip install "het-ai[examples]"

----

Quick Start
-----------

Subclass ``BaseTrainer``, declare your search space, and implement three hooks:

.. code-block:: python

   from het_ai.studio import BaseTrainer, TrainConfig
   from het_ai.studio.types import TunableFloat, TunableInt


   class MyTrainer(BaseTrainer):

       @BaseTrainer.search(
           lr=TunableFloat(1e-4, 1e-2, log=True),
           hidden=TunableInt(64, 512),
       )
       def train(self, data, lr, hidden):
           """Core training logic. Called once per Optuna trial."""
           model = build_model(hidden)
           score = fit(model, data, lr=lr)
           return score, model          # (metric value, exportable artifact)

       def load_data(self, dvc_data_root):
           """Load and return a DataBundle from the DVC data root."""
           ...

       def mock_data(self):
           """Return synthetic data for dry-run validation (no real data needed)."""
           ...

       def export_model(self, artifact, export_dir):
           """Persist the model artifact (e.g. ONNX, pickle) to export_dir."""
           ...


   # ── Local validation (no production environment required) ────────────────
   trainer = MyTrainer(TrainConfig(n_trials=2))
   report  = trainer.dry_run()
   print(report)
   # {'score': 0.91, 'elapsed': 1.23, 'export_path': '/tmp/...'}

   # ── Full HPO run (loads real data, launches Optuna study) ────────────────
   result = trainer.run()

----

Multi-Objective Optimisation
-----------------------------

Declare an ``objectives`` dict to enable Pareto-front HPO.
The framework switches Optuna to multi-directional mode automatically:

.. code-block:: python

   from het_ai.studio.types import Result


   class MultiObjectiveTrainer(BaseTrainer):

       objectives = {
           "accuracy":      "maximize",
           "model_size_mb": "minimize",
       }

       @BaseTrainer.search(lr=TunableFloat(1e-4, 1e-2))
       def train(self, data, lr):
           ...
           return Result(accuracy=acc, model_size_mb=size), model

Override ``select_best_trial(pareto_front)`` to apply a custom trade-off policy
(default: highest value on the first objective).

----

TrainConfig Reference
---------------------

All fields support environment-variable overrides.
Precedence: **env var > explicit argument > default value**.

.. list-table::
   :header-rows: 1
   :widths: 22 24 38 16

   * - Field
     - Environment Variable
     - Description
     - Default
   * - ``n_trials``
     - ``N_TRIALS``
     - Number of HPO trials
     - ``100``
   * - ``direction``
     - ``OPTUNA_DIRECTION``
     - Optimisation direction (``maximize`` / ``minimize``)
     - ``maximize``
   * - ``n_jobs``
     - ``OPTUNA_N_JOBS``
     - Number of parallel trials
     - ``1``
   * - ``timeout``
     - ``OPTUNA_TIMEOUT``
     - Search timeout in seconds
     - ``None``
   * - ``pruner``
     - ``OPTUNA_PRUNER``
     - Early-stopping strategy (``median`` / ``threshold`` / ``none``)
     - ``median``
   * - ``storage``
     - ``OPTUNA_STORAGE``
     - Optuna persistence storage URL
     - ``None``
   * - ``dvc_data_root``
     - ``DVC_DATA_ROOT``
     - Root path of the DVC dataset
     - ``dvc_data``
   * - ``trial_root_dir``
     - ``TRIAL_ROOT_DIR``
     - Directory for per-trial artefacts
     - auto-generated
   * - ``log_level``
     - ``LOG_LEVEL``
     - Logging verbosity
     - ``INFO``

----

Supported Training Scenarios
-----------------------------

The bundled test suite covers eight representative scenarios, all runnable via
``dry_run()`` without any external dependencies:

.. list-table::
   :header-rows: 1
   :widths: 8 30 30 32

   * - #
     - Scenario
     - Framework
     - Notes
   * - 1
     - Tabular single-target classification
     - PyTorch
     - ONNX export
   * - 2
     - Multi-target classification
     - PyTorch
     - ``Result`` multi-metric return
   * - 3
     - Image classification
     - TensorFlow / Keras
     - Pillow preprocessing
   * - 4
     - Gradient-boosted trees
     - scikit-learn
     - joblib serialisation
   * - 5
     - Unsupervised clustering (dual-objective)
     - NumPy / scikit-learn
     - Pareto-front HPO
   * - 6
     - Bayesian mixture model
     - PyMC + ArviZ
     - Posterior sampling
   * - 7
     - Hand-written logistic regression
     - Pure NumPy
     - Zero ML-framework dependency
   * - 8
     - Black-box external process
     - Subprocess
     - Wraps any CLI training command

----

Development & Testing
---------------------

.. code-block:: bash

   # Clone and install in editable mode with all development dependencies
   git clone https://github.com/HeT-FTI/het-ai.git
   cd het-ai
   pip install -e ".[dev,examples]"

   # Run the full test suite (8 dry-run smoke tests)
   pytest tests/ -v

----

License
-------

- **Source code** — `Apache License 2.0 <LICENSE>`_
- **Documentation** — `Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) <https://creativecommons.org/licenses/by-nc/4.0/>`_

For commercial use of the documentation, please contact HeT directly.

Copyright © 2025 Shenzhen HeT Intelligent Control Co., Ltd.
