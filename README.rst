het-ai
======

.. |badge1| image:: https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white
   :target: https://www.python.org/downloads/

.. |badge2| image:: https://img.shields.io/badge/license-Apache%202.0-green
   :target: https://www.apache.org/licenses/LICENSE-2.0

.. |badge3| image:: https://img.shields.io/badge/HPO-Optuna-blue?logo=optuna
   :target: https://optuna.org/

|badge1| |badge2| |badge3|

**het-ai** is a framework-agnostic MLOps training DSL that provides a structured
abstraction for the full training lifecycle, powered by `Optuna <https://optuna.org/>`_-driven
automated hyperparameter optimisation (HPO).

Whether you work with PyTorch, TensorFlow, scikit-learn, PyMC, plain NumPy, or any
black-box external process, het-ai lets you express the complete
*data loading → hyperparameter search → model training → export* pipeline through
a single, consistent interface.

When combined with the optional ``dvc`` and ``mlflow`` extras, het-ai becomes a
**data-version-driven platform**: a new DVC release tag on your data repository
automatically drives a full HPO run, and every MLflow experiment is tagged with the
exact data version that produced it — closing the data → experiment → model
traceability loop.

Developed and maintained by
`Shenzhen HeT Intelligent Control Co., Ltd. <https://github.com/HeT-FTI>`_

----

Key Features
------------

- **Framework-agnostic** — optional dependency groups bring first-class support for
  PyTorch, TensorFlow, scikit-learn, PyMC, and arbitrary black-box processes.
- **Broad task coverage** — supports high-frequency AI model development workflows across
  **classification, regression, clustering, anomaly detection, time-series forecasting,
  text classification, Bayesian inference, and black-box optimisation**.
- **Declarative search spaces** — annotate tunable parameters directly in the
  ``train()`` signature with ``@BaseTrainer.search(...)``. No Optuna boilerplate required.
- **Single- and multi-objective HPO** — native Pareto-front support for multi-metric
  optimisation; override ``select_best_trial()`` for custom selection logic.
- **Environment-variable-driven config** — every ``TrainConfig`` field can be
  overridden by an environment variable, making container and CI/CD integration seamless.
- **Dry-run validation** — verify the entire pipeline locally without production data
  or a remote training server using a single ``trainer.dry_run()`` call.
- **DVC + MinIO integration** — ``het_ai.dvc`` resolves the latest data version tag from
  GitHub, pulls actual data from MinIO via DVC, and injects version metadata into the
  ``DataBundle`` for downstream traceability.
- **MLflow auto-logging** — set ``TrainConfig(mlflow=MLflowConfig(...))`` and the
  framework automatically logs params, metrics, dataset lineage, and registers the model
  after every run. No MLflow code required in your ``Trainer``.
- **Flexible model export** — ONNX / TFLite / Keras / joblib / JSON / NetCDF and
  custom formats via ``export_model()``.

----

Installation
------------

Core package (HPO only, minimal dependencies):

.. code-block:: bash

   pip install het-ai

Full platform (HPO + DVC data versioning + MLflow tracking):

.. code-block:: bash

   pip install "het-ai[platform]"

Install optional ML framework extras as needed:

.. code-block:: bash

   # PyTorch + ONNX export
   pip install "het-ai[torch]"

   # TensorFlow / Keras / TFLite
   pip install "het-ai[tensorflow]"

   # scikit-learn
   pip install "het-ai[sklearn]"

   # Bayesian modelling with PyMC
   pip install "het-ai[bayes]"

   # Full platform + PyTorch
   pip install "het-ai[platform,torch]"

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

Data-Version-Driven Platform
------------------------------

When DVC and MLflow extras are installed, het-ai drives the entire pipeline
from a data version tag to a registered model — fully automatically.

.. code-block:: python

   from het_ai.studio import BaseTrainer, TrainConfig
   from het_ai.studio.bundle import DataBundle
   from het_ai.dvc import DVCLoader, DVCConfig
   from het_ai.mlflow import MLflowConfig
   from het_ai.studio.types import TunableFloat
   from pathlib import Path
   import pandas as pd


   class MyTrainer(BaseTrainer):

       @BaseTrainer.search(lr=TunableFloat(1e-4, 1e-2, log=True))
       def train(self, data: DataBundle, lr: float):
           ...

       def load_data(self, dvc_data_root: str) -> DataBundle:
           # Pull versioned data: GitHub tag → MinIO → local
           loader = DVCLoader(self.config.dvc or DVCConfig())
           tag, sha = loader.pull(Path(dvc_data_root))

           df = pd.read_csv(f"{dvc_data_root}/data.csv")
           bundle = DataBundle(
               splits={"train": {"X": df[features].values,
                                 "y": df["label"].values}},
               feature_list=features,
               target_list=["label"],
           )
           # Inject version metadata — MLflow will tag the Run automatically
           return loader.enrich_bundle(bundle, tag, sha)

       def mock_data(self) -> DataBundle: ...
       def export_model(self, artifact, export_dir) -> str: ...


   config = TrainConfig(
       n_trials=100,
       dvc=DVCConfig(),        # reads DVC_GITHUB_* and MINIO_* env vars
       mlflow=MLflowConfig(    # reads MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT
           experiment_name="my-project",
       ),
   )

   # DVC pull → Optuna HPO → model export →
   # MLflow Run logged with dvc_version tag → model registered
   result = MyTrainer(config).run()

After ``run()`` completes, every MLflow Run is tagged with the DVC version
(e.g. ``dvc_version = release-20250525``) that produced it, and the model is
registered as ``my-project_prod`` in the MLflow Model Registry.

The minimum required environment variables for the full platform:

.. code-block:: bash

   # DVC / GitHub
   export DVC_GITHUB_REPO=org/data-repo
   export DVC_GITHUB_TOKEN=ghp_xxxxxxxxxxxx
   export MINIO_ENDPOINT=minio.internal:9000
   export MINIO_ACCESS_KEY=minioadmin
   export MINIO_SECRET_KEY=minioadmin
   export MINIO_BUCKET=dvc-store

   # MLflow
   export MLFLOW_TRACKING_URI=http://mlflow.internal:5000
   export MLFLOW_EXPERIMENT=my-project

----

Multi-Objective Optimisation
------------------------------

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
   * - ``dvc``
     - —
     - ``DVCConfig`` instance; consumed by ``load_data()`` to pull versioned data
     - ``None``
   * - ``mlflow``
     - —
     - ``MLflowConfig`` instance; triggers automatic MLflow logging after ``run()``
     - ``None``

----

Supported Training Scenarios
------------------------------

The bundled test suite covers **twelve** representative scenarios, all runnable via
``dry_run()`` without any external dependencies:

.. list-table::
   :header-rows: 1
   :widths: 6 32 28 34

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
     - Gradient-boosted trees classification
     - scikit-learn
     - joblib serialisation
   * - 5
     - Unsupervised clustering (dual-objective)
     - NumPy / scikit-learn
     - Pareto-front HPO
   * - 6
     - Bayesian parameter inference
     - PyMC
     - NetCDF export
   * - 7
     - Hand-written logistic regression
     - Pure NumPy
     - Zero ML-framework dependency
   * - 8
     - Black-box external process
     - Subprocess
     - Wraps any CLI training command
   * - 9
     - Tabular regression
     - scikit-learn
     - Multi-objective (R2 + RMSE) selection
   * - 10
     - Time-series forecasting
     - PyTorch (LSTM)
     - Sequence modelling + ONNX export
   * - 11
     - Unsupervised anomaly detection
     - scikit-learn (Isolation Forest)
     - Multi-metric anomaly quality
   * - 12
     - Text classification
     - PyTorch (TextCNN)
     - NLP workflow + ONNX export

----

Development & Testing
---------------------

.. code-block:: bash

   # Clone and install in editable mode with all development dependencies
   git clone https://github.com/HeT-FTI/het-ai.git
   cd het-ai
   pip install -e ".[dev,examples]"

   # Run the full test suite
   pytest tests/ -v

   # Run scenario smoke tests only
   pytest tests/cases -v

The ``tests/cases`` directory currently contains **12** representative end-to-end
dry-run scenarios for core workflow validation.

----

License
-------

- **Source code** — `Apache License 2.0 <LICENSE>`_
- **Documentation** — `Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) <https://creativecommons.org/licenses/by-nc/4.0/>`_

For commercial use of the documentation, please contact HeT directly.

Copyright © 2026 Shenzhen HeT Intelligent Control Co., Ltd.
