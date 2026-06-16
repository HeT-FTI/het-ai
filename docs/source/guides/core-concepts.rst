Core Concepts
==============

This page explains the architecture and key abstractions of het-ai.

Architecture Overview
---------------------

het-ai splits the ML training lifecycle into three layers:

.. kroki::
    :type: plantuml
    :format: svg

    @startuml
    skinparam componentStyle rectangle
    skinparam backgroundColor #FEFEFE
    skinparam defaultFontSize 13

    title het-ai Architecture Overview

    package "User Code" as User #E8F0FE {
        [my_trainer = MyTrainer(config)] as uc1
        [my_trainer.dry_run()  // local validation] as uc2
        [my_trainer.run()      // full HPO pipeline] as uc3
    }

    package "het_ai.studio (Core)" as Studio #FFF3E0 {
        [BaseTrainer\nuser-facing abstraction] as s1
        [TrainConfig\nenv-variable-driven config] as s2
        [DataBundle\nframework-agnostic data container] as s3
        [GhostInjector\n@search decorator (no Optuna import)] as s4
        [WorkflowRunner\ninternal orchestration engine] as s5
        [TrainResult\nstandardised output protocol] as s6
    }

    package "het_ai.dvc" as DVC #E8F5E9 {
        [DVCLoader\ndata version pull (DVC + MinIO)] as d1
    }

    package "het_ai.mlflow" as MLflow #FCE4EC {
        [MLflowRunLogger\nauto-log params/metrics/model/lineage] as m1
    }

    User --> Studio : depends on
    Studio --> DVC : optional integration
    Studio --> MLflow : optional integration
    @enduml

Concept 1: TrainConfig
----------------------

:class:`~het_ai.studio.config.TrainConfig` is the central configuration dataclass.
Every field is **environment-variable-driven** — set a variable to override the default.

.. list-table:: Key Config Fields
    :header-rows: 1

    * - Field
      - Default
      - Env Variable
      - Description
    * - ``n_trials``
      - 100
      - ``N_TRIALS``
      - Number of Optuna trials
    * - ``direction``
      - ``"maximize"``
      - ``OPTUNA_DIRECTION``
      - Optimisation direction
    * - ``n_jobs``
      - 1
      - ``OPTUNA_N_JOBS``
      - Parallel trials (threads)
    * - ``timeout``
      - *None*
      - ``OPTUNA_TIMEOUT``
      - Time limit in seconds
    * - ``pruner``
      - ``"median"``
      - ``OPTUNA_PRUNER``
      - ``"median"``, ``"threshold"``, or ``"none"``
    * - ``storage``
      - *None*
      - ``OPTUNA_STORAGE``
      - Optuna DB URL for distributed HPO
    * - ``dvc_data_root``
      - ``"dvc_data"``
      - ``DVC_DATA_ROOT``
      - Local path for DVC-pulled data
    * - ``export_formats``
      - ``["onnx"]``
      - —
      - Model export format(s)

Nested configs for optional integrations:

- ``mlflow: Optional[MLflowConfig]`` — enables MLflow auto-logging
- ``dvc: Optional[DVCConfig]`` — enables automatic data pull via DVC

Concept 2: BaseTrainer
----------------------

:class:`~het_ai.studio.base.BaseTrainer` is the main user-facing class.
Subclass it and implement the required hooks. The class carries:

- **Class attributes** — ``search`` (the ``@search`` decorator), ``TunableInt``,
  ``TunableFloat``, ``TunableCategorical``, ``Result`` (all accessible without
  importing from elsewhere).
- **``objectives``** — an optional ``Dict[str, Direction]`` for multi-objective HPO.

Required methods to implement:

.. list-table::
    :header-rows: 1

    * - Method
      - Purpose
    * - ``load_data(dvc_data_root) -> DataBundle``
      - Load real training data
    * - ``train(data, **hparams) -> float | tuple | Result``
      - Core training logic (must be decorated with ``@search``)
    * - ``mock_data() -> DataBundle``
      - Return synthetic data for dry-run validation

Optional hooks (override for custom behaviour):

.. list-table::
    :header-rows: 1

    * - Hook
      - When Called
    * - ``export_model(artifact, export_dir) -> str``
      - After best trial is selected
    * - ``select_best_trial(pareto_front) -> Any``
      - Custom Pareto-front selection logic
    * - ``on_study_end(study, best_trial) -> dict``
      - After all trials complete (add extra tags)
    * - ``before_mlflow_log(result) -> TrainResult``
      - Before MLflow logging (modify result)
    * - ``on_model_registered(result) -> None``
      - After MLflow model registration (deploy callback)
    * - ``predict(model_path, inputs) -> Any``
      - Inference for registered models

Concept 3: DataBundle
---------------------

:class:`~het_ai.studio.bundle.DataBundle` is a lightweight, framework-agnostic data
container. It makes no assumptions about data format:

.. code-block:: python

    from het_ai.studio.bundle import DataBundle

    bundle = DataBundle(
        splits={
            "train": {"X": X_train, "y": y_train},
            "val":   {"X": X_val,   "y": y_val},
            "test":  {"X": X_test,  "y": y_test},
        },
        feature_list=["f1", "f2", "f3"],
        target_list=["label"],
        meta={"description": "Iris dataset"},
    )

    # Convenience accessors
    X_train, y_train = bundle.X_train, bundle.y_train
    X_val, y_val = bundle.X_val, bundle.y_val

- ``splits`` — arbitrary dict of named data splits
- ``feature_list`` / ``target_list`` — column names for MLflow logging
- ``meta`` — free-form metadata injected by DVC integration (data version, commit SHA)
- ``lineage_datasets`` — optional list of MLflow dataset objects for lineage tracking

Concept 4: Tunable Types & @search
------------------------------------

The heart of het-ai's declarative approach. Three "ghost types" carry metadata:

.. list-table::
    :header-rows: 1

    * - Type
      - Subclasses
      - Metadata
    * - ``TunableInt(low, high, step, log)``
      - ``int``
      - ``low``, ``high``, ``step``, ``log``
    * - ``TunableFloat(low, high, step, log)``
      - ``float``
      - ``low``, ``high``, ``step``, ``log``
    * - ``TunableCategorical(choices)``
      - ``str``
      - ``choices`` (list of values)

They look like ordinary values but carry ``_meta`` dictionaries attached. The
``@BaseTrainer.search(...)`` decorator (implemented by :class:`~het_ai.studio.injector.GhostInjector`)
simply stores the metadata. At runtime, :class:`~het_ai.studio.runner.WorkflowRunner`
reads the metadata and calls the appropriate ``trial.suggest_*()``:

.. code-block:: python

    @BaseTrainer.search(
        lr=TunableFloat(1e-4, 1e-2, log=True),        # float
        hidden=TunableInt(32, 256, step=32),            # int
        dropout=TunableCategorical([0.0, 0.2, 0.5]),   # categorical
    )
    def train(self, data, lr, hidden, dropout):
        # lr is a plain float, hidden is a plain int, dropout is a plain str
        # No Optuna imports needed here.
        ...

Key benefits:

- **Zero Optuna imports** in user training code
- **IDE-friendly** — ``train()`` parameters are real Python types
- **Validation at class-definition time** — ``__init_subclass__`` checks that
  ``@search`` parameters match the ``train()`` signature

How ``run()`` Works (Execution Pipeline)
-----------------------------------------

When you call ``trainer.run()``, the framework executes the following pipeline:

1. **Validate** — check that ``train()`` is decorated with ``@search``
2. **DVC pull** (if ``config.dvc`` is set) — resolve latest data version via
   :class:`~het_ai.dvc.loader.DVCLoader`, pull data, enrich ``DataBundle.meta``
   with version metadata
3. **Load data** — call ``trainer.load_data(dvc_data_root)``
4. **Run Optuna study** — :class:`~het_ai.studio.runner.WorkflowRunner` creates
   a study, samples hyperparameters from ``@search`` metadata, calls ``train()``
   for each trial, applies pruning
5. **Pick best trial** — from the study, optionally using ``select_best_trial()``
   for custom Pareto-front selection
6. **Export model** — call ``trainer.export_model(artifact, export_dir)``
7. **Build TrainResult** — standardised result with params, metrics, model path,
   dataset splits, and tags
8. **MLflow logging** (if ``config.mlflow`` is set) — auto-log params, metrics,
   dataset lineage, and register the model via
   :class:`~het_ai.mlflow.logger.MLflowRunLogger`
9. **Call post-hooks** — ``on_model_registered()`` for deployment callbacks

The entire pipeline is exposed through a single method — no need to orchestrate
DVC, Optuna, or MLflow separately.

Multi-Objective Optimisation
-----------------------------

For multi-objective HPO, declare ``objectives`` on your Trainer class and return
a :class:`~het_ai.studio.types.Result` from ``train()``:

.. code-block:: python

    class MultiTrainer(BaseTrainer):
        objectives = {"accuracy": "maximize", "latency": "minimize"}

        @BaseTrainer.search(...)
        def train(self, data, **hp):
            ...
            return BaseTrainer.Result(accuracy=0.95, latency=3.2)

Optuna handles the Pareto-front. Override ``select_best_trial()`` to implement
custom selection logic (e.g., pick the trial with the best accuracy among the
front, or apply a weighted utility function).

Thread Safety
~~~~~~~~~~~~~

When ``n_jobs > 1``, Optuna runs trials concurrently. het-ai uses
``threading.local()`` for per-trial state (``_trial_local``) to ensure thread
safety without requiring user awareness.

Next Steps
----------

- :doc:`dvc-integration` — data versioning with DVC + MinIO
- :doc:`mlflow-integration` — automatic MLflow logging
- :doc:`advanced` — multi-objective, pruning, custom hooks, parallel trials

