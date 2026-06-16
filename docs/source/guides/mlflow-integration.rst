MLflow Integration
===================

The ``het_ai.mlflow`` module provides **automatic MLflow logging** — params,
metrics, dataset lineage, and model registration — with zero boilerplate in
your training code.

Configuration
-------------

:class:`~het_ai.mlflow.config.MLflowConfig` controls the MLflow integration:

.. list-table:: MLflowConfig Fields
    :header-rows: 1

    * - Field
      - Env Variable
      - Default
      - Description
    * - ``tracking_uri``
      - ``MLFLOW_TRACKING_URI``
      - ``http://localhost:5000``
      - MLflow server URI
    * - ``experiment_name``
      - ``MLFLOW_EXPERIMENT_NAME``
      - ``"mlops_demo"``
      - Experiment name
    * - ``register_model``
      - ``MLFLOW_REGISTER_MODEL``
      - ``False``
      - Register model in MLflow Model Registry
    * - ``registered_model_name``
      - ``MLFLOW_REGISTERED_MODEL_NAME``
      - ``(same as experiment_name)``
      - Name for Model Registry entry
    * - ``model_name_suffix``
      - ``MLFLOW_MODEL_NAME_SUFFIX``
      - ``""``
      - Suffix appended to registered model name
    * - ``log_dataset_lineage``
      - ``MLFLOW_LOG_DATASET_LINEAGE``
      - ``True``
      - Log dataset lineage metadata

Enable MLflow by passing an ``MLflowConfig`` to ``TrainConfig``:

.. code-block:: python

    from het_ai.studio import BaseTrainer, TrainConfig
    from het_ai.mlflow import MLflowConfig

    config = TrainConfig(
        n_trials=50,
        direction="maximize",
        mlflow=MLflowConfig(
            tracking_uri="http://mlflow-server:5000",
            experiment_name="het-ai-experiments",
            register_model=True,
        ),
    )
    trainer = MyTrainer(config)
    result = trainer.run()
    # MLflow run is automatically created and populated

What Gets Logged
----------------

When ``config.mlflow`` is set, :class:`~het_ai.mlflow.logger.MLflowRunLogger`
automatically logs the following after each ``trainer.run()``:

.. list-table::
    :header-rows: 1

    * - Category
      - Content
    * - **Params**
      - Best trial's hyperparameters, auto-flattened (e.g. ``best_trial_params.lr``)
    * - **Metrics**
      - Scalar metrics (single values) and per-step metric histories
    * - **Dataset Lineage**
      - ``mlflow.data.from_pandas()`` per data split, tagged with DVC version
    * - **Tags**
      - ``result.tag_dict`` merged with DVC metadata (``dvc_version_tag``, ``dvc_commit_sha``)
    * - **Model**
      - Dispatched by file extension via the **Flavor Registry**
    * - **Artifacts**
      - All files and directories in ``result.artifact_file_paths``

No MLflow code is required in your ``Trainer`` subclass.

Flavor Registry
---------------

Instead of hardcoded ``if/elif`` chains, ``MLflowRunLogger`` uses a **class-level
flavor registry** — a dict mapping file extensions to log functions:

.. list-table:: Default Flavors
    :header-rows: 1

    * - Extension
      - MLflow Flavor
    * - ``.onnx``
      - ``mlflow.onnx.log_model()``
    * - ``.pt``, ``.pth``
      - ``mlflow.pytorch.log_model()``
    * - ``.pkl``, ``.joblib``
      - ``mlflow.sklearn.log_model()``
    * - ``.tflite``
      - Custom ``TFLiteWrapper`` (``mlflow.pyfunc.log_model()``)

**Register a custom model flavor:**

.. code-block:: python

    from het_ai.mlflow import MLflowRunLogger

    def log_keras(model_path, signature, input_example, registered_name, register):
        import keras
        import mlflow
        mlflow.keras.log_model(
            keras.load_model(model_path),
            artifact_path="model",
            registered_model_name=registered_name if register else None,
        )

    # Register for `.keras` files before any run
    MLflowRunLogger.register_flavor(".keras", log_keras)

The fallback for unrecognised extensions is to log the file as a generic MLflow
artifact.

Pre-MLflow Hook
~~~~~~~~~~~~~~~

Override ``before_mlflow_log(result)`` on your Trainer to modify the result or
add extra artifacts before MLflow logging occurs:

.. code-block:: python

    class MyTrainer(BaseTrainer):
        def before_mlflow_log(self, result):
            result.tag_dict["custom_field"] = "extra metadata"
            return result

Post-Registration Hook
~~~~~~~~~~~~~~~~~~~~~~

Override ``on_model_registered(result)`` for deployment callbacks (e.g., trigger
a CI pipeline, send a notification, push to a model serving endpoint):

.. code-block:: python

    class MyTrainer(BaseTrainer):
        def on_model_registered(self, result):
            print(f"Model registered: {result.model_path}")
            # Trigger deployment pipeline...

Dataset Lineage
---------------

When ``log_dataset_lineage=True``, the logger constructs ``mlflow.data.from_pandas()``
objects for each data split and logs them as MLflow dataset entities. Each dataset
is tagged with:

- ``split`` — ``"train"``, ``"val"``, ``"test"``
- ``dvc_version`` — the resolved data version tag
- ``dvc_commit_sha`` — the corresponding commit SHA

This appears in the MLflow UI under the **Datasets** tab of each run.

Full Example
------------

.. code-block:: python

    import os
    from het_ai.studio import BaseTrainer, TrainConfig, DataBundle
    from het_ai.mlflow import MLflowConfig, MLflowRunLogger

    # Register a custom flavor for NetCDF files (case6: PyMC Bayesian)
    def log_netcdf(model_path, signature, input_example, registered_name, register):
        import mlflow
        mlflow.log_artifact(model_path, artifact_path="model_netcdf")

    MLflowRunLogger.register_flavor(".nc", log_netcdf)

    class MyTrainer(BaseTrainer):
        @BaseTrainer.search(lr=BaseTrainer.TunableFloat(1e-4, 1e-2, log=True))
        def train(self, data, lr):
            # ... training logic ...
            return score, model

        def load_data(self, dvc_data_root):
            # ... data loading ...
            return DataBundle(...)

        def mock_data(self):
            return self.load_data(None)

    config = TrainConfig(
        n_trials=100,
        mlflow=MLflowConfig(
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            experiment_name="bayesian-inference",
            register_model=True,
            registered_model_name="pymc-model",
        ),
    )
    trainer = MyTrainer(config)
    trainer.dry_run()       # Validate first
    result = trainer.run()  # Full run → params, metrics, model logged

Next Steps
----------

- :doc:`dvc-integration` — combine with DVC for full traceability
- :doc:`../api/mlflow` — full MLflow API reference

