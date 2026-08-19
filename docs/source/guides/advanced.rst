Advanced Features
==================

This guide covers het-ai's more advanced capabilities: multi-objective
optimisation, pruning, custom hooks, parallel execution, and custom export.

Multi-Objective Optimisation
-----------------------------

het-ai supports multi-objective HPO via Optuna's Pareto-front capabilities.

**Step 1:** Declare ``objectives`` on your Trainer class:

.. code-block:: python

    class MyTrainer(BaseTrainer):
        objectives = {"accuracy": "maximize", "inference_latency": "minimize"}

**Step 2:** Return a :class:`~het_ai.studio.types.Result` from ``train()``:

.. code-block:: python

    @BaseTrainer.search(lr=TunableFloat(1e-4, 1e-2, log=True))
    def train(self, data, lr):
        ...
        return BaseTrainer.Result(
            accuracy=0.95,
            inference_latency=2.3,  # milliseconds
        )

Optuna automatically computes the Pareto front across all objectives.

Custom Best-Trial Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Override ``select_best_trial()`` to define your own selection logic from the
Pareto front:

.. code-block:: python

    class MyTrainer(BaseTrainer):
        objectives = {"accuracy": "maximize", "latency": "minimize"}

        def select_best_trial(self, pareto_front):
            """Prefer accuracy: pick the trial with highest accuracy among the front."""
            return max(pareto_front, key=lambda t: t.values[0])

The default implementation picks the trial with the maximum first objective value.

Intermediate Reporting & Pruning
--------------------------------

Use ``self.report(step, value)`` inside ``train()`` to report intermediate
values. This enables Optuna's pruners to terminate unpromising trials early:

.. code-block:: python

    @BaseTrainer.search(lr=TunableFloat(1e-4, 1e-2, log=True))
    def train(self, data, lr):
        model = build_model()
        for epoch in range(100):
            loss = train_one_epoch(model, data)
            val_acc = evaluate(model, data)

            # Report intermediate value every epoch
            # Returns False if the trial was pruned
            if not self.report(epoch, val_acc):
                break  # Early stop — trial pruned

        return val_acc, model

**Available pruners** (set via ``TrainConfig(pruner=...)``):

.. list-table::
    :header-rows: 1

    * - Pruner
      - Description
    * - ``"median"``
      - MedianPruner: prunes trials below the median of previous trials
    * - ``"threshold"``
      - ThresholdPruner: prunes trials below a fixed threshold
    * - ``"none"``
      - No pruning (default for single-objective without ``report()``)

Custom Model Export
-------------------

Override ``export_model()`` to persist your trained model in any format:

.. code-block:: python

    class MyTrainer(BaseTrainer):
        def export_model(self, artifact, export_dir):
            """Export model to a custom directory."""
            model_path = f"{export_dir}/my_model_format.bin"
            # artifact is whatever train() returns as the second element
            save_my_model(artifact, model_path)
            return model_path  # Returned path is logged by MLflow

The default implementation simply returns ``export_dir`` (no-op).

Post-Training Hooks
-------------------

het-ai provides three hooks for extending the training lifecycle:

``on_study_end(study, best_trial) -> dict``
    Called after all trials complete. Return a dict of extra tags to add to
    the ``TrainResult``:

    .. code-block:: python

        def on_study_end(self, study, best_trial):
            return {"total_trials": len(study.trials),
                    "best_trial_number": best_trial.number}

``before_mlflow_log(result) -> TrainResult``
    Called before MLflow logging. Modify the result or add custom artifacts:

    .. code-block:: python

        def before_mlflow_log(self, result):
            # Add a confusion matrix plot as an artifact
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix
            # ...
            plt.savefig(f"{result.model_path}/confusion_matrix.png")
            result.artifact_file_paths.append(f"{result.model_path}/confusion_matrix.png")
            return result

``on_model_registered(result) -> None``
    Called after MLflow model registration. Use for deployment triggers:

    .. code-block:: python

        def on_model_registered(self, result):
            # Trigger CI/CD pipeline, send webhook, push to serving endpoint
            import requests
            requests.post("https://deploy.example.com/webhook", json={
                "model_path": result.model_path,
                "metrics": result.metric_dict,
            })

Parallel Trials
---------------

Set ``n_jobs > 1`` in ``TrainConfig`` to run trials concurrently:

.. code-block:: python

    config = TrainConfig(n_trials=100, n_jobs=4)  # 4 trials in parallel

**Thread safety:** het-ai uses ``threading.local()`` (exposed as
``BaseTrainer._trial_local``) for per-trial state. Each trial has its own
isolated ``_trial_local`` namespace — no action needed from your side.

.. warning::

    When using ``n_jobs > 1``, ensure your ``train()`` method does not rely on
    shared mutable state. Use ``_trial_local`` for any trial-scoped state:

    .. code-block:: python

        def train(self, data, lr):
            self._trial_local.model = build_model(lr)
            # ... use self._trial_local.model safely across trials ...

Dry-Run Mode
------------

``trainer.dry_run()`` runs the complete pipeline with mock data and default
hyperparameter values — no remote services, no real data, no Optuna database:

.. code-block:: python

    trainer = MyTrainer(TrainConfig(n_trials=100))
    result = trainer.dry_run()  # ~1 trial, mock data, no MLflow/DVC
    print("Pipeline validates successfully")

Use dry-run to:

- Verify that your ``load_data()``, ``train()``, and ``export_model()`` work
  end-to-end
- Catch signature mismatches between ``@search`` and ``train()`` early
- Test in CI before running full HPO

Distributed HPO
---------------

Connect to a shared Optuna storage to run distributed HPO across multiple workers:

.. code-block:: python

    config = TrainConfig(
        n_trials=1000,
        storage="postgresql://user:pass@host:5432/optuna",
        study_name="my-distributed-study",
    )

All workers sharing the same ``storage`` and ``study_name`` contribute trials
to the same study. This enables horizontal scaling across a cluster.

Custom TagResolver for DVC
---------------------------

When the built-in GitHub resolver doesn't fit your workflow, implement the
:class:`~het_ai.dvc.loader.TagResolver` protocol:

.. code-block:: python

    from het_ai.dvc.loader import TagResolver

    class SemverLatestResolver:
        """Pick the highest semver tag."""
        def __init__(self, repo_url):
            self._repo = repo_url

        def resolve(self) -> tuple[str, str]:
            # Fetch tags, parse semver, return highest
            ...
            return tag_name, commit_sha

    # Pass to DVCConfig (manual instantiation of DVCLoader)
    from het_ai.dvc import DVCLoader, DVCConfig
    loader = DVCLoader(DVCConfig(...), tag_resolver=SemverLatestResolver("..."))
    tag, sha = loader.pull(output_path)

The ``TagResolver`` protocol is the **only decoupling point** for version
resolution — swap resolvers without modifying any other module.

Next Steps
----------

- :doc:`../api/studio` — full Studio API reference
- :doc:`../examples/cases` — see these features in action across 12 examples

