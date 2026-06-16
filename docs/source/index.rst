.. het-ai documentation master file

het-ai
======

**A framework-agnostic MLOps training DSL powered by Optuna.**

het-ai provides a structured abstraction for the full training lifecycle —
*data loading → hyperparameter search → model training → export* —
through a single, consistent interface, regardless of your ML framework.

.. grid:: 2
    :gutter: 3

    .. grid-item-card::
        :link: guides/getting-started
        :link-type: doc

        🚀 Getting Started
        ^^^^^^^^^^^^^^^^^^
        Install het-ai and run your first hyperparameter optimisation
        in under 5 minutes.

    .. grid-item-card::
        :link: guides/core-concepts
        :link-type: doc

        🧠 Core Concepts
        ^^^^^^^^^^^^^^^^
        Understand the architecture: TrainConfig, BaseTrainer, DataBundle,
        tunable types, and the execution pipeline.

    .. grid-item-card::
        :link: guides/dvc-integration
        :link-type: doc

        📦 DVC Integration
        ^^^^^^^^^^^^^^^^^^
        Data versioning with DVC + MinIO. Resolve tags, pull data,
        and trace every experiment back to its data version.

    .. grid-item-card::
        :link: guides/mlflow-integration
        :link-type: doc

        📊 MLflow Integration
        ^^^^^^^^^^^^^^^^^^^^^
        Automatic MLflow logging: params, metrics, dataset lineage,
        and model registration — zero boilerplate.

    .. grid-item-card::
        :link: guides/advanced
        :link-type: doc

        🔧 Advanced Features
        ^^^^^^^^^^^^^^^^^^^^
        Multi-objective optimisation, Pareto-front selection, pruning,
        custom export, post-training hooks, and parallel trials.

    .. grid-item-card::
        :link: examples/cases
        :link-type: doc

        📚 Examples
        ^^^^^^^^^^
        12 end-to-end examples covering PyTorch, TensorFlow, scikit-learn,
        PyMC, NumPy, and black-box processes across diverse task types.

.. raw:: html

    <hr style="margin: 2rem 0;">

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Guides

    guides/getting-started
    guides/core-concepts
    guides/dvc-integration
    guides/mlflow-integration
    guides/advanced

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: API Reference

    api/studio
    api/dvc
    api/mlflow
    api/types

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Examples

    examples/cases

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Contributing

    contributing

Quick Install
-------------

.. code-block:: bash

    # Core package (HPO only, minimal dependencies)
    pip install het-ai

    # Full platform (HPO + DVC + MLflow)
    pip install "het-ai[platform]"

    # With framework extras
    pip install "het-ai[torch,sklearn]"

Key Features
------------

.. list-table::
    :header-rows: 0

    * - ✅ **Framework-agnostic**
      - One API for PyTorch, TensorFlow, scikit-learn, PyMC, NumPy, and black-box.
    * - 🎯 **Declarative HPO**
      - Annotate search spaces with ``@search`` — no Optuna boilerplate.
    * - 📈 **Multi-objective**
      - Native Pareto-front support with customisable selection logic.
    * - 🔌 **DVC + MinIO**
      - Automatic data version resolution and pull on every run.
    * - 📊 **MLflow auto-logging**
      - Params, metrics, lineage, and model registration — zero extra code.
    * - 🧪 **Dry-run mode**
      - ``trainer.dry_run()`` validates the full pipeline locally with mock data.

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

