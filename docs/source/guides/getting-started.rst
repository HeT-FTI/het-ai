Getting Started
===============

Installation
------------

het-ai is available on PyPI. Choose the installation scope that matches your needs:

.. code-block:: bash

    # Core package — HPO only, minimal dependencies
    pip install het-ai

    # Full platform — HPO + DVC data versioning + MLflow tracking
    pip install "het-ai[platform]"

    # With ML framework extras
    pip install "het-ai[torch]"          # PyTorch + ONNX
    pip install "het-ai[tensorflow]"     # TensorFlow / Keras / TFLite
    pip install "het-ai[sklearn]"        # scikit-learn
    pip install "het-ai[bayes]"          # PyMC for Bayesian modelling

    # Combine extras
    pip install "het-ai[platform,torch]"

    # All dependencies needed for bundled examples
    pip install "het-ai[examples]"

Prerequisites
~~~~~~~~~~~~~

- **Python** >= 3.10
- **Optuna** >= 3.5.0 (installed automatically)
- Framework-specific dependencies are optional; install only what you need.

5-Minute Quick Start
--------------------

Here is the minimal example — a PyTorch tabular classifier with hyperparameter
optimisation in fewer than 60 lines:

.. code-block:: python

    import numpy as np
    import torch
    import torch.nn as nn
    from het_ai.studio import BaseTrainer, TrainConfig, DataBundle

    class SimpleNN(nn.Module):
        def __init__(self, in_features, out_features, hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_features),
            )

        def forward(self, x):
            return self.net(x)

    class MyTrainer(BaseTrainer):
        objectives = {"accuracy": "maximize"}

        @BaseTrainer.search(
            lr=BaseTrainer.TunableFloat(1e-4, 1e-2, log=True),
            hidden=BaseTrainer.TunableInt(32, 256, step=32),
        )
        def train(self, data, lr, hidden):
            X, y = data.X_train, data.y_train
            X_val, y_val = data.X_val, data.y_val

            n_classes = len(np.unique(y))
            model = SimpleNN(4, n_classes, hidden)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(50):
                model.train()
                opt.zero_grad()
                loss = loss_fn(model(torch.tensor(X, dtype=torch.float32)),
                               torch.tensor(y, dtype=torch.long))
                loss.backward()
                opt.step()

                # Intermediate reporting enables pruning
                model.eval()
                with torch.no_grad():
                    preds = model(torch.tensor(X_val, dtype=torch.float32)).argmax(1)
                    acc = (preds.numpy() == y_val).mean()
                self.report(epoch, acc)

            return acc, model

        def load_data(self, dvc_data_root):
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split

            X, y = load_iris(return_X_y=True)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            return DataBundle(splits={
                "train": {"X": X_train, "y": y_train},
                "val": {"X": X_val, "y": y_val},
            })

        def mock_data(self):
            return self.load_data(None)

    # Configure and run
    config = TrainConfig(n_trials=20, direction="maximize")
    trainer = MyTrainer(config)

    # Step 1: Dry-run to validate the pipeline
    trainer.dry_run()

    # Step 2: Full HPO run
    result = trainer.run()
    print("Best score:", result.metric_dict)
    print("Model at:", result.model_path)

What's Happening?
~~~~~~~~~~~~~~~~~

1. **Subclass** :class:`~het_ai.studio.base.BaseTrainer` and declare ``objectives``.
2. **Decorate** ``train()`` with ``@BaseTrainer.search(...)``, annotating each
   tunable parameter with a ghost type (:class:`~het_ai.studio.types.TunableFloat`,
   :class:`~het_ai.studio.types.TunableInt`).
3. **Implement** ``load_data()`` → :class:`~het_ai.studio.bundle.DataBundle` and
   ``mock_data()`` for dry-run validation.
4. **Configure** via :class:`~het_ai.studio.config.TrainConfig` — number of trials,
   direction, pruner, parallelism.
5. **Dry-run** first (``trainer.dry_run()``) to catch errors early.
6. **Run** (``trainer.run()``) for the full Optuna-powered HPO pipeline.

Next Steps
----------

- :doc:`core-concepts` — deep dive into the architecture
- :doc:`../examples/cases` — 12 end-to-end examples across frameworks and tasks
- :doc:`advanced` — multi-objective, pruning, custom hooks, parallel trials

