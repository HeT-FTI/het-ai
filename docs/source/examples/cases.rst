Examples
========

The ``tests/cases/`` directory contains 12 end-to-end examples covering diverse
frameworks, task types, and optimisation modes. Each case follows the same pattern:
subclass :class:`~het_ai.studio.base.BaseTrainer`, implement the required hooks,
and run with :class:`~het_ai.studio.config.TrainConfig`.

All examples are tested via ``pytest tests/test_simulate_all.py`` which runs
``dry_run()`` on each case to validate the pipeline end-to-end.

.. list-table:: Example Matrix
    :header-rows: 1

    * - #
      - Framework
      - Task
      - Export Format
      - HPO Type
    * - 1
      - PyTorch
      - Tabular Classification
      - ONNX
      - Single-objective
    * - 2
      - PyTorch
      - Multi-target Classification
      - ONNX
      - Multi-objective (2)
    * - 3
      - TensorFlow / Keras
      - Image Segmentation (YOLO)
      - TFLite / ONNX
      - Single-objective
    * - 4
      - scikit-learn
      - GBT Classification
      - joblib (.pkl)
      - Single-objective
    * - 5
      - scikit-learn
      - KMeans Clustering
      - joblib
      - Multi-objective (2)
    * - 6
      - PyMC
      - Bayesian Parameter Estimation
      - NetCDF
      - Single-objective
    * - 7
      - Pure NumPy
      - Logistic Regression (Softmax + SGD)
      - JSON
      - Single-objective
    * - 8
      - Subprocess (black-box)
      - External Process HPO
      - External binary
      - Single-objective
    * - 9
      - scikit-learn
      - Random Forest Regression
      - joblib
      - Multi-objective (2)
    * - 10
      - PyTorch
      - LSTM Time-Series Forecasting
      - ONNX
      - Single-objective
    * - 11
      - scikit-learn
      - Isolation Forest Anomaly Detection
      - joblib
      - Multi-objective (2)
    * - 12
      - PyTorch
      - TextCNN Text Classification
      - ONNX
      - Single-objective

----

Tabular Tasks
-------------

**Case 1: PyTorch Tabular Classification**
    A simple feedforward network trained on tabular data with ONNX export.
    Demonstrates the standard single-objective workflow with ``@search``
    annotations on learning rate, hidden size, and batch size.

    .. code-block:: python

        class PytorchTabularTrainer(BaseTrainer):
            @BaseTrainer.search(
                lr=BaseTrainer.TunableFloat(1e-4, 1e-2, log=True),
                hidden=BaseTrainer.TunableInt(32, 256, step=32),
            )
            def train(self, data, lr, hidden):
                model = SimpleNet(data.X_train.shape[1], n_classes)
                # ... training loop with self.report() for pruning ...
                return accuracy, model

    Source: ``tests/cases/case1_torch_tabular.py``

**Case 2: PyTorch Multi-target Classification**
    A single model predicting multiple labels simultaneously. Demonstrates
    multi-objective HPO with two competing metrics.

    .. code-block:: python

        class PytorchMultitargetTrainer(BaseTrainer):
            objectives = {"acc_target1": "maximize", "acc_target2": "maximize"}

            def train(self, data, lr, hidden_dim):
                # ... multi-head model ...
                return BaseTrainer.Result(
                    acc_target1=acc1,
                    acc_target2=acc2,
                )

    Source: ``tests/cases/case2_torch_multitarget.py``

**Case 4: scikit-learn GBT Classification**
    Wraps ``GradientBoostingClassifier`` inside a ``Pipeline`` with
    ``StandardScaler``. Demonstrates sklearn integration with joblib export.

    .. code-block:: python

        class SklearnGBTrainer(BaseTrainer):
            @BaseTrainer.search(
                n_estimators=BaseTrainer.TunableInt(50, 300, step=50),
                learning_rate=BaseTrainer.TunableFloat(0.01, 0.3, log=True),
            )
            def train(self, data, n_estimators, learning_rate):
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("gbt", GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                    )),
                ])
                pipe.fit(data.X_train, data.y_train)
                f1 = f1_score(data.y_val, pipe.predict(data.X_val), average="weighted")
                return f1, pipe

    Source: ``tests/cases/case4_sklearn_gbt.py``

**Case 9: scikit-learn Random Forest Regression**
    Multi-objective regression optimisation with R² and negative MSE.

    .. code-block:: python

        class SklearnRegressionTrainer(BaseTrainer):
            objectives = {"r2": "maximize", "neg_mse": "maximize"}

            def train(self, data, n_estimators, max_depth):
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                )
                model.fit(data.X_train, data.y_train)
                r2 = model.score(data.X_val, data.y_val)
                mse = -((model.predict(data.X_val) - data.y_val) ** 2).mean()
                return BaseTrainer.Result(r2=r2, neg_mse=mse), model

    Source: ``tests/cases/case9_sklearn_regression.py``

----

Image & Computer Vision
-----------------------

**Case 3: TensorFlow/Keras Image Segmentation**
    Uses a YOLO-style segmentation model with TFLite/ONNX export. Demonstrates
    TensorFlow integration, image data handling, and multi-format export.

    Source: ``tests/cases/case3_tflite_image.py``

----

Time-Series
-----------

**Case 10: PyTorch LSTM Forecasting**
    An LSTM network for time-series forecasting. Demonstrates sequence data
    handling, ONNX export, and time-series specific HPO (look-back window,
    hidden size).

    Source: ``tests/cases/case10_torch_timeseries.py``

----

Text & NLP
----------

**Case 12: PyTorch TextCNN Classification**
    A TextCNN (convolutional neural network for text) for document classification.
    Demonstrates text preprocessing, embedding layers, and ONNX export.

    Source: ``tests/cases/case12_torch_text.py``

----

Clustering & Anomaly Detection
------------------------------

**Case 5: scikit-learn KMeans Clustering**
    Unsupervised clustering with silhouette score and inertia as multi-objective
    metrics. Demonstrates HPO for clustering tasks (``n_clusters``, ``init``).

    .. code-block:: python

        class ClusteringTrainer(BaseTrainer):
            objectives = {"silhouette": "maximize", "inertia": "minimize"}

    Source: ``tests/cases/case5_clustering.py``

**Case 11: scikit-learn Isolation Forest Anomaly Detection**
    Anomaly detection with multi-objective optimisation. Demonstrates HPO for
    unsupervised anomaly detection tasks.

    .. code-block:: python

        class AnomalyTrainer(BaseTrainer):
            objectives = {"precision": "maximize", "recall": "maximize"}

    Source: ``tests/cases/case11_sklearn_anomaly.py``

----

Bayesian Inference
------------------

**Case 6: PyMC Bayesian Parameter Estimation**
    Bayesian model fitting with PyMC. Demonstrates integration with
    probabilistic programming frameworks and NetCDF export.

    .. code-block:: python

        class BayesianTrainer(BaseTrainer):
            @BaseTrainer.search(
                prior_mu=BaseTrainer.TunableFloat(-5, 5, step=0.5),
                prior_sigma=BaseTrainer.TunableFloat(0.1, 5, log=True),
            )
            def train(self, data, prior_mu, prior_sigma):
                import pymc as pm
                with pm.Model() as model:
                    mu = pm.Normal("mu", prior_mu, prior_sigma)
                    # ... sampling ...
                    trace = pm.sample(...)
                # ... compute WAIC or LOO ...
                return score, trace

    Source: ``tests/cases/case6_pymc_bayesian.py``

----

Black-Box / External Processes
------------------------------

**Case 8: Black-Box External Process**
    Hyperparameter optimisation for an external binary or script. Demonstrates
    how het-ai can tune parameters for any process — not just Python ML models.

    The external trainer is launched as a subprocess with hyperparameters
    passed as command-line arguments. Results are read from stdout or a
    results file.

    .. code-block:: python

        class BlackBoxTrainer(BaseTrainer):
            @BaseTrainer.search(
                param_a=BaseTrainer.TunableInt(1, 10),
                param_b=BaseTrainer.TunableFloat(0.1, 1.0, log=True),
            )
            def train(self, data, param_a, param_b):
                import subprocess
                result = subprocess.run(
                    ["python", "external_trainer.py",
                     "--param_a", str(param_a),
                     "--param_b", str(param_b)],
                    capture_output=True, text=True,
                )
                score = float(result.stdout.strip())
                return score, None  # No model artifact for external processes

    Source: ``tests/cases/case8_blackbox.py`` (trainer) and
    ``tests/external_trainer.py`` (standalone external process)

----

Pure NumPy (No Framework)
-------------------------

**Case 7: NumPy Logistic Regression**
    A softmax logistic regression implemented from scratch using only NumPy.
    Demonstrates that het-ai works with zero ML framework dependencies —
    pure Python + NumPy + manual SGD.

    .. code-block:: python

        class NumpyLogisticTrainer(BaseTrainer):
            @BaseTrainer.search(
                lr=BaseTrainer.TunableFloat(1e-3, 1.0, log=True),
                epochs=BaseTrainer.TunableInt(50, 500, step=50),
            )
            def train(self, data, lr, epochs):
                X, y = data.X_train, data.y_train
                W = np.random.randn(X.shape[1], n_classes) * 0.01
                # ... manual softmax + cross-entropy + SGD ...
                for epoch in range(epochs):
                    logits = X @ W
                    probs = softmax(logits)
                    loss = cross_entropy(probs, y)
                    grad = X.T @ (probs - one_hot(y)) / len(y)
                    W -= lr * grad
                    self.report(epoch, accuracy(probs, y))
                return accuracy, {"weights": W}

    Source: ``tests/cases/case7_numpy_logistic.py``

----

Running the Examples
--------------------

All examples are designed to work with synthetic (mock) data so you can run
them without any external dependencies beyond the core package:

.. code-block:: bash

    pip install "het-ai[examples]"
    cd tests
    python -m pytest test_simulate_all.py -v

To run a single case:

.. code-block:: python

    from tests.cases.case1_torch_tabular import PytorchTabularTrainer
    from het_ai.studio import TrainConfig

    trainer = PytorchTabularTrainer(TrainConfig(n_trials=5))
    trainer.dry_run()

Each case file is self-contained and can serve as a template for your own
training tasks.

