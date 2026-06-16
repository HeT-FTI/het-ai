DVC Integration
================

The ``het_ai.dvc`` module provides **automatic data versioning** via DVC + MinIO.
When enabled, every HPO run automatically:

1. Resolves the appropriate data version tag
2. Pulls the actual data from MinIO remote storage
3. Injects version metadata into the ``DataBundle``
4. Tags the corresponding MLflow experiment with the data version

This closes the **data → experiment → model** traceability loop.

Configuration
-------------

:class:`~het_ai.dvc.config.DVCConfig` controls the DVC integration.
All fields are environment-variable-driven:

.. list-table:: DVCConfig Fields
    :header-rows: 1

    * - Field
      - Env Variable
      - Description
    * - ``github_repo``
      - ``DVC_GITHUB_REPO``
      - GitHub repo URL or ``owner/repo``
    * - ``github_token``
      - ``DVC_GITHUB_TOKEN``
      - GitHub API token (needs ``repo`` scope)
    * - ``github_api_base``
      - ``DVC_GITHUB_API_BASE``
      - API base URL (default: ``https://api.github.com``)
    * - ``minio_endpoint``
      - ``DVC_MINIO_ENDPOINT``
      - MinIO server address (e.g. ``minio.example.com:9000``)
    * - ``minio_access_key``
      - ``DVC_MINIO_ACCESS_KEY``
      - MinIO access key
    * - ``minio_secret_key``
      - ``DVC_MINIO_SECRET_KEY``
      - MinIO secret key
    * - ``minio_bucket``
      - ``DVC_MINIO_BUCKET``
      - Bucket name (typically ``dvc``)
    * - ``minio_virtual_folder``
      - ``DVC_MINIO_VIRTUAL_FOLDER``
      - Optional prefix within the bucket
    * - ``tag_strategy``
      - ``DVC_TAG_STRATEGY``
      - ``"release"`` (match ``release*``) or ``"latest"`` (most recent)

Enable DVC by passing a ``DVCConfig`` to ``TrainConfig``:

.. code-block:: python

    from het_ai.studio import BaseTrainer, TrainConfig
    from het_ai.dvc import DVCConfig

    config = TrainConfig(
        n_trials=50,
        dvc=DVCConfig(
            github_repo="my-org/ml-data-repo",
            tag_strategy="release",
        ),
    )
    trainer = MyTrainer(config)
    result = trainer.run()
    # DataBundle.meta now contains dvc_version and dvc_commit_sha

Tag Resolution Strategy
-----------------------

The :class:`~het_ai.dvc.loader.TagResolver` protocol defines the version
resolution interface. The framework ships with two built-in resolvers:

.. list-table::
    :header-rows: 1

    * - Resolver
      - Behaviour
    * - :class:`~het_ai.dvc.loader.GitHubTagResolver`
      - Fetches tags from GitHub API. ``"release"`` strategy matches tags
        starting with ``release``; ``"latest"`` picks the most recent by commit date.
    * - :class:`~het_ai.dvc.loader.FixedTagResolver`
      - Returns a fixed tag/commit SHA — useful for reproducibility, CI, or
        offline scenarios.

You can implement custom resolvers by conforming to the ``TagResolver`` protocol:

.. code-block:: python

    class GitLabTagResolver:
        def resolve(self) -> tuple[str, str]:
            # Custom logic to fetch tags from GitLab API
            ...
            return tag_name, commit_sha

The ``TagResolver`` protocol is the **only decoupling point** — swap resolvers
without touching any other module.

Data Pull Workflow
------------------

When ``trainer.run()`` is called with ``config.dvc`` set:

1. ``DVCLoader`` is instantiated with the ``DVCConfig``.
2. ``TagResolver.resolve()`` returns ``(tag_name, commit_sha)``.
3. ``DVCLoader`` downloads the ``.dvc`` pointer files from GitHub at the resolved tag.
4. DVC remote is configured to point to the MinIO bucket.
5. ``dvc pull`` + ``dvc checkout`` fetches and checks out the actual data files.
6. ``DVCLoader.enrich_bundle()`` injects version metadata into ``DataBundle.meta``:

.. code-block:: python

    bundle.meta["dvc_version"]    = tag_name      # e.g. "release-v1.2"
    bundle.meta["dvc_commit_sha"] = commit_sha    # full 40-char SHA

MLflow Traceability
~~~~~~~~~~~~~~~~~~~

When both DVC and MLflow are enabled, the data version is automatically:

- Added as an MLflow **tag** (``data_version``)
- Attached to each **dataset lineage** entry in the MLflow run
- Included in the ``TrainResult.tag_dict``

This gives you full traceability: any deployed model can be traced back to the
exact data version that produced it.

Full Example
------------

.. code-block:: python

    import os
    from het_ai.studio import BaseTrainer, TrainConfig, DataBundle
    from het_ai.dvc import DVCConfig
    from het_ai.mlflow import MLflowConfig
    from sklearn.datasets import load_diabetes

    class MyTrainer(BaseTrainer):
        objectives = {"r2": "maximize"}

        @BaseTrainer.search(
            n_estimators=BaseTrainer.TunableInt(50, 300, step=50),
        )
        def train(self, data, n_estimators):
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=n_estimators)
            model.fit(data.X_train, data.y_train)
            score = model.score(data.X_val, data.y_val)
            return score, model

        def load_data(self, dvc_data_root):
            # In production, this loads from dvc_data_root
            X, y = load_diabetes(return_X_y=True)
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            return DataBundle(splits={
                "train": {"X": X_train, "y": y_train},
                "val": {"X": X_val, "y": y_val},
            })

        def mock_data(self):
            return self.load_data(None)

    config = TrainConfig(
        n_trials=20,
        dvc=DVCConfig(),
        mlflow=MLflowConfig(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        ),
    )
    trainer = MyTrainer(config)
    result = trainer.run()
    # result.tag_dict includes "data_version_tag" and "data_commit_sha"
    # MLflow run is tagged with the same data version

Next Steps
----------

- :doc:`mlflow-integration` — automatic MLflow logging details
- :doc:`../api/dvc` — full DVC API reference

