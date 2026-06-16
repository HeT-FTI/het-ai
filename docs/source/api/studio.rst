Studio API Reference
====================

The ``het_ai.studio`` package is the core framework. It provides the
user-facing abstractions for the full training lifecycle.

BaseTrainer
-----------

.. autoclass:: het_ai.studio.base.BaseTrainer
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: search, TunableInt, TunableFloat, TunableCategorical, Result, _trial_local

TrainConfig
-----------

.. autoclass:: het_ai.studio.config.TrainConfig
    :members:
    :undoc-members:
    :show-inheritance:

DataBundle
----------

.. autoclass:: het_ai.studio.bundle.DataBundle
    :members:
    :undoc-members:
    :show-inheritance:

TrainResult
-----------

.. autoclass:: het_ai.studio.result.TrainResult
    :members:
    :undoc-members:
    :show-inheritance:

GhostInjector
-------------

.. autoclass:: het_ai.studio.injector.GhostInjector
    :members:
    :undoc-members:
    :show-inheritance:

WorkflowRunner
--------------

.. autoclass:: het_ai.studio.runner.WorkflowRunner
    :members:
    :undoc-members:
    :show-inheritance:

