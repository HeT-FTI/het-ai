Type System
============

The ``het_ai.studio.types`` module defines the core type system for
declarative hyperparameter optimisation.

TunableInt
----------

.. autoclass:: het_ai.studio.types.TunableInt
    :members:
    :undoc-members:
    :show-inheritance:

    A ghost integer that carries HPO metadata through its ``_meta`` attribute.

    **Constructor:** ``TunableInt(low, high, step=1, log=False)``

    .. code-block:: python

        from het_ai.studio.types import TunableInt

        # In your @search decorator:
        hidden = TunableInt(32, 256, step=32)       # linear search
        layers = TunableInt(1, 10, log=True)         # log-scale search

    At runtime, the framework calls ``trial.suggest_int()`` with the metadata.


TunableFloat
------------

.. autoclass:: het_ai.studio.types.TunableFloat
    :members:
    :undoc-members:
    :show-inheritance:

    A ghost float that carries HPO metadata.

    **Constructor:** ``TunableFloat(low, high, step=None, log=False)``

    .. code-block:: python

        from het_ai.studio.types import TunableFloat

        lr = TunableFloat(1e-4, 1e-2, log=True)      # log-scale float
        dropout = TunableFloat(0.0, 0.5, step=0.1)    # linear float

    At runtime, the framework calls ``trial.suggest_float()`` with the metadata.


TunableCategorical
------------------

.. autoclass:: het_ai.studio.types.TunableCategorical
    :members:
    :undoc-members:
    :show-inheritance:

    A ghost string that carries categorical choice metadata.

    **Constructor:** ``TunableCategorical(choices)``

    .. code-block:: python

        from het_ai.studio.types import TunableCategorical

        optimizer = TunableCategorical(["adam", "sgd", "adamw"])
        activation = TunableCategorical(["relu", "gelu", "silu"])

    At runtime, the framework calls ``trial.suggest_categorical()`` with the choices.


Result
------

.. autoclass:: het_ai.studio.types.Result
    :members:
    :undoc-members:
    :show-inheritance:

    A multi-objective return wrapper used in ``train()``.

    .. code-block:: python

        from het_ai.studio.types import Result

        # In train():
        return Result(accuracy=0.95, latency=2.3, memory_mb=150)

        # Methods:
        result.get_values()   # (0.95, 2.3, 150)
        result.get_names()    # ("accuracy", "latency", "memory_mb")


Direction
---------

.. autoclass:: het_ai.studio.types.Direction
    :members:
    :undoc-members:

    Type alias: ``Literal["maximize", "minimize"]``. Used in the ``objectives``
    class attribute of ``BaseTrainer``.

