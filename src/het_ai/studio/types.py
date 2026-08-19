from typing import Any, ClassVar, Literal

Direction = Literal["maximize", "minimize"]


class TunableBase:
    _meta: ClassVar[dict[str, Any]] = {}


class TunableInt(int, TunableBase):
    def __new__(cls, low: int, high: int, step: int = 1, log: bool = False):
        instance = super().__new__(cls, low)
        instance._meta = {
            "type": "int",
            "low": low,
            "high": high,
            "step": step,
            "log": log,
        }
        return instance


class TunableFloat(float, TunableBase):
    def __new__(cls, low: float, high: float, step: float | None = None, log: bool = False):
        instance = super().__new__(cls, low)
        instance._meta = {
            "type": "float",
            "low": low,
            "high": high,
            "step": step,
            "log": log,
        }
        return instance


class TunableCategorical(str, TunableBase):
    def __new__(cls, choices: list[Any]):
        instance = super().__new__(cls, str(choices[0]))
        instance._meta = {
            "type": "categorical",
            "choices": choices,
        }
        return instance


class Result:
    """
    A wrapper for multi-objective optimization return values.
    The optimization directions are declared in the BaseTrainer.objectives class
    attribute; this class only carries the values.
    """

    def __init__(self, **kwargs: float):
        self.data: dict[str, float] = kwargs

    def get_values(self) -> list[float]:
        return list(self.data.values())

    def get_names(self) -> list[str]:
        return list(self.data.keys())
