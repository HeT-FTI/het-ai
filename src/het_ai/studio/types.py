from typing import Any, Dict, List, Literal

Direction = Literal["maximize", "minimize"]


class TunableBase:
    _meta: Dict[str, Any] = {}


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
    def __new__(cls, low: float, high: float, step: float = None, log: bool = False):
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
    def __new__(cls, choices: List[Any]):
        instance = super().__new__(cls, str(choices[0]))
        instance._meta = {
            "type": "categorical",
            "choices": choices,
        }
        return instance


class Result:
    """
    多目标优化返回值包装器。
    优化方向在 BaseTrainer.objectives 类属性中声明，此处只携带数值。
    """

    def __init__(self, **kwargs: float):
        self.data: Dict[str, float] = kwargs

    def get_values(self) -> List[float]:
        return list(self.data.values())

    def get_names(self) -> List[str]:
        return list(self.data.keys())
