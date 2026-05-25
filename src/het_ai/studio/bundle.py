from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DataBundle:
    """
    通用数据容器，不预设任务类型、框架或数据结构。
    """

    splits: Dict[str, Any] = field(default_factory=dict)
    feature_list: List[str] = field(default_factory=list)
    target_list: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def X_train(self):
        return self.splits.get("train", {}).get("X")

    @property
    def y_train(self):
        return self.splits.get("train", {}).get("y")

    @property
    def X_val(self):
        return self.splits.get("val", {}).get("X")

    @property
    def y_val(self):
        return self.splits.get("val", {}).get("y")

    @property
    def X_test(self):
        return self.splits.get("test", {}).get("X")

    @property
    def y_test(self):
        return self.splits.get("test", {}).get("y")
