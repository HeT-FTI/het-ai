from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataBundle:
    """
    A generic data container that does not presume any task type, framework,
    or data structure.
    """

    splits: dict[str, Any] = field(default_factory=dict)
    feature_list: list[str] = field(default_factory=list)
    target_list: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # 可选：用户在 load_data() 中预先填入已构造好的 mlflow.data.Dataset 列表，
    # 供 MLflowRunLogger 直接使用，以支持非表格数据结构（多目标、聚类、图像等）。
    # 若为 None，MLflowRunLogger 将尝试从 splits["X"] / splits["y"] 自动构建。
    lineage_datasets: list[Any] | None = field(default=None)

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
