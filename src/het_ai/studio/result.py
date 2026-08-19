from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from het_ai.studio.base import BaseTrainer


@dataclass
class TrainResult:
    """
    A standardized training result, supporting to_tuple() for backward
    compatibility with the MLOps platform protocol.
    """

    tag_dict: dict[str, Any]
    params_dict: dict[str, Any]
    metric_dict: dict[str, Any]
    feature_list: list[str]
    target_list: list[str]
    dataset_splits_dict: dict[str, Any]
    model_path: str | None
    artifact_file_paths: list[str] = field(default_factory=list)

    trainer: BaseTrainer | None = field(default=None, repr=False)

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise KeyError(key)
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise KeyError(key)
            setattr(self, key, value)

    def to_tuple(self) -> tuple:
        return (
            self.tag_dict,
            self.params_dict,
            self.metric_dict,
            self.feature_list,
            self.target_list,
            self.dataset_splits_dict,
            self.model_path,
            self.artifact_file_paths,
        )
