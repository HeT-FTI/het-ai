from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from het_ai.studio.base import BaseTrainer


@dataclass
class TrainResult:
    """
    标准化训练结果，支持 to_tuple() 向后兼容 MLOps 平台协议。
    """

    tag_dict: Dict[str, Any]
    params_dict: Dict[str, Any]
    metric_dict: Dict[str, Any]
    feature_list: List[str]
    target_list: List[str]
    dataset_splits_dict: Dict[str, Any]
    model_path: Optional[str]
    artifact_file_paths: List[str] = field(default_factory=list)

    trainer: Optional["BaseTrainer"] = field(default=None, repr=False)

    def to_tuple(self) -> Tuple:
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
