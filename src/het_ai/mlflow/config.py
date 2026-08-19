from dataclasses import dataclass, field
import os


@dataclass
class MLflowConfig:
    """MLflow reporting configuration."""

    tracking_uri: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    experiment_name: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_EXPERIMENT", "default")
    )

    # 模型注册
    register_model: bool = True
    registered_model_name: str = ""   # 空时用 experiment_name
    model_name_suffix: str = "_prod"  # 注册名后缀，可覆盖

    # 数据集血缘自动上报
    log_dataset_lineage: bool = True

    # 注意：export_formats 由 TrainConfig 管理，此处不重复声明
