from dataclasses import dataclass, field
import os

@dataclass
class MLflowConfig:
    """MLflow 上报配置。"""

    tracking_uri: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    experiment_name: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_EXPERIMENT", "default")
    )

    # 模型注册
    register_model: bool = True
    registered_model_name: str = ""          # 空时用 experiment_name

    # 数据集血缘自动上报
    log_dataset_lineage: bool = True

    # 支持的导出格式："onnx" | "pt" | "tflite"
    # 从 TrainConfig.export_formats 继承，此处可覆盖
    export_formats: list = field(default_factory=lambda: ["onnx"])
