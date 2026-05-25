import os
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    """
    平台级训练配置。
    优先级：环境变量 > 显式传参 > 默认值。
    """

    n_trials: int = field(
        default_factory=lambda: int(os.environ.get("N_TRIALS", 100))
    )
    direction: str = field(
        default_factory=lambda: os.environ.get("OPTUNA_DIRECTION", "maximize")
    )
    n_jobs: int = field(
        default_factory=lambda: int(os.environ.get("OPTUNA_N_JOBS", 1))
    )
    timeout: Optional[float] = field(
        default_factory=lambda: (
            float(os.environ["OPTUNA_TIMEOUT"])
            if "OPTUNA_TIMEOUT" in os.environ else None
        )
    )
    pruner: str = field(
        default_factory=lambda: os.environ.get("OPTUNA_PRUNER", "median")
    )
    storage: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPTUNA_STORAGE", None)
    )
    study_name: Optional[str] = None

    dvc_data_root: str = field(
        default_factory=lambda: os.environ.get("DVC_DATA_ROOT", "dvc_data")
    )
    test_size: float = 0.2
    random_state: int = 42

    export_formats: List[str] = field(default_factory=lambda: ["onnx"])
    onnx_opset_version: int = 18
    trial_root_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("TRIAL_ROOT_DIR", None)
    )

    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO")
    )

    def __post_init__(self):
        if self.study_name is None:
            self.study_name = f"study_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        if self.trial_root_dir is None:
            self.trial_root_dir = f"optuna_trials_{self.study_name}"
