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

    # # 新增：可选集成，None = 不启用
    # dvc: Optional["DVCConfig"] = None       # from het_ai.dvc
    # mlflow: Optional["MLflowConfig"] = None # from het_ai.mlflow


    # ── 集成配置 ─────────────────────────────────────────────────────
    #
    # mlflow: 框架自动触发。
    #   设置后，WorkflowRunner 在训练结束时自动完成全部 MLflow 上报。
    #   用户无需在 Trainer 里写任何 mlflow 代码。
    mlflow: Optional["MLflowConfig"] = None

    # dvc: 用户主动消费的配置载体，框架不自动触发 DVC 拉取。
    #
    # 设计原因：load_data() 是用户控制数据来源的唯一入口，
    # 数据目录结构、多仓库合并、版本检测逻辑因项目而异，
    # 框架无法替用户做这些决定。
    #
    # 推荐用法：
    #   def load_data(self, dvc_data_root: str) -> DataBundle:
    #       loader = DVCLoader(self.config.dvc or DVCConfig())
    #       tag, sha = loader.pull(Path(dvc_data_root))
    #       bundle = DataBundle(...)
    #       return loader.enrich_bundle(bundle, tag, sha)
    #
    # 不使用 DVC 的场景（本地文件、其他数据源）无需设置此字段。
    dvc: Optional["DVCConfig"] = None

    def __post_init__(self):
        if self.study_name is None:
            self.study_name = f"study_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        if self.trial_root_dir is None:
            self.trial_root_dir = f"optuna_trials_{self.study_name}"
