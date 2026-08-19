import inspect
import logging
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from het_ai.studio.bundle import DataBundle
from het_ai.studio.config import TrainConfig
from het_ai.studio.injector import GhostInjector
from het_ai.studio.result import TrainResult
from het_ai.studio.types import (
    Direction,
    Result,
    TunableCategorical,
    TunableFloat,
    TunableInt,
)


class BaseTrainer(ABC):
    """
    Platform-level base class for trainers.
    """

    search = staticmethod(GhostInjector.search)
    TunableInt = TunableInt
    TunableFloat = TunableFloat
    TunableCategorical = TunableCategorical
    Result = Result

    objectives: dict[str, Direction] | None = None
    _trial_local = threading.local()

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()
        self._in_dry_run = False
        self._logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=self.config.log_level,
            format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        train_fn = cls.__dict__.get("train")
        if train_fn is None or not getattr(train_fn, "_is_tunable", False):
            return

        search_space = train_fn._search_space
        sig_params = set(inspect.signature(train_fn).parameters) - {"self", "data", "trial"}

        extra = set(search_space) - sig_params
        if extra:
            raise TypeError(
                f"\n[{cls.__name__}] @search 声明了参数 {extra}，"
                f"但 train() 签名中找不到。\n"
                f"  签名中的参数 : {sig_params}\n"
                f"  请检查是否有拼写错误。"
            )

        if cls.objectives is not None and not isinstance(cls.objectives, dict):
            raise TypeError(
                f"[{cls.__name__}] objectives 必须是 dict，"
                f"例如: {{'accuracy': 'maximize', 'size': 'minimize'}}"
            )

    @abstractmethod
    def load_data(self, dvc_data_root: str) -> DataBundle:
        ...

    @abstractmethod
    def train(self, data: DataBundle, **hparams) -> "float | tuple | Result":
        ...

    def mock_data(self) -> DataBundle:
        raise NotImplementedError

    def export_model(self, artifact: Any, export_dir: str) -> str:
        return export_dir

    def select_best_trial(self, pareto_front: list) -> Any:
        return max(pareto_front, key=lambda t: t.values[0])

    def on_study_end(self, study: Any, best_trial: Any) -> dict:
        return {}

    def before_mlflow_log(self, result: "TrainResult") -> "TrainResult":
        """
        Callback hook invoked before the training result is uploaded to MLflow.
        Subclasses may modify the result here, e.g. append to artifact_file_paths
        or adjust metric_dict.

        Args:
            result: The normalized result object about to be uploaded to MLflow.

        Returns:
            The modified result; may also return the same object after in-place
            modification.
        """
        return result

    def on_model_registered(self, result: "TrainResult") -> None:
        """
        Callback hook invoked after the model is registered in the MLflow Model
        Registry. Subclasses may override it to trigger deployment (K8s, Seldon,
        TorchServe, etc.) or send notifications.

        Args:
            result: The normalized result object after training completes.
        """

    def predict(self, model_path: str, inputs: Any) -> Any:
        raise NotImplementedError("如需推理验证，请覆写 predict()。")

    def _validate_integration_configs(self) -> None:
        """
        Validate the required fields of the integration configs in TrainConfig.
        Called before dry_run() and run() to surface configuration errors early.
        """
        dvc_cfg = getattr(self.config, "dvc", None)
        if dvc_cfg is not None:
            missing = []
            if not getattr(dvc_cfg, "seaweedfs_endpoint", ""):
                missing.append("dvc.seaweedfs_endpoint")
            if not getattr(dvc_cfg, "seaweedfs_access_key", ""):
                missing.append("dvc.seaweedfs_access_key")
            if not getattr(dvc_cfg, "seaweedfs_secret_key", ""):
                missing.append("dvc.seaweedfs_secret_key")
            if not getattr(dvc_cfg, "github_repo", ""):
                missing.append("dvc.github_repo")
            if not getattr(dvc_cfg, "github_token", ""):
                missing.append("dvc.github_token")
            if missing:
                raise ValueError(
                    f"[{type(self).__name__}] DVCConfig 缺少必填字段: "
                    f"{', '.join(missing)}。"
                    f"请通过显式传参或环境变量完成配置后再运行。"
                )

        mlflow_cfg = getattr(self.config, "mlflow", None)
        if mlflow_cfg is not None:
            missing = []
            if not getattr(mlflow_cfg, "tracking_uri", ""):
                missing.append("mlflow.tracking_uri")
            if not getattr(mlflow_cfg, "experiment_name", ""):
                missing.append("mlflow.experiment_name")
            if missing:
                raise ValueError(
                    f"[{type(self).__name__}] MLflowConfig 缺少必填字段: "
                    f"{', '.join(missing)}。"
                    f"请通过显式传参或环境变量完成配置后再运行。"
                )

    def report(self, step: int, value: float) -> bool:
        trial = getattr(self._trial_local, "current", None)
        if trial is None:
            return False
        trial.report(value, step=step)
        return trial.should_prune()

    def dry_run(self, dvc_data_root: str | None = None) -> dict:
        # ── 配置结构预检：提前发现外部集成配置错误 ──────────────────────────
        self._validate_integration_configs()

        try:
            self._logger.info("[dry_run] ① 尝试 mock_data()...")
            data = self.mock_data()
            self._logger.info("[dry_run]    ✓ 使用合成数据（mock_data）")
        except NotImplementedError:
            root = dvc_data_root or self.config.dvc_data_root
            self._logger.info(
                f"[dry_run] ① mock_data() 未实现，回退到 load_data({root})"
            )
            data = self.load_data(root)

        search_space = type(self).train._search_space
        default_params = {}
        for name, ghost in search_space.items():
            meta = ghost._meta
            default_params[name] = (
                meta["choices"][0] if meta["type"] == "categorical" else meta["low"]
            )

        self._logger.info(f"[dry_run] ② 默认参数: {default_params}")

        self._in_dry_run = True
        try:
            t0 = time.time()
            raw = self.train(data=data, **default_params)
            elapsed = time.time() - t0
        finally:
            self._in_dry_run = False

        score, artifact, _metrics = self._unpack_train_result(raw)
        self._logger.info(
            f"[dry_run] ③ train() 完成  score={score}  elapsed={elapsed:.2f}s"
        )

        export_path = None
        if artifact is not None:
            export_dir = tempfile.mkdtemp()
            export_path = self.export_model(artifact, export_dir)
            self._logger.info(f"[dry_run] ④ export_model() → {export_path}")
        else:
            self._logger.info("[dry_run] ④ artifact=None，跳过 export_model()")

        self._logger.info("[dry_run] ✅ 全流程验证通过")
        return {
            "score": score,
            "artifact": artifact,
            "elapsed": elapsed,
            "export_path": export_path,
        }

    def run(self, dvc_data_root: str | None = None) -> TrainResult:
        from het_ai.studio.runner import WorkflowRunner

        self._validate_integration_configs()
        root = dvc_data_root or self.config.dvc_data_root
        return WorkflowRunner(self).execute(root)

    @staticmethod
    def _unpack_train_result(raw) -> tuple:
        """Unpack the return value of train(), supporting three-element returns.

        Returns: (score, artifact, metrics_dict)
        """
        if isinstance(raw, tuple) and len(raw) == 3:
            return raw[0], raw[1], raw[2]
        if isinstance(raw, tuple) and len(raw) == 2:
            return raw[0], raw[1], None
        if isinstance(raw, (int, float, Result)):
            return raw, None, None
        raise TypeError(
            f"train() 返回类型不合法: {type(raw)}\n"
            "合法返回值:\n"
            "  float                              — 单目标\n"
            "  (float, artifact)                  — 单目标 + 导出产物\n"
            "  (float, artifact, dict)            — 单目标 + 导出产物 + 自定义指标\n"
            "  Result(...)                        — 多目标\n"
            "  (Result(...), artifact)            — 多目标 + 导出产物\n"
            "  (Result(...), artifact, dict)      — 多目标 + 导出产物 + 自定义指标"
        )
