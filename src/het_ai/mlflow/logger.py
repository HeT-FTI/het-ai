from __future__ import annotations

import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import mlflow
import mlflow.data
import numpy as np
import pandas as pd

from het_ai.mlflow.config import MLflowConfig
from het_ai.studio.bundle import DataBundle
from het_ai.studio.result import TrainResult


# ──────────────────────────────────────────────────────────────────────────────
# 模型 Flavor 注册表（替代 if/elif 硬编码链）
# ──────────────────────────────────────────────────────────────────────────────

# 类型：(model_path, signature, input_example, registered_name, register_model) -> None
FlavorLogFn = Callable[[str, object, object, str, bool], None]

def _log_onnx(model_path, signature, input_example, registered_name, register):
    import onnx
    mlflow.onnx.log_model(
        onnx_model=onnx.load(model_path),
        name="model_onnx",
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_name if register else None,
    )

def _log_pytorch(model_path, signature, input_example, registered_name, register):
    import torch
    mlflow.pytorch.log_model(
        pytorch_model=torch.load(model_path, map_location="cpu", weights_only=False),
        name="model_pt",
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_name if register else None,
    )

def _log_sklearn(model_path, signature, input_example, registered_name, register):
    import joblib
    mlflow.sklearn.log_model(
        sk_model=joblib.load(model_path),
        name="model_sklearn",
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_name if register else None,
    )

def _log_tflite(model_path, signature, input_example, registered_name, register):
    class TFLiteWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite
            self.interpreter = tflite.Interpreter(
                model_path=context.artifacts["tflite_file"]
            )
            self.interpreter.allocate_tensors()
            self._in  = self.interpreter.get_input_details()[0]
            self._out = self.interpreter.get_output_details()[0]

        def predict(self, context, model_input):
            if not isinstance(model_input, pd.DataFrame):
                model_input = pd.DataFrame(model_input)
            target_dtype = np.dtype(self._in["dtype"])
            target_shape = list(self._in["shape"])
            if target_shape[0] == -1:
                target_shape[0] = len(model_input)
            if np.issubdtype(target_dtype, np.number):
                arr = model_input.to_numpy().astype(target_dtype).reshape(target_shape)
            else:
                arr = model_input.to_numpy(dtype=object).reshape(target_shape)
            self.interpreter.set_tensor(self._in["index"], arr)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self._out["index"])
            if self._out["dtype"] in (np.object_, np.bytes_):
                try:
                    output = np.char.decode(output.astype(np.bytes_), "utf-8")
                except Exception:
                    pass
            return output

    mlflow.pyfunc.log_model(
        python_model=TFLiteWrapper(),
        artifacts={"tflite_file": model_path},
        name="model_tflite",
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_name if register else None,
    )


# 默认注册表：文件后缀 → flavor 函数
# 用户可通过 MLflowRunLogger.register_flavor() 扩展
_DEFAULT_FLAVOR_REGISTRY: dict[str, FlavorLogFn] = {
    "onnx":   _log_onnx,
    "pt":     _log_pytorch,
    "pth":    _log_pytorch,
    "pkl":    _log_sklearn,
    "joblib": _log_sklearn,
    "tflite": _log_tflite,
}


# ──────────────────────────────────────────────────────────────────────────────
# MLflowRunLogger
# ──────────────────────────────────────────────────────────────────────────────

class MLflowRunLogger:
    """
    在 WorkflowRunner 执行结束后，将 TrainResult 完整上报到 MLflow。

    上报内容：
      ① mlflow.log_params       ← result.params_dict（自动展平一层嵌套）
      ② mlflow.log_metric(s)    ← result.metric_dict（list → per-step）
      ③ mlflow.log_input        ← DataBundle 各 split 的数据集血缘（含 DVC 版本标签）
      ④ mlflow.set_tags         ← result.tag_dict + bundle.meta 中所有 dvc_* 字段
      ⑤ mlflow.log_model        ← 通过 flavor 注册表按后缀自动分发
      ⑥ mlflow.log_artifact(s)  ← result.artifact_file_paths

    扩展自定义 flavor：
        MLflowRunLogger.register_flavor("h5", my_keras_log_fn)
    """

    # 类级注册表，所有实例共享，支持运行时扩展
    _flavor_registry: dict[str, FlavorLogFn] = dict(_DEFAULT_FLAVOR_REGISTRY)

    @classmethod
    def register_flavor(cls, extension: str, log_fn: FlavorLogFn) -> None:
        """
        注册自定义模型 flavor。

        Args:
            extension: 文件后缀（不含点），如 "h5", "cbm"
            log_fn:    签名为 (model_path, signature, input_example,
                                registered_name, register_model) -> None
        """
        cls._flavor_registry[extension.lower().lstrip(".")] = log_fn

    def __init__(self, config: MLflowConfig):
        self.config = config
        self._owned_run_id: Optional[str] = None
        mlflow.set_tracking_uri(config.tracking_uri)

    # ── 公共入口 ──────────────────────────────────────────────────────────────

    def log(
        self,
        bundle: DataBundle,
        result: TrainResult,
        run_name: Optional[str] = None,
    ) -> None:
        """一次性完成所有 MLflow 上报，被 WorkflowRunner.execute() 自动调用。"""
        with self._active_or_new_run(run_name=run_name):
            self._log_params(result)
            self._log_metrics(result)
            if self.config.log_dataset_lineage:
                self._log_dataset_lineage(bundle, result)
            self._log_tags(bundle, result)
            if result.model_path:
                self._log_model(bundle, result)
            self._log_artifacts(result)

    def log_text(self, text: str, artifact_file: str, run_name: Optional[str] = None) -> None:
        """
        记录文本到 MLflow Artifact。

        若当前已有 active run，则复用；否则新建 run 后写入。
        """
        if not artifact_file or text is None:
            return

        with self._active_or_new_run(run_name=run_name):
            mlflow.log_text(str(text), artifact_file=artifact_file)

    def log_exception(
        self,
        exc: Exception,
        artifact_file: str = "logs/exception_traceback.txt",
        run_name: Optional[str] = None,
        extra_context: Optional[dict[str, str]] = None,
    ) -> None:
        """
        将异常信息格式化后写入 MLflow Artifact。
        """
        lines = []
        if extra_context:
            for k, v in extra_context.items():
                lines.append(f"{k}: {v}")
        lines.extend([
            f"exception_type: {type(exc).__name__}",
            f"exception_message: {str(exc)}",
            "",
            "traceback:",
            traceback.format_exc(),
        ])

        active = mlflow.active_run()
        if active is not None:
            mlflow.set_tag("run_status", "failed")
            mlflow.set_tag("exception_type", type(exc).__name__)

        self.log_text(
            text="\n".join(lines),
            artifact_file=artifact_file,
            run_name=run_name,
        )

    # ── ① params：展平一层嵌套 dict ──────────────────────────────────────────

    def _log_params(self, result: TrainResult) -> None:
        """
        MLflow 要求所有 param value 是标量。
        对一层嵌套 dict 自动展平：
          {"best_trial_params": {"lr": 0.01}} → {"best_trial_params.lr": 0.01}
        """
        flat: dict = {}
        for k, v in result.params_dict.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[f"{k}.{sub_k}"] = sub_v
            else:
                flat[k] = v
        mlflow.log_params(flat)

    # ── ② metrics ─────────────────────────────────────────────────────────────

    def _log_metrics(self, result: TrainResult) -> None:
        for key, value in result.metric_dict.items():
            if isinstance(value, list):
                for step, v in enumerate(value):
                    mlflow.log_metric(key, float(v), step=step)
            else:
                mlflow.log_metric(key, float(value))

    # ── ③ 数据集血缘 ───────────────────────────────────────────────────────────

    def _log_dataset_lineage(self, bundle: DataBundle, result: TrainResult) -> None:
        dvc_version = bundle.meta.get("dvc_version", "")
        dvc_repo    = bundle.meta.get("dvc_repo", "")
        source_name = f"{dvc_repo}@{dvc_version}" if dvc_version else "local"

        # 优先使用用户在 load_data() / mock_data() 中预先填入的 lineage_datasets，
        # 适用于多目标、聚类、图像等无法自动转为表格的场景。
        if bundle.lineage_datasets is not None:
            for dataset in bundle.lineage_datasets:
                try:
                    mlflow.log_input(dataset)
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"[MLflowRunLogger] log_input 失败 (lineage_datasets): {exc}"
                    )
            return

        for split_name, split_data in bundle.splits.items():
            X = split_data.get("X")
            y = split_data.get("y")
            if X is None or y is None:
                continue
            try:
                df = self._build_dataframe(X, y, result.feature_list, result.target_list)
            except (ValueError, TypeError) as exc:
                import logging
                logging.getLogger(__name__).warning(
                    f"[MLflowRunLogger] 跳过 split='{split_name}' 血缘上报: {exc}"
                )
                continue
            for target_col in result.target_list:
                try:
                    mlflow.log_input(
                        mlflow.data.from_pandas(df, name=source_name, targets=target_col),
                        context=split_name,
                    )
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"[MLflowRunLogger] log_input 失败 "
                        f"(split={split_name}, target={target_col}): {exc}"
                    )

    # ── ④ tags ────────────────────────────────────────────────────────────────

    def _log_tags(self, bundle: DataBundle, result: TrainResult) -> None:
        tags = dict(result.tag_dict)
        tags.update({k: str(v) for k, v in bundle.meta.items() if k.startswith("dvc_")})
        mlflow.set_tags(tags)

    # ── ⑤ 模型上报（注册表分发，无 if/elif 硬编码）──────────────────────────

    def _log_model(self, bundle: DataBundle, result: TrainResult) -> None:
        model_path = result.model_path
        if not model_path or not Path(model_path).exists():
            return

        ext = Path(model_path).suffix.lower().lstrip(".")
        registered_name = (
            self.config.registered_model_name or self.config.experiment_name
        ) + self.config.model_name_suffix

        try:
            sample_df   = self._get_sample_dataframe(bundle, result)
            model_input  = sample_df[result.feature_list]
            model_output = sample_df[result.target_list]
            signature    = mlflow.models.infer_signature(model_input, model_output)
        except Exception:
            signature   = None
            model_input = None

        log_fn = self._flavor_registry.get(ext)
        if log_fn is not None:
            log_fn(model_path, signature, model_input,
                   registered_name, self.config.register_model)
        else:
            # 未知格式：作为通用 artifact 保留，并给出明确提示
            import logging
            logging.getLogger(__name__).warning(
                f"[MLflowRunLogger] 未找到 '.{ext}' 对应的 MLflow flavor，"
                f"已作为普通 artifact 上报。"
                f"可通过 MLflowRunLogger.register_flavor('{ext}', your_fn) 注册。"
            )
            mlflow.log_artifact(model_path, artifact_path="model")

    # ── ⑥ artifacts ───────────────────────────────────────────────────────────

    def _log_artifacts(self, result: TrainResult) -> None:
        for path_str in result.artifact_file_paths:
            p = Path(path_str)
            if not p.exists():
                continue
            if p.is_file():
                mlflow.log_artifact(
                    str(p),
                    artifact_path=str(p.parent).lstrip("/").lstrip("./\\") or None,
                )
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        mlflow.log_artifact(
                            str(f),
                            artifact_path=str(f.parent).lstrip("/").lstrip("./\\"),
                        )

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    @contextmanager
    def _active_or_new_run(self, run_name: Optional[str] = None):
        active = mlflow.active_run()
        if active is not None:
            yield active
            return

        # 仅在需要内部创建 run 时设置 experiment。
        # 外部已有 active run 的场景会在上面直接复用，不触发此调用。
        if self._owned_run_id is None:
            mlflow.set_experiment(self.config.experiment_name)

        start_kwargs = (
            {"run_id": self._owned_run_id}
            if self._owned_run_id is not None
            else {"run_name": run_name}
        )

        with mlflow.start_run(**start_kwargs) as run:
            if self._owned_run_id is None:
                self._owned_run_id = run.info.run_id
            yield run

    @staticmethod
    def _build_dataframe(X, y, feature_list, target_list) -> pd.DataFrame:
        X_np = MLflowRunLogger._to_numpy(X)
        y_np = MLflowRunLogger._to_numpy(y)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        if X_np.shape[1] != len(feature_list):
            raise ValueError(f"feature_list 长度 ({len(feature_list)}) ≠ X 列数 ({X_np.shape[1]})")
        if y_np.shape[1] != len(target_list):
            raise ValueError(f"target_list 长度 ({len(target_list)}) ≠ y 列数 ({y_np.shape[1]})")
        df = pd.DataFrame(X_np, columns=feature_list, dtype=object)
        for i, col in enumerate(target_list):
            df[col] = y_np[:, i]
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
        return df

    @staticmethod
    def _to_numpy(data) -> np.ndarray:
        if data is None:
            return np.array([])
        try:
            import torch
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
        except ImportError:
            pass
        try:
            import tensorflow as tf
            if isinstance(data, tf.Tensor):
                return data.numpy()
        except ImportError:
            pass
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        if isinstance(data, np.ndarray):
            return data
        return np.array(data, dtype=object)

    def _get_sample_dataframe(self, bundle, result, n=3) -> pd.DataFrame:
        split_data = None
        for key in ("training", "train"):
            if key in bundle.splits and bundle.splits[key].get("X") is not None:
                split_data = bundle.splits[key]
                break
        if split_data is None:
            for v in bundle.splits.values():
                if v.get("X") is not None:
                    split_data = v
                    break
        if split_data is None:
            raise ValueError("DataBundle 中没有可用的 split 数据")
        return self._build_dataframe(
            split_data["X"], split_data["y"],
            result.feature_list, result.target_list,
        ).iloc[:n].reset_index(drop=True)
