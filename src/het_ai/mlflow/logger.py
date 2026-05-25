from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import mlflow
import mlflow.data
import numpy as np
import pandas as pd

from het_ai.mlflow.config import MLflowConfig
from het_ai.studio.bundle import DataBundle
from het_ai.studio.result import TrainResult

if TYPE_CHECKING:
    pass


class MLflowRunLogger:
    """
    在 WorkflowRunner 执行结束后，将 TrainResult 完整上报到 MLflow。

    上报内容：
      ① mlflow.log_params       ← result.params_dict
      ② mlflow.log_metric(s)    ← result.metric_dict（list → per-step）
      ③ mlflow.log_input        ← DataBundle 各 split 的数据集血缘（含 DVC 版本标签）
      ④ mlflow.set_tags         ← result.tag_dict + bundle.meta 中所有 dvc_* 字段
      ⑤ mlflow.log_model        ← 自动按后缀分发 onnx / pt / tflite，生成 signature
      ⑥ mlflow.log_artifact(s)  ← result.artifact_file_paths
    """

    def __init__(self, config: MLflowConfig):
        self.config = config
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    # ──────────────────────────────────────────────────────────────────────────
    # 公共入口
    # ──────────────────────────────────────────────────────────────────────────

    def log(
        self,
        bundle: DataBundle,
        result: TrainResult,
        run_name: Optional[str] = None,
    ) -> None:
        """一次性完成所有 MLflow 上报，被 WorkflowRunner.execute() 自动调用。"""
        with mlflow.start_run(run_name=run_name):
            self._log_params(result)
            self._log_metrics(result)
            if self.config.log_dataset_lineage:
                self._log_dataset_lineage(bundle, result)
            self._log_tags(bundle, result)
            if result.model_path:
                self._log_model(bundle, result)
            self._log_artifacts(result)

    # ──────────────────────────────────────────────────────────────────────────
    # ① params
    # ──────────────────────────────────────────────────────────────────────────

    def _log_params(self, result: TrainResult) -> None:
        mlflow.log_params(result.params_dict)

    # ──────────────────────────────────────────────────────────────────────────
    # ② metrics
    # ──────────────────────────────────────────────────────────────────────────

    def _log_metrics(self, result: TrainResult) -> None:
        for key, value in result.metric_dict.items():
            if isinstance(value, list):
                # 训练曲线（loss per epoch 等），按 step 上报
                for step, v in enumerate(value):
                    mlflow.log_metric(key, float(v), step=step)
            else:
                mlflow.log_metric(key, float(value))

    # ──────────────────────────────────────────────────────────────────────────
    # ③ 数据集血缘
    # ──────────────────────────────────────────────────────────────────────────

    def _log_dataset_lineage(self, bundle: DataBundle, result: TrainResult) -> None:
        """
        遍历 DataBundle.splits，将每个 split（train / val / test）的
        X + y 拼成 DataFrame，上报到 MLflow Dataset。

        DVC 版本信息（bundle.meta 里的 dvc_version / dvc_repo 等）
        会作为 dataset 的 source name 写入，实现数据→实验的完整溯源。
        """
        # 数据集来源描述（优先用 DVC 标签，没有就用 "local"）
        dvc_version = bundle.meta.get("dvc_version", "")
        dvc_repo    = bundle.meta.get("dvc_repo", "")
        dataset_source_name = (
            f"{dvc_repo}@{dvc_version}" if dvc_version else "local"
        )

        for split_name, split_data in bundle.splits.items():
            X = split_data.get("X")
            y = split_data.get("y")

            # 跳过空 split（例如未提供 test set 的情况）
            if X is None or y is None:
                continue

            try:
                df = self._build_dataframe(
                    X=X,
                    y=y,
                    feature_list=result.feature_list,
                    target_list=result.target_list,
                )
            except (ValueError, TypeError) as exc:
                # 上报失败不应中断训练记录，降级为警告
                import logging
                logging.getLogger(__name__).warning(
                    f"[MLflowRunLogger] 跳过 split='{split_name}' 的数据集血缘上报: {exc}"
                )
                continue

            # 对每个 target 列分别上报一个 MLflow Dataset
            # （MLflow 的 from_pandas 要求 targets 是单列名）
            for target_col in result.target_list:
                try:
                    mlflow_dataset = mlflow.data.from_pandas(
                        df=df,
                        name=dataset_source_name,
                        targets=target_col,
                    )
                    mlflow.log_input(mlflow_dataset, context=split_name)
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"[MLflowRunLogger] mlflow.log_input 失败 "
                        f"(split={split_name}, target={target_col}): {exc}"
                    )

    @staticmethod
    def _build_dataframe(
        X,
        y,
        feature_list: list[str],
        target_list: list[str],
    ) -> pd.DataFrame:
        """
        将任意框架的 X / y 统一转换成 pandas DataFrame。

        支持：
          - PyTorch Tensor（GPU / CPU）
          - TensorFlow Tensor
          - NumPy ndarray
          - pandas DataFrame / Series
          - Python list

        拼接顺序：feature 列在前，target 列在后。
        列名来自 feature_list / target_list，长度不匹配时抛出 ValueError。
        """
        X_np = MLflowRunLogger._to_numpy(X)   # shape: (N, F)
        y_np = MLflowRunLogger._to_numpy(y)   # shape: (N,) 或 (N, T)

        # 保证 2D
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        # 列数校验
        if X_np.shape[1] != len(feature_list):
            raise ValueError(
                f"feature_list 长度 ({len(feature_list)}) "
                f"≠ X 列数 ({X_np.shape[1]})"
            )
        if y_np.shape[1] != len(target_list):
            raise ValueError(
                f"target_list 长度 ({len(target_list)}) "
                f"≠ y 列数 ({y_np.shape[1]})"
            )

        # 构造 DataFrame（dtype=object 先保留字符串，后续尝试数值化）
        df = pd.DataFrame(X_np, columns=feature_list, dtype=object)
        for i, col in enumerate(target_list):
            df[col] = y_np[:, i]

        # 能转数值的列自动转（保留字符串标签列原样）
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        return df

    @staticmethod
    def _to_numpy(data) -> np.ndarray:
        """将任意框架的张量/数组统一转换为 NumPy ndarray。"""
        if data is None:
            return np.array([])

        # PyTorch Tensor
        try:
            import torch
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
        except ImportError:
            pass

        # TensorFlow Tensor
        try:
            import tensorflow as tf
            if isinstance(data, tf.Tensor):
                return data.numpy()
        except ImportError:
            pass

        # pandas
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values

        # NumPy（直接返回，避免不必要拷贝）
        if isinstance(data, np.ndarray):
            return data

        # list / tuple 等其他可迭代对象
        return np.array(data, dtype=object)

    # ──────────────────────────────────────────────────────────────────────────
    # ④ tags
    # ──────────────────────────────────────────────────────────────────────────

    def _log_tags(self, bundle: DataBundle, result: TrainResult) -> None:
        tags = dict(result.tag_dict)
        # 将 DataBundle.meta 中所有 dvc_* 字段打入 MLflow tag
        # 这是"数据版本驱动"溯源闭环的关键一步
        dvc_meta = {k: str(v) for k, v in bundle.meta.items() if k.startswith("dvc_")}
        tags.update(dvc_meta)
        mlflow.set_tags(tags)

    # ──────────────────────────────────────────────────────────────────────────
    # ⑤ 模型上报与注册
    # ──────────────────────────────────────────────────────────────────────────

    def _log_model(self, bundle: DataBundle, result: TrainResult) -> None:
        """
        根据模型文件后缀自动分发到对应的 mlflow.*. log_model 方法。

        支持格式：
          .onnx   → mlflow.onnx.log_model
          .pt     → mlflow.pytorch.log_model
          .tflite → mlflow.pyfunc.log_model（TFLiteWrapper）

        同时自动推断 ModelSignature（输入 feature / 输出 target）。
        """
        model_path = result.model_path
        if not model_path or not Path(model_path).exists():
            return

        ext = Path(model_path).suffix.lower().lstrip(".")

        # 注册模型名称（优先用配置，其次用 experiment_name）
        registered_name = (
            self.config.registered_model_name
            or self.config.experiment_name
        ) + "_prod"

        # 构造 signature 所需的示例 DataFrame
        # 只取训练集的前 3 行，避免大数据集传输
        try:
            sample_df = self._get_sample_dataframe(bundle, result, n=3)
            model_input  = sample_df[result.feature_list]
            model_output = sample_df[result.target_list]
            signature = mlflow.models.infer_signature(model_input, model_output)
        except Exception:
            # signature 推断失败不阻断模型注册
            signature = None
            model_input = None

        if ext == "onnx":
            self._log_onnx(model_path, signature, model_input, registered_name)

        elif ext == "pt":
            self._log_pytorch(model_path, signature, model_input, registered_name)

        elif ext == "tflite":
            self._log_tflite(model_path, signature, model_input, registered_name)

        else:
            # 其他格式（joblib pickle、h5 等）作为通用 artifact 上报
            # 不注册为 MLflow Model，但保留文件
            mlflow.log_artifact(model_path, artifact_path="model")

    def _log_onnx(self, model_path, signature, input_example, registered_name):
        import onnx
        model_obj = onnx.load(model_path)
        mlflow.onnx.log_model(
            onnx_model=model_obj,
            name="model_onnx",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_name if self.config.register_model else None,
        )

    def _log_pytorch(self, model_path, signature, input_example, registered_name):
        import torch
        # weights_only=False 保留完整对象（含自定义类）
        model_obj = torch.load(model_path, map_location="cpu", weights_only=False)
        mlflow.pytorch.log_model(
            pytorch_model=model_obj,
            name="model_pt",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_name if self.config.register_model else None,
        )

    def _log_tflite(self, model_path, signature, input_example, registered_name):
        """
        TFLite 没有原生 mlflow flavor，用 pyfunc 包装。
        TFLiteWrapper 封装了 Interpreter 的完整推理逻辑，
        使注册后的模型可以直接通过 mlflow.pyfunc.load_model() 调用。
        """

        class TFLiteWrapper(mlflow.pyfunc.PythonModel):

            def load_context(self, context):
                # 优先用轻量的 tflite_runtime，回退到 tensorflow.lite
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
                # MLflow Serving 传入的始终是 DataFrame
                if not isinstance(model_input, pd.DataFrame):
                    model_input = pd.DataFrame(model_input)

                target_dtype = np.dtype(self._in["dtype"])
                target_shape = list(self._in["shape"])

                # 动态 batch size（-1 表示可变）
                if target_shape[0] == -1:
                    target_shape[0] = len(model_input)

                # 类型转换
                if np.issubdtype(target_dtype, np.number):
                    input_array = (
                        model_input.to_numpy()
                        .astype(target_dtype)
                        .reshape(target_shape)
                    )
                else:
                    input_array = (
                        model_input.to_numpy(dtype=object)
                        .reshape(target_shape)
                    )

                self.interpreter.set_tensor(self._in["index"], input_array)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self._out["index"])

                # bytes → str 解码（分类标签场景）
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
            registered_model_name=registered_name if self.config.register_model else None,
        )

    def _get_sample_dataframe(
        self,
        bundle: DataBundle,
        result: TrainResult,
        n: int = 3,
    ) -> pd.DataFrame:
        """从训练集取前 n 行，用于 signature 推断和 input_example。"""
        # 优先取 training，其次 train，最后取第一个非空 split
        preferred = ("training", "train")
        split_data = None
        for key in preferred:
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

        df = self._build_dataframe(
            X=split_data["X"],
            y=split_data["y"],
            feature_list=result.feature_list,
            target_list=result.target_list,
        )
        return df.iloc[:n].reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────────
    # ⑥ artifacts
    # ──────────────────────────────────────────────────────────────────────────

    def _log_artifacts(self, result: TrainResult) -> None:
        """
        上报所有附加文件（study_summary.json、train_config.yaml 等）。
        支持单文件和目录，目录会递归展开。
        """
        for path_str in result.artifact_file_paths:
            p = Path(path_str)
            if not p.exists():
                continue
            if p.is_file():
                # artifact_path 保留相对目录结构，方便在 MLflow UI 中浏览
                mlflow.log_artifact(
                    str(p),
                    artifact_path=str(p.parent).lstrip("/").lstrip("./\\") or None,
                )
            elif p.is_dir():
                for file in p.rglob("*"):
                    if file.is_file():
                        mlflow.log_artifact(
                            str(file),
                            artifact_path=str(file.parent).lstrip("/").lstrip("./\\"),
                        )
