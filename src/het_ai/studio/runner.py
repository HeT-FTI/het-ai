import json
import logging
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import optuna

from het_ai.studio.result import TrainResult
from het_ai.studio.types import Result, TunableBase

if TYPE_CHECKING:
    from het_ai.studio.base import BaseTrainer
    from het_ai.studio.bundle import DataBundle

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class WorkflowRunner:
    """
    Internal orchestration engine; not intended for direct use by users.
    """

    def __init__(self, trainer: "BaseTrainer"):
        self.trainer = trainer
        self.config = trainer.config

    def execute(self, dvc_data_root: str) -> TrainResult:
        mlflow_cfg = getattr(self.config, "mlflow", None)
        self._validate()

        # ── Step 0: 自动 DVC 数据拉取（仅当 config.dvc 已配置时）────────────
        _dvc_tag, _dvc_sha = None, None
        dvc_cfg = getattr(self.config, "dvc", None)
        if dvc_cfg is not None:
            from het_ai.dvc.loader import DVCLoader
            loader = DVCLoader(dvc_cfg)
            logger.info("Step 0/4  自动拉取 DVC 数据...")
            _dvc_tag, _dvc_sha = loader.pull(Path(dvc_data_root))
            logger.info(
                f"Step 0/4  DVC 数据版本: tag={_dvc_tag}  sha="
                f"{_dvc_sha[:8] if _dvc_sha else ''}"
            )

        logger.info("Step 1/4  加载数据...")
        data = self.trainer.load_data(dvc_data_root)

        # 自动将 DVC 版本元数据注入 DataBundle.meta（不覆盖用户已设置的字段）
        if dvc_cfg is not None and _dvc_tag is not None:
            loader.enrich_bundle(data, _dvc_tag, _dvc_sha or "")

        logger.info("Step 2/4  启动 Optuna HPO...")
        study = self._run_study(data)

        logger.info("Step 3/4  处理最优 Trial...")
        best_trial = self._pick_best_trial(study)
        best_dir = best_trial.user_attrs.get("trial_dir", self.config.trial_root_dir)
        artifact = best_trial.user_attrs.get("__artifact__")

        logger.info("Step 4/4  导出模型...")
        export_path = self.trainer.export_model(artifact, best_dir)

        extra_tags = self.trainer.on_study_end(study, best_trial)
        summary_path = self._save_summary(study, best_trial, export_path)

        result = self._build_result(
            data, best_trial, export_path, summary_path, extra_tags
        )

        result = self.trainer.before_mlflow_log(result) or result

        if mlflow_cfg is not None:
            from het_ai.mlflow.logger import MLflowRunLogger
            MLflowRunLogger(mlflow_cfg).log(
                bundle=data,
                result=result,
                run_name=self.config.study_name,
            )

        # P6: 模型注册后回调钩子（供子类触发部署/通知）
        self.trainer.on_model_registered(result)

        return result

    def _validate(self):
        train_fn = type(self.trainer).train
        if not getattr(train_fn, "_is_tunable", False):
            raise RuntimeError(
                f"[{type(self.trainer).__name__}] train() 必须用 "
                "@BaseTrainer.search(...) 装饰。"
            )

    @staticmethod
    def _sample(trial, name: str, ghost):
        meta = ghost._meta
        t = meta["type"]

        if t == "int":
            return trial.suggest_int(
                name,
                meta["low"],
                meta["high"],
                step=meta.get("step", 1),
                log=meta.get("log", False),
            )

        if t == "float":
            kw = {"log": meta.get("log", False)}
            if meta.get("step") is not None:
                kw["step"] = meta["step"]
            return trial.suggest_float(name, meta["low"], meta["high"], **kw)

        if t == "categorical":
            return trial.suggest_categorical(name, meta["choices"])

        raise TypeError(f"未知 Tunable 类型: {type(ghost)}")

    def _build_objective(self, data: "DataBundle"):
        search_space = type(self.trainer).train._search_space
        trainer = self.trainer
        config = self.config

        def objective(trial: optuna.trial.Trial):
            trainer._trial_local.current = trial
            try:
                sampled = {
                    name: WorkflowRunner._sample(trial, name, ghost)
                    for name, ghost in search_space.items()
                    if isinstance(ghost, TunableBase)
                }

                trial_dir = Path(config.trial_root_dir) / str(uuid.uuid4())
                trial_dir.mkdir(parents=True, exist_ok=True)
                trial.set_user_attr("trial_dir", str(trial_dir))

                raw = trainer.train(data=data, **sampled)

                from het_ai.studio.base import BaseTrainer
                score, artifact, metrics_dict = BaseTrainer._unpack_train_result(raw)

                trial.set_user_attr("__artifact__", artifact)
                trial.set_user_attr("trial_params", sampled)

                # 记录自定义指标（支持列表和标量）
                if metrics_dict:
                    for k, v in metrics_dict.items():
                        trial.set_user_attr(f"metric_{k}", v)

                if isinstance(score, Result):
                    for k, v in score.data.items():
                        trial.set_user_attr(f"metric_{k}", float(v))
                    return score.get_values()

                trial.set_user_attr("metric_objective", float(score))
                return float(score)

            except Exception:
                # 保留 trial 原始异常堆栈，供上层一次性汇总并上报。
                trial.set_user_attr("__fail_reason__", traceback.format_exc())
                raise

            finally:
                trainer._trial_local.current = None

        return objective

    def _run_study(self, data: "DataBundle") -> optuna.Study:
        os.makedirs(self.config.trial_root_dir, exist_ok=True)

        pruner_map = {
            "median": optuna.pruners.MedianPruner(),
            "threshold": optuna.pruners.ThresholdPruner(lower=0.0),
            "none": optuna.pruners.NopPruner(),
        }
        pruner = pruner_map.get(self.config.pruner, optuna.pruners.MedianPruner())

        objectives_def = getattr(self.trainer, "objectives", None)

        if objectives_def:
            study = optuna.create_study(
                directions=list(objectives_def.values()),
                study_name=self.config.study_name,
                storage=self.config.storage,
                load_if_exists=True,
            )
        else:
            study = optuna.create_study(
                direction=self.config.direction,
                study_name=self.config.study_name,
                pruner=pruner,
                storage=self.config.storage,
                load_if_exists=True,
            )

        study.optimize(
            self._build_objective(data),
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            catch=(Exception,),
        )
        return study

    def _pick_best_trial(self, study: optuna.Study):
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise RuntimeError(self._format_no_completed_trial_error(study))

        objectives_def = getattr(self.trainer, "objectives", None)
        if objectives_def:
            pareto = study.best_trials
            if not pareto:
                raise RuntimeError(self._format_no_completed_trial_error(study))
            return self.trainer.select_best_trial(pareto)
        return study.best_trial

    def _format_no_completed_trial_error(self, study: optuna.Study) -> str:
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        if not failed:
            return "Optuna study 没有任何完成的 trial。"

        lines = [
            "Optuna study 没有任何完成的 trial。",
            f"failed_trials={len(failed)}",
            "failed_reasons:",
        ]

        for t in failed:
            reason = (
                t.user_attrs.get("__fail_reason__")
                or t.system_attrs.get("fail_reason")
                or "<no fail_reason>"
            )
            lines.append(f"- trial={t.number}")
            lines.append(f"  params={t.params}")
            lines.append(f"  reason={reason}")

        return "\n".join(lines)

    def _save_summary(self, study, best_trial, export_path: str) -> str:
        summary = {
            "study_name": self.config.study_name,
            "best_trial_number": best_trial.number,
            "best_value": best_trial.values or best_trial.value,
            "best_params": best_trial.params,
            "best_trial_dir": best_trial.user_attrs.get("trial_dir"),
            "export_model_path": export_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = Path(self.config.trial_root_dir) / "study_summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Study summary → {path}")
        return str(path)

    def _build_result(self, data, best_trial, export_path, summary_path, extra_tags) -> TrainResult:
        attrs = best_trial.user_attrs
        metric_dict = {
            k.replace("metric_", ""): v
            for k, v in attrs.items()
            if k.startswith("metric_")
        }
        return TrainResult(
            tag_dict={
                "hpo_framework": "Optuna",
                "study_name": self.config.study_name,
                **extra_tags,
            },
            params_dict={
                "n_trials": self.config.n_trials,
                "best_trial_number": best_trial.number,
                "best_trial_params": best_trial.params,
            },
            metric_dict=metric_dict,
            feature_list=data.feature_list,
            target_list=data.target_list,
            dataset_splits_dict=data.splits,
            model_path=export_path,
            artifact_file_paths=[summary_path],
            trainer=self.trainer,
        )
