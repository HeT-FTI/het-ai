"""
案例四：sklearn Pipeline（梯度提升 + 标准化）
框架: scikit-learn  |  任务: 监督分类  |  导出: joblib pkl
"""
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from het_ai.mlflow import MLflowConfig
from het_ai.studio import BaseTrainer, DataBundle, TrainConfig, TrainResult


class SklearnGBTrainer(BaseTrainer):

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        if dvc_data_root == '__mock__':
            return self.mock_data()

        import pandas as pd
        from sklearn.model_selection import train_test_split

        df     = pd.read_csv(
            f"{dvc_data_root}/repository_0/files/label_data/data.csv"
        )
        X      = df.drop(columns=['label']).values
        y      = df['label'].values
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        return DataBundle(
            splits={
                'train': {'X': X_tr,  'y': y_tr},
                'val':   {'X': X_val, 'y': y_val},
            },
            feature_list=df.drop(columns=['label']).columns.tolist(),
            target_list=['label'],
        )

    def mock_data(self) -> DataBundle:
        rng = np.random.default_rng(0)
        X   = StandardScaler().fit_transform(rng.random((150, 6)))
        y   = rng.integers(0, 3, 150)
        return DataBundle(
            splits={
                'train': {'X': X[:120], 'y': y[:120]},
                'val':   {'X': X[120:], 'y': y[120:]},
            },
            feature_list=[f'f{i}' for i in range(6)],
            target_list=['label'],
        )

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        n_estimators      = BaseTrainer.TunableInt(50, 500, step=50),
        max_depth         = BaseTrainer.TunableInt(2, 8),
        learning_rate     = BaseTrainer.TunableFloat(0.01, 0.3, log=True),
        subsample         = BaseTrainer.TunableFloat(0.6, 1.0),
        min_samples_split = BaseTrainer.TunableInt(2, 20),
    )
    def train(self, data: DataBundle, n_estimators, max_depth,
              learning_rate, subsample, min_samples_split):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    GradientBoostingClassifier(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                subsample=subsample,
                min_samples_split=int(min_samples_split),
                random_state=self.config.random_state,
            )),
        ])
        pipe.fit(data.X_train, data.y_train)

        scaler = pipe.named_steps['scaler']
        clf = pipe.named_steps['clf']
        X_train_scaled = scaler.transform(data.X_train)
        X_val_scaled = scaler.transform(data.X_val)

        train_loss_history = []
        val_loss_history = []
        for train_proba, val_proba in zip(
            clf.staged_predict_proba(X_train_scaled),
            clf.staged_predict_proba(X_val_scaled),
        ):
            train_loss_history.append(float(log_loss(
                data.y_train, train_proba, labels=clf.classes_
            )))
            val_loss_history.append(float(log_loss(
                data.y_val, val_proba, labels=clf.classes_
            )))

        score = f1_score(
            data.y_val, pipe.predict(data.X_val), average='weighted'
        )
        return score, pipe, {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        path = f"{export_dir}/pipeline.pkl"
        joblib.dump(artifact, path)
        return path

    def predict(self, model_path: str, inputs):
        return joblib.load(model_path).predict(inputs)

    def before_mlflow_log(self, result: TrainResult):
        # 在 MLflow 记录之前附加额外的文件或信息
        result.artifact_file_paths.append("./tests/cases/case4_sklearn_gbt.py")
        return result

def main(dvc_data_root: str = "__mock__", mlflow_config: MLflowConfig | None = None):
    config  = TrainConfig(
        n_trials=10,
        mlflow=MLflowConfig(
            tracking_uri="http://10.12.8.110:5000",
            experiment_name="case4_sklearn_gbt",
        ),
    )
    trainer = SklearnGBTrainer(config)
    result = trainer.run(dvc_data_root)
    return result.to_tuple()

if __name__ == "__main__":
    print(main())