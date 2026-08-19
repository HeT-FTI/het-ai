"""
案例十一：sklearn 异常检测（Isolation Forest）
框架: scikit-learn  |  任务: 无监督异常检测  |  导出: joblib pkl
"""
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from het_ai.mlflow import MLflowConfig
from het_ai.studio import BaseTrainer, DataBundle, TrainConfig, TrainResult

class AnomalyDetectionTrainer(BaseTrainer):

    objectives = {
        'f1':        'maximize',
        'precision': 'maximize',
    }

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        if dvc_data_root == '__mock__':
            return self.mock_data()

        import pandas as pd

        df = pd.read_csv(
            f"{dvc_data_root}/repository_0/files/label_data/data.csv"
        )
        X = StandardScaler().fit_transform(
            df.drop(columns=['is_anomaly']).values
        )
        y = df['is_anomaly'].values   # 1=正常, -1=异常
        return DataBundle(
            splits={'all': {'X': X, 'y': y}},
            feature_list=df.drop(columns=['is_anomaly']).columns.tolist(),
            target_list=['is_anomaly'],
            meta={'input_dim': X.shape[1]},
        )

    def mock_data(self) -> DataBundle:
        rng    = np.random.default_rng(0)
        normal = rng.normal(0, 1, (180, 5))
        anomal = rng.normal(0, 1, (20, 5)) + 5        # 离群
        X      = StandardScaler().fit_transform(
            np.vstack([normal, anomal])
        )
        y      = np.array([1] * 180 + [-1] * 20)
        return DataBundle(
            splits={'all': {'X': X, 'y': y}},
            feature_list=[f'f{i}' for i in range(5)],
            target_list=['is_anomaly'],
            meta={'input_dim': 5},
        )

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        n_estimators    = BaseTrainer.TunableInt(50, 500, step=50),
        max_samples     = BaseTrainer.TunableFloat(0.3, 1.0),
        contamination   = BaseTrainer.TunableFloat(0.01, 0.2),
        max_features    = BaseTrainer.TunableFloat(0.5, 1.0),
    )
    def train(self, data: DataBundle, n_estimators, max_samples,
              contamination, max_features):
        np.random.seed(self.config.random_state)
        from sklearn.model_selection import train_test_split

        X      = data.splits['all']['X']
        y_true = data.splits['all']['y']
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_true,
            test_size=0.2,
            random_state=self.config.random_state,
            stratify=y_true,
        )

        model = IsolationForest(
            n_estimators=1,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=self.config.random_state,
            n_jobs=-1,
            warm_start=True,
        )

        train_loss_history = []
        val_loss_history = []
        best_f1 = 0.0
        best_pre = 0.0
        for epoch in range(1, int(n_estimators) + 1):
            model.set_params(n_estimators=epoch)
            model.fit(X_train)

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            train_f1 = float(f1_score(y_train, y_train_pred, pos_label=-1))
            val_f1 = float(f1_score(y_val, y_val_pred, pos_label=-1))
            val_pre = float(precision_score(y_val, y_val_pred, pos_label=-1))
            train_loss_history.append(float(1.0 - train_f1))
            val_loss_history.append(float(1.0 - val_f1))

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_pre = val_pre

            try:
                should_stop = self.report(step=epoch - 1, value=val_f1)
            except NotImplementedError:
                should_stop = False
            if should_stop:
                break

        return BaseTrainer.Result(
            f1=best_f1,
            precision=best_pre,
        ), model, {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }

    def select_best_trial(self, pareto_front):
        return max(
            pareto_front,
            key=lambda t: 0.5 * t.values[0] + 0.5 * t.values[1],
        )

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        path = f"{export_dir}/isolation_forest.pkl"
        joblib.dump(artifact, path)
        return path

    def predict(self, model_path: str, inputs):
        return joblib.load(model_path).predict(inputs)

    def before_mlflow_log(self, result: TrainResult):
        # 在 MLflow 记录之前附加额外的文件或信息
        result.artifact_file_paths.append("./tests/cases/case11_sklearn_anomaly.py")
        return result

def main(dvc_data_root: str = "__mock__", mlflow_config: MLflowConfig | None = None):
    config  = TrainConfig(
        n_trials=10,
        mlflow=MLflowConfig(
            tracking_uri="http://10.12.8.110:5000",
            experiment_name="case11_sklearn_anomaly",
        ),
    )
    trainer = AnomalyDetectionTrainer(config)
    result = trainer.run(dvc_data_root)
    return result.to_tuple()

if __name__ == "__main__":
    print(main())
