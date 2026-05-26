"""
案例十一：sklearn 异常检测（Isolation Forest）
框架: scikit-learn  |  任务: 无监督异常检测  |  导出: joblib pkl
"""
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from het_ai.studio import BaseTrainer, DataBundle, TrainConfig


class AnomalyDetectionTrainer(BaseTrainer):

    objectives = {
        'f1':        'maximize',
        'precision': 'maximize',
    }

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
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
        X      = data.splits['all']['X']
        y_true = data.splits['all']['y']

        model = IsolationForest(
            n_estimators=int(n_estimators),
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        model.fit(X)
        y_pred = model.predict(X)

        f1  = f1_score(y_true, y_pred, pos_label=-1)
        pre = precision_score(y_true, y_pred, pos_label=-1)

        return BaseTrainer.Result(
            f1=float(f1),
            precision=float(pre),
        ), model

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


def main(dvc_data_root: str = "dvc_data"):
    config  = TrainConfig()
    trainer = AnomalyDetectionTrainer(config)
    trainer.dry_run(dvc_data_root)
    return trainer.run(dvc_data_root).to_tuple()
