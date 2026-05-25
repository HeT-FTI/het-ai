"""
案例五：无监督聚类（KMeans，双目标）
框架: scikit-learn  |  任务: 无监督聚类  |  导出: joblib pkl
"""
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from het_ai.studio import BaseTrainer, DataBundle, TrainConfig


class ClusteringTrainer(BaseTrainer):

    objectives = {
        'silhouette':        'maximize',
        'calinski_harabasz': 'maximize',
    }

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        import pandas as pd

        df = pd.read_csv(
            f"{dvc_data_root}/repository_0/files/label_data/data.csv"
        )
        X  = StandardScaler().fit_transform(df.values)
        return DataBundle(
            splits={'all': {'X': X}},
            feature_list=df.columns.tolist(),
        )

    def mock_data(self) -> DataBundle:
        rng   = np.random.default_rng(0)
        blobs = np.vstack([
            rng.normal(loc=c, scale=0.5, size=(50, 4))
            for c in [[0, 0, 0, 0], [5, 5, 0, 0], [0, 5, 5, 5]]
        ])
        X = StandardScaler().fit_transform(blobs)
        return DataBundle(
            splits={'all': {'X': X}},
            feature_list=[f'f{i}' for i in range(4)],
        )

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        n_clusters = BaseTrainer.TunableInt(2, 20),
        init       = BaseTrainer.TunableCategorical(['k-means++', 'random']),
        n_init     = BaseTrainer.TunableInt(5, 20),
        max_iter   = BaseTrainer.TunableInt(100, 500),
    )
    def train(self, data: DataBundle, n_clusters, init, n_init, max_iter):
        X      = data.splits['all']['X']
        model  = KMeans(
            n_clusters=int(n_clusters), init=init,
            n_init=int(n_init), max_iter=int(max_iter),
            random_state=self.config.random_state,
        )
        labels = model.fit_predict(X)

        if len(set(labels)) < 2:
            return BaseTrainer.Result(silhouette=0.0, calinski_harabasz=0.0), None

        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))
        ch  = calinski_harabasz_score(X, labels)
        return BaseTrainer.Result(
            silhouette=float(sil),
            calinski_harabasz=float(ch),
        ), model

    def select_best_trial(self, pareto_front):
        return max(
            pareto_front,
            key=lambda t: 0.7 * t.values[0] + 0.3 * t.values[1],
        )

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        path = f"{export_dir}/kmeans.pkl"
        joblib.dump(artifact, path)
        return path

    def predict(self, model_path: str, inputs):
        return joblib.load(model_path).predict(inputs)


def main(dvc_data_root: str = "dvc_data"):
    config  = TrainConfig()
    trainer = ClusteringTrainer(config)
    trainer.dry_run(dvc_data_root)
    return trainer.run(dvc_data_root).to_tuple()
