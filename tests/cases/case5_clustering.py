"""
案例五：无监督聚类（KMeans，双目标）
框架: scikit-learn  |  任务: 无监督聚类  |  导出: joblib pkl
"""
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from het_ai.mlflow import MLflowConfig
from het_ai.studio import BaseTrainer, DataBundle, TrainConfig, TrainResult

class ClusteringTrainer(BaseTrainer):

    objectives = {
        'silhouette':        'maximize',
        'calinski_harabasz': 'maximize',
    }

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        if dvc_data_root == '__mock__':
            return self.mock_data()

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
        from sklearn.model_selection import train_test_split

        X      = data.splits['all']['X']
        X_loss_train, X_loss_val = train_test_split(
            X,
            test_size=0.2,
            random_state=self.config.random_state,
            shuffle=True,
        )
        model  = KMeans(
            n_clusters=int(n_clusters), init=init,
            n_init=int(n_init), max_iter=1,
            random_state=self.config.random_state,
        )

        train_loss_history = []
        val_loss_history = []
        epochs = int(max_iter)
        for epoch in range(epochs):
            if epoch > 0:
                model = KMeans(
                    n_clusters=int(n_clusters),
                    init=model.cluster_centers_,
                    n_init=1,
                    max_iter=1,
                    random_state=self.config.random_state,
                )
            model.fit(X)
            centers = model.cluster_centers_
            train_loss = float(np.mean(np.min(
                np.sum((X_loss_train[:, None, :] - centers[None, :, :]) ** 2, axis=2),
                axis=1,
            )))
            val_loss = float(np.mean(np.min(
                np.sum((X_loss_val[:, None, :] - centers[None, :, :]) ** 2, axis=2),
                axis=1,
            )))
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            try:
                should_stop = self.report(step=epoch, value=-val_loss)
            except NotImplementedError:
                should_stop = False
            if should_stop:
                break

        labels = model.predict(X)

        if len(set(labels)) < 2:
            return BaseTrainer.Result(silhouette=0.0, calinski_harabasz=0.0), None, {
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
            }

        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))
        ch  = calinski_harabasz_score(X, labels)
        return BaseTrainer.Result(
            silhouette=float(sil),
            calinski_harabasz=float(ch),
        ), model, {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }

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
    
    def before_mlflow_log(self, result: TrainResult):
        # 在 MLflow 记录之前附加额外的文件或信息
        result.artifact_file_paths.append("./tests/cases/case5_clustering.py")
        return result


def main(dvc_data_root: str = "__mock__", mlflow_config: MLflowConfig | None = None):
    config  = TrainConfig(
        n_trials=10,
        mlflow=MLflowConfig(
            tracking_uri="http://10.12.8.110:5000",
            experiment_name="case5_clustering",
        ),
    )
    trainer = ClusteringTrainer(config)
    result = trainer.run(dvc_data_root)
    return result.to_tuple()

if __name__ == "__main__":
    print(main())
