"""
案例九：sklearn 随机森林回归
框架: scikit-learn  |  任务: 监督回归  |  导出: joblib pkl
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from het_ai.studio import BaseTrainer, DataBundle, TrainConfig


class SklearnRegressionTrainer(BaseTrainer):

    objectives = {
        'r2':       'maximize',
        'neg_rmse': 'maximize',   # 取负以统一 maximize 方向
    }

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(
            f"{dvc_data_root}/repository_0/files/label_data/data.csv"
        )
        X = df.drop(columns=['target']).values.astype(np.float32)
        y = df['target'].values.astype(np.float32)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        return DataBundle(
            splits={
                'train': {'X': X_tr, 'y': y_tr},
                'val':   {'X': X_val, 'y': y_val},
            },
            feature_list=df.drop(columns=['target']).columns.tolist(),
            target_list=['target'],
            meta={'input_dim': X.shape[1]},
        )

    def mock_data(self) -> DataBundle:
        rng = np.random.default_rng(0)
        X   = StandardScaler().fit_transform(rng.random((200, 5)))
        y   = (X @ rng.random(5) + rng.normal(0, 0.1, 200)).astype(np.float32)
        return DataBundle(
            splits={
                'train': {'X': X[:160], 'y': y[:160]},
                'val':   {'X': X[160:], 'y': y[160:]},
            },
            feature_list=[f'f{i}' for i in range(5)],
            target_list=['target'],
            meta={'input_dim': 5},
        )

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        n_estimators     = BaseTrainer.TunableInt(50, 500, step=50),
        max_depth        = BaseTrainer.TunableInt(3, 20),
        min_samples_leaf = BaseTrainer.TunableInt(1, 20),
        max_features     = BaseTrainer.TunableCategorical(['sqrt', 'log2', None]),
    )
    def train(self, data: DataBundle, n_estimators, max_depth,
              min_samples_leaf, max_features):
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        model.fit(data.X_train, data.y_train)

        y_pred = model.predict(data.X_val)
        r2     = r2_score(data.y_val, y_pred)
        rmse   = float(np.sqrt(mean_squared_error(data.y_val, y_pred)))

        return BaseTrainer.Result(
            r2=float(r2),
            neg_rmse=-rmse,
        ), model

    def select_best_trial(self, pareto_front):
        return max(
            pareto_front,
            key=lambda t: 0.6 * t.values[0] + 0.4 * t.values[1],
        )

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        path = f"{export_dir}/rf_regressor.pkl"
        joblib.dump(artifact, path)
        return path

    def predict(self, model_path: str, inputs):
        return joblib.load(model_path).predict(inputs)


def main(dvc_data_root: str = "dvc_data"):
    config  = TrainConfig()
    trainer = SklearnRegressionTrainer(config)
    trainer.dry_run(dvc_data_root)
    return trainer.run(dvc_data_root).to_tuple()
