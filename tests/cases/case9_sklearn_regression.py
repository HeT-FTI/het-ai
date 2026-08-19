"""
案例九：sklearn 随机森林回归
框架: scikit-learn  |  任务: 监督回归  |  导出: joblib pkl
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from het_ai.mlflow import MLflowConfig
from het_ai.studio import BaseTrainer, DataBundle, TrainConfig, TrainResult

class SklearnRegressionTrainer(BaseTrainer):

    objectives = {
        'r2':       'maximize',
        'neg_rmse': 'maximize',   # 取负以统一 maximize 方向
    }

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        if dvc_data_root == '__mock__':
            return self.mock_data()

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
            meta={'input_dim': X.shape[1], 'mock_mode': False},
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
            meta={'input_dim': 5, 'mock_mode': True},
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
        np.random.seed(self.config.random_state)
        total_estimators = int(n_estimators)
        if getattr(self, '_in_dry_run', False):
            total_estimators = min(total_estimators, 50)
        if data.meta.get('mock_mode', False):
            total_estimators = min(total_estimators, 20)

        model = RandomForestRegressor(
            n_estimators=1,
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            random_state=self.config.random_state,
            n_jobs=-1,
            warm_start=True,
        )
        train_loss_history = []
        val_loss_history = []
        best_rmse = float('inf')
        best_r2 = -float('inf')

        for epoch in range(1, total_estimators + 1):
            model.set_params(n_estimators=epoch)
            model.fit(data.X_train, data.y_train)

            y_tr_pred = model.predict(data.X_train)
            y_pred = model.predict(data.X_val)
            train_rmse = float(np.sqrt(mean_squared_error(data.y_train, y_tr_pred)))
            val_rmse = float(np.sqrt(mean_squared_error(data.y_val, y_pred)))
            train_loss_history.append(train_rmse)
            val_loss_history.append(val_rmse)
            cur_r2 = float(r2_score(data.y_val, y_pred))
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_r2 = cur_r2

            try:
                should_stop = self.report(step=epoch - 1, value=-val_rmse)
            except NotImplementedError:
                should_stop = False
            if should_stop:
                break

        return BaseTrainer.Result(
            r2=best_r2,
            neg_rmse=-best_rmse,
        ), model, {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }

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

    def before_mlflow_log(self, result: TrainResult):
        # 在 MLflow 记录之前附加额外的文件或信息
        result.artifact_file_paths.append("./tests/cases/case9_sklearn_regression.py")
        return result

def main(dvc_data_root: str = "__mock__", mlflow_config: MLflowConfig | None = None):
    n_trials = 10
    if dvc_data_root == "__mock__":
        n_trials = 3

    config  = TrainConfig(
        n_trials=n_trials,
        mlflow=MLflowConfig(
            tracking_uri="http://10.12.8.110:5000",
            experiment_name="case9_sklearn_regression",
        ),
    )
    trainer = SklearnRegressionTrainer(config)
    result = trainer.run(dvc_data_root)
    return result.to_tuple()

if __name__ == "__main__":
    print(main())
