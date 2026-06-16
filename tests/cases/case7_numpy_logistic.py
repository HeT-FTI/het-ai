"""
案例七：纯 NumPy 手写 Logistic Regression（Softmax + SGD）
框架: 无（纯 NumPy）  |  任务: 监督分类  |  导出: JSON
"""
import json
import numpy as np
from het_ai.mlflow import MLflowConfig
from het_ai.studio import BaseTrainer, DataBundle, TrainConfig, TrainResult

class NumpyLogisticTrainer(BaseTrainer):

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        if dvc_data_root == '__mock__':
            return self.mock_data()

        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        df     = pd.read_csv(
            f"{dvc_data_root}/repository_0/files/label_data/data.csv"
        )
        X      = StandardScaler().fit_transform(
            df.drop(columns=['label']).values
        )
        y      = LabelEncoder().fit_transform(df['label'].values)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        return DataBundle(
            splits={
                'train': {'X': X_tr,  'y': y_tr},
                'val':   {'X': X_val, 'y': y_val},
            },
            feature_list=df.drop(columns=['label']).columns.tolist(),
            target_list=['label'],
            meta={'num_classes': len(np.unique(y)), 'input_dim': X.shape[1]},
        )

    def mock_data(self) -> DataBundle:
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        X   = StandardScaler().fit_transform(rng.random((150, 4)))
        y   = rng.integers(0, 3, 150)
        return DataBundle(
            splits={
                'train': {'X': X[:120], 'y': y[:120]},
                'val':   {'X': X[120:], 'y': y[120:]},
            },
            feature_list=['f1', 'f2', 'f3', 'f4'],
            target_list=['label'],
            meta={'num_classes': 3, 'input_dim': 4},
        )

    # ── 静态工具 ─────────────────────────────────────────────────

    @staticmethod
    def _softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        lr         = BaseTrainer.TunableFloat(1e-4, 1.0, log=True),
        epochs     = BaseTrainer.TunableInt(50, 500, step=50),
        batch_size = BaseTrainer.TunableInt(16, 128, step=16),
        l2_lambda  = BaseTrainer.TunableFloat(0.0, 0.1),
    )
    def train(self, data: DataBundle, lr, epochs, batch_size, l2_lambda):
        X_tr, y_tr   = data.X_train, data.y_train
        X_val, y_val = data.X_val,   data.y_val
        n, d         = X_tr.shape
        C            = data.meta['num_classes']

        rng      = np.random.default_rng(self.config.random_state)
        W        = rng.standard_normal((d, C)) * 0.01
        b        = np.zeros(C)
        best_acc = 0.0
        best_W, best_b = W.copy(), b.copy()
        train_loss_history = []
        val_loss_history = []

        for epoch in range(int(epochs)):
            idx = rng.permutation(n)
            for start in range(0, n, int(batch_size)):
                batch = idx[start:start + int(batch_size)]
                Xb, yb = X_tr[batch], y_tr[batch]

                probs               = self._softmax(Xb @ W + b)
                dZ                  = probs.copy()
                dZ[np.arange(len(yb)), yb] -= 1
                dZ                 /= len(yb)

                W -= lr * (Xb.T @ dZ + l2_lambda * W)
                b -= lr * dZ.sum(axis=0)

            val_acc = (
                self._softmax(X_val @ W + b).argmax(axis=1) == y_val
            ).mean()

            train_probs = self._softmax(X_tr @ W + b)
            val_probs = self._softmax(X_val @ W + b)
            train_loss = -np.log(train_probs[np.arange(len(y_tr)), y_tr] + 1e-12).mean()
            val_loss = -np.log(val_probs[np.arange(len(y_val)), y_val] + 1e-12).mean()
            train_loss_history.append(float(train_loss))
            val_loss_history.append(float(val_loss))

            if val_acc > best_acc:
                best_acc       = val_acc
                best_W, best_b = W.copy(), b.copy()

            if self.report(step=epoch, value=val_acc):
                break

        return best_acc, {'W': best_W, 'b': best_b}, {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        path = f"{export_dir}/model_weights.json"
        with open(path, 'w') as f:
            json.dump(
                {'W': artifact['W'].tolist(), 'b': artifact['b'].tolist()}, f
            )
        return path

    def predict(self, model_path: str, inputs):
        with open(model_path) as f:
            w = json.load(f)
        W = np.array(w['W'])
        b = np.array(w['b'])
        return self._softmax(inputs @ W + b).argmax(axis=1)

    def before_mlflow_log(self, result: TrainResult):
        # 在 MLflow 记录之前附加额外的文件或信息
        result.artifact_file_paths.append("./tests/cases/case7_numpy_logistic.py")
        return result

def main(dvc_data_root: str = "__mock__", mlflow_config: MLflowConfig | None = None):
    config  = TrainConfig(
        n_trials=10,
        mlflow=MLflowConfig(
            tracking_uri="http://10.12.8.110:5000",
            experiment_name="case7_numpy_logistic",
        ),
    )
    trainer = NumpyLogisticTrainer(config)
    result = trainer.run(dvc_data_root)
    return result.to_tuple()

if __name__ == "__main__":
    print(main())
