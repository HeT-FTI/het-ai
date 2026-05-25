"""
案例一：PyTorch 表格单目标分类
框架: PyTorch  |  任务: 监督分类  |  导出: ONNX
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from het_ai.studio import BaseTrainer, DataBundle, TrainConfig


class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class PytorchTabularTrainer(BaseTrainer):

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        df     = pd.read_csv(
            f"{dvc_data_root}/repository_0/files/label_data/data.csv"
        )
        X      = StandardScaler().fit_transform(
            df.drop(columns=['label']).values.astype(np.float32)
        )
        y      = df['label'].values.astype(np.int64)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        return DataBundle(
            splits={
                'train': {'X': torch.tensor(X_tr),  'y': torch.tensor(y_tr)},
                'val':   {'X': torch.tensor(X_val), 'y': torch.tensor(y_val)},
            },
            feature_list=df.drop(columns=['label']).columns.tolist(),
            target_list=['label'],
            meta={
                'num_classes': int(y.max()) + 1,
                'input_dim':   X.shape[1],
            },
        )

    def mock_data(self) -> DataBundle:
        torch.manual_seed(0)
        X_tr  = torch.randn(120, 4)
        y_tr  = torch.randint(0, 3, (120,))
        X_val = torch.randn(30, 4)
        y_val = torch.randint(0, 3, (30,))
        return DataBundle(
            splits={
                'train': {'X': X_tr,  'y': y_tr},
                'val':   {'X': X_val, 'y': y_val},
            },
            feature_list=['f1', 'f2', 'f3', 'f4'],
            target_list=['label'],
            meta={'num_classes': 3, 'input_dim': 4},
        )

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        lr           = BaseTrainer.TunableFloat(1e-4, 1e-2, log=True),
        batch_size   = BaseTrainer.TunableCategorical([16, 32, 64]),
        epochs       = BaseTrainer.TunableInt(10, 50),
        weight_decay = BaseTrainer.TunableFloat(0.0, 0.01),
    )
    def train(self, data: DataBundle, lr, batch_size, epochs, weight_decay):
        model     = SimpleNet(data.meta['input_dim'], data.meta['num_classes'])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loader = DataLoader(
            TensorDataset(data.X_train, data.y_train),
            batch_size=int(batch_size), shuffle=True,
        )
        best_acc = 0.0

        for epoch in range(int(epochs)):
            model.train()
            for bx, by in loader:
                loss = nn.CrossEntropyLoss()(model(bx), by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                acc = (
                    model(data.X_val).argmax(1) == data.y_val
                ).float().mean().item()

            best_acc = max(best_acc, acc)
            if self.report(step=epoch, value=acc):
                break

        return best_acc, model

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        import onnxruntime as ort

        path  = f"{export_dir}/model.onnx"
        dummy = torch.randn(1, artifact.fc[0].in_features)
        torch.onnx.export(
            artifact, dummy, path,
            input_names=['input'], output_names=['output'],
            opset_version=self.config.onnx_opset_version,
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        )
        # 验证导出正确性
        sess = ort.InferenceSession(path)
        sess.run(None, {'input': dummy.numpy()})
        return path

    def predict(self, model_path: str, inputs):
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path)
        return sess.run(None, {'input': inputs})[0]


def main(dvc_data_root: str = "dvc_data"):
    config  = TrainConfig()
    trainer = PytorchTabularTrainer(config)
    trainer.dry_run(dvc_data_root)
    return trainer.run(dvc_data_root).to_tuple()
