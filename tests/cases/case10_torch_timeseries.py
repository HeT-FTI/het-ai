"""
案例十：PyTorch LSTM 时序预测
框架: PyTorch  |  任务: 时序回归预测  |  导出: ONNX
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from het_ai.studio import BaseTrainer, DataBundle, TrainConfig


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class TimeSeriesTrainer(BaseTrainer):

    # ── 数据 ─────────────────────────────────────────────────────

    @staticmethod
    def _create_sequences(values, seq_len: int):
        X, y = [], []
        for i in range(len(values) - seq_len):
            X.append(values[i:i + seq_len])
            y.append(values[i + seq_len])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def load_data(self, dvc_data_root: str) -> DataBundle:
        import pandas as pd

        df     = pd.read_csv(
            f"{dvc_data_root}/repository_0/files/label_data/timeseries.csv"
        )
        values = df['value'].values.astype(np.float32)
        mean, std = values.mean(), values.std() + 1e-8
        values = (values - mean) / std

        seq_len   = 20
        X, y      = self._create_sequences(values, seq_len)
        split_idx = int(len(X) * (1 - self.config.test_size))

        return DataBundle(
            splits={
                'train': {
                    'X': torch.tensor(X[:split_idx]).unsqueeze(-1),
                    'y': torch.tensor(y[:split_idx]),
                },
                'val': {
                    'X': torch.tensor(X[split_idx:]).unsqueeze(-1),
                    'y': torch.tensor(y[split_idx:]),
                },
            },
            feature_list=['value'],
            target_list=['value_next'],
            meta={'seq_len': seq_len, 'input_dim': 1,
                  'mean': float(mean), 'std': float(std)},
        )

    def mock_data(self) -> DataBundle:
        rng    = np.random.default_rng(0)
        t      = np.linspace(0, 4 * np.pi, 300)
        values = (np.sin(t) + rng.normal(0, 0.1, 300)).astype(np.float32)

        seq_len   = 20
        X, y      = self._create_sequences(values, seq_len)
        split_idx = int(len(X) * 0.8)

        return DataBundle(
            splits={
                'train': {
                    'X': torch.tensor(X[:split_idx]).unsqueeze(-1),
                    'y': torch.tensor(y[:split_idx]),
                },
                'val': {
                    'X': torch.tensor(X[split_idx:]).unsqueeze(-1),
                    'y': torch.tensor(y[split_idx:]),
                },
            },
            feature_list=['value'],
            target_list=['value_next'],
            meta={'seq_len': seq_len, 'input_dim': 1,
                  'mean': 0.0, 'std': 1.0},
        )

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        lr         = BaseTrainer.TunableFloat(1e-4, 1e-2, log=True),
        hidden_dim = BaseTrainer.TunableCategorical([32, 64, 128]),
        num_layers = BaseTrainer.TunableInt(1, 3),
        dropout    = BaseTrainer.TunableFloat(0.0, 0.5),
        batch_size = BaseTrainer.TunableCategorical([16, 32, 64]),
        epochs     = BaseTrainer.TunableInt(20, 100),
    )
    def train(self, data: DataBundle, lr, hidden_dim, num_layers,
              dropout, batch_size, epochs):
        model     = LSTMForecaster(
            data.meta['input_dim'], int(hidden_dim),
            int(num_layers), dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        loader    = DataLoader(
            TensorDataset(data.X_train, data.y_train),
            batch_size=int(batch_size), shuffle=True,
        )
        best_mse = float('inf')

        for epoch in range(int(epochs)):
            model.train()
            for bx, by in loader:
                loss = criterion(model(bx), by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(data.X_val)
                val_mse  = criterion(val_pred, data.y_val).item()

            best_mse = min(best_mse, val_mse)
            if self.report(step=epoch, value=val_mse):
                break

        return -best_mse, model     # 取负数以 maximize

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        import onnxruntime as ort

        path  = f"{export_dir}/lstm_forecaster.onnx"
        dummy = torch.randn(1, 20, 1)        # (batch, seq_len, input_dim)
        torch.onnx.export(
            artifact, dummy, path,
            input_names=['input'], output_names=['output'],
            opset_version=self.config.onnx_opset_version,
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        )
        sess = ort.InferenceSession(path)
        sess.run(None, {'input': dummy.numpy()})
        return path

    def predict(self, model_path: str, inputs):
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path)
        return sess.run(None, {'input': inputs})[0]


def main(dvc_data_root: str = "dvc_data"):
    config  = TrainConfig(direction='maximize')
    trainer = TimeSeriesTrainer(config)
    trainer.dry_run(dvc_data_root)
    return trainer.run(dvc_data_root).to_tuple()
