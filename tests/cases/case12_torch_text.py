"""
案例十二：PyTorch 文本分类（Embedding + TextCNN）
框架: PyTorch  |  任务: 文本监督分类  |  导出: ONNX
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from het_ai.mlflow import MLflowConfig
from het_ai.studio import BaseTrainer, DataBundle, TrainConfig, TrainResult

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int,
                 num_filters: int, filter_sizes: list, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len) long tensor
        emb = self.embedding(x).transpose(1, 2)   # (batch, embed, seq)
        conv_outs = [
            torch.relu(conv(emb)).max(dim=2).values for conv in self.convs
        ]
        cat = self.dropout(torch.cat(conv_outs, dim=1))
        return self.fc(cat)


class TextClassificationTrainer(BaseTrainer):

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        if dvc_data_root == '__mock__':
            return self.mock_data()

        import pandas as pd
        from sklearn.model_selection import train_test_split

        df    = pd.read_csv(
            f"{dvc_data_root}/repository_0/files/label_data/text.csv"
        )
        texts  = df['text'].tolist()
        labels = df['label'].values.astype(np.int64)

        # 简单 char-level tokenizer（生产环境可替换为任意 tokenizer）
        vocab  = {c: i + 1 for i, c in enumerate(
            sorted(set(''.join(texts)))
        )}
        max_len = 128

        def encode(text):
            ids = [vocab.get(c, 0) for c in text[:max_len]]
            return ids + [0] * (max_len - len(ids))

        X = np.array([encode(t) for t in texts], dtype=np.int64)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=labels,
        )
        return DataBundle(
            splits={
                'train': {'X': torch.tensor(X_tr), 'y': torch.tensor(y_tr)},
                'val':   {'X': torch.tensor(X_val), 'y': torch.tensor(y_val)},
            },
            feature_list=['text'],
            target_list=['label'],
            meta={
                'vocab_size':  len(vocab) + 1,
                'num_classes': int(labels.max()) + 1,
                'max_len':     max_len,
            },
        )

    def mock_data(self) -> DataBundle:
        rng     = np.random.default_rng(0)
        max_len = 128
        vocab_size = 50
        n_tr, n_val, n_cls = 100, 30, 4

        X_tr  = torch.tensor(rng.integers(0, vocab_size, (n_tr, max_len)))
        X_val = torch.tensor(rng.integers(0, vocab_size, (n_val, max_len)))
        y_tr  = torch.tensor(rng.integers(0, n_cls, n_tr))
        y_val = torch.tensor(rng.integers(0, n_cls, n_val))

        return DataBundle(
            splits={
                'train': {'X': X_tr, 'y': y_tr},
                'val':   {'X': X_val, 'y': y_val},
            },
            feature_list=['text'],
            target_list=['label'],
            meta={
                'vocab_size':  vocab_size,
                'num_classes': n_cls,
                'max_len':     max_len,
            },
        )

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        lr          = BaseTrainer.TunableFloat(1e-4, 1e-2, log=True),
        embed_dim   = BaseTrainer.TunableCategorical([64, 128, 256]),
        num_filters = BaseTrainer.TunableCategorical([64, 128, 256]),
        dropout     = BaseTrainer.TunableFloat(0.1, 0.5),
        batch_size  = BaseTrainer.TunableCategorical([32, 64]),
        epochs      = BaseTrainer.TunableInt(10, 50),
    )
    def train(self, data: DataBundle, lr, embed_dim, num_filters,
              dropout, batch_size, epochs):
        # Reproducibility: fix the global RNG so dry_run scores are stable.
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        model = TextCNN(
            vocab_size=data.meta['vocab_size'],
            embed_dim=int(embed_dim),
            num_classes=data.meta['num_classes'],
            num_filters=int(num_filters),
            filter_sizes=[2, 3, 4, 5],
            dropout=dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loader    = DataLoader(
            TensorDataset(data.X_train, data.y_train),
            batch_size=int(batch_size), shuffle=True,
        )
        best_acc = 0.0
        train_loss_history = []
        val_loss_history = []

        for epoch in range(int(epochs)):
            model.train()
            train_loss_sum = 0.0
            train_samples = 0
            for bx, by in loader:
                loss = criterion(model(bx), by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * by.size(0)
                train_samples += by.size(0)

            train_loss = train_loss_sum / max(1, train_samples)
            train_loss_history.append(float(train_loss))

            model.eval()
            with torch.no_grad():
                logits = model(data.X_val)
                val_loss = criterion(logits, data.y_val).item()
                val_loss_history.append(float(val_loss))
                acc = (logits.argmax(1) == data.y_val).float().mean().item()

            best_acc = max(best_acc, acc)
            if self.report(step=epoch, value=acc):
                break

        return best_acc, model, {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        import onnxruntime as ort

        path  = f"{export_dir}/text_cnn.onnx"
        dummy = torch.randint(0, 50, (1, 128))
        torch.onnx.export(
            artifact, dummy, path,
            input_names=['input_ids'], output_names=['logits'],
            opset_version=self.config.onnx_opset_version,
            dynamic_axes={
                'input_ids': {0: 'batch'},
                'logits':    {0: 'batch'},
            },
        )
        sess = ort.InferenceSession(path)
        sess.run(None, {'input_ids': dummy.numpy()})
        return path

    def predict(self, model_path: str, inputs):
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path)
        return sess.run(None, {'input_ids': inputs})[0]

    def before_mlflow_log(self, result: TrainResult):
        # 在 MLflow 记录之前附加额外的文件或信息
        result.artifact_file_paths.append("./tests/cases/case12_torch_text.py")
        return result

def main(dvc_data_root: str = "__mock__", mlflow_config: MLflowConfig | None = None):
    config  = TrainConfig(
        n_trials=10,
        mlflow=MLflowConfig(
            tracking_uri="http://10.12.8.110:5000",
            experiment_name="case12_torch_text",
        ),
    )
    trainer = TextClassificationTrainer(config)
    result = trainer.run(dvc_data_root)
    return result.to_tuple()

if __name__ == "__main__":
    print(main())
