"""
案例二：PyTorch 多目标分类（共享骨干 + 双头）
框架: PyTorch  |  任务: 多目标监督分类  |  导出: ONNX
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from het_ai.studio import BaseTrainer, DataBundle, TrainConfig


class MultiHeadNet(nn.Module):
    def __init__(self, input_dim: int, out_dims: list):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.heads    = nn.ModuleList([nn.Linear(32, d) for d in out_dims])

    def forward(self, x):
        feat = self.backbone(x)
        return [h(feat) for h in self.heads]


class MultiTargetTrainer(BaseTrainer):

    objectives = {'acc_target_1': 'maximize', 'acc_target_2': 'maximize'}

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df  = pd.read_csv(
            f"{dvc_data_root}/repository_0/files/label_data/data.csv"
        )
        X   = torch.tensor(
            df[['f1', 'f2', 'f3', 'f4']].values, dtype=torch.float32
        )
        y1  = torch.tensor(df['target_1'].values, dtype=torch.long)
        y2  = torch.tensor(df['target_2'].values, dtype=torch.long)

        idx                 = list(range(len(X)))
        tr_idx, val_idx     = train_test_split(
            idx,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        return DataBundle(
            splits={
                'train': {
                    'X': X[tr_idx], 'y1': y1[tr_idx], 'y2': y2[tr_idx],
                },
                'val': {
                    'X': X[val_idx], 'y1': y1[val_idx], 'y2': y2[val_idx],
                },
            },
            feature_list=['f1', 'f2', 'f3', 'f4'],
            target_list=['target_1', 'target_2'],
            meta={
                'input_dim':   4,
                'num_classes': [int(y1.max()) + 1, int(y2.max()) + 1],
            },
        )

    def mock_data(self) -> DataBundle:
        torch.manual_seed(0)
        X_tr  = torch.randn(120, 4)
        X_val = torch.randn(30, 4)
        return DataBundle(
            splits={
                'train': {
                    'X':  X_tr,
                    'y1': torch.randint(0, 3, (120,)),
                    'y2': torch.randint(0, 2, (120,)),
                },
                'val': {
                    'X':  X_val,
                    'y1': torch.randint(0, 3, (30,)),
                    'y2': torch.randint(0, 2, (30,)),
                },
            },
            feature_list=['f1', 'f2', 'f3', 'f4'],
            target_list=['target_1', 'target_2'],
            meta={'input_dim': 4, 'num_classes': [3, 2]},
        )

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        lr             = BaseTrainer.TunableFloat(1e-4, 1e-2, log=True),
        batch_size     = BaseTrainer.TunableCategorical([16, 32, 64]),
        epochs         = BaseTrainer.TunableInt(10, 50),
        loss_weight_t1 = BaseTrainer.TunableFloat(0.2, 0.8),
    )
    def train(self, data: DataBundle, lr, batch_size, epochs, loss_weight_t1):
        out_dims  = data.meta['num_classes']
        model     = MultiHeadNet(data.meta['input_dim'], out_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_w    = [loss_weight_t1, 1.0 - loss_weight_t1]

        tr     = data.splits['train']
        loader = DataLoader(
            TensorDataset(tr['X'], tr['y1'], tr['y2']),
            batch_size=int(batch_size), shuffle=True,
        )
        best_accs = [0.0, 0.0]

        for epoch in range(int(epochs)):
            model.train()
            for bx, by1, by2 in loader:
                outs = model(bx)
                loss = sum(
                    w * nn.CrossEntropyLoss()(o, by)
                    for w, o, by in zip(loss_w, outs, [by1, by2])
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val = data.splits['val']
            with torch.no_grad():
                outs = model(val['X'])
                accs = [
                    (o.argmax(1) == y).float().mean().item()
                    for o, y in zip(outs, [val['y1'], val['y2']])
                ]
            best_accs = [max(b, a) for b, a in zip(best_accs, accs)]

            if self.report(step=epoch, value=sum(accs) / len(accs)):
                break

        return BaseTrainer.Result(
            acc_target_1=best_accs[0],
            acc_target_2=best_accs[1],
        ), model

    def select_best_trial(self, pareto_front):
        return max(
            pareto_front,
            key=lambda t: sum(t.values) / len(t.values),
        )

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        path  = f"{export_dir}/model_multi.onnx"
        dummy = torch.randn(1, 4)
        torch.onnx.export(
            artifact, dummy, path,
            input_names=['input'],
            output_names=[f'output_{t}' for t in self.objectives],
            opset_version=self.config.onnx_opset_version,
            dynamic_axes={
                'input': {0: 'batch'},
                **{f'output_{t}': {0: 'batch'} for t in self.objectives},
            },
        )
        return path


def main(dvc_data_root: str = "dvc_data"):
    config  = TrainConfig()
    trainer = MultiTargetTrainer(config)
    trainer.dry_run(dvc_data_root)
    return trainer.run(dvc_data_root).to_tuple()
