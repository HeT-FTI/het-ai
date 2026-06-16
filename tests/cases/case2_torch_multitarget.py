"""
案例二：PyTorch 多目标分类（共享骨干 + 双头）
框架: PyTorch  |  任务: 多目标监督分类  |  导出: ONNX
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from os.path import exists
from het_ai.mlflow import MLflowConfig
from het_ai.studio import BaseTrainer, DataBundle, TrainConfig, TrainResult


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

        data_path = f"{dvc_data_root}/repository_0/files/label_data/data.csv"
        if not exists(data_path):
            return self.mock_data()

        df  = pd.read_csv(data_path)
        required_cols = {'f1', 'f2', 'f3', 'f4', 'target_1', 'target_2'}
        if not required_cols.issubset(set(df.columns)):
            return self.mock_data()

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
        train_loss_history = []
        val_loss_history = []

        for epoch in range(int(epochs)):
            model.train()
            train_loss_sum = 0.0
            train_samples = 0
            for bx, by1, by2 in loader:
                outs = model(bx)
                loss = sum(
                    w * nn.CrossEntropyLoss()(o, by)
                    for w, o, by in zip(loss_w, outs, [by1, by2])
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * by1.size(0)
                train_samples += by1.size(0)

            train_loss = train_loss_sum / max(1, train_samples)
            train_loss_history.append(float(train_loss))

            model.eval()
            val = data.splits['val']
            with torch.no_grad():
                outs = model(val['X'])
                val_loss = sum(
                    w * nn.CrossEntropyLoss()(o, y)
                    for w, o, y in zip(loss_w, outs, [val['y1'], val['y2']])
                )
                val_loss_history.append(float(val_loss.item()))
                accs = [
                    (o.argmax(1) == y).float().mean().item()
                    for o, y in zip(outs, [val['y1'], val['y2']])
                ]
            best_accs = [max(b, a) for b, a in zip(best_accs, accs)]

            # Optuna 的 multi-objective trial 不支持 report/pruning。
            try:
                if self.report(step=epoch, value=sum(accs) / len(accs)):
                    break
            except NotImplementedError:
                pass

        return BaseTrainer.Result(
            acc_target_1=best_accs[0],
            acc_target_2=best_accs[1],
        ), model, {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }

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
    
    def before_mlflow_log(self, result: TrainResult):
        # 在 MLflow 记录之前附加额外的文件或信息
        result.artifact_file_paths.append("./tests/cases/case2_torch_multitarget.py")
        return result


def main(dvc_data_root: str = "__mock__",):
    config  = TrainConfig(
        n_trials=10,
        mlflow=MLflowConfig(
            tracking_uri="http://10.12.8.110:5000",
            experiment_name="case2_torch_multitarget",
        ),
    )
    trainer = MultiTargetTrainer(config)
    result = trainer.run(dvc_data_root)
    return result.to_tuple()

if __name__ == "__main__":
    print(main())
