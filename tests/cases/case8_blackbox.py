"""
案例八：黑盒外部进程优化（调用任意可执行文件）
框架: subprocess  |  任务: 任意  |  导出: 外部程序产物
dry_run 时通过 _in_dry_run 标志完全短路外部调用。
"""
import json
import os
import shutil
import subprocess
import tempfile

from het_ai.studio import BaseTrainer, DataBundle, TrainConfig


class BlackBoxTrainer(BaseTrainer):

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        data_path = (
            f"{dvc_data_root}/repository_0/files/label_data/data.csv"
        )
        return DataBundle(
            splits={'all': {'data_path': data_path}},
            feature_list=['files'],
            target_list=['label'],
            meta={'data_path': data_path},
        )

    def mock_data(self) -> DataBundle:
        # dry_run 时 data_path 不会被真正读取
        tmp       = tempfile.mkdtemp()
        mock_path = os.path.join(tmp, 'mock_data.csv')
        return DataBundle(
            splits={'all': {'data_path': mock_path}},
            feature_list=['files'],
            target_list=['label'],
            meta={'data_path': mock_path},
        )

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        learning_rate = BaseTrainer.TunableFloat(1e-4, 1e-1, log=True),
        num_trees     = BaseTrainer.TunableInt(50, 500, step=50),
        max_depth     = BaseTrainer.TunableInt(3, 12),
        subsample     = BaseTrainer.TunableFloat(0.5, 1.0),
    )
    def train(self, data: DataBundle, learning_rate, num_trees,
              max_depth, subsample):

        # ── dry_run 短路：跳过真实 subprocess 调用 ───────────────
        if getattr(self, '_in_dry_run', False):
            self._logger.info("[mock] 跳过外部进程，返回合成结果")
            return 0.75, {'model_path': None, 'tmp_dir': None}

        # ── 生产路径 ──────────────────────────────────────────────
        tmp_dir     = tempfile.mkdtemp()
        config_path = os.path.join(tmp_dir, 'hparams.json')
        output_path = os.path.join(tmp_dir, 'result.json')

        with open(config_path, 'w') as f:
            json.dump({
                'data_path':     data.meta['data_path'],
                'learning_rate': learning_rate,
                'num_trees':     int(num_trees),
                'max_depth':     int(max_depth),
                'subsample':     subsample,
                'random_state':  self.config.random_state,
            }, f)

        proc = subprocess.run(
            ['./external_trainer',
             '--config', config_path,
             '--output', output_path],
            capture_output=True,
            timeout=300,
        )

        if proc.returncode != 0:
            self._logger.error(proc.stderr.decode())
            return 0.0

        with open(output_path) as f:
            result = json.load(f)

        return float(result['val_accuracy']), {
            'model_path': result.get('model_path'),
            'tmp_dir':    tmp_dir,
        }

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None or artifact.get('model_path') is None:
            return export_dir
        src = artifact['model_path']
        dst = os.path.join(export_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        return dst


def main(dvc_data_root: str = "dvc_data"):
    config  = TrainConfig(timeout=3600.0)
    trainer = BlackBoxTrainer(config)
    trainer.dry_run(dvc_data_root)
    return trainer.run(dvc_data_root).to_tuple()
