"""
案例三：YOLOv8-seg 预训练实例分割（ONNX + TFLite 导出）
框架: Ultralytics/YOLO  |  任务: 监督分割（box+mask）  |  导出: ONNX / TFLite
"""
import numpy as np
from pathlib import Path
from het_ai.mlflow import MLflowConfig
from het_ai.studio import BaseTrainer, DataBundle, TrainConfig, TrainResult


class TFImageTrainer(BaseTrainer):

    def __init__(self, config=None):
        super().__init__(config)
        self._last_onnx_path = None
        self._last_run_dir = None

    # ── 数据 ─────────────────────────────────────────────────────

    @staticmethod
    def _discover_repositories(base_dir: Path) -> list[Path]:
        repos = []
        if not base_dir.exists():
            return repos
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.startswith('repository_'):
                repos.append(item)
        repos.sort(key=lambda p: p.name)
        return repos

    @staticmethod
    def _collect_seg_pairs(repositories: list[Path]) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        for repo in repositories:
            modal_root = repo / 'files' / 'modal_data'
            if not modal_root.exists():
                continue
            imgs = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                imgs.extend(modal_root.rglob(ext))

            for img in imgs:
                stem = img.stem
                label_candidates = [
                    repo / 'files' / 'label_data' / 'txt' / f'{stem}.txt',
                    repo / 'files' / 'label_data' / f'{stem}.txt',
                ]
                lbl = None
                for cand in label_candidates:
                    if cand.exists():
                        lbl = cand
                        break
                if lbl is None:
                    label_root = repo / 'files' / 'label_data'
                    if label_root.exists():
                        found = list(label_root.rglob(f'{stem}.txt'))
                        if found:
                            lbl = found[0]
                if lbl is not None:
                    pairs.append((img, lbl))
        return pairs

    @staticmethod
    def _infer_class_names_from_labels(label_paths: list[Path]) -> list[str]:
        cls_ids = set()
        for p in label_paths:
            try:
                with p.open('r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        cls_ids.add(int(line.split()[0]))
            except Exception:
                continue
        if not cls_ids:
            return ['class_0']
        max_id = max(cls_ids)
        return [f'class_{i}' for i in range(max_id + 1)]

    def _prepare_seg_dataset(self, data_root: Path) -> DataBundle:
        import random
        import tempfile
        import os
        import shutil

        repositories = self._discover_repositories(data_root)
        if repositories:
            pairs = self._collect_seg_pairs(repositories)
        else:
            # 兼容扁平目录: dvc_data/files/modal_data/jpg + dvc_data/files/label_data/txt
            pairs = []
            flat_img_root = data_root / 'files' / 'modal_data' / 'jpg'
            flat_lbl_root = data_root / 'files' / 'label_data' / 'txt'
            if flat_img_root.exists() and flat_lbl_root.exists():
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                    for img in flat_img_root.rglob(ext):
                        txt = flat_lbl_root / f'{img.stem}.txt'
                        tx = flat_lbl_root / f'{img.stem}.tx'
                        if txt.exists():
                            pairs.append((img, txt))
                        elif tx.exists():
                            pairs.append((img, tx))

            # 兼容直接给到 YOLO 标准目录的场景
            if not pairs:
                img_root = data_root / 'images'
                lbl_root = data_root / 'labels'
                if img_root.exists() and lbl_root.exists():
                    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                        for img in img_root.rglob(ext):
                            txt = lbl_root / f'{img.stem}.txt'
                            tx = lbl_root / f'{img.stem}.tx'
                            if txt.exists():
                                pairs.append((img, txt))
                            elif tx.exists():
                                pairs.append((img, tx))

        if not pairs:
            raise ValueError(f'在 {data_root} 中未找到可用的分割图像-标签对。')

        rng = random.Random(self.config.random_state)
        rng.shuffle(pairs)
        split_idx = int(len(pairs) * (1.0 - self.config.test_size))
        split_idx = min(max(split_idx, 1), len(pairs) - 1)
        train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]

        seg_root = Path(tempfile.mkdtemp(prefix='yolo_seg_'))
        for split in ('train', 'val'):
            (seg_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (seg_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

        def _link_or_copy(src: Path, dst: Path) -> None:
            try:
                os.symlink(str(src.resolve()), str(dst))
            except Exception:
                shutil.copy2(src, dst)

        for split_name, split_pairs in (('train', train_pairs), ('val', val_pairs)):
            for idx, (img, lbl) in enumerate(split_pairs):
                img_dst = seg_root / 'images' / split_name / f'{idx}_{img.name}'
                lbl_dst = seg_root / 'labels' / split_name / f'{idx}_{img.stem}.txt'
                _link_or_copy(img, img_dst)
                _link_or_copy(lbl, lbl_dst)

        class_names = self._infer_class_names_from_labels([lbl for _, lbl in pairs])
        data_yaml = seg_root / 'data.yaml'
        yaml_lines = [
            f'path: {seg_root.resolve()}',
            'train: images/train',
            'val: images/val',
            f'nc: {len(class_names)}',
            'names:',
        ]
        for idx, name in enumerate(class_names):
            yaml_lines.append(f'  {idx}: {name}')
        data_yaml.write_text('\n'.join(yaml_lines) + '\n', encoding='utf-8')

        return DataBundle(
            splits={
                'train': {
                    'X': np.array([str(p[0]) for p in train_pairs], dtype=object),
                    'y': np.array([str(p[1]) for p in train_pairs], dtype=object),
                },
                'val': {
                    'X': np.array([str(p[0]) for p in val_pairs], dtype=object),
                    'y': np.array([str(p[1]) for p in val_pairs], dtype=object),
                },
            },
            feature_list=['image_path'],
            target_list=['label_path'],
            meta={
                'mock_mode': False,
                'data_yaml': str(data_yaml),
                'seg_dataset_root': str(seg_root),
                'num_classes': len(class_names),
                'class_names': class_names,
            },
        )

    def _load_from_image_dir(self, image_root: str) -> DataBundle:
        import os
        import re
        from PIL import Image
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        IMG_SIZE                  = (224, 224)
        images, labels, image_paths = [], [], []

        for root, _, files in os.walk(image_root):
            for fname in files:
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                m = re.match(r'^(.*?)[_-]\d+', os.path.splitext(fname)[0])
                label = m.group(1) if m else os.path.splitext(fname)[0]
                try:
                    img = (
                        Image.open(os.path.join(root, fname))
                        .convert('RGB')
                        .resize(IMG_SIZE)
                    )
                    images.append(
                        np.array(img, dtype=np.float32) / 255.0
                    )
                    labels.append(label)
                    image_paths.append(os.path.join(root, fname))
                except Exception:
                    continue

        if not images:
            raise ValueError(
                f"在目录 {image_root} 中未找到可用图片（jpg/jpeg/png）"
            )

        le = LabelEncoder()
        y_enc = le.fit_transform(labels)
        X = np.stack(images)

        # 检查每个类的样本数，如果都>=2则使用分层抽样
        from collections import Counter
        class_counts = Counter(y_enc)
        use_stratify = all(count >= 2 for count in class_counts.values())

        (X_tr, X_val,
         y_tr, y_val,
         p_tr, p_val) = train_test_split(
            X, y_enc, image_paths,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_enc if use_stratify else None,
        )
        return DataBundle(
            splits={
                'train':       {'X': X_tr,  'y': y_tr},
                'val':         {'X': X_val, 'y': y_val},
            },
            feature_list=['files'],
            target_list=['class'],
            meta={
                'num_classes': len(le.classes_),
                'class_names': le.classes_.tolist(),
                'img_size':    IMG_SIZE,
                'train_paths': p_tr,
                'val_paths':   p_val,
                'mock_mode':   False,
            },
        )

    def load_data(self, dvc_data_root: str) -> DataBundle:
        if dvc_data_root == '__mock__':
            return self.mock_data()

        if dvc_data_root == '__dvc__':
            import os
            from het_ai.dvc import DVCConfig, DVCLoader

            dvc_config = DVCConfig(
                github_repo='https://github.com/qshixing/food_segmentation_data.git',
                github_token='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',
                minio_endpoint='10.12.8.110:9000',
                minio_access_key='admin',
                minio_secret_key='het@1234',
                minio_bucket='dvc-store',
                minio_virtual_folder='food_segmentation_data',
                minio_secure=False,
            )
            loader = DVCLoader(dvc_config)
            local_root = Path('dvc_data')
            tag, commit_sha = loader.pull(local_root)
            bundle = self._prepare_seg_dataset(local_root)
            return loader.enrich_bundle(bundle, tag, commit_sha)

        return self._prepare_seg_dataset(Path(dvc_data_root))

    def mock_data(self) -> DataBundle:
        return DataBundle(
            splits={
                'train':       {'X': np.array([], dtype=object), 'y': np.array([], dtype=object)},
                'val':         {'X': np.array([], dtype=object), 'y': np.array([], dtype=object)},
            },
            feature_list=['image_path'],
            target_list=['label_path'],
            meta={
                'mock_mode':   True,
                'data_yaml':   '',
                'seg_dataset_root': '',
            },
        )

    # ── 训练 ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_pretrained_weights() -> str:
        # 对齐参考脚本：优先使用项目内显式权重路径。
        local_weight = Path(__file__).resolve().parents[2] / 'yolov8n-seg.pt'
        return str(local_weight) if local_weight.exists() else 'yolov8n-seg.pt'

    @staticmethod
    def _resolve_device():
        try:
            import torch
            return 0 if torch.cuda.is_available() else 'cpu'
        except Exception:
            return 'cpu'

    @staticmethod
    def _extract_loss_history_from_csv(save_dir: Path) -> dict[str, list[float]]:
        import csv

        csv_path = save_dir / 'results.csv'
        if not csv_path.exists():
            return {
                'train_box_loss_history': [],
                'train_mask_loss_history': [],
                'val_box_loss_history': [],
                'val_mask_loss_history': [],
            }

        with csv_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return {
                'train_box_loss_history': [],
                'train_mask_loss_history': [],
                'val_box_loss_history': [],
                'val_mask_loss_history': [],
            }

        columns = [c.strip() for c in (reader.fieldnames or [])]

        def _pick_col(candidates: list[str]) -> str | None:
            lower_map = {c.lower(): c for c in columns}
            for key in candidates:
                if key in lower_map:
                    return lower_map[key]
            return None

        train_box_col = _pick_col(['train/box_loss', 'box_loss'])
        train_mask_col = _pick_col(['train/seg_loss', 'train/mask_loss', 'seg_loss', 'mask_loss'])
        val_box_col = _pick_col(['val/box_loss'])
        val_mask_col = _pick_col(['val/seg_loss', 'val/mask_loss'])

        def _to_float_list(col: str | None) -> list[float]:
            if not col:
                return []
            out: list[float] = []
            for row in rows:
                raw = (row.get(col) or '').strip()
                if not raw:
                    continue
                try:
                    out.append(float(raw))
                except ValueError:
                    continue
            return out

        return {
            'train_box_loss_history': _to_float_list(train_box_col),
            'train_mask_loss_history': _to_float_list(train_mask_col),
            'val_box_loss_history': _to_float_list(val_box_col),
            'val_mask_loss_history': _to_float_list(val_mask_col),
        }

    @BaseTrainer.search(
        learning_rate = BaseTrainer.TunableFloat(1e-4, 1e-2, log=True),
        freeze_layers = BaseTrainer.TunableInt(0, 10),
        batch_size    = BaseTrainer.TunableCategorical([8, 16]),
        epochs        = BaseTrainer.TunableInt(30, 80),
        imgsz         = BaseTrainer.TunableCategorical([640]),
        patience      = BaseTrainer.TunableInt(10, 30),
    )
    def train(self, data: DataBundle, learning_rate, freeze_layers,
              batch_size, epochs, imgsz, patience):
        if data.meta.get('mock_mode'):
            return 0.0, None, {
                'train_box_loss_history': [],
                'train_mask_loss_history': [],
                'val_box_loss_history': [],
                'val_mask_loss_history': [],
            }

        from ultralytics import YOLO
        trial_num = getattr(getattr(self._trial_local, 'current', None), 'number', -1)
        project_dir = Path(__file__).resolve().parents[2] / 'runs' / 'seg_train'

        model = YOLO(self._resolve_pretrained_weights())
        train_kwargs = {
            'data': data.meta['data_yaml'],
            'epochs': int(epochs),
            'batch': int(batch_size),
            'imgsz': int(imgsz),
            'lr0': float(learning_rate),
            'freeze': int(freeze_layers),
            'patience': int(patience),
            'workers': 4,
            'device': self._resolve_device(),
            'amp': False,
            'project': str(project_dir),
            'name': f'trial_{trial_num}',
            'exist_ok': True,
            'verbose': True,
        }
        try:
            results = model.train(**train_kwargs)
        except RuntimeError as e:
            if 'no trainable parameters left' not in str(e):
                raise
            self._logger.warning('freeze 参数过大导致全冻结，自动回退 freeze=0 重试。')
            train_kwargs['freeze'] = 0
            results = model.train(**train_kwargs)

        metrics = getattr(results, 'results_dict', {}) or {}
        best_acc = float(
            metrics.get('metrics/mAP50(M)')
            or metrics.get('metrics/mAP50-95(M)')
            or metrics.get('metrics/mAP50(B)')
            or 0.0
        )

        best_weights = None
        save_dir = getattr(results, 'save_dir', None)
        self._last_run_dir = str(save_dir) if save_dir is not None else None
        if save_dir is not None:
            candidate = Path(save_dir) / 'weights' / 'best.pt'
            if candidate.exists():
                best_weights = str(candidate)

        loss_hist = (
            self._extract_loss_history_from_csv(Path(save_dir))
            if save_dir is not None else {
                'train_box_loss_history': [],
                'train_mask_loss_history': [],
                'val_box_loss_history': [],
                'val_mask_loss_history': [],
            }
        )

        trained_artifact = YOLO(best_weights) if best_weights else model

        return best_acc, trained_artifact, {
            **loss_hist,
            'mask_map50': float(metrics.get('metrics/mAP50(M)', 0.0) or 0.0),
            'mask_map50_95': float(metrics.get('metrics/mAP50-95(M)', 0.0) or 0.0),
            'box_map50': float(metrics.get('metrics/mAP50(B)', 0.0) or 0.0),
            'box_map50_95': float(metrics.get('metrics/mAP50-95(B)', 0.0) or 0.0),
        }

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        import shutil

        self._last_onnx_path = None
        export_root = Path(export_dir)
        export_root.mkdir(parents=True, exist_ok=True)

        onnx_raw = Path(artifact.export(format='onnx', dynamic=True))
        tflite_raw = Path(artifact.export(format='tflite'))

        onnx_path = export_root / 'model.onnx'
        tflite_path = export_root / 'model.tflite'
        shutil.copy2(onnx_raw, onnx_path)
        shutil.copy2(tflite_raw, tflite_path)

        self._last_onnx_path = str(onnx_path)
        # 返回 tflite 作为 model_path，用于后续 MLflow 注册。
        return str(tflite_path)
        
    def before_mlflow_log(self, result: TrainResult):
        # 在 MLflow 记录之前附加额外的文件或信息
        result.artifact_file_paths.append("./tests/cases/case3_tflite_image.py")
        onnx_path = getattr(self, '_last_onnx_path', None)
        if onnx_path and Path(onnx_path).exists():
            result.artifact_file_paths.append(onnx_path)
        run_dir = getattr(self, '_last_run_dir', None)
        if run_dir and Path(run_dir).exists() and Path(run_dir).is_dir():
            result.artifact_file_paths.append(run_dir)
        # tflite 作为 model_path 进行注册，onnx 作为 artifact 额外上报。
        return result

    def on_study_end(self, study, best_trial) -> dict:
        return {'model_type': 'YOLOv8n-seg (box+mask)'}


def main(dvc_data_root: str = "__dvc__", mlflow_config: MLflowConfig | None = None):
    config  = TrainConfig(
        n_trials=2,
        mlflow=MLflowConfig(
            tracking_uri="http://10.12.8.110:5000",
            experiment_name="case3_tflite_image",
        ),
    )
    trainer = TFImageTrainer(config)
    result = trainer.run(dvc_data_root)
    return result.to_tuple()

if __name__ == "__main__":
    print(main())
