"""
案例三：TensorFlow 图像分类（MobileNetV3 + TFLite 导出）
框架: TensorFlow/Keras  |  任务: 监督图像分类  |  导出: TFLite / Keras
"""
import numpy as np

from het_ai.studio import BaseTrainer, DataBundle, TrainConfig


class TFImageTrainer(BaseTrainer):

    # ── 数据 ─────────────────────────────────────────────────────

    def load_data(self, dvc_data_root: str) -> DataBundle:
        import os
        import re
        from PIL import Image
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        IMG_SIZE                    = (224, 224)
        images, labels, filenames   = [], [], []

        for root, _, files in os.walk(dvc_data_root):
            for fname in files:
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                m     = re.match(r'^(.*?)[_-]\d+', os.path.splitext(fname)[0])
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
                    filenames.append(fname)
                except Exception:
                    continue

        le    = LabelEncoder()
        y_enc = le.fit_transform(labels)
        X     = np.stack(images)

        (X_tr, X_val,
         y_tr, y_val,
         f_tr, f_val) = train_test_split(
            X, y_enc, filenames,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_enc,
        )
        return DataBundle(
            splits={
                'train':       {'X': X_tr,  'y': y_tr},
                'val':         {'X': X_val, 'y': y_val},
                'train_paths': f_tr,
                'val_paths':   f_val,
            },
            feature_list=['files'],
            target_list=['class'],
            meta={
                'num_classes': len(le.classes_),
                'class_names': le.classes_.tolist(),
                'img_size':    IMG_SIZE,
            },
        )

    def mock_data(self) -> DataBundle:
        rng                     = np.random.default_rng(0)
        IMG_SIZE                = (224, 224)
        n_tr, n_val, n_cls      = 20, 8, 3

        X_tr  = rng.random((n_tr,  *IMG_SIZE, 3), dtype=np.float32)
        X_val = rng.random((n_val, *IMG_SIZE, 3), dtype=np.float32)
        y_tr  = rng.integers(0, n_cls, n_tr)
        y_val = rng.integers(0, n_cls, n_val)

        return DataBundle(
            splits={
                'train':       {'X': X_tr,  'y': y_tr},
                'val':         {'X': X_val, 'y': y_val},
                'train_paths': [f'mock_train_{i}.jpg' for i in range(n_tr)],
                'val_paths':   [f'mock_val_{i}.jpg'   for i in range(n_val)],
            },
            feature_list=['files'],
            target_list=['class'],
            meta={
                'num_classes': n_cls,
                'class_names': [f'class_{i}' for i in range(n_cls)],
                'img_size':    IMG_SIZE,
            },
        )

    # ── 训练 ─────────────────────────────────────────────────────

    @BaseTrainer.search(
        learning_rate = BaseTrainer.TunableFloat(1e-5, 1e-2, log=True),
        dropout_rate  = BaseTrainer.TunableFloat(0.1, 0.5),
        dense_units   = BaseTrainer.TunableCategorical([128, 256, 512]),
        freeze_layers = BaseTrainer.TunableInt(60, 140, step=10),
        batch_size    = BaseTrainer.TunableCategorical([16, 32, 64]),
        es_patience   = BaseTrainer.TunableInt(3, 8),
    )
    def train(self, data: DataBundle, learning_rate, dropout_rate,
              dense_units, freeze_layers, batch_size, es_patience):
        import tensorflow as tf
        tf.keras.backend.clear_session()

        num_classes = data.meta['num_classes']
        img_size    = data.meta['img_size']

        base = tf.keras.applications.MobileNetV3Small(
            input_shape=(*img_size, 3),
            include_top=False, weights='imagenet', pooling='avg',
        )
        base.trainable = True
        for layer in base.layers[:int(freeze_layers)]:
            layer.trainable = False

        inputs  = tf.keras.Input(shape=(*img_size, 3))
        x       = base(inputs, training=False)
        x       = tf.keras.layers.Dropout(dropout_rate)(x)
        x       = tf.keras.layers.Dense(int(dense_units), activation='relu')(x)
        x       = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model   = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        history = model.fit(
            data.X_train, data.y_train,
            validation_data=(data.X_val, data.y_val),
            epochs=100,
            batch_size=int(batch_size),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=int(es_patience),
                    restore_best_weights=True,
                ),
            ],
            verbose=0,
        )
        best_acc = max(history.history['val_accuracy'])
        return best_acc, model

    # ── 导出 ─────────────────────────────────────────────────────

    def export_model(self, artifact, export_dir: str) -> str:
        if artifact is None:
            return export_dir
        import tensorflow as tf

        tflite_path = f"{export_dir}/model.tflite"
        try:
            inp = artifact.inputs[0]

            @tf.function(input_signature=[
                tf.TensorSpec(shape=inp.shape, dtype=inp.dtype)
            ])
            def serving_fn(x):
                return artifact(x, training=False)

            converter = tf.lite.TFLiteConverter.from_concrete_functions(
                [serving_fn.get_concrete_function()], artifact
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            with open(tflite_path, 'wb') as f:
                f.write(converter.convert())
            return tflite_path

        except Exception as e:
            self._logger.warning(f"TFLite 导出失败，回退为 Keras 格式: {e}")
            keras_path = f"{export_dir}/model.keras"
            artifact.save(keras_path)
            return keras_path

    def on_study_end(self, study, best_trial) -> dict:
        return {'model_type': 'MobileNetV3Small'}


def main(dvc_data_root: str = "dvc_data"):
    config  = TrainConfig()
    trainer = TFImageTrainer(config)
    trainer.dry_run(dvc_data_root)
    return trainer.run(dvc_data_root).to_tuple()
