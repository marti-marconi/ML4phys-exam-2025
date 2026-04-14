#!/usr/bin/env python3

# Train the Galaxy10 CNN once and saves the trained model and a compact metrics report

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau


SEED = int(os.getenv("SEED", "42"))
NUM_CLASSES = 10
EPOCHS = int(os.getenv("EPOCHS", "75"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "0") == "1"

# If present, these can further reduce thread pressure on shared machines.
if "TF_NUM_INTRAOP_THREADS" in os.environ:
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TF_NUM_INTRAOP_THREADS"]))
if "TF_NUM_INTEROP_THREADS" in os.environ:
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ["TF_NUM_INTEROP_THREADS"]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(69, 69, 3)),
            tf.keras.layers.Conv2D(32, (5, 5), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv2D(64, (5, 5), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(128, (5, 5), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


def augment_galaxy_rot_flips(
    x: np.ndarray,
    y: np.ndarray,
    class_aug_config: dict[int, dict[str, int | bool]],
) -> tuple[np.ndarray, np.ndarray]:
    x_aug: list[np.ndarray] = []
    y_aug: list[int] = []

    for c in np.unique(y):
        x_c = x[y == c]
        config = class_aug_config.get(int(c), {})

        for img in x_c:
            aug_imgs = [img.copy()]

            if config.get("rotations", False):
                n_rot = int(config.get("num_rotations", 0))
                if n_rot > 0:
                    temp = []
                    for im in aug_imgs:
                        rot_options = [np.rot90(im, k=k) for k in range(1, 4)]
                        temp.extend(random.sample(rot_options, k=min(n_rot, len(rot_options))))
                    aug_imgs.extend(temp)

            if config.get("flips", False):
                n_flip = int(config.get("num_flips", 0))
                if n_flip > 0:
                    temp = []
                    for im in aug_imgs:
                        flip_options = [np.fliplr(im), np.flipud(im), np.flipud(np.fliplr(im))]
                        temp.extend(random.sample(flip_options, k=min(n_flip, len(flip_options))))
                    aug_imgs.extend(temp)

            x_aug.extend(aug_imgs)
            y_aug.extend([int(c)] * len(aug_imgs))

    return np.asarray(x_aug, dtype=np.float32), np.asarray(y_aug, dtype=np.int32)


class ValidationMetricsCallback(Callback):
    """Track imbalance-aware validation metrics at the end of each epoch."""

    def __init__(self, x_val: np.ndarray, y_val: np.ndarray) -> None:
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.macro_f1: list[float] = []
        self.balanced_acc: list[float] = []
        self.per_class_metrics: list[dict[str, list[float] | list[int]]] = []

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        y_pred = np.argmax(self.model.predict(self.x_val, verbose=0), axis=1)
        macro_f1 = f1_score(self.y_val, y_pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(self.y_val, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_val,
            y_pred,
            zero_division=0,
        )

        self.macro_f1.append(float(macro_f1))
        self.balanced_acc.append(float(bal_acc))
        self.per_class_metrics.append(
            {
                "precision": [float(v) for v in precision],
                "recall": [float(v) for v in recall],
                "f1": [float(v) for v in f1],
                "support": [int(v) for v in support],
            }
        )

        # Expose custom metrics in fit logs for progress visibility.
        if logs is not None:
            logs["macro_f1"] = float(macro_f1)
            logs["balanced_acc"] = float(bal_acc)

        print(f"Epoch {epoch + 1}: Macro-F1={macro_f1:.3f}, Balanced Acc={bal_acc:.3f}")


def main() -> None:
    set_seed(SEED)

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    data_path = base_dir / "data" / "Galaxy10.h5"
    artifacts_dir = base_dir / "artifacts"
    logs_dir = base_dir / "logs"
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "galaxy_cnn.keras"
    metrics_path = artifacts_dir / "train_once_metrics.json"

    if model_path.exists() and not FORCE_RETRAIN:
        print(f"Model already exists at {model_path}. Skipping training.")
        print("Set FORCE_RETRAIN=1 to force retraining.")
        return

    with h5py.File(data_path, "r") as f:
        images = f["images"][:]
        labels = f["ans"][:]

    images = images.astype("float32") / 255.0

    x_train, x_temp, y_train, y_temp = train_test_split(
        images,
        labels,
        test_size=0.3,
        random_state=SEED,
        stratify=labels,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=y_temp,
    )

    class_aug_config = {
        0: {"rotations": False, "flips": False},
        1: {"rotations": False, "flips": False},
        2: {"rotations": False, "flips": False},
        3: {"rotations": False, "flips": False},
        4: {"rotations": False, "flips": False},
        5: {"rotations": True, "num_rotations": 3, "flips": True, "num_flips": 3},
        6: {"rotations": False, "flips": False},
        7: {"rotations": False, "flips": True, "num_flips": 1},
        8: {"rotations": False, "flips": True, "num_flips": 1},
        9: {"rotations": False, "flips": True, "num_flips": 1},
    }

    x_train_aug, y_train_aug = augment_galaxy_rot_flips(x_train, y_train, class_aug_config)

    classes = np.unique(y_train_aug)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_aug)
    class_weight_dict = dict(zip(classes.tolist(), class_weights.tolist()))

    y_train_aug_oh = tf.keras.utils.to_categorical(y_train_aug, NUM_CLASSES)
    y_val_oh = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

    model = build_model(NUM_CLASSES)

    val_metrics_cb = ValidationMetricsCallback(x_val=x_val, y_val=y_val)

    callbacks = [
        val_metrics_cb,
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, min_lr=1e-6, verbose=1),
    ]

    print("Starting training...")
    history = model.fit(
        x_train_aug,
        y_train_aug_oh,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val_oh),
        shuffle=True,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    history_path = artifacts_dir / "train_once_history.json"
    history_payload = {k: [float(vv) for vv in v] for k, v in history.history.items()}
    history_payload["macro_f1"] = [float(v) for v in val_metrics_cb.macro_f1]
    history_payload["balanced_acc"] = [float(v) for v in val_metrics_cb.balanced_acc]
    history_payload["per_class_metrics"] = val_metrics_cb.per_class_metrics

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    print(f"Saved history to {history_path}")

    model.save(model_path)
    print(f"Saved model to {model_path}")

    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "epochs_ran": len(history.history.get("loss", [])),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
