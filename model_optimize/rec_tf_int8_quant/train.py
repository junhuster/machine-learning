"""
train.py — 训练模块
训练 MMOE Keras 模型，保存最优检查点和 SavedModel
"""

import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from evaluate import evaluate_keras, avg_auc

logger = logging.getLogger(__name__)


class AucCallback(keras.callbacks.Callback):
    """每个 epoch 结束后计算 AUC，保存最优权重。"""

    def __init__(self, val_ds, checkpoint_path):
        super().__init__()
        self.val_ds = val_ds
        self.checkpoint_path = checkpoint_path
        self.best_auc = 0.0
        self.best_metrics = {}

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate_keras(self.model, self.val_ds)
        current = avg_auc(metrics)
        logger.info(
            "Epoch %d | ctr_auc=%.4f | cvr_auc=%.4f | avg_auc=%.4f",
            epoch + 1, metrics["ctr_auc"], metrics["cvr_auc"], current,
        )
        if current > self.best_auc:
            self.best_auc = current
            self.best_metrics = metrics
            self.model.save_weights(self.checkpoint_path)
            logger.info("  >> New best model saved (avg_auc=%.4f)", self.best_auc)


def _make_tf_dataset_with_labels(ds):
    """
    将 RecDataset 的 tf.data 输出重新整理为
    (inputs_dict, labels_dict) 格式，供 model.fit() 使用。
    """
    return ds.map(
        lambda b: (
            {"sparse": b["sparse"], "dense": b["dense"]},
            {"ctr": b["label_ctr"], "cvr": b["label_cvr"]},
        )
    )


def train(model, train_loader, val_loader, cfg):
    """
    完整训练流程。

    Returns:
        best_metrics (dict): 最优 epoch 的 ctr_auc / cvr_auc
    """
    t_cfg = cfg["train"]
    os.makedirs(os.path.dirname(t_cfg["checkpoint_path"]), exist_ok=True)

    train_ds = _make_tf_dataset_with_labels(train_loader)
    auc_cb   = AucCallback(val_loader, t_cfg["checkpoint_path"])

    lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=1, verbose=0
    )

    logger.info("Training for %d epochs...", t_cfg["epochs"])
    model.fit(
        train_ds,
        epochs=t_cfg["epochs"],
        callbacks=[auc_cb, lr_cb],
        verbose=0,
    )

    # 加载最优权重
    model.load_weights(t_cfg["checkpoint_path"])
    logger.info("Training complete. Best avg_auc=%.4f", auc_cb.best_auc)

    # 保存 SavedModel（用于后续 ONNX 导出）
    saved_model_dir = t_cfg["saved_model_dir"]
    os.makedirs(saved_model_dir, exist_ok=True)
    model.save(saved_model_dir)
    logger.info("SavedModel saved to: %s", saved_model_dir)

    return auc_cb.best_metrics
