"""
evaluate.py — 评估模块
支持 tf.keras 模型、onnxruntime session、tensorrt engine 的 AUC 计算
"""

import logging
import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def evaluate_keras(model, val_ds):
    """
    用 tf.keras 模型在 val_ds 上评估，返回 ctr_auc / cvr_auc。

    Args:
        model:  MMOERecModel
        val_ds: tf.data.Dataset，每个 batch 为 dict

    Returns:
        dict: {"ctr_auc": float, "cvr_auc": float}
    """
    import tensorflow as tf

    all_pred_ctr, all_pred_cvr = [], []
    all_label_ctr, all_label_cvr = [], []

    for batch in val_ds:
        preds = model({"sparse": batch["sparse"], "dense": batch["dense"]},
                      training=False)
        all_pred_ctr.append(preds[0].numpy())
        all_pred_cvr.append(preds[1].numpy())
        all_label_ctr.append(batch["label_ctr"].numpy())
        all_label_cvr.append(batch["label_cvr"].numpy())

    return _compute_auc(all_pred_ctr, all_pred_cvr, all_label_ctr, all_label_cvr)


def evaluate_onnx(session, val_dataset, batch_size):
    """
    用 onnxruntime session 评估，返回 ctr_auc / cvr_auc。

    Args:
        session:     onnxruntime.InferenceSession
        val_dataset: RecDataset（numpy）
        batch_size:  int
    """
    all_pred_ctr, all_pred_cvr = [], []
    all_label_ctr, all_label_cvr = [], []

    for batch in val_dataset.as_numpy_batches(batch_size):
        feeds = {
            "sparse": batch["sparse"],
            "dense":  batch["dense"],
        }
        outputs = session.run(None, feeds)
        # 模型输出顺序：[pred_ctr, pred_cvr]
        all_pred_ctr.append(outputs[0].flatten())
        all_pred_cvr.append(outputs[1].flatten())
        all_label_ctr.append(batch["label_ctr"])
        all_label_cvr.append(batch["label_cvr"])

    return _compute_auc(all_pred_ctr, all_pred_cvr, all_label_ctr, all_label_cvr)


def evaluate_trt(engine_runner, val_dataset, batch_size):
    """
    用 TensorRT engine runner 评估，返回 ctr_auc / cvr_auc。

    Args:
        engine_runner: TRTRunner 实例（见 quantize_trt.py）
        val_dataset:   RecDataset（numpy）
        batch_size:    int
    """
    all_pred_ctr, all_pred_cvr = [], []
    all_label_ctr, all_label_cvr = [], []

    for batch in val_dataset.as_numpy_batches(batch_size):
        outputs = engine_runner.infer(batch["sparse"], batch["dense"])
        all_pred_ctr.append(outputs[0].flatten())
        all_pred_cvr.append(outputs[1].flatten())
        all_label_ctr.append(batch["label_ctr"])
        all_label_cvr.append(batch["label_cvr"])

    return _compute_auc(all_pred_ctr, all_pred_cvr, all_label_ctr, all_label_cvr)


def _compute_auc(pred_ctr_list, pred_cvr_list, label_ctr_list, label_cvr_list):
    pred_ctr  = np.concatenate(pred_ctr_list)
    pred_cvr  = np.concatenate(pred_cvr_list)
    label_ctr = np.concatenate(label_ctr_list)
    label_cvr = np.concatenate(label_cvr_list)

    ctr_auc = roc_auc_score(label_ctr, pred_ctr)
    cvr_auc = roc_auc_score(label_cvr, pred_cvr)

    logger.debug("ctr_auc=%.4f  cvr_auc=%.4f", ctr_auc, cvr_auc)
    return {"ctr_auc": ctr_auc, "cvr_auc": cvr_auc}


def avg_auc(metrics):
    """返回 ctr_auc 和 cvr_auc 的均值。"""
    return (metrics["ctr_auc"] + metrics["cvr_auc"]) / 2.0
