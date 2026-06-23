"""
evaluate.py — 评估模块
计算 CTR / CVR 任务的 AUC
"""

import logging
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def evaluate(model, loader, device):
    """
    在 loader 上跑一遍推理，返回 ctr_auc 和 cvr_auc。

    Returns:
        dict: {"ctr_auc": float, "cvr_auc": float}
    """
    model.eval()
    all_pred_ctr, all_pred_cvr = [], []
    all_label_ctr, all_label_cvr = [], []

    with torch.no_grad():
        for batch in loader:
            sparse = batch["sparse"].to(device)
            dense  = batch["dense"].to(device)
            pred_ctr, pred_cvr = model(sparse, dense)
            all_pred_ctr.append(pred_ctr.cpu().numpy())
            all_pred_cvr.append(pred_cvr.cpu().numpy())
            all_label_ctr.append(batch["label_ctr"].numpy())
            all_label_cvr.append(batch["label_cvr"].numpy())

    pred_ctr  = np.concatenate(all_pred_ctr)
    pred_cvr  = np.concatenate(all_pred_cvr)
    label_ctr = np.concatenate(all_label_ctr)
    label_cvr = np.concatenate(all_label_cvr)

    ctr_auc = roc_auc_score(label_ctr, pred_ctr)
    cvr_auc = roc_auc_score(label_cvr, pred_cvr)

    logger.debug("Evaluation done: ctr_auc=%.4f  cvr_auc=%.4f", ctr_auc, cvr_auc)
    return {"ctr_auc": ctr_auc, "cvr_auc": cvr_auc}


def avg_auc(metrics):
    """返回 ctr_auc 和 cvr_auc 的均值，作为综合指标。"""
    return (metrics["ctr_auc"] + metrics["cvr_auc"]) / 2.0
