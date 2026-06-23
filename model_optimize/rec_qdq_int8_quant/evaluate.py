"""
evaluate.py — 评估模块，支持 PyTorch 模型和 TensorRT engine
"""
import logging
import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def evaluate(model, loader, device):
    """PyTorch 模型评估，返回 {ctr_auc, cvr_auc}。"""
    import torch
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

    return _calc_auc(all_pred_ctr, all_pred_cvr, all_label_ctr, all_label_cvr)


def evaluate_trt(runner, val_dataset, batch_size):
    """TensorRT engine 评估，返回 {ctr_auc, cvr_auc}。"""
    all_pred_ctr, all_pred_cvr = [], []
    all_label_ctr, all_label_cvr = [], []
    n = len(val_dataset)

    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        sp   = val_dataset.sparse[start:end].numpy().astype(np.int32)
        de   = val_dataset.dense[start:end].numpy()
        out  = runner.infer(sp, de)
        all_pred_ctr.append(out[0].flatten())
        all_pred_cvr.append(out[1].flatten())
        all_label_ctr.append(val_dataset.label_ctr[start:end].numpy())
        all_label_cvr.append(val_dataset.label_cvr[start:end].numpy())

    return _calc_auc(all_pred_ctr, all_pred_cvr, all_label_ctr, all_label_cvr)


def _calc_auc(pred_ctr_list, pred_cvr_list, label_ctr_list, label_cvr_list):
    pred_ctr  = np.concatenate(pred_ctr_list)
    pred_cvr  = np.concatenate(pred_cvr_list)
    label_ctr = np.concatenate(label_ctr_list)
    label_cvr = np.concatenate(label_cvr_list)
    ctr_auc   = roc_auc_score(label_ctr, pred_ctr)
    cvr_auc   = roc_auc_score(label_cvr, pred_cvr)
    logger.debug("ctr_auc=%.4f  cvr_auc=%.4f", ctr_auc, cvr_auc)
    return {"ctr_auc": ctr_auc, "cvr_auc": cvr_auc}


def avg_auc(metrics):
    return (metrics["ctr_auc"] + metrics["cvr_auc"]) / 2.0
