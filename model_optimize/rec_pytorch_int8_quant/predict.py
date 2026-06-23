"""
predict.py — 推理对比脚本
分别加载 FP32 和 INT8 模型，用同一份假数据做推理，对比结果和耗时。
"""

import argparse
import copy
import logging
import time

import numpy as np
import torch
import yaml

from data import RecDataset
from model import build_model
from quantize import quantize_model, get_quantizable_layer_names
from evaluate import avg_auc
from logger_setup import init_logger

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  推理工具
# ------------------------------------------------------------------ #

def run_inference(model, sparse, dense, num_warmup=3, num_repeats=20):
    """
    对一个 batch 跑多次推理，返回预测结果和平均耗时（ms）。

    Args:
        model:       模型（eval 模式，CPU）
        sparse:      (B, F) long tensor
        dense:       (B, D) float tensor
        num_warmup:  预热次数，不计入耗时
        num_repeats: 正式计时次数

    Returns:
        pred_ctr (np.ndarray), pred_cvr (np.ndarray), avg_ms (float)
    """
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(num_warmup):
            model(sparse, dense)

        # 计时
        start = time.perf_counter()
        for _ in range(num_repeats):
            pred_ctr, pred_cvr = model(sparse, dense)
        elapsed = time.perf_counter() - start

    avg_ms = elapsed / num_repeats * 1000
    return pred_ctr.numpy(), pred_cvr.numpy(), avg_ms


# ------------------------------------------------------------------ #
#  模型加载
# ------------------------------------------------------------------ #

def load_fp32_model(cfg, checkpoint_path, device):
    """加载 FP32 模型权重到指定设备。"""
    model = build_model(cfg, device=device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("FP32 model loaded from: %s  device: %s", checkpoint_path, device)
    return model


def load_int8_model(int8_model_path):
    """
    加载保存的 INT8 模型。
    注意：PyTorch PTQ INT8 量化只支持 CPU，模型始终在 CPU 上运行。
    """
    model = torch.load(int8_model_path, map_location="cpu")
    model.eval()
    logger.info("INT8 model loaded from: %s  (PTQ runs on CPU)", int8_model_path)
    return model


# ------------------------------------------------------------------ #
#  主流程
# ------------------------------------------------------------------ #

def main(config_path, fp32_path, int8_path, num_predict_samples, batch_size):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    init_logger(cfg.get("logging", {}))

    d = cfg["data"]
    t = cfg["train"]

    # ---- 构造推理数据（固定 seed，保证两个模型用同一份数据）----
    logger.info("Building predict dataset: %d samples", num_predict_samples)
    dataset = RecDataset(
        num_samples=num_predict_samples,
        num_sparse_fields=d["num_sparse_fields"],
        sparse_vocab_size=d["sparse_vocab_size"],
        num_dense_features=d["num_dense_features"],
        seed=t["seed"] + 999,        # 用不同 seed，模拟线上新数据
    )
    sparse = dataset.sparse[:num_predict_samples]   # (N, F)
    dense  = dataset.dense[:num_predict_samples]    # (N, D)
    label_ctr = dataset.label_ctr[:num_predict_samples].numpy()
    label_cvr = dataset.label_cvr[:num_predict_samples].numpy()

    # 取第一个 batch 用于耗时对比（模拟线上单 batch 推理）
    sparse_batch = sparse[:batch_size]
    dense_batch  = dense[:batch_size]

    # ---- 加载 FP32 模型（放到 GPU）----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference device: %s", device)
    fp32_model = load_fp32_model(cfg, fp32_path, device)

    # ---- 加载 INT8 模型（PTQ 只支持 CPU）----
    int8_model = load_int8_model(int8_path)
    int8_device = torch.device("cpu")

    # ---- 全量推理（用于 AUC 对比）----
    logger.info("Running full inference on %d samples...", num_predict_samples)
    fp32_all_ctr, fp32_all_cvr = [], []
    int8_all_ctr, int8_all_cvr = [], []

    with torch.no_grad():
        for start in range(0, num_predict_samples, batch_size):
            end = min(start + batch_size, num_predict_samples)
            sp = sparse[start:end]
            de = dense[start:end]

            # FP32 推理（GPU）
            p_ctr, p_cvr = fp32_model(sp.to(device), de.to(device))
            fp32_all_ctr.append(p_ctr.cpu().numpy())
            fp32_all_cvr.append(p_cvr.cpu().numpy())

            # INT8 推理（CPU，PTQ 限制）
            p_ctr, p_cvr = int8_model(sp, de)
            int8_all_ctr.append(p_ctr.numpy())
            int8_all_cvr.append(p_cvr.numpy())

    fp32_pred_ctr = np.concatenate(fp32_all_ctr)
    fp32_pred_cvr = np.concatenate(fp32_all_cvr)
    int8_pred_ctr = np.concatenate(int8_all_ctr)
    int8_pred_cvr = np.concatenate(int8_all_cvr)

    # ---- AUC 对比 ----
    from sklearn.metrics import roc_auc_score
    fp32_ctr_auc = roc_auc_score(label_ctr, fp32_pred_ctr)
    fp32_cvr_auc = roc_auc_score(label_cvr, fp32_pred_cvr)
    int8_ctr_auc = roc_auc_score(label_ctr, int8_pred_ctr)
    int8_cvr_auc = roc_auc_score(label_cvr, int8_pred_cvr)

    fp32_avg = (fp32_ctr_auc + fp32_cvr_auc) / 2
    int8_avg = (int8_ctr_auc + int8_cvr_auc) / 2

    # ---- 单 batch 耗时对比 ----
    logger.info("Measuring latency on batch_size=%d ...", batch_size)
    _, _, fp32_ms = run_inference(fp32_model, sparse_batch.to(device), dense_batch.to(device))
    _, _, int8_ms = run_inference(int8_model, sparse_batch, dense_batch)
    speedup = fp32_ms / int8_ms if int8_ms > 0 else float("inf")

    # ---- 预测一致性（MSE）----
    mse_ctr = float(np.mean((fp32_pred_ctr - int8_pred_ctr) ** 2))
    mse_cvr = float(np.mean((fp32_pred_cvr - int8_pred_cvr) ** 2))

    # ---- 打印报告 ----
    logger.info("=" * 60)
    logger.info("PREDICT COMPARISON REPORT")
    logger.info("  Samples : %d  |  Batch size : %d", num_predict_samples, batch_size)
    logger.info("")
    logger.info("  AUC Comparison:")
    logger.info("  %-10s  ctr_auc=%.4f  cvr_auc=%.4f  avg=%.4f",
                "FP32", fp32_ctr_auc, fp32_cvr_auc, fp32_avg)
    logger.info("  %-10s  ctr_auc=%.4f  cvr_auc=%.4f  avg=%.4f",
                "INT8", int8_ctr_auc, int8_cvr_auc, int8_avg)
    logger.info("  AUC drop   : %.4f (avg)", fp32_avg - int8_avg)
    logger.info("")
    logger.info("  Latency (single batch, avg of 20 runs):")
    logger.info("  FP32 : %.3f ms", fp32_ms)
    logger.info("  INT8 : %.3f ms", int8_ms)
    logger.info("  Speedup : %.2fx", speedup)
    logger.info("")
    logger.info("  Prediction MSE (FP32 vs INT8):")
    logger.info("  CTR MSE : %.6f", mse_ctr)
    logger.info("  CVR MSE : %.6f", mse_cvr)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP32 vs INT8 inference comparison")
    parser.add_argument("--config",   default="config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--fp32",     default="checkpoints/best_model.pt",
                        help="FP32 model checkpoint path")
    parser.add_argument("--int8",     default="checkpoints/quantized_model.pt",
                        help="INT8 quantized model path")
    parser.add_argument("--samples",  type=int, default=10000,
                        help="Number of samples for inference")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for latency measurement")
    args = parser.parse_args()

    main(
        config_path=args.config,
        fp32_path=args.fp32,
        int8_path=args.int8,
        num_predict_samples=args.samples,
        batch_size=args.batch_size,
    )
