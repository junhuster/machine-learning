"""
predict.py — FP32 vs TRT INT8 推理对比脚本

分别加载：
  1. FP32 PyTorch 模型（GPU 推理）
  2. TRT INT8 engine（GPU 推理，由 QDQ ONNX 构建）

用同一份假数据做推理，对比 AUC、延迟、预测一致性。

用法：
  python predict.py --config config.yaml
  python predict.py --config config.yaml \
      --fp32 checkpoints/best_model.pt \
      --trt  checkpoints/model_int8.trt \
      --samples 10000 --batch_size 512
"""

import argparse
import logging
import time

import numpy as np
import torch
import yaml

from logger_setup import init_logger
from data import RecDataset
from model import build_model
from onnx2trt import load_engine, TRTRunner, benchmark
from evaluate import avg_auc
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  延迟测量
# ------------------------------------------------------------------ #

def _latency_pytorch(model, sparse, dense, num_warmup=5, num_repeats=20):
    """测量 PyTorch 模型单 batch 推理延迟（ms）。"""
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            model(sparse, dense)
        torch.cuda.synchronize() if sparse.is_cuda else None

        start = time.perf_counter()
        for _ in range(num_repeats):
            model(sparse, dense)
        torch.cuda.synchronize() if sparse.is_cuda else None
        elapsed = time.perf_counter() - start

    return elapsed / num_repeats * 1000


def _latency_trt(runner, sparse_np, dense_np, num_warmup=5, num_repeats=20):
    """测量 TRT engine 单 batch 推理延迟（ms）。"""
    for _ in range(num_warmup):
        runner.infer(sparse_np, dense_np)
    start = time.perf_counter()
    for _ in range(num_repeats):
        runner.infer(sparse_np, dense_np)
    return (time.perf_counter() - start) / num_repeats * 1000


# ------------------------------------------------------------------ #
#  主流程
# ------------------------------------------------------------------ #

def main(config_path, fp32_path, trt_path, num_samples, batch_size):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    init_logger(cfg.get("logging", {}))

    d = cfg["data"]
    t = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference device: %s", device)

    # ---- 构造推理数据 ----
    logger.info("Building predict dataset: %d samples", num_samples)
    dataset = RecDataset(
        num_samples=num_samples,
        num_sparse_fields=d["num_sparse_fields"],
        sparse_vocab_size=d["sparse_vocab_size"],
        num_dense_features=d["num_dense_features"],
        seed=t["seed"] + 999,
    )
    sparse_all = dataset.sparse[:num_samples]
    dense_all  = dataset.dense[:num_samples]
    label_ctr  = dataset.label_ctr[:num_samples].numpy()
    label_cvr  = dataset.label_cvr[:num_samples].numpy()

    sparse_batch_np = sparse_all[:batch_size].numpy().astype(np.int32)
    dense_batch_np  = dense_all[:batch_size].numpy()

    # ---- 加载 FP32 PyTorch 模型 ----
    fp32_model = build_model(cfg, device)
    fp32_model.load_state_dict(torch.load(fp32_path, map_location=device))
    fp32_model.eval()
    logger.info("FP32 model loaded: %s", fp32_path)

    # ---- 加载 TRT INT8 engine ----
    trt_engine = load_engine(trt_path)
    trt_runner = TRTRunner(trt_engine)
    logger.info("TRT engine loaded: %s", trt_path)

    # ---- 全量推理（AUC 对比）----
    logger.info("Running full inference on %d samples...", num_samples)
    fp32_ctr_list, fp32_cvr_list = [], []
    trt_ctr_list,  trt_cvr_list  = [], []

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            sp  = sparse_all[start:end].to(device)
            de  = dense_all[start:end].to(device)

            p_ctr, p_cvr = fp32_model(sp, de)
            fp32_ctr_list.append(p_ctr.cpu().numpy())
            fp32_cvr_list.append(p_cvr.cpu().numpy())

            sp_np = sparse_all[start:end].numpy().astype(np.int32)
            de_np = dense_all[start:end].numpy()
            out   = trt_runner.infer(sp_np, de_np)
            trt_ctr_list.append(out["pred_ctr"].flatten())
            trt_cvr_list.append(out["pred_cvr"].flatten())

    fp32_pred_ctr = np.concatenate(fp32_ctr_list)
    fp32_pred_cvr = np.concatenate(fp32_cvr_list)
    trt_pred_ctr  = np.concatenate(trt_ctr_list)
    trt_pred_cvr  = np.concatenate(trt_cvr_list)

    fp32_ctr_auc = roc_auc_score(label_ctr, fp32_pred_ctr)
    fp32_cvr_auc = roc_auc_score(label_cvr, fp32_pred_cvr)
    trt_ctr_auc  = roc_auc_score(label_ctr, trt_pred_ctr)
    trt_cvr_auc  = roc_auc_score(label_cvr, trt_pred_cvr)
    fp32_avg = (fp32_ctr_auc + fp32_cvr_auc) / 2
    trt_avg  = (trt_ctr_auc  + trt_cvr_auc)  / 2

    # ---- 单 batch 延迟 ----
    logger.info("Measuring latency on batch_size=%d ...", batch_size)
    sp_t = sparse_all[:batch_size].to(device)
    de_t = dense_all[:batch_size].to(device)
    fp32_ms = _latency_pytorch(fp32_model, sp_t, de_t)
    trt_ms  = _latency_trt(trt_runner, sparse_batch_np, dense_batch_np)
    speedup = fp32_ms / trt_ms if trt_ms > 0 else float("inf")

    # ---- 预测一致性 ----
    mse_ctr = float(np.mean((fp32_pred_ctr - trt_pred_ctr) ** 2))
    mse_cvr = float(np.mean((fp32_pred_cvr - trt_pred_cvr) ** 2))

    # ---- 报告 ----
    logger.info("=" * 60)
    logger.info("PREDICT COMPARISON REPORT  [FP32 PyTorch vs TRT INT8]")
    logger.info("  Samples=%d  Batch=%d", num_samples, batch_size)
    logger.info("")
    logger.info("  AUC Comparison:")
    logger.info("  %-12s ctr_auc=%.4f  cvr_auc=%.4f  avg=%.4f",
                "FP32", fp32_ctr_auc, fp32_cvr_auc, fp32_avg)
    logger.info("  %-12s ctr_auc=%.4f  cvr_auc=%.4f  avg=%.4f",
                "TRT INT8", trt_ctr_auc, trt_cvr_auc, trt_avg)
    logger.info("  AUC drop   : %.4f", fp32_avg - trt_avg)
    logger.info("")
    logger.info("  Latency (avg of 20 runs):")
    logger.info("  FP32 PyTorch : %.3f ms", fp32_ms)
    logger.info("  TRT INT8     : %.3f ms", trt_ms)
    logger.info("  Speedup      : %.2fx", speedup)
    logger.info("")
    logger.info("  Prediction MSE (FP32 vs TRT INT8):")
    logger.info("  CTR MSE : %.6f", mse_ctr)
    logger.info("  CVR MSE : %.6f", mse_cvr)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP32 PyTorch vs TRT INT8 inference comparison")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--fp32",       default=None)
    parser.add_argument("--trt",        default=None)
    parser.add_argument("--samples",    type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    with open(args.config) as f:
        _cfg = yaml.safe_load(f)

    main(
        config_path=args.config,
        fp32_path=args.fp32 or _cfg["train"]["checkpoint_path"],
        trt_path=args.trt  or _cfg["tensorrt"]["engine_path"],
        num_samples=args.samples,
        batch_size=args.batch_size,
    )
