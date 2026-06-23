"""
predict.py — FP32 vs INT8 推理对比脚本（TF 版本）
支持 onnxruntime 和 TensorRT 两种 INT8 后端。
"""

import argparse
import logging
import time
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from data import RecDataset
from evaluate import evaluate_keras, avg_auc
import quantize_onnx
import quantize_trt
from logger_setup import init_logger

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  推理耗时测量
# ------------------------------------------------------------------ #

def _measure_latency_keras(model, sparse_np, dense_np,
                            num_warmup=3, num_repeats=20):
    import tensorflow as tf
    sparse = tf.constant(sparse_np)
    dense  = tf.constant(dense_np)
    for _ in range(num_warmup):
        model({"sparse": sparse, "dense": dense}, training=False)
    start = time.perf_counter()
    for _ in range(num_repeats):
        model({"sparse": sparse, "dense": dense}, training=False)
    return (time.perf_counter() - start) / num_repeats * 1000


def _measure_latency_onnx(session, sparse_np, dense_np,
                           num_warmup=3, num_repeats=20):
    input_names = [i.name for i in session.get_inputs()]
    feeds = {}
    for name in input_names:
        if "sparse" in name.lower():
            feeds[name] = sparse_np
        else:
            feeds[name] = dense_np
    for _ in range(num_warmup):
        session.run(None, feeds)
    start = time.perf_counter()
    for _ in range(num_repeats):
        session.run(None, feeds)
    return (time.perf_counter() - start) / num_repeats * 1000


def _measure_latency_trt(runner, sparse_np, dense_np,
                          num_warmup=3, num_repeats=20):
    for _ in range(num_warmup):
        runner.infer(sparse_np, dense_np)
    start = time.perf_counter()
    for _ in range(num_repeats):
        runner.infer(sparse_np, dense_np)
    return (time.perf_counter() - start) / num_repeats * 1000


# ------------------------------------------------------------------ #
#  主流程
# ------------------------------------------------------------------ #

def main(config_path, fp32_model_dir, int8_path, num_predict_samples, batch_size):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    init_logger(cfg.get("logging", {}))

    d     = cfg["data"]
    t     = cfg["train"]
    q_cfg = cfg["quantize"]
    backend = q_cfg["backend"]

    # ---- 构造推理数据 ----
    logger.info("Building predict dataset: %d samples  backend=%s",
                num_predict_samples, backend)
    dataset = RecDataset(
        num_samples=num_predict_samples,
        num_sparse_fields=d["num_sparse_fields"],
        sparse_vocab_size=d["sparse_vocab_size"],
        num_dense_features=d["num_dense_features"],
        seed=t["seed"] + 999,
    )
    sparse_all = dataset.sparse[:num_predict_samples]
    dense_all  = dataset.dense[:num_predict_samples]
    label_ctr  = dataset.label_ctr[:num_predict_samples]
    label_cvr  = dataset.label_cvr[:num_predict_samples]

    sparse_batch = sparse_all[:batch_size]
    dense_batch  = dense_all[:batch_size]

    # ---- 加载 FP32 Keras 模型 ----
    import tensorflow as tf
    from model import build_model
    fp32_model = build_model(cfg)
    fp32_model.load_weights(t["checkpoint_path"])
    fp32_model.trainable = False
    logger.info("FP32 model loaded from: %s", t["checkpoint_path"])

    # ---- 加载 INT8 模型 ----
    if backend == "onnxruntime":
        int8_session = quantize_onnx.load_session(int8_path)
        logger.info("INT8 (onnxruntime) session loaded: %s", int8_path)
    elif backend == "tensorrt":
        trt_engine = quantize_trt.load_engine(int8_path)
        trt_runner = quantize_trt.TRTRunner(trt_engine)
        logger.info("INT8 (TensorRT) engine loaded: %s", int8_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # ---- 全量推理（AUC）----
    logger.info("Running full inference on %d samples...", num_predict_samples)
    fp32_pred_ctr_list, fp32_pred_cvr_list = [], []
    int8_pred_ctr_list, int8_pred_cvr_list = [], []

    for batch in dataset.as_numpy_batches(batch_size):
        sp = batch["sparse"]
        de = batch["dense"]

        # FP32
        p = fp32_model({"sparse": tf.constant(sp),
                         "dense":  tf.constant(de)}, training=False)
        fp32_pred_ctr_list.append(p[0].numpy())
        fp32_pred_cvr_list.append(p[1].numpy())

        # INT8
        if backend == "onnxruntime":
            out = quantize_onnx.infer(int8_session, sp, de)
        else:
            out = trt_runner.infer(sp, de)
        int8_pred_ctr_list.append(out[0].flatten())
        int8_pred_cvr_list.append(out[1].flatten())

    fp32_pred_ctr = np.concatenate(fp32_pred_ctr_list)
    fp32_pred_cvr = np.concatenate(fp32_pred_cvr_list)
    int8_pred_ctr = np.concatenate(int8_pred_ctr_list)
    int8_pred_cvr = np.concatenate(int8_pred_cvr_list)

    # ---- AUC ----
    fp32_ctr_auc = roc_auc_score(label_ctr, fp32_pred_ctr)
    fp32_cvr_auc = roc_auc_score(label_cvr, fp32_pred_cvr)
    int8_ctr_auc = roc_auc_score(label_ctr, int8_pred_ctr)
    int8_cvr_auc = roc_auc_score(label_cvr, int8_pred_cvr)
    fp32_avg = (fp32_ctr_auc + fp32_cvr_auc) / 2
    int8_avg = (int8_ctr_auc + int8_cvr_auc) / 2

    # ---- 单 batch 耗时 ----
    logger.info("Measuring latency on batch_size=%d ...", batch_size)
    fp32_ms = _measure_latency_keras(fp32_model, sparse_batch, dense_batch)
    if backend == "onnxruntime":
        int8_ms = _measure_latency_onnx(int8_session, sparse_batch, dense_batch)
    else:
        int8_ms = _measure_latency_trt(trt_runner, sparse_batch, dense_batch)
    speedup = fp32_ms / int8_ms if int8_ms > 0 else float("inf")

    # ---- 预测一致性 ----
    mse_ctr = float(np.mean((fp32_pred_ctr - int8_pred_ctr) ** 2))
    mse_cvr = float(np.mean((fp32_pred_cvr - int8_pred_cvr) ** 2))

    # ---- 报告 ----
    logger.info("=" * 60)
    logger.info("PREDICT COMPARISON REPORT  [backend=%s]", backend)
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
    parser = argparse.ArgumentParser(description="TF FP32 vs INT8 inference comparison")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--fp32",       default="checkpoints/saved_model")
    parser.add_argument("--int8",       default=None,
                        help="INT8 model path (.onnx or .trt, auto-detected from config)")
    parser.add_argument("--samples",    type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    with open(args.config) as f:
        _cfg = yaml.safe_load(f)

    # 若未指定 --int8，从 config 自动选择
    if args.int8 is None:
        backend = _cfg["quantize"]["backend"]
        if backend == "onnxruntime":
            int8_path = _cfg["quantize"]["onnx_int8_path"]
        else:
            int8_path = _cfg["quantize"]["trt_engine_path"]
    else:
        int8_path = args.int8

    main(
        config_path=args.config,
        fp32_model_dir=args.fp32,
        int8_path=int8_path,
        num_predict_samples=args.samples,
        batch_size=args.batch_size,
    )
