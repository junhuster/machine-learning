"""
main.py — 主入口
串联：数据构建 → 模型训练 → ONNX 导出 → 量化搜索 → 保存最终模型和报告
"""

import argparse
import json
import logging
import os
import random

import numpy as np
import yaml

from logger_setup import init_logger

from data import build_datasets
from model import build_model
from train import train
from export import export_to_onnx, get_onnx_matmul_nodes, get_onnx_input_names
from evaluate import evaluate_onnx, evaluate_trt, avg_auc
from search import search_quantization_plan
import quantize_onnx
import quantize_trt


# ------------------------------------------------------------------ #
#  工具函数
# ------------------------------------------------------------------ #

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  主流程
# ------------------------------------------------------------------ #

def main(config_path):
    cfg = load_config(config_path)
    init_logger(cfg.get("logging", {}))
    logger = logging.getLogger("main")

    set_seed(cfg["train"]["seed"])
    backend = cfg["quantize"]["backend"]
    logger.info("Backend: %s", backend)

    # ---- Step 1: 构建数据 ----
    logger.info("Step 1: Building datasets...")
    train_loader, val_loader, train_ds, val_ds = build_datasets(cfg)

    # ---- Step 2: 训练 FP32 Keras 模型 ----
    logger.info("Step 2: Training FP32 model...")
    model = build_model(cfg)
    best_metrics = train(model, train_loader, val_loader, cfg)
    logger.info("FP32 best metrics: ctr_auc=%.4f  cvr_auc=%.4f  avg=%.4f",
                best_metrics["ctr_auc"], best_metrics["cvr_auc"],
                avg_auc(best_metrics))

    # ---- Step 3: 导出 ONNX ----
    logger.info("Step 3: Exporting to ONNX...")
    e_cfg = cfg["export"]
    export_to_onnx(
        saved_model_dir=cfg["train"]["saved_model_dir"],
        onnx_path=e_cfg["onnx_path"],
        opset=e_cfg["opset"],
    )

    # 获取所有可量化节点和输入名
    all_nodes   = get_onnx_matmul_nodes(e_cfg["onnx_path"])
    input_names = get_onnx_input_names(e_cfg["onnx_path"])
    logger.info("ONNX: %d quantizable nodes, inputs=%s",
                len(all_nodes), input_names)

    # ---- Step 4: 量化搜索 ----
    logger.info("Step 4: Searching quantization plan...")
    plan = search_quantization_plan(
        onnx_path=e_cfg["onnx_path"],
        all_nodes=all_nodes,
        val_dataset=val_ds,
        input_names=input_names,
        baseline_metrics=best_metrics,
        cfg=cfg,
    )

    # ---- Step 5: 构建并保存最终量化模型 ----
    q_cfg = cfg["quantize"]
    os.makedirs(os.path.dirname(q_cfg.get(
        "onnx_int8_path", "checkpoints/x")), exist_ok=True)

    nodes_to_exclude = [n for n in all_nodes if n not in plan["final_nodes"]]

    if plan["final_nodes"]:
        logger.info("Step 5: Building final quantized model (%s)...", backend)

        if backend == "onnxruntime":
            quantize_onnx.quantize(
                onnx_path=e_cfg["onnx_path"],
                output_path=q_cfg["onnx_int8_path"],
                nodes_to_exclude=nodes_to_exclude,
                val_dataset=val_ds,
                input_names=input_names,
                calibration_samples=q_cfg["calibration_samples"],
                batch_size=cfg["data"]["batch_size"],
            )
            logger.info("INT8 ONNX model saved to: %s", q_cfg["onnx_int8_path"])

        elif backend == "tensorrt":
            quantize_trt.build_engine(
                onnx_path=e_cfg["onnx_path"],
                engine_path=q_cfg["trt_engine_path"],
                layers_to_keep_fp32=nodes_to_exclude,
                val_dataset=val_ds,
                calibration_samples=q_cfg["calibration_samples"],
                cfg=cfg,
            )
            logger.info("TRT INT8 engine saved to: %s", q_cfg["trt_engine_path"])
    else:
        logger.warning("Step 5: No nodes quantized — skipping model save.")

    # ---- Step 6: 保存报告 ----
    report = {
        "backend":              backend,
        "baseline_ctr_auc":     best_metrics["ctr_auc"],
        "baseline_cvr_auc":     best_metrics["cvr_auc"],
        "baseline_avg_auc":     avg_auc(best_metrics),
        "final_avg_auc":        plan["final_auc"],
        "auc_drop":             plan["auc_drop"],
        "num_quantized_nodes":  len(plan["final_nodes"]),
        "num_sensitive_nodes":  len(plan["sensitive_nodes"]),
        "quantized_nodes":      plan["final_nodes"],
        "sensitive_nodes":      plan["sensitive_nodes"],
        "node_deltas":          plan["deltas"],
    }
    report_path = q_cfg["report_path"]
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to: %s", report_path)

    # ---- 最终摘要 ----
    logger.info("=" * 60)
    logger.info("SUMMARY  [backend=%s]", backend)
    logger.info("  FP32  avg_auc : %.4f", avg_auc(best_metrics))
    logger.info("  INT8  avg_auc : %.4f", plan["final_auc"])
    logger.info("  AUC drop      : %.4f", plan["auc_drop"])
    logger.info("  Quantized nodes : %d / %d",
                len(plan["final_nodes"]),
                len(plan["final_nodes"]) + len(plan["sensitive_nodes"]))
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF Rec INT8 Quantization")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
