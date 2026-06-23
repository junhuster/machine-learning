"""
main.py — 主入口
完整流程：训练 → calibration → 量化搜索 → 导出 ONNX → 构建 TRT engine
"""

import argparse
import copy
import json
import logging
import os
import random

import numpy as np
import torch
import yaml
from pytorch_quantization import quant_modules

from logger_setup import init_logger
from data import build_dataloaders
from model import build_model
from train import train
from evaluate import evaluate, avg_auc
from quantize import (calibrate, get_quantizer_names,
                      only_disable_with_keywords, enable_all)
from search import search_quantization_plan
from torch2onnx import export_onnx
from onnx2trt import build_engine

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path):
    cfg = load_config(config_path)
    init_logger(cfg.get("logging", {}))
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Step 1: 数据 ----
    logger.info("Step 1: Building data loaders...")
    train_loader, val_loader, val_ds = build_dataloaders(cfg)

    # ---- Step 2: 训练 FP32 模型 ----
    logger.info("Step 2: Training FP32 model...")
    fp32_model = build_model(cfg, device)
    best_metrics = train(fp32_model, train_loader, val_loader, cfg, device)
    logger.info("FP32 best: ctr_auc=%.4f  cvr_auc=%.4f  avg=%.4f",
                best_metrics["ctr_auc"], best_metrics["cvr_auc"],
                avg_auc(best_metrics))

    # ---- Step 3: 构建 QDQ 模型 ----
    # 必须在 quant_modules.initialize() 后重新 build_model，
    # 这样 nn.Linear 等会被自动替换为 QuantLinear（含 TensorQuantizer）
    logger.info("Step 3: Building QDQ model...")
    quant_modules.initialize()
    qdq_model = build_model(cfg, device)
    qdq_model.load_state_dict(
        torch.load(cfg["train"]["checkpoint_path"], map_location=device)
    )

    # ---- Step 4: Calibration（搜索最优 percentile）----
    q_cfg = cfg["quantize"]
    best_percentile = q_cfg.get("best_percentile")

    if best_percentile is None:
        logger.info("Step 4: Searching best percentile via binary search...")
        best_percentile, best_p_auc = _search_percentile(
            qdq_model, val_loader, device, q_cfg, cfg
        )
        logger.info("Best percentile=%.4f  avg_auc=%.4f", best_percentile, best_p_auc)
    else:
        logger.info("Step 4: Using specified percentile=%.4f", best_percentile)

    qdq_model = calibrate(
        qdq_model=qdq_model,
        loader=val_loader,
        calibration_samples=q_cfg["calibration_samples"],
        device=device,
        method="histogram",
        percentile=best_percentile,
    )

    # 全量化基线
    enable_all(qdq_model)
    full_quant_metrics = evaluate(qdq_model, val_loader, device)
    logger.info("Full quant: ctr_auc=%.4f  cvr_auc=%.4f  avg=%.4f",
                full_quant_metrics["ctr_auc"], full_quant_metrics["cvr_auc"],
                avg_auc(full_quant_metrics))

    # ---- Step 5: 量化层搜索 ----
    logger.info("Step 5: Searching quantization plan...")
    plan = search_quantization_plan(
        qdq_model=qdq_model,
        val_loader=val_loader,
        baseline_metrics=best_metrics,
        device=device,
        cfg=cfg,
    )

    # ---- Step 6: 导出最终 QDQ ONNX ----
    logger.info("Step 6: Exporting final QDQ ONNX...")
    sensitive_layers = plan["sensitive_layers"]
    only_disable_with_keywords(qdq_model, sensitive_layers)
    onnx_qdq_path = cfg["export"]["onnx_qdq_path"]
    export_onnx(qdq_model, onnx_qdq_path, cfg, device)

    # 同时导出 FP32 基线 ONNX
    onnx_fp32_path = cfg["export"]["onnx_fp32_path"]
    fp32_model.eval()
    export_onnx(fp32_model, onnx_fp32_path, cfg, device)

    # ---- Step 7: 构建 TRT INT8 engine ----
    logger.info("Step 7: Building TRT INT8 engine...")
    build_engine(
        onnx_path=onnx_qdq_path,
        engine_path=cfg["tensorrt"]["engine_path"],
        precision="int8",
        cfg=cfg,
    )

    # ---- Step 8: 保存报告 ----
    report = {
        "best_percentile":      best_percentile,
        "baseline_ctr_auc":     best_metrics["ctr_auc"],
        "baseline_cvr_auc":     best_metrics["cvr_auc"],
        "baseline_avg_auc":     avg_auc(best_metrics),
        "full_quant_avg_auc":   avg_auc(full_quant_metrics),
        "final_avg_auc":        plan["final_auc"],
        "auc_drop":             plan["auc_drop"],
        "num_quantized_layers": len(plan["final_layers"]),
        "num_sensitive_layers": len(plan["sensitive_layers"]),
        "quantized_layers":     plan["final_layers"],
        "sensitive_layers":     plan["sensitive_layers"],
        "layer_deltas":         plan["deltas"],
    }
    report_path = q_cfg["report_path"]
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved: %s", report_path)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  FP32  avg_auc     : %.4f", avg_auc(best_metrics))
    logger.info("  FullQ avg_auc     : %.4f", avg_auc(full_quant_metrics))
    logger.info("  Final avg_auc     : %.4f", plan["final_auc"])
    logger.info("  AUC drop          : %.4f", plan["auc_drop"])
    logger.info("  Quantized layers  : %d / %d",
                len(plan["final_layers"]),
                len(plan["final_layers"]) + len(plan["sensitive_layers"]))
    logger.info("  QDQ ONNX          : %s", onnx_qdq_path)
    logger.info("  TRT engine        : %s", cfg["tensorrt"]["engine_path"])
    logger.info("=" * 60)


# ------------------------------------------------------------------ #
#  Percentile 二分搜索（简化版）
# ------------------------------------------------------------------ #

def _search_percentile(qdq_model, val_loader, device, q_cfg, cfg):
    """
    在 [percentile_left, percentile_right] 范围内二分搜索最优 percentile。
    每次评估：calibrate → 全量化 → evaluate，取 avg_auc 最高的 percentile。
    """
    p_l   = q_cfg["percentile_left"]
    p_mid = q_cfg["percentile_mid"]
    p_r   = q_cfg["percentile_right"]
    calib_n = q_cfg["calibration_samples"]

    p2auc = {}

    def _eval(p):
        if p in p2auc:
            return p2auc[p]
        m = calibrate(qdq_model, val_loader, calib_n, device,
                      method="histogram", percentile=p)
        enable_all(m)
        metrics = evaluate(m, val_loader, device)
        auc = avg_auc(metrics)
        p2auc[p] = auc
        logger.info("  percentile=%.4f  avg_auc=%.4f", p, auc)
        return auc

    for _ in range(8):
        auc_l   = _eval(p_l)
        auc_mid = _eval(p_mid)
        auc_r   = _eval(p_r)
        if auc_l >= auc_mid and auc_l >= auc_r:
            p_r   = p_mid
            p_mid = (p_l + p_mid) / 2
        elif auc_r >= auc_mid and auc_r >= auc_l:
            p_l   = p_mid
            p_mid = (p_mid + p_r) / 2
        else:
            p_l = (p_l + p_mid) / 2
            p_r = (p_mid + p_r) / 2

    best_p   = max(p2auc, key=p2auc.get)
    best_auc = p2auc[best_p]
    return best_p, best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rec QDQ INT8 Quantization")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
