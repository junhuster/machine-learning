"""
main.py — 主入口
串联：数据构建 → 模型训练 → 量化搜索 → 保存最终模型和报告
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

from logger_setup import init_logger

from data import build_dataloaders
from model import build_model
from train import train
from evaluate import evaluate, avg_auc
from search import search_quantization_plan
from quantize import quantize_model


# ------------------------------------------------------------------ #
#  工具函数
# ------------------------------------------------------------------ #

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- Step 1: 构建数据 ----
    logger.info("Step 1: Building data loaders...")
    train_loader, val_loader = build_dataloaders(cfg)

    # ---- Step 2: 训练 FP32 模型 ----
    logger.info("Step 2: Training FP32 model...")
    model = build_model(cfg, device)
    best_metrics = train(model, train_loader, val_loader, cfg, device)
    logger.info("FP32 best metrics: ctr_auc=%.4f  cvr_auc=%.4f  avg=%.4f",
                best_metrics["ctr_auc"], best_metrics["cvr_auc"], avg_auc(best_metrics))

    # 加载最优检查点，用于量化搜索
    # PTQ 量化只支持 CPU，需将模型迁移到 CPU
    model.load_state_dict(torch.load(cfg["train"]["checkpoint_path"], map_location=device))
    fp32_model_cpu = copy.deepcopy(model).cpu().eval()
    logger.info("Model copied to CPU for PTQ quantization search")

    # ---- Step 3: 量化层搜索 ----
    logger.info("Step 3: Searching quantization plan...")
    plan = search_quantization_plan(fp32_model_cpu, val_loader, best_metrics, cfg)

    # ---- Step 4: 构建并保存最终 INT8 模型 ----
    q_cfg = cfg["quantize"]
    os.makedirs(os.path.dirname(q_cfg["quantized_model_path"]), exist_ok=True)

    if plan["final_layers"]:
        logger.info("Step 4: Building final quantized model...")
        final_int8 = quantize_model(
            fp32_model_cpu,
            plan["final_layers"],
            val_loader,
            q_cfg["calibration_samples"],
        )
        torch.save(final_int8, q_cfg["quantized_model_path"])
        logger.info("Quantized model saved to: %s", q_cfg["quantized_model_path"])
    else:
        logger.warning("Step 4: No layers quantized — skipping model save.")

    # ---- Step 5: 保存报告 ----
    report = {
        "baseline_ctr_auc": best_metrics["ctr_auc"],
        "baseline_cvr_auc": best_metrics["cvr_auc"],
        "baseline_avg_auc": avg_auc(best_metrics),
        "final_avg_auc":    plan["final_auc"],
        "auc_drop":         plan["auc_drop"],
        "num_quantized_layers": len(plan["final_layers"]),
        "num_sensitive_layers": len(plan["sensitive_layers"]),
        "quantized_layers": plan["final_layers"],
        "sensitive_layers": plan["sensitive_layers"],
        "layer_deltas":     plan["deltas"],
    }
    with open(q_cfg["report_path"], "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to: %s", q_cfg["report_path"])

    # ---- 最终摘要 ----
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  FP32  avg_auc : %.4f", avg_auc(best_metrics))
    logger.info("  INT8  avg_auc : %.4f", plan["final_auc"])
    logger.info("  AUC drop      : %.4f", plan["auc_drop"])
    logger.info("  Quantized layers : %d / %d",
                len(plan["final_layers"]),
                len(plan["final_layers"]) + len(plan["sensitive_layers"]))
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rec INT8 Quantization")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
