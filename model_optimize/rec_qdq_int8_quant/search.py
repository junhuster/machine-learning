"""
search.py — QDQ 量化层搜索模块（三阶段贪心策略）

逻辑与 rec_pytorch_int8_quant/search.py 一致，
差异在于量化控制使用 quantize.py 中的 QDQ 接口：
  - disable_with_keywords / only_disable_with_keywords
  - 通过 TensorQuantizer 精细控制哪些层量化

Phase 1: 单层扫描  — 逐层单独量化，记录 AUC 降幅
Phase 2: 分组验证  — 组内同时量化，检测累积误差
Phase 3: 全量验证  — 所有候选层同时量化，最终确认
"""

import copy
import logging
from evaluate import evaluate, avg_auc
from quantize import (only_disable_with_keywords, enable_all,
                      get_quantizer_names)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  内部辅助
# ------------------------------------------------------------------ #

def _eval_with_disabled(qdq_model, keywords_to_disable, val_loader, device):
    """
    关闭指定关键词对应的量化器，评估 AUC。

    Args:
        qdq_model:           已 calibrated 的 QDQ 模型
        keywords_to_disable: list[str]，要关闭量化的层名关键词
        val_loader:          验证集 DataLoader
        device:              推理设备

    Returns:
        (avg_auc_value, metrics_dict)
    """
    # 先全开，再按黑名单关闭
    only_disable_with_keywords(qdq_model, keywords_to_disable)
    metrics = evaluate(qdq_model, val_loader, device)
    # 恢复全开
    enable_all(qdq_model)
    return avg_auc(metrics), metrics


def _group_layers(layer_names):
    """按名称前缀分组（取前两段），组内同时量化检验累积误差。"""
    groups = {}
    for name in layer_names:
        parts = name.split(".")
        key   = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        groups.setdefault(key, []).append(name)
    return groups


# ------------------------------------------------------------------ #
#  Phase 1 — 单层扫描
# ------------------------------------------------------------------ #

def phase1_single_layer_scan(qdq_model, all_layers, baseline_auc,
                              val_loader, device, cfg):
    """
    逐层单独量化（其余全部关闭），记录 AUC 降幅。

    Returns:
        safe_layers, sensitive_layers, deltas
    """
    threshold = cfg["quantize"]["single_layer_auc_drop_threshold"]
    logger.info("=== Phase 1: Single-layer scan (%d layers) ===", len(all_layers))

    safe_layers, sensitive_layers, deltas = [], [], {}

    for i, layer in enumerate(all_layers):
        # 只量化当前层，其余全部排除
        exclude = [l for l in all_layers if l != layer]
        auc, metrics = _eval_with_disabled(qdq_model, exclude, val_loader, device)
        delta  = baseline_auc - auc
        deltas[layer] = delta
        status = "SAFE" if delta < threshold else "SENSITIVE"
        logger.info(
            "  [%2d/%2d] %-60s | delta=%.4f | ctr=%.4f cvr=%.4f | %s",
            i + 1, len(all_layers), layer,
            delta, metrics["ctr_auc"], metrics["cvr_auc"], status,
        )
        (safe_layers if delta < threshold else sensitive_layers).append(layer)

    logger.info("Phase 1 done: safe=%d  sensitive=%d",
                len(safe_layers), len(sensitive_layers))
    return safe_layers, sensitive_layers, deltas


# ------------------------------------------------------------------ #
#  Phase 2 — 分组验证
# ------------------------------------------------------------------ #

def phase2_group_validation(qdq_model, safe_layers, all_layers, baseline_auc,
                             deltas, val_loader, device, cfg):
    """
    将安全层按组同时量化，检测累积误差，超阈值则逐层剔除。
    """
    threshold = cfg["quantize"]["group_auc_drop_threshold"]
    groups    = _group_layers(safe_layers)
    logger.info("=== Phase 2: Group validation (%d groups) ===", len(groups))

    validated = []
    for key, members in groups.items():
        exclude = [l for l in all_layers if l not in members]
        auc, metrics = _eval_with_disabled(qdq_model, exclude, val_loader, device)
        delta = baseline_auc - auc
        logger.info("  Group %-40s  layers=%d | delta=%.4f | ctr=%.4f cvr=%.4f",
                    key, len(members), delta,
                    metrics["ctr_auc"], metrics["cvr_auc"])

        if delta < threshold:
            validated.extend(members)
            logger.info("    >> Accepted.")
        else:
            logger.info("    >> Exceeds threshold, pruning...")
            survivors = list(members)
            for rm in sorted(members, key=lambda l: deltas.get(l, 0), reverse=True):
                survivors.remove(rm)
                if not survivors:
                    break
                exc2 = [l for l in all_layers if l not in survivors]
                auc2, _ = _eval_with_disabled(qdq_model, exc2, val_loader, device)
                if baseline_auc - auc2 < threshold:
                    logger.info("    >> Accepted after removing %d layer(s).",
                                len(members) - len(survivors))
                    break
            validated.extend(survivors)

    logger.info("Phase 2 done: validated=%d", len(validated))
    return validated


# ------------------------------------------------------------------ #
#  Phase 3 — 全量验证
# ------------------------------------------------------------------ #

def phase3_final_validation(qdq_model, candidates, all_layers, baseline_auc,
                             deltas, val_loader, device, cfg):
    """
    所有候选层同时量化，做最终验证，超阈值逐层剔除。
    """
    threshold = cfg["quantize"]["final_auc_drop_threshold"]
    logger.info("=== Phase 3: Final validation (%d layers) ===", len(candidates))

    exclude = [l for l in all_layers if l not in candidates]
    auc, metrics = _eval_with_disabled(qdq_model, exclude, val_loader, device)
    delta = baseline_auc - auc
    logger.info("Full quantization: delta=%.4f | ctr=%.4f | cvr=%.4f",
                delta, metrics["ctr_auc"], metrics["cvr_auc"])

    if delta < threshold:
        logger.info("Phase 3 passed with all %d layers.", len(candidates))
        return candidates, auc

    logger.info("Exceeds threshold, pruning...")
    survivors = list(candidates)
    for rm in sorted(candidates, key=lambda l: deltas.get(l, 0), reverse=True):
        survivors.remove(rm)
        logger.info("  Removed: %s", rm)
        if not survivors:
            logger.warning("All layers removed.")
            return [], baseline_auc
        exc2 = [l for l in all_layers if l not in survivors]
        auc2, m2 = _eval_with_disabled(qdq_model, exc2, val_loader, device)
        delta2 = baseline_auc - auc2
        logger.info("  Remaining %d | delta=%.4f | ctr=%.4f cvr=%.4f",
                    len(survivors), delta2, m2["ctr_auc"], m2["cvr_auc"])
        if delta2 < threshold:
            logger.info("Phase 3 passed after pruning %d layer(s).",
                        len(candidates) - len(survivors))
            return survivors, auc2

    logger.warning("Could not meet threshold; no quantization applied.")
    return [], baseline_auc


# ------------------------------------------------------------------ #
#  主入口
# ------------------------------------------------------------------ #

def search_quantization_plan(qdq_model, val_loader, baseline_metrics,
                              device, cfg):
    """
    三阶段量化搜索，返回最终量化方案。

    Args:
        qdq_model:         已 calibrated 的 QDQ 模型
        val_loader:        验证集 DataLoader
        baseline_metrics:  FP32 baseline 的评估结果
        device:            推理设备
        cfg:               全局配置

    Returns:
        plan (dict): final_layers, final_auc, baseline_auc, auc_drop,
                     sensitive_layers, deltas
    """
    baseline_auc = avg_auc(baseline_metrics)
    all_layers   = get_quantizer_names(qdq_model)

    logger.info("Baseline avg_auc=%.4f  quantizer count=%d",
                baseline_auc, len(all_layers))

    safe, sensitive, deltas = phase1_single_layer_scan(
        qdq_model, all_layers, baseline_auc, val_loader, device, cfg)

    if not safe:
        logger.warning("No safe layers — no quantization applied.")
        return _plan([], baseline_auc, baseline_auc, sensitive, deltas)

    validated = phase2_group_validation(
        qdq_model, safe, all_layers, baseline_auc,
        deltas, val_loader, device, cfg)

    final_layers, final_auc = phase3_final_validation(
        qdq_model, validated, all_layers, baseline_auc,
        deltas, val_loader, device, cfg)

    plan = _plan(final_layers, final_auc, baseline_auc, sensitive, deltas)
    logger.info("=== Search complete: baseline=%.4f  final=%.4f  drop=%.4f  "
                "quantized=%d  sensitive=%d ===",
                baseline_auc, final_auc, plan["auc_drop"],
                len(final_layers), len(sensitive))
    return plan


def _plan(final_layers, final_auc, baseline_auc, sensitive, deltas):
    return {
        "final_layers":     final_layers,
        "final_auc":        final_auc,
        "baseline_auc":     baseline_auc,
        "auc_drop":         round(baseline_auc - final_auc, 6),
        "sensitive_layers": sensitive,
        "deltas":           {k: round(v, 6) for k, v in deltas.items()},
    }
