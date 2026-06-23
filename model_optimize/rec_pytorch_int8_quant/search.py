"""
search.py — 量化层搜索模块

搜索策略（三阶段）：
  Phase 1 — 单层筛选（贪心）
    逐层单独开量化，测 AUC 下降，超阈值列为敏感层。

  Phase 2 — 分组验证
    将安全层按网络位置分组，组内同时开量化，验证累积误差。
    若某组导致 AUC 降幅超阈值，则从组中逐层剔除最敏感的层。

  Phase 3 — 全量验证
    把通过分组验证的层一次性全部开量化，做最终确认。
    若整体 AUC 降幅超阈值，按 Phase-1 的 delta 从大到小逐层剔除，
    直到满足约束。
"""

import logging
import copy
import json
import torch
from evaluate import evaluate, avg_auc
from quantize import quantize_model, get_quantizable_layer_names

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  内部辅助
# ------------------------------------------------------------------ #

def _eval_int8(fp32_model, layer_names, val_loader, calib_samples, device):
    """量化指定层并在 val_loader 上评估，返回 avg_auc。"""
    if not layer_names:
        return None
    int8_model = quantize_model(
        copy.deepcopy(fp32_model).cpu().eval(),
        layer_names, val_loader, calib_samples
    )
    int8_model.eval()
    metrics = evaluate(int8_model, val_loader, device="cpu")
    return avg_auc(metrics), metrics


def _group_layers(all_layer_names):
    """
    按网络组件对层名分组。
    规则：层名前缀相同（如 experts.0、towers.1）归为一组，
    其余每层单独一组。
    """
    groups = {}
    for name in all_layer_names:
        # 取前两段作为组 key（如 "experts.0" / "gates.1" / "towers.0"）
        parts = name.split(".")
        group_key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        groups.setdefault(group_key, []).append(name)
    return groups


# ------------------------------------------------------------------ #
#  Phase 1 — 单层筛选
# ------------------------------------------------------------------ #

def phase1_single_layer_scan(fp32_model, all_layers, baseline_auc,
                              val_loader, cfg):
    """
    逐层单独量化，记录 AUC 下降幅度。

    Returns:
        safe_layers:      list[str]  AUC 降幅 < 阈值
        sensitive_layers: list[str]  AUC 降幅 >= 阈值
        deltas:           dict[str -> float]  每层的 AUC 降幅
    """
    q_cfg = cfg["quantize"]
    threshold = q_cfg["single_layer_auc_drop_threshold"]
    calib_samples = q_cfg["calibration_samples"]

    logger.info("=== Phase 1: Single-layer scan (%d layers) ===", len(all_layers))

    safe_layers, sensitive_layers, deltas = [], [], {}

    for i, layer in enumerate(all_layers):
        result = _eval_int8(fp32_model, [layer], val_loader, calib_samples, device="cpu")
        if result is None:
            continue
        auc, metrics = result
        delta = baseline_auc - auc
        deltas[layer] = delta
        status = "SAFE" if delta < threshold else "SENSITIVE"
        logger.info(
            "  [%2d/%2d] %-50s | delta=%.4f | ctr=%.4f cvr=%.4f | %s",
            i + 1, len(all_layers), layer,
            delta, metrics["ctr_auc"], metrics["cvr_auc"], status,
        )
        if delta < threshold:
            safe_layers.append(layer)
        else:
            sensitive_layers.append(layer)

    logger.info("Phase 1 done: safe=%d  sensitive=%d", len(safe_layers), len(sensitive_layers))
    return safe_layers, sensitive_layers, deltas


# ------------------------------------------------------------------ #
#  Phase 2 — 分组验证
# ------------------------------------------------------------------ #

def phase2_group_validation(fp32_model, safe_layers, baseline_auc,
                             deltas, val_loader, cfg):
    """
    将安全层按组同时量化，检测累积误差。
    若某组超阈值，按 delta 从大到小逐层剔除，直到组内 AUC 达标。

    Returns:
        validated_layers: list[str]  通过分组验证的层
    """
    q_cfg = cfg["quantize"]
    threshold   = q_cfg["group_auc_drop_threshold"]
    calib_samples = q_cfg["calibration_samples"]

    groups = _group_layers(safe_layers)
    logger.info("=== Phase 2: Group validation (%d groups) ===", len(groups))

    validated_layers = []
    for group_key, group_members in groups.items():
        result = _eval_int8(fp32_model, group_members, val_loader, calib_samples, "cpu")
        if result is None:
            continue
        auc, metrics = result
        delta = baseline_auc - auc
        logger.info(
            "  Group %-30s  layers=%d | delta=%.4f | ctr=%.4f cvr=%.4f",
            group_key, len(group_members),
            delta, metrics["ctr_auc"], metrics["cvr_auc"],
        )

        if delta < threshold:
            validated_layers.extend(group_members)
            logger.info("    >> Group accepted.")
        else:
            logger.info("    >> Group exceeds threshold, pruning...")
            # 按 delta 从大到小排序，逐层剔除
            sorted_members = sorted(group_members, key=lambda l: deltas.get(l, 0), reverse=True)
            survivors = list(group_members)
            for rm_layer in sorted_members:
                survivors.remove(rm_layer)
                logger.info("      Removed layer: %s", rm_layer)
                if not survivors:
                    break
                result2 = _eval_int8(fp32_model, survivors, val_loader, calib_samples, "cpu")
                if result2 is None:
                    break
                auc2, _ = result2
                if baseline_auc - auc2 < threshold:
                    logger.info("      Group now accepted after removing %d layer(s).",
                                len(group_members) - len(survivors))
                    break
            validated_layers.extend(survivors)

    logger.info("Phase 2 done: validated_layers=%d", len(validated_layers))
    return validated_layers


# ------------------------------------------------------------------ #
#  Phase 3 — 全量验证
# ------------------------------------------------------------------ #

def phase3_final_validation(fp32_model, candidate_layers, baseline_auc,
                             deltas, val_loader, cfg):
    """
    把所有候选层同时量化做最终验证。
    若 AUC 下降超阈值，按 delta 逐层剔除直到满足约束。

    Returns:
        final_layers: list[str]  最终确认量化的层
        final_auc:    float
    """
    q_cfg = cfg["quantize"]
    threshold     = q_cfg["final_auc_drop_threshold"]
    calib_samples = q_cfg["calibration_samples"]

    logger.info("=== Phase 3: Final validation (%d layers) ===", len(candidate_layers))

    result = _eval_int8(fp32_model, candidate_layers, val_loader, calib_samples, "cpu")
    if result is None:
        return [], baseline_auc

    auc, metrics = result
    delta = baseline_auc - auc
    logger.info(
        "Full quantization: delta=%.4f | ctr=%.4f | cvr=%.4f",
        delta, metrics["ctr_auc"], metrics["cvr_auc"],
    )

    if delta < threshold:
        logger.info("Phase 3 passed with all %d layers.", len(candidate_layers))
        return candidate_layers, auc

    # 超阈值：按 delta 从大到小逐层剔除
    logger.info("Exceeds threshold, pruning layers...")
    sorted_candidates = sorted(candidate_layers, key=lambda l: deltas.get(l, 0), reverse=True)
    survivors = list(candidate_layers)
    for rm_layer in sorted_candidates:
        survivors.remove(rm_layer)
        logger.info("  Removed: %s", rm_layer)
        if not survivors:
            logger.warning("All layers removed — no quantization applied.")
            return [], baseline_auc
        result2 = _eval_int8(fp32_model, survivors, val_loader, calib_samples, "cpu")
        if result2 is None:
            break
        auc2, m2 = result2
        delta2 = baseline_auc - auc2
        logger.info(
            "  Remaining %d layers | delta=%.4f | ctr=%.4f cvr=%.4f",
            len(survivors), delta2, m2["ctr_auc"], m2["cvr_auc"],
        )
        if delta2 < threshold:
            logger.info("Phase 3 passed after pruning %d layer(s).",
                        len(candidate_layers) - len(survivors))
            return survivors, auc2

    logger.warning("Could not meet threshold; returning empty quantization plan.")
    return [], baseline_auc


# ------------------------------------------------------------------ #
#  主入口
# ------------------------------------------------------------------ #

def search_quantization_plan(fp32_model, val_loader, baseline_metrics, cfg):
    """
    完整三阶段量化搜索，返回最终量化方案。

    Returns:
        plan (dict): {
            "final_layers": list[str],
            "final_auc": float,
            "baseline_auc": float,
            "auc_drop": float,
            "sensitive_layers": list[str],
            "deltas": dict[str, float],
        }
    """
    baseline_auc = avg_auc(baseline_metrics)
    logger.info("Baseline avg_auc=%.4f (ctr=%.4f, cvr=%.4f)",
                baseline_auc, baseline_metrics["ctr_auc"], baseline_metrics["cvr_auc"])

    all_layers = get_quantizable_layer_names(fp32_model)
    logger.info("Total quantizable layers: %d", len(all_layers))

    # Phase 1
    safe_layers, sensitive_layers, deltas = phase1_single_layer_scan(
        fp32_model, all_layers, baseline_auc, val_loader, cfg)

    if not safe_layers:
        logger.warning("No safe layers found — no quantization will be applied.")
        return {
            "final_layers": [],
            "final_auc": baseline_auc,
            "baseline_auc": baseline_auc,
            "auc_drop": 0.0,
            "sensitive_layers": sensitive_layers,
            "deltas": deltas,
        }

    # Phase 2
    validated_layers = phase2_group_validation(
        fp32_model, safe_layers, baseline_auc, deltas, val_loader, cfg)

    # Phase 3
    final_layers, final_auc = phase3_final_validation(
        fp32_model, validated_layers, baseline_auc, deltas, val_loader, cfg)

    auc_drop = baseline_auc - final_auc
    logger.info("=== Search complete ===")
    logger.info("  Baseline AUC : %.4f", baseline_auc)
    logger.info("  Final AUC    : %.4f", final_auc)
    logger.info("  AUC drop     : %.4f", auc_drop)
    logger.info("  Quantized layers (%d): %s", len(final_layers), final_layers)
    logger.info("  Sensitive layers (%d): %s", len(sensitive_layers), sensitive_layers)

    return {
        "final_layers": final_layers,
        "final_auc": final_auc,
        "baseline_auc": baseline_auc,
        "auc_drop": auc_drop,
        "sensitive_layers": sensitive_layers,
        "deltas": {k: round(v, 6) for k, v in deltas.items()},
    }
