"""
search.py — 量化层搜索模块（三阶段贪心策略）

与 PyTorch 版本逻辑相同，差异在于：
  - 量化和推理通过 backend 适配层调用 quantize_onnx 或 quantize_trt
  - 层名使用 ONNX 的 MatMul/Gemm 节点名
  - "黑名单" = nodes_to_exclude（onnxruntime）或 layers_to_keep_fp32（TRT）
    两者语义相同：哪些节点不做 INT8 量化
"""

import logging
import json
import os
import tempfile

from evaluate import avg_auc
import quantize_onnx
import quantize_trt

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  后端适配层
# ------------------------------------------------------------------ #

def _eval_with_excluded(onnx_path, nodes_to_exclude, val_dataset,
                        input_names, cfg):
    """
    量化 ONNX 模型（排除指定节点），在 val_dataset 上评估，返回 avg_auc。

    Args:
        onnx_path:        FP32 ONNX 路径
        nodes_to_exclude: list[str]，不量化的节点（黑名单）
        val_dataset:      RecDataset
        input_names:      ONNX 输入名列表
        cfg:              全局配置

    Returns:
        (avg_auc_value, metrics_dict) 或 None（量化失败时）
    """
    q_cfg    = cfg["quantize"]
    backend  = q_cfg["backend"]
    batch_sz = cfg["data"]["batch_size"]
    calib_n  = q_cfg["calibration_samples"]

    try:
        if backend == "onnxruntime":
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                tmp_path = f.name
            quantize_onnx.quantize(
                onnx_path=onnx_path,
                output_path=tmp_path,
                nodes_to_exclude=nodes_to_exclude,
                val_dataset=val_dataset,
                input_names=input_names,
                calibration_samples=calib_n,
                batch_size=batch_sz,
            )
            session = quantize_onnx.load_session(tmp_path)
            from evaluate import evaluate_onnx
            metrics = evaluate_onnx(session, val_dataset, batch_sz)
            os.unlink(tmp_path)

        elif backend == "tensorrt":
            with tempfile.NamedTemporaryFile(suffix=".trt", delete=False) as f:
                tmp_path = f.name
            quantize_trt.build_engine(
                onnx_path=onnx_path,
                engine_path=tmp_path,
                layers_to_keep_fp32=nodes_to_exclude,
                val_dataset=val_dataset,
                calibration_samples=calib_n,
                cfg=cfg,
            )
            engine = quantize_trt.load_engine(tmp_path)
            runner = quantize_trt.TRTRunner(engine)
            from evaluate import evaluate_trt
            metrics = evaluate_trt(runner, val_dataset, batch_sz)
            os.unlink(tmp_path)

        else:
            raise ValueError(f"Unknown backend: {backend}")

        return avg_auc(metrics), metrics

    except Exception as e:
        logger.warning("Quantization eval failed (excluded=%s): %s",
                       nodes_to_exclude, e)
        return None


def _group_nodes(all_nodes):
    """
    按节点名前缀分组（取 '/' 分隔的前两段）。
    同组节点同时量化，检验累积误差。
    """
    groups = {}
    for name in all_nodes:
        parts = name.split("/")
        key = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        groups.setdefault(key, []).append(name)
    return groups


# ------------------------------------------------------------------ #
#  Phase 1 — 单节点扫描
# ------------------------------------------------------------------ #

def phase1_single_node_scan(onnx_path, all_nodes, baseline_auc,
                             val_dataset, input_names, cfg):
    """
    逐节点单独量化，记录 AUC 降幅。
    注意：这里"量化某节点"= 仅该节点量化（其余全部排除）。
    等价于：nodes_to_exclude = all_nodes - {当前节点}。

    Returns:
        safe_nodes:      list[str]
        sensitive_nodes: list[str]
        deltas:          dict[str -> float]
    """
    q_cfg     = cfg["quantize"]
    threshold = q_cfg["single_layer_auc_drop_threshold"]

    logger.info("=== Phase 1: Single-node scan (%d nodes) ===", len(all_nodes))

    safe_nodes, sensitive_nodes, deltas = [], [], {}

    for i, node in enumerate(all_nodes):
        # 只量化当前节点，其余都排除
        exclude = [n for n in all_nodes if n != node]
        result = _eval_with_excluded(onnx_path, exclude, val_dataset, input_names, cfg)
        if result is None:
            continue

        auc, metrics = result
        delta  = baseline_auc - auc
        deltas[node] = delta
        status = "SAFE" if delta < threshold else "SENSITIVE"

        logger.info(
            "  [%2d/%2d] %-60s | delta=%.4f | ctr=%.4f cvr=%.4f | %s",
            i + 1, len(all_nodes), node,
            delta, metrics["ctr_auc"], metrics["cvr_auc"], status,
        )

        if delta < threshold:
            safe_nodes.append(node)
        else:
            sensitive_nodes.append(node)

    logger.info("Phase 1 done: safe=%d  sensitive=%d",
                len(safe_nodes), len(sensitive_nodes))
    return safe_nodes, sensitive_nodes, deltas


# ------------------------------------------------------------------ #
#  Phase 2 — 分组验证
# ------------------------------------------------------------------ #

def phase2_group_validation(onnx_path, safe_nodes, all_nodes, baseline_auc,
                             deltas, val_dataset, input_names, cfg):
    """
    将安全节点按组同时量化（其余排除），检测累积误差。
    若某组超阈值，按 delta 从大到小逐节点剔除。

    Returns:
        validated_nodes: list[str]
    """
    q_cfg     = cfg["quantize"]
    threshold = q_cfg["group_auc_drop_threshold"]

    groups = _group_nodes(safe_nodes)
    logger.info("=== Phase 2: Group validation (%d groups) ===", len(groups))

    validated_nodes = []
    for group_key, group_members in groups.items():
        # 量化该组节点，其余全部排除
        exclude = [n for n in all_nodes if n not in group_members]
        result  = _eval_with_excluded(onnx_path, exclude, val_dataset, input_names, cfg)
        if result is None:
            continue

        auc, metrics = result
        delta = baseline_auc - auc
        logger.info(
            "  Group %-40s  nodes=%d | delta=%.4f | ctr=%.4f cvr=%.4f",
            group_key, len(group_members),
            delta, metrics["ctr_auc"], metrics["cvr_auc"],
        )

        if delta < threshold:
            validated_nodes.extend(group_members)
            logger.info("    >> Group accepted.")
        else:
            logger.info("    >> Group exceeds threshold, pruning...")
            sorted_members = sorted(group_members,
                                    key=lambda n: deltas.get(n, 0), reverse=True)
            survivors = list(group_members)
            for rm in sorted_members:
                survivors.remove(rm)
                logger.info("      Removed node: %s", rm)
                if not survivors:
                    break
                exclude2 = [n for n in all_nodes if n not in survivors]
                result2  = _eval_with_excluded(onnx_path, exclude2,
                                               val_dataset, input_names, cfg)
                if result2 is None:
                    break
                auc2, _ = result2
                if baseline_auc - auc2 < threshold:
                    logger.info("      Group accepted after removing %d node(s).",
                                len(group_members) - len(survivors))
                    break
            validated_nodes.extend(survivors)

    logger.info("Phase 2 done: validated_nodes=%d", len(validated_nodes))
    return validated_nodes


# ------------------------------------------------------------------ #
#  Phase 3 — 全量验证
# ------------------------------------------------------------------ #

def phase3_final_validation(onnx_path, candidate_nodes, all_nodes, baseline_auc,
                             deltas, val_dataset, input_names, cfg):
    """
    把所有候选节点同时量化（排除其余），做最终验证。
    若超阈值按 delta 逐节点剔除。

    Returns:
        final_nodes: list[str]
        final_auc:   float
    """
    q_cfg     = cfg["quantize"]
    threshold = q_cfg["final_auc_drop_threshold"]

    logger.info("=== Phase 3: Final validation (%d nodes) ===",
                len(candidate_nodes))

    exclude = [n for n in all_nodes if n not in candidate_nodes]
    result  = _eval_with_excluded(onnx_path, exclude, val_dataset, input_names, cfg)
    if result is None:
        return [], baseline_auc

    auc, metrics = result
    delta = baseline_auc - auc
    logger.info(
        "Full quantization: delta=%.4f | ctr=%.4f | cvr=%.4f",
        delta, metrics["ctr_auc"], metrics["cvr_auc"],
    )

    if delta < threshold:
        logger.info("Phase 3 passed with all %d nodes.", len(candidate_nodes))
        return candidate_nodes, auc

    logger.info("Exceeds threshold, pruning nodes...")
    sorted_candidates = sorted(candidate_nodes,
                               key=lambda n: deltas.get(n, 0), reverse=True)
    survivors = list(candidate_nodes)
    for rm in sorted_candidates:
        survivors.remove(rm)
        logger.info("  Removed: %s", rm)
        if not survivors:
            logger.warning("All nodes removed — no quantization applied.")
            return [], baseline_auc
        exclude2 = [n for n in all_nodes if n not in survivors]
        result2  = _eval_with_excluded(onnx_path, exclude2,
                                       val_dataset, input_names, cfg)
        if result2 is None:
            break
        auc2, m2 = result2
        delta2 = baseline_auc - auc2
        logger.info(
            "  Remaining %d nodes | delta=%.4f | ctr=%.4f cvr=%.4f",
            len(survivors), delta2, m2["ctr_auc"], m2["cvr_auc"],
        )
        if delta2 < threshold:
            logger.info("Phase 3 passed after pruning %d node(s).",
                        len(candidate_nodes) - len(survivors))
            return survivors, auc2

    logger.warning("Could not meet threshold; returning empty quantization plan.")
    return [], baseline_auc


# ------------------------------------------------------------------ #
#  主入口
# ------------------------------------------------------------------ #

def search_quantization_plan(onnx_path, all_nodes, val_dataset,
                              input_names, baseline_metrics, cfg):
    """
    完整三阶段量化搜索。

    Returns:
        plan (dict)
    """
    baseline_auc = avg_auc(baseline_metrics)
    logger.info("Baseline avg_auc=%.4f (ctr=%.4f, cvr=%.4f)",
                baseline_auc, baseline_metrics["ctr_auc"], baseline_metrics["cvr_auc"])
    logger.info("Total quantizable nodes: %d  backend: %s",
                len(all_nodes), cfg["quantize"]["backend"])

    # Phase 1
    safe_nodes, sensitive_nodes, deltas = phase1_single_node_scan(
        onnx_path, all_nodes, baseline_auc, val_dataset, input_names, cfg)

    if not safe_nodes:
        logger.warning("No safe nodes found — no quantization will be applied.")
        return _make_plan([], baseline_auc, baseline_auc, sensitive_nodes, deltas)

    # Phase 2
    validated_nodes = phase2_group_validation(
        onnx_path, safe_nodes, all_nodes, baseline_auc,
        deltas, val_dataset, input_names, cfg)

    # Phase 3
    final_nodes, final_auc = phase3_final_validation(
        onnx_path, validated_nodes, all_nodes, baseline_auc,
        deltas, val_dataset, input_names, cfg)

    plan = _make_plan(final_nodes, final_auc, baseline_auc, sensitive_nodes, deltas)

    logger.info("=== Search complete ===")
    logger.info("  Baseline AUC : %.4f", baseline_auc)
    logger.info("  Final AUC    : %.4f", final_auc)
    logger.info("  AUC drop     : %.4f", plan["auc_drop"])
    logger.info("  Quantized nodes (%d): %s", len(final_nodes), final_nodes)
    logger.info("  Sensitive nodes (%d): %s", len(sensitive_nodes), sensitive_nodes)
    return plan


def _make_plan(final_nodes, final_auc, baseline_auc, sensitive_nodes, deltas):
    return {
        "final_nodes":      final_nodes,
        "final_auc":        final_auc,
        "baseline_auc":     baseline_auc,
        "auc_drop":         round(baseline_auc - final_auc, 6),
        "sensitive_nodes":  sensitive_nodes,
        "deltas":           {k: round(v, 6) for k, v in deltas.items()},
    }
