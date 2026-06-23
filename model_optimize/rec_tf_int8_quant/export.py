"""
export.py — SavedModel → ONNX 转换模块
同时建立 TF 层名 → ONNX 节点名的映射，供量化搜索使用。
"""

import logging
import os
import subprocess
import json
import onnx

logger = logging.getLogger(__name__)


def export_to_onnx(saved_model_dir, onnx_path, opset=13):
    """
    调用 tf2onnx 将 SavedModel 转换为 ONNX 格式。

    Args:
        saved_model_dir: SavedModel 目录
        onnx_path:       输出 ONNX 文件路径
        opset:           ONNX opset 版本
    """
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output",      onnx_path,
        "--opset",       str(opset),
    ]
    logger.info("Exporting SavedModel to ONNX: %s", onnx_path)
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("tf2onnx failed:\n%s", result.stderr)
        raise RuntimeError(f"tf2onnx conversion failed: {result.stderr}")

    logger.info("ONNX export complete: %s", onnx_path)


def get_onnx_matmul_nodes(onnx_path):
    """
    读取 ONNX 模型，返回所有 MatMul / Gemm 节点名列表。
    这些节点对应 Dense 层的权重乘法，是 INT8 量化的主要目标。

    Returns:
        list[str]: ONNX 节点名列表
    """
    model = onnx.load(onnx_path)
    nodes = []
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            nodes.append(node.name)
    logger.info("Found %d MatMul/Gemm nodes in ONNX model.", len(nodes))
    logger.debug("Nodes: %s", nodes)
    return nodes


def get_onnx_input_names(onnx_path):
    """返回 ONNX 模型的输入名列表。"""
    model = onnx.load(onnx_path)
    return [inp.name for inp in model.graph.input]


def save_node_mapping(node_names, mapping_path):
    """将节点名列表保存为 JSON，方便后续复用。"""
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(node_names, f, indent=2)
    logger.info("Node mapping saved to: %s", mapping_path)


def load_node_mapping(mapping_path):
    """从 JSON 加载节点名列表。"""
    with open(mapping_path, "r") as f:
        return json.load(f)
