"""
quantize.py — 量化工具模块
封装 PyTorch PTQ（训练后静态量化）的核心操作：
  - 准备模型（插入 Observer）
  - Calibration
  - 转换为 INT8
  - 支持指定层的量化配置
"""

import logging
import copy
import torch
import torch.quantization as tq

logger = logging.getLogger(__name__)


# 默认量化配置：per-tensor、fbgemm 后端
DEFAULT_QCONFIG = tq.get_default_qconfig("fbgemm")


def get_quantizable_layer_names(model):
    """
    返回模型中所有 nn.Linear 层的名称列表。
    这些层是 INT8 量化的主要候选对象。
    Embedding 和 BN 层通常不做 INT8 量化。
    """
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names.append(name)
    logger.debug("Quantizable Linear layers (%d): %s", len(names), names)
    return names


def build_quantized_model(fp32_model, layer_names_to_quantize):
    """
    给定要量化的层名称列表，返回 INT8 量化后的模型副本。

    Args:
        fp32_model: 原始 FP32 模型（CPU）
        layer_names_to_quantize: list[str]，要量化的层名

    Returns:
        量化后的模型（CPU）
    """
    model = copy.deepcopy(fp32_model).cpu().eval()

    # 构建 qconfig_spec：只对指定层设置量化配置
    qconfig_spec = {}
    for name, module in model.named_modules():
        if name in layer_names_to_quantize and isinstance(module, torch.nn.Linear):
            qconfig_spec[name] = DEFAULT_QCONFIG

    if not qconfig_spec:
        logger.warning("No layers selected for quantization, returning FP32 model.")
        return model

    # 逐层设置 qconfig
    for name, module in model.named_modules():
        if name in qconfig_spec:
            module.qconfig = qconfig_spec[name]
        elif hasattr(module, "qconfig"):
            module.qconfig = None

    tq.prepare(model, inplace=True)
    return model


def calibrate(model, loader, num_samples, device="cpu"):
    """
    用 calibration 数据跑前向，让 Observer 收集激活分布统计。

    Args:
        model: prepare 后的模型
        loader: DataLoader
        num_samples: 最多使用的样本数
        device: 推理设备（PTQ calibration 在 CPU 上进行）
    """
    model.eval()
    collected = 0
    with torch.no_grad():
        for batch in loader:
            if collected >= num_samples:
                break
            sparse = batch["sparse"].to(device)
            dense  = batch["dense"].to(device)
            model(sparse, dense)
            collected += sparse.size(0)
    logger.debug("Calibration done: %d samples used.", collected)


def convert_to_int8(prepared_model):
    """将 prepare + calibrate 后的模型转换为 INT8。"""
    model = copy.deepcopy(prepared_model)
    tq.convert(model, inplace=True)
    return model


def quantize_model(fp32_model, layer_names, val_loader, calibration_samples):
    """
    完整量化流程：prepare → calibrate → convert。

    Args:
        fp32_model: FP32 模型（CPU eval 模式）
        layer_names: 要量化的层名列表
        val_loader: 用于 calibration 的数据加载器
        calibration_samples: calibration 样本数

    Returns:
        int8_model: 量化后的模型
    """
    logger.debug("Quantizing %d layers: %s", len(layer_names), layer_names)
    prepared = build_quantized_model(fp32_model, layer_names)
    calibrate(prepared, val_loader, calibration_samples, device="cpu")
    int8_model = convert_to_int8(prepared)
    return int8_model
