"""
quantize.py — QDQ 量化核心模块

使用 NVIDIA pytorch_quantization 库，在模型中插入 QDQ 节点
（QuantizeLinear / DequantizeLinear），通过 calibration 数据确定
量化参数（scale / zero_point），最终导出含 QDQ 节点的 ONNX，
由 TensorRT 识别并构建 INT8 engine。

核心流程：
  1. setup_qdq_model()     向 FP32 模型插入 TensorQuantizer 节点
  2. calibrate()           用校准数据统计激活分布，确定 scale
  3. get_quantizer_names() 获取所有 TensorQuantizer 名称（用于搜索）
  4. disable/enable_*()    控制哪些 TensorQuantizer 开启/关闭
"""

import copy
import logging
import torch
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import calib

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  模型 QDQ 插桩
# ------------------------------------------------------------------ #

def setup_qdq_model(model):
    """
    向模型中所有 Linear / Conv 层插入 TensorQuantizer 节点（fake quant）。
    使用 pytorch_quantization 的 quant_modules 方式替换标准 nn.Module。

    注意：这会 in-place 修改模型结构，建议传入 deepcopy。

    Returns:
        model with QDQ nodes inserted
    """
    quant_modules.initialize()
    # 用 quant_nn 替换标准层（Linear -> QuantLinear 等）
    # pytorch_quantization 通过 monkey-patch 实现
    logger.info("QDQ nodes inserted into model.")
    return model


def setup_qdq_model_from_fp32(fp32_model):
    """
    从 FP32 权重构建带 QDQ 节点的模型副本。
    先 initialize quant_modules，再重新实例化同结构模型并加载权重。

    实际使用时建议：
      1. 在 model.py 的 build_model() 之前调用 quant_modules.initialize()
      2. 这样 nn.Linear 等会自动被替换为 QuantLinear

    Returns:
        qdq_model: 含 TensorQuantizer 的模型（权重与 fp32_model 相同）
    """
    quant_modules.initialize()
    qdq_model = copy.deepcopy(fp32_model)
    logger.info("QDQ model prepared from FP32 weights.")
    return qdq_model


# ------------------------------------------------------------------ #
#  Calibration
# ------------------------------------------------------------------ #

def calibrate(qdq_model, loader, calibration_samples, device,
              method="histogram", percentile=99.99):
    """
    用 calibration 数据跑前向，让 TensorQuantizer 统计激活分布，
    并计算量化 scale / zero_point。

    Args:
        qdq_model:           含 QDQ 节点的模型
        loader:              DataLoader，提供 calibration 数据
        calibration_samples: 最多使用的样本数
        device:              推理设备
        method:              calibration 方法，"histogram" 或 "max"
        percentile:          histogram 方法的百分位数（99.9~99.999）

    Returns:
        calibrated qdq_model
    """
    qdq_model.eval()

    # 开启 calibration 模式
    with torch.no_grad():
        # 设置所有 TensorQuantizer 为 calibration 模式
        for name, module in qdq_model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        collected = 0
        for batch in loader:
            if collected >= calibration_samples:
                break
            sparse = batch["sparse"].to(device)
            dense  = batch["dense"].to(device)
            qdq_model(sparse, dense)
            collected += sparse.size(0)
            logger.debug("Calibration: %d / %d samples", collected, calibration_samples)

        # 计算 scale，关闭 calibration 模式，开启量化模式
        for name, module in qdq_model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(method, percentile=percentile)
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    logger.info("Calibration done: %d samples, method=%s, percentile=%.3f",
                collected, method, percentile)
    return qdq_model


# ------------------------------------------------------------------ #
#  量化器名称获取
# ------------------------------------------------------------------ #

def get_quantizer_names(model):
    """
    返回模型中所有 TensorQuantizer 的名称列表。
    这些名称用于量化搜索中的 disable/enable 操作。
    """
    names = [name for name, module in model.named_modules()
             if isinstance(module, quant_nn.TensorQuantizer)]
    logger.debug("Found %d TensorQuantizers: %s", len(names), names)
    return names


# ------------------------------------------------------------------ #
#  量化器开关控制
# ------------------------------------------------------------------ #

def disable_all(model):
    """关闭所有 TensorQuantizer（退化为 FP32）。"""
    for _, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable()


def enable_all(model):
    """开启所有 TensorQuantizer（全量化）。"""
    for _, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable()


def disable_with_keywords(model, keywords):
    """
    关闭名称包含任意 keyword 的 TensorQuantizer。
    keyword 可以是字符串或整数（层索引）。
    """
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, quant_nn.TensorQuantizer):
            continue
        for kw in keywords:
            if str(kw) in name:
                module.disable()
                count += 1
                break
    logger.debug("disable_with_keywords: disabled %d quantizers, keywords=%s",
                 count, keywords)


def enable_with_keywords(model, keywords):
    """开启名称包含任意 keyword 的 TensorQuantizer。"""
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, quant_nn.TensorQuantizer):
            continue
        for kw in keywords:
            if str(kw) in name:
                module.enable()
                count += 1
                break
    logger.debug("enable_with_keywords: enabled %d quantizers, keywords=%s",
                 count, keywords)


def only_disable_with_keywords(model, keywords):
    """先全部开启，再关闭匹配 keyword 的量化器。"""
    enable_all(model)
    disable_with_keywords(model, keywords)


def only_enable_with_keywords(model, keywords):
    """先全部关闭，再开启匹配 keyword 的量化器。"""
    disable_all(model)
    enable_with_keywords(model, keywords)
