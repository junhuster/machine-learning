"""
torch2onnx.py — PyTorch (QDQ) → ONNX 导出工具脚本

支持两种导出模式：
  --mode fp32  : 导出标准 FP32 ONNX（无量化节点），用于对比基线
  --mode qdq   : 导出含 QDQ 节点的 ONNX（QuantizeLinear/DequantizeLinear），
                 供 TensorRT 直接读取量化参数，无需再做 calibration

用法：
  python torch2onnx.py --config config.yaml --mode qdq
  python torch2onnx.py --config config.yaml --mode fp32
"""

import argparse
import logging
import os
import torch
import yaml
from pytorch_quantization import quant_modules

from logger_setup import init_logger
from data import build_dataloaders
from model import build_model
from quantize import setup_qdq_model_from_fp32, calibrate

logger = logging.getLogger(__name__)


def export_onnx(model, onnx_path, cfg, device):
    """
    将模型导出为 ONNX 格式。
    模型中若含 TensorQuantizer，导出时会自动转换为
    QuantizeLinear / DequantizeLinear 标准算子。

    Args:
        model:     PyTorch 模型（FP32 或含 QDQ 节点）
        onnx_path: 输出路径
        cfg:       全局配置
        device:    当前设备
    """
    d = cfg["data"]
    opset = cfg["export"]["opset"]
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    model.eval()
    dummy_sparse = torch.zeros(1, d["num_sparse_fields"], dtype=torch.long).to(device)
    dummy_dense  = torch.zeros(1, d["num_dense_features"], dtype=torch.float32).to(device)

    # pytorch_quantization 要求用 quant_nn.TensorQuantizer.use_fb_fake_quant
    # 保证导出的 ONNX 含标准 QDQ 算子而非 fake_quant 私有节点
    from pytorch_quantization.nn import TensorQuantizer
    TensorQuantizer.use_fb_fake_quant = True

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_sparse, dummy_dense),
            onnx_path,
            opset_version=opset,
            input_names=["sparse", "dense"],
            output_names=["pred_ctr", "pred_cvr"],
            dynamic_axes={
                "sparse":   {0: "batch_size"},
                "dense":    {0: "batch_size"},
                "pred_ctr": {0: "batch_size"},
                "pred_cvr": {0: "batch_size"},
            },
            do_constant_folding=True,
        )

    TensorQuantizer.use_fb_fake_quant = False
    logger.info("ONNX exported to: %s", onnx_path)


def main():
    parser = argparse.ArgumentParser(description="PyTorch → ONNX export (FP32 or QDQ)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode",   choices=["fp32", "qdq"], default="qdq",
                        help="fp32: 无量化节点基线; qdq: 含QDQ节点供TRT INT8")
    parser.add_argument("--output", default=None,
                        help="覆盖 config 中的输出路径（可选）")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    init_logger(cfg.get("logging", {}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Export mode=%s  device=%s", args.mode, device)

    # 确定输出路径
    if args.output:
        onnx_path = args.output
    elif args.mode == "fp32":
        onnx_path = cfg["export"]["onnx_fp32_path"]
    else:
        onnx_path = cfg["export"]["onnx_qdq_path"]

    # 加载 FP32 权重
    model = build_model(cfg, device)
    model.load_state_dict(
        torch.load(cfg["train"]["checkpoint_path"], map_location=device)
    )
    logger.info("FP32 weights loaded from: %s", cfg["train"]["checkpoint_path"])

    if args.mode == "fp32":
        # 直接导出 FP32 ONNX
        logger.info("Exporting FP32 ONNX...")
        export_onnx(model, onnx_path, cfg, device)

    else:
        # 插入 QDQ 节点 → calibration → 导出 QDQ ONNX
        logger.info("Inserting QDQ nodes and calibrating...")

        # 需要在 quant_modules.initialize() 之后重新构建模型
        quant_modules.initialize()
        qdq_model = build_model(cfg, device)
        qdq_model.load_state_dict(
            torch.load(cfg["train"]["checkpoint_path"], map_location=device)
        )

        _, val_loader, _ = build_dataloaders(cfg)
        q_cfg = cfg["quantize"]
        percentile = q_cfg.get("best_percentile") or q_cfg["percentile_mid"]

        calibrate(
            qdq_model=qdq_model,
            loader=val_loader,
            calibration_samples=q_cfg["calibration_samples"],
            device=device,
            method="histogram",
            percentile=percentile,
        )

        logger.info("Exporting QDQ ONNX (percentile=%.3f)...", percentile)
        export_onnx(qdq_model, onnx_path, cfg, device)

    logger.info("Done. ONNX saved to: %s", onnx_path)


if __name__ == "__main__":
    main()
