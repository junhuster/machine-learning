"""
Nsight Compute (ncu) 专用短脚本 — 推理模式（轻量版）
- 加载训练好的模型，跑1个推理step，配合 ncu 命令做 kernel 级分析
- 用法: ncu --set speedoflight --launch-skip 5 --launch-count 30 -o ncu_report python3 profile_ncu.py

优化点（针对ncu慢的问题）：
  1. 用 ncu_batch_size（默认64），比 infer_batch_size(512) 小很多，kernel 数量和计算量大幅减少
     kernel 类型和大batch完全一样，ncu 看单个kernel硬件指标不受batch影响
  2. 预生成1个batch复用，避免每步CPU侧重新生成随机数据
  3. 去掉warmup（ncu用 --launch-skip 跳过初始kernel即可）
  4. 只跑1个推理step（推理是重复性的，1步包含所有kernel类型）
  5. 去掉多余的后处理D2H传输
"""
import os
import yaml
import torch
import torch.nn as nn

from model import DCNDIN
from fake_data import generate_batch

# 日志配置：只写文件，不输出到控制台
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_ncu.log')
    _fh = logging.FileHandler(_log_path, mode='a', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(_fh)
    _logger.propagate = False


def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', "Nsight Compute requires CUDA GPU"

    # 加载训练好的模型
    save_path = config.get('save_path', './dcn_din_model.pt')
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model = DCNDIN(ckpt['config']).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        _logger.info(f"[NCU] Loaded trained model from {save_path}")
    else:
        _logger.warning(f"[NCU] Model file {save_path} not found, using fresh model")
        model = DCNDIN(config).to(device)

    model.eval()

    # ncu专用小batch：默认64，可在config里加 ncu_batch_size 覆盖
    # 小batch = 少kernel + 小计算量 = ncu跑得快，kernel类型和大batch完全一样
    ncu_batch_size = config.get('ncu_batch_size', 64)

    # 尝试导入 NVTX（用于在 ncu timeline 中标注推理阶段）
    try:
        import torch.cuda.nvtx as nvtx
        _has_nvtx = True
    except ImportError:
        _has_nvtx = False

    # 预生成1个batch，复用（避免每步CPU侧重新生成随机数据，减少CPU开销）
    _logger.info(f"[NCU] Pre-generating batch (size={ncu_batch_size})...")
    batch = generate_batch(config, ncu_batch_size, device)

    # 确保所有CUDA操作完成，再开始计时/采集
    torch.cuda.synchronize()

    # 只跑1个推理step（推理是重复性的，1步包含所有kernel类型）
    _logger.info(f"[NCU] Running 1 inference step (batch_size={ncu_batch_size}) for kernel profiling...")

    with torch.no_grad():
        if _has_nvtx:
            nvtx.range_push("inference_step")

        # 模型推理（核心，ncu采集这里的kernel）
        if _has_nvtx:
            nvtx.range_push("nn_inference")
        logits = model(**batch)
        if _has_nvtx:
            nvtx.range_pop()

        # 轻量后处理（只sigmoid，不做D2H传输，避免多余kernel）
        if _has_nvtx:
            nvtx.range_push("postprocess")
        probs = torch.sigmoid(logits)
        if _has_nvtx:
            nvtx.range_pop()

        if _has_nvtx:
            nvtx.range_pop()

    torch.cuda.synchronize()
    _logger.info(f"[NCU] Done. output shape={logits.shape}, mean_prob={probs.mean().item():.4f}")


if __name__ == '__main__':
    main()
