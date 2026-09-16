"""
训练脚本（精排场景）
- 加载 fake 数据，每步随机采 1 条样本（1 个用户 + N 个候选 item）
- 前向得到 N 个分数，BCEWithLogitsLoss 对 N 个 label 求 loss
- 每 log_every 步打印 loss / 耗时 / ETA
- 训练结束保存模型
- 内置 NVTX range 标记，配合 nsys profile 使用
"""
import os
import time
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn

from model import RankModel

# 日志配置：只写文件，不输出到控制台
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.log')
    _fh = logging.FileHandler(_log_path, mode='a', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(_fh)
    _logger.propagate = False

# NVTX：用于 nsys timeline 标记（CPU-only 环境下做运行时检测）
HAS_NVTX = False
nvtx = None
try:
    import torch.cuda.nvtx as _nvtx
    _nvtx.range_push("nvtx_probe")
    _nvtx.range_pop()
    nvtx = _nvtx
    HAS_NVTX = True
except Exception:
    HAS_NVTX = False


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def train():
    config = load_config()
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.info(f"[INFO] Device: {device}")
    if device.type == 'cuda':
        _logger.info(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # 加载 fake 数据
    data = load_data(config['data_path'])
    n_samples = len(data['user_id'])
    _logger.info(f"[INFO] Loaded {n_samples} samples from {config['data_path']}")

    # 模型 & 优化器
    model = RankModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    num_candidates = config['num_candidates']
    max_steps = config['max_steps']
    log_every = config['log_every']

    total_params = sum(p.numel() for p in model.parameters())
    _logger.info(f"[INFO] Model params: {total_params:,}")
    _logger.info(f"[INFO] num_candidates={num_candidates}, max_steps={max_steps}, log_every={log_every}")
    _logger.info("=" * 60)

    model.train()
    start_time = time.time()
    indices = np.arange(n_samples)

    for step in range(1, max_steps + 1):
        if HAS_NVTX:
            nvtx.range_push(f"step_{step}")

        # 随机采 1 条样本
        idx = np.random.choice(indices)

        if HAS_NVTX:
            nvtx.range_push("data_load_cpu")
        user_id_cpu = torch.tensor(data['user_id'][idx])
        hist_ids_cpu = torch.tensor(data['hist_ids'][idx])
        item_ids_cpu = torch.tensor(data['item_ids'][idx])
        label_cpu = torch.tensor(data['labels'][idx])
        if HAS_NVTX:
            nvtx.range_pop()

        if HAS_NVTX:
            nvtx.range_push("h2d_transfer")
        user_id = user_id_cpu.to(device)
        hist_ids = hist_ids_cpu.to(device)
        item_ids = item_ids_cpu.to(device)
        label = label_cpu.to(device)
        if HAS_NVTX:
            nvtx.range_pop()

        # Forward
        if HAS_NVTX:
            nvtx.range_push("forward")
        logits = model(user_id, hist_ids, item_ids)   # [N]
        loss = criterion(logits, label)
        if HAS_NVTX:
            nvtx.range_pop()

        # Backward & Optimize
        if HAS_NVTX:
            nvtx.range_push("backward")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if HAS_NVTX:
            nvtx.range_pop()

        # 日志
        if step % log_every == 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            avg_per_step = elapsed / step
            remaining = avg_per_step * (max_steps - step)
            gpu_info = ""
            if device.type == 'cuda':
                gpu_util = torch.cuda.utilization()
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                gpu_info = f"  gpu_util={gpu_util:>3d}%  mem={mem_alloc:>6.0f}/{mem_reserved:>6.0f}MB"
            _logger.info(
                f"[Step {step:>5d}/{max_steps}] "
                f"loss={loss.item():.4f}  "
                f"elapsed={elapsed:>7.1f}s  "
                f"avg={avg_per_step * 1000:>6.1f}ms/step  "
                f"ETA={remaining:>7.1f}s"
                f"{gpu_info}"
            )

        if HAS_NVTX:
            nvtx.range_pop()

    # 保存模型
    if device.type == 'cuda':
        torch.cuda.synchronize()
    save_path = config['save_path']
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, save_path)

    total_time = time.time() - start_time
    _logger.info("=" * 60)
    _logger.info(f"[DONE] Training finished in {total_time:.1f}s")
    _logger.info(f"[DONE] Model saved to {save_path}")
    _logger.info(f"[DONE] Avg step time: {total_time / max_steps * 1000:.1f}ms")


if __name__ == '__main__':
    train()
