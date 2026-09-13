"""
训练脚本
- 加载fake数据，对1000条样本无限循环采样，直到达到max_steps
- 每log_every步打印日志: loss, step, 已耗时, 平均每步耗时, 剩余时间(ETA)
- 训练结束保存模型
- 内置NVTX range标记，配合nsys profile使用
"""
import os
import sys
import time
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn

from model import OneTrans

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

# NVTX: 用于nsight system timeline标记
# 注意: CPU-only torch 上 import 成功但运行时调用会失败，需做运行时检测
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

    # 加载fake数据
    data = load_data(config['data_path'])
    n_samples = len(data['label'])
    _logger.info(f"[INFO] Loaded {n_samples} samples from {config['data_path']}")

    # 模型 & 优化器
    model = OneTrans(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    batch_size = config['batch_size']
    max_steps = config['max_steps']
    log_every = config['log_every']

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    _logger.info(f"[INFO] Model params: {total_params:,}")
    _logger.info(f"[INFO] batch_size={batch_size}, max_steps={max_steps}, log_every={log_every}")
    _logger.info("=" * 60)

    model.train()
    start_time = time.time()
    indices = np.arange(n_samples)

    for step in range(1, max_steps + 1):
        if HAS_NVTX:
            nvtx.range_push(f"step_{step}")

        # 从1000条样本中随机采样一个batch（无限循环）
        batch_idx = np.random.choice(indices, size=batch_size, replace=True)

        if HAS_NVTX:
            nvtx.range_push("data_load_cpu")
        # 先在CPU上创建tensor（模拟特征拉取/预处理后的数据在CPU内存）
        user_id_cpu = torch.tensor(data['user_id'][batch_idx])
        item_id_cpu = torch.tensor(data['item_id'][batch_idx])
        cate_id_cpu = torch.tensor(data['cate_id'][batch_idx])
        history_cpu = torch.tensor(data['history'][batch_idx])
        history_mask_cpu = torch.tensor(data['history_mask'][batch_idx])
        numeric_cpu = torch.tensor(data['numeric'][batch_idx])
        label_cpu = torch.tensor(data['label'][batch_idx])
        if HAS_NVTX:
            nvtx.range_pop()

        # 显式H2D数据搬运（nsys可观测cudaMemcpyHtoD）
        if HAS_NVTX:
            nvtx.range_push("h2d_transfer")
        user_id = user_id_cpu.to(device)
        item_id = item_id_cpu.to(device)
        cate_id = cate_id_cpu.to(device)
        history = history_cpu.to(device)
        history_mask = history_mask_cpu.to(device)
        numeric = numeric_cpu.to(device)
        label = label_cpu.to(device)
        if HAS_NVTX:
            nvtx.range_pop()

        # Forward
        if HAS_NVTX:
            nvtx.range_push("forward")
        logits = model(user_id, item_id, cate_id, history, history_mask, numeric)
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
