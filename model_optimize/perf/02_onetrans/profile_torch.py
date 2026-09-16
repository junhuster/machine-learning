"""
PyTorch Profiler 性能分析脚本 — 推理模式
- 使用 torch.profiler 采集 CPU + CUDA 活动（推理 forward only）
- 导出 Chrome trace，可用 Perfetto (https://ui.perfetto.dev/) 或 chrome://tracing 打开
- 分析: 推理算子耗时、GPU kernel耗时、CPU/GPU时间分布、调用栈

使用方法:
  python3 profile_torch.py
  # 产出 trace_chrome.json，用 Perfetto 或 chrome://tracing 打开
"""
import os
import json
import yaml
import torch
import torch.nn as nn

from model import OneTrans
from fake_data import generate_batch

# 日志配置：只写文件，不输出到控制台
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_torch.log')
    _fh = logging.FileHandler(_log_path, mode='a', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(_fh)
    _logger.propagate = False


def load_config():
    path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.info(f"[INFO] Device: {device}")

    # 加载训练好的模型（推理模式）
    save_path = config.get('save_path', './onetrans_model.pt')
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model = OneTrans(ckpt['config']).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        _logger.info(f"[INFO] Loaded trained model from {save_path}")
    else:
        _logger.warning(f"[WARN] Model file {save_path} not found, using fresh model")
        model = OneTrans(config).to(device)

    model.eval()
    infer_batch_size = config.get('infer_batch_size', config.get('batch_size', 64))
    _logger.info(f"[INFO] Inference batch size: {infer_batch_size}")

    # 预热（推理 forward only）
    _logger.info("[INFO] Warmup 5 steps (inference)...")
    with torch.no_grad():
        for _ in range(5):
            batch = generate_batch(config, infer_batch_size, device)
            _ = model(**batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Profiler 采集（推理模式）
    # wait=1: 第1步不采集
    # warmup=1: 第2步预热
    # active=3: 第3-5步采集
    # repeat=1: 重复1轮
    chrome_path = os.path.join(os.path.dirname(__file__), 'trace_chrome.json')

    _logger.info("[INFO] Profiling 5 inference steps (wait=1, warmup=1, active=3)...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        # 注意：不使用 on_trace_ready，因为它会在 prof.step() 内部保存 trace，
        # 导致之后 export_chrome_trace 报 "Trace is already saved"。
        # 改为在 with 块结束后手动导出。
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                batch = generate_batch(config, infer_batch_size, device)
                _ = model(**batch)
                prof.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 导出 Chrome trace（只能导出一次，不能同时用 tensorboard_trace_handler，
    # 因为两者内部都调用 kineto_results.save()，会报 "Trace is already saved"）
    prof.export_chrome_trace(chrome_path)
    _logger.info(f"[INFO] Chrome trace exported to {chrome_path}")

    # 归一化时间轴（原地修复 trace_chrome.json），解决查看器 x 轴显示"年"的问题：
    # 剔除离群/坏时间戳事件（ts=0 的元数据事件、CUPTI 异常造成的负 dur / 超大 dur），
    # 把 ts 平移到 0 点；若整份时间戳单位是纳秒则自动换算成微秒。
    # 实现见同目录 fix_trace_time.py（也可单独对任意 trace 跑：python3 fix_trace_time.py trace.json）
    try:
        from fix_trace_time import normalize_chrome_trace
        _stats = normalize_chrome_trace(chrome_path, inplace=True, verbose=False)
        _logger.info(
            f"[INFO] Trace normalized: events={_stats['events']} dropped={_stats['dropped']} "
            f"span_after={_stats['span_after']:.4g} {_stats['unit']} -> {_stats['output']}"
        )
    except Exception as e:
        _logger.warning(f"[WARN] Trace normalization failed: {e}")

    # 打印CPU算子耗时TOP
    _logger.info("\n===== CPU Operators (by CPU time, TOP 20) =====")
    _logger.info(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20,
    ))

    # 打印CUDA kernel耗时TOP
    if device.type == 'cuda':
        _logger.info("\n===== CUDA Kernels (by CUDA time, TOP 20) =====")
        _logger.info(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        ))

    _logger.info(f"\n[DONE] Chrome trace: {chrome_path}")
    _logger.info(f"[DONE] 用 Perfetto (https://ui.perfetto.dev/) 或 chrome://tracing 打开 → 选择 {chrome_path}")


if __name__ == '__main__':
    main()
