"""
PyTorch Profiler 性能分析脚本 —— 推理模式（精排打分）
- 加载训练好的模型，1 个用户对 N 个候选 item 打分（前向 only）
- 导出 Chrome trace（Perfetto 打开）
- 配合本工程埋的两个坑，分析：
   坑1：item_tower 内大量微小 kernel + 空隙（launch overhead）
   坑2：user_tower 区间重复 N 次（user 特征冗余）

使用方法:
  python3 profile_torch.py
  # 产出 trace_chrome.json，用 https://ui.perfetto.dev/ 打开
"""
import os
import json
import yaml
import torch

from model import RankModel
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

    # 加载训练好的模型
    save_path = config.get('save_path', './rank_model.pt')
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model = RankModel(ckpt['config']).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        _logger.info(f"[INFO] Loaded trained model from {save_path}")
    else:
        _logger.warning(f"[WARN] Model file {save_path} not found, using fresh model")
        model = RankModel(config).to(device)

    model.eval()
    num_candidates = config['num_candidates']

    # 预生成 1 个用户 + N 个候选（复用，避免每步 CPU 侧重新生成）
    batch = generate_batch(config, device=device)

    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(batch['user_id'], batch['hist_ids'], batch['item_ids'])
    torch.cuda.synchronize()

    # Profiler 采集：wait=1, warmup=1, active=3（采第 3~5 步）
    chrome_path = os.path.join(os.path.dirname(__file__), 'trace_chrome.json')
    _logger.info(f"[INFO] Profiling 5 steps (wait=1, warmup=1, active=3), N={num_candidates}...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                _ = model(batch['user_id'], batch['hist_ids'], batch['item_ids'])
                prof.step()

    torch.cuda.synchronize()

    # 导出 Chrome trace（只能导出一次）
    prof.export_chrome_trace(chrome_path)
    _logger.info(f"[INFO] Chrome trace exported to {chrome_path}")

    # 归一化时间轴（修复查看器 x 轴显示"年"的问题）
    try:
        from fix_trace_time import normalize_chrome_trace
        _stats = normalize_chrome_trace(chrome_path, inplace=True, verbose=False)
        _logger.info(
            f"[INFO] Trace normalized: events={_stats['events']} dropped={_stats['dropped']} "
            f"span_after={_stats['span_after']:.4g} {_stats['unit']}"
        )
    except Exception as e:
        _logger.warning(f"[WARN] Trace normalization failed: {e}")

    # 打印 CPU 算子耗时 TOP（对照 Perfetto 看）
    _logger.info("\n===== CPU Operators (by CPU time, TOP 15) =====")
    _logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    # 打印 CUDA kernel 耗时 TOP（这里能直接看到"哪些 kernel 重复 N 次"）
    if device.type == 'cuda':
        _logger.info("\n===== CUDA Kernels (by CUDA time, TOP 15) =====")
        _logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    _logger.info(f"\n[DONE] Chrome trace: {chrome_path}")
    _logger.info(f"[DONE] 用 https://ui.perfetto.dev/ 打开，分析两个坑（见 README）")


if __name__ == '__main__':
    main()
