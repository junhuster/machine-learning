"""
推理压测脚本（性能分析增强版）
- 端到端延迟拆解: 数据生成 → CPU预处理(模拟特征拉取) → H2D搬运 → NN推理 → D2H搬运 → 后处理
- 显式H2D/D2H数据搬运（nsys可观测Memcpy）
- 以指定QPS持续发送请求，达到target_requests后停止
- 输出延迟统计: avg/p50/p90/p99/max + 各阶段占比
- 支持batch_sweep模式: 测试不同batch_size下的延迟和吞吐
"""
import os
import time
import yaml
import numpy as np
import torch

from model import DCNDIN
from fake_data import generate_batch

# 日志配置：只写文件，不输出到控制台
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference.log')
    _fh = logging.FileHandler(_log_path, mode='a', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(_fh)
    _logger.propagate = False

# NVTX
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


def cpu_preprocess(batch_cpu, config, device):
    """
    模拟CPU侧特征拉取/预处理
    真实场景: 从PS/Redis拉取embedding key → int化 → 拼接 → padding
    这里用可控的CPU计算+sleep模拟，制造GPU等CPU的场景
    """
    if not config.get('enable_cpu_preprocess', True):
        return batch_cpu

    if HAS_NVTX:
        nvtx.range_push("cpu_preprocess")

    # 模拟特征预处理: 对数据做一些CPU侧numpy操作
    # (真实场景中这一步是特征拉取和int化，耗时不可忽略)
    _ = batch_cpu['user_id'].numpy()
    _ = batch_cpu['history'].numpy().sum()
    _ = batch_cpu['numeric'].numpy().mean()

    # 可控的模拟耗时（调大可放大CPU bottleneck）
    simulate_ms = config.get('cpu_preprocess_ms', 2)
    if simulate_ms > 0:
        time.sleep(simulate_ms / 1000.0)

    if HAS_NVTX:
        nvtx.range_pop()
    return batch_cpu


def run_inference_step(model, batch_cpu, config, device):
    """
    执行一次完整推理，返回各阶段耗时(ms)和输出
    阶段: cpu_preprocess → h2d → nn → d2h → postprocess
    """
    times = {}

    # 1. CPU预处理（模拟特征拉取）
    t0 = time.time()
    batch_cpu = cpu_preprocess(batch_cpu, config, device)
    times['cpu_preprocess'] = (time.time() - t0) * 1000

    # 2. 显式H2D数据搬运
    if config.get('enable_h2d_d2h', True):
        if HAS_NVTX:
            nvtx.range_push("h2d_transfer")
        t0 = time.time()
        batch = {k: v.to(device) for k, v in batch_cpu.items()}
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times['h2d'] = (time.time() - t0) * 1000
        if HAS_NVTX:
            nvtx.range_pop()
    else:
        batch = batch_cpu
        times['h2d'] = 0.0

    # 3. NN推理
    if HAS_NVTX:
        nvtx.range_push("nn_inference")
    t0 = time.time()
    with torch.no_grad():
        logits = model(**batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times['nn'] = (time.time() - t0) * 1000
    if HAS_NVTX:
        nvtx.range_pop()

    # 4. 显式D2H数据搬运
    if config.get('enable_h2d_d2h', True):
        if HAS_NVTX:
            nvtx.range_push("d2h_transfer")
        t0 = time.time()
        logits_cpu = logits.cpu()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times['d2h'] = (time.time() - t0) * 1000
        if HAS_NVTX:
            nvtx.range_pop()
    else:
        logits_cpu = logits
        times['d2h'] = 0.0

    # 5. 后处理（sigmoid → numpy）
    t0 = time.time()
    probs = torch.sigmoid(logits_cpu).numpy()
    times['postprocess'] = (time.time() - t0) * 1000

    times['total'] = sum(times.values())
    return times, probs


def benchmark_fixed_qps(config, model, device):
    """固定QPS压测模式"""
    batch_size = config['infer_batch_size']
    target_qps = config['target_qps']
    target_requests = config['target_requests']
    warmup = config.get('warmup_requests', 100)
    interval = 1.0 / target_qps

    # 各阶段耗时记录
    stage_keys = ['cpu_preprocess', 'h2d', 'nn', 'd2h', 'postprocess', 'total']
    stage_times = {k: [] for k in stage_keys}

    _logger.info(f"[INFO] batch_size={batch_size}, target_qps={target_qps}, "
          f"target_requests={target_requests}, warmup={warmup}")
    _logger.info(f"[INFO] cpu_preprocess={config.get('enable_cpu_preprocess', True)} "
          f"({config.get('cpu_preprocess_ms', 2)}ms), "
          f"h2d_d2h={config.get('enable_h2d_d2h', True)}")

    # 预热
    _logger.info(f"[INFO] Warming up {warmup} requests...")
    with torch.no_grad():
        for _ in range(warmup):
            batch_cpu = generate_batch(config, batch_size, device='cpu')
            run_inference_step(model, batch_cpu, config, device)
    _logger.info("[INFO] Warmup done.")

    # 正式压测
    _logger.info("=" * 60)
    _logger.info(f"[BENCH] Start: target_qps={target_qps}, total={target_requests}")
    start = time.time()
    next_send = start

    with torch.no_grad():
        for req_id in range(1, target_requests + 1):
            # QPS控制
            now = time.time()
            sleep_time = next_send - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_send += interval

            # 生成CPU侧数据（模拟请求到达，数据在CPU内存）
            batch_cpu = generate_batch(config, batch_size, device='cpu')

            # 执行推理
            times, _ = run_inference_step(model, batch_cpu, config, device)
            for k in stage_keys:
                stage_times[k].append(times[k])

            # 进度日志
            if req_id % 1000 == 0:
                elapsed = time.time() - start
                actual_qps = req_id / elapsed
                nn_recent = stage_times['nn'][-1000:]
                _logger.info(
                    f"[{req_id:>6d}/{target_requests}] "
                    f"actual_qps={actual_qps:>7.1f}  "
                    f"nn_lat={np.mean(nn_recent):>6.2f}ms  "
                    f"total_lat={np.mean(stage_times['total'][-1000:]):>6.2f}ms"
                )

    # 结果统计
    total_time = time.time() - start
    total_samples = target_requests * batch_size

    _logger.info("=" * 60)
    _logger.info("===== Benchmark Results =====")
    _logger.info(f"  Total requests     : {target_requests}")
    _logger.info(f"  Batch size         : {batch_size}")
    _logger.info(f"  Total samples      : {total_samples}")
    _logger.info(f"  Total time         : {total_time:.2f}s")
    _logger.info(f"  Actual QPS (req/s) : {target_requests / total_time:.2f}")
    _logger.info(f"  Throughput (samp/s): {total_samples / total_time:.2f}")

    # 端到端延迟拆解
    if config.get('enable_latency_breakdown', True):
        _logger.info("\n===== End-to-End Latency Breakdown =====")
        total_avg = np.mean(stage_times['total'])
        for k in ['cpu_preprocess', 'h2d', 'nn', 'd2h', 'postprocess']:
            avg = np.mean(stage_times[k])
            pct = (avg / total_avg * 100) if total_avg > 0 else 0
            _logger.info(f"  {k:<18s}: avg={avg:>7.3f}ms  ({pct:>5.1f}%)")
        _logger.info(f"  {'total':<18s}: avg={total_avg:>7.3f}ms  (100.0%)")

    # NN延迟统计
    nn_lat = np.array(stage_times['nn'])
    _logger.info(f"\n===== NN Inference Latency =====")
    _logger.info(f"  avg  : {np.mean(nn_lat):.3f} ms")
    _logger.info(f"  p50  : {np.percentile(nn_lat, 50):.3f} ms")
    _logger.info(f"  p90  : {np.percentile(nn_lat, 90):.3f} ms")
    _logger.info(f"  p99  : {np.percentile(nn_lat, 99):.3f} ms")
    _logger.info(f"  max  : {np.max(nn_lat):.3f} ms")

    # 总延迟统计
    total_lat = np.array(stage_times['total'])
    _logger.info(f"\n===== Total End-to-End Latency =====")
    _logger.info(f"  avg  : {np.mean(total_lat):.3f} ms")
    _logger.info(f"  p50  : {np.percentile(total_lat, 50):.3f} ms")
    _logger.info(f"  p90  : {np.percentile(total_lat, 90):.3f} ms")
    _logger.info(f"  p99  : {np.percentile(total_lat, 99):.3f} ms")
    _logger.info(f"  max  : {np.max(total_lat):.3f} ms")
    _logger.info("=" * 60)


def benchmark_batch_sweep(config, model, device):
    """Batch sweep模式: 测试不同batch_size下的延迟和吞吐"""
    batch_sizes = [1, 4, 16, 64, 256]
    num_iters = 200  # 每个batch测200次
    warmup = 20

    _logger.info("=" * 70)
    _logger.info("===== Batch Size Sweep (延迟 vs 吞吐) =====")
    _logger.info(f"{'batch':>8s} {'avg_lat(ms)':>12s} {'p50(ms)':>10s} {'p99(ms)':>10s} "
          f"{'throughput(s/s)':>16s} {'gpu_util':>10s}")
    _logger.info("-" * 70)

    for bs in batch_sizes:
        # 预热
        with torch.no_grad():
            for _ in range(warmup):
                batch_cpu = generate_batch(config, bs, device='cpu')
                run_inference_step(model, batch_cpu, config, device)

        # 正式测量
        lats = []
        with torch.no_grad():
            for _ in range(num_iters):
                batch_cpu = generate_batch(config, bs, device='cpu')
                times, _ = run_inference_step(model, batch_cpu, config, device)
                lats.append(times['nn'])

        lats = np.array(lats)
        avg_lat = np.mean(lats)
        throughput = bs / (avg_lat / 1000) if avg_lat > 0 else 0
        gpu_util = torch.cuda.utilization() if device.type == 'cuda' else 'N/A'
        gpu_str = f"{gpu_util}%" if isinstance(gpu_util, int) else gpu_util

        _logger.info(f"{bs:>8d} {avg_lat:>12.3f} {np.percentile(lats,50):>10.3f} "
              f"{np.percentile(lats,99):>10.3f} {throughput:>16.1f} {gpu_str:>10s}")

    _logger.info("=" * 70)
    _logger.info("分析要点:")
    _logger.info("  - batch=1: 延迟最低但吞吐最低，kernel launch overhead占比高")
    _logger.info("  - batch增大: 延迟上升但吞吐提升，GPU利用率提高")
    _logger.info("  - 找到延迟和吞吐的平衡点（通常batch=16~64）")
    _logger.info("  - 用nsys观察不同batch下kernel间隙和launch overhead")


def benchmark():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.info(f"[INFO] Device: {device}")
    if device.type == 'cuda':
        _logger.info(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # 加载模型
    ckpt = torch.load(config['save_path'], map_location=device, weights_only=False)
    model = DCNDIN(ckpt['config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    _logger.info(f"[INFO] Model loaded from {config['save_path']}")

    # 根据配置选择模式
    if config.get('batch_sweep', False):
        benchmark_batch_sweep(config, model, device)
    else:
        benchmark_fixed_qps(config, model, device)


if __name__ == '__main__':
    benchmark()
