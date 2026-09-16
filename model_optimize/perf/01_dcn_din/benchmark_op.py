"""
微基准测试脚本（对应文档第四层：微基准验证 + Roofline分析）
用 torch.utils.benchmark 精确测量推荐模型中各类关键算子的耗时，
对照文档中的Roofline表判断每个算子是memory-bound还是compute-bound。

测试算子:
  1. Embedding Lookup   - 随机访问，典型memory-bound
  2. GEMM (Linear)      - 矩阵乘，小GEMM memory-bound，大GEMM compute-bound
  3. Element-wise Mul+Add - 逐元素运算，典型memory-bound
  4. Attention (QK^T+Softmax+AV^T) - 推荐注意力，memory-bound
  5. LayerNorm           - 多次读写，memory-bound
  6. Concat              - 纯数据搬运，memory-bound
"""
import os
import yaml
import torch
from torch.utils import benchmark

# 日志配置：只写文件，不输出到控制台
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark.log')
    _fh = logging.FileHandler(_log_path, mode='a', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(_fh)
    _logger.propagate = False


def load_config():
    path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def benchmark_embedding_lookup(device, ed, num_items, batch_size, seq_len):
    """Embedding Lookup: 随机gather，典型memory-bound"""
    _logger.info("\n--- 1. Embedding Lookup (memory-bound) ---")
    emb = torch.nn.Embedding(num_items, ed).to(device)
    indices = torch.randint(0, num_items, (batch_size, seq_len), device=device)

    timer = benchmark.Timer(
        stmt='emb(indices)',
        globals={'emb': emb, 'indices': indices},
        num_threads=1,
    )
    result = timer.blocked_autorange(min_run_time=1.0)
    _logger.info(f"  batch={batch_size}, seq_len={seq_len}, vocab={num_items}, dim={ed}")
    _logger.info(f"  avg = {result.mean * 1000:.4f} ms")
    # 估算: 读取字节数 = batch * seq_len * ed * 4bytes (FP32)
    bytes_read = batch_size * seq_len * ed * 4
    flops = 0  # embedding lookup纯读取，无计算
    ai = flops / bytes_read if bytes_read > 0 else 0
    _logger.info(f"  bytes_read = {bytes_read / 1024:.1f} KB, FLOPs = {flops}, AI = {ai:.4f} FLOPs/byte")
    _logger.info(f"  => memory-bound (AI=0, 纯随机内存访问)")
    return result.mean * 1000


def benchmark_gemm(device, m, k, n, label=""):
    """GEMM: 矩阵乘，小GEMM memory-bound，大GEMM compute-bound"""
    _logger.info(f"\n--- 2. GEMM {label} ---")
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)

    timer = benchmark.Timer(
        stmt='a @ b',
        globals={'a': a, 'b': b},
        num_threads=1,
    )
    result = timer.blocked_autorange(min_run_time=1.0)
    _logger.info(f"  M={m}, K={k}, N={n}")
    _logger.info(f"  avg = {result.mean * 1000:.4f} ms")
    # FLOPs = 2 * M * N * K (乘加)
    flops = 2 * m * n * k
    # 读写字节 = (M*K + K*N + M*N) * 4
    bytes_rw = (m * k + k * n + m * n) * 4
    ai = flops / bytes_rw
    _logger.info(f"  FLOPs = {flops / 1e6:.1f} M, bytes_rw = {bytes_rw / 1024:.1f} KB, AI = {ai:.2f} FLOPs/byte")
    # T4平衡点大约: 峰值算力8.1TFLOPS / 带宽320GB/s ≈ 25 FLOPs/byte (FP32)
    balance_point = 8.1e12 / 320e9
    if ai < balance_point:
        _logger.info(f"  => memory-bound (AI={ai:.1f} < 平衡点{balance_point:.1f})")
    else:
        _logger.info(f"  => compute-bound (AI={ai:.1f} >= 平衡点{balance_point:.1f})")
    return result.mean * 1000


def benchmark_elementwise(device, size, label=""):
    """Element-wise Mul+Add: 逐元素运算，典型memory-bound"""
    _logger.info(f"\n--- 3. Element-wise Mul+Add {label} ---")
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    c = torch.randn(size, device=device)

    timer = benchmark.Timer(
        stmt='a * b + c',
        globals={'a': a, 'b': b, 'c': c},
        num_threads=1,
    )
    result = timer.blocked_autorange(min_run_time=1.0)
    _logger.info(f"  size = {size:,}")
    _logger.info(f"  avg = {result.mean * 1000:.4f} ms")
    # 读3个tensor + 写1个 = 4 * size * 4 bytes
    bytes_rw = 4 * size * 4
    flops = 2 * size  # 1次mul + 1次add
    ai = flops / bytes_rw
    _logger.info(f"  FLOPs = {flops / 1e6:.1f} M, bytes_rw = {bytes_rw / 1024:.1f} KB, AI = {ai:.4f} FLOPs/byte")
    _logger.info(f"  => memory-bound (AI极低，每元素只做2次运算但读写4次)")
    return result.mean * 1000


def benchmark_attention(device, batch_size, seq_len, dim, label=""):
    """Attention: QK^T + Softmax + AV^T，推荐场景通常memory-bound"""
    _logger.info(f"\n--- 4. Attention {label} ---")
    q = torch.randn(batch_size, seq_len, dim, device=device)
    k = torch.randn(batch_size, seq_len, dim, device=device)
    v = torch.randn(batch_size, seq_len, dim, device=device)
    scale = dim ** -0.5

    def attention():
        scores = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, L, L]
        weights = torch.softmax(scores, dim=-1)
        out = torch.bmm(weights, v)  # [B, L, D]
        return out

    timer = benchmark.Timer(
        stmt='attention()',
        globals={'attention': attention},
        num_threads=1,
    )
    result = timer.blocked_autorange(min_run_time=1.0)
    _logger.info(f"  batch={batch_size}, seq_len={seq_len}, dim={dim}")
    _logger.info(f"  avg = {result.mean * 1000:.4f} ms")
    # FLOPs: QK^T = 2*B*L*L*D, AV^T = 2*B*L*L*D, softmax ≈ B*L*L*5
    flops = 4 * batch_size * seq_len * seq_len * dim + 5 * batch_size * seq_len * seq_len
    # 字节: 读写Q,K,V,scores,weights,out
    bytes_rw = (3 * batch_size * seq_len * dim + 3 * batch_size * seq_len * seq_len) * 4
    ai = flops / bytes_rw if bytes_rw > 0 else 0
    _logger.info(f"  FLOPs = {flops / 1e6:.1f} M, bytes_rw = {bytes_rw / 1024:.1f} KB, AI = {ai:.2f} FLOPs/byte")
    _logger.info(f"  => 短序列时memory-bound（kernel launch overhead + 多次小GEMM）")
    return result.mean * 1000


def benchmark_layernorm(device, batch_size, seq_len, dim):
    """LayerNorm: 多次读写，计算量小，memory-bound"""
    _logger.info(f"\n--- 5. LayerNorm (memory-bound) ---")
    ln = torch.nn.LayerNorm(dim).to(device)
    x = torch.randn(batch_size, seq_len, dim, device=device)

    timer = benchmark.Timer(
        stmt='ln(x)',
        globals={'ln': ln, 'x': x},
        num_threads=1,
    )
    result = timer.blocked_autorange(min_run_time=1.0)
    _logger.info(f"  batch={batch_size}, seq_len={seq_len}, dim={dim}")
    _logger.info(f"  avg = {result.mean * 1000:.4f} ms")
    # 读x + 写out + 读gamma/beta = 约3 * B*L*D * 4
    bytes_rw = 3 * batch_size * seq_len * dim * 4
    # 计算: mean, var, normalize, scale, shift ≈ 7 * B*L*D
    flops = 7 * batch_size * seq_len * dim
    ai = flops / bytes_rw
    _logger.info(f"  FLOPs = {flops / 1e6:.1f} M, bytes_rw = {bytes_rw / 1024:.1f} KB, AI = {ai:.4f} FLOPs/byte")
    _logger.info(f"  => memory-bound (AI≈2.3，每元素7次运算但读写3次)")
    return result.mean * 1000


def benchmark_concat(device, num_tensors, batch_size, dim):
    """Concat: 纯数据搬运，无计算，memory-bound"""
    _logger.info(f"\n--- 6. Concat (memory-bound, 纯数据搬运) ---")
    tensors = [torch.randn(batch_size, dim, device=device) for _ in range(num_tensors)]

    timer = benchmark.Timer(
        stmt='torch.cat(tensors, dim=-1)',
        globals={'tensors': tensors},
        num_threads=1,
    )
    result = timer.blocked_autorange(min_run_time=1.0)
    _logger.info(f"  num_tensors={num_tensors}, batch={batch_size}, dim={dim}")
    _logger.info(f"  avg = {result.mean * 1000:.4f} ms")
    bytes_rw = 2 * num_tensors * batch_size * dim * 4  # 读+写
    flops = 0
    ai = 0
    _logger.info(f"  bytes_rw = {bytes_rw / 1024:.1f} KB, FLOPs = 0, AI = 0")
    _logger.info(f"  => memory-bound (纯数据搬运，无任何计算)")
    return result.mean * 1000


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.info(f"[INFO] Device: {device}")
    if device.type == 'cuda':
        _logger.info(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        _logger.info(f"[INFO] 峰值算力(FP32): ~8.1 TFLOPS, 显存带宽: ~320 GB/s")
        _logger.info(f"[INFO] Roofline平衡点: ~{8.1e12 / 320e9:.1f} FLOPs/byte")

    ed = config['embedding_dim']
    bs = config['batch_size']
    seq_len = config['history_len']
    num_items = config['num_items']

    _logger.info("=" * 60)
    _logger.info("微基准测试: 推荐模型关键算子 Roofline 分析")
    _logger.info("=" * 60)

    results = {}
    results['embedding_lookup'] = benchmark_embedding_lookup(device, ed, num_items, bs, seq_len)
    results['gemm_small'] = benchmark_gemm(device, bs, ed * 4, 128, label="(小GEMM, M=batch)")
    results['gemm_large'] = benchmark_gemm(device, 1024, 1024, 1024, label="(大GEMM)")
    results['elementwise'] = benchmark_elementwise(device, bs * ed * 4, label="(Cross层逐元素乘加)")
    results['attention'] = benchmark_attention(device, bs, seq_len, ed, label="(DIN/OneTrans注意力)")
    results['layernorm'] = benchmark_layernorm(device, bs, seq_len + 5, ed)
    results['concat'] = benchmark_concat(device, 5, bs, ed)

    _logger.info("\n" + "=" * 60)
    _logger.info("汇总: 各算子耗时对比")
    _logger.info("=" * 60)
    _logger.info(f"{'算子':<25s} {'avg(ms)':>10s} {'瓶颈类型':>15s}")
    _logger.info("-" * 55)
    bottleneck = {
        'embedding_lookup': 'memory-bound',
        'gemm_small': 'memory-bound',
        'gemm_large': 'compute-bound',
        'elementwise': 'memory-bound',
        'attention': 'memory-bound',
        'layernorm': 'memory-bound',
        'concat': 'memory-bound',
    }
    for name, t in results.items():
        _logger.info(f"{name:<25s} {t:>10.4f} {bottleneck.get(name, ''):>15s}")
    _logger.info("=" * 60)
    _logger.info("\n结论: 推荐模型绝大多数算子是memory-bound，优化方向是减少内存访问")
    _logger.info("  (算子融合、量化、数据复用、减少H2D/D2H搬运)")
    _logger.info("  只有大GEMM是compute-bound，可用Tensor Core/混合精度加速")


if __name__ == '__main__':
    main()
