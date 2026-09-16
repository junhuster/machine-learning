#!/usr/bin/env python3
"""压测框架：统一的延迟/吞吐度量 + 结果表格化。

用 torch.cuda.Event 计时（GPU 侧），比 time.time() 准；
每次迭代都 record + synchronize，得到的是「单请求端到端延迟」，可直接取百分位。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch


@dataclass
class LatencyStats:
    batch_size: int
    iters: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    samples_per_sec: float   # 吞吐：batch_size * 1000 / mean_ms
    ms_per_sample: float     # 单样本摊薄延迟

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def time_loop(fn: Callable[[], None], iters: int) -> np.ndarray:
    """执行 fn iters 次，返回每次耗时（ms）。fn 内部应完成一次完整推理。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    return times


def summarize(times: np.ndarray, batch_size: int) -> LatencyStats:
    mean = float(times.mean())
    return LatencyStats(
        batch_size=batch_size,
        iters=int(times.shape[0]),
        mean_ms=mean,
        p50_ms=float(np.percentile(times, 50)),
        p90_ms=float(np.percentile(times, 90)),
        p99_ms=float(np.percentile(times, 99)),
        min_ms=float(times.min()),
        max_ms=float(times.max()),
        samples_per_sec=batch_size * 1000.0 / mean if mean > 0 else float("nan"),
        ms_per_sample=mean / batch_size if batch_size > 0 else float("nan"),
    )


def bench_callable(
    run_once: Callable[[], None],
    batch_size: int,
    warmup: int = 20,
    iters: int = 100,
) -> LatencyStats:
    """run_once() 执行一次推理；先 warmup 再计时。"""
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    times = time_loop(run_once, iters)
    return summarize(times, batch_size)


def bench_engine(
    engine,
    input_factory: Callable[[int], Dict[str, torch.Tensor]],
    batch_sizes: Sequence[int],
    warmup: int = 20,
    iters: int = 100,
) -> List[LatencyStats]:
    """对一个 TRTEngine 扫 batch size，逐档出 LatencyStats。"""
    results: List[LatencyStats] = []
    for bs in batch_sizes:
        inputs = input_factory(bs)
        engine.infer(inputs)  # 触发一次，确保 context 形状已绑定
        stats = bench_callable(lambda: engine.infer(inputs), bs, warmup=warmup, iters=iters)
        results.append(stats)
        print(f"  batch={bs:4d}  mean={stats.mean_ms:8.3f} ms  "
              f"p99={stats.p99_ms:8.3f} ms  thr={stats.samples_per_sec:10.1f} samples/s")
    return results


# ---------- 结果输出 ----------
def format_markdown_table(rows: Sequence[Dict], headers: Sequence[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = []
        for h in headers:
            v = r[h]
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def format_csv(rows: Sequence[Dict], headers: Sequence[str]) -> str:
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in headers))
    return "\n".join(lines)
