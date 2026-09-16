#!/usr/bin/env python3
"""⑥ 压测对比 FP32 vs INT8（任务2：DLRM 结构），产出结论报告。

用法:
    python3 bench_compare.py
    python3 bench_compare.py --batch-sizes 1,8,32,64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.bench import bench_engine, format_markdown_table  # noqa: E402
from common.conf import get_device, load_config, resolve  # noqa: E402
from common.dataio import iter_batches, load_split  # noqa: E402
from common.metrics import auc_score  # noqa: E402
from common.trt_util import TRTEngine, trt_version  # noqa: E402
from inputs import make_inputs  # noqa: E402
from model import INPUT_NAMES  # noqa: E402


@torch.no_grad()
def engine_auc(engine, val_set, device, batch_size: int = 64, max_batches: int = 40) -> float:
    scores, labels = [], []
    for i, b in enumerate(iter_batches(val_set, batch_size)):
        if i >= max_batches:
            break
        inputs = {k: getattr(b, k).to(device) for k in INPUT_NAMES}
        out = engine.infer(inputs)
        scores.append(torch.sigmoid(out["logit"].float()).cpu().numpy())
        labels.append(b.label.numpy())
    return auc_score(np.concatenate(labels), np.concatenate(scores))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=str, default="")
    args = ap.parse_args()

    cfg = load_config("config.yaml", __file__)
    ben = cfg["bench"]
    device = get_device("cuda")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else ben["batch_sizes"]

    data_path = resolve(cfg, cfg["data"]["data_path"])
    val_set = load_split(data_path, "val")

    engines = {}
    for tag, fname in (("FP32", "model_fp32.plan"), ("INT8", "model_int8.plan")):
        p = resolve(cfg, f"outputs/{fname}")
        if p.exists():
            engines[tag] = TRTEngine(str(p))
        else:
            print(f"[bench] ⚠️ 跳过 {tag}: 找不到 {p}")
    if not engines:
        raise SystemExit("没有 engine 可压测，请先跑 build_engine.py")

    import torch as _t

    report = ["# 压测报告 · 任务2（DQRM 简化版 / DLRM 结构）\n",
              f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
              f"- TensorRT: {trt_version()}",
              f"- torch: {_t.__version__} | CUDA: {_t.version.cuda} | GPU: {_t.cuda.get_device_name(0)}",
              f"- 输入来源: 验证集真实分布 | warmup={ben['warmup']} iters={ben['iters']}\n"]

    report.append("## 1. 精度对比（AUC，验证集）\n")
    auc_rows = []
    for tag, eng in engines.items():
        auc = engine_auc(eng, val_set, device)
        auc_rows.append({"engine": tag, "auc": auc})
        print(f"[bench] {tag} AUC = {auc:.5f}")
    report.append(format_markdown_table(auc_rows, ["engine", "auc"]))
    if len(auc_rows) == 2:
        report.append(f"\n**ΔAUC(INT8 - FP32) = {auc_rows[1]['auc'] - auc_rows[0]['auc']:+.5f}**\n")

    report.append("\n## 2. 性能对比（延迟 ms / 吞吐 samples·s⁻¹）\n")
    all_stats = {}
    for tag, eng in engines.items():
        print(f"\n[bench] 压测 {tag} ...")
        factory = lambda bs: make_inputs(cfg, bs, device, source="val", data_path=str(data_path))[0]
        all_stats[tag] = bench_engine(eng, factory, batch_sizes, warmup=ben["warmup"], iters=ben["iters"])

    rows = []
    for bs in batch_sizes:
        row = {"batch": bs}
        for tag, stats in all_stats.items():
            for s in stats:
                if s.batch_size == bs:
                    row[f"{tag} mean_ms"] = s.mean_ms
                    row[f"{tag} p99_ms"] = s.p99_ms
                    row[f"{tag} samples/s"] = s.samples_per_sec
        if "FP32 mean_ms" in row and "INT8 mean_ms" in row:
            row["加速比"] = row["FP32 mean_ms"] / row["INT8 mean_ms"]
        rows.append(row)
    headers = ["batch"]
    for tag in all_stats:
        headers += [f"{tag} mean_ms", f"{tag} p99_ms", f"{tag} samples/s"]
    if len(all_stats) == 2:
        headers.append("加速比")
    report.append(format_markdown_table(rows, headers))

    csv_path = resolve(cfg, "outputs/bench_stats.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("engine,batch_size,mean_ms,p50_ms,p90_ms,p99_ms,samples_per_sec,ms_per_sample\n")
        for tag, stats in all_stats.items():
            for s in stats:
                f.write(f"{tag},{s.batch_size},{s.mean_ms:.4f},{s.p50_ms:.4f},{s.p90_ms:.4f},"
                        f"{s.p99_ms:.4f},{s.samples_per_sec:.2f},{s.ms_per_sample:.4f}\n")

    report.append("\n## 3. 结论\n")
    if len(all_stats) == 2:
        small = next((r for r in rows if r["batch"] == 1), None)
        best = max((r for r in rows if "加速比" in r), key=lambda r: r["加速比"], default=None)
        if small:
            report.append(f"- **batch=1**: FP32 {small['FP32 mean_ms']:.2f} ms → "
                          f"INT8 {small['INT8 mean_ms']:.2f} ms，加速 "
                          f"**{small.get('加速比', float('nan')):.2f}x**")
        if best:
            report.append(f"- **最佳档位**: batch={best['batch']}，{best['加速比']:.2f}x")
        if len(auc_rows) == 2:
            report.append(f"- **精度代价**: ΔAUC = {auc_rows[1]['auc'] - auc_rows[0]['auc']:+.5f}")
        report.append(
            "\n解读要点（也是与任务1 的对比结论）：\n"
            "  1. DLRM 类模型的**大头是 embedding 查表**，而 Gather 既没有 INT8 加速、\n"
            "     也不会被 torch.ao / TensorRT 量化 —— 所以 INT8 只能加速 dense 的 MLP 部分；\n"
            "  2. 模型里 sparse 占比越高，INT8 的端到端加速比越低（这正是精排模型的现实）；\n"
            "  3. 对比任务1（Transformer，dense 占主导）的加速比，可以直接说明\n"
            "     「量化收益取决于 dense/sparse 比例」这个结论。\n"
        )

    out = resolve(cfg, ben["report_path"])
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"[bench] 报告: {out}")
    print("\n" + "\n".join(report[-16:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
