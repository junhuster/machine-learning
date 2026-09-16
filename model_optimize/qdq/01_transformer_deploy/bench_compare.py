#!/usr/bin/env python3
"""⑥ 压测对比 FP32 vs INT8，产出结论报告。

用法:
    python3 bench_compare.py
    python3 bench_compare.py --batch-sizes 1,8,32,64,128
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
def engine_auc(engine, val_set, device, batch_size: int = 64, max_batches: int = 40):
    scores, labels = [], []
    for i, b in enumerate(iter_batches(val_set, batch_size)):
        if i >= max_batches:
            break
        inputs = {k: getattr(b, k).to(device) for k in INPUT_NAMES}
        out = engine.infer(inputs)
        scores.append(torch.sigmoid(out["logit"].float()).cpu().numpy())
        labels.append(b.label.numpy())
    y = np.concatenate(labels)
    s = np.concatenate(scores)
    return auc_score(y, s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=str, default="", help="覆盖 config，如 1,8,32,64")
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
        raise SystemExit("没有任何 engine 可压测，请先跑 build_engine.py")

    report = [f"# 压测报告 · 任务1（Transformer 精排）\n"]
    report.append(f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"- TensorRT: {trt_version()}")
    import torch as _t

    report.append(f"- torch: {_t.__version__} | CUDA: {_t.version.cuda} | "
                  f"GPU: {_t.cuda.get_device_name(0)}")
    report.append(f"- 输入来源: 验证集真实分布 | warmup={ben['warmup']} iters={ben['iters']}\n")

    # ---------- 精度 ----------
    report.append("## 1. 精度对比（AUC，验证集）\n")
    auc_rows = []
    for tag, eng in engines.items():
        auc = engine_auc(eng, val_set, device)
        auc_rows.append({"engine": tag, "auc": auc})
        print(f"[bench] {tag} AUC = {auc:.5f}")
    report.append(format_markdown_table(auc_rows, ["engine", "auc"]))
    if len(auc_rows) == 2:
        d = auc_rows[1]["auc"] - auc_rows[0]["auc"]
        report.append(f"\n**ΔAUC(INT8 - FP32) = {d:+.5f}**\n")

    # ---------- 性能 ----------
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

    # csv 明细
    csv_path = resolve(cfg, "outputs/bench_stats.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("engine,batch_size,mean_ms,p50_ms,p90_ms,p99_ms,samples_per_sec,ms_per_sample\n")
        for tag, stats in all_stats.items():
            for s in stats:
                f.write(f"{tag},{s.batch_size},{s.mean_ms:.4f},{s.p50_ms:.4f},{s.p90_ms:.4f},"
                        f"{s.p99_ms:.4f},{s.samples_per_sec:.2f},{s.ms_per_sample:.4f}\n")
    print(f"\n[bench] 明细: {csv_path}")

    # ---------- 结论 ----------
    report.append("\n## 3. 结论\n")
    if len(all_stats) == 2:
        best = max((r for r in rows if "加速比" in r), key=lambda r: r["加速比"], default=None)
        small = next((r for r in rows if r["batch"] == 1), None)
        if small:
            report.append(f"- **batch=1（在线单请求场景）**: FP32 {small['FP32 mean_ms']:.2f} ms → "
                          f"INT8 {small['INT8 mean_ms']:.2f} ms，加速 **{small.get('加速比', float('nan')):.2f}x**")
        if best:
            report.append(f"- **加速比最大的档位**: batch={best['batch']}，"
                          f"{best['加速比']:.2f}x")
        if len(auc_rows) == 2:
            report.append(f"- **精度代价**: ΔAUC = {auc_rows[1]['auc'] - auc_rows[0]['auc']:+.5f}")
        report.append(
            "\n解读要点：\n"
            "  1. INT8 的收益主要来自 dense 部分（Linear/attention 的大 GEMM 走 INT8 Tensor Core）；\n"
            "  2. embedding 查表（Gather）不会被量化，所以模型里 sparse 占比越高，加速比越低；\n"
            "  3. batch 越大，GPU 越饱和，INT8 的算力优势越明显；batch=1 时更受 launch/延迟限制。\n"
        )

    out = resolve(cfg, ben["report_path"])
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"[bench] 报告: {out}")
    print("\n" + "\n".join(report[-14:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
