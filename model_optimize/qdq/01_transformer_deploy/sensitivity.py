#!/usr/bin/env python3
"""④ 逐层敏感度分析 —— 决定「哪些层可以量化、哪些必须保持 FP32」。

两条互补的证据（都要看，结论才硬）：

  A) per-layer-only（单独量化一层）
     只把第 i 层打开量化，其余 FP32，看 AUC 掉多少。
     掉得多 = 这一层单独就扛不住 INT8 = 敏感层。

  B) leave-one-out（全量化后逐层恢复）
     先把所有层都量化，再逐个把第 i 层恢复成 FP32，看 AUC 回血多少。
     回血多 = 这一层是被量化坑得最狠的。

A 与 B 结论一致 → 可信；不一致 → 说明层间有交互，以 B 为主（因为线上就是全量化态）。

注意：整个分析**只在 PyTorch 里跑伪量化**，不建任何 TRT engine，所以很快
（~50 层 × 验证集前向 = 几分钟）。

用法:
    python3 sensitivity.py
    python3 sensitivity.py --max-val 8192 --layers "blocks.0,head"   # 只看部分层
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.conf import get_device, load_config, resolve  # noqa: E402
from common.dataio import iter_batches, load_split  # noqa: E402
from common.metrics import auc_score  # noqa: E402
from model import INPUT_NAMES, build_model  # noqa: E402
from qdq_lib import calibrate, list_linear_names, set_modes, wrap_model  # noqa: E402


@torch.no_grad()
def evaluate_auc(model, bundle, device, max_samples: int = 0) -> float:
    model.eval()
    scores, labels, seen = [], [], 0
    for batch in iter_batches(bundle, 4096):
        out = model(**batch.to(device).inputs(INPUT_NAMES))
        scores.append(torch.sigmoid(out).float().cpu().numpy())
        labels.append(batch.label.numpy())
        seen += batch.batch_size()
        if max_samples and seen >= max_samples:
            break
    return auc_score(np.concatenate(labels), np.concatenate(scores))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-val", type=int, default=8192, help="评估用多少条验证样本（0=全量）")
    ap.add_argument("--layers", type=str, default="", help="只分析名字含这些子串的层，逗号分隔")
    ap.add_argument("--skip-loo", action="store_true", help="跳过 leave-one-out（省时间）")
    args = ap.parse_args()

    cfg = load_config("config.yaml", __file__)
    q = cfg["quant"]
    device = get_device("cuda")

    data_path = resolve(cfg, cfg["data"]["data_path"])
    val_set = load_split(data_path, "val")
    calib_set = load_split(data_path, "calib")

    # 1) 建模型 + 载权重（**必须在包装之前载入**）
    ckpt = torch.load(resolve(cfg, cfg["train"]["save_path"]), map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    # 2) 包装所有 Linear
    wrappers = wrap_model(model, method=q["calib_method"])
    names = [w.name for w in wrappers]
    if args.layers:
        keep = tuple(args.layers.split(","))
        names = [n for n in names if any(k in n for k in keep)]
    print(f"[sensitivity] 待分析 Linear 层数: {len(names)}")

    # 3) 先校准一遍，拿到所有层的 scale
    print("[sensitivity] 校准中（统计 amax）...")
    calibrate(
        model, wrappers,
        calib_iter=(b.to(device).inputs(INPUT_NAMES)
                    for b in iter_batches(calib_set, q["calib_batch_size"])),
    )

    # 基线：全 FP32
    auc_fp32 = evaluate_auc(model, val_set, device, args.max_val)
    print(f"[sensitivity] 基线 AUC(FP32) = {auc_fp32:.5f}")

    rows = []
    # ---- A) per-layer-only ----
    print("[sensitivity] A) 逐层单独量化 ...")
    for i, name in enumerate(names):
        set_modes(wrappers, "quant", only={name})
        auc = evaluate_auc(model, val_set, device, args.max_val)
        drop = auc_fp32 - auc
        rows.append({"layer": name, "auc_only": auc, "delta_only": drop,
                     "auc_loo": float("nan"), "delta_loo": float("nan")})
        flag = "敏感" if drop > q["sensitive_auc_drop"] else ""
        print(f"  [{i + 1:3d}/{len(names)}] {name:42s} AUC={auc:.5f} Δ={drop:+.5f} {flag}")

    # ---- B) leave-one-out ----
    if not args.skip_loo:
        print("[sensitivity] B) 全量化后逐层恢复（leave-one-out）...")
        set_modes(wrappers, "quant")
        auc_all = evaluate_auc(model, val_set, device, args.max_val)
        print(f"  全量化 AUC = {auc_all:.5f} (Δ={auc_all - auc_fp32:+.5f})")
        for i, name in enumerate(names):
            set_modes(wrappers, "quant", disabled={name})
            auc = evaluate_auc(model, val_set, device, args.max_val)
            rows[i]["auc_loo"] = auc
            rows[i]["delta_loo"] = auc - auc_all
            print(f"  [{i + 1:3d}/{len(names)}] 恢复 {name:42s} -> AUC={auc:.5f} "
                  f"回血={auc - auc_all:+.5f}")

    # 4) 汇总判定：只要任一证据显示敏感，就判敏感（保守）
    sensitive = []
    for r in rows:
        hot = False
        if r["delta_only"] > q["sensitive_auc_drop"]:
            hot = True
        if not np.isnan(r["delta_loo"]) and r["delta_loo"] > q["sensitive_auc_drop"]:
            hot = True
        r["verdict"] = "sensitive" if hot else "quantizable"
        if hot:
            sensitive.append(r["layer"])

    report = resolve(cfg, q["report_path"])
    with open(report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "auc_only", "delta_only",
                                              "auc_loo", "delta_loo", "verdict"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sensitivity] 明细已写入: {report}")

    sen_path = resolve(cfg, q["sensitive_path"])
    with open(sen_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sensitive) + ("\n" if sensitive else ""))
    print(f"[sensitivity] 敏感层 {len(sensitive)}/{len(names)} 已写入: {sen_path}")
    if sensitive:
        print("  敏感层（不量化）:")
        for s in sensitive:
            print(f"    - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
