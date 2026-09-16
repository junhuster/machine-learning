#!/usr/bin/env python3
"""④ 逐层敏感度分析（torch.ao 观察器版，PTQ）。

流程：用 torch.ao 的观察器**校准一次**拿到所有层的量化参数，之后逐层开关即可，
不需要重复准备/校准，所以很快。

两条互补证据：
  A) per-layer-only : 只开启第 i 层的量化，其余 FP32  -> ΔAUC
  B) leave-one-out  : 全部开启后把第 i 层关掉        -> AUC 回血

判定：任一证据 ΔAUC > `quant.sensitive_auc_drop` → 敏感层 → 不量化。

用法:
    python3 sensitivity.py
    python3 sensitivity.py --max-val 4096 --skip-loo
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
import ao_quant as A  # noqa: E402


@torch.no_grad()
def eval_auc(model, val_set, device, max_samples: int = 0) -> float:
    model.eval()
    scores, labels, seen = [], [], 0
    for b in iter_batches(val_set, 4096):
        b = b.to(device)
        out = model(*(getattr(b, k) for k in INPUT_NAMES))
        scores.append(torch.sigmoid(out.float()).cpu().numpy())
        labels.append(b.label.cpu().numpy())
        seen += b.batch_size()
        if max_samples and seen >= max_samples:
            break
    return auc_score(np.concatenate(labels), np.concatenate(scores))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-val", type=int, default=16384)
    ap.add_argument("--skip-loo", action="store_true")
    args = ap.parse_args()

    cfg = load_config("config.yaml", __file__)
    q = cfg["quant"]
    device = get_device("cuda")

    data_path = resolve(cfg, cfg["data"]["data_path"])
    val_set = load_split(data_path, "val")
    calib_set = load_split(data_path, "calib")

    ckpt = torch.load(resolve(cfg, cfg["train"]["save_path"]), map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    names = A.list_linear_names(model)
    print(f"[sensitivity] 可量化 Linear 层: {len(names)}")
    for n in names:
        print(f"    - {n}")

    print("[sensitivity] 用 torch.ao 观察器校准一次 ...")
    adt, aqs, wdt, wqs = A.resolve_dtypes(
        q["act_dtype"], q["act_qscheme"], q["w_dtype"], q["w_qscheme"])
    qp = A.compute_qparams(
        model,
        calib_iter=({k: getattr(b, k).to(device) for k in INPUT_NAMES}
                    for b in iter_batches(calib_set, q["calib_batch_size"])),
        input_names=INPUT_NAMES,
        act_dtype=adt, act_qscheme=aqs, w_dtype=wdt, w_qscheme=wqs,
        reduce_range=False,
    )
    wrappers = A.wrap_model(model, qp)

    A.set_enabled(wrappers, enabled=set())          # 全 FP32
    auc_fp32 = eval_auc(model, val_set, device, args.max_val)
    print(f"[sensitivity] 基线 AUC(FP32) = {auc_fp32:.5f}")

    rows = []
    print("[sensitivity] A) 逐层单独量化 ...")
    for i, name in enumerate(names):
        A.set_enabled(wrappers, enabled={name})
        auc = eval_auc(model, val_set, device, args.max_val)
        drop = auc_fp32 - auc
        rows.append({"layer": name, "auc_only": auc, "delta_only": drop,
                     "auc_loo": float("nan"), "delta_loo": float("nan")})
        print(f"  [{i + 1:3d}/{len(names)}] {name:16s} AUC={auc:.5f} Δ={drop:+.5f}"
              f"{'  敏感' if drop > q['sensitive_auc_drop'] else ''}")

    if not args.skip_loo:
        print("[sensitivity] B) 全量化后逐层恢复 ...")
        A.set_enabled(wrappers, enabled=None)
        auc_all = eval_auc(model, val_set, device, args.max_val)
        print(f"  全量化 AUC = {auc_all:.5f} (Δ={auc_all - auc_fp32:+.5f})")
        for i, name in enumerate(names):
            A.set_enabled(wrappers, enabled=None, disabled={name})
            auc = eval_auc(model, val_set, device, args.max_val)
            rows[i]["auc_loo"] = auc
            rows[i]["delta_loo"] = auc - auc_all
            print(f"  [{i + 1:3d}/{len(names)}] 恢复 {name:16s} -> AUC={auc:.5f} "
                  f"回血={auc - auc_all:+.5f}")

    sensitive = []
    for r in rows:
        hot = r["delta_only"] > q["sensitive_auc_drop"]
        if not np.isnan(r["delta_loo"]) and r["delta_loo"] > q["sensitive_auc_drop"]:
            hot = True
        r["verdict"] = "sensitive" if hot else "quantizable"
        if hot:
            sensitive.append(r["layer"])

    report = resolve(cfg, q["report_path"])
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "auc_only", "delta_only",
                                         "auc_loo", "delta_loo", "verdict"])
        w.writeheader()
        w.writerows(rows)
    print(f"[sensitivity] 明细: {report}")

    sen = resolve(cfg, q["sensitive_path"])
    sen.write_text("\n".join(sensitive) + ("\n" if sensitive else ""), encoding="utf-8")
    print(f"[sensitivity] 敏感层 {len(sensitive)}/{len(names)} -> {sen}")
    for s in sensitive:
        print(f"    - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
