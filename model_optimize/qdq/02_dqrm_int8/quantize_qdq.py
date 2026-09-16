#!/usr/bin/env python3
"""④ 训练后量化（PTQ）：用 torch.ao 观察器求量化参数 -> 插 Q/DQ -> 导出 ONNX。

用法:
    python3 quantize_qdq.py                 # 按敏感度名单跳过敏感层
    python3 quantize_qdq.py --all-layers    # 不跳过（全量化对照）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.conf import get_device, load_config, resolve  # noqa: E402
from common.dataio import iter_batches, load_split  # noqa: E402
from common.metrics import auc_score  # noqa: E402
from model import INPUT_NAMES, OUTPUT_NAMES, build_model  # noqa: E402
import ao_quant as A  # noqa: E402


@torch.no_grad()
def eval_auc(model, val_set, device, max_samples: int = 16384) -> float:
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


def load_disabled(cfg, all_layers: bool) -> set:
    disabled = set(cfg["quant"].get("disable_list") or [])
    if all_layers or not cfg["quant"].get("skip_sensitive", True):
        print("[quantize] 关闭敏感层跳过，全部 Linear 都量化")
        return disabled
    p = resolve(cfg, cfg["quant"]["sensitive_path"])
    if p.exists():
        names = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
        disabled |= names
        print(f"[quantize] 从 {p.name} 读入 {len(names)} 个敏感层（不量化）")
    else:
        print(f"[quantize] 未找到 {p.name}，先跑 python3 sensitivity.py 可自动跳过敏感层")
    return disabled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-layers", action="store_true")
    args = ap.parse_args()

    cfg = load_config("config.yaml", __file__)
    q, d = cfg["quant"], cfg["data"]
    device = get_device("cuda")

    data_path = resolve(cfg, cfg["data"]["data_path"])
    calib_set = load_split(data_path, "calib")
    val_set = load_split(data_path, "val")

    ckpt = torch.load(resolve(cfg, cfg["train"]["save_path"]), map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    all_names = set(A.list_linear_names(model))
    disabled = load_disabled(cfg, args.all_layers) & all_names

    auc_fp32 = eval_auc(model, val_set, device, max_samples=8192)

    print("[quantize] PTQ：用 torch.ao 观察器校准求量化参数 ...")
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
    for n in sorted(qp):
        mark = "跳过" if n in disabled else "量化"
        print(f"    [{mark}] {n:16s} {qp[n]}")

    wrappers = A.wrap_model(model, qp)
    A.set_enabled(wrappers, enabled=None, disabled=disabled)
    n_on = sum(1 for w in wrappers if w.enabled)
    print(f"[quantize] 共 {len(qp)} 层可量化，实际开启 {n_on} 层")

    auc_q = eval_auc(model, val_set, device)
    print(f"[quantize] 基线 AUC(FP32) = {auc_fp32:.5f}")
    print(f"[quantize] AUC(INT8)      = {auc_q:.5f}   ΔAUC = {auc_q - auc_fp32:+.5f}")

    out_path = resolve(cfg, cfg["export"]["quant_onnx_path"])
    ex = [
        torch.randn(8, d["num_dense"], device=device),
        torch.zeros(8, d["num_sparse"], dtype=torch.long, device=device),
    ]
    print(f"[quantize] 导出量化 ONNX: {out_path}")
    A.export_qdq_onnx(model, ex, str(out_path), INPUT_NAMES, OUTPUT_NAMES, cfg["export"]["opset"])
    print(f"[quantize] 完成 | {out_path.stat().st_size / 1024 ** 2:.1f} MB")

    # ---- 导出后自检：给三个数字，便于判断导出是否可信 ----
    try:
        import onnxruntime as ort

        probe = [
            torch.randn(32, d["num_dense"], device=device),
            torch.randint(0, d["sparse_vocab"], (32, d["num_sparse"]), device=device),
        ]
        A.set_enabled(wrappers, enabled=None, disabled=disabled)
        with torch.no_grad():
            y_int8 = model(*probe).float().cpu().numpy()
        A.set_enabled(wrappers, enabled=set())
        with torch.no_grad():
            y_fp32 = model(*probe).float().cpu().numpy()
        A.set_enabled(wrappers, enabled=None, disabled=disabled)

        # 注意：ONNX Runtime 的「全量图优化」会错误处理 Q→DQ 模式，必须用 BASIC
        # 才能与 torch 一致（这是 ORT 的优化问题，与导出的图无关）。
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess = ort.InferenceSession(str(out_path), so,
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        y_ort = sess.run([OUTPUT_NAMES[0]],
                         {k: v.cpu().numpy() for k, v in zip(INPUT_NAMES, probe)})[0]

        d_quant = float(np.abs(y_fp32 - y_int8).max())
        d_export = float(np.abs(y_int8 - y_ort).max())
        print(f"[quantize] 自检 ① logit 量级 |max| = {float(np.abs(y_fp32).max()):.3f}")
        print(f"[quantize] 自检 ② FP32 -> INT8 差异 = {d_quant:.3e}   (量化本身的代价)")
        print(f"[quantize] 自检 ③ INT8 -> ONNX 差异 = {d_export:.3e}   (导出是否忠实)")
        print("[quantize] 判读：③ 应显著小于 ②；若 ③ 与 ② 同量级，导出可能有问题")
        print("[quantize] 权威精度以 bench_compare.py 里 TRT engine 实测 AUC 为准")
    except ImportError:
        print("[quantize] 未安装 onnxruntime，跳过导出自检（pip install onnxruntime）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
