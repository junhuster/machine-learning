#!/usr/bin/env python3
"""④ 插 Q/DQ 并导出量化的 ONNX。

流程：
    载入 FP32 权重 -> 包装 Linear -> 校准(amax) -> 按敏感层名单跳过若干层
    -> 打开量化 -> 评估 ΔAUC -> 导出带 Q/DQ 的 ONNX

用法:
    python3 quantize_qdq.py                 # 用 sensitivity.py 的结果自动跳过敏感层
    python3 quantize_qdq.py --all-layers    # 不跳过任何层（全量化，做对照）
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
import qdq_lib as Q  # noqa: E402


@torch.no_grad()
def evaluate_auc(model, bundle, device, max_samples: int = 16384) -> float:
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


def load_disabled(cfg, all_layers: bool) -> set:
    """敏感层名单 = sensitivity.py 的产物 ∪ config 里手工指定的 disable_list。"""
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
    ap.add_argument("--all-layers", action="store_true", help="不跳过敏感层（全量化对照）")
    args = ap.parse_args()

    cfg = load_config("config.yaml", __file__)
    q, e = cfg["quant"], cfg["export"]
    device = get_device("cuda")

    data_path = resolve(cfg, cfg["data"]["data_path"])
    val_set = load_split(data_path, "val")
    calib_set = load_split(data_path, "calib")

    ckpt = torch.load(resolve(cfg, cfg["train"]["save_path"]), map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    wrappers = Q.wrap_model(model, method=q["calib_method"])
    disabled = load_disabled(cfg, args.all_layers)
    known = {w.name for w in wrappers}
    unknown = disabled - known
    if unknown:
        print(f"[quantize] ⚠️ config/sensitive 里有 {len(unknown)} 个名字不在模型里，已忽略: "
              f"{sorted(unknown)[:3]} ...")
        disabled &= known

    print(f"[quantize] Linear 层 {len(known)} 个，其中跳过 {len(disabled)} 个")

    auc_fp32 = evaluate_auc(model, val_set, device)
    print(f"[quantize] 基线 AUC(FP32) = {auc_fp32:.5f}")

    # ---- 校准 ----
    print("[quantize] 校准（统计 amax）...")
    Q.calibrate(
        model, wrappers,
        calib_iter=(b.to(device).inputs(INPUT_NAMES)
                    for b in iter_batches(calib_set, q["calib_batch_size"])),
        disabled=disabled,
    )

    # ---- 估算量化后精度（伪量化，快）----
    Q.set_modes(wrappers, "quant", disabled=disabled)
    auc_q = evaluate_auc(model, val_set, device)
    print(f"[quantize] AUC(INT8 伪量化) = {auc_q:.5f}   ΔAUC = {auc_q - auc_fp32:+.5f}")

    # ---- 导出 ONNX（Q/DQ 节点）----
    out_path = resolve(cfg, e["quant_onnx_path"])
    Q.set_modes(wrappers, "export", disabled=disabled)

    L, D = cfg["data"]["history_len"], cfg["data"]["num_dense"]
    b = 2
    dummy = (
        torch.zeros(b, dtype=torch.long, device=device),
        torch.zeros(b, dtype=torch.long, device=device),
        torch.zeros(b, dtype=torch.long, device=device),
        torch.zeros(b, L, dtype=torch.long, device=device),
        torch.zeros(b, D, dtype=torch.float32, device=device),
    )
    print(f"[quantize] 导出量化 ONNX: {out_path}")
    Q.export_qdq_onnx(model, dummy, str(out_path), INPUT_NAMES, OUTPUT_NAMES, e["opset"],
                      simplify=bool(q.get("simplify", False)))

    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"[quantize] 完成 | {size_mb:.1f} MB")

    # ---- 导出后自检：给三个数字，便于判断导出是否可信 ----
    try:
        import onnxruntime as ort

        probe = (
            torch.randint(0, cfg["data"]["num_users"], (32,), device=device),
            torch.randint(0, cfg["data"]["num_items"], (32,), device=device),
            torch.randint(0, cfg["data"]["num_categories"], (32,), device=device),
            torch.randint(0, cfg["data"]["num_items"], (32, L), device=device),
            torch.randn(32, D, device=device),
        )
        with torch.no_grad():
            y_int8 = model(*probe).float().cpu().numpy()      # export 模式 = 图里的语义
        Q.set_modes(wrappers, "off")
        with torch.no_grad():
            y_fp32 = model(*probe).float().cpu().numpy()
        Q.set_modes(wrappers, "export", disabled=disabled)

        # 注意：ONNX Runtime 的「全量图优化」会错误处理本图的 Q→DQ 模式
        # （实测误差 2.4e-01）；用 BASIC 才与 torch 一致（3.6e-07）。
        # 这是 ORT 的已知优化问题，与导出的图本身无关 —— TRT 按显式量化规范解析，不受影响。
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

    Q.set_modes(wrappers, "off")
    print(f"[quantize] 提示：拿这份 ONNX 去 build_engine.py 建 int8 engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
