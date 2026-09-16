#!/usr/bin/env python3
"""③ torch → ONNX（动态 batch）。

用法:
    python3 export_onnx.py
    python3 export_onnx.py --check     # 额外用 onnxruntime 与 torch 对拍
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
from model import build_model  # noqa: E402

INPUT_NAMES = ["user_id", "item_id", "cate_id", "history", "dense"]
OUTPUT_NAMES = ["logit"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="用 onnxruntime 与 torch 对拍")
    args = ap.parse_args()

    cfg = load_config("config.yaml", __file__)
    d, e = cfg["data"], cfg["export"]
    device = get_device("cuda") if args.check else torch.device("cpu")

    ckpt = torch.load(resolve(cfg, cfg["train"]["save_path"]), map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    L, D = d["history_len"], d["num_dense"]
    b = 2
    dummy = (
        torch.zeros(b, dtype=torch.long, device=device),
        torch.zeros(b, dtype=torch.long, device=device),
        torch.zeros(b, dtype=torch.long, device=device),
        torch.zeros(b, L, dtype=torch.long, device=device),
        torch.zeros(b, D, dtype=torch.float32, device=device),
    )

    out_path = resolve(cfg, e["onnx_path"])
    dynamic_axes = {name: {0: "batch"} for name in INPUT_NAMES}
    dynamic_axes.update({name: {0: "batch"} for name in OUTPUT_NAMES})

    print(f"[export] 导出 ONNX: {out_path}")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic_axes,
        opset_version=e["opset"],
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )

    import onnx

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"[export] 校验通过 | {size_mb:.1f} MB | opset={e['opset']}")
    for inp in onnx_model.graph.input:
        dims = [dim.dim_param or dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        print(f"  IN   {inp.name:12s} {dims}")
    for o in onnx_model.graph.output:
        dims = [dim.dim_param or dim.dim_value for dim in o.type.tensor_type.shape.dim]
        print(f"  OUT  {o.name:12s} {dims}")

    if args.check:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        np_inputs = {k: v.cpu().numpy() for k, v in zip(INPUT_NAMES, dummy)}
        ort_out = sess.run(OUTPUT_NAMES, np_inputs)[0]
        with torch.no_grad():
            torch_out = model(*dummy).cpu().numpy()
        diff = float(np.abs(ort_out - torch_out).max())
        print(f"[export] onnxruntime vs torch 最大绝对误差 = {diff:.3e} "
              f"({'OK' if diff < 1e-4 else '⚠️ 偏大，检查一下'})")

    print("[export] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
