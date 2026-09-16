#!/usr/bin/env python3
"""③ torch → ONNX（FP32，动态 batch）。

用法:
    python3 export_onnx.py
    python3 export_onnx.py --check     # 用 onnxruntime 与 torch 对拍
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
from model import INPUT_NAMES, OUTPUT_NAMES, build_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cfg = load_config("config.yaml", __file__)
    d, e = cfg["data"], cfg["export"]
    device = get_device("cuda") if args.check else torch.device("cpu")

    ckpt = torch.load(resolve(cfg, cfg["train"]["save_path"]), map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    b = 2
    dummy = (
        torch.randn(b, d["num_dense"], device=device),
        torch.zeros(b, d["num_sparse"], dtype=torch.long, device=device),
    )
    out_path = resolve(cfg, e["onnx_path"])
    dynamic_axes = {n: {0: "batch"} for n in INPUT_NAMES + OUTPUT_NAMES}

    print(f"[export] 导出 ONNX: {out_path}")
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic_axes, opset_version=e["opset"],
        do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL,
    )

    import onnx

    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    print(f"[export] 校验通过 | {out_path.stat().st_size / 1024 ** 2:.1f} MB | opset={e['opset']}")
    for inp in m.graph.input:
        print(f"  IN   {inp.name:10s} {[x.dim_param or x.dim_value for x in inp.type.tensor_type.shape.dim]}")
    for o in m.graph.output:
        print(f"  OUT  {o.name:10s} {[x.dim_param or x.dim_value for x in o.type.tensor_type.shape.dim]}")

    if args.check:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path),
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        np_in = {k: v.cpu().numpy() for k, v in zip(INPUT_NAMES, dummy)}
        ort_out = sess.run(OUTPUT_NAMES, np_in)[0]
        with torch.no_grad():
            ref = model(*dummy).cpu().numpy()
        diff = float(np.abs(ort_out - ref).max())
        print(f"[export] onnxruntime vs torch 最大绝对误差 = {diff:.3e} "
              f"({'OK' if diff < 1e-4 else '⚠️ 偏大'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
