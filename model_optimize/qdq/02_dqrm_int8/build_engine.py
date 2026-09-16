#!/usr/bin/env python3
"""③ ONNX → TensorRT engine（fp32 / int8）。

用法:
    python3 build_engine.py
    python3 build_engine.py --only int8 --verbose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.conf import load_config, resolve  # noqa: E402
from common.trt_util import build_engine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fp32", "int8"], default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config("config.yaml", __file__)
    e, d = cfg["export"], cfg["data"]

    shapes = {
        "dense": (d["num_dense"],),
        "sparse": (d["num_sparse"],),
    }
    common = dict(shapes=shapes, min_batch=e["min_batch"], opt_batch=e["opt_batch"],
                  max_batch=e["max_batch"], verbose=args.verbose)

    out_dir = resolve(cfg, "outputs/")
    jobs = []
    if args.only in (None, "fp32"):
        jobs.append(("fp32", resolve(cfg, e["onnx_path"]), out_dir / "model_fp32.plan"))
    if args.only in (None, "int8"):
        jobs.append(("int8", resolve(cfg, e["quant_onnx_path"]), out_dir / "model_int8.plan"))

    for precision, src, dst in jobs:
        if not src.exists():
            print(f"[build] 跳过 {precision}: 找不到 {src}")
            print("       先跑 export_onnx.py / quantize_qdq.py")
            continue
        build_engine(str(src), str(dst), precision=precision, **common)

    print("[build] 完成，产物在 outputs/*.plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
