#!/usr/bin/env python3
"""③ ONNX → TensorRT engine（fp32 / int8）。

    fp32 : 用未量化的 model.onnx
    int8 : 用带 Q/DQ 的 model_quant_qdq.onnx（显式量化，无需校准缓存）

用法:
    python3 build_engine.py              # 两个都建
    python3 build_engine.py --only int8
    python3 build_engine.py --verbose    # 打印 TRT 详细日志（看哪些层真的用了 INT8）
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

    # 每个输入的「非 batch」维度；batch 维由 min/opt/max 自动加在最前面
    shapes = {
        "user_id": (),
        "item_id": (),
        "cate_id": (),
        "history": (d["history_len"],),
        "dense": (d["num_dense"],),
    }
    common = dict(
        shapes=shapes,
        min_batch=e["min_batch"],
        opt_batch=e["opt_batch"],
        max_batch=e["max_batch"],
        verbose=args.verbose,
    )

    out_dir = resolve(cfg, "outputs/")
    jobs = []
    if args.only in (None, "fp32"):
        jobs.append(("fp32", resolve(cfg, e["onnx_path"]), out_dir / "model_fp32.plan"))
    if args.only in (None, "int8"):
        jobs.append(("int8", resolve(cfg, e["quant_onnx_path"]), out_dir / "model_int8.plan"))

    for precision, src, dst in jobs:
        if not src.exists():
            print(f"[build] 跳过 {precision}: 找不到 {src}")
            if precision == "int8":
                print("       先跑 python3 quantize_qdq.py 生成量化 ONNX")
            else:
                print("       先跑 python3 export_onnx.py 生成 ONNX")
            continue
        build_engine(str(src), str(dst), precision=precision, **common)

    print("[build] 全部完成，产物在 outputs/*.plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
