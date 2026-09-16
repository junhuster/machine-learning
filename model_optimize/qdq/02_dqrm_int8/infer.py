#!/usr/bin/env python3
"""⑤ 用 TensorRT engine 推理（任务2：dense + sparse 两路输入）。

用法:
    python3 infer.py --engine outputs/model_int8.plan --batch 32
    python3 infer.py --engine outputs/model_int8.plan --batch 32 --compare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.conf import get_device, load_config, resolve  # noqa: E402
from common.trt_util import TRTEngine  # noqa: E402
from inputs import make_inputs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--source", choices=["val", "random"], default="val")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    cfg = load_config("config.yaml", __file__)
    device = get_device("cuda")

    engine_path = Path(args.engine)
    if not engine_path.is_absolute():
        engine_path = HERE / engine_path
    if not engine_path.exists():
        raise SystemExit(f"找不到 engine: {engine_path}")

    eng = TRTEngine(str(engine_path))
    print(eng.describe())

    inputs, labels = make_inputs(cfg, args.batch, device, source=args.source)
    print("\n输入形状: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in inputs.items()))

    out = eng.infer(inputs)
    logit = out["logit"].float()
    print(f"\n输出 logit 形状 = {tuple(logit.shape)}")
    print(f"  前 8 个 prob : {[round(x, 4) for x in torch.sigmoid(logit)[:8].tolist()]}")
    print(f"  均值/最大/最小: {logit.mean():.4f} / {logit.max():.4f} / {logit.min():.4f}")
    if labels is not None:
        print(f"  真实标签正例率: {labels.mean():.3f}")

    if args.compare:
        from common.metrics import auc_score
        from model import INPUT_NAMES, build_model

        ckpt = torch.load(resolve(cfg, cfg["train"]["save_path"]), map_location="cpu")
        model = build_model(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)
        with torch.no_grad():
            ref = model(*(inputs[k] for k in INPUT_NAMES))
        diff = (ref - logit).abs()
        print("\n[compare] torch vs TRT:")
        print(f"  最大绝对误差 = {float(diff.max()):.4e} | 平均 = {float(diff.mean()):.4e}")
        if labels is not None:
            print(f"  AUC(torch) = {auc_score(labels, ref.cpu().numpy()):.5f}")
            print(f"  AUC(TRT)   = {auc_score(labels, logit.cpu().numpy()):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
