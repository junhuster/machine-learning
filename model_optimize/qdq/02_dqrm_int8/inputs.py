#!/usr/bin/env python3
"""构造 TensorRT 推理输入（任务2：dense + sparse 两路输入）。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from common.dataio import load_split
from model import INPUT_NAMES


def make_inputs(
    cfg: dict,
    batch_size: int,
    device: torch.device,
    source: str = "val",
    data_path: Optional[str] = None,
    seed: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Optional[np.ndarray]]:
    d = cfg["data"]
    if source == "val":
        path = data_path or str(Path(cfg["_base_dir"]) / d["data_path"])
        bundle = load_split(path, "val")
        if bundle.batch_size() < batch_size:
            raise ValueError(f"验证集只有 {bundle.batch_size()} 条，凑不够 batch={batch_size}")
        inputs = {name: getattr(bundle, name)[:batch_size].to(device) for name in INPUT_NAMES}
        return inputs, bundle.label[:batch_size].numpy()

    if source == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        inputs = {
            "dense": torch.randn(batch_size, d["num_dense"], generator=g).to(device),
            "sparse": torch.randint(
                0, d["sparse_vocab"], (batch_size, d["num_sparse"]), generator=g
            ).to(device),
        }
        return inputs, None

    raise ValueError(f"未知 source: {source}")
