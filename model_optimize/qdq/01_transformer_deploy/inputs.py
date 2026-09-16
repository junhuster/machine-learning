#!/usr/bin/env python3
"""构造 TensorRT 推理输入。

两种来源：
  val    : 从验证集取真实分布样本（推荐——数值分布和训练一致，量化误差才有代表性）
  random : 随机生成（只为了压测形状，不关心分布）
"""
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
        n = min(batch_size, bundle.batch_size())
        if n < batch_size:
            raise ValueError(f"验证集只有 {n} 条，凑不够 batch={batch_size}")
        kw = {}
        for name in INPUT_NAMES:
            kw[name] = getattr(bundle, name)[:batch_size].to(device)
        labels = bundle.label[:batch_size].numpy()
        return kw, labels

    if source == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        kw = {
            "user_id": torch.randint(0, d["num_users"], (batch_size,), generator=g).to(device),
            "item_id": torch.randint(0, d["num_items"], (batch_size,), generator=g).to(device),
            "cate_id": torch.randint(0, d["num_categories"], (batch_size,), generator=g).to(device),
            "history": torch.randint(
                0, d["num_items"], (batch_size, d["history_len"]), generator=g
            ).to(device),
            "dense": torch.randn(batch_size, d["num_dense"], generator=g).to(device),
        }
        return kw, None

    raise ValueError(f"未知 source: {source}")
