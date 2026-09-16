#!/usr/bin/env python3
"""数据读取与批处理：两个任务共用。

约定：npz 里按 `{split}_{field}` 命名，split ∈ {train, val, calib}。
本模块把某个 split 读成 torch 张量，并按 batch 迭代。
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch


@dataclass
class FeatureBundle:
    """一个 batch 的特征。不同模型的字段子集不同，缺的字段为 None。"""

    user_id: Optional[torch.Tensor] = None
    item_id: Optional[torch.Tensor] = None
    cate_id: Optional[torch.Tensor] = None
    history: Optional[torch.Tensor] = None
    dense: Optional[torch.Tensor] = None
    sparse: Optional[torch.Tensor] = None   # [B, num_sparse] 多列类别特征（任务2/DLRM 用）
    label: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "FeatureBundle":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = None if v is None else v.to(device, non_blocking=True)
        return FeatureBundle(**kw)

    def inputs(self, names: List[str]) -> Dict[str, torch.Tensor]:
        """按名字取模型输入（不含 label）。"""
        return {n: getattr(self, n) for n in names if getattr(self, n) is not None}

    def batch_size(self) -> int:
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None and f.name != "label":
                return int(v.shape[0])
        raise ValueError("空 batch")


_ID_FIELDS = ("user_id", "item_id", "cate_id", "sparse")
_FLOAT_FIELDS = ("dense", "label")


def _cast(name: str, arr: np.ndarray) -> torch.Tensor:
    base = name.split("_", 1)[1]
    if base in _ID_FIELDS:
        return torch.from_numpy(np.ascontiguousarray(arr).astype(np.int64))
    if base == "history":
        return torch.from_numpy(np.ascontiguousarray(arr).astype(np.int64))
    if base in _FLOAT_FIELDS:
        return torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32))
    raise KeyError(f"未知字段: {name}")


def load_split(npz_path: str | Path, split: str) -> FeatureBundle:
    """读入整个 split（全量放 GPU 显存，本项目数据量小，够用）。"""
    data = np.load(npz_path)
    prefix = f"{split}_"
    kw: Dict[str, torch.Tensor] = {}
    for key in data.files:
        if key.startswith(prefix):
            kw[key[len(prefix):]] = _cast(key, data[key])
    if not kw:
        raise KeyError(f"npz 里没有 {split}_* 字段，可用: {data.files[:6]} ...")
    # data 里的字段名必须都在 FeatureBundle 里
    valid = {f.name for f in fields(FeatureBundle)}
    unknown = set(kw) - valid
    if unknown:
        raise KeyError(f"FeatureBundle 不认识的字段: {unknown}")
    return FeatureBundle(**kw)


def iter_batches(
    bundle: FeatureBundle,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
) -> Iterator[FeatureBundle]:
    n = bundle.batch_size()
    order = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, n, batch_size):
        sel = order[start : start + batch_size]
        if drop_last and sel.shape[0] < batch_size:
            continue
        kw = {}
        for f in fields(FeatureBundle):
            v = getattr(bundle, f.name)
            kw[f.name] = None if v is None else v[torch.from_numpy(sel)]
        yield FeatureBundle(**kw)
