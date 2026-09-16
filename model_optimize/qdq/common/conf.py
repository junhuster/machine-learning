#!/usr/bin/env python3
"""小工具：路径、随机种子、config 加载。所有脚本统一从这里取，保证行为一致。"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def project_dir() -> Path:
    """返回调用脚本所在目录（各子项目根目录）。"""
    import inspect

    frame = inspect.stack()[2]
    return Path(frame.filename).resolve().parent


def load_config(path: str | os.PathLike, script_file: str) -> Dict[str, Any]:
    """相对脚本目录解析 config 路径。"""
    base = Path(script_file).resolve().parent
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(p)
    cfg["_base_dir"] = str(base)
    return cfg


def resolve(cfg: Dict[str, Any], rel: str) -> Path:
    """把 config 里的相对路径解析成绝对路径，并确保父目录存在。"""
    p = Path(rel)
    if not p.is_absolute():
        p = Path(cfg["_base_dir"]) / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass


def get_device(name: str = "cuda"):
    import torch

    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回落到 CPU（TRT 相关步骤仍需 GPU）")
        return torch.device("cpu")
    return torch.device(name)
