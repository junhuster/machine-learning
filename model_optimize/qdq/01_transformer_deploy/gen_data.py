#!/usr/bin/env python3
"""① 生成 fake 训练数据并落盘。

为什么要造「有学习信号」的数据：如果 label 纯随机，AUC 恒等于 0.5，
量化前后的 ΔAUC 全是噪声，根本判断不了敏感层。所以这里用一组固定的
隐因子模型来造 label：

    logit  = <w_dense, x_dense>                      # 数值特征线性项
           + <U[user_id], V[item_id]> * a            # 用户-物品交互（emb 能学到）
           + b_cate[cate_id]                         # 类目偏置
           + <V[item_id], mean(V[history])> * c      # 序列偏好（attention 能学到）
    label ~ Bernoulli(sigmoid(logit))

这样 AUC 能稳定在 0.75 上下，量化造成的精度损失才可测。

输出：outputs/fake_data.npz
    train_/val_/calib_ 各前缀一份：
      user_id[int32] item_id[int32] cate_id[int32] history[int32,B,L] dense[float32,B,D] label[float32,B]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.conf import load_config, resolve, set_seed  # noqa: E402

HERE = Path(__file__).resolve().parent


def make_latent_tables(rng: np.random.Generator, n: int, dim: int, scale: float) -> np.ndarray:
    return (rng.standard_normal((n, dim)) * scale).astype(np.float32)


def main() -> int:
    cfg = load_config("config.yaml", __file__)
    d = cfg["data"]
    seed = d["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)

    n = d["num_samples"]
    L = d["history_len"]
    D = d["num_dense"]

    print(f"[gen_data] 样本 {n} | history_len {L} | dense {D} | vocab "
          f"user={d['num_users']} item={d['num_items']} cate={d['num_categories']}")

    # ---- 隐因子与权重（固定，保证可复现且模型可学）----
    K = 16
    U = make_latent_tables(rng, d["num_users"], K, 0.6)     # 用户隐因子
    V = make_latent_tables(rng, d["num_items"], K, 0.6)     # 物品隐因子
    w_dense = (rng.standard_normal(D) / np.sqrt(D)).astype(np.float32)
    b_cate = (rng.standard_normal(d["num_categories"]) * 0.2).astype(np.float32)
    a, c = 1.1, 0.8

    # ---- 特征 ----
    user_id = rng.integers(0, d["num_users"], size=n)
    item_id = rng.integers(0, d["num_items"], size=n)
    cate_id = rng.integers(0, d["num_categories"], size=n)
    history = rng.integers(0, d["num_items"], size=(n, L))
    dense = rng.standard_normal((n, D)).astype(np.float32)

    # ---- label ----
    hist_fac = V[history].mean(axis=1)                      # [n, K]
    logit = (
        dense @ w_dense
        + (U[user_id] * V[item_id]).sum(axis=1) * a
        + b_cate[cate_id]
        + (V[item_id] * hist_fac).sum(axis=1) * c
        + rng.standard_normal(n).astype(np.float32) * 0.5   # 噪声
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    label = (rng.random(n) < prob).astype(np.float32)

    # ---- 切分：train / val / calib ----
    n_val = int(n * d["val_ratio"])
    n_calib = int(n * d["calib_ratio"])
    idx = rng.permutation(n)
    val_idx = idx[:n_val]
    calib_idx = idx[n_val : n_val + n_calib]
    train_idx = idx[n_val + n_calib :]

    print(f"[gen_data] train={len(train_idx)} val={len(val_idx)} calib={len(calib_idx)} "
          f"| 正样本率={label.mean():.3f}")

    arrays = {}
    for prefix, sel in (("train", train_idx), ("val", val_idx), ("calib", calib_idx)):
        arrays[f"{prefix}_user_id"] = user_id[sel].astype(np.int32)
        arrays[f"{prefix}_item_id"] = item_id[sel].astype(np.int32)
        arrays[f"{prefix}_cate_id"] = cate_id[sel].astype(np.int32)
        arrays[f"{prefix}_history"] = history[sel].astype(np.int32)
        arrays[f"{prefix}_dense"] = dense[sel].astype(np.float32)
        arrays[f"{prefix}_label"] = label[sel].astype(np.float32)

    out = resolve(cfg, d["data_path"])
    np.savez_compressed(out, **arrays)
    print(f"[gen_data] 已保存: {out}  ({out.stat().st_size / 1024 ** 2:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
