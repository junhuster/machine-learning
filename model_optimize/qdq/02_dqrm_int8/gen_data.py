#!/usr/bin/env python3
"""① 生成 fake 训练数据并落盘（DQRM/DLRM 版）。

label 同样用隐因子模型造，保证 AUC 有意义（否则 ΔAUC 全是噪声）：
    logit = <w, dense>
          + Σ_j <E_j[sparse_j], v> * a      # 每列类别特征对同一组隐因子贡献
          + 噪声
    label ~ Bernoulli(sigmoid(logit))

输出：outputs/fake_data.npz
    train_/val_/calib_ 各前缀一份：dense[B,D] float32 | sparse[B,C] int32 | label[B] float32
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.conf import load_config, resolve, set_seed  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    cfg = load_config("config.yaml", __file__)
    d = cfg["data"]
    seed = d["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)

    n, D, C, V = d["num_samples"], d["num_dense"], d["num_sparse"], d["sparse_vocab"]
    print(f"[gen_data] 样本 {n} | dense {D} | sparse {C} 列 x vocab {V}")

    w_dense = (rng.standard_normal(D) / np.sqrt(D)).astype(np.float32)
    share = (rng.standard_normal(16) * 0.7).astype(np.float32)          # 共享隐因子方向
    tables = (rng.standard_normal((C, V, 16)) * 0.5).astype(np.float32)  # 每列一张隐因子表

    dense = rng.standard_normal((n, D)).astype(np.float32)
    sparse = rng.integers(0, V, size=(n, C)).astype(np.int32)

    acc = np.zeros(n, dtype=np.float32)
    for j in range(C):
        acc += (tables[j][sparse[:, j]] * share).sum(axis=1) * 0.35
    logit = dense @ w_dense + acc + rng.standard_normal(n).astype(np.float32) * 0.5
    prob = 1.0 / (1.0 + np.exp(-logit))
    label = (rng.random(n) < prob).astype(np.float32)

    n_val = int(n * d["val_ratio"])
    n_calib = int(n * d["calib_ratio"])
    idx = rng.permutation(n)
    val_idx, calib_idx, train_idx = idx[:n_val], idx[n_val : n_val + n_calib], idx[n_val + n_calib :]
    print(f"[gen_data] train={len(train_idx)} val={len(val_idx)} calib={len(calib_idx)} "
          f"| 正样本率={label.mean():.3f}")

    arrays = {}
    for prefix, sel in (("train", train_idx), ("val", val_idx), ("calib", calib_idx)):
        arrays[f"{prefix}_dense"] = dense[sel]
        arrays[f"{prefix}_sparse"] = sparse[sel]
        arrays[f"{prefix}_label"] = label[sel]

    out = resolve(cfg, d["data_path"])
    np.savez_compressed(out, **arrays)
    print(f"[gen_data] 已保存: {out}  ({out.stat().st_size / 1024 ** 2:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
