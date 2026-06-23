"""
data.py — 假数据生成模块
生成搜推场景下的稀疏特征 + 连续特征 + 二分类标签数据
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)


class RecDataset(Dataset):
    """
    搜推场景假数据集。
    每条样本包含：
      - sparse_fields: (num_sparse_fields,) int64  稀疏特征 id
      - dense_features: (num_dense_features,) float32  连续特征
      - label_ctr: float32  点击标签
      - label_cvr: float32  转化标签
    """

    def __init__(self, num_samples, num_sparse_fields, sparse_vocab_size,
                 num_dense_features, seed=42):
        rng = np.random.default_rng(seed)
        self.sparse = torch.from_numpy(
            rng.integers(0, sparse_vocab_size,
                         size=(num_samples, num_sparse_fields))
        ).long()
        self.dense = torch.from_numpy(
            rng.standard_normal((num_samples, num_dense_features)).astype(np.float32)
        )
        # 用简单线性关系 + 噪声生成标签，保证 AUC 有意义
        logit = (self.dense[:, :3].sum(dim=1) +
                 (self.sparse[:, 0] % 10).float() * 0.1)
        prob_ctr = torch.sigmoid(logit + torch.randn(num_samples) * 0.5)
        prob_cvr = torch.sigmoid(logit * 0.8 + torch.randn(num_samples) * 0.5)
        self.label_ctr = (prob_ctr > 0.5).float()
        self.label_cvr = (prob_cvr > 0.5).float()

        logger.debug("Dataset created: %d samples, sparse_fields=%d, dense_features=%d",
                     num_samples, num_sparse_fields, num_dense_features)

    def __len__(self):
        return len(self.label_ctr)

    def __getitem__(self, idx):
        return {
            "sparse": self.sparse[idx],
            "dense":  self.dense[idx],
            "label_ctr": self.label_ctr[idx],
            "label_cvr": self.label_cvr[idx],
        }


def build_dataloaders(cfg):
    """根据配置构建训练集和验证集 DataLoader。"""
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    logger.info("Building datasets: train=%d  val=%d",
                data_cfg["num_train_samples"], data_cfg["num_val_samples"])

    common_kw = dict(
        num_sparse_fields=data_cfg["num_sparse_fields"],
        sparse_vocab_size=data_cfg["sparse_vocab_size"],
        num_dense_features=data_cfg["num_dense_features"],
    )
    train_ds = RecDataset(data_cfg["num_train_samples"], seed=train_cfg["seed"], **common_kw)
    val_ds   = RecDataset(data_cfg["num_val_samples"],   seed=train_cfg["seed"] + 1, **common_kw)

    loader_kw = dict(
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    logger.info("DataLoaders ready: batch_size=%d", data_cfg["batch_size"])
    return train_loader, val_loader
