"""
data.py — 假数据生成模块（与 rec_pytorch_int8_quant 保持一致）
"""
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)


class RecDataset(Dataset):
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
        logit   = (self.dense[:, :3].sum(dim=1) +
                   (self.sparse[:, 0] % 10).float() * 0.1)
        prob_ctr = torch.sigmoid(logit + torch.randn(num_samples) * 0.5)
        prob_cvr = torch.sigmoid(logit * 0.8 + torch.randn(num_samples) * 0.5)
        self.label_ctr = (prob_ctr > 0.5).float()
        self.label_cvr = (prob_cvr > 0.5).float()
        logger.debug("Dataset created: %d samples", num_samples)

    def __len__(self):
        return len(self.label_ctr)

    def __getitem__(self, idx):
        return {
            "sparse":    self.sparse[idx],
            "dense":     self.dense[idx],
            "label_ctr": self.label_ctr[idx],
            "label_cvr": self.label_cvr[idx],
        }


def build_dataloaders(cfg):
    d, t = cfg["data"], cfg["train"]
    logger.info("Building datasets: train=%d  val=%d",
                d["num_train_samples"], d["num_val_samples"])
    common = dict(num_sparse_fields=d["num_sparse_fields"],
                  sparse_vocab_size=d["sparse_vocab_size"],
                  num_dense_features=d["num_dense_features"])
    train_ds = RecDataset(d["num_train_samples"], seed=t["seed"],     **common)
    val_ds   = RecDataset(d["num_val_samples"],   seed=t["seed"] + 1, **common)
    kw = dict(batch_size=d["batch_size"], num_workers=d["num_workers"], pin_memory=True)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw),
            val_ds)
