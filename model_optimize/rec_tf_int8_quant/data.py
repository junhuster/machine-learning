"""
data.py — 假数据生成模块（TensorFlow 版本）
生成搜推场景下的稀疏特征 + 连续特征 + 二分类标签数据
"""

import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class RecDataset:
    """
    搜推场景假数据集。
    每条样本包含：
      - sparse: (num_sparse_fields,) int32   稀疏特征 id
      - dense:  (num_dense_features,) float32 连续特征
      - label_ctr: float32  点击标签
      - label_cvr: float32  转化标签
    """

    def __init__(self, num_samples, num_sparse_fields, sparse_vocab_size,
                 num_dense_features, seed=42):
        rng = np.random.default_rng(seed)

        self.sparse = rng.integers(
            0, sparse_vocab_size,
            size=(num_samples, num_sparse_fields)
        ).astype(np.int32)

        self.dense = rng.standard_normal(
            (num_samples, num_dense_features)
        ).astype(np.float32)

        # 用简单线性关系 + 噪声生成标签，保证 AUC 有意义
        logit = (self.dense[:, :3].sum(axis=1) +
                 (self.sparse[:, 0] % 10).astype(np.float32) * 0.1)
        noise = rng.standard_normal(num_samples).astype(np.float32)

        prob_ctr = 1.0 / (1.0 + np.exp(-(logit + noise * 0.5)))
        prob_cvr = 1.0 / (1.0 + np.exp(-(logit * 0.8 + noise * 0.5)))

        self.label_ctr = (prob_ctr > 0.5).astype(np.float32)
        self.label_cvr = (prob_cvr > 0.5).astype(np.float32)

        logger.debug(
            "Dataset created: %d samples, sparse_fields=%d, dense_features=%d",
            num_samples, num_sparse_fields, num_dense_features,
        )

    def __len__(self):
        return len(self.label_ctr)

    def as_tf_dataset(self, batch_size, shuffle=False, seed=42):
        """返回 tf.data.Dataset，输出 (inputs_dict, labels_dict)。"""
        ds = tf.data.Dataset.from_tensor_slices({
            "sparse":    self.sparse,
            "dense":     self.dense,
            "label_ctr": self.label_ctr,
            "label_cvr": self.label_cvr,
        })
        if shuffle:
            ds = ds.shuffle(buffer_size=10000, seed=seed)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def as_numpy_batches(self, batch_size):
        """逐 batch 产出 numpy dict，用于 onnx/trt calibration 和推理。"""
        n = len(self.label_ctr)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield {
                "sparse":    self.sparse[start:end],
                "dense":     self.dense[start:end],
                "label_ctr": self.label_ctr[start:end],
                "label_cvr": self.label_cvr[start:end],
            }


def build_datasets(cfg):
    """根据配置构建训练集和验证集。"""
    d = cfg["data"]
    t = cfg["train"]

    logger.info("Building datasets: train=%d  val=%d",
                d["num_train_samples"], d["num_val_samples"])

    common = dict(
        num_sparse_fields=d["num_sparse_fields"],
        sparse_vocab_size=d["sparse_vocab_size"],
        num_dense_features=d["num_dense_features"],
    )
    train_ds = RecDataset(d["num_train_samples"], seed=t["seed"],     **common)
    val_ds   = RecDataset(d["num_val_samples"],   seed=t["seed"] + 1, **common)

    train_loader = train_ds.as_tf_dataset(d["batch_size"], shuffle=True, seed=t["seed"])
    val_loader   = val_ds.as_tf_dataset(d["batch_size"], shuffle=False)

    logger.info("DataLoaders ready: batch_size=%d", d["batch_size"])
    return train_loader, val_loader, train_ds, val_ds
