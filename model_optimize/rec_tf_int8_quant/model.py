"""
model.py — MMOE 推荐模型定义（TensorFlow/Keras 版本）
结构：Embedding + Dense BN + FM 交叉 + MMOE (3 experts) + 双 Tower (CTR/CVR)
"""

import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  基础模块
# ------------------------------------------------------------------ #

def build_mlp(input_dim, hidden_dims, dropout=0.1, use_bn=True,
              output_activation=True, name_prefix="mlp"):
    """构建带 BN 和 Dropout 的 MLP，返回 keras.Sequential。"""
    model = keras.Sequential(name=name_prefix)
    for i, h in enumerate(hidden_dims):
        model.add(layers.Dense(h, name=f"{name_prefix}_dense_{i}"))
        if use_bn:
            model.add(layers.BatchNormalization(name=f"{name_prefix}_bn_{i}"))
        model.add(layers.ReLU(name=f"{name_prefix}_relu_{i}"))
        if dropout > 0:
            model.add(layers.Dropout(dropout, name=f"{name_prefix}_drop_{i}"))
    if not output_activation and hidden_dims:
        # 移除最后一个激活层（Tower 输出前不需要）
        model.layers.pop()
        if dropout > 0 and len(model.layers) > 0:
            model.layers.pop()
    return model


class FMLayer(layers.Layer):
    """
    FM 二阶特征交叉层。
    输入: (batch, num_fields, emb_dim)
    输出: (batch, emb_dim)
    """

    def call(self, x):
        sum_emb = tf.reduce_sum(x, axis=1)           # (B, D)
        sum_sq  = tf.reduce_sum(tf.square(x), axis=1) # (B, D)
        return 0.5 * (tf.square(sum_emb) - sum_sq)


# ------------------------------------------------------------------ #
#  MMOE 主模型
# ------------------------------------------------------------------ #

class MMOERecModel(keras.Model):
    """
    Multi-gate Mixture-of-Experts 推荐模型。

    输入:
      sparse: (B, num_sparse_fields)  int32
      dense:  (B, num_dense_features) float32

    输出:
      [pred_ctr, pred_cvr]: 各 (B, 1) float，经过 sigmoid
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        m = cfg["model"]
        self.emb_dim        = m["emb_dim"]
        self.num_sparse     = m["num_sparse_fields"]
        self.num_dense      = m["num_dense_features"]
        self.num_experts    = m["num_experts"]
        self.num_tasks      = m["num_tasks"]
        dropout             = m["dropout"]
        use_bn              = m["use_bn"]
        expert_dims         = m["expert_hidden_dims"]
        gate_hidden         = m["gate_hidden_dim"]
        tower_dims          = m["tower_hidden_dims"]
        sparse_vocab        = cfg["data"]["sparse_vocab_size"]

        # --- Embedding ---
        self.embedding = layers.Embedding(sparse_vocab, self.emb_dim,
                                          name="embedding")

        # --- Dense 特征处理 ---
        self.dense_bn   = layers.BatchNormalization(name="dense_bn")
        self.dense_proj = layers.Dense(self.emb_dim, name="dense_proj")

        # --- FM ---
        self.fm = FMLayer(name="fm")

        # --- Expert 网络 ---
        self.experts = [
            build_mlp(0, expert_dims, dropout=dropout, use_bn=use_bn,
                      name_prefix=f"expert_{i}")
            for i in range(self.num_experts)
        ]

        # --- Gate 网络 ---
        self.gate_dense1 = [
            layers.Dense(gate_hidden, activation="relu", name=f"gate_{i}_dense1")
            for i in range(self.num_tasks)
        ]
        self.gate_dense2 = [
            layers.Dense(self.num_experts, activation="softmax", name=f"gate_{i}_dense2")
            for i in range(self.num_tasks)
        ]

        # --- Tower 网络 ---
        self.towers = [
            build_mlp(0, tower_dims, dropout=dropout, use_bn=use_bn,
                      output_activation=False, name_prefix=f"tower_{i}")
            for i in range(self.num_tasks)
        ]

        # --- 输出层 ---
        self.output_layers = [
            layers.Dense(1, activation="sigmoid", name=f"output_{i}")
            for i in range(self.num_tasks)
        ]

    def call(self, inputs, training=False):
        sparse = inputs["sparse"]   # (B, F)
        dense  = inputs["dense"]    # (B, D)

        # --- Embedding ---
        emb      = self.embedding(sparse)                     # (B, F, emb_dim)
        flat_emb = tf.reshape(emb, (tf.shape(emb)[0], -1))   # (B, F*emb_dim)

        # --- FM ---
        fm_out = self.fm(emb)                                  # (B, emb_dim)

        # --- Dense ---
        dense_out = self.dense_proj(
            self.dense_bn(dense, training=training)
        )                                                      # (B, emb_dim)

        # --- MMOE 输入拼接 ---
        mmoe_input = tf.concat([fm_out, dense_out, flat_emb], axis=1)

        # --- Expert ---
        expert_outs = tf.stack(
            [e(mmoe_input, training=training) for e in self.experts],
            axis=1
        )  # (B, num_experts, expert_dim)

        # --- Gate + Tower ---
        preds = []
        for i in range(self.num_tasks):
            gate_w = self.gate_dense2[i](
                self.gate_dense1[i](mmoe_input)
            )                                                  # (B, num_experts)
            gate_w = tf.expand_dims(gate_w, axis=-1)           # (B, num_experts, 1)
            mixed  = tf.reduce_sum(expert_outs * gate_w, axis=1)  # (B, expert_dim)
            tower_out = self.towers[i](mixed, training=training)
            pred = self.output_layers[i](tower_out)            # (B, 1)
            preds.append(tf.squeeze(pred, axis=-1))            # (B,)

        return preds  # [pred_ctr, pred_cvr]


def build_model(cfg):
    """构建并编译模型。"""
    model = MMOERecModel(cfg, name="mmoe_rec")

    # 构建计算图（需要一次 call 才能确定权重）
    dummy_sparse = tf.zeros((1, cfg["model"]["num_sparse_fields"]), dtype=tf.int32)
    dummy_dense  = tf.zeros((1, cfg["model"]["num_dense_features"]), dtype=tf.float32)
    model({"sparse": dummy_sparse, "dense": dummy_dense}, training=False)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["train"]["lr"]),
        loss={
            "ctr": keras.losses.BinaryCrossentropy(),
            "cvr": keras.losses.BinaryCrossentropy(),
        },
        loss_weights={"ctr": 1.0, "cvr": 1.0},
    )

    num_params = model.count_params()
    logger.info("MMOERecModel built: params=%d (%.2fM)", num_params, num_params / 1e6)
    return model
