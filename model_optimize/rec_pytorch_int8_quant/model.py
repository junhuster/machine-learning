"""
model.py — MMOE 推荐模型定义
结构：Embedding + Dense BN + FM 交叉 + MMOE (3 experts) + 双 Tower (CTR/CVR)
该结构含有足够多的 Linear / BN 层，适合验证 INT8 量化效果。
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  基础模块
# ------------------------------------------------------------------ #

class MLP(nn.Module):
    """带 BatchNorm 和 Dropout 的多层感知机。"""

    def __init__(self, input_dim, hidden_dims, dropout=0.1, use_bn=True, output_activation=True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        if not output_activation and layers:
            # 移除最后一个 ReLU（Tower 输出层不需要）
            layers = layers[:-1] if not dropout else layers[:-2]
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x):
        return self.net(x)


class FMLayer(nn.Module):
    """
    FM 二阶特征交叉层。
    输入: (batch, num_fields, emb_dim)
    输出: (batch, emb_dim)
    公式: 0.5 * (sum^2 - sum_of_square)
    """

    def forward(self, x):
        # x: (B, F, D)
        sum_emb = x.sum(dim=1)               # (B, D)
        sum_sq  = (x ** 2).sum(dim=1)        # (B, D)
        return 0.5 * (sum_emb ** 2 - sum_sq) # (B, D)


# ------------------------------------------------------------------ #
#  MMOE 主模型
# ------------------------------------------------------------------ #

class MMOERecModel(nn.Module):
    """
    Multi-gate Mixture-of-Experts 推荐模型。

    输入:
      sparse: (B, num_sparse_fields)  long
      dense:  (B, num_dense_features) float

    输出:
      (pred_ctr, pred_cvr): 各 (B,) float，经过 sigmoid
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        emb_dim          = m["emb_dim"]
        num_sparse       = m["num_sparse_fields"]
        num_dense        = m["num_dense_features"]
        num_experts      = m["num_experts"]
        expert_dims      = m["expert_hidden_dims"]
        gate_hidden      = m["gate_hidden_dim"]
        tower_dims       = m["tower_hidden_dims"]
        num_tasks        = m["num_tasks"]
        dropout          = m["dropout"]
        use_bn           = m["use_bn"]
        sparse_vocab     = cfg["data"]["sparse_vocab_size"]

        # --- Embedding 层 ---
        self.embedding = nn.Embedding(sparse_vocab, emb_dim, padding_idx=0)

        # --- 连续特征归一化 ---
        self.dense_bn = nn.BatchNorm1d(num_dense)
        self.dense_proj = nn.Linear(num_dense, emb_dim)

        # --- FM 交叉 ---
        self.fm = FMLayer()

        # 输入到 MMOE 的维度：FM 输出 + dense_proj + 拼接所有 embedding
        # FM: emb_dim, dense_proj: emb_dim, flat_emb: num_sparse * emb_dim
        mmoe_input_dim = emb_dim + emb_dim + num_sparse * emb_dim

        # --- Expert 网络 ---
        self.experts = nn.ModuleList([
            MLP(mmoe_input_dim, expert_dims, dropout=dropout, use_bn=use_bn)
            for _ in range(num_experts)
        ])
        expert_output_dim = self.experts[0].output_dim

        # --- Gate 网络（每个任务一个）---
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mmoe_input_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, num_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_tasks)
        ])

        # --- Tower 网络（每个任务一个）---
        self.towers = nn.ModuleList([
            MLP(expert_output_dim, tower_dims, dropout=dropout,
                use_bn=use_bn, output_activation=False)
            for _ in range(num_tasks)
        ])
        tower_output_dim = self.towers[0].output_dim

        # --- 输出层 ---
        self.output_layers = nn.ModuleList([
            nn.Linear(tower_output_dim, 1)
            for _ in range(num_tasks)
        ])

        self._init_weights()
        logger.info("MMOERecModel built: experts=%d, expert_dims=%s, tower_dims=%s",
                    num_experts, expert_dims, tower_dims)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, sparse, dense):
        """
        sparse: (B, F) long
        dense:  (B, D) float
        """
        # --- Embedding ---
        emb = self.embedding(sparse)       # (B, F, emb_dim)
        flat_emb = emb.view(emb.size(0), -1)  # (B, F * emb_dim)

        # --- FM 交叉 ---
        fm_out = self.fm(emb)              # (B, emb_dim)

        # --- Dense 特征 ---
        dense_out = self.dense_proj(self.dense_bn(dense))  # (B, emb_dim)

        # --- 拼接 MMOE 输入 ---
        mmoe_input = torch.cat([fm_out, dense_out, flat_emb], dim=1)  # (B, mmoe_input_dim)

        # --- Expert 计算 ---
        expert_outs = torch.stack(
            [e(mmoe_input) for e in self.experts], dim=1
        )  # (B, num_experts, expert_dim)

        # --- 双任务 Tower ---
        preds = []
        for gate, tower, out_layer in zip(self.gates, self.towers, self.output_layers):
            gate_w = gate(mmoe_input).unsqueeze(-1)          # (B, num_experts, 1)
            mixed  = (expert_outs * gate_w).sum(dim=1)       # (B, expert_dim)
            tower_out = tower(mixed)                          # (B, tower_dim)
            logit = out_layer(tower_out).squeeze(-1)          # (B,)
            preds.append(torch.sigmoid(logit))

        return preds[0], preds[1]   # pred_ctr, pred_cvr


def build_model(cfg, device):
    """构建模型并移至目标设备。"""
    model = MMOERecModel(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d  (%.2fM)", num_params, num_params / 1e6)
    return model
