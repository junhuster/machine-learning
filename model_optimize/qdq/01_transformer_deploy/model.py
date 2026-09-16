#!/usr/bin/env python3
"""模型结构：推荐场景的 Transformer 精排模型。

结构（刻意把 dense 部分做大，量化才有意义）：
    user / item / cate embedding
    dense 特征 -> Linear 投影
    candidate token + 历史序列 token -> 位置编码 -> N x Transformer Block
    head: [user, candidate, 序列聚合] 拼接 -> MLP -> logit

设计约束（为了 ONNX/TRT 友好 + 便于逐层插 Q/DQ）：
  * 所有 Linear 都是独立的 nn.Linear 子模块（不用融合算子），方便按名定位、逐个量化；
  * attention 手写（qkv / proj 两个 Linear），不依赖 nn.MultiheadAttention 的内部结构；
  * 不使用动态控制流、不做 in-place 赋值。
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn


class MHSA(nn.Module):
    """手写多头自注意力：两个 Linear（qkv、proj），便于逐层量化。"""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, seq, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # [3, B, H, L, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                # [B, H, L, Dh]
        out = out.transpose(1, 2).reshape(b, seq, self.dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN 的 Transformer Block（attention + FFN，两个残差）。"""

    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerRanker(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        m = cfg["model"]
        d = cfg["data"]
        ed = m["embed_dim"]
        self.cfg = cfg

        self.user_emb = nn.Embedding(d["num_users"], ed)
        self.item_emb = nn.Embedding(d["num_items"], ed)
        self.cate_emb = nn.Embedding(d["num_categories"], ed)
        self.hist_emb = nn.Embedding(d["num_items"], ed)   # 历史物品单独一张表
        self.pos_emb = nn.Embedding(d["history_len"] + 1, ed)
        self.dense_proj = nn.Linear(d["num_dense"], ed)

        self.blocks = nn.ModuleList(
            [TransformerBlock(ed, m["num_heads"], m["ffn_dim"], m["dropout"]) for _ in range(m["num_layers"])]
        )
        self.norm = nn.LayerNorm(ed)

        head: List[nn.Module] = []
        in_dim = ed * 3
        for h in m["head_hidden"]:
            head += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(m["dropout"])]
            in_dim = h
        head += [nn.Linear(in_dim, 1)]
        self.head = nn.Sequential(*head)

        self.register_buffer(
            "pos_index", torch.arange(d["history_len"] + 1).unsqueeze(0), persistent=False
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        user_id: torch.Tensor,     # [B] int64
        item_id: torch.Tensor,     # [B] int64
        cate_id: torch.Tensor,     # [B] int64
        history: torch.Tensor,     # [B, L] int64
        dense: torch.Tensor,       # [B, D] float32
    ) -> torch.Tensor:             # [B] float32 (logit)
        b, seq = history.shape

        u = self.user_emb(user_id)                  # [B, ed]
        i = self.item_emb(item_id)
        c = self.cate_emb(cate_id)
        x_dense = self.dense_proj(dense)            # [B, ed]

        hist = self.hist_emb(history)               # [B, L, ed]
        candidate = (i + x_dense).unsqueeze(1)      # [B, 1, ed]
        tokens = torch.cat([candidate, hist], dim=1)            # [B, L+1, ed]
        tokens = tokens + self.pos_emb(self.pos_index[:, : seq + 1])

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        cand_out = tokens[:, 0, :]                            # 候选位
        seq_out = tokens[:, 1:, :].mean(dim=1)                # 序列聚合
        feat = torch.cat([u + c, cand_out, seq_out], dim=-1)
        return self.head(feat).squeeze(-1)


def build_model(cfg: Dict) -> TransformerRanker:
    return TransformerRanker(cfg)


# 模型前向的输入名（顺序即位置参数顺序），ONNX / TRT 都按这个名字对齐
INPUT_NAMES = ["user_id", "item_id", "cate_id", "history", "dense"]
OUTPUT_NAMES = ["logit"]


def forward_inputs(bundle, device):
    """把 FeatureBundle 转成模型需要的张量（放到 device）。"""
    b = bundle.to(device)
    return b.inputs(INPUT_NAMES)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(open(Path(__file__).resolve().parent / "config.yaml"))
    m = build_model(cfg)
    print(f"参数量: {count_params(m) / 1e6:.2f} M")
    b = 2
    out = m(
        torch.zeros(b, dtype=torch.long),
        torch.zeros(b, dtype=torch.long),
        torch.zeros(b, dtype=torch.long),
        torch.zeros(b, cfg["data"]["history_len"], dtype=torch.long),
        torch.zeros(b, cfg["data"]["num_dense"]),
    )
    print("输出形状:", tuple(out.shape))
