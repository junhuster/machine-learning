#!/usr/bin/env python3
"""模型结构：DQRM 简化版（DLRM 结构）。

对应论文（DQRM: Deep Quantized Recommendation Models, arXiv:2410.20046）里的
标准推荐模型结构：Embedding tables + MLP layers，两者在特征交互层汇合。

    dense 特征 ──▶ bottom MLP ──▶ d [B, emb_dim]
    sparse 特征 ──▶ 26 张 Embedding 表 ──▶ e_1..e_26 [B, emb_dim]
                                    │
        interaction: <d, e_i> 逐列点积 ──▶ [B, num_sparse]
                                    │
        concat([d, interaction]) ──▶ top MLP ──▶ logit

**与论文的差异（重要，README 里也有说明）**：
  * 原文对 embedding 表做 INT4 量化（目的是省显存 / 上端侧，且发现 embedding 对低精度鲁棒）；
  * 本项目量化的是 **dense 部分（bottom/top MLP）**，因为：
      1) 目的是「在线推理提速」，而 embedding 查表（Gather）没有 INT8 加速、
         且 TensorRT / torch.ao 默认都不会量化 Embedding；
      2) T4 没有 INT4 推理加速。
  * 原文的 Lazy Quantization / 梯度压缩属于**分布式训练**优化，不属于本次推理量化范围。

设计约束：所有 Linear 都是独立 nn.Module（便于 FX 追踪与逐层量化）；
不使用动态控制流；Embedding 的查表用 `table(idx)` 保持为独立 Gather 节点。
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def _mlp(dims: List[int], dropout: float, final_linear: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        last = i == len(dims) - 2
        if not (last and final_linear):
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class DLRM(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        m, d = cfg["model"], cfg["data"]
        self.cfg = cfg
        self.emb_dim = m["emb_dim"]
        self.num_sparse = d["num_sparse"]

        bottom = [d["num_dense"], *m["bottom_mlp"], m["emb_dim"]]
        self.bottom_mlp = _mlp(bottom, m["dropout"], final_linear=True)

        self.emb_tables = nn.ModuleList(
            [nn.Embedding(d["sparse_vocab"], m["emb_dim"]) for _ in range(d["num_sparse"])]
        )

        top = [d["num_sparse"] + m["emb_dim"], *m["top_mlp"], 1]
        self.top_mlp = _mlp(top, m["dropout"], final_linear=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "emb_tables" in name:
                nn.init.normal_(p, 0.0, 0.02)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        """dense: [B, num_dense] float32 ; sparse: [B, num_sparse] int64 ; 返回 [B] logit"""
        d = self.bottom_mlp(dense)                       # [B, emb_dim]
        embs = [table(sparse[:, i]) for i, table in enumerate(self.emb_tables)]
        stacked = torch.stack(embs, dim=1)               # [B, num_sparse, emb_dim]
        inter = (stacked * d.unsqueeze(1)).sum(dim=-1)   # [B, num_sparse]  逐列点积
        feat = torch.cat([d, inter], dim=-1)             # [B, num_sparse + emb_dim]
        return self.top_mlp(feat).squeeze(-1)


def build_model(cfg: Dict) -> DLRM:
    return DLRM(cfg)


INPUT_NAMES = ["dense", "sparse"]
OUTPUT_NAMES = ["logit"]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_params_by_part(model: nn.Module) -> Dict[str, int]:
    emb = sum(p.numel() for n, p in model.named_parameters() if n.startswith("emb_tables"))
    dense = sum(p.numel() for p in model.parameters()) - emb
    return {"dense(被量化的部分)": dense, "embedding(不量化)": emb, "合计": emb + dense}


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(open(Path(__file__).resolve().parent / "config.yaml"))
    m = build_model(cfg)
    for k, v in count_params_by_part(m).items():
        print(f"  {k:22s} {v / 1e6:8.2f} M")
    b = 2
    out = m(
        torch.zeros(b, cfg["data"]["num_dense"]),
        torch.zeros(b, cfg["data"]["num_sparse"], dtype=torch.long),
    )
    print("输出形状:", tuple(out.shape))
