"""
DCNv2 + DIN 推荐模型
- DIN: 目标感知注意力，对用户历史行为序列做target-aware加权池化
- DCNv2: Cross网络（低秩分解）+ Deep MLP，显式高阶bit-wise特征交叉
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINAttention(nn.Module):
    """
    DIN Local Activation Unit
    用候选item (query) 对历史行为序列 (keys) 做目标感知注意力
    输入拼接: [query, key, query-key, query*key] -> MLP -> score
    """
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query, keys, mask=None):
        """
        query: [B, D]  候选item embedding
        keys:  [B, L, D] 历史行为序列embedding
        mask:  [B, L]  1=有效, 0=padding
        返回:   [B, D]  加权后的用户兴趣向量
        """
        B, L, D = keys.shape
        query_exp = query.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        concat = torch.cat(
            [query_exp, keys, query_exp - keys, query_exp * keys],
            dim=-1,
        )  # [B, L, 4D]
        scores = self.mlp(concat).squeeze(-1)  # [B, L]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)  # [B, L]
        output = (weights.unsqueeze(-1) * keys).sum(dim=1)  # [B, D]
        return output


class CrossLayer(nn.Module):
    """
    DCNv2 Cross层（低秩分解版本）
    x_{l+1} = x0 * (V @ U @ x_l) + bias + x_l
    其中 W = U @ V^T，U: [d, r], V: [r, d], r << d
    """
    def __init__(self, dim, low_rank=16):
        super().__init__()
        self.U = nn.Linear(dim, low_rank, bias=False)
        self.V = nn.Linear(low_rank, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, xl):
        transformed = self.V(self.U(xl))  # [B, d]
        return x0 * transformed + self.bias + xl


class DCNDIN(nn.Module):
    """
    DCNv2 + DIN 完整模型
    特征: user_id, item_id, cate_id, history序列, numeric数值特征
    流程: Embedding -> DIN注意力 -> 拼接 -> CrossNet + Deep并行 -> 最终CTR
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        ed = config['embedding_dim']

        # Embedding层
        self.user_emb = nn.Embedding(config['num_users'], ed)
        self.item_emb = nn.Embedding(config['num_items'], ed)
        self.cate_emb = nn.Embedding(config['num_categories'], ed)

        # DIN目标注意力
        self.din = DINAttention(ed, hidden_dim=64)

        # 拼接维度: user + item + cate + din_output + numeric
        num_numeric = config.get('num_numeric', 3)
        concat_dim = ed * 4 + num_numeric

        # DCNv2 Cross网络
        self.cross_layers = nn.ModuleList([
            CrossLayer(concat_dim, config.get('cross_low_rank', 16))
            for _ in range(config['num_cross_layers'])
        ])

        # Deep MLP
        deep_layers = []
        in_dim = concat_dim
        for h in config['deep_hidden']:
            deep_layers.append(nn.Linear(in_dim, h))
            deep_layers.append(nn.ReLU())
            in_dim = h
        self.deep = nn.Sequential(*deep_layers)

        # 最终输出层: Cross输出 + Deep输出
        self.final = nn.Linear(concat_dim + in_dim, 1)

    def forward(self, user_id, item_id, cate_id, history, history_mask, numeric):
        # Embedding
        u = self.user_emb(user_id)       # [B, D]
        i = self.item_emb(item_id)       # [B, D]
        c = self.cate_emb(cate_id)       # [B, D]
        hist = self.item_emb(history)    # [B, L, D]

        # DIN: 用候选item做query，加权历史序列
        din_out = self.din(i, hist, history_mask)  # [B, D]

        # 拼接所有特征
        x = torch.cat([u, i, c, din_out, numeric], dim=-1)  # [B, concat_dim]

        # Cross网络（显式高阶交叉）
        xc = x
        for layer in self.cross_layers:
            xc = layer(x, xc)

        # Deep网络（隐式非线性交互）
        xd = self.deep(x)

        # 合并Cross和Deep输出
        out = torch.cat([xc, xd], dim=-1)
        return self.final(out).squeeze(-1)  # [B]
