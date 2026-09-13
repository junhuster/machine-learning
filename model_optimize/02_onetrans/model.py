"""
OneTrans: 字节跳动单Transformer精排模型
核心思想: 用一个Transformer同时做 特征交互 + 序列建模，
替代 DIN(序列建模) + DCN(特征交叉) 的分离范式。

所有特征（用户、物品、上下文、历史行为序列）都被token化，
加上type embedding和position embedding，送入单个Transformer Encoder，
最终用<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token的输出做CTR预测。

Token顺序: <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>, user, item, cate, numeric, history_1, history_2, ..., history_L
"""
import torch
import torch.nn as nn


class OneTrans(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ed = config['embedding_dim']
        history_len = config['history_len']
        num_numeric = config.get('num_numeric', 3)

        # ---- Embedding层 ----
        self.user_emb = nn.Embedding(config['num_users'], ed)
        self.item_emb = nn.Embedding(config['num_items'], ed)
        self.cate_emb = nn.Embedding(config['num_categories'], ed)
        # 数值特征投影到ed维
        self.numeric_proj = nn.Linear(num_numeric, ed)

        # ---- Token Type Embedding ----
        # 0=CLS, 1=user, 2=item, 3=cate, 4=numeric, 5=history
        self.type_emb = nn.Embedding(6, ed)

        # ---- Position Embedding (仅用于history序列) ----
        self.pos_emb = nn.Embedding(history_len, ed)

        # ---- <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> Token ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, ed) * 0.02)

        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ed,
            nhead=config['num_heads'],
            dim_feedforward=config['ffn_dim'],
            dropout=config.get('dropout', 0.1),
            batch_first=True,
            activation='gelu',
            norm_first=True,  # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_layers'],
        )

        # 最终LayerNorm
        self.final_norm = nn.LayerNorm(ed)

        # ---- 预测头 ----
        self.head = nn.Sequential(
            nn.Linear(ed, ed),
            nn.GELU(),
            nn.Linear(ed, 1),
        )

        # 预计算type ids（避免每次forward重复创建）
        self._type_ids = None

    def _get_type_ids(self, device, num_history):
        """生成token type ids: [cls=0, user=1, item=2, cate=3, numeric=4, history*=5]"""
        # 固定部分: cls, user, item, cate, numeric = 5个token
        fixed = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.long)
        history_types = torch.full((num_history,), 5, device=device, dtype=torch.long)
        return torch.cat([fixed, history_types])

    def forward(self, user_id, item_id, cate_id, history, history_mask, numeric):
        """
        user_id:     [B]
        item_id:     [B]
        cate_id:     [B]
        history:     [B, L]  历史行为item ids
        history_mask:[B, L]  1=有效, 0=padding
        numeric:     [B, N]  数值特征
        """
        B = user_id.shape[0]
        L = history.shape[1]
        device = user_id.device

        # ---- 各特征转embedding ----
        u = self.user_emb(user_id).unsqueeze(1)        # [B, 1, D]
        i = self.item_emb(item_id).unsqueeze(1)        # [B, 1, D]
        c = self.cate_emb(cate_id).unsqueeze(1)        # [B, 1, D]
        n = self.numeric_proj(numeric).unsqueeze(1)    # [B, 1, D]
        h = self.item_emb(history)                       # [B, L, D]

        # ---- <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token ----
        cls = self.cls_token.expand(B, -1, -1)          # [B, 1, D]

        # ---- 拼接所有token ----
        # 顺序: cls, user, item, cate, numeric, history_1...history_L
        tokens = torch.cat([cls, u, i, c, n, h], dim=1)  # [B, 5+L, D]

        # ---- 加Type Embedding ----
        type_ids = self._get_type_ids(device, L)          # [5+L]
        tokens = tokens + self.type_emb(type_ids).unsqueeze(0)  # broadcast到 [B, 5+L, D]

        # ---- 历史序列加Position Embedding ----
        # history token在位置 5..5+L
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # [B, L]
        pos_emb = self.pos_emb(pos_ids)  # [B, L, D]
        tokens[:, 5:5+L, :] = tokens[:, 5:5+L, :] + pos_emb

        # ---- Attention Mask: 屏蔽padding的history token ----
        # key_padding_mask: [B, 5+L], True=屏蔽
        key_padding_mask = torch.zeros(B, 5 + L, device=device, dtype=torch.bool)
        # history部分: mask=0(padding)的位置设为True
        key_padding_mask[:, 5:5+L] = (history_mask == 0)

        # ---- Transformer Encoder ----
        out = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        out = self.final_norm(out)

        # ---- 取<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>输出做预测 ----
        cls_out = out[:, 0, :]  # [B, D]
        logits = self.head(cls_out).squeeze(-1)  # [B]
        return logits
