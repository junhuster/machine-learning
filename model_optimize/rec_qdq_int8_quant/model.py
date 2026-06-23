"""
model.py — MMOE 推荐模型（与 rec_pytorch_int8_quant 保持一致）
注意：不再使用 torch.quantization，QDQ 节点由 quantize.py 动态插入。
"""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1, use_bn=True,
                 output_activation=True):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        if not output_activation and layers:
            layers = layers[:-1] if not dropout else layers[:-2]
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x):
        return self.net(x)


class FMLayer(nn.Module):
    def forward(self, x):
        sum_emb = x.sum(dim=1)
        sum_sq  = (x ** 2).sum(dim=1)
        return 0.5 * (sum_emb ** 2 - sum_sq)


class MMOERecModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m            = cfg["model"]
        emb_dim      = m["emb_dim"]
        num_sparse   = m["num_sparse_fields"]
        num_dense    = m["num_dense_features"]
        num_experts  = m["num_experts"]
        expert_dims  = m["expert_hidden_dims"]
        gate_hidden  = m["gate_hidden_dim"]
        tower_dims   = m["tower_hidden_dims"]
        num_tasks    = m["num_tasks"]
        dropout      = m["dropout"]
        use_bn       = m["use_bn"]
        sparse_vocab = cfg["data"]["sparse_vocab_size"]

        self.embedding  = nn.Embedding(sparse_vocab, emb_dim, padding_idx=0)
        self.dense_bn   = nn.BatchNorm1d(num_dense)
        self.dense_proj = nn.Linear(num_dense, emb_dim)
        self.fm         = FMLayer()

        mmoe_input_dim = emb_dim + emb_dim + num_sparse * emb_dim

        self.experts = nn.ModuleList([
            MLP(mmoe_input_dim, expert_dims, dropout=dropout, use_bn=use_bn)
            for _ in range(num_experts)
        ])
        expert_out_dim = self.experts[0].output_dim

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mmoe_input_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, num_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_tasks)
        ])
        self.towers = nn.ModuleList([
            MLP(expert_out_dim, tower_dims, dropout=dropout,
                use_bn=use_bn, output_activation=False)
            for _ in range(num_tasks)
        ])
        tower_out_dim = self.towers[0].output_dim
        self.output_layers = nn.ModuleList([
            nn.Linear(tower_out_dim, 1) for _ in range(num_tasks)
        ])
        self._init_weights()
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("MMOERecModel: params=%d (%.2fM)", num_params, num_params / 1e6)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, sparse, dense):
        emb      = self.embedding(sparse)
        flat_emb = emb.view(emb.size(0), -1)
        fm_out   = self.fm(emb)
        dense_out = self.dense_proj(self.dense_bn(dense))
        mmoe_input = torch.cat([fm_out, dense_out, flat_emb], dim=1)

        expert_outs = torch.stack(
            [e(mmoe_input) for e in self.experts], dim=1
        )
        preds = []
        for gate, tower, out_layer in zip(self.gates, self.towers, self.output_layers):
            gate_w    = gate(mmoe_input).unsqueeze(-1)
            mixed     = (expert_outs * gate_w).sum(dim=1)
            tower_out = tower(mixed)
            logit     = out_layer(tower_out).squeeze(-1)
            preds.append(torch.sigmoid(logit))
        return preds[0], preds[1]


def build_model(cfg, device):
    model = MMOERecModel(cfg).to(device)
    return model
