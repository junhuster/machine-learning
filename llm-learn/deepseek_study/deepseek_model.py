import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# =============================================================================
# 1. RMSNorm （DeepSeek 官方归一化）
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# =============================================================================
# 2. RoPE 旋转位置编码（DeepSeek 官方实现）
# =============================================================================
def apply_rope(q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor, theta: float = 10000.0):
    # q,k: [B, n_heads, seq_len, head_dim]
    B, H, S, D = q.shape
    assert D % 2 == 0, "head_dim 必须是偶数，适配RoPE"

    # 生成频率
    dims = torch.arange(0, D, 2, device=q.device).float()
    freqs = 1.0 / (theta ** (dims / D))

    # 位置编码
    pos = pos_ids.float()
    freqs = torch.outer(pos, freqs)  # [seq_len, D//2]

    # 构造 cos sin
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos[None, None, :, :]  # [1,1,S,D//2]
    sin = sin[None, None, :, :]

    # 拆分奇偶维度
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    # 旋转
    q_out = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_out = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_out, k_out

# =============================================================================
# 3. 带RoPE的多头注意力 GQA（对齐DeepSeek）
# =============================================================================
class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        self.repeat_times = self.n_heads // self.n_kv_heads

        # QKV 三合一投影 你之前问的那行核心代码
        self.qkv_proj = nn.Linear(dim, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, pos_ids, mask=None):
        B, S, _ = x.shape

        # QKV 投影
        qkv = self.qkv_proj(x)

        # 拆分 Q KV
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # 变形分头
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)      # [B,H,S,D]
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 加入 RoPE 位置编码
        q, k = apply_rope(q, k, pos_ids)

        # GQA 复制KV头
        k = k.repeat_interleave(self.repeat_times, dim=1)
        v = v.repeat_interleave(self.repeat_times, dim=1)

        # 缩放点积注意力
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        # 合并多头
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.dim)
        return self.o_proj(attn_out)

# =============================================================================
# 4. MoE 混合专家层（DeepSeek 经典MOE结构）
# =============================================================================
class MoE(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=4, topk=2):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.gate = nn.Linear(dim, num_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=False)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        # 路由
        gate_logits = self.gate(x_flat)
        topk_vals, topk_idx = torch.topk(gate_logits, self.topk, dim=-1)
        topk_vals = F.softmax(topk_vals, dim=-1)

        out = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            token_pos = (topk_idx == expert_idx).any(dim=-1)
            if not token_pos.any():
                continue
            tokens_in = x_flat[token_pos]
            expert_out = self.experts[expert_idx](tokens_in)

            # 加权合并
            weight = topk_vals[token_pos, (topk_idx[token_pos] == expert_idx).nonzero(as_tuple=True)[1]]
            out[token_pos] += expert_out * weight.unsqueeze(-1)

        return out.view(B, S, D)

# =============================================================================
# 5. FFN / MOE 自适应切换
# =============================================================================
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, use_moe=False, num_experts=4):
        super().__init__()
        self.use_moe = use_moe
        if use_moe:
            self.net = MoE(dim, hidden_dim, num_experts=num_experts)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=False)
            )

    def forward(self, x):
        return self.net(x)

# =============================================================================
# 6. Transformer 层 Block
# =============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, n_kv_heads=None, use_moe=False):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, use_moe=use_moe)

    def forward(self, x, pos_ids, mask=None):
        # 前置归一化
        x = x + self.attn(self.attn_norm(x), pos_ids, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

# =============================================================================
# 7. 完整 DeepSeek 模型（含RoPE+MOE+RMSNorm）
# =============================================================================
class DeepSeek(nn.Module):
    def __init__(
        self,
        vocab_size=65024,
        dim=512,
        n_heads=8,
        n_kv_heads=4,
        n_layers=8,
        hidden_dim=1536,
        max_seq_len=512,
        use_moe=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 词嵌入
        self.embeddings = nn.Embedding(vocab_size, dim)
        # 多层Transformer
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim, n_kv_heads, use_moe=use_moe)
            for _ in range(n_layers)
        ])
        # 最终归一化
        self.norm = RMSNorm(dim)
        # 语言模型头
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape
        # 生成位置id 给RoPE用
        pos_ids = torch.arange(0, S, device=input_ids.device, dtype=torch.long)

        # 词嵌入
        x = self.embeddings(input_ids)

        # 逐层前向
        for layer in self.layers:
            x = layer(x, pos_ids, mask=attention_mask)

        # 输出
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits