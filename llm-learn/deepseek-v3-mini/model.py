"""
DeepSeek-Mini: 基于DeepSeek-V3架构的轻量化单卡训练版本

与原版的主要改动：
- 移除bf16/fp8量化，全部使用fp16
- 移除多GPU分布式并行（world_size=1，单卡）
- 移除Triton kernel依赖
- 简化Linear层（无量化scale）
- 模型默认参数缩小，适合T4 16GB显存学习
- 支持训练（causal LM loss）和推理（KV cache）两种模式
"""

import math
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    """
    模型超参数。默认值为Mini版本，适合T4显卡单卡运行。
    如果显存不够，可以继续缩小dim、n_layers等参数。
    """
    # 词表
    vocab_size: int = 102400          # DeepSeek-V3分词器词表大小，不要修改

    # 模型尺寸（Mini默认值，比原16B小很多）
    dim: int = 512                    # 隐层维度（原16B是2048）
    inter_dim: int = 1344             # 密集层FFN中间维度（~2.625x dim）
    moe_inter_dim: int = 256          # MoE每个专家的中间维度

    # Transformer层
    n_layers: int = 8                 # 总层数（原16B是27层）
    n_dense_layers: int = 1           # 前N层用普通MLP，其余用MoE

    # 注意力（MLA）
    n_heads: int = 8                  # 注意力头数
    q_lora_rank: int = 0              # Query LoRA压缩rank；0=不压缩
    kv_lora_rank: int = 128           # KV低秩压缩rank（节省KV cache）
    qk_nope_head_dim: int = 64        # 不带位置编码的Q/K每头维度
    qk_rope_head_dim: int = 32        # 带RoPE位置编码的Q/K每头维度
    v_head_dim: int = 64              # Value每头维度

    # MoE路由
    n_routed_experts: int = 16        # 路由专家总数
    n_shared_experts: int = 1         # 共享专家数（每个token都会经过）
    n_activated_experts: int = 2      # 每个token激活的路由专家数
    n_expert_groups: int = 1          # 专家分组数（>1时按组路由）
    n_limited_groups: int = 1         # 限制选择的组数
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0          # 路由权重缩放系数

    # RoPE位置编码 / YaRN长度外推
    original_seq_len: int = 2048      # 基础序列长度
    max_seq_len: int = 2048           # 模型支持的最大序列长度
    rope_theta: float = 10000.0       # RoPE基础频率
    rope_factor: float = 1.0          # >1时启用YaRN外推
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0               # YaRN注意力缩放系数

    # 推理KV cache（用于inference.py）
    max_batch_size: int = 4


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE / YaRN)
# ---------------------------------------------------------------------------

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """预计算RoPE旋转位置编码的复数形式频率张量，shape=(max_seq_len, qk_rope_head_dim//2)"""
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta
    factor = args.rope_factor

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # YaRN长度外推（仅当序列长度超过original_seq_len且rope_factor>1时启用）
    if seqlen > args.original_seq_len and factor > 1.0:
        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(mn, mx, d):
            if mn == mx:
                mx += 0.001
            t = (torch.arange(d, dtype=torch.float32) - mn) / (mx - mn)
            return torch.clamp(t, 0, 1)

        low, high = find_correction_range(args.beta_fast, args.beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    将旋转位置编码应用到张量x上。
    x: (..., seq_len, n_heads, head_dim)  其中head_dim必须为偶数
    freqs_cis: (seq_len, head_dim//2) 复数
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


# ---------------------------------------------------------------------------
# Multi-Head Latent Attention (MLA)
# ---------------------------------------------------------------------------

class MLA(nn.Module):
    """
    DeepSeek-V3的核心注意力机制：Multi-Head Latent Attention。

    通过低秩分解压缩KV，大幅减少KV cache内存占用。
    训练时不用KV cache，全序列计算；推理时用KV cache逐步生成。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.dim = args.dim

        # Query投影
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(args.dim, args.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(args.dim, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, args.n_heads * self.qk_head_dim, bias=False)

        # KV低秩投影
        self.wkv_a = nn.Linear(args.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            args.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # 输出投影
        self.wo = nn.Linear(args.n_heads * self.v_head_dim, args.dim, bias=False)

        # Softmax缩放系数
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len and args.rope_factor > 1.0:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # 推理用KV cache buffers（训练时不使用）
        self.register_buffer(
            "k_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len, args.n_heads, self.qk_head_dim),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len, args.n_heads, self.v_head_dim),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            freqs_cis: 位置编码复数，shape=(seq_len, qk_rope_head_dim//2)
            mask: 因果掩码，shape=(seq_len, kv_len)，上三角为-inf
            start_pos: 推理时KV cache的写入起始位置
            use_cache: True=推理模式，使用并更新KV cache；False=训练模式
        Returns:
            输出tensor，shape同x
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # --- Query ---
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)  # (B, S, H, qk_head_dim)

        # --- Key/Value ---
        kv = self.wkv_a(x)
        kv_latent, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # (B, S, 1, rope_dim)

        kv_full = self.wkv_b(self.kv_norm(kv_latent))
        kv_full = kv_full.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_full, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)  # (B, S, H, qk_head_dim)

        if use_cache:
            # 写入cache，读取历史+当前
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            k_use = self.k_cache[:bsz, :end_pos]
            v_use = self.v_cache[:bsz, :end_pos]
        else:
            k_use = k
            v_use = v

        # --- Attention ---
        # scores: (B, S, H, T)  S=query len, T=key/value len
        scores = torch.einsum("bshd,bthd->bsht", q, k_use) * self.softmax_scale
        if mask is not None:
            scores = scores + mask.unsqueeze(1)  # broadcast over heads
        scores = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.einsum("bsht,bthd->bshd", scores, v_use)
        out = self.wo(out.flatten(2))
        return out


# ---------------------------------------------------------------------------
# MLP (dense feed-forward with SwiGLU)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Mixture-of-Experts (MoE)
# ---------------------------------------------------------------------------

class Gate(nn.Module):
    """MoE路由门控：根据输入token决定路由到哪些专家"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, dim)  N = batch * seq_len
        Returns:
            weights: (N, topk) 路由权重
            indices: (N, topk) 选中的专家索引
        """
        scores = F.linear(x, self.weight)  # (N, n_experts)

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()

        original_scores = scores

        if self.n_groups > 1:
            scores_grouped = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores_grouped.amax(dim=-1)
            group_indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            group_mask = torch.ones(x.size(0), self.n_groups, dtype=torch.bool, device=x.device)
            group_mask.scatter_(1, group_indices, False)
            scores = scores_grouped.masked_fill(group_mask.unsqueeze(-1), float("-inf")).flatten(1)

        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)

        if self.score_func == "sigmoid":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        weights = (weights * self.route_scale).to(x.dtype)
        return weights, indices


class Expert(nn.Module):
    """MoE路由专家模块（SwiGLU FFN）"""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Mixture-of-Experts层：路由专家 + 共享专家"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim)
            for _ in range(args.n_routed_experts)
        ])
        # 共享专家：每个token都会经过，不参与路由
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x_flat = x.view(-1, self.dim)  # (N, dim)

        weights, indices = self.gate(x_flat)  # (N, topk), (N, topk)
        y = torch.zeros_like(x_flat)

        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += self.experts[i](x_flat[idx]) * weights[idx, top, None]

        z = self.shared_experts(x_flat)
        return (y + z).view(shape)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """单个Transformer层：Pre-norm + MLA注意力 + Pre-norm + FFN（密集或MoE）"""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        # 前n_dense_layers层使用密集MLP，其余使用MoE
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask, start_pos, use_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    DeepSeek-Mini完整Transformer模型。

    支持两种使用模式：
    1. 训练模式（use_cache=False）：输入(B, T)，输出logits (B, T, V)
    2. 推理模式（use_cache=True）：逐步生成，内置KV cache
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.vocab_size = args.vocab_size

        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 位置编码buffer（不参与训练，不保存到checkpoint）
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_kv_cache(self):
        """推理前清零所有层的KV cache"""
        for layer in self.layers:
            if hasattr(layer.attn, "k_cache"):
                layer.attn.k_cache.zero_()
                layer.attn.v_cache.zero_()

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            tokens: token id张量，shape=(batch, seq_len)
            start_pos: KV cache写入起始位置（推理时使用）
            use_cache: True=推理模式（返回最后位置logits）；False=训练模式（返回全序列logits）
        Returns:
            训练模式: (batch, seq_len, vocab_size)
            推理模式: (batch, vocab_size)  仅最后一个位置
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)

        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        # 因果掩码（序列长度>1时才需要）
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            # 推理时prefill：key来自cache（包含历史），需要扩展mask
            if use_cache and start_pos > 0:
                # query位置 [start_pos, end_pos)，key位置 [0, end_pos)
                # 只需要对query内部的因果约束，对历史key全部可见
                prefix_mask = torch.zeros(seqlen, start_pos, device=tokens.device)
                mask = torch.cat([prefix_mask, mask], dim=1)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask, start_pos, use_cache)

        h = self.norm(h)

        if use_cache:
            # 推理模式：只返回最后一个token位置的logits
            logits = self.head(h[:, -1, :])
        else:
            # 训练模式：返回所有位置的logits
            logits = self.head(h)

        return logits
