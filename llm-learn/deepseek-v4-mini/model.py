"""
DeepSeek-V4-Mini: 基于 DeepSeek-V4-Pro 架构的轻量化单卡训练版本

与原版的主要改动：
- 去掉 FP8/FP4 量化，全部使用 FP16
- 去掉多 GPU 分布式并行（单卡）
- 去掉 Triton kernel 依赖（sparse_attn / hc_split_sinkhorn / act_quant 等）
- Hyper-Connections 用纯 PyTorch Sinkhorn 迭代实现
- 稀疏 Attention 改为 Dense Attention（保留 Compressor KV 压缩，去掉 Indexer）
- 去掉 Indexer（依赖 rotate_activation / fp4_act_quant）
- 支持训练（causal LM loss + MTP 辅助 loss）和推理（KV cache）两种模式
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal, List
from functools import lru_cache

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    """模型超参数，字段名与 config JSON 键名对应。"""
    max_batch_size: int = 4
    max_seq_len: int = 2048
    vocab_size: int = 10000
    dim: int = 896
    moe_inter_dim: int = 512
    n_layers: int = 8
    n_hash_layers: int = 1           # 前 n_hash_layers 层使用 hash-based 专家路由
    n_mtp_layers: int = 1            # Multi-Token Prediction 辅助层数
    n_heads: int = 14
    # MoE
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.0
    swiglu_limit: float = 10.0       # SwiGLU 激活值截断阈值（0=不截断）
    # MLA attention
    q_lora_rank: int = 256
    head_dim: int = 128              # 每头的完整维度（nope + rope）
    rope_head_dim: int = 32          # RoPE 部分的维度
    norm_eps: float = 1e-6
    o_groups: int = 2                # 输出投影的分组数（低秩输出投影）
    o_lora_rank: int = 128           # 输出低秩投影的 rank
    window_size: int = 64            # 滑动窗口 attention 窗口大小
    compress_ratios: Tuple[int, ...] = (0, 0, 4, 0, 4, 0, 4, 0)  # 每层的 KV 压缩比
    # YaRN RoPE
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0        # >0 时对压缩层启用 YaRN
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    # Hyper-Connections
    hc_mult: int = 2                 # HC 副本数（原版 4，Mini 版 2 节省显存）
    hc_sinkhorn_iters: int = 5       # Sinkhorn 迭代次数（原版 20，Mini 版 5）
    hc_eps: float = 1e-6


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
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight.float() * x).to(dtype)


# ---------------------------------------------------------------------------
# Linear（FP16，单卡）
# ---------------------------------------------------------------------------

class Linear(nn.Module):
    """标准 FP16 线性层，无量化，无分布式切分。"""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE / YaRN)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def precompute_freqs_cis(
        dim: int,
        seqlen: int,
        original_seq_len: int,
        base: float,
        factor: float,
        beta_fast: int,
        beta_slow: int,
) -> torch.Tensor:
    """预计算 RoPE 旋转位置编码（支持 YaRN 长度外推）。
    返回 shape=(seqlen, dim//2) 的复数张量。
    """
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

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """将 RoPE 旋转位置编码应用到 x 上，返回新 tensor（非 in-place）。
    x: (..., seq_len, head_dim) 或 (..., seq_len, n_heads, rope_dim)
    freqs_cis: (seq_len, rope_dim//2) 复数
    """
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        # (B, S, rope_dim//2)
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        # (B, S, H, rope_dim//2)
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    return torch.view_as_real(x_complex * freqs_cis).flatten(-2).type_as(x)


# ---------------------------------------------------------------------------
# Hyper-Connections：纯 PyTorch Sinkhorn 实现
# ---------------------------------------------------------------------------

def sinkhorn_split(
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        iters: int,
        eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    纯 PyTorch 实现，替代原版 hc_split_sinkhorn Triton kernel。

    Args:
        mixes:    (..., mix_hc) — Block.hc_pre 输出的混合特征，
                  其中 mix_hc = (2 + hc_mult) * hc_mult（与原版一致）
        hc_scale: (3,) — pre/post/comb 三部分各自的缩放系数
        hc_base:  (mix_hc,) — 各部分的偏置（按顺序排列：pre[:hc_mult], post[hc_mult:2*hc_mult], comb[2*hc_mult:]）
        hc_mult:  HC 副本数
        iters:    Sinkhorn 迭代次数
        eps:      数值稳定 epsilon

    Returns:
        pre:  (..., hc_mult)          — 对 hc 副本的加权系数（行归一化）
        post: (..., hc_mult)          — 扩展时各副本的权重（sigmoid）
        comb: (..., hc_mult, hc_mult) — 副本间的组合矩阵（softmax）
    """
    # mixes: (..., mix_hc)  其中 mix_hc = (2 + hc_mult) * hc_mult
    # 例：hc_mult=2 → mix_hc=8

    # 拆分三段
    pre_logits  = mixes[..., :hc_mult]              # (..., hc_mult)
    post_logits = mixes[..., hc_mult : 2 * hc_mult] # (..., hc_mult)
    comb_logits = mixes[..., 2 * hc_mult :]         # (..., hc_mult * hc_mult)

    # --- pre: Sinkhorn 归一化 ---
    log_pre = pre_logits * hc_scale[0] + hc_base[:hc_mult]  # (..., hc_mult)
    for _ in range(iters):
        log_pre = log_pre - log_pre.logsumexp(dim=-1, keepdim=True)  # 行归一化（softmax in log-space）
    pre = log_pre.exp() + eps                                # (..., hc_mult)，归一化后加 eps 防止 0

    # --- post: sigmoid ---
    post = torch.sigmoid(post_logits * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]) + eps
    # post: (..., hc_mult)

    # --- comb: softmax ---
    comb_bias   = hc_base[2 * hc_mult:].view(hc_mult, hc_mult)             # (hc_mult, hc_mult)
    comb_logits = comb_logits.view(*comb_logits.shape[:-1], hc_mult, hc_mult)
    # comb_logits: (..., hc_mult, hc_mult)
    comb = torch.softmax(comb_logits * hc_scale[2] + comb_bias, dim=-1)    # (..., hc_mult, hc_mult)

    return pre, post, comb


# ---------------------------------------------------------------------------
# Compressor：KV Cache 压缩（保留，去掉量化 kernel）
# ---------------------------------------------------------------------------

class Compressor(nn.Module):
    """通过 gated pooling 把连续 compress_ratio 个 token 的 KV 压缩为 1 个。
    保留原版逻辑，去掉 act_quant/fp4_act_quant 量化调用，直接存 FP16 KV。
    """

    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 128):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        # ratio=4 时使用重叠压缩（overlap），ratio=128 时不使用
        self.overlap = compress_ratio == 4
        coff = 1 + self.overlap  # overlap=True → coff=2，否则 coff=1

        self.ape = nn.Parameter(torch.zeros(compress_ratio, coff * head_dim))
        self.wkv = Linear(self.dim, coff * head_dim)
        self.wgate = Linear(self.dim, coff * head_dim)
        self.norm = RMSNorm(head_dim, args.norm_eps)
        self.kv_cache: Optional[torch.Tensor] = None  # 由 Attention 外部赋值

        # decode 阶段的增量状态 buffer
        self.register_buffer(
            "kv_state",
            torch.zeros(args.max_batch_size, coff * compress_ratio, coff * head_dim),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full((args.max_batch_size, coff * compress_ratio, coff * head_dim), float("-inf")),
            persistent=False,
        )
        self.freqs_cis: Optional[torch.Tensor] = None  # 由 Attention 外部赋值

    def overlap_transform(self, tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        """重叠压缩的窗口变换：[b,s,r,2d] → [b,s,2r,d]"""
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int) -> Optional[torch.Tensor]:
        """
        x: (B, S, dim)
        start_pos: 当前序列的起始位置（prefill 时为 0，decode 时为当前位置）
        返回压缩后的 KV token，写入 self.kv_cache；如果本步不需要压缩则返回 None。
        """
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype

        kv    = self.wkv(x)     # (B, S, coff*head_dim)  coff=2(overlap) 或 1
        score = self.wgate(x)   # (B, S, coff*head_dim)

        if start_pos == 0:
            # --- prefill 阶段 ---
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio     # 不足一个 ratio 的尾部 token 数
            cutoff    = seqlen - remainder # 能整除 ratio 的有效长度
            offset    = ratio if overlap else 0

            if overlap and cutoff >= ratio:
                # 保存最后一个 ratio 的 kv 到 kv_state 前半部分，供下一块 overlap 用
                self.kv_state[:bsz, :ratio]   = kv[:, cutoff - ratio: cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio: cutoff] + self.ape

            if remainder > 0:
                # 尾部 token 存入 kv_state，本轮不压缩
                kv, self.kv_state[:bsz, offset: offset + remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset: offset + remainder]   = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]

            # (B, cutoff, coff*d) → (B, cutoff//ratio, ratio, coff*d)
            kv    = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape  # + position encoding

            if overlap:
                kv    = self.overlap_transform(kv, 0.0)          # (B, cutoff//ratio, 2*ratio, d)
                score = self.overlap_transform(score, float("-inf"))

            # gated pooling：softmax 加权平均
            kv = (kv * score.softmax(dim=2)).sum(dim=2)          # (B, cutoff//ratio, coff*d 或 d)

        else:
            # --- decode 阶段（每次处理 1 个 token）---
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score = score + self.ape[start_pos % ratio]          # (B, 1, coff*d)

            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio]   = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_s = torch.cat([self.kv_state[:bsz, :ratio, :d],   self.kv_state[:bsz, ratio:, d:]],   dim=1)
                    sc_s = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv   = (kv_s * sc_s.softmax(dim=1)).sum(dim=1, keepdim=True)  # (B, 1, d)
                    self.kv_state[:bsz, :ratio]   = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio]   = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
                    # kv: (B, 1, d)

        if not should_compress:
            return None

        kv = self.norm(kv.to(dtype))   # (B, S//ratio 或 1, head_dim)  RMSNorm

        # 施加 RoPE（compress kv 使用压缩层自己的 freqs_cis）
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]              # (S//ratio, rope_dim//2) 复数
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)  # (1, rope_dim//2)
        kv = torch.cat([kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], freqs_cis)], dim=-1)

        # 写入 kv_cache（不做量化，直接存 fp16）
        if not self.training:
            if start_pos == 0:
                self.kv_cache[:bsz, :seqlen // ratio] = kv  # kv_cache: (B, max_seq//ratio, head_dim)
            else:
                self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)  # 写入对应压缩位置

        return kv  # (B, S//ratio 或 1, head_dim)


# ---------------------------------------------------------------------------
# Attention（MLA + 滑动窗口 + 可选 KV 压缩，Dense Attention）
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head Latent Attention (MLA)，支持滑动窗口 + 可选 KV 压缩。
    稀疏 Attention 改为标准 Dense Attention（F.scaled_dot_product_attention）。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        # attention sink：对第 0 个 token 额外保留注意力（数值稳定）
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads))

        # Q 低秩投影
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)

        # KV 共享投影（MLA：所有 head 共享一组 KV）
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # 输出低秩投影（分组）
        self.wo_a = Linear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank)
        self.wo_b = Linear(self.n_groups * args.o_lora_rank, self.dim)

        self.softmax_scale = self.head_dim ** -0.5

        # KV cache：滑动窗口部分 + 压缩部分
        compress_slots = args.max_seq_len // self.compress_ratio if self.compress_ratio else 0
        kv_cache_size = args.window_size + compress_slots
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim),
            persistent=False,
        )

        # Compressor（如果该层有 KV 压缩）
        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)

        # RoPE freqs（压缩层用 compress_rope_theta + YaRN，普通层用 rope_theta）
        if self.compress_ratio:
            orig_seq_len = args.original_seq_len
            rope_base    = args.compress_rope_theta
        else:
            orig_seq_len = 0
            rope_base    = args.rope_theta

        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            orig_seq_len,
            rope_base,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        x: (B, S, dim)
        start_pos: 当前 token 的起始位置（prefill=0，decode=当前位置）
        返回: (B, S, dim)
        """
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]  # (S, rope_head_dim//2) 复数
        win = self.window_size
        rd  = self.rope_head_dim

        # --- 初始化 Compressor 的 kv_cache 引用（惰性赋值）---
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache  = self.kv_cache[:, win:]  # 压缩 KV 存在滑窗之后
            self.compressor.freqs_cis = self.freqs_cis

        # ---- Q ----
        # (B, S, dim) → wq_a → (B, S, q_lora_rank) → norm → wq_b → (B, S, n_heads*head_dim)
        q = self.wq_b(self.q_norm(self.wq_a(x))).unflatten(-1, (self.n_heads, self.head_dim))
        # q: (B, S, n_heads, head_dim)
        # Q-Norm（L2 normalize over head_dim）
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        # 施加 RoPE 到 rope 部分：(B, S, n_heads, rope_head_dim)
        q = torch.cat([q[..., :-rd], apply_rotary_emb(q[..., -rd:], freqs_cis)], dim=-1)

        # ---- KV（MLA：单组共享 KV）----
        # (B, S, dim) → wkv → (B, S, head_dim)
        kv = self.wkv(x)
        kv = self.kv_norm(kv)                   # (B, S, head_dim)  RMSNorm
        kv = torch.cat([kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], freqs_cis)], dim=-1)  # rope 部分旋转

        # ---- 更新滑动窗口 KV cache ----
        if not self.training:
            if start_pos == 0:
                # prefill：将最后 window_size 个 token 循环写入 kv_cache[:, :win]
                if seqlen <= win:
                    self.kv_cache[:bsz, :seqlen] = kv                   # (B, S, head_dim)
                else:
                    cutoff = seqlen % win
                    self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = \
                        kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            else:
                # decode：循环写入，位置 = start_pos % win
                self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)    # (B, head_dim)

        # ---- KV 压缩（如果该层有 Compressor）----
        if self.compress_ratio:
            kv_compress = self.compressor(x, start_pos)
            # kv_compress: (B, S//ratio 或 1, head_dim) 或 None

        # ---- 构建 K、V 用于 attention 计算 ----
        if start_pos == 0:
            # prefill：用完整序列 kv
            k_full = kv                                              # (B, S, head_dim)
            if self.compress_ratio and kv_compress is not None:
                k_full = torch.cat([kv, kv_compress], dim=1)        # (B, S + S//ratio, head_dim)
            v_full = k_full  # MLA 共享 KV
        else:
            # decode：从 kv_cache 取历史 KV
            end_pos  = start_pos + seqlen
            win_end  = min(end_pos, win)
            k_win    = self.kv_cache[:bsz, :win_end]                # (B, T_win, head_dim)

            if self.compress_ratio:
                compress_end = end_pos // self.compress_ratio
                k_compress   = self.kv_cache[:bsz, win: win + compress_end]  # (B, T_comp, head_dim)
                k_full       = torch.cat([k_win, k_compress], dim=1)          # (B, T_win+T_comp, head_dim)
            else:
                k_full = k_win
            v_full = k_full

        # ---- Attention 计算（Dense SDPA） ----
        # q:   (B, S, n_heads, head_dim) → transpose → (B, n_heads, S, head_dim)
        # k/v: (B, T, head_dim) → unsqueeze+expand → (B, n_heads, T, head_dim)
        q_t = q.transpose(1, 2)                                             # (B, n_heads, S, head_dim)
        T   = k_full.size(1)
        k_t = k_full.unsqueeze(1).expand(-1, self.n_heads, -1, -1)         # (B, n_heads, T, head_dim)
        v_t = v_full.unsqueeze(1).expand(-1, self.n_heads, -1, -1)         # (B, n_heads, T, head_dim)

        # 因果 mask（仅 prefill 且 seqlen > 1 时需要）
        attn_mask = None
        if start_pos == 0 and seqlen > 1:
            causal  = torch.zeros(seqlen, T, device=x.device, dtype=x.dtype)
            offset  = T - seqlen
            mask_tri = torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool).triu(1)
            full_mask = torch.zeros(seqlen, T, device=x.device, dtype=torch.bool)
            full_mask[:, offset:] = mask_tri
            causal   = causal.masked_fill(full_mask, float("-inf"))
            attn_mask = causal.unsqueeze(0).unsqueeze(0)                    # (1, 1, S, T)

        o = F.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=attn_mask, scale=self.softmax_scale,
        )  # (B, n_heads, S, head_dim)

        # 反旋转 RoPE（attention output 需要 de-rotate rope 维度）
        o_t = o.transpose(1, 2)                                             # (B, S, n_heads, head_dim)
        o_t = torch.cat([o_t[..., :-rd], apply_rotary_emb(o_t[..., -rd:], freqs_cis, inverse=True)], dim=-1)

        # ---- 输出低秩投影（分组） ----
        # (B, S, n_heads, head_dim) → view → (B, S, n_groups, n_heads//n_groups * head_dim)
        o_grouped = o_t.view(bsz, seqlen, self.n_groups, -1)
        # wo_a.weight: (n_groups * o_lora_rank, n_heads//n_groups * head_dim)
        # → view → (n_groups, o_lora_rank, n_heads//n_groups * head_dim)
        wo_a_w = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        # einsum: (B,S,n_groups,d) × (n_groups,o_lora,d) → (B,S,n_groups,o_lora)
        o_lora = torch.einsum("bsgd,grd->bsgr", o_grouped, wo_a_w)         # (B, S, n_groups, o_lora_rank)
        out = self.wo_b(o_lora.flatten(2))                                   # (B, S, dim)
        return out

    def reset_kv_cache(self):
        """推理前清零 KV cache 和 Compressor 状态。"""
        self.kv_cache.zero_()
        if self.compress_ratio:
            self.compressor.kv_cache = None
            self.compressor.kv_state.zero_()
            self.compressor.score_state.fill_(float("-inf"))


# ---------------------------------------------------------------------------
# Gate（MoE 路由，支持 hash-based 和 score-based）
# ---------------------------------------------------------------------------

class Gate(nn.Module):
    """MoE 路由门控。
    前 n_hash_layers 层使用 hash-based 路由（token ID → 专家索引）；
    其余层使用 score-based 路由（sqrtsoftplus/softmax/sigmoid）。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.hash:
            # tid2eid: token_id → [topk 个专家索引]
            self.tid2eid = nn.Parameter(
                torch.zeros(args.vocab_size, args.n_activated_experts, dtype=torch.int32),
                requires_grad=False,
            )
            self.bias = None
        else:
            # bias 用于 load balancing，不影响路由权重
            self.bias = nn.Parameter(torch.zeros(args.n_routed_experts))

    def forward(
            self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, dim)  N = batch * seq_len
        input_ids: (N,)  hash 路由时使用
        返回: weights (N, topk), indices (N, topk)
        """
        scores = F.linear(x.float(), self.weight.float())  # (N, n_experts)

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).sqrt()

        original_scores = scores

        if self.bias is not None:
            scores = scores + self.bias  # bias 只影响选择，不影响权重

        if self.hash:
            assert input_ids is not None
            indices = self.tid2eid[input_ids]  # (N, topk) int32
        else:
            indices = scores.topk(self.topk, dim=-1)[1]  # (N, topk)

        weights = original_scores.gather(1, indices.long())

        if self.score_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        weights = (weights * self.route_scale).to(x.dtype)
        return weights, indices


# ---------------------------------------------------------------------------
# Expert & MoE
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    """单个 MoE 专家：SwiGLU FFN（w1, w2, w3），支持激活截断。"""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (n_tokens, dim)
        gate = self.w1(x).float()
        up   = self.w3(x).float()
        if self.swiglu_limit > 0:
            up   = torch.clamp(up,   min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(self.w2.weight.dtype))


class MoE(nn.Module):
    """Mixture-of-Experts：gate 路由 + n_routed_experts 路由专家 + 1 共享专家。"""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.gate = Gate(layer_id, args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
            for _ in range(args.n_routed_experts)
        ])
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, S, dim)
        shape   = x.size()
        x_flat  = x.reshape(-1, self.dim)         # (N, dim)  N = B*S
        weights, indices = self.gate(x_flat, input_ids.reshape(-1))
        # weights: (N, topk), indices: (N, topk)

        y = torch.zeros_like(x_flat, dtype=torch.float32)
        counts = torch.bincount(indices.flatten().long(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices.long() == i)
            y[idx] += self.experts[i](x_flat[idx], weights[idx, top, None]).float()

        y += self.shared_experts(x_flat).float()
        return y.to(x.dtype).view(shape)


# ---------------------------------------------------------------------------
# Block（Transformer 层，使用 Hyper-Connections）
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Transformer 解码层，使用 Hyper-Connections（HC）残差机制。
    HC 维护 hc_mult 份隐层副本，通过 Sinkhorn 归一化做加权混合。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn  = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm  = RMSNorm(args.dim, self.norm_eps)

        self.hc_mult          = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps           = args.hc_eps

        mix_hc = (2 + args.hc_mult) * args.hc_mult  # 混合特征维度数
        hc_dim = args.hc_mult * args.dim             # HC 展开维度

        # Attention HC 参数
        self.hc_attn_fn    = nn.Parameter(torch.zeros(mix_hc, hc_dim))
        self.hc_ffn_fn     = nn.Parameter(torch.zeros(mix_hc, hc_dim))
        self.hc_attn_base  = nn.Parameter(torch.zeros(mix_hc))
        self.hc_ffn_base   = nn.Parameter(torch.zeros(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.ones(3))
        self.hc_ffn_scale  = nn.Parameter(torch.ones(3))

    def hc_pre(
            self,
            x: torch.Tensor,
            hc_fn: torch.Tensor,
            hc_scale: torch.Tensor,
            hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        HC 预处理：将 hc_mult 份副本加权混合为 1 份，供子层计算。
        x: (B, S, hc_mult, dim)
        返回: y (B, S, dim)，以及 post/comb 用于 hc_post
        """
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()                              # (B, S, hc_mult*dim)
        rsqrt  = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        # hc_fn: (mix_hc, hc_mult*dim)  linear 等价于 x_flat @ hc_fn^T
        mixes  = F.linear(x_flat, hc_fn.float()) * rsqrt          # (B, S, mix_hc)  其中 mix_hc=(2+hc_mult)*hc_mult

        pre, post, comb = sinkhorn_split(
            mixes, hc_scale.float(), hc_base.float(),
            self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps,
        )
        # pre:  (B, S, hc_mult)
        # post: (B, S, hc_mult)
        # comb: (B, S, hc_mult, hc_mult)

        # 加权求和 hc_mult 份副本 → 单份
        y = torch.sum(pre.unsqueeze(-1) * x.float(), dim=2)       # (B, S, dim)
        return y.to(dtype), post, comb

    def hc_post(
            self,
            x: torch.Tensor,
            residual: torch.Tensor,
            post: torch.Tensor,
            comb: torch.Tensor,
    ) -> torch.Tensor:
        """
        HC 后处理：将子层输出扩展回 hc_mult 份副本。
        x:        (B, S, dim)          子层输出
        residual: (B, S, hc_mult, dim) 更新前的 hc 状态
        post:     (B, S, hc_mult)      扩展权重（sigmoid）
        comb:     (B, S, hc_mult, hc_mult) 副本间组合矩阵（softmax）
        返回: (B, S, hc_mult, dim)
        """
        # post.unsqueeze(-1): (B, S, hc_mult, 1)
        # x.unsqueeze(-2):    (B, S, 1, dim)
        # → 广播乘积: (B, S, hc_mult, dim)  —— 子层输出的 hc_mult 份加权副本
        #
        # comb.unsqueeze(-1):      (B, S, hc_mult, hc_mult, 1)
        # residual.unsqueeze(-2):  (B, S, 1, hc_mult, dim)
        # → 乘积 sum(dim=2):       (B, S, hc_mult, dim)  —— 旧副本间的线性组合
        y = (post.unsqueeze(-1) * x.unsqueeze(-2)
             + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2))
        # y: (B, S, hc_mult, dim)
        return y.type_as(x)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            input_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        x: (B, S, hc_mult, dim)
        返回: (B, S, hc_mult, dim)
        """
        # --- Attention 子层 ---
        residual = x
        y, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        y = self.attn_norm(y)
        y = self.attn(y, start_pos)
        x = self.hc_post(y, residual, post, comb)

        # --- FFN/MoE 子层 ---
        residual = x
        y, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        y = self.ffn_norm(y)
        y = self.ffn(y, input_ids)
        x = self.hc_post(y, residual, post, comb)

        return x


# ---------------------------------------------------------------------------
# HC Head（最终从 hc_mult 份副本输出 logits）
# ---------------------------------------------------------------------------

class HCHead(nn.Module):
    """从 hc_mult 份 HC 副本加权合并后输出 logits。"""

    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def hc_collapse(
            self,
            x: torch.Tensor,
            hc_fn: torch.Tensor,
            hc_scale: torch.Tensor,
            hc_base: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, S, hc_mult, dim) → (B, S, dim) 加权合并 HC 副本。
        hc_fn:   (hc_mult, hc_mult*dim)  — 注意：HCHead 的 collapse 只有 hc_mult 行
        hc_scale: (1,)
        hc_base:  (hc_mult,)
        """
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()                          # (B, S, hc_mult*dim)
        rsqrt  = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes  = F.linear(x_flat, hc_fn.float()) * rsqrt      # (B, S, hc_mult)
        # sigmoid 激活：每个副本的权重 ∈ (0, 1)
        pre    = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + self.hc_eps  # (B, S, hc_mult)
        # pre.unsqueeze(-1): (B, S, hc_mult, 1) × x: (B, S, hc_mult, dim) → sum(dim=2) → (B, S, dim)
        y = torch.sum(pre.unsqueeze(-1) * x.float(), dim=2)   # (B, S, dim)
        return y.to(dtype)

    def forward(
            self,
            x: torch.Tensor,
            hc_fn: torch.Tensor,
            hc_scale: torch.Tensor,
            hc_base: torch.Tensor,
            norm: RMSNorm,
    ) -> torch.Tensor:
        """
        x: (B, S, hc_mult, dim)
        返回 logits: (B, S, vocab_size) 或推理时 (B, vocab_size)
        """
        x = self.hc_collapse(x, hc_fn, hc_scale, hc_base)  # (B, S, dim)
        x = norm(x)
        logits = F.linear(x.float(), self.weight.float())   # (B, S, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# MTPBlock（Multi-Token Prediction 辅助层）
# ---------------------------------------------------------------------------

class MTPBlock(Block):
    """Multi-Token Prediction Block：在主干隐层基础上预测下一个 token。
    共享 embed 和 head 与主干 Transformer，额外有 e_proj/h_proj 融合层。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        self.e_proj = Linear(args.dim, args.dim)
        self.h_proj = Linear(args.dim, args.dim)
        self.enorm  = RMSNorm(args.dim, args.norm_eps)
        self.hnorm  = RMSNorm(args.dim, args.norm_eps)
        self.norm   = RMSNorm(args.dim, args.norm_eps)

        # HC head 专用参数（用于 MTP 输出 logits）
        hc_dim = args.hc_mult * args.dim
        self.hc_head_fn    = nn.Parameter(torch.zeros(args.hc_mult, hc_dim))
        self.hc_head_base  = nn.Parameter(torch.zeros(args.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.ones(1))

        # 由 Transformer 外部赋值
        self.embed: Optional[nn.Embedding] = None
        self.head:  Optional[HCHead]       = None

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, S, hc_mult, dim) — 主干最后一层的 HC 隐层状态
        input_ids: (B, S) — 当前步的输入 token（用于 MTP 融合）
        返回 logits: (B, S, vocab_size)
        """
        assert self.embed is not None and self.head is not None

        e = self.embed(input_ids)  # (B, S, dim)  token embedding
        e = self.enorm(e)          # (B, S, dim)  RMSNorm

        # 取 hc 副本的均值作为主干隐层表示
        h = x.mean(dim=2)          # (B, S, dim)  hc_mult 份副本均值
        h = self.hnorm(h)          # (B, S, dim)  RMSNorm

        # 融合 embed 和主干隐层，扩展为 hc_mult 份
        fused = self.e_proj(e) + self.h_proj(h)                        # (B, S, dim)
        fused = fused.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)    # (B, S, hc_mult, dim)
        fused = fused.contiguous()

        # 过 Block（Attention + MoE），同父类 Block.forward
        out = super().forward(fused, start_pos, input_ids)             # (B, S, hc_mult, dim)

        # 输出 logits（共享主干 head）
        logits = self.head(out, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        return logits  # (B, S, vocab_size)


# ---------------------------------------------------------------------------
# Transformer（完整模型）
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """DeepSeek-V4-Mini 完整 Transformer 模型。

    支持两种使用模式：
    1. 训练模式（start_pos=0，use_cache=False）：输入 (B, T)，输出 logits (B, T, V)
    2. 推理模式（use_cache=True）：逐步生成，内置 KV cache
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.vocab_size = args.vocab_size

        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = HCHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)

        # MTP 层
        self.mtp = nn.ModuleList()
        for layer_id in range(args.n_mtp_layers):
            mtp_block = MTPBlock(args.n_layers + layer_id, args)
            mtp_block.embed = self.embed  # 共享 embed
            mtp_block.head  = self.head   # 共享 head
            self.mtp.append(mtp_block)

        # HC head 参数（主干用）
        hc_dim = args.hc_mult * args.dim
        self.hc_head_fn    = nn.Parameter(torch.zeros(args.hc_mult, hc_dim))
        self.hc_head_base  = nn.Parameter(torch.zeros(args.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (Linear, nn.Embedding, HCHead)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # nn.Parameter 需要通过 named_parameters 遍历初始化
        for name, param in self.named_parameters():
            if "hc_" in name and "scale" in name:
                nn.init.ones_(param)
            elif "hc_" in name and ("fn" in name or "base" in name):
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif "hc_" in name and "base" in name:
                nn.init.zeros_(param)
            elif "attn_sink" in name:
                nn.init.zeros_(param)

    def reset_kv_cache(self):
        """推理前清零所有层的 KV cache。"""
        for layer in self.layers:
            layer.attn.reset_kv_cache()
        for mtp in self.mtp:
            mtp.attn.reset_kv_cache()

    def forward(
            self,
            input_ids: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False,
            return_hidden: bool = False,
    ):
        """
        训练模式（use_cache=False）：
            input_ids: (B, T) → logits: (B, T, vocab_size)
        推理模式（use_cache=True）：
            input_ids: (B, T) → logits: (B, vocab_size)（只返回最后位置）
        return_hidden=True 时额外返回最后一层隐层状态 h，供 MTP 复用。

        注意：MTP 层仅在训练时被 train.py 单独调用，forward 不调用 mtp。
        """
        # embed: (B, T) → (B, T, dim)
        h = self.embed(input_ids)

        # 扩展为 hc_mult 份 HC 副本
        # unsqueeze(2): (B, T, 1, dim) → expand: (B, T, hc_mult, dim)
        h = h.unsqueeze(2).expand(-1, -1, self.args.hc_mult, -1).contiguous()

        # N 个 Block：每层输入输出均为 (B, T, hc_mult, dim)
        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, input_ids)

        # HCHead 从 hc_mult 副本合并：(B, T, hc_mult, dim) → (B, T, vocab_size)
        logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)

        if use_cache:
            # 推理模式：只返回最后一个 token 的 logits
            return logits[:, -1, :]  # (B, vocab_size)
        if return_hidden:
            return logits, h
        return logits  # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# 快速测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16

    args = ModelArgs(
        n_layers=4,
        n_hash_layers=1,
        n_mtp_layers=1,
        dim=128,
        moe_inter_dim=64,
        n_heads=4,
        n_routed_experts=4,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=16,
        window_size=16,
        compress_ratios=(0, 0, 4, 0),
        hc_mult=2,
        hc_sinkhorn_iters=3,
        max_seq_len=64,
        max_batch_size=2,
        vocab_size=10000,
    )

    model = Transformer(args).to(device=device, dtype=dtype)
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 训练模式前向
    x = torch.randint(0, args.vocab_size, (2, 16), device=device)
    logits = model(x, start_pos=0, use_cache=False)
    print(f"训练模式 logits shape: {logits.shape}")  # (2, 16, 1024)

    # MTP 前向
    h = model.embed(x).unsqueeze(2).expand(-1, -1, args.hc_mult, -1).contiguous()
    for layer in model.layers:
        h = layer(h, 0, x)
    mtp_logits = model.mtp[0](h, 0, x)
    print(f"MTP logits shape: {mtp_logits.shape}")  # (2, 16, 1024)

    # 推理模式：prefill
    model.reset_kv_cache()
    logits_inf = model(x, start_pos=0, use_cache=True)
    print(f"推理prefill logits shape: {logits_inf.shape}")  # (2, 1024)

    # 推理模式：decode
    x_next = torch.randint(0, args.vocab_size, (2, 1), device=device)
    logits_dec = model(x_next, start_pos=16, use_cache=True)
    print(f"推理decode logits shape: {logits_dec.shape}")  # (2, 1024)

    print("✓ 所有测试通过")
