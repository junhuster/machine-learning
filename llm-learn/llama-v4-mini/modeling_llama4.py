# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# 文件说明：Llama4 文本语言模型的 PyTorch 实现（本地实现，不依赖 HuggingFace）
#
# 整体架构（从底层到顶层）：
#   归一化层：Llama4TextRMSNorm（带权重）、Llama4TextL2Norm（无权重，用于QK）
#   FFN 层：  Llama4TextMLP（稠密）、Llama4TextExperts + Llama4Router → Llama4TextMoe（稀疏MoE）
#   位置编码：Llama4TextRotaryEmbedding → apply_rotary_emb（RoPE，复数域旋转）
#   注意力：  Llama4TextAttention（GQA + 可选RoPE + 可选QKNorm）
#   解码层：  Llama4TextDecoderLayer（Pre-Norm，注意力 + FFN/MoE）
#   主干模型：Llama4TextModel（Embedding + N层解码器 + 最终归一化）
#   因果LM：  Llama4ForCausalLM（主干 + lm_head，支持训练loss和generate()）
# ============================================================================

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration_llama4 import Llama4TextConfig

logger = logging.getLogger(__name__)


# ============================================================================
# 本地实现：激活函数
# ============================================================================

def _silu(x):
    return F.silu(x)

ACT2FN = {
    "silu": _silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


# ============================================================================
# 本地实现：输出数据类
# ============================================================================

@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[object] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    def __getitem__(self, idx):
        return (self.last_hidden_state, self.past_key_values,
                self.hidden_states, self.attentions)[idx]


@dataclass
class CausalLMOutputWithPast:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[object] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    def __getitem__(self, idx):
        return (self.loss, self.logits, self.past_key_values,
                self.hidden_states, self.attentions)[idx]


# ============================================================================
# 本地实现：KV Cache
# ============================================================================

class DynamicCache:
    """简单的动态 KV Cache，每层存储 (key, value) tensor。"""

    def __init__(self, config=None):
        self._cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 扩展列表到 layer_idx
        while len(self._cache) <= layer_idx:
            self._cache.append(None)

        if self._cache[layer_idx] is None:
            self._cache[layer_idx] = (key_states, value_states)
        else:
            past_key, past_value = self._cache[layer_idx]
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            self._cache[layer_idx] = (key_states, value_states)

        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._cache) and self._cache[layer_idx] is not None:
            return self._cache[layer_idx][0].shape[2]
        return 0


# ============================================================================
# 本地实现：Attention Mask 构建
# ============================================================================

def create_causal_mask(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[DynamicCache],
    position_ids: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    构建标准因果掩码（下三角 mask），shape: (batch, 1, seq_q, seq_k)。
    已有 past_key_values 时，seq_k = past + seq_q。
    """
    batch, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    total_len = past_len + seq_len

    # 下三角因果掩码：(seq_q, total_len)
    mask = torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=past_len + 1)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_q, total_len)

    # 应用 padding mask
    if attention_mask is not None and attention_mask.dim() == 2:
        # attention_mask: (batch, total_len)，0=pad，1=有效
        pad_mask = (1.0 - attention_mask[:, None, None, :].float()) * float("-inf")
        pad_mask = pad_mask.nan_to_num(nan=0.0)
        mask = mask + pad_mask

    return mask.expand(batch, 1, seq_len, total_len)


def create_chunked_causal_mask(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[DynamicCache],
    position_ids: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    构建分块因果掩码（NoPE 层使用局部窗口注意力）。
    每个 token 只能看到同一 chunk 内的历史 token。
    chunk_size = config.attention_chunk_size
    """
    batch, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    total_len = past_len + seq_len
    chunk_size = config.attention_chunk_size or total_len

    # 先构建标准因果掩码
    mask = torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=past_len + 1)

    # 再加 chunk 限制：超过 chunk_size 距离的位置也屏蔽
    for q_idx in range(seq_len):
        abs_q = past_len + q_idx
        chunk_start = (abs_q // chunk_size) * chunk_size
        # 屏蔽 chunk_start 之前的所有位置
        if chunk_start > 0:
            mask[q_idx, :chunk_start] = float("-inf")

    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, seq_len, total_len)

    if attention_mask is not None and attention_mask.dim() == 2:
        pad_mask = (1.0 - attention_mask[:, None, None, :].float()) * float("-inf")
        pad_mask = pad_mask.nan_to_num(nan=0.0)
        mask = mask + pad_mask

    return mask


# ============================================================================
# MoE（混合专家）相关模块
# ============================================================================

class Llama4TextExperts(nn.Module):
    """
    所有路由专家的参数容器，用批量矩阵乘法（bmm）一次性计算所有专家的输出。

    权重布局：
      gate_up_proj: (num_experts, hidden_size, 2 * expert_dim)
      down_proj:    (num_experts, expert_dim, hidden_size)

    前向流程（SwiGLU 激活）：
      1. hidden: (num_experts, tokens_per_expert, hidden_size)
      2. gate_up = bmm(hidden, gate_up_proj)  → (num_experts, tokens, 2*expert_dim)
      3. gate, up = chunk(2)
      4. output = bmm(up * act(gate), down_proj) → (num_experts, tokens, hidden_size)
      5. 展平回 (total_tokens, hidden_size)
    """
    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(self.gate_up_proj.shape[0], -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
        next_states = next_states.view(-1, self.hidden_size)
        return next_states


class Llama4TextMLP(nn.Module):
    """
    标准稠密前馈网络（SwiGLU 变体）。
    output = down_proj(act(gate_proj(x)) * up_proj(x))
    """
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.activation_fn(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# 归一化层
# ============================================================================

class Llama4TextL2Norm(nn.Module):
    """无可学习参数的 L2 归一化，用于 QK Norm。"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)


class Llama4TextRMSNorm(nn.Module):
    """
    RMS 归一化：output = (x / RMS(x)) * weight
    """
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


# ============================================================================
# MoE 路由器
# ============================================================================

class Llama4Router(nn.Linear):
    """
    MoE 路由器：为每个 token 打分，选出 top-k 专家。
    返回 (router_scores, router_logits)。
    """
    def __init__(self, config):
        super().__init__(config.hidden_size, config.num_local_experts, bias=False)
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states):
        router_logits = super().forward(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value)
        router_scores = torch.nn.functional.sigmoid(router_scores.float()).to(router_scores.dtype)
        return router_scores, router_logits


class Llama4TextMoe(nn.Module):
    """
    混合专家（Mixture of Experts）前馈层。

    结构：
      - router：为每个 token 选择 top-k 路由专家
      - experts：num_local_experts 个专家（批量 bmm 计算）
      - shared_expert：一个始终激活的稠密 MLP
    """
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(config)
        self.router = Llama4Router(config)
        self.shared_expert = Llama4TextMLP(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_scores, router_logits = self.router(hidden_states)

        routed_in = hidden_states.repeat(router_scores.shape[1], 1)
        routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
        routed_out = self.experts(routed_in)

        out = self.shared_expert(hidden_states)
        out.add_(routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim=0))
        return out, router_logits


# ============================================================================
# 旋转位置编码（RoPE）
# ============================================================================

class Llama4TextRotaryEmbedding(nn.Module):
    """
    旋转位置编码（RoPE）。
    inv_freq：逆频率向量，形状 (head_dim/2,)。
    forward 返回复数张量 freqs_cis。
    """
    def __init__(self, config: Llama4TextConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters.get("rope_type", "default")

        inv_freq, self.attention_scaling = self._compute_default_rope_parameters(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _compute_default_rope_parameters(config, device=None):
        base = config.rope_parameters.get("rope_theta", 500000.0)
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        计算旋转复数张量 freqs_cis。
        Args:
            x:            任意张量，仅用于获取 device
            position_ids: (batch, seq_len)
        Returns:
            freqs_cis: 复数张量 (batch, seq_len, head_dim/2)
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            freqs_cis = freqs_cis * self.attention_scaling

        return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 RoPE 旋转应用到 Query 和 Key 上。

    Args:
        xq, xk:    (batch, seq_len, num_heads, head_dim) 实数张量
        freqs_cis: (batch, seq_len, head_dim/2) 复数张量
    Returns:
        旋转后的 xq, xk，形状与输入相同
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs = freqs_cis[:, :, None, :]  # (batch, seq_len, 1, head_dim/2)
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================================
# 注意力辅助函数
# ============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 KV head 复制 n_rep 次，以匹配 Q head 数量（GQA 所需）。
    (batch, num_kv_heads, seq, head_dim) → (batch, num_q_heads, seq, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    标准 Scaled Dot-Product Attention（PyTorch 原生实现）。
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ============================================================================
# 注意力层
# ============================================================================

class Llama4TextAttention(nn.Module):
    """
    多头注意力，支持 GQA、RoPE、QK Norm、注意力温度调节。
    """

    def __init__(self, config: Llama4TextConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = config.no_rope_layers[layer_idx]

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[DynamicCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:
            query_states, key_states = apply_rotary_emb(
                query_states, key_states, position_embeddings.to(query_states.device)
            )

        if hasattr(self, "qk_norm"):
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        if self.attn_temperature_tuning and not self.use_rope:
            past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
            positions = torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_seen_tokens
            attn_scales = (
                torch.log1p(torch.floor((positions.float() + 1.0) / self.floor_scale)) * self.attn_scale + 1.0
            )
            attn_scales = attn_scales.view((1, input_shape[-1], 1, 1)).expand((*input_shape, 1, 1))
            query_states = (query_states * attn_scales).to(query_states.dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ============================================================================
# Transformer 解码层
# ============================================================================

class Llama4TextDecoderLayer(nn.Module):
    """
    单个 Transformer 解码层（Pre-Norm 架构）：
      x = x + Attention(RMSNorm(x))
      x = x + FFN/MoE(RMSNorm(x))
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Llama4TextAttention(config, layer_idx)
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:
            self.feed_forward = Llama4TextMoe(config)
        else:
            self.feed_forward = Llama4TextMLP(config, intermediate_size=config.intermediate_size_mlp)

        self.input_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = False,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + attention_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states.view(residual.shape)
        return hidden_states


# ============================================================================
# 主干模型：Llama4TextModel
# ============================================================================

class Llama4TextModel(nn.Module):
    """
    Llama4 文本编码器主干（不含 lm_head）。
    结构：embed_tokens → N × Llama4TextDecoderLayer → RMSNorm
    """

    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Llama4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Llama4TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self._init_weights()

    def _init_weights(self):
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, Llama4TextExperts):
                nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
                nn.init.normal_(module.down_proj, mean=0.0, std=std)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("必须且只能指定 input_ids 或 inputs_embeds 之一")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # 预先构建两种注意力掩码
        mask_kwargs = dict(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "chunked_attention": create_chunked_causal_mask(**mask_kwargs),
        }

        hidden_states = inputs_embeds
        freq_cis = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[:self.config.num_hidden_layers]):
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    causal_mask_mapping[self.config.layer_types[i]],
                    position_ids,
                    past_key_values,
                    use_cache,
                    freq_cis,
                    use_reentrant=False,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=freq_cis,
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ============================================================================
# 因果语言模型：Llama4ForCausalLM
# ============================================================================

class Llama4ForCausalLM(nn.Module):
    """
    带语言模型头的 Llama4，支持训练 loss 计算和自回归生成。

    结构：Llama4TextModel（主干） + lm_head（Linear，hidden_size → vocab_size）
    """

    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.config = config
        self.model = Llama4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重绑定：lm_head 与 embed_tokens 共享权重
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        hidden_states = outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # labels 已经是右移后的目标序列（由调用方处理），直接对齐计算
            # 忽略 label=-100 的位置（pad 位置）
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    def state_dict(self, **kwargs):
        return super().state_dict(**kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 200,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        eos_token_id=None,
        pad_token_id=None,
        num_beams: int = 1,
        **kwargs,
    ) -> torch.LongTensor:
        """
        自回归文本生成（贪心 / 采样，不含 beam search）。
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        past_key_values = DynamicCache(config=self.config)
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # 只输入最新的 token（利用 KV cache）
            cur_input = generated[:, -1:] if past_key_values.get_seq_length() > 0 else generated

            outputs = self.forward(
                input_ids=cur_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]  # (batch, vocab)
            past_key_values = outputs.past_key_values

            # 重复惩罚
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for token_id in set(generated[b].tolist()):
                        if logits[b, token_id] < 0:
                            logits[b, token_id] *= repetition_penalty
                        else:
                            logits[b, token_id] /= repetition_penalty

            # 采样或贪心
            if do_sample and temperature > 0:
                logits = logits / temperature
                # top_k 过滤
                if top_k > 0:
                    topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                    min_val = topk_vals[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < min_val, float("-inf"))
                # top_p 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) > top_p
                    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                    logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # 检查是否所有序列都生成了 eos
            for eid in eos_token_id:
                finished = finished | (next_token.squeeze(-1) == eid)
            if finished.all():
                break

        return generated


__all__ = [
    "Llama4TextModel",
    "Llama4ForCausalLM",
    "DynamicCache",
]
