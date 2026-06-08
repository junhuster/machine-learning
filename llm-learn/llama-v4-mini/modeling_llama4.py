# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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
# 文件说明：Llama4 文本语言模型的 PyTorch 实现
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

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_chunked_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    dynamic_rope_update,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from configuration_llama4 import Llama4TextConfig


logger = logging.get_logger(__name__)


# ============================================================================
# MoE（混合专家）相关模块
# ============================================================================

class Llama4TextExperts(nn.Module):
    """
    所有路由专家的参数容器，用批量矩阵乘法（bmm）一次性计算所有专家的输出。

    权重布局：
      gate_up_proj: (num_experts, hidden_size, 2 * expert_dim)
        — 将 gate 和 up 投影合并为一个参数，chunk(2) 后分别取 gate 和 up
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
        # 所有专家的 gate+up 投影合并为一个 3D 参数，第 0 维是专家索引
        self.gate_up_proj = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (num_experts * tokens_per_expert, hidden_size)
                           输入须已按专家分组排好顺序（由 Llama4TextMoe 的 repeat+mask 完成）
        Returns:
            (num_experts * tokens_per_expert, hidden_size)
        """
        # (num_experts*tokens_per_expert, hidden_size) -> (num_experts, tokens_per_expert, hidden_size)
        hidden_states = hidden_states.view(self.gate_up_proj.shape[0], -1, self.hidden_size)
        # bmm: (num_experts, tokens_per_expert, hidden_size) x (num_experts, hidden_size, 2*expert_dim)
        #   -> (num_experts, tokens_per_expert, 2*expert_dim)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        # chunk: 沿最后维度切成两半
        # gate: (num_experts, tokens_per_expert, expert_dim)
        # up:   (num_experts, tokens_per_expert, expert_dim)
        gate, up = gate_up.chunk(2, dim=-1)
        # SwiGLU: act(gate) * up -> (num_experts, tokens_per_expert, expert_dim)
        # bmm: (num_experts, tokens_per_expert, expert_dim) x (num_experts, expert_dim, hidden_size)
        #   -> (num_experts, tokens_per_expert, hidden_size)
        next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
        # (num_experts, tokens_per_expert, hidden_size) -> (num_experts*tokens_per_expert, hidden_size)
        next_states = next_states.view(-1, self.hidden_size)
        return next_states


class Llama4TextMLP(nn.Module):
    """
    标准稠密前馈网络（用于非 MoE 层，以及 MoE 层中的共享专家）。

    结构：SwiGLU 变体
      output = down_proj(act(gate_proj(x)) * up_proj(x))

    在 MoE 层中作为 shared_expert 始终激活；
    在非 MoE 层中，intermediate_size 使用更大的 intermediate_size_mlp。
    """
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        # 默认用 config.intermediate_size（MoE 的专家维度）
        # 非 MoE 层调用时会传入 config.intermediate_size_mlp（更大）
        if intermediate_size is None:
            intermediate_size = config.intermediate_size

        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # x: (batch, seq, hidden_size)
        # gate_proj(x): (batch, seq, intermediate_size)
        # up_proj(x):   (batch, seq, intermediate_size)
        # act(gate) * up: (batch, seq, intermediate_size)  SwiGLU 激活
        down_proj = self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
        # down_proj: (batch, seq, hidden_size)
        return self.down_proj(down_proj)


# ============================================================================
# 归一化层
# ============================================================================

class Llama4TextL2Norm(torch.nn.Module):
    """
    无可学习参数的 L2 归一化，用于 QK Norm（对 Query 和 Key 做归一化，
    防止点积过大导致注意力分布退化）。
    仅作用于开启了 RoPE 的层（use_rope=True）。
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 在 float32 下计算归一化，再转回原始精度（防止 fp16 数值溢出）
        return self._norm(x.float()).type_as(x)

    def extra_repr(self):
        return f"eps={self.eps}"


class Llama4TextRMSNorm(nn.Module):
    """
    RMS 归一化（等价于 T5LayerNorm）：带可学习缩放权重 weight。

    公式：output = (x / RMS(x)) * weight
    其中 RMS(x) = sqrt(mean(x^2) + eps)

    相比 LayerNorm 省去了减均值的步骤，计算更高效，
    是现代大语言模型（LLaMA、Mistral 等）的标准选择。
    """
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习的缩放因子，初始化为全1

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 在 float32 精度下归一化，再乘以可学习权重，最后转回原始精度
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# ============================================================================
# MoE 路由器
# ============================================================================

class Llama4Router(nn.Linear):
    """
    MoE 路由器：为每个 token 打分，选出 top-k 专家。

    流程：
      1. 线性层：hidden → logits (num_experts,)
      2. topk：找出分数最高的 k 个专家索引
      3. scatter：将非 top-k 的位置填为 -inf
      4. sigmoid：将 top-k 的 logits 转为路由权重（0~1），非 top-k 为 0

    返回：
      router_scores:  (tokens, num_experts)，仅 top-k 位置非零，作为加权系数
      router_logits:  (tokens, num_experts)，原始 logits，用于计算辅助 loss
    """
    def __init__(self, config):
        super().__init__(config.hidden_size, config.num_local_experts, bias=False)
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states):
        # hidden_states: (batch*seq, hidden_size)
        # super().forward: Linear(hidden_size -> num_experts)
        router_logits = super().forward(hidden_states)          # (batch*seq, num_experts)
        # 取 top-k 专家的 logit 值和索引
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        # router_top_value: (batch*seq, top_k)
        # router_indices:   (batch*seq, top_k)
        # 非 top-k 位置填 -inf，再 sigmoid → 0；top-k 位置保留原值 sigmoid → 路由权重
        router_scores = torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value)
        # router_scores: (batch*seq, num_experts)  仅 top-k 位置有值，其余为 -inf
        router_scores = torch.nn.functional.sigmoid(router_scores.float()).to(router_scores.dtype)
        # router_scores: (batch*seq, num_experts)  仅 top-k 位置非零（0~1），其余为 0
        return router_scores, router_logits


@use_kernel_forward_from_hub("Llama4TextMoe")  # 允许从 Hub 加载优化的 kernel 替换 forward
class Llama4TextMoe(nn.Module):
    """
    混合专家（Mixture of Experts）前馈层。

    结构：
      - router：为每个 token 选择 top-k 路由专家
      - experts：num_local_experts 个专家（批量 bmm 计算）
      - shared_expert：一个始终激活的稠密 MLP

    前向流程：
      1. router 打分，得到路由权重 (tokens, num_experts)
      2. 将输入 repeat num_experts 次，乘以对应路由权重（非激活专家权重为0）
      3. 所有专家并行计算，输出求和（实际只有 top-k 非零）
      4. 加上 shared_expert 的输出
    """
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(config)    # 所有路由专家（批量计算）
        self.router = Llama4Router(config)           # 路由器
        self.shared_expert = Llama4TextMLP(config)  # 始终激活的共享专家

    def forward(self, hidden_states):
        # hidden_states: (batch, seq, hidden_size)
        # 展平为 token 维度，方便 token 级别路由
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)  # (batch*seq, hidden_size)  N=batch*seq
        router_scores, router_logits = self.router(hidden_states)
        # router_scores:  (N, num_experts)  仅 top-k 位置非零
        # router_logits:  (N, num_experts)  原始 logits，用于辅助 loss

        # 将每个 token 复制 num_experts 份
        routed_in = hidden_states.repeat(router_scores.shape[1], 1)           # (num_experts*N, hidden_size)
        # router_scores.transpose(0,1): (num_experts, N)
        # .reshape(-1, 1):              (num_experts*N, 1)  与 routed_in 广播相乘
        # 非激活专家对应权重为 0，乘后该专家对应 token 输入为全 0
        routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)  # (num_experts*N, hidden_size)

        # 所有专家并行计算（非激活专家因输入为 0，输出也为 0）
        routed_out = self.experts(routed_in)                                   # (num_experts*N, hidden_size)

        # shared_expert 始终对所有 token 计算
        out = self.shared_expert(hidden_states)                                # (N, hidden_size)

        # routed_out.reshape(num_experts, N, hidden_size).sum(dim=0): (N, hidden_size)
        # 等效于将每个 token 的 top-k 专家输出加权求和
        out.add_(routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim=0))
        # out: (N, hidden_size)  注意：上层 Llama4TextDecoderLayer 会通过 .view(residual.shape) 还原为 (batch, seq, hidden_size)
        return out, router_logits


# ============================================================================
# 旋转位置编码（RoPE）
# ============================================================================

class Llama4TextRotaryEmbedding(nn.Module):
    """
    旋转位置编码（Rotary Position Embedding，RoPE）。

    核心思想：将 Query/Key 向量视为复数，通过乘以与位置相关的旋转因子来编码位置信息。
    相对位置信息通过 Q·K 的点积自然体现（旋转相消），无需额外的位置 Embedding 参数。

    inv_freq：逆频率向量，形状 (head_dim/2,)，每个维度对应不同频率的旋转。
    forward 返回的 freqs_cis 是复数张量，直接与 Q/K 做复数乘法完成旋转。
    """
    inv_freq: torch.Tensor  # 注册为 buffer（不参与梯度，但跟随模型保存/加载）

    def __init__(self, config: Llama4TextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        # 根据 rope_type 选择初始化函数（default / llama3 / yarn 等）
        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        # persistent=False：不保存到 checkpoint，每次从 inv_freq 重建
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
            config: Llama4TextConfig | None = None,
            device: Optional["torch.device"] = None,
            seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        标准 RoPE 逆频率计算：
          inv_freq[i] = 1 / (theta ^ (2i / head_dim))，i = 0, 1, ..., head_dim/2 - 1
        theta 越大，低频维度旋转越慢，有利于长序列外推。
        Llama4 默认 theta = 500000.0（比 LLaMA2 的 10000 大得多）。
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0  # default 类型不缩放注意力
        inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # 支持动态序列长度扩展（如 LongRoPE）
    def forward(self, x, position_ids):
        """
        计算旋转复数张量 freqs_cis。

        Args:
            x:            任意张量，仅用于获取 device
            position_ids: (batch, seq_len)，每个 token 的绝对位置索引
        Returns:
            freqs_cis: 复数张量 (batch, seq_len, head_dim/2)
        """
        # inv_freq: (head_dim/2,) -> (1, head_dim/2, 1) -> (batch, head_dim/2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids: (batch, seq_len) -> (batch, 1, seq_len)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # 强制 float32，防止精度损失
            # matmul: (batch, head_dim/2, 1) x (batch, 1, seq_len) -> (batch, head_dim/2, seq_len)
            # transpose(1,2): (batch, seq_len, head_dim/2)
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            # polar: 转换为复数 e^(i*freq)，shape 不变: (batch, seq_len, head_dim/2) 复数
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            freqs_cis = freqs_cis * self.attention_scaling  # 缩放因子，shape 不变

        return freqs_cis  # (batch, seq_len, head_dim/2) 复数


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将 RoPE 旋转应用到 Query 和 Key 上。

    原理：将实数向量的相邻两维视为复数的实部和虚部，
    与旋转因子 freqs_cis 做复数乘法，再转回实数。
    这等价于对每对维度 (2i, 2i+1) 做角度为 freq*pos 的二维旋转。

    Args:
        xq, xk:    (batch, seq_len, num_heads, head_dim) 实数张量
        freqs_cis: (batch, seq_len, head_dim/2) 复数张量
    Returns:
        旋转后的 xq, xk，形状与输入相同
    """
    # xq: (batch, seq_len, num_heads, head_dim)
    # reshape: (batch, seq_len, num_heads, head_dim/2, 2)
    # view_as_complex: (batch, seq_len, num_heads, head_dim/2) 复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis: (batch, seq_len, head_dim/2)
    # [:, :, None, :]: (batch, seq_len, 1, head_dim/2)  在 num_heads 维广播
    # 复数乘法: (batch, seq_len, num_heads, head_dim/2) 复数
    # view_as_real: (batch, seq_len, num_heads, head_dim/2, 2) 实数
    # flatten(3): (batch, seq_len, num_heads, head_dim)
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================================
# 注意力辅助函数
# ============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 KV head 复制 n_rep 次，以匹配 Q head 数量（GQA/MQA 所需）。

    GQA（Grouped Query Attention）：多个 Q head 共享同一组 KV head，
    降低 KV cache 的内存占用。此函数将每个 KV head 展开为 n_rep 份。

    等价于 torch.repeat_interleave(x, dim=1, repeats=n_rep)。
    (batch, num_kv_heads, seq, head_dim) → (batch, num_q_heads, seq, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # unsqueeze(2): (batch, num_kv_heads, 1, seq, head_dim)
    # expand:       (batch, num_kv_heads, n_rep, seq, head_dim)
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # reshape: (batch, num_kv_heads*n_rep, seq, head_dim) = (batch, num_q_heads, seq, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
):
    """
    标准 Scaled Dot-Product Attention（PyTorch 原生实现，无 Flash Attention）。

    注意：Llama4 不将注意力权重 upcast 到 float32（与标准 Llama 不同），
    直接在当前精度（fp16）下做 softmax，节省显存。

    Args:
        query:   (batch, num_q_heads, seq_q, head_dim)
        key:     (batch, num_kv_heads, seq_k, head_dim)
        value:   (batch, num_kv_heads, seq_k, head_dim)
        scaling: 缩放因子，通常为 head_dim^(-0.5)
    Returns:
        (attn_output, attn_weights)
    """
    # GQA：将 kv_heads 扩展到 q_heads 数量
    key_states = repeat_kv(key, module.num_key_value_groups)     # (batch, num_q_heads, seq_k, head_dim)
    value_states = repeat_kv(value, module.num_key_value_groups) # (batch, num_q_heads, seq_k, head_dim)

    # QK^T: (batch, num_q_heads, seq_q, head_dim) x (batch, num_q_heads, head_dim, seq_k)
    #     -> (batch, num_q_heads, seq_q, seq_k)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask  # 加因果掩码（-inf 位置 softmax 后为 0）

    # softmax over seq_k 维: (batch, num_q_heads, seq_q, seq_k)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # attn_weights x V: (batch, num_q_heads, seq_q, seq_k) x (batch, num_q_heads, seq_k, head_dim)
    #                -> (batch, num_q_heads, seq_q, head_dim)
    attn_output = torch.matmul(attn_weights, value_states)
    # transpose: (batch, seq_q, num_q_heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ============================================================================
# 注意力层
# ============================================================================

class Llama4TextAttention(nn.Module):
    """
    多头注意力，支持以下特性：
      - GQA（Grouped Query Attention）：num_key_value_heads < num_attention_heads
      - RoPE：由 no_rope_layers[layer_idx] 决定该层是否使用位置编码
        - use_rope=1：使用 RoPE（大多数层）
        - use_rope=0：NoPE 层，不加位置编码（每隔 no_rope_layer_interval 层）
      - QK Norm：对开启 RoPE 的层的 Q/K 做 L2 归一化（use_qk_norm=True 时）
      - 注意力温度调节（attn_temperature_tuning）：仅对 NoPE 层生效，
        在长序列时动态放大 query scale，防止注意力分布过于集中
      - 分块注意力（chunked attention）：NoPE 层使用局部注意力窗口以节省内存
    """

    def __init__(self, config: Llama4TextConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads  # GQA 分组数
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5  # 标准缩放因子 1/sqrt(d_k)
        self.attn_scale = config.attn_scale           # 温度调节强度系数
        self.floor_scale = config.floor_scale         # 温度调节的基准位置（token数）
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        # no_rope_layers[layer_idx] = 1 表示该层使用 RoPE，= 0 表示 NoPE
        self.use_rope = config.no_rope_layers[layer_idx]

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # QK Norm 只加在有 RoPE 的层
        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_values: Cache | None = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # hidden_states: (batch, seq, hidden_size)
        input_shape = hidden_states.shape[:-1]   # (batch, seq)
        hidden_shape = (*input_shape, -1, self.head_dim)  # (batch, seq, num_heads, head_dim)

        # 线性投影，view 为多头格式
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        # query_states: (batch, seq, num_q_heads, head_dim)
        key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
        # key_states:   (batch, seq, num_kv_heads, head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # value_states: (batch, num_q_heads, seq, head_dim)  注意这里已经 transpose

        # 有 RoPE 的层：施加旋转位置编码（shape 不变）
        if self.use_rope:
            query_states, key_states = apply_rotary_emb(
                query_states, key_states, position_embeddings.to(query_states.device)
            )
            # query_states: (batch, seq, num_q_heads, head_dim)
            # key_states:   (batch, seq, num_kv_heads, head_dim)

        # QK Norm：L2 归一化（shape 不变）
        if hasattr(self, "qk_norm"):
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        # NoPE 层的注意力温度调节：序列越长，query 乘以越大的 scale
        # 公式：scale = log1p(floor((pos+1)/floor_scale)) * attn_scale + 1.0
        # 当 pos < floor_scale 时 scale ≈ 1，超过后随对数缓慢增大
        if self.attn_temperature_tuning and not self.use_rope:
            past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
            positions = torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_seen_tokens
            # attn_scales: (seq,) -> (1, seq, 1, 1) 广播到 (batch, seq, num_heads, head_dim)
            attn_scales = (
                    torch.log1p(torch.floor((positions.float() + 1.0) / self.floor_scale)) * self.attn_scale + 1.0
            )
            attn_scales = attn_scales.view((1, input_shape[-1], 1, 1)).expand((*input_shape, 1, 1))
            query_states = (query_states * attn_scales).to(query_states.dtype)

        # 转置为 (batch, heads, seq, head_dim)，统一后续计算格式
        query_states = query_states.transpose(1, 2)  # (batch, num_q_heads, seq, head_dim)
        key_states = key_states.transpose(1, 2)      # (batch, num_kv_heads, seq, head_dim)

        # 更新 KV cache（推理时），k/v shape 含历史 token
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            # key_states:   (batch, num_kv_heads, past+seq, head_dim)
            # value_states: (batch, num_q_heads,  past+seq, head_dim)

        # 根据配置选择注意力实现（eager / sdpa / flash_attn）
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        # attn_output: (batch, seq, num_q_heads, head_dim)

        # (batch, seq, num_q_heads, head_dim) -> (batch, seq, num_q_heads*head_dim) = (batch, seq, hidden_size)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # o_proj: (batch, seq, hidden_size) -> (batch, seq, hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ============================================================================
# Transformer 解码层
# ============================================================================

class Llama4TextDecoderLayer(GradientCheckpointingLayer):
    """
    单个 Transformer 解码层，采用 Pre-Norm 架构：

      x = x + Attention(RMSNorm(x))      # 残差连接 + 注意力
      x = x + FFN/MoE(RMSNorm(x))        # 残差连接 + 前馈网络

    FFN 类型由 layer_idx 决定：
      - layer_idx 在 config.moe_layers 中 → Llama4TextMoe（稀疏）
      - 否则 → Llama4TextMLP（稠密，使用更大的 intermediate_size_mlp）

    继承 GradientCheckpointingLayer，支持梯度检查点（以计算换显存）。
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
            # 非 MoE 层使用更大的 intermediate_size_mlp
            self.feed_forward = Llama4TextMLP(config, intermediate_size=config.intermediate_size_mlp)

        self.input_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            use_cache: bool | None = False,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        # hidden_states: (batch, seq, hidden_size)

        # ---- 注意力子层 ----
        residual = hidden_states                                       # (batch, seq, hidden_size)
        hidden_states = self.input_layernorm(hidden_states)            # (batch, seq, hidden_size)  Pre-Norm
        attention_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        # attention_states: (batch, seq, hidden_size)
        hidden_states = residual + attention_states                    # (batch, seq, hidden_size)  残差连接

        # ---- FFN/MoE 子层 ----
        residual = hidden_states                                       # (batch, seq, hidden_size)
        hidden_states = self.post_attention_layernorm(hidden_states)   # (batch, seq, hidden_size)  Pre-Norm
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            # MoE 返回 (output, router_logits)，output shape: (batch*seq, hidden_size)
            hidden_states, _ = hidden_states
        # view(residual.shape) 将 MoE 展平的 (batch*seq, hidden_size) 还原为 (batch, seq, hidden_size)
        hidden_states = residual + hidden_states.view(residual.shape)  # (batch, seq, hidden_size)  残差连接
        return hidden_states


# ============================================================================
# 预训练基类
# ============================================================================

@auto_docstring
class Llama4PreTrainedModel(PreTrainedModel):
    """
    所有 Llama4 模型的基类，提供权重初始化和公共配置。

    注意力实现选择（_supports_*）：
      - flash_attn：不支持（T4/Turing 架构不支持 Flash Attention）
      - sdpa：支持（PyTorch 2.0+ 的 scaled_dot_product_attention，T4 可用）
      - flex_attn：支持（PyTorch 2.5+ 的 FlexAttention）
    """
    config: Llama4TextConfig
    input_modalities = ("text",)
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False   # T4 不支持 Flash Attention
    _supports_sdpa = True          # 支持 PyTorch SDPA（T4 可用）
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        """
        权重初始化：
          - 普通 Linear/Embedding：由父类 PreTrainedModel._init_weights 处理（正态分布）
          - Llama4TextExperts：单独处理，因其权重是 nn.Parameter 而非 nn.Linear
        """
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, Llama4TextExperts):
            # 专家参数是直接的 nn.Parameter，需显式初始化
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)


# ============================================================================
# 主干模型：Llama4TextModel
# ============================================================================

@auto_docstring
class Llama4TextModel(Llama4PreTrainedModel):
    """
    Llama4 文本编码器主干（不含 lm_head）。

    结构：
      embed_tokens → N × Llama4TextDecoderLayer → RMSNorm

    输出 last_hidden_state，供上层的 lm_head 或其他任务头使用。

    注意力掩码策略：
      每个 decoder layer 根据自身类型（layer_types）选择掩码：
        - "full_attention"：标准因果掩码，所有历史 token 可见
        - "chunked_attention"：分块因果掩码，只能看到当前块内的历史 token
      两种掩码提前计算好，通过字典按层类型分发。
    """
    _no_split_modules = ["Llama4TextDecoderLayer"]  # 不在此模块边界做张量并行切分
    base_model_prefix = "model"
    input_modalities = ("text",)
    config: Llama4TextConfig
    _can_record_outputs = {
        "attentions": Llama4TextAttention,
        "hidden_states": Llama4TextDecoderLayer,
        "router_logits": Llama4TextMoe,
    }

    def __init__(self, config: Llama4TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Llama4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化
        self.rotary_emb = Llama4TextRotaryEmbedding(config=config)  # 所有层共享同一个 RoPE 模块
        self.gradient_checkpointing = False

        self.post_init()  # 权重初始化 + tied weights 处理

    @can_return_tuple
    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            use_cache: bool | None = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))
            # inputs_embeds: (batch, seq, hidden_size)

        # 推理时自动创建 KV cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # 自动计算位置 id（支持续写，从 past_seen_tokens 处开始）
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)  # (1, seq) -> broadcast 到 (batch, seq)

        # 预先构建两种注意力掩码，按层类型分发
        # attention_mask 如果已是 dict（由 generate 预处理），则直接使用
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),          # 标准因果掩码
                "chunked_attention": create_chunked_causal_mask(**mask_kwargs),  # 分块因果掩码
            }

        hidden_states = inputs_embeds  # (batch, seq, hidden_size)

        # 所有层共享同一套 RoPE 频率（一次计算，逐层传入）
        freq_cis = self.rotary_emb(hidden_states, position_ids)
        # freq_cis: (batch, seq, head_dim/2) 复数

        # 逐层前向传播，每层输入输出均为 (batch, seq, hidden_size)
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=freq_cis,
                **kwargs,
            )
        # 最终 RMSNorm
        hidden_states = self.norm(hidden_states)  # (batch, seq, hidden_size)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ============================================================================
# 因果语言模型：Llama4ForCausalLM
# ============================================================================

class Llama4ForCausalLM(Llama4PreTrainedModel, GenerationMixin):
    """
    带语言模型头的 Llama4，用于：
      1. 训练：传入 labels，自动计算交叉熵 loss（内置 loss_function）
      2. 推理：继承 GenerationMixin，支持 model.generate() 自回归生成

    结构：
      Llama4TextModel（主干） + lm_head（Linear，hidden_size → vocab_size）

    lm_head 与 embed_tokens 权重绑定（tied_weights），节省参数：
      lm_head.weight == embed_tokens.weight
    """
    _no_split_modules = ["Llama4TextDecoderLayer"]
    base_model_prefix = "language_model"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 权重绑定
    _tp_plan = {"lm_head": "colwise_gather_output"}
    config: Llama4TextConfig

    def __init__(self, config: Llama4TextConfig):
        super().__init__(config)
        self.model = Llama4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            use_cache: bool | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        labels: (batch, seq_len)，下一个 token 的 id。
          - 设为 -100 的位置（如 pad）不参与 loss 计算。
          - 训练时通常传入 input_ids 右移一位的结果。

        logits_to_keep: 只计算最后 N 个位置的 logits，节省显存。
          - 训练时设为 0（保留全部）；推理生成时通常只需最后一个位置。
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]  # (batch, seq, hidden_size)

        # 按需只计算部分位置的 logits（推理时节省显存）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # lm_head: (batch, seq_slice, hidden_size) -> (batch, seq_slice, vocab_size)
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # (batch, seq_slice, vocab_size)

        loss = None
        if labels is not None:
            # 内置 loss_function：交叉熵，自动忽略 label=-100 的位置
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ============================================================================
# 输出数据类
# ============================================================================

@auto_docstring(
    custom_intro="""
    Llama4 因果语言模型的输出容器。
    """
)
@dataclass
class Llama4CausalLMOutputWithPast(ModelOutput):
    r"""
    loss:             标量，训练时（传入 labels）返回的交叉熵损失。
    logits:           (batch, seq_len, vocab_size)，lm_head 的原始输出（未经 softmax）。
    past_key_values:  KV cache，推理时加速自回归生成（use_cache=True 时返回）。
    hidden_states:    各层隐状态（output_hidden_states=True 时返回）。
    attentions:       各层注意力权重（output_attentions=True 时返回）。
    """
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


__all__ = [
    "Llama4PreTrainedModel",
    "Llama4TextModel",
    "Llama4ForCausalLM",
]
