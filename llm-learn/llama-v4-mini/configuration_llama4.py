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
# 文件说明：Llama4 文本模型的配置类（本地实现，不依赖 HuggingFace）
# ============================================================================

import json
import logging

logger = logging.getLogger(__name__)


class Llama4TextConfig:
    """
    Llama4 文本语言模型的配置类（独立实现，无需 transformers/huggingface_hub）。

    主要参数说明：

    【模型尺寸】
    vocab_size:            词表大小，默认 202048
    hidden_size:           隐层维度，默认 5120
    num_hidden_layers:     Transformer 层数，默认 48
    num_attention_heads:   Query 头数，默认 40
    num_key_value_heads:   KV 头数（<num_attention_heads 时启用 GQA），默认 8
    head_dim:              每个注意力头的维度，默认 128

    【FFN 尺寸】
    intermediate_size:     MoE 专家的中间维度，默认 8192
    intermediate_size_mlp: 非 MoE 稠密层的中间维度，默认 16384

    【MoE（混合专家）】
    num_local_experts:         每层的专家总数，默认 16
    num_experts_per_tok:       每个 token 激活的专家数（top-k），默认 1
    moe_layers:                指定哪些层使用 MoE（None 时由 interleave_moe_layer_step 自动生成）
    interleave_moe_layer_step: MoE 层的间隔步长，默认 1

    【注意力机制】
    use_qk_norm:           对 RoPE 层的 Q/K 做 L2 归一化，默认 True
    attention_chunk_size:  分块注意力的块大小（NoPE 层使用局部窗口），默认 8192

    【NoPE 层（无位置编码层）】
    no_rope_layers:        长度等于层数的列表，1=使用RoPE，0=NoPE
    no_rope_layer_interval: NoPE 层的间隔，默认 4

    【注意力温度调节（仅对 NoPE 层生效）】
    attn_temperature_tuning: 是否启用，默认 True
    floor_scale:              开始缩放的位置基准（token数），默认 8192
    attn_scale:               缩放强度，默认 0.1

    【位置编码】
    rope_parameters: RoPE 参数字典，含 rope_type 和 rope_theta
    """

    model_type = "llama4_text"

    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 5120,
        intermediate_size: int = 8192,
        intermediate_size_mlp: int = 16384,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 40,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096 * 32,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings: bool = False,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 1,
        num_local_experts: int = 16,
        moe_layers=None,
        interleave_moe_layer_step: int = 1,
        use_qk_norm: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        rope_parameters=None,
        no_rope_layers=None,
        no_rope_layer_interval: int = 4,
        attention_chunk_size=8192,
        layer_types=None,
        attn_temperature_tuning: bool = True,
        floor_scale: int = 8192,
        attn_scale: float = 0.1,
        attention_bias: bool = False,
        **kwargs,  # 兼容旧 config.json 中可能存在的多余字段
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_size_mlp = intermediate_size_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.interleave_moe_layer_step = interleave_moe_layer_step
        self.use_qk_norm = use_qk_norm
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.no_rope_layer_interval = no_rope_layer_interval
        self.attention_chunk_size = attention_chunk_size
        self.attn_temperature_tuning = attn_temperature_tuning
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale
        self.attention_bias = attention_bias

        # rope_parameters 默认值
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": 500000.0}
        self.rope_parameters = rope_parameters

        # 自动生成 no_rope_layers：每隔 no_rope_layer_interval 层放置一个 NoPE 层（值为0）
        default_no_rope_layers = [
            int((layer_idx + 1) % self.no_rope_layer_interval != 0)
            for layer_idx in range(self.num_hidden_layers)
        ]
        self.no_rope_layers = no_rope_layers if no_rope_layers else default_no_rope_layers

        # 自动生成 moe_layers
        self.moe_layers = (
            moe_layers
            if moe_layers is not None
            else list(range(
                self.interleave_moe_layer_step - 1,
                self.num_hidden_layers,
                self.interleave_moe_layer_step,
            ))
        )

        # 根据 no_rope_layers 自动生成 layer_types
        if layer_types is None:
            self.layer_types = [
                "chunked_attention" if not no_rope else "full_attention"
                for no_rope in self.no_rope_layers
            ]
        else:
            self.layer_types = layer_types

        # 注意力实现，默认 eager（T4 不支持 flash_attention）
        self._attn_implementation = kwargs.get("_attn_implementation", "eager")

    @classmethod
    def from_json(cls, path: str) -> "Llama4TextConfig":
        with open(path) as f:
            cfg = json.load(f)
        return cls(**cfg)

    def __repr__(self):
        return (
            f"Llama4TextConfig(hidden_size={self.hidden_size}, "
            f"num_hidden_layers={self.num_hidden_layers}, "
            f"num_attention_heads={self.num_attention_heads}, "
            f"num_key_value_heads={self.num_key_value_heads}, "
            f"num_local_experts={self.num_local_experts})"
        )


__all__ = ["Llama4TextConfig"]
