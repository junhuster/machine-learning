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
# 文件说明：Llama4 文本模型的配置类
#
# 只保留 Llama4TextConfig（文本语言模型配置）。
# 原始文件中的 Llama4VisionConfig（视觉编码器）和
# Llama4Config（多模态组合配置）已删除，不需要。
# ============================================================================

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="meta-llama/Llama-4-Scout-17B-16E")
@strict  # 严格模式：不允许传入未定义的字段（防止配置拼写错误）
class Llama4TextConfig(PreTrainedConfig):
    r"""
    Llama4 文本语言模型的配置类，继承自 PreTrainedConfig。

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
    intermediate_size_mlp: 非 MoE 稠密层的中间维度，默认 16384（更大）

    【MoE（混合专家）】
    num_local_experts:         每层的专家总数，默认 16
    num_experts_per_tok:       每个 token 激活的专家数（top-k），默认 1
    moe_layers:                指定哪些层使用 MoE（None 时由 interleave_moe_layer_step 自动生成）
    interleave_moe_layer_step: MoE 层的间隔步长，默认 1（即每层都是 MoE）
    output_router_logits:      是否输出路由 logits（计算辅助 loss 用），默认 False
    router_aux_loss_coef:      路由辅助 loss 系数，用于鼓励负载均衡，默认 0.001

    【注意力机制】
    use_qk_norm:           对 RoPE 层的 Q/K 做 L2 归一化，默认 True
    attention_chunk_size:  分块注意力的块大小（NoPE 层使用局部窗口），默认 8192
                           减小此值可降低 NoPE 层的显存占用，代价是感受野变小

    【NoPE 层（无位置编码层）】
    no_rope_layers:        长度等于层数的列表，1=使用RoPE，0=NoPE（None时自动生成）
    no_rope_layer_interval: NoPE 层的间隔，默认 4（即每4层有1个NoPE层）
                            NoPE 层使用分块注意力（chunked_attention）

    【注意力温度调节（仅对 NoPE 层生效）】
    attn_temperature_tuning: 是否启用，默认 True（长序列时动态调整 query scale）
    floor_scale:              开始缩放的位置基准（token数），默认 8192
    attn_scale:               缩放强度，默认 0.1

    【位置编码】
    rope_parameters: RoPE 参数字典，含 rope_type 和 rope_theta
                     例：{"rope_type": "default", "rope_theta": 500000.0}
                     theta=500000 比 LLaMA2 的 10000 大得多，有利于长序列外推

    【其他】
    layer_types:  每层的注意力类型列表（None时自动生成）
                  取值："full_attention"（RoPE层）或 "chunked_attention"（NoPE层）

    Example:
    """

    model_type = "llama4_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0

    # 张量并行（TP）分片计划：指定各层权重的切分方式（单卡训练时忽略）
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.feed_forward.shared_expert.gate_proj": "colwise",
        "layers.*.feed_forward.shared_expert.up_proj": "colwise",
        "layers.*.feed_forward.shared_expert.down_proj": "rowwise",
        "layers.*.feed_forward.experts.gate_up_proj": "packed_rowwise",
        "layers.*.feed_forward.experts.down_proj": "colwise",
        "layers.*.feed_forward.gate_proj": "colwise",
        "layers.*.feed_forward.up_proj": "colwise",
        "layers.*.feed_forward.down_proj": "rowwise",
    }
    # 专家并行（EP）分片计划（单卡训练时忽略）
    base_model_ep_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.feed_forward.experts.gate_up_proj": "grouped_gemm",
        "layers.*.feed_forward.experts.down_proj": "grouped_gemm",
        "layers.*.feed_forward.gate_proj": "colwise",
        "layers.*.feed_forward.up_proj": "colwise",
        "layers.*.feed_forward.down_proj": "rowwise",
        "layers.*.feed_forward.router": "ep_router",
    }

    # 以下为所有配置字段及默认值（原始完整模型尺寸）
    # config_mini.json 中会覆盖这些值为适合 T4 单卡的小尺寸
    vocab_size: int = 202048
    hidden_size: int = 5120
    intermediate_size: int = 8192
    intermediate_size_mlp: int = 16384
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096 * 32
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    attention_dropout: float | int = 0.0
    num_experts_per_tok: int = 1
    num_local_experts: int = 16
    moe_layers: list[int] | None = None
    interleave_moe_layer_step: int = 1
    use_qk_norm: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    rope_parameters: RopeParameters | dict | None = None
    no_rope_layers: list[int] | None = None
    no_rope_layer_interval: int = 4
    attention_chunk_size: int | None = 8192
    layer_types: list[str] | None = None
    attn_temperature_tuning: bool = True
    floor_scale: int = 8192
    attn_scale: float = 0.1
    attention_bias: bool = False

    def __post_init__(self, **kwargs):
        # num_key_value_heads 未设置时退化为标准 MHA（每头独立 KV）
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # 自动生成 no_rope_layers：每隔 no_rope_layer_interval 层放置一个 NoPE 层（值为0）
        # 例：interval=4，8层模型 → [1,1,1,0, 1,1,1,0]（第4、8层为NoPE）
        default_no_rope_layers = [
            int((layer_idx + 1) % self.no_rope_layer_interval != 0) for layer_idx in range(self.num_hidden_layers)
        ]
        self.no_rope_layers = self.no_rope_layers if self.no_rope_layers else default_no_rope_layers

        # head_dim 未指定时由 hidden_size / num_attention_heads 计算
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads

        # 自动生成 moe_layers：从第 (step-1) 层开始，每隔 step 层一个 MoE 层
        # 例：step=2，8层 → [1, 3, 5, 7]（1-indexed: 第2,4,6,8层）
        self.moe_layers = (
            self.moe_layers
            if self.moe_layers is not None
            else list(
                range(
                    self.interleave_moe_layer_step - 1,
                    self.num_hidden_layers,
                    self.interleave_moe_layer_step,
                )
            )
        )

        # 根据 no_rope_layers 自动生成 layer_types：
        # RoPE 层 → "full_attention"（全序列因果注意力）
        # NoPE 层 → "chunked_attention"（分块局部注意力）
        if self.layer_types is None:
            self.layer_types = [
                "chunked_attention" if no_rope else "full_attention" for no_rope in self.no_rope_layers
            ]

        super().__post_init__(**kwargs)


__all__ = ["Llama4TextConfig"]
