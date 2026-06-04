# Llama4 模型结构文档

## 整体架构

Llama4 是一个基于 Transformer Decoder 的自回归语言模型，整体结构从底层到顶层如下：

```
输入 token ids
      ↓
 embed_tokens（词嵌入）
      ↓
 N × Llama4TextDecoderLayer（解码层）
      ↓
 RMSNorm（最终归一化）
      ↓
 lm_head（线性层，输出词表 logits）
      ↓
输出 logits / loss
```

Llama4 相比标准 LLaMA 有三个核心改进：**混合专家（MoE）**、**NoPE 层**、**GQA**。

---

## 一、配置参数（Llama4TextConfig）

Mini 版本（`config_mini.json`）与原始完整模型的关键参数对比：

| 参数 | Mini 版本 | 完整模型 | 说明 |
|------|-----------|---------|------|
| `vocab_size` | 202048 | 202048 | 词表大小 |
| `hidden_size` | 512 | 5120 | 隐层维度 |
| `num_hidden_layers` | 8 | 48 | Transformer 层数 |
| `num_attention_heads` | 8 | 40 | Query 头数 |
| `num_key_value_heads` | 2 | 8 | KV 头数（GQA） |
| `head_dim` | 64 | 128 | 每个注意力头的维度 |
| `intermediate_size` | 1024 | 8192 | MoE 专家中间维度 |
| `intermediate_size_mlp` | 2048 | 16384 | 稠密层中间维度 |
| `num_local_experts` | 4 | 16 | 路由专家总数 |
| `num_experts_per_tok` | 1 | 1 | 每个 token 激活的专家数 |
| `interleave_moe_layer_step` | 2 | 1 | MoE 层间隔（1=每层都是MoE） |
| `no_rope_layer_interval` | 4 | 4 | NoPE 层间隔 |
| `max_position_embeddings` | 2048 | 131072 | 最大序列长度 |

---

## 二、归一化层

### RMSNorm（Llama4TextRMSNorm）

标准现代大语言模型的归一化方式，相比 LayerNorm 省去了减均值的步骤。

```
output = (x / RMS(x)) * weight
RMS(x) = sqrt(mean(x²) + eps)
```

- 带可学习缩放参数 `weight`，初始化为全 1
- 在 float32 下计算，再转回原始精度（防止 fp16 溢出）
- 用于每个解码层的前置归一化（Pre-Norm），以及最终输出归一化

### L2Norm（Llama4TextL2Norm）

无可学习参数的 L2 归一化，仅用于注意力层的 **QK Norm**。

```
output = x / ||x||₂
```

- 只作用于开启了 RoPE 的层（`use_rope=True`）
- 目的：防止 Q/K 点积过大导致注意力分布退化（极端的 softmax 分布）

---

## 三、前馈网络（FFN）

Llama4 中的 FFN 有两种形态，均使用 **SwiGLU** 激活：

```
output = down_proj(act(gate_proj(x)) * up_proj(x))
```

### 稠密 MLP（Llama4TextMLP）

用于非 MoE 层，`intermediate_size` 使用更大的 `intermediate_size_mlp`。
也作为 MoE 层中的 **共享专家**（每个 token 都会经过）使用，此时使用较小的 `intermediate_size`。

### 混合专家（Llama4TextMoe）

```
输入 (tokens, hidden)
      ↓
   Router 打分，选出 top-k 专家
      ↓
   路由专家并行计算（Llama4TextExperts）
      ↓
   共享专家计算（Llama4TextMLP，始终激活）
      ↓
   路由专家输出 + 共享专家输出
```

**路由器（Llama4Router）**：

1. 线性层将 hidden 映射到 `(tokens, num_experts)` 的得分
2. topk 选出分数最高的 k 个专家
3. 非 top-k 位置填 `-inf`，再 sigmoid → 权重为 0
4. top-k 位置 sigmoid → 路由权重（0~1）

**专家计算（Llama4TextExperts）**：

- 所有专家的权重合并为 3D 参数，批量 bmm 并行计算，效率高
- `gate_up_proj`：`(num_experts, hidden_size, 2 * expert_dim)`
- `down_proj`：`(num_experts, expert_dim, hidden_size)`
- 非激活专家因路由权重为 0，输入为 0，输出也为 0

**哪些层用 MoE**：由 `moe_layers` 列表控制，`interleave_moe_layer_step` 决定间隔。
Mini 版本 `step=2`，8 层模型中第 1、3、5、7 层（0-indexed）为 MoE 层。

---

## 四、旋转位置编码（RoPE）

### 原理

将 Query/Key 向量的相邻两维视为复数，通过与位置相关的旋转因子相乘来编码位置信息。相对位置通过 Q·K 点积自然体现（旋转角相消），无需额外位置 Embedding 参数。

```
inv_freq[i] = 1 / (theta ^ (2i / head_dim))
freqs_cis = e^(i * position * inv_freq) = cos + i·sin
Q_rotated = Q * freqs_cis（复数乘法）
```

Llama4 默认 `theta = 500000.0`，比 LLaMA2 的 `10000` 大得多，有利于长序列外推。

### NoPE 层

每隔 `no_rope_layer_interval`（默认 4）层，设置一个不使用 RoPE 的层（`use_rope=0`）。

NoPE 层没有显式的位置编码，依靠数据中的上下文信息隐式建模位置关系，对长序列有一定的泛化能力。

Mini 版本 8 层模型中：第 3、7 层（0-indexed）为 NoPE 层，其余为 RoPE 层。

---

## 五、注意力机制（Llama4TextAttention）

### GQA（Grouped Query Attention）

多个 Query 头共享同一组 KV 头，降低 KV Cache 的内存占用。

```
num_key_value_groups = num_attention_heads / num_key_value_heads
```

Mini 版本：8 个 Q 头，2 个 KV 头，每组 4 个 Q 头共享 1 对 KV。

### QK Norm

对开启 RoPE 的层，在施加旋转编码之后对 Q、K 做 L2 归一化，防止注意力分布退化。

### 注意力温度调节（attn_temperature_tuning）

仅对 **NoPE 层**生效，在处理长序列时动态放大 query 的缩放系数：

```
scale = log1p(floor((position + 1) / floor_scale)) * attn_scale + 1.0
query = query * scale
```

当 `position < floor_scale` 时 scale ≈ 1，超过后随位置对数缓慢增大，防止长序列下注意力过于分散。

### 分块注意力（Chunked Attention）

NoPE 层使用局部注意力窗口（块大小 `attention_chunk_size`），每个 token 只能看到当前块内的历史 token，而不是全部历史。

- 好处：降低显存，NoPE 层不需要全局位置信息
- 代价：跨块的远距离依赖需通过多层叠加传递

RoPE 层使用标准因果掩码（`full_attention`），可见全部历史。

---

## 六、解码层（Llama4TextDecoderLayer）

每个解码层采用 **Pre-Norm** 架构：

```
x = x + Attention(RMSNorm(x))    # 注意力子层
x = x + FFN/MoE(RMSNorm(x))      # 前馈子层
```

- 先归一化再计算，训练更稳定（梯度不易爆炸/消失）
- 残差连接让梯度直接流回浅层

FFN 类型由层索引决定：在 `moe_layers` 列表中的层使用 MoE，其余使用稠密 MLP。

---

## 七、权重绑定

`lm_head`（输出层）与 `embed_tokens`（输入嵌入层）共享同一套权重：

```
lm_head.weight == embed_tokens.weight
```

这是语言模型中的常见做法，减少参数量，同时让 token 的输入表示和输出预测在同一语义空间中。

---

## 八、Mini 版本层结构一览

以下是 8 层 Mini 模型每层的完整配置（`no_rope_layer_interval=4`，`interleave_moe_layer_step=2`）：

`moe_layers = [1, 3, 5, 7]`（从第 `step-1=1` 层开始，每隔 2 层一个 MoE 层）
`no_rope_layers = [1,1,1,0, 1,1,1,0]`（每隔 4 层一个 NoPE 层，即第 3、7 层）

| 层索引 | 注意力类型 | 使用 RoPE | FFN 类型 |
|--------|-----------|-----------|---------|
| 0 | full_attention | ✓ | 稠密 MLP |
| 1 | full_attention | ✓ | MoE |
| 2 | full_attention | ✓ | 稠密 MLP |
| 3 | chunked_attention | ✗（NoPE） | MoE |
| 4 | full_attention | ✓ | 稠密 MLP |
| 5 | full_attention | ✓ | MoE |
| 6 | full_attention | ✓ | 稠密 MLP |
| 7 | chunked_attention | ✗（NoPE） | MoE |

---

## 九、参数量估算

基本参数：`H=512, Im=2048（稠密）, I=1024（MoE专家）, E=4（路由专家）, V=202048`
注意力头：`nH=8（Q头）, nKV=2（KV头）, dh=64（head_dim）`

**注意力层（每层）**

| 子模块 | 计算 | 参数量 |
|--------|------|--------|
| q_proj | 512 × (8×64) | 262,144 |
| k_proj | 512 × (2×64) | 65,536 |
| v_proj | 512 × (2×64) | 65,536 |
| o_proj | (8×64) × 512 | 262,144 |
| input_layernorm + post_attn_layernorm | 512 × 2 | 1,024 |
| **小计** | | **656,384** |

**稠密 FFN（4 层，intermediate_size_mlp=2048）**

| 子模块 | 计算 | 参数量 |
|--------|------|--------|
| gate_proj | 512 × 2048 | 1,048,576 |
| up_proj | 512 × 2048 | 1,048,576 |
| down_proj | 2048 × 512 | 1,048,576 |
| **小计（单层）** | | **3,145,728** |

**MoE 层（4 层，intermediate_size=1024，num_local_experts=4）**

| 子模块 | 计算 | 参数量 |
|--------|------|--------|
| 路由专家 gate_up_proj | 4 × 512 × (2×1024) | 4,194,304 |
| 路由专家 down_proj | 4 × 1024 × 512 | 2,097,152 |
| 共享专家（MLP，inter=1024）| 512×1024×3 | 1,572,864 |
| router | 512 × 4 | 2,048 |
| **小计（单层）** | | **7,866,368** |

**汇总**

| 模块 | 参数量 | 说明 |
|------|--------|------|
| embed_tokens | 202048 × 512 = **103.4M** | 词嵌入 |
| 注意力层 × 8 | 656,384 × 8 ≈ **5.3M** | 含 RMSNorm |
| 稠密 FFN × 4 | 3,145,728 × 4 ≈ **12.6M** | |
| MoE 层 × 4 | 7,866,368 × 4 ≈ **31.5M** | 路由专家 + 共享专家 + router |
| final_norm + lm_head | 512 + 103,448,576 ≈ **103.4M** | lm_head 不与 embed 共享权重 |
| **总计** | **≈ 256M** | |

> embed_tokens 和 lm_head 各占约 103M，合计占总参数量的 **81%**，模型主干约 49M。
> `tie_word_embeddings=false`，两者独立存储。若开启权重绑定可节省约 100M 参数。

---

## 十、训练显存估算

以 fp16 混合精度，`batch_size=4, max_seq_len=256` 为例：

| 项目 | 显存 | 说明 |
|------|------|------|
| 模型参数 fp16 | 256M × 2 = **512 MB** | |
| 梯度 fp16 | 256M × 2 = **512 MB** | 形状与参数完全相同 |
| AdamW 优化器状态 fp32 | 256M × 8 = **2,048 MB** | 一阶矩 + 二阶矩各一份 fp32 |
| Activation | ~80 MB | 8 层，每层约 10 MB |
| 框架/临时缓冲 | ~200 MB | |
| **合计** | **≈ 3.4 GB** | |

T4（16GB）显存充裕，可大幅增大 `batch_size` 或 `max_seq_len`。

---

## 十一、训练与推理模式

### 预训练模式（`train.py`）

- 全序列一次性前向传播，不使用 KV Cache（`use_cache=False`）
- 传入 `labels`（input_ids 右移一位），内置交叉熵 loss
- label=-100 的位置（如 padding）不参与 loss 计算

### SFT 训练（`train_sft.py`）

- 只对 assistant 回复部分计算 loss（`loss_mask`）
- 手动计算 `F.cross_entropy(reduction="none")` 再乘以 mask，取 mask 位置的平均
- 优先从 SFT checkpoint 续训；无 SFT checkpoint 时加载预训练权重初始化

### 梯度累积（`grad_accum_steps`）

两个训练脚本均支持梯度累积，通过 `--grad_accum_steps` 参数控制：

- `grad_accum_steps=1`（默认）：每个 batch 立即更新参数，行为与不开启时完全一致
- `grad_accum_steps=N`（N>1）：累积 N 个 mini-batch 的梯度后才执行一次参数更新，等价于将有效 batch size 扩大 N 倍

```bash
# 等价于 batch_size=32 的训练效果，但显存只占 batch_size=4 的量
python train.py --batch_size 4 --grad_accum_steps 8
```

实现要点：
- 每个 mini-batch 的 loss 除以 `grad_accum_steps`，保证梯度幅度等价于大 batch
- `optimizer.zero_grad()`、参数更新、`scheduler.step()` 均只在累积满时执行
- 日志和 checkpoint 的步数以 optimizer 更新次数（`global_step`）计，`total_steps = batch总数 // grad_accum_steps`

### 推理模式（`inference.py`）

- 继承 HuggingFace `GenerationMixin`，支持 `model.generate()`
- 使用 `DynamicCache` 管理 KV Cache，避免重复计算历史 token
- 支持 temperature 采样、top-p nucleus sampling
- `--chat` 参数：用 `apply_chat_template` 拼接对话格式，适用于 SFT 模型
