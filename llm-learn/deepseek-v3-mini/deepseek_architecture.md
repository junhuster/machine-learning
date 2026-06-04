# DeepSeek-Mini 模型结构文档

## 整体架构

DeepSeek-Mini 基于 DeepSeek-V3 架构裁剪而来，是一个自回归 Decoder-Only 语言模型，整体结构如下：

```
输入 token ids
      ↓
 embed（词嵌入）
      ↓
 N × Block（解码层）
      ↓
 RMSNorm（最终归一化）
      ↓
 head（Linear，输出词表 logits）
      ↓
输出 logits / loss
```

相比标准 Transformer，DeepSeek-V3 最核心的两个创新是：
- **MLA（Multi-Head Latent Attention）**：通过低秩压缩大幅减少 KV Cache 显存占用
- **MoE（Mixture of Experts）**：用稀疏激活替代稠密 FFN，在参数量大的同时保持计算量可控

与 llama4-mini 相比，deepseek-mini **没有 GQA**，也没有 NoPE 层，每层均使用 RoPE。

---

## 一、配置参数（config_mini.json）

Mini 版本（`config_mini.json`）与 DeepSeek-V3 完整模型的关键参数对比：

| 参数 | Mini 版本 | DeepSeek-V3 完整版 | 说明 |
|------|-----------|-------------------|------|
| `vocab_size` | 102400 | 102400 | 词表大小 |
| `dim` | 512 | 7168 | 隐层维度 |
| `n_layers` | 8 | 61 | Transformer 层数 |
| `n_dense_layers` | 1 | 3 | 前 N 层用稠密 MLP，其余用 MoE |
| `n_heads` | 8 | 128 | 注意力头数（Q/K/V 头数相同，无 GQA） |
| `inter_dim` | 1344 | 18432 | 稠密层 FFN 中间维度 |
| `moe_inter_dim` | 256 | 2048 | MoE 每个专家的中间维度 |
| `n_routed_experts` | 16 | 256 | 路由专家总数 |
| `n_shared_experts` | 1 | 1 | 共享专家数（每个 token 都激活） |
| `n_activated_experts` | 2 | 8 | 每个 token 激活的路由专家数（top-k） |
| `kv_lora_rank` | 128 | 512 | MLA KV 低秩压缩维度 |
| `qk_nope_head_dim` | 64 | 128 | 不带位置编码的 Q/K 每头维度 |
| `qk_rope_head_dim` | 32 | 64 | 带 RoPE 的 Q/K 每头维度 |
| `v_head_dim` | 64 | 128 | Value 每头维度 |
| `max_seq_len` | 2048 | 163840 | 最大序列长度 |
| `rope_theta` | 10000.0 | 10000.0 | RoPE 基础频率 |

---

## 二、归一化层（RMSNorm）

```
output = (x / RMS(x)) * weight
RMS(x) = sqrt(mean(x²) + eps)
```

- 带可学习缩放参数 `weight`，初始化为全 1
- 相比 LayerNorm 省去减均值步骤，计算更高效
- 在 float32 下计算后转回原始精度，防止 fp16 溢出
- 用于每层的 Pre-Norm 以及最终输出归一化

---

## 三、旋转位置编码（RoPE / YaRN）

### 标准 RoPE

将 Query/Key 向量的相邻两维视为复数，通过与位置相关的旋转因子相乘编码位置信息：

```
inv_freq[i] = 1 / (theta ^ (2i / head_dim))
freqs_cis = e^(i * position * inv_freq) = cos + i·sin
Q_rotated = Q * freqs_cis（复数乘法）
```

### YaRN 长度外推

当序列长度超过 `original_seq_len` 且 `rope_factor > 1` 时自动启用，通过对频率进行非线性插值支持更长序列。

Mini 版本 `rope_factor=1.0`，不启用 YaRN，使用标准 RoPE。

### MLA 中的 RoPE 拆分

MLA 将 Q/K 的头维度拆分为两部分：
- `qk_nope_head_dim=64`：不加位置编码，用于语义匹配
- `qk_rope_head_dim=32`：加 RoPE，用于位置感知

---

## 四、MLA（Multi-Head Latent Attention）

DeepSeek-V3 的核心注意力机制，通过低秩分解压缩 KV，减少 KV Cache 内存占用。

### 与标准 MHA 的对比

| | 标准 MHA | MLA |
|--|---------|-----|
| KV Cache 大小 | `2 × n_heads × head_dim` | `kv_lora_rank`（远小于前者） |
| 原理 | 直接缓存 K/V | 缓存压缩后的隐向量，推理时再展开 |

### 前向流程

```
输入 x (B, S, dim)
  ↓
【Query 投影】
  q_lora_rank=0：wq 直接投影到 n_heads × qk_head_dim
  q_lora_rank>0：wq_a → RMSNorm → wq_b（低秩压缩）

【KV 低秩投影】
  wkv_a(x) → 拆分为：
    kv_latent (kv_lora_rank)   ← 压缩的 KV 语义信息
    k_pe (qk_rope_head_dim)    ← 用于位置编码的 K 分量

  kv_latent → RMSNorm → wkv_b → 展开为 k_nope + v

【拼接】
  q = [q_nope, q_pe(+RoPE)]
  k = [k_nope, k_pe(+RoPE).expand(n_heads)]

【注意力计算】
  scores = einsum(q, k) * scale → softmax → einsum(v) → wo
```

推理时 KV Cache 只存 `kv_lora_rank` 大小的隐向量，远小于标准 MHA 的 KV Cache。

---

## 五、前馈网络（FFN）

### 稠密 MLP（SwiGLU）

用于前 `n_dense_layers` 层（Mini 版本只有第 0 层）：

```
output = w2(silu(w1(x)) * w3(x))
```

- `w1/w3`：门控和上投影，`dim → inter_dim`
- `w2`：下投影，`inter_dim → dim`
- Mini 版本 `inter_dim=1344 ≈ 2.625 × dim`

### 混合专家（MoE）

用于第 1 层及之后的所有层：

```
输入 x
  ↓
Gate 路由打分 → top-k 专家索引 + 路由权重
  ↓
路由专家并行计算（每个 token 激活 n_activated_experts 个）
  ↓
共享专家计算（始终激活，n_shared_experts 个）
  ↓
路由专家加权输出 + 共享专家输出
```

**路由器（Gate）**：

1. 线性层 `dim → n_routed_experts` 打分
2. softmax（或 sigmoid）归一化
3. topk 选出 `n_activated_experts` 个专家
4. 路由权重乘以 `route_scale` 缩放

Mini 版本：16 个路由专家，每次激活 2 个，加 1 个共享专家。

---

## 六、解码层（Block）

每层采用 **Pre-Norm** 架构：

```
x = x + Attention(RMSNorm(x))    # 注意力子层（MLA）
x = x + FFN(RMSNorm(x))          # 前馈子层（稠密或 MoE）
```

FFN 类型由层索引决定：
- `layer_idx < n_dense_layers`：稠密 MLP
- `layer_idx >= n_dense_layers`：MoE

Mini 版本 `n_dense_layers=1`，即第 0 层稠密，第 1~7 层均为 MoE。

---

## 七、Mini 版本层结构一览

| 层索引 | FFN 类型 | 说明 |
|--------|---------|------|
| 0 | 稠密 MLP | `inter_dim=1344` |
| 1 | MoE | 16 路由专家 + 1 共享专家，激活 2 个 |
| 2 | MoE | 同上 |
| 3 | MoE | 同上 |
| 4 | MoE | 同上 |
| 5 | MoE | 同上 |
| 6 | MoE | 同上 |
| 7 | MoE | 同上 |

所有层均使用 RoPE，无 NoPE 层。

---

## 八、参数量估算

基本参数：`dim=512, inter_dim=1344, moe_inter_dim=256, vocab_size=102400`

| 模块 | 参数量 | 说明 |
|------|--------|------|
| embed_tokens | 102400 × 512 ≈ **52.4M** | 词嵌入 |
| 注意力层 × 8 | ~5.3M | wq/wkv_a/wkv_b/wo + RMSNorm |
| 稠密 FFN × 1 | 512×1344×3 ≈ **2.1M** | w1/w2/w3 |
| MoE 层 × 7 | ~31.5M | 路由专家 + 共享专家 + router |
| lm_head | 512 × 102400 ≈ **52.4M** | 输出投影（不与 embed 共享） |
| **总计** | **≈ 144M** | |

> embed 和 lm_head 各占约 52M，合计占总参数量的 **72%**，模型主干约 40M。

---

## 九、训练显存估算

以 fp16 混合精度，`batch_size=8, max_seq_len=256` 为例：

| 项目 | 显存 |
|------|------|
| 模型参数 fp16 | 144M × 2 = **288 MB** |
| 梯度 fp16 | 144M × 2 = **288 MB** |
| AdamW 优化器状态 fp32 | 144M × 8 = **1,152 MB** |
| Activation | ~60 MB |
| 框架缓冲 | ~100 MB |
| **合计** | **≈ 1.9 GB** |

T4（16GB）显存充裕，可大幅增大 `batch_size` 或 `max_seq_len`。

---

## 十、训练与推理模式

### 预训练模式（`train.py`）

- 全序列前向传播，不使用 KV Cache（`use_cache=False`）
- 手动计算交叉熵 loss，忽略 padding 位置（`ignore_index=pad_id`）
- 分词器：直接复用 DeepSeek-V3 官方分词器（`deepseek-ai/DeepSeek-V3`）

### SFT 训练（`train_sft.py`）

- 只对 assistant 回复部分计算 loss（`loss_mask`）
- 手动计算 `F.cross_entropy(reduction="none")` 再乘以 mask，取 mask 位置的平均
- 优先从 SFT checkpoint 续训；无 SFT checkpoint 时加载预训练权重初始化

### 梯度累积（`grad_accum_steps`）

两个训练脚本均支持梯度累积，通过 `--grad_accum_steps` 参数控制：

- `grad_accum_steps=1`（默认）：每个 batch 立即更新参数，与不开启时完全一致
- `grad_accum_steps=N`（N>1）：累积 N 个 mini-batch 的梯度后才执行一次参数更新，等价于将有效 batch size 扩大 N 倍

```bash
# 等价于 batch_size=64 的训练效果，但显存只占 batch_size=8 的量
python train.py --batch_size 8 --grad_accum_steps 8
```

实现要点：
- 每个 mini-batch 的 loss 除以 `grad_accum_steps`，保证梯度幅度等价于大 batch
- `optimizer.zero_grad()`、参数更新、`scheduler.step()` 均只在累积满时执行
- 日志和 checkpoint 的步数以 optimizer 更新次数（`global_step`）计，`total_steps = batch总数 // grad_accum_steps`

### 推理模式（`inference.py`）

- 自实现 KV Cache（`k_cache/v_cache` buffer），支持增量解码
- 推理分两阶段：prefill（处理完整 prompt）+ decode（逐 token 生成）
- 支持 temperature 采样、top-p nucleus sampling
- `generate_chat`：用 `apply_chat_template` 拼接对话格式，适用于 SFT 模型
