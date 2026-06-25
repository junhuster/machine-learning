# DeepSeek-V4-Mini 架构说明

## 整体架构

DeepSeek-V4-Mini 是基于 DeepSeek-V4-Pro 架构的轻量化单卡训练版本。相比原版，采用纯 PyTorch 实现，不依赖 Triton kernel，完全采用 FP16 精度，支持单卡训练和推理。整体结构为自回归 Decoder-Only 语言模型，核心创新包括：

- **MLA（Multi-Head Latent Attention）**：通过低秩分解压缩 KV，减少推理 KV Cache 显存占用
- **MoE（Mixture of Experts）**：稀疏激活专家网络，在参数量大的同时保持计算量可控
- **Hyper-Connections（HC）**：通过 Sinkhorn 归一化维护多份隐层副本，改进残差连接机制
- **KV Compressor**：对特定层的 KV 进行有损压缩，进一步减少 KV Cache
- **Hash-based 路由**：前若干层使用确定性 hash 路由，提高专家负载均衡
- **sqrtsoftplus 激活函数**：替代传统 softmax 用于 MoE 路由打分
- **Multi-Token Prediction（MTP）**：辅助训练目标，预测后续多个 token

```
输入 token ids (B, T)
      ↓
 embed（词嵌入）→ (B, T, dim)
      ↓
 扩展为 HC 副本 → (B, T, hc_mult, dim)
      ↓
 N × Block（解码层，Attention + MoE）
      ↓
 HCHead 合并 HC 副本 → (B, T, dim)
      ↓
 RMSNorm → lm_head
      ↓
输出 logits (B, T, vocab_size)
```

---

## 一、配置参数（config_mini.json）

| 参数 | 值 | 说明 |
|------|-----|------|
| `vocab_size` | 129280 | 词表大小 |
| `dim` | 896 | 隐层维度 |
| `n_layers` | 8 | Transformer 块数 |
| `n_heads` | 14 | MLA 注意力头数 |
| `head_dim` | 128 | 每头的完整维度（nope + rope） |
| `rope_head_dim` | 32 | RoPE 部分维度（nope = 128 - 32 = 96） |
| `q_lora_rank` | 256 | Query 低秩投影维度 |
| `o_lora_rank` | 128 | 输出低秩投影维度 |
| `o_groups` | 2 | 输出投影的分组数 |
| `moe_inter_dim` | 512 | MoE 专家中间维度 |
| `n_routed_experts` | 16 | 路由专家总数 |
| `n_shared_experts` | 1 | 共享专家数（每个 token 都激活） |
| `n_activated_experts` | 2 | 每个 token 激活的路由专家数（top-k） |
| `score_func` | `sqrtsoftplus` | MoE 路由打分函数 |
| `route_scale` | 1.0 | 路由权重缩放系数 |
| `swiglu_limit` | 10.0 | SwiGLU 激活值截断阈值（0=不截断） |
| `n_hash_layers` | 1 | 前 N 层使用 hash-based 路由 |
| `n_mtp_layers` | 1 | Multi-Token Prediction 辅助层数 |
| `window_size` | 64 | 滑动窗口 attention 大小 |
| `compress_ratios` | `[0,0,4,0,4,0,4,0]` | 每层 KV 压缩比（0=无压缩，4=4:1压缩） |
| `compress_rope_theta` | 40000.0 | 压缩层的 RoPE 基础频率 |
| `original_seq_len` | 0 | 压缩层 YaRN 原始序列长度（0=禁用） |
| `rope_theta` | 10000.0 | 普通层 RoPE 基础频率 |
| `rope_factor` | 1.0 | RoPE 外推因子（1.0=无外推） |
| `hc_mult` | 2 | Hyper-Connections 副本数（原版=4） |
| `hc_sinkhorn_iters` | 5 | Sinkhorn 迭代次数（原版=20） |
| `hc_eps` | 1e-6 | HC 数值稳定 epsilon |
| `norm_eps` | 1e-6 | RMSNorm epsilon |
| `max_seq_len` | 2048 | 最大序列长度 |
| `max_batch_size` | 4 | 最大批次大小 |

---

## 二、与 V3-Mini 的主要架构差异

### 1. Hyper-Connections（HC）残差机制

**V3-Mini**：标准残差连接 `x = x + sublayer(norm(x))`

**V4-Mini**：HC 维护 `hc_mult=2` 份隐层副本，通过 Sinkhorn 归一化做加权混合

```
hc_pre:
  (B, S, hc_mult, dim) → Sinkhorn 加权 → (B, S, dim) + pre/post/comb 权重

子层计算（Attention 或 MoE）：
  (B, S, dim) → (B, S, dim)

hc_post:
  (B, S, dim) + residual(B, S, hc_mult, dim) → (B, S, hc_mult, dim)
```

**优势**：改进梯度流，增强表达能力；Sinkhorn 确保数值稳定性

### 2. KV 压缩 Compressor

**V3-Mini**：仅缓存完整 KV

**V4-Mini**：对第 2、4、6 层（compress_ratio=4）应用 KV 压缩
- 通过 gated pooling 将 4 个连续 token 的 KV 压缩为 1 个
- 滑动窗口保留最近 64 个完整 KV；历史 KV 以 4:1 压缩存储
- 支持重叠压缩（overlap）以减少边界损失

### 3. Hash-based 路由

**V3-Mini**：所有层都用 score-based 路由（softmax 打分 → topk）

**V4-Mini**：前 `n_hash_layers=1` 层使用 hash-based 路由
```
indices = tid2eid[input_ids]  # 从查表得到，O(1)，无计算开销
```
提高负载均衡性，前层确定性路由更稳定。

### 4. sqrtsoftplus 激活函数

**V3-Mini**：路由器用 softmax 或 sigmoid

**V4-Mini**：默认用 `sqrtsoftplus`（即 `sqrt(softplus(x))`）
```python
scores = F.softplus(scores).sqrt()
weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)
```
介于 softmax 和 sigmoid 之间，梯度特性更稳定。

### 5. Multi-Token Prediction（MTP）

**V3-Mini**：无辅助训练目标

**V4-Mini**：`n_mtp_layers=1` 个 MTP 层
- 共享 embed 和 head，额外参数少
- 训练时与主 loss 加权组合：`total_loss = main_loss + 0.1 * mtp_loss`
- 推理时不使用 MTP

### 6. Dense Attention（替代 Sparse Attention）

**V3-Mini**：标准 MLA（全量 attention）

**V4-Mini**：标准 Dense Attention + 滑动窗口 KV Cache + 可选 KV 压缩
```python
o = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, scale=self.softmax_scale)
```
去掉 Indexer 和 sparse_attn kernel 依赖，纯 PyTorch 实现。

---

## 三、参数量估算（分模块）

| 模块 | 参数量估算 | 说明 |
|------|-----------|------|
| **embedding** | 129280 × 896 ≈ **115.8M** | 词嵌入矩阵 |
| **lm_head** | 129280 × 896 ≈ **115.8M** | 输出投影（与 embedding 不共享） |
| **Attention ×8** | ~2.4M/层 ≈ **19.2M** | wq_a/wq_b/wkv/wo_a/wo_b + norms |
| **MoE ×8** | ~12M/层 ≈ **96M** | gate + 16 experts + 1 shared |
| **HC 参数 ×8** | ~0.5M/层 ≈ **4M** | hc_attn_fn/ffn_fn/base/scale |
| **MTP 层 ×1** | ~3.5M | e_proj/h_proj/norms/hc_head_params |
| **其他** | ~2M | 全局 norm、hc_head 参数等 |
| **合计** | **~356M** | 实际可训练参数 |

> embedding + lm_head 合计约 231.6M，占 65%，是主要参数来源。

---

## 四、各主要模块说明

### RMSNorm

```
output = weight * (x / sqrt(mean(x²) + eps))
```
- 在 float32 计算后转回原 dtype，防止 fp16 溢出
- 省去减均值步骤，比 LayerNorm 更高效

### Linear

标准 FP16 线性层，无量化，无分布式切分：`output = x @ weight^T`

### Rotary Positional Embeddings（RoPE）

将 Q/K 的相邻两维视为复数，通过旋转编码位置：
```
inv_freq[i] = 1 / (theta ^ (2i / rope_head_dim))
freqs_cis[pos] = e^(i × pos × inv_freq)  → (pos, rope_head_dim//2) 复数
```

Q/K 拆分为两部分：
- `nope_head_dim = 96`：不加位置编码，用于语义匹配
- `rope_head_dim = 32`：加 RoPE，用于位置感知

### sinkhorn_split（Hyper-Connections 核心）

输入 `mixes: (..., mix_hc)` 中 `mix_hc = (2 + hc_mult) * hc_mult = 8`，拆分为：
- `pre`：前 hc_mult 列，Sinkhorn 迭代归一化 → 加权系数
- `post`：中 hc_mult 列，sigmoid → 扩展权重
- `comb`：后 hc_mult² 列，softmax → 副本间组合矩阵

### Compressor（KV 缓存压缩）

通过 gated pooling 压缩连续 `compress_ratio` 个 token 的 KV：
```
kv = wkv(x)      # (B, S, head_dim)
score = wgate(x)  # (B, S, head_dim)
compressed_kv = (kv * softmax(score + ape)).sum(ratio_dim)  # (B, S//ratio, head_dim)
```
压缩后应用 RMSNorm 和 RoPE，写入 kv_cache 的压缩部分。

### Attention（MLA + 滑动窗口 + 可选压缩）

```
Q: wq_b(RMSNorm(wq_a(x))) → Q-Norm → RoPE(rope_dim)
KV: RMSNorm(wkv(x)) → RoPE(rope_dim)   ← 单组共享所有头

KV Cache:
  Window: 滑动窗口最近 window_size 个完整 KV
  Compressed: 历史 KV 经 Compressor 压缩

Attention:
  o = SDPA(q, [k_window || k_compressed], [v_window || v_compressed])
  apply_RoPE_inverse(o)  → de-rotate

Output: o → wo_a（分组）→ wo_b → dim
```

### Gate（MoE 路由）

```
# Hash-based（layer < n_hash_layers）
indices = tid2eid[input_ids]

# Score-based（其余层）
scores = sqrt(softplus(W @ x))
indices = topk(scores + bias, n_activated_experts)
weights = scores.gather(indices) / sum(weights)
```

### Expert & MoE

```
# SwiGLU FFN（单个 Expert）
output = w2(silu(clamp(w1(x))) * clamp(w3(x)))

# MoE 层
y = sum(expert_i(x) * weight_i for activated experts) + shared_expert(x)
```

### Block（使用 Hyper-Connections）

```
# 前向（x: (B,S,hc,d)）
# Attention:
y, post, comb = hc_pre(x)    # (B,S,hc,d) → (B,S,d) + 权重
y = attn(attn_norm(y))       # (B,S,d)
x = hc_post(y, x, post, comb) # 扩展回 (B,S,hc,d)

# MoE:
y, post, comb = hc_pre(x)
y = moe(ffn_norm(y), input_ids)
x = hc_post(y, x, post, comb)
```

### HCHead

从 hc_mult 份副本加权合并输出 logits：
```
pre = sigmoid(linear(flatten(x)) * hc_scale + hc_base)  # (B,S,hc_mult)
y = sum(pre * x, dim=hc_mult)  # (B,S,dim)
logits = linear(norm(y))
```

### MTPBlock

融合 embed + 主干隐层，预测下一个 token：
```
e = norm(embed(input_ids))    # (B,S,dim)
h = norm(mean(x, dim=hc))    # (B,S,dim) HC 副本均值
fused = e_proj(e) + h_proj(h) → expand → (B,S,hc,dim)
out = Block.forward(fused)
logits = head(out)
```

---

## 五、训练与推理数据流

### 训练模式（`use_cache=False`）

```
input_ids: (B, T)
    → embed → (B, T, 896)
    → unsqueeze + expand → (B, T, 2, 896)  [hc_mult=2]
    → N×Block(start_pos=0) → (B, T, 2, 896)
    → HCHead → (B, T, 129280)
    → cross_entropy(target_ids) → main_loss

MTP（可选）：
    → 取最后 Block 输出 h
    → MTPBlock(h, start_pos=0, input_ids) → (B, T, 129280)
    → cross_entropy → mtp_loss

total_loss = main_loss + 0.1 * mtp_loss
```

### 推理模式（`use_cache=True`）

**Prefill**（一次处理整个 prompt）：
```
input_ids: (B, T_prompt)
    → embed/expand/Blocks(start_pos=0)
    → 各层更新 KV cache（window + compressed）
    → HCHead → logits[:, -1, :]  # 取最后位置
    → 采样 → next_token
```

**Decode**（逐 token）：
```
input_ids: (B, 1)
    → embed/expand/Blocks(start_pos=T_prompt+k)
    → 各层读取 KV cache（window 最近64 + compressed 历史）
    → HCHead → logits[:, -1, :]
    → 采样 → next_token
```

---

## 六、KV Cache 详细说明

每一 Attention 层的 KV cache 由两部分组成：

| 部分 | 大小 | 内容 |
|------|------|------|
| 滑动窗口 | `(B, window_size=64, head_dim=128)` | 最近 64 个完整 KV |
| 压缩部分（仅压缩层） | `(B, max_seq_len//4=512, head_dim=128)` | 历史 KV 4:1 压缩 |

对于 `compress_ratios=[0,0,4,0,4,0,4,0]`：
- 第 0/1/3/5/7 层：仅滑动窗口，cache 大小 = `(B, 64, 128)`
- 第 2/4/6 层：滑动窗口 + 压缩，cache 大小 = `(B, 64+512, 128)`

---

## 七、显存占用估算

### 推理阶段（`batch_size=1, max_seq_len=2048`）

| 项目 | 显存 |
|------|------|
| 模型参数 fp16 | 356M × 2 ≈ **712 MB** |
| KV Cache（8层） | ~**120 MB** |
| 激活值（推理中间） | ~**50 MB** |
| **合计** | **~900 MB** |

### 训练阶段（`batch_size=4, seq_len=256`）

| 项目 | 显存 |
|------|------|
| 模型参数 fp16 | **712 MB** |
| 梯度 fp16 | **712 MB** |
| AdamW 优化器（fp32 m+v） | 356M × 8 ≈ **2848 MB** |
| 激活值（前向保留） | ~**400 MB** |
| **合计** | **~4.7 GB** |

T4 16GB 完全可以承载，还有约 11GB 余量。

---

## 八、文件说明

| 文件 | 说明 |
|------|------|
| `model.py` | 模型定义（ModelArgs、所有模块类、Transformer） |
| `config_mini.json` | Mini 版本超参数配置 |
| `inference.py` | 推理脚本（generate/generate_chat/CLI） |
| `train.py` | 预训练脚本（主 loss + MTP loss） |
| `train_sft.py` | SFT 微调脚本（loss_mask 只算 assistant 部分） |
| `dataset.py` | 数据集类（PretrainDataset/SFTDataset） |
| `requirements.txt` | 依赖（torch>=2.1.0, transformers>=4.46.0） |
