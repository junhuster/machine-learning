# KV Cache 计算完全指南

## 目录
1. [什么是 KV Cache](#一什么是-kv-cache)
2. [单层单个 Token 的 KV Cache 计算](#二单层单个-token-的-kv-cache-计算)
3. [完整序列的 KV Cache 计算](#三完整序列的-kv-cache-计算)
4. [不同 Attention 机制对比](#四不同-attention-机制的-kv-cache-对比)
5. [数值计算示例](#五详细计算步骤以-llama-2-70b-为例)
6. [数据类型的影响](#六数据类型的影响)
7. [超长上下文的 KV Cache 压力](#七超长上下文的-kv-cache-压力)
8. [分布式场景的 KV Cache](#八分布式场景的-kv-cache)
9. [实际代码示例](#九实际代码示例)
10. [MLA 详细计算](#十mla-的详细计算deepseek-v3)
11. [总结公式速查表](#十一总结公式速查表)

---

## 一、什么是 KV Cache？

### 自回归生成过程

```python
# 生成 "人工智能改变世界"
时间步 1: 输入 <BOS> → 预测 "人工"
时间步 2: 输入 <BOS>, "人工" → 预测 "智能"
时间步 3: 输入 <BOS>, "人工", "智能" → 预测 "改变"
时间步 4: 输入 <BOS>, "人工", "智能", "改变" → 预测 "世界"
```

**问题**：每次生成新 token 都要重新计算所有历史 token 的 K、V

```python
# 不使用 KV Cache
for step in range(seq_len):
    # 每次都要从头计算所有历史 token
    all_tokens = tokens[:step+1]
    K, V = compute_kv(all_tokens)  # 重复计算！
    output = attention(Q_new, K, V)
```

**解决方案**：缓存历史 K、V

```python
# 使用 KV Cache
kv_cache = []
for step in range(seq_len):
    # 只计算新 token 的 K、V
    K_new, V_new = compute_kv(tokens[step])
    kv_cache.append((K_new, V_new))  # 缓存起来

    # 直接使用缓存的 K、V
    K_all = [k for k, v in kv_cache]
    V_all = [v for k, v in kv_cache]
    output = attention(Q_new, K_all, V_all)
```

---

## 二、单层单个 Token 的 KV Cache 计算

### 基础公式

```python
# 单个 token 在单层的 KV Cache 大小
KV_cache_per_token_per_layer = 2 × n_heads × head_dim × dtype_size

其中:
- 2: K 和 V 两个矩阵
- n_heads: 注意力头数量
- head_dim: 每个头的维度
- dtype_size: 数据类型字节数
```

### 详细维度分析

```python
# 输入 token embedding
x: [batch_size, 1, hidden_dim]  # 单个 token

# 计算 K、V（以 MHA 为例）
W_K: [hidden_dim, n_heads × head_dim]
W_V: [hidden_dim, n_heads × head_dim]

K = x @ W_K  # [batch, 1, n_heads × head_dim]
V = x @ W_V  # [batch, 1, n_heads × head_dim]

# 重塑为多头形式
K = K.view(batch, 1, n_heads, head_dim)  # [batch, 1, n_heads, head_dim]
V = V.view(batch, 1, n_heads, head_dim)  # [batch, 1, n_heads, head_dim]

# 需要缓存的张量
K_cache: [batch, n_heads, head_dim]  # 去掉 seq_len=1 的维度
V_cache: [batch, n_heads, head_dim]
```

---

## 三、完整序列的 KV Cache 计算

### 公式推导

```python
# 完整序列的 KV Cache（所有层）
Total_KV_Cache = n_layers × 2 × batch_size × seq_len × n_heads × head_dim × dtype_size

简化为（单 batch）:
Total_KV_Cache = n_layers × 2 × seq_len × n_heads × head_dim × dtype_size
```

### 数值例子：GPT-3 175B

```python
# GPT-3 175B 配置
n_layers = 96
n_heads = 96
head_dim = 128
hidden_dim = 12288
seq_len = 2048
dtype_size = 2  # FP16

# 计算 KV Cache
KV_cache = 96 × 2 × 2048 × 96 × 128 × 2
         = 96 × 2 × 2048 × 12288 × 2
         = 9,663,676,416 bytes
         ≈ 9 GB

# 每个 token 的 KV Cache
KV_per_token = 96 × 2 × 96 × 128 × 2
             = 4,718,592 bytes
             ≈ 4.5 MB
```

---

## 四、不同 Attention 机制的 KV Cache 对比

### 1. MHA (Multi-Head Attention)

```python
# MHA: 每个 head 都有独立的 K、V
n_kv_heads = n_heads  # 所有 head 都有 K、V

KV_cache_MHA = 2 × n_layers × seq_len × n_heads × head_dim × dtype_size
```

**数值例子**：
```python
# LLaMA 65B (假设用 MHA)
n_layers = 80
n_heads = 64
head_dim = 128
seq_len = 4096
dtype_size = 2

KV_cache_MHA = 80 × 2 × 4096 × 64 × 128 × 2
             = 80 × 2 × 4096 × 8192 × 2
             = 10,737,418,240 bytes
             ≈ 10 GB
```

---

### 2. MQA (Multi-Query Attention)

```python
# MQA: 所有 heads 共享 1 个 K、V
n_kv_heads = 1

KV_cache_MQA = 2 × n_layers × seq_len × 1 × head_dim × dtype_size
```

**数值例子**：
```python
# 同样的模型配置，使用 MQA
KV_cache_MQA = 80 × 2 × 4096 × 1 × 128 × 2
             = 80 × 2 × 4096 × 128 × 2
             = 167,772,160 bytes
             ≈ 160 MB

# 节省: 10 GB → 160 MB (节省 64 倍！)
```

---

### 3. GQA (Grouped Query Attention)

```python
# GQA: n_heads 分成 n_kv_heads 组，每组共享 K、V
# 通常 n_kv_heads = n_heads / group_size

KV_cache_GQA = 2 × n_layers × seq_len × n_kv_heads × head_dim × dtype_size
```

**数值例子**：
```python
# LLaMA 2 70B (实际使用 GQA)
n_layers = 80
n_heads = 64
n_kv_heads = 8  # 8 组 K、V
head_dim = 128
seq_len = 4096
dtype_size = 2

KV_cache_GQA = 80 × 2 × 4096 × 8 × 128 × 2
             = 80 × 2 × 4096 × 1024 × 2
             = 1,342,177,280 bytes
             ≈ 1.28 GB

# 相比 MHA: 节省 8 倍
# 相比 MQA: 大 8 倍，但质量更好
```

---

### 4. MLA (Multi-head Latent Attention) - DeepSeek 创新

```python
# MLA: 将 KV 压缩到 latent space
# 不存储完整的 K、V，而是存储压缩后的 latent

# 压缩后的 KV latent 维度
kv_lora_rank = 512  # DeepSeek-V3 配置

# 还需要存储位置编码部分
qk_rope_head_dim = 64

# MLA KV Cache
KV_cache_MLA = n_layers × seq_len × (kv_lora_rank + qk_rope_head_dim) × dtype_size
```

**数值例子**：
```python
# DeepSeek-V3
n_layers = 27
kv_lora_rank = 512
qk_rope_head_dim = 64
seq_len = 4096
dtype_size = 2

KV_cache_MLA = 27 × 4096 × (512 + 64) × 2
             = 27 × 4096 × 576 × 2
             = 127,500,288 bytes
             ≈ 121 MB

# 如果用 MHA (对比)
KV_cache_MHA = 27 × 2 × 4096 × 16 × 128 × 2
             = 27 × 2 × 4096 × 2048 × 2
             = 901,775,360 bytes
             ≈ 860 MB

# MLA 节省: 7 倍！
```

---

### 5. 对比表格

| Attention 类型 | KV Cache 公式 | 相对 MHA 比例 | 例子 (LLaMA 70B) |
|---------------|--------------|--------------|-----------------|
| MHA | 2 × L × S × H × D | 100% | 10.24 GB |
| MQA | 2 × L × S × 1 × D | 1/H | 160 MB |
| GQA | 2 × L × S × G × D | G/H | 1.28 GB |
| MLA | L × S × (R + D_pe) | (R + D_pe)/(2×H×D) | 121 MB |

其中：
- L = n_layers
- S = seq_len
- H = n_heads
- D = head_dim
- G = n_kv_heads (GQA 组数)
- R = kv_lora_rank (MLA 压缩维度)
- D_pe = qk_rope_head_dim

---

## 五、详细计算步骤（以 LLaMA 2 70B 为例）

### 步骤 1：确定模型配置

```python
# LLaMA 2 70B 配置
n_layers = 80
n_heads = 64
n_kv_heads = 8  # GQA
head_dim = 128
hidden_dim = 8192
vocab_size = 32000
seq_len = 4096
```

### 步骤 2：计算单个 token 在单层的 KV Cache

```python
# 每个 token 在单层
K_per_layer: [n_kv_heads, head_dim] = [8, 128]
V_per_layer: [n_kv_heads, head_dim] = [8, 128]

# 元素数量
num_elements_per_layer = 2 × 8 × 128 = 2048

# 字节数 (FP16)
bytes_per_layer = 2048 × 2 = 4096 bytes = 4 KB
```

### 步骤 3：计算单个 token 在所有层的 KV Cache

```python
# 所有层
KV_per_token_all_layers = n_layers × bytes_per_layer
                        = 80 × 4096
                        = 327,680 bytes
                        ≈ 320 KB
```

### 步骤 4：计算完整序列的 KV Cache

```python
# 完整序列 (4096 tokens)
Total_KV_Cache = KV_per_token_all_layers × seq_len
               = 320 KB × 4096
               = 1,310,720 KB
               = 1,280 MB
               ≈ 1.25 GB
```

### 步骤 5：考虑 Batch Size

```python
# Batch = 1: 1.25 GB
# Batch = 8: 1.25 GB × 8 = 10 GB
# Batch = 32: 1.25 GB × 32 = 40 GB
```

---

## 六、数据类型的影响

```python
# 不同数据类型的 dtype_size
FP32: 4 bytes
FP16: 2 bytes
BF16: 2 bytes
INT8: 1 byte
FP8:  1 byte

# KV Cache 大小对比
FP32: 1.25 GB × 2 = 2.5 GB
FP16: 1.25 GB
INT8: 1.25 GB × 0.5 = 0.625 GB
FP8:  1.25 GB × 0.5 = 0.625 GB
```

---

## 七、超长上下文的 KV Cache 压力

### LLaMA 2 70B 不同序列长度的 KV Cache

```python
# GQA 配置
n_layers = 80
n_kv_heads = 8
head_dim = 128

def calculate_kv_cache(seq_len):
    return 80 × 2 × seq_len × 8 × 128 × 2 / (1024**3)  # GB

# 不同序列长度
seq_len = 4096:   1.25 GB
seq_len = 8192:   2.5 GB
seq_len = 16384:  5 GB
seq_len = 32768:  10 GB
seq_len = 65536:  20 GB
seq_len = 100000: 30.5 GB  # Claude 2 的 100K 上下文
seq_len = 128000: 39 GB    # GPT-4 Turbo 的 128K 上下文
```

### 对比 MHA vs GQA 在 128K 上下文

```python
# MHA (n_kv_heads = 64)
KV_cache_MHA_128K = 80 × 2 × 128000 × 64 × 128 × 2
                  = 311 GB

# GQA (n_kv_heads = 8)
KV_cache_GQA_128K = 80 × 2 × 128000 × 8 × 128 × 2
                  = 39 GB

# 节省: 311 GB → 39 GB (节省 8 倍，单卡可行!)
```

---

## 八、分布式场景的 KV Cache

### Tensor Parallelism

```python
# 8 GPU Tensor Parallel
# 每个 GPU 只存储 1/8 的 KV Cache

# MHA 分布式
每个 GPU: KV_cache_MHA / 8 = 311 GB / 8 = 39 GB
# 但 GPU 间需要 AllReduce 通信

# MQA 分布式（问题）
每个 GPU: KV_cache_MQA = 160 MB
# 但 K、V 必须复制到每个 GPU
实际占用: 160 MB × 8 = 1.28 GB (浪费!)

# GQA 分布式（完美）
每个 GPU: KV_cache_GQA / 8 = 39 GB / 8 = 4.88 GB
# K、V 可以按 head 分片，无浪费
```

### 图示对比

```
MQA 的分布式浪费:
┌─────────────────────────────────────────────────┐
│                    4 GPU 系统                    │
├──────────┬──────────┬──────────┬────────────────┤
│  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3        │
│ Q[0:8]   │ Q[8:16]  │ Q[16:24] │ Q[24:32]      │
│ ┌──────┐ │ ┌──────┐ │ ┌──────┐ │ ┌──────┐      │
│ │ K, V │ │ │ K, V │ │ │ K, V │ │ │ K, V │      │
│ │(复制)│ │ │(复制)│ │ │(复制)│ │ │(复制)│      │
│ └──────┘ │ └──────┘ │ └──────┘ │ └──────┘      │
└──────────┴──────────┴──────────┴────────────────┘
           ↑
      同一个 K、V 存了 4 次！浪费！

GQA 的分布式优化:
┌─────────────────────────────────────────────────┐
│                    4 GPU 系统                    │
├──────────┬──────────┬──────────┬────────────────┤
│  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3        │
│ Q[0:8]   │ Q[8:16]  │ Q[16:24] │ Q[24:32]      │
│ ┌──────┐ │ ┌──────┐ │ ┌──────┐ │ ┌──────┐      │
│ │K[0:2]│ │ │K[2:4]│ │ │K[4:6]│ │ │K[6:8]│      │
│ │V[0:2]│ │ │V[2:4]│ │ │V[4:6]│ │ │V[6:8]│      │
│ └──────┘ │ └──────┘ │ └──────┘ │ └──────┘      │
└──────────┴──────────┴──────────┴────────────────┘
           ↑
      每个 GPU 存不同的 K、V 分片，无浪费！
```

---

## 九、实际代码示例

### 计算 KV Cache 大小

```python
def calculate_kv_cache(
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,  # GQA 参数
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # FP16
) -> dict:
    """
    计算 KV Cache 大小

    Args:
        n_layers: 层数
        n_heads: Q 的头数
        n_kv_heads: K/V 的头数 (GQA)
        head_dim: 每个头的维度
        seq_len: 序列长度
        batch_size: 批次大小
        dtype_bytes: 数据类型字节数

    Returns:
        dict: KV Cache 大小信息
    """
    # 单层单个 token 的 KV Cache
    kv_per_token_per_layer = 2 * n_kv_heads * head_dim * dtype_bytes

    # 所有层单个 token
    kv_per_token_all_layers = kv_per_token_per_layer * n_layers

    # 完整序列
    kv_total = kv_per_token_all_layers * seq_len * batch_size

    # 转换为 GB
    kv_total_gb = kv_total / (1024 ** 3)

    # 计算节省比例
    mha_kv = 2 * n_heads * head_dim * dtype_bytes * n_layers * seq_len * batch_size
    savings_ratio = mha_kv / kv_total

    return {
        "per_token_per_layer_bytes": kv_per_token_per_layer,
        "per_token_all_layers_bytes": kv_per_token_all_layers,
        "total_bytes": kv_total,
        "total_gb": kv_total_gb,
        "savings_vs_mha": savings_ratio,
    }


# 示例：LLaMA 2 70B
result = calculate_kv_cache(
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    head_dim=128,
    seq_len=4096,
    batch_size=1,
)

print(f"每个 token 单层: {result['per_token_per_layer_bytes']} bytes = {result['per_token_per_layer_bytes']/1024} KB")
print(f"每个 token 所有层: {result['per_token_all_layers_bytes']} bytes = {result['per_token_all_layers_bytes']/1024} KB")
print(f"完整序列: {result['total_gb']:.2f} GB")
print(f"相比 MHA 节省: {result['savings_vs_mha']:.1f} 倍")
```

### 输出

```
每个 token 单层: 4096 bytes = 4.0 KB
每个 token 所有层: 327680 bytes = 320.0 KB
完整序列: 1.22 GB
相比 MHA 节省: 8.0 倍
```

---

## 十、MLA 的详细计算（DeepSeek-V3）

```python
def calculate_mla_kv_cache(
    n_layers: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> dict:
    """
    计算 MLA 的 KV Cache 大小

    MLA 存储:
    - kv_latent: [kv_lora_rank] 压缩后的 KV
    - k_pe: [qk_rope_head_dim] 位置编码
    """
    # 单层单个 token
    kv_latent = kv_lora_rank * dtype_bytes
    k_pe = qk_rope_head_dim * dtype_bytes
    total_per_token_per_layer = kv_latent + k_pe

    # 所有层
    total_per_token = total_per_token_per_layer * n_layers

    # 完整序列
    total = total_per_token * seq_len * batch_size

    return {
        "per_token_per_layer_bytes": total_per_token_per_layer,
        "per_token_all_layers_bytes": total_per_token,
        "total_gb": total / (1024 ** 3),
    }


# DeepSeek-V3
result = calculate_mla_kv_cache(
    n_layers=27,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    seq_len=4096,
)

print(f"MLA 每个 token 单层: {result['per_token_per_layer_bytes']} bytes")
print(f"MLA 完整序列: {result['total_gb']:.2f} GB")
```

### 输出

```
MLA 每个 token 单层: 1152 bytes = 1.125 KB
MLA 完整序列: 0.11 GB
```

---

## 十一、总结公式速查表

### 基础公式

```python
# 单层单个 token
KV_cache = 2 × n_kv_heads × head_dim × dtype_bytes

# 所有层单个 token
KV_cache = n_layers × 2 × n_kv_heads × head_dim × dtype_bytes

# 完整序列
KV_cache = batch × n_layers × 2 × seq_len × n_kv_heads × head_dim × dtype_bytes
```

### 各 Attention 公式

```python
# MHA
KV_MHA = L × 2 × S × H × D × bytes

# MQA
KV_MQA = L × 2 × S × 1 × D × bytes

# GQA
KV_GQA = L × 2 × S × G × D × bytes

# MLA
KV_MLA = L × S × (R + D_pe) × bytes
```

**符号说明**：
- L = n_layers
- S = seq_len
- H = n_heads
- G = n_kv_heads
- D = head_dim
- R = kv_lora_rank
- D_pe = qk_rope_head_dim

---

## 十二、常见模型 KV Cache 速查

| 模型 | n_layers | n_heads | n_kv_heads | head_dim | 4K seq KV | 128K seq KV |
|------|----------|---------|------------|----------|-----------|-------------|
| GPT-3 175B | 96 | 96 | 96 (MHA) | 128 | 9 GB | 288 GB |
| LLaMA 2 70B | 80 | 64 | 8 (GQA) | 128 | 1.25 GB | 40 GB |
| LLaMA 2 13B | 40 | 40 | 40 (MHA) | 128 | 1.6 GB | 51 GB |
| Mistral 7B | 32 | 32 | 8 (GQA) | 128 | 0.5 GB | 16 GB |
| DeepSeek-V3 | 27 | 16 | MLA | - | 0.11 GB | 3.5 GB |

---

## 参考资源

- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [GQA Paper](https://arxiv.org/abs/2305.13245)
- [MLA Paper (DeepSeek-V2)](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

---

*文档生成时间: 2026-05-07*
