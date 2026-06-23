# LLM 推理优化论文导读合集

> 本合集涵盖 LLM 推理优化的核心论文，分为两大块：
> - **第一部分：工业界主流应用**——这些技术已在 vLLM、TGI、TensorRT-LLM 等主流框架中落地，或已被 LLaMA/Mistral/Gemma 等主流模型采用，是生产部署的标配。
> - **第二部分：学术界研究热门**——这些方向在论文和研究社区中活跃，工业落地程度不一，代表前沿探索方向。

---

## 目录

### 第一部分：工业界主流应用

1. [MQA：Multi-Query Attention](#1-mqa-multi-query-attention)
2. [GQA：Grouped-Query Attention](#2-gqa-grouped-query-attention)
3. [FlashAttention](#3-flashattention)
4. [FlashAttention-2](#4-flashattention-2)
5. [FlashAttention-3](#5-flashattention-3)
6. [PagedAttention](#6-pagedattention)
7. [ORCA：Continuous Batching](#7-orca-continuous-batching)
8. [LLM.int8()](#8-llmint8)
9. [GPTQ](#9-gptq)
10. [AWQ](#10-awq)
11. [Speculative Decoding（Leviathan et al.）](#11-speculative-decoding)

### 第二部分：学术界研究热门

12. [Speculative Decoding（Seq2seq，Xia et al.）](#12-speculative-decoding-seq2seq)
13. [BigBird：稀疏 Attention 长序列](#13-bigbird)
14. [Early Exit / PABEE](#14-early-exit--pabee)
15. [LayerSkip：自投机解码](#15-layerskip)
16. [Sorted LLaMA：动态深度推理](#16-sorted-llama)
17. [KVQuant：KV Cache 量化](#17-kvquant)
18. [SparseGPT：LLM 剪枝](#18-sparsegpt)
19. [Distilling Step-by-Step](#19-distilling-step-by-step)
20. [FlexGen：单 GPU 高吞吐推理](#20-flexgen)

---

# 第一部分：工业界主流应用

---

## 1. MQA：Multi-Query Attention

**论文**：Fast Transformer Decoding: One Write-Head is All You Need
**作者**：Noam Shazeer（Google）
**发表**：2019（arXiv:1911.02150）
**链接**：https://arxiv.org/abs/1911.02150

---

### 定位

这篇论文提出了 **Multi-Query Attention（MQA）**，是现代 LLM 推理优化的基础技术之一。
PaLM、Falcon、StarCoder 等模型直接采用，后续 GQA（Grouped-Query Attention）是其泛化版本，
Llama 2/3、Mistral、Gemma 均在此基础上演进。

---

### 背景：推理为什么慢？

Transformer 训练和推理的计算特征完全不同：

**训练时**：
```
所有 token 并行处理，矩阵乘法规模大
计算密集型（compute-bound）
GPU 算力是瓶颈
```

**自回归解码时（逐 token 生成）**：
```
每步只生成 1 个 token
需要读取之前所有 token 的 KV cache
计算量极小（只有一行向量 × 矩阵），但内存读取量很大
内存带宽密集型（memory-bandwidth-bound）
内存带宽是瓶颈，不是算力
```

**关键问题**：每生成一个 token，需要把整个 KV cache 从 HBM 读到计算单元，
KV cache 越大，读取越慢，解码越慢。

---

### MQA 的核心思想

**只保留一组 K 和 V，所有 query head 共享**：

```
MHA（原始）：
  Q: [batch, heads, seq_len, head_dim]   ← h 组
  K: [batch, heads, seq_len, head_dim]   ← h 组
  V: [batch, heads, seq_len, head_dim]   ← h 组

MQA（本文）：
  Q: [batch, heads, seq_len, head_dim]   ← h 组（不变）
  K: [batch, 1,     seq_len, head_dim]   ← 1 组（共享）
  V: [batch, 1,     seq_len, head_dim]   ← 1 组（共享）
```

**KV cache 缩小 heads 倍**：
```
MHA KV cache：2 × batch × heads × seq_len × head_dim
MQA KV cache：2 × batch × 1     × seq_len × head_dim

节省比例 = heads 倍（通常 heads=32~96，节省 32~96×）
```

---

### 实验结果

**翻译任务（WMT14 英德）**：

| 模型 | BLEU（beam=4） | 解码速度（μs/token） |
|------|--------------|-------------------|
| MHA（基线） | 28.4 | 46 |
| **MQA** | **28.5** | **3.8** |

解码速度提升 **~12×**，质量几乎不变（甚至略好）。

---

### 与后续工作的关系

```
MHA：H 组 KV，质量最好，KV cache 最大
MQA：1 组 KV，速度最快，KV cache 最小，质量略降
GQA：G 组 KV（1 < G < H），质量和速度的折中（Llama 2/3 采用）

GQA（2023）= MQA 的泛化版本：
  G=1  → MQA
  G=H  → MHA
  G=8  → 典型 GQA（Llama 3 用 8 个 KV head，32 个 Q head）
```

---

### 论文结构速览

```
Section 1: Introduction       ← 推理 memory-bound 问题，MQA 的动机
Section 2: Multi-Head Attention ← MHA 回顾
Section 3: Multi-Query Attention ← MQA 定义，与 MHA 的差异
Section 4: Incremental Inference ← 逐 token 解码的内存分析（重点）
Section 5: Experiments        ← WMT14 翻译，速度 vs 质量
Section 6: Conclusion         ← 总结，与其他优化正交
```

---

### 核心主线

> 自回归解码是 memory-bandwidth-bound，瓶颈是每步读取完整的 KV cache。
> MQA 让所有 Q head 共享同一组 K/V，KV cache 缩小 H 倍，内存读取减少 H 倍，
> 解码速度提升约 12×，训练质量几乎不变。
> 核心洞察：模型需要多个"读头"（Q），但"写头"（K/V）只需一个。

---

## 2. GQA：Grouped-Query Attention

**论文**：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
**作者**：Joshua Ainslie, James Lee-Thorp et al.（Google Research）
**发表**：EMNLP 2023（arXiv:2305.13245）
**链接**：https://arxiv.org/abs/2305.13245

---

### 定位：MHA 和 MQA 之间的工程折中

上一篇 MQA 把 KV head 压到 1，速度快但质量有所下降。
本文提出 GQA，在两者之间找到最优平衡点，同时解决一个工程问题：
**如何把已训练好的 MHA 模型转换为 GQA，而不需要从头训练**。

```
MHA：H 组 KV → 质量最好，KV cache 最大，速度最慢
MQA：1 组 KV → 速度最快，KV cache 最小，质量略降，训练不稳定
GQA：G 组 KV → 质量接近 MHA，速度接近 MQA（1 < G < H）
```

---

### GQA 的结构设计

**核心思想**：把 H 个 Query head 分成 G 组，每组共享一对 KV head。

```
H = 8 个 Query head，G = 2 组 KV head：

组 0：Q_0, Q_1, Q_2, Q_3 共享 K_0, V_0
组 1：Q_4, Q_5, Q_6, Q_7 共享 K_1, V_1
```

**三种方案对比**：

| 方案 | Q heads | KV heads | KV cache 大小 | 质量 |
|------|---------|---------|--------------|------|
| MHA | H | H | H × seq × d | 最好 |
| GQA-G | H | G | G × seq × d | 接近 MHA |
| MQA | H | 1 | 1 × seq × d | 略降 |

当 G=1 → MQA；当 G=H → MHA；GQA 是统一的通用形式。

---

### Uptraining：从 MHA checkpoint 转换（论文核心贡献）

**Step 1：Mean Pooling 构造初始 GQA 权重**

每组 KV head 通过对该组内所有原始 KV head 做**均值池化**得到：

```
原始 MHA：H 个 K 投影矩阵 W_K_0, W_K_1, ..., W_K_{H-1}

GQA-G 转换（H=8，G=2）：
  组 0 的 KV head：W_K_new_0 = mean(W_K_0, W_K_1, W_K_2, W_K_3)
  组 1 的 KV head：W_K_new_1 = mean(W_K_4, W_K_5, W_K_6, W_K_7)
```

**Step 2：Uptraining（少量继续预训练）**

用原始预训练数据量的 **5%** 继续训练，让模型适应新的 GQA 结构，5% 的数据量足以恢复 GQA 的质量，不需要完整重训。

---

### 工业界采用情况

| 模型 | 方案 | KV head 数 |
|------|------|-----------|
| Llama 2 7B/13B | MHA | = Q head 数 |
| **Llama 2 70B** | **GQA-8** | 8 |
| **Llama 3 全系列** | **GQA-8** | 8 |
| Mistral 7B | GQA-8 | 8 |
| Gemma 系列 | GQA | 视规格 |
| Falcon | MQA | 1 |

GQA-8 已成为大模型的事实标准配置。

---

### 核心主线

> GQA 的核心是两件事：
> 1. **架构**：把 H 个 Q head 分成 G 组，每组共享一对 KV，KV cache 缩小 H/G 倍，是 MHA 和 MQA 的统一泛化（G=1→MQA，G=H→MHA）。
> 2. **Uptraining**：用 mean pooling 把 MHA checkpoint 转成 GQA 初始化，再用 5% 数据继续预训练，无需从头训练即可获得 GQA 模型。

---

## 3. FlashAttention

**论文**：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
**作者**：Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré（Stanford）
**发表**：NeurIPS 2022（arXiv:2205.14135）
**链接**：https://arxiv.org/abs/2205.14135
**代码**：https://github.com/Dao-AILab/flash-attention

---

### 定位

FlashAttention 是目前所有主流 LLM 训练和推理框架的标配 Attention 实现，
解决的是标准 Attention 的 **IO 瓶颈**——不是通过近似，而是精确计算，
通过重新组织计算顺序让数据尽量留在快速的片上 SRAM 而不是来回读写慢速的 HBM。

---

### 背景：GPU 的内存层次与带宽差异

```
SRAM（片上缓存）：容量 ~20 MB，带宽 ~19 TB/s
HBM（显存）：    容量 40~80 GB，带宽 ~2 TB/s

带宽比：SRAM : HBM ≈ 10 : 1
```

**关键结论**：数据在 SRAM 里计算极快，一旦需要从 HBM 读写就慢很多。

---

### 标准 Attention 的 IO 问题

```python
S = Q @ K.T          # [N, N]  ← 写入 HBM
P = softmax(S)        # [N, N]  ← 写入 HBM（需要读 S）
O = P @ V            # [N, d]  ← 写入 HBM（需要读 P）

总 HBM 访问量：Θ(Nd + N²)
N=8192 时：N² = 64M → 平方增长，带宽成为瓶颈
```

**根本问题**：N×N 的 attention 矩阵太大，装不进 SRAM，只能在 HBM 里来回读写。

---

### 核心思想：Tiling（分块计算）

**目标**：永远不把完整的 N×N attention 矩阵写到 HBM，在 SRAM 内完成所有中间计算。

把 Q、K、V 切成小块，每次只取一小块到 SRAM 里计算，累积结果后再写回 HBM。

```
外层循环（遍历 K、V 的 block）：
  for j = 1 to T_c:
    从 HBM 加载 K_j, V_j → SRAM

    内层循环（遍历 Q 的 block）：
      for i = 1 to T_r:
        从 HBM 加载 Q_i → SRAM
        计算 S_ij = Q_i × K_j^T（在 SRAM 内）
        用 online softmax 更新 O_i（在 SRAM 内）
        写回 O_i → HBM

整个过程：N×N 的 attention 矩阵从未出现在 HBM 中！
```

---

### Online Softmax：精确计算的数学基础

维护两个统计量 (m, ℓ) 做增量更新，处理完所有 block 后精确归一化，结果与标准实现完全一致（exact，非近似）。

---

### 反向传播：重计算替代存储

前向只存 O 和统计量 (m, ℓ)（O(N) 显存），反向时重新计算 P，N×N 矩阵不需要存在 HBM。

---

### 实验结果

| 指标 | 标准 Attention | FlashAttention |
|------|--------------|----------------|
| HBM 访问量 | Θ(Nd + N²) | Θ(N²d²/M)，减少 5~9× |
| 显存 | O(N²) | **O(N)** |
| 训练速度（GPT-2） | 基准 | **3×** |
| 长序列（8K~64K） | OOM | **正常运行** |

---

### 核心主线

> FlashAttention 把 Attention 从**计算问题**重新定义为 **IO 问题**。
> 解法是 **Tiling**：分块在 SRAM 内完成所有中间计算，
> 用 **Online Softmax** 保证结果精确，用**反向重计算**替代 N×N 矩阵存储。
> 最终：HBM 访问减少 5~9×，显存从 O(N²) 降到 O(N)，训练速度 2~4×，结果完全精确。

---

## 4. FlashAttention-2

**论文**：FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
**作者**：Tri Dao（Princeton University）
**发表**：ICLR 2024（arXiv:2307.08691）
**链接**：https://arxiv.org/abs/2307.08691

---

### 定位：FA1 的工程深度优化

FlashAttention-1 解决了"把计算搬进 SRAM"的算法问题，
但在 A100 上实际只达到理论峰值的 **25~40%**。

FlashAttention-2 不改变核心算法思路，专注于三个 GPU 工程层面的优化：

```
问题 1：每个 block 迭代都做一次输出 rescaling → 多余的非矩阵乘法计算
问题 2：Warp 之间工作分配不合理 → 共享内存读写多，同步开销大
问题 3：只在 batch × heads 维度并行 → 长序列时 GPU 占用率低
```

---

### 三大改进

**改进一：减少非矩阵乘法 FLOPs**

推迟 rescaling 到最后一步，把 T_c 次 exp 操作降为 1 次除法。（exp 比 matmul 慢 16×，效果显著）

**改进二：Split-Q Warp 分工**

```
FA1（Split-K）：K/V 分片 → 各 Warp 算部分 → 需要跨 Warp 合并（SRAM 通信）
FA2（Split-Q）：Q 分片   → 各 Warp 独立算整个 attention → 无需通信
```

**改进三：序列长度维度并行**

```
FA1：Thread Block 数 = batch × heads
FA2：Thread Block 数 = batch × heads × (seq_len / B_r)
→ 长序列时 SM 利用率大幅提升
```

---

### 性能结果

| 指标 | FA1 | FA2 | 提升 |
|------|-----|-----|------|
| A100 MFU | 25~40% | **50~73%** | ~1.5× |
| 端到端加速（vs FA1） | 基准 | **~2×** | 2× |
| 新增支持 | MHA | MHA + **MQA/GQA/ALiBi/滑窗** | — |

---

### 核心主线

> FA2 的本质是**在 FA1 的算法框架内，把 GPU 利用率从 25~40% 提升到 50~73%**。
> 1. 减少 rescaling → 省掉 16× 慢的非 matmul FLOPs
> 2. Split-Q warp 分工 → 消除跨 Warp 的 SRAM 通信和同步
> 3. 序列长度维度并行 → 长序列时 SM 满负荷

---

## 5. FlashAttention-3

**论文**：FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
**作者**：Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thambidurai, Atri Rudra, Tri Dao
**发表**：2024（arXiv:2407.08608）
**链接**：https://arxiv.org/abs/2407.08608

---

### 定位：专为 H100 硬件重新设计

FA2 在 H100 上只达到 **35% 的理论峰值**——H100 有大量 FA2 未利用的新特性。

```
H100 三个新特性          FA3 对应解法
TMA（异步传输）     →  Producer-Consumer 异步流水线
WGMMA（异步 matmul）→  GEMM-Softmax Ping-Pong 交错执行
FP8 Tensor Core   →  块量化 + 不相干处理
```

---

### 三大核心改进

**改进一：Producer-Consumer 异步流水线（利用 TMA）**

```
Producer Warp：专门异步加载 K/V tiles，发完立即预取下一个
Consumer Warp：计算 WGMMA，数据就绪后直接用

时间线：
  TMA 预取 K_{j+1}, V_{j+1} ────────────────┐
  WGMMA 计算 Q × K_j ──────┐                │
  WGMMA 计算 P × V_j ──────┘─────┐           │
  Softmax               ─────┘           │
                                         ↓
  下一轮：数据已就绪，消除等待
```

**改进二：Ping-Pong 调度（隐藏 Softmax 延迟）**

```
H100 WGMMA vs 非 matmul 吞吐比：~250:1

两个 Warpgroup 交替执行：
  WG_A 做 Softmax（慢）时，WG_B 同时做 WGMMA（快）
  WGMMA 单元几乎无空闲
```

**改进三：FP8 块量化 + 不相干处理**

```
不相干处理：量化前乘以随机正交矩阵 R，把 Outlier "分散"到所有维度
效果：FP8 量化误差降低 2.6×
FP8 吞吐：~1.2 PetaFLOPS（单卡 H100）
```

---

### 性能结果

| 指标 | FA2（H100） | FA3（H100） |
|------|------------|------------|
| FP16 吞吐 | ~350 TFLOPS | **~740 TFLOPS** |
| H100 MFU | 35% | **75%** |
| FP8 吞吐 | N/A | **~1200 TFLOPS（>1 PetaFLOPS）** |

---

### 核心主线

> FA3 充分利用 H100 三个新硬件特性：
> 1. **TMA** → 异步预取，数据加载与计算完全重叠
> 2. **WGMMA 异步** → Ping-Pong 调度，Softmax 和 GEMM 交错，WGMMA 满负荷
> 3. **FP8** → 块量化 + 不相干处理，2× 吞吐，精度可控
> H100 MFU 从 35% 提升到 75%。

---

## 6. PagedAttention

**论文**：Efficient Memory Management for Large Language Model Serving with PagedAttention
**作者**：Woosuk Kwon, Zhuohan Li, Siyuan Zhuang et al.（UC Berkeley & Stanford）
**发表**：SOSP 2023（arXiv:2309.06180）
**链接**：https://arxiv.org/abs/2309.06180
**项目**：https://github.com/vllm-project/vllm

---

### 定位

PagedAttention 是目前最主流的 LLM 推理引擎（vLLM）的核心技术，
灵感来自操作系统的**虚拟内存和分页机制**，解决 KV cache 的内存管理问题。

---

### 背景：KV Cache 的内存浪费有多严重

```
传统系统（预分配连续内存块）：
  按最大序列长度预分配 → 大量内部碎片
  不同请求大小不同 → 大量外部碎片

实测：13B 模型在 A100 上，KV cache 利用率只有 20.4%~38.2%
```

---

### 核心思想：类操作系统分页

| OS 概念 | PagedAttention 对应 |
|---------|-------------------|
| 虚拟内存页 | 逻辑 KV Block |
| 物理内存页 | 物理 KV Block |
| 页表 | Block Table |
| 写时复制（CoW） | KV block CoW |
| 页面置换 | Swap / Recompute |

**核心机制**：
- KV cache 切成固定大小的 block，逻辑连续但物理不连续
- Block Table 动态映射，按需分配
- Copy-on-Write 支持 parallel sampling 和 beam search 的 prompt 共享

---

### 内存利用率分析

```
传统系统：利用率 20.4%~38.2%
PagedAttention：接近 100%（内外碎片接近零）

→ 同等 GPU 显存，能同时服务更多请求 → 吞吐大幅提升
```

---

### 性能结果

| 系统 | 吞吐（相对 HF TGI） |
|------|-----------------|
| HuggingFace TGI | 1× |
| FasterTransformer | ~1.5× |
| Orca | ~2× |
| **vLLM（PagedAttention）** | **2~4×** |

---

### 核心主线

> PagedAttention 把操作系统的虚拟内存分页思想引入 KV cache 管理：
> 逻辑连续但物理不连续，block table 动态映射，按需分配，
> 内存利用率从 20~38% 提升到接近 100%。
> Copy-on-Write 让共享 prompt 的多个序列复用物理 block。
> 最终：vLLM 吞吐比 HuggingFace TGI 高 2~4×。

---

## 7. ORCA：Continuous Batching

**论文**：ORCA: A Distributed Serving System for Transformer-Based Generative Models
**作者**：Gyeong-In Yu, Joo Seong Jeong et al.（Seoul National University）
**发表**：OSDI 2022
**链接**：https://arxiv.org/abs/2206.04083

---

### 定位

ORCA 是现代 LLM serving 系统的奠基论文，提出了**迭代级调度（Iteration-Level Scheduling）**，
即今天广泛使用的 **Continuous Batching** 的原型。

---

### 背景：Request-Level Batching 的问题

```
传统批处理：
  batch = {A(500步), B(10步), C(200步)}，等所有人结束才返回
  → B 第 10 步完成，却要等 A 的 500 步 → Head-of-Line Blocking
  → GPU 利用率只有 10~30%
```

---

### 核心创新一：Iteration-Level Scheduling

**核心思想**：把调度粒度从"整个请求"改为"每一次 forward 迭代"。

```
每次迭代后重新调度：
  1. 完成的请求立刻移出 batch，立刻返回结果
  2. 等待队列中有空位的请求立刻加入
  → GPU 持续满负荷，这就是 Continuous Batching 名称的由来
```

---

### 核心创新二：Selective Batching

**问题**：batch 中不同请求 KV cache 长度不同，无法统一做 attention。

**解法**：
```
Linear / FFN / Embedding：所有 token 拼接 → 一次批处理（高效）
Attention：每个请求单独计算（各自的 KV cache 长度不同）
```

---

### 性能结果

```
吞吐量提升：
  小模型（GPT-2 1.5B）：提升约 10×
  大模型（GPT-3 175B 规格）：提升约 36.9×
```

---

### 核心主线

> ORCA 的核心洞察：**LLM 的生成长度不可预测，不能等整个 batch 都结束再换批。**
> 把调度粒度从"每个请求"改为"每次迭代"——完成的请求立刻离开，新请求立刻加入，
> GPU 始终保持满负荷（Continuous Batching）。
> Selective Batching 解决变长序列的批处理难题。
> 这两个思想奠定了现代 LLM serving 系统的基础架构。

---

## 8. LLM.int8()

**论文**：LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
**作者**：Tim Dettmers, Mike Lewis, Yannic Belanger, Luke Zettlemoyer（UW + Meta AI）
**发表**：NeurIPS 2022（arXiv:2208.07339）
**链接**：https://arxiv.org/abs/2208.07339
**代码**：https://github.com/TimDettmers/bitsandbytes

---

### 定位

LLM.int8() 的核心贡献是发现并解决了大规模 LLM（>6.7B）中的 **Outlier 特征涌现**问题，
使得 INT8 量化在不损失精度的前提下成为可能。

---

### 核心发现：大模型中的涌现 Outlier 特征

```
特定隐层维度上，激活值系统性地远大于其他维度：
  维度  57: 62.3, -58.1, 71.2, ...（Outlier！量级 100× 以上）
  其他维度：0.x 级别

特点：
  1. 只有少数维度（~0.1%，约 6 个维度）是 Outlier
  2. 同一维度在所有 token 上都是 Outlier（系统性）
  3. 是一个相变（Phase Transition）现象，>6.7B 参数后突然涌现
```

---

### 核心方法：混合精度分解

```
把矩阵乘法按维度拆分：
  C = X[:, O] × W[O, :]^T  （Outlier 维度，~0.1%，FP16 计算）
    + X[:, R] × W[R, :]^T  （Regular 维度，~99.9%，INT8 计算）

结果：显存减少 50%，精度损失 <1%
OPT-175B FP16：350 GB → INT8：176 GB（可从 8 卡降至 3 卡）
```

---

### 核心主线

> LLM.int8() 的核心贡献是**发现大模型激活值存在系统性 Outlier 涌现现象**，
> 并提出**混合精度分解**解决：Outlier 维度（~0.1%）用 FP16，其余（~99.9%）用 INT8。
> 结果：显存减少 50%，精度损失 <1%，让单卡运行 175B 模型成为可能。

---

## 9. GPTQ

**论文**：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
**作者**：Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh（ETH Zurich & IST Austria）
**发表**：ICLR 2023（arXiv:2210.17323）
**链接**：https://arxiv.org/abs/2210.17323

---

### 定位：INT4 权重量化的标准方案

```
LLM.int8()：权重 + 激活值都量化为 INT8，2× 压缩
GPTQ：      只量化权重为 INT4（激活值保持 FP16），4× 压缩，3~4.5× 加速
```

GPTQ 是目前最广泛使用的 LLM 量化方案之一，TheBloke 等社区发布的大量量化模型都基于 GPTQ。

---

### 理论基础：OBS 框架

```
量化一个权重 w_q 后，对其他未量化权重做补偿更新：
  δW_F = -(w_q - quant(w_q)) / [H⁻¹]_{qq} × H⁻¹_{:,q}

含义：量化 w_q 产生误差后，通过调整其他权重来最小化对输出的影响
```

---

### 三个关键优化

**优化一：任意顺序量化**
大模型高度过参数化，贪心顺序 vs 固定顺序效果差异极小，复杂度从 O(d_row × d_col³) 降到 O(d_row × d_col²)。

**优化二：懒惰批量更新（B=128）**
块内立刻更新，块外延迟一次性更新，内存 I/O 减少 128 倍。

**优化三：Cholesky 分解（数值稳定）**
预计算 H 的 Cholesky 分解，避免反复更新 H⁻¹ 导致的浮点误差累积，175B 参数不崩溃。

---

### 性能结果

```
OPT-175B：4.2 小时量化，PPL 仅从 8.34 升至 8.69
GEMM 吞吐加速：A100 上 3.25×，A6000 上 4.5×
```

---

### 核心主线

> GPTQ 通过三个关键优化把 OBS 扩展到 175B 参数 LLM：
> 任意顺序（降低计算复杂度）+ 懒惰批量更新（减少内存 I/O）+ Cholesky（数值稳定）。
> 结果：INT4 量化，精度损失 <1%，推理速度提升 3~4.5×，单卡可运行 175B 模型。

---

## 10. AWQ

**论文**：AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration
**作者**：Ji Lin, Jiaming Tang, Haotian Tang et al.（MIT Han Lab，Song Han 组）
**发表**：MLSys 2024（arXiv:2306.00978）
**链接**：https://arxiv.org/abs/2306.00978

---

### 定位：比 GPTQ 更快、更硬件友好的 INT4 量化

```
GPTQ：二阶 Hessian 信息，量化慢（175B 需 4+ 小时），强精度补偿
AWQ： 激活统计量，量化极快（几分钟），全 INT4 硬件友好，泛化性更好
```

---

### 核心发现：权重重要性由激活决定

```
线性层输出：Y = W × X

某个输入通道 i 的激活值大（X_i 大）：
  → W[:, i] 的量化误差 × X_i 被放大
  → 这些权重对量化误差极其敏感

→ 重要的权重不是权重值大的，而是对应输入激活值大的
```

---

### 核心技术：Per-Channel Scaling

```
原始：Y = W × X
等价变换：Y = (W × diag(s)) × (diag(s)⁻¹ × X) = W' × X'

效果：
  s_i 增大 → 重要通道权重的相对精度提高
  diag(s)⁻¹ 吸收到前一层（LayerNorm）→ 运行时零额外计算

搜索：s_i = mean(|X_i|)^α，α ∈ [0, 1]，网格搜索，无需反向传播
```

---

### 性能结果

| 指标 | GPTQ | AWQ |
|------|------|-----|
| 量化速度 | ~20 分钟（7B） | **~几分钟** |
| INT4 精度 | 5.85 PPL（LLaMA-7B） | **5.78 PPL** |
| 泛化性 | 指令微调略下降 | **指令微调/多模态有效** |
| 推理加速 | 3~4.5× | **3.2×（RTX 4090）** |

---

### 核心主线

> AWQ 的核心洞察：**权重重要性由对应输入激活幅值决定**。
> 解法是 Per-Channel Scaling：对重要通道权重乘以 scale > 1（提升精度），
> 对应激活除以同一 scale（吸收到前一层），数学恒等变换，全 INT4 硬件友好。
> 量化速度快 10×，精度略优于 GPTQ，泛化到指令微调和多模态模型。
> vLLM、TGI、llama.cpp 均原生支持 AWQ。

---

## 11. Speculative Decoding

**论文**：Fast Inference from Transformers via Speculative Decoding
**作者**：Yaniv Leviathan, Matan Kalman, Yossi Matias（Google）
**发表**：ICML 2023（arXiv:2211.17192）
**链接**：https://arxiv.org/abs/2211.17192

---

### 定位

Speculative Decoding 是目前 LLM 推理加速最重要的技术之一，
核心思想是用一个**小模型猜，大模型验**，
在**不改变输出分布**的前提下，每次大模型调用可以生成多个 token。

---

### 为什么自回归解码慢？

```
大模型解码时：
  每步只生成 1 个 token，但需要加载整个模型权重
  是 Memory-Bandwidth Bound，GPU 算力远未被利用

关键推论：
  1 个 token 的计算时间 ≈ γ 个 token 的计算时间
  （瓶颈是内存带宽，不是计算量）
  → 如果能一次 forward 产出 γ 个 token，吞吐量提升 γ 倍
```

---

### 核心算法：三阶段流程

```
Phase 1：小模型 M_q 自回归生成 γ 个候选 token

Phase 2：大模型 M_p 并行验证
  输入：原始前缀 + γ 个 draft token
  一次 forward → 同时得到 γ+1 个位置的概率分布

Phase 3：修正拒绝采样（保证分布完全正确）
  对每个 draft token：
    r ~ Uniform(0,1)
    if r < min(1, p(x̃)/q(x̃))：接受，继续
    else：拒绝，从修正分布 norm(max(0, p-q)) 采样，停止

  全部接受：额外采样 1 个 → 本轮共产出 γ+1 个 token！
```

**数学保证**：每个 token 被采纳的概率恰好等于 p(x)，输出分布与直接使用大模型完全一致（精确，非近似）。

---

### 性能结果

```
T5-XXL（11B）为目标模型，T5-Small（77M）为 draft 模型：
  英德翻译：2.6× 加速（温度=1），3.4× 加速（greedy）
  文本摘要：2.3× 加速（温度=1），3.1× 加速（greedy）
```

---

### 核心主线

> Speculative Decoding 的洞察是：**自回归解码是 memory-bandwidth bound，
> 一次 forward 生成 1 个 token 和生成 γ 个 token 耗时相近。**
> 用小模型猜 γ 个 token，大模型并行验证，通过修正拒绝采样保证输出分布完全一致。
> 期望加速 2~3×，输出质量与直接使用大模型完全相同（非近似）。
> 已集成到 vLLM、TGI 等主流推理框架。

---

# 第二部分：学术界研究热门

---

## 12. Speculative Decoding（Seq2seq）

**论文**：Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation
**作者**：Heming Xia, Tao Ge, Si-Qing Chen, Furu Wei, Wei Wang, Zhifang Sui（微软研究院 + 北京大学）
**发表**：EMNLP 2023 Findings（arXiv:2203.16487，首版 2022.03）
**链接**：https://arxiv.org/abs/2203.16487

---

### 定位：与 Leviathan et al. 并行独立提出的投机解码

两篇论文几乎同时独立提出了投机解码的思想，但在关键维度上有所不同：

| 维度 | Leviathan et al.（第11篇） | Xia et al.（本篇）|
|------|---------------------|-----------------|
| 目标架构 | Decoder-only（GPT 类） | Encoder-Decoder（BART 类）|
| Draft 方式 | 小模型自回归（AR） | 专用非自回归（NAR）|
| 验证策略 | 修正拒绝采样（严格） | Top-β 接受（宽松）|
| 分布保证 | 完全相同（精确） | 略有差异（近似）|
| 实测加速 | 2~3× | **4.6~5.5×** |

---

### Spec-Drafter：非自回归专用 Draft 模型

```
架构：完整 Encoder（共享目标模型权重）+ 浅层 NAR Decoder
  → 编码只需做一次，浅层 Decoder 速度极快

训练（Mask-Predict）：
  [CLS] x_1...x_n [SEP] y_1...y_t [MASK][MASK][MASK]
                                    ↓      ↓      ↓
                                  ỹ_{t+1} ỹ_{t+2} ỹ_{t+3}

k 个位置同时预测，一步 forward 完成（vs AR 需 k 步串行）
```

---

### 实验结果

```
WMT EN→DE（mBART-large 610M）：Spec-Drafter → 5.5× 加速
CNN/DailyMail 摘要（BART-large）：5.1× 加速
```

---

### 核心主线

> 本文核心是**为 Seq2seq 模型设计非自回归 Spec-Drafter**：
> 共享 Encoder（保证质量），浅层 NAR Decoder（保证速度），k 个 token 一步并行生成。
> Top-β 宽松验证换取更高接受率，综合效果 4.6~5.5×。

---

## 13. BigBird

**论文**：Big Bird: Transformers for Longer Sequences
**作者**：Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey et al.（Google Research）
**发表**：NeurIPS 2020（arXiv:2007.14062）
**链接**：https://arxiv.org/abs/2007.14062

---

### 定位：稀疏 Attention 支持长序列的奠基工作

```
标准 Transformer：Attention O(n²)，BERT 最大 512 token
BigBird：降到 O(n)，支持 4096~16384 token

对比 FlashAttention：
  FlashAttention：精确 O(n²) Attention，IO 优化加速
  BigBird：近似 O(n) Attention，改变计算结构
  两者思路完全不同
```

---

### 三种稀疏 Attention 的组合

```
组件一：局部窗口 Attention（w=3，两侧各3个邻居）
  → 捕捉短程语法依赖，复杂度 O(n)

组件二：随机 Attention（r=3，随机连接全局位置）
  → 基于图论：随机图任意两点路径长 O(log n)，保证全局信息流动

组件三：全局 Token（g=1，CLS token attend 所有位置）
  → 作为信息聚合中枢，收集全序列信息再广播

三者缺一不可：局部→近邻依赖，随机→全局流动，全局→信息汇聚
```

---

### 理论保证

```
定理一：BigBird 是序列到序列函数的通用近似器（Universal Approximator）
定理二：BigBird 是图灵完备的

结论：稀疏 Attention 理论上具备完整 Transformer 的计算能力
```

---

### 实验结果

| 数据集 | 先前 SOTA | BigBird | 提升 |
|--------|----------|---------|------|
| Natural Questions | 60.7 F1 | **72.8 F1** | +12.1 |
| arXiv 摘要（ROUGE-2） | 19.77 | **21.90** | +2.13 |
| DNA 启动子预测 | 95.6% | **99.9%** | +4.3% |

---

### 核心主线

> BigBird 把 O(n²) Attention 替换为三种稀疏模式：
> **局部窗口**（近邻依赖）+ **随机 Attention**（全局信息流动）+ **全局 Token**（信息聚合）。
> 复杂度降到 O(n)，理论上保持通用近似和图灵完备性，Block Sparse 实现 GPU 高效运行。

---

## 14. Early Exit / PABEE

**论文**：BERT Loses Patience: Fast and Robust Inference with Early Exit
**作者**：Wangchunshu Zhou, Canwen Xu, Tao Ge et al.（微软研究院 + UC San Diego）
**发表**：NeurIPS 2020（arXiv:2006.04152）
**链接**：https://arxiv.org/abs/2006.04152

---

### 定位：用样本难度差异加速推理

```
核心洞察：不同样本难度不同
  简单样本：浅层就能做出准确预测，不需要跑完所有层
  难的样本：才需要跑到深层
  → 按需分配计算，平均节省大量计算
```

---

### 核心发现："过度思考"问题

```
实验发现：深层有时比浅层更差（Overthinking）
层数：  1   2  ...  7   8   9  10  11  12
准确率: 75  82 ...  91  90  92  93  92  93
                        ↑
               某些样本在第7层正确，第8层却错了

→ 提前退出既能加速，有时还能提升准确率
```

---

### PABEE：Patience-Based Early Exit

```
每层插入轻量级分类器 Ci
退出判据：连续 p 层预测相同 → 退出（"模型已想清楚了"）

优于阈值方法：
  - 更直观（层数而非置信度概率）
  - 更鲁棒（softmax overconfident 问题被规避）
```

---

### 实验结果

| 任务 | 全层准确率 | PABEE 准确率 | 加速比 |
|------|----------|------------|-------|
| SST-2 | 93.2% | 93.0% | **2.0×** |
| MNLI | 84.5% | 84.3% | **1.7×** |
| QQP | 91.1% | 91.0% | **2.2×** |

---

### 核心主线

> PABEE：在每层后插入分类器，联合训练；推理时连续 p 层预测稳定则退出。
> 比阈值方法更稳定，比知识蒸馏更灵活（动态分配而非静态压缩）。
> GLUE 任务 1.7~2.2× 加速，准确率损失 <0.5%。

---

## 15. LayerSkip

**论文**：LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding
**作者**：Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich et al.（Meta AI / FAIR）
**发表**：ACL 2024（arXiv:2404.16710）
**链接**：https://arxiv.org/abs/2404.16710

---

### 定位：把 Early Exit 和投机解码统一到生成式 LLM

```
PABEE（14）：BERT 分类，无法用于生成任务
Spec Decoding（11）：需要独立 draft 模型，额外显存

LayerSkip：
  1. 让生成式 LLM 能 Early Exit（专门训练）
  2. 用模型自身浅层做 draft（零额外参数）
```

---

### 训练方案

**Layer Dropout（递增 Dropout 率）**：
```
越深的层 dropout 率越高（0% → 50%）
→ 深层经常被跳过 → 浅层被迫学会独立推理
```

**Early Exit Loss（共享 LM Head）**：
```
所有层共享同一个 LM Head，都参与 loss 计算
→ 浅层特征空间逐渐对齐到 LM head 期望输入
→ 零额外参数（复用同一 LM head）
```

---

### Self-Speculative Decoding（核心贡献）

```
Draft Phase：只运行层 1~E，生成 γ 个 draft tokens
Verify Phase：只运行层 E+1~L，复用层 1~E 的 KV cache！

关键优势：验证时前 E 层不需要重新计算
当 E=8, L=32, γ=4：
  标准：128 次层计算
  LayerSkip：32 + 24 = 56 次 → ~2.3× 理论加速
```

---

### 实验结果

| 任务 | 加速比（Llama-2 7B） |
|------|-------------------|
| 摘要（CNN/DM） | **1.86×** |
| 代码生成（HumanEval） | **1.82×** |
| 语义解析（TOPv2） | **2.0×** |

---

### 核心主线

> LayerSkip 打通训练和推理两端：
> **训练**：递增 Layer Dropout（强迫浅层独立推理）+ 共享 LM Head 的 Early Exit Loss；
> **推理**：Early Exit 或 Self-Speculative Decoding（浅层 draft，深层 verify，复用 KV cache）。
> 无需额外模型权重，Llama-2 上实现 1.8~2.0× 加速。

---

## 16. Sorted LLaMA

**论文**：Sorted LLaMA: Unlocking the Potential of Intermediate Layers of Large Language Models for Dynamic Inference
**作者**：Parsa Kavehzadeh, Mojtaba Valipour et al.（华为诺亚方舟实验室 + 滑铁卢大学）
**发表**：arXiv:2309.08968（2023）
**链接**：https://arxiv.org/abs/2309.08968

---

### 定位：把 SortedNet 训练思想引入 LLM 推理

```
动态深度推理三部曲：
  PABEE（14）：     BERT 分类，每层加分类头，Patience 退出
  LayerSkip（15）：  LLaMA 生成，专门训练出口 + 自投机解码（有额外参数）
  Sorted LLaMA：    LLaMA 生成，SortedNet 训练，零额外参数

核心差异：
  LayerSkip：每层有独立出口 LM Head（额外参数）
  Sorted LLaMA：复用原始 LM Head（零额外参数）
```

---

### 核心方法：SortedNet 训练

```
传统训练：只用完整模型计算 loss
SortedNet：同时训练多个深度子网络

每个 step，同一 batch 上：
  子网络 8：  [L1~L8]  → LM Head → Loss_8
  子网络 16： [L1~L16] → LM Head → Loss_16
  子网络 24： [L1~L24] → LM Head → Loss_24
  完整模型：  [L1~L32] → LM Head → Loss_32

  总 Loss = Σ wi × Loss_i

关键：所有子网络共享同一套参数和同一个 LM Head
训练代价：原始预训练的 1~2%，零额外参数
```

---

### 实验结果

| 出口层 | 原始 LLaMA（直接截断） | Sorted LLaMA | 计算节省 |
|--------|---------------------|-------------|---------|
| 8 层 | ~120 PPL（几乎不可用） | **~18 PPL** | 75% |
| 16 层 | ~35 PPL | **~9.5 PPL** | 50% |
| 24 层 | ~8.5 PPL | **~7.5 PPL** | 25% |
| 32 层 | 7.0（基准） | 7.0（相同） | 0% |

---

### 核心主线

> Sorted LLaMA：通过 SortedNet 训练使单一 LM Head 在任意深度都能产生合理输出，
> 零额外参数，单模型支持 8/16/24/32 层灵活切换，或按 token 动态退出。
> 16 层时节省 50% 计算，PPL 仅轻微上升。

---

## 17. KVQuant

**论文**：KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
**作者**：Coleman Hooper, Sehoon Kim, Hasan Genc et al.（UC Berkeley / SqueezeAI Lab）
**发表**：NeurIPS 2024（arXiv:2401.18079）
**链接**：https://arxiv.org/abs/2401.18079

---

### 定位：专注 KV Cache 量化的系统性工作

```
权重量化（GPTQ/AWQ）：压缩模型参数（静态，一次性）
KV Cache 量化（本篇）：压缩推理时动态生成的 KV 激活值（动态，在线）

KV cache 是长上下文推理的主要显存瓶颈：
  LLaMA-2 70B，seq=1M：KV cache ≈ 2.6 TB → 完全不可能
  3-bit 量化后：显存节省 5×，支持 8 卡 10M token 上下文
```

---

### 核心发现：Key 和 Value 的分布特性不同

```
Key Cache：
  特定 channel 始终有 outlier（channel-wise 稳定）
  → 适合 Per-Channel Quantization

Value Cache：
  outlier 分散在不同 token 的不同位置（token-wise 不稳定）
  → 适合 Per-Token Quantization
```

---

### 五大技术

1. **非对称量化策略**：Key 用 Per-Channel，Value 用 Per-Token
2. **Pre-RoPE Key 量化**：RoPE 旋转前量化，保持 channel 统计稳定性
3. **非均匀量化**：k-means 码本，适应非均匀分布，误差降低 0.33 PPL
4. **稠密+稀疏混合**：99% 用 3-bit，1% outlier 用 FP16，精度大幅提升
5. **Q-Norm**：2-bit 时调整码本对齐均值/方差，无运行时开销

---

### 精度结果

| 方法 | bits | Perplexity | vs FP16 |
|------|------|-----------|---------|
| 均匀量化（per-tensor） | 3 | 8.32 | +2.85 |
| **KVQuant（全部技术）** | **3** | **5.54** | **+0.07** |

3-bit 量化仅带来 0.07 的 perplexity 损失，1.4× attention 加速。

---

### 核心主线

> KVQuant 的核心：**Key 和 Value 的 outlier 结构不同，必须分别对待**（Per-Channel vs Per-Token）。
> 叠加 Pre-RoPE、非均匀码本、稀疏存储、Q-Norm 四项优化，
> 3-bit 量化 PPL 损失 <0.1，显存节省 5×，8 卡支持 **10M token 上下文**。

---

## 18. SparseGPT

**论文**：SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot
**作者**：Elias Frantar, Dan Alistarh（IST Austria）
**发表**：ICML 2023（arXiv:2301.00774）
**链接**：https://arxiv.org/abs/2301.00774

---

### 定位：LLM 剪枝的标准方案

```
量化（GPTQ/AWQ）：保留所有权重，降低精度
剪枝（本篇）：    直接置零权重，保留精度

两者出自同一团队（IST Austria），共享 OBS 理论框架，可以叠加：
  50% 稀疏 + INT4 量化 → ~8× 综合压缩比
```

---

### 为什么 LLM 剪枝很难

```
朴素幅度剪枝（OPT-175B，50% 稀疏）：
  PPL：8.34 → 14+（灾难性退化）

SparseGPT（50% 稀疏）：
  PPL：8.34 → 8.71（几乎无损）

原因：幅度小 ≠ 不重要，必须用二阶信息做补偿
```

---

### 核心算法：懒惰块更新

```
每列分组（B=128），块内立刻更新，块间延迟更新：
  复杂度：从 O(d_row × d_col³) 降到 O(d_row × d_col²)
  OPT-175B：从不可行 → 单卡 A100 上 4.5 小时完成
  Cholesky 分解保证数值稳定
```

---

### 支持的稀疏模式

```
非结构化：任意位置置零，精度最好，GPU 无法加速
2:4 半结构化：每4个权重恰好2个非零，A100+ GPU ~2× 加速
联合量化：50% 稀疏 + INT4 量化，~8× 压缩比
```

---

### 实验结果

| 模型 | Dense | 朴素幅度（50%） | SparseGPT（50%） |
|------|-------|--------------|----------------|
| OPT-175B | 8.34 | 14.43 | **8.71** |
| OPT-66B | 9.34 | 11.22 | **9.72** |

---

### 核心主线

> SparseGPT 把 OBS 扩展到 175B 参数 LLM 剪枝。
> 懒惰块更新（B=128）+ Cholesky 分解，把不可行的复杂度降到 4.5 小时可完成。
> OPT-175B 50% 稀疏，PPL 仅升 0.37，而朴素幅度剪枝升 6+。

---

## 19. Distilling Step-by-Step

**论文**：Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes
**作者**：Cheng-Yu Hsieh, Chun-Liang Li et al.（Google Research）
**发表**：ACL 2023 Findings（arXiv:2212.09561）
**链接**：https://arxiv.org/abs/2212.09561

---

### 核心思想

将 LLM 链式思维（Chain-of-Thought）生成的**推理过程（rationale）**作为额外监督信号，
与答案标签一起训练小模型。

让小模型不仅学习"**答案是什么**"，还学习"**为什么是这个答案**"。

---

### 两阶段流程

**阶段一：Rationale 提取**

用带 CoT 提示的大型 LLM 对每个训练样本生成推理过程，全程无需人工标注。

**阶段二：多任务学习训练小模型**

| 任务 | 输入 | 输出 |
|------|------|------|
| 标签预测 | 问题/文本 | 正确答案 |
| 推理生成 | 问题/文本 | LLM 生成的 rationale |

---

### 实验结果

```
770M T5（Distilling Step-by-Step，用 80% 数据）
  超越 540B PaLM 的少样本性能
  参数量缩小 ~700 倍，数据量减少 20%
```

---

### 核心主线

> 将"推理过程"作为监督信号，比仅使用"答案标签"包含更多信息，
> 大幅提升小模型的数据效率和泛化能力。
> 在知识蒸馏与链式思维推理的交叉点上开辟了新方向。

---

## 20. FlexGen

**论文**：FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU
**作者**：Ying Sheng, Lianmin Zheng et al.（UC Berkeley、Stanford、ETH Zurich 等）
**发表**：ICML 2023（arXiv:2303.06865）
**链接**：https://arxiv.org/abs/2303.06865

---

### 核心问题

能否在**单张普通 GPU**（如 16GB T4）上，以**可用的吞吐量**运行 175B 参数量的模型？

---

### 方法：三级存储层次 + 线性规划

**三级内存层次**：GPU 显存（40~80GB） → CPU 内存（1~2TB） → NVMe SSD（10~100TB）

**线性规划（LP）搜索最优卸载策略**：
```
目标：最大化生成吞吐量
约束：GPU/CPU/磁盘容量，PCIe/NVMe 带宽，GPU 算力

LP 自动输出每层最优的张量放置比例（权重/激活/KV cache 各放多少在哪里）
```

**锯齿形块调度（Zig-Zag）**：
```
GPU 计算第 i 层
  同时：预加载第 i+1 层的权重和 KV cache（从 CPU/磁盘）
  同时：回写第 i-1 层的结果
→ I/O 与计算重叠，消除等待
```

**压缩技术**：分组 4-bit 量化（权重 + KV cache）+ KV 稀疏化（保留 top-20%）

---

### 实验结果（单张 T4 16GB）

| 系统 | OPT-175B 吞吐 |
|------|-------------|
| HuggingFace Accelerate | OOM |
| DeepSpeed ZeRO-Inference | <0.3 tokens/s |
| **FlexGen（无压缩）** | **~25 tokens/s** |
| **FlexGen（4-bit + KV 稀疏）** | **~69 tokens/s** |

---

### 核心主线

> 将 LLM 推理的内存管理抽象为三级存储层次上的张量放置优化问题，
> 配合 I/O-计算流水线和压缩技术，在单张普通 GPU 上实现原本需要多 GPU 集群才能完成的大模型推理。
> 主要价值：降低研究门槛，让普通研究者用单 GPU 跑完整大模型评测。
> 注：面向批处理吞吐设计，单请求延迟高，不适合交互式场景。

---

*全文完*
