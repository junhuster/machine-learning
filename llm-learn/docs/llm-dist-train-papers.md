# LLM 分布式训练论文导读合集

> 本合集涵盖 LLM 分布式训练的核心论文，按照推荐阅读顺序排列。
> 整体脉络：数据并行基础（DDP）→ 显存优化（ZeRO 系列）→ 模型并行（Megatron TP）→ 流水线并行（GPipe → PipeDream → Megatron 3D）→ 激活值优化（Selective Recomputation）

---

## 目录

1. [PyTorch DDP](#1-pytorch-ddp)
2. [ZeRO](#2-zero)
3. [ZeRO-Offload](#3-zero-offload)
4. [ZeRO-Infinity](#4-zero-infinity)
5. [Megatron-LM（张量并行）](#5-megatron-lm张量并行)
6. [GPipe](#6-gpipe)
7. [PipeDream](#7-pipedream)
8. [Megatron-LM v2（3D 并行）](#8-megatron-lm-v23d-并行)
9. [Reducing Activation Recomputation](#9-reducing-activation-recomputation)

---

## 1. PyTorch DDP

**论文**：PyTorch Distributed: Experiences on Accelerating Data Parallel Training
**作者**：Shen Li et al.（Meta/Facebook AI Research）
**发表**：VLDB 2020
**链接**：https://arxiv.org/abs/2006.15704

---

### 背景：为什么要重新设计

PyTorch 早期有 `DataParallel`（DP），但它是单进程多线程，有 GIL 瓶颈，主卡负载不均衡。
这篇论文描述的是重新设计的 `DistributedDataParallel`（DDP）——多进程方案，每个进程独立持有一份模型副本，是现在 PyTorch 分布式训练的标准用法。

---

### 读论文前需要理解的基础概念

**数据并行的基本逻辑**：
```
每张卡：持有完整模型副本
每个 step：
  1. 各卡拿不同的 mini-batch 做 forward，得到各自的梯度
  2. 所有卡的梯度做 AllReduce（求平均）
  3. 每张卡用相同的平均梯度更新参数
  → 所有卡参数始终保持一致
```

**AllReduce** 是核心通信原语：把所有进程的张量加和/平均，结果广播回所有进程。底层用 NCCL（GPU 场景）实现。

---

### 论文的三个核心贡献

#### 1. Gradient Bucketing（梯度分桶）

**问题**：模型有很多层，每层 backward 完产生一个梯度张量，如果每个张量单独发一次 AllReduce，网络请求太碎，延迟高。

**解法**：把多个梯度张量打包成一个 bucket，凑满一个 bucket 再发一次 AllReduce。

```
layer_100 梯度 ─┐
layer_99  梯度 ─┼→ bucket_1（25MB）→ 一次 AllReduce
layer_98  梯度 ─┘

layer_97  梯度 ─┐
layer_96  梯度 ─┼→ bucket_2（25MB）→ 一次 AllReduce
...
```

**关键参数**：`bucket_cap_mb`（默认 25MB），论文里有实验说明这个值的选取对性能影响。

---

#### 2. Computation-Communication Overlap（计算通信重叠）

**问题**：朴素实现是 backward 全部完成后再做 AllReduce，GPU 在通信期间空闲。

**解法**：利用 backward 从后往前逐层计算的顺序——**后面的层梯度算完就立刻发 AllReduce，同时前面的层继续 backward**。

```
时间轴：
朴素方案：[backward 全部完成] → [AllReduce] → 下一个 step
DDP方案： [backward layer N] → [AllReduce bucket_1 开始]
                [backward layer N-1...]
                                [AllReduce bucket_1 完成]
          [backward 全部完成]  [AllReduce bucket_2 完成]
          → 下一个 step（几乎不等通信）
```

实现方式：给每个参数注册 **autograd hook**，梯度一产生就触发 bucket 的累积逻辑，bucket 满了自动发 AllReduce。

---

#### 3. Gradient Synchronization Skipping（梯度同步跳过）

**问题**：梯度累积（gradient accumulation）场景下，用户跑多个 micro-batch 才做一次 optimizer.step()，中间的 AllReduce 是浪费的。

**解法**：`no_sync()` 上下文管理器，在累积阶段跳过 AllReduce，只在最后一个 micro-batch 才同步。

```python
for i, batch in enumerate(dataloader):
    if i % accumulation_steps != 0:
        with model.no_sync():       # 跳过 AllReduce
            loss = model(batch)
            loss.backward()
    else:
        loss = model(batch)
        loss.backward()             # 这里才真正 AllReduce
        optimizer.step()
```

---

### 系统设计细节（读论文时不要跳过）

**参数分桶的顺序问题**：
- DDP 按 backward 顺序（从后往前）分配 bucket
- 有些模型参数在 forward 里不一定被用到（unused parameters），论文讨论了怎么处理

**初始化同步**：
- 所有进程启动时，用 broadcast 把 rank 0 的参数广播给所有进程，保证初始状态一致

**与 optimizer 的边界**：
- DDP 只负责梯度同步，不管 optimizer
- optimizer 在各进程独立运行，但因为梯度已经同步，结果一致

---

### 论文结构速览

```
Section 1: Introduction        ← 为什么 DP 不够用，DDP 要解决什么
Section 2: Background          ← AllReduce、NCCL 基础，先搞懂这里
Section 3: System Design       ← 三大核心贡献的设计思路（重点）
Section 4: Implementation      ← autograd hook、bucket 管理的实现细节
Section 5: Evaluation          ← 各优化的性能收益实验（选读）
Section 6: Discussion          ← 局限性和未来方向
```

---

### 核心主线

> 论文的核心就一件事：**如何让 AllReduce 通信尽量不阻塞 backward 计算**。
> Gradient Bucketing 减少通信次数，Overlap 让通信和计算并行，no_sync 避免无效通信——三个优化都服务于这一个目标。

带着这个主线去读，每个设计决策都会很清晰。

---

### 性能结论

论文实验表明：配置合理时，DDP 在 **256 GPU** 上达到接近线性的扩展性（near-linear scalability）。

---

## 2. ZeRO

**论文**：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
**作者**：Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He（Microsoft Research）
**发表**：SC'20（超算顶会）
**链接**：https://arxiv.org/abs/1910.02054
**实现**：DeepSpeed 库

---

### 背景：训练大模型的显存墙

混合精度训练下，一个参数占用的显存分布：

```
参数本身（FP16）：       2 bytes
梯度（FP16）：           2 bytes
优化器状态（FP32）：
  - FP32 参数备份：      4 bytes
  - Adam momentum：      4 bytes
  - Adam variance：      4 bytes
                        ──────────
合计：                  16 bytes / 参数
```

**例子**：GPT-3（175B 参数）× 16 bytes = **2.8 TB** 显存

单卡装不下，传统数据并行（DDP）每张卡存完整副本也装不下。

---

### 传统数据并行的问题

```
DDP 的做法：
  每张卡：完整模型参数 + 完整梯度 + 完整优化器状态
  N 张卡：N 份完全一样的副本

显存利用率极低——N 张卡存了 N 份相同的东西，纯浪费
```

ZeRO 的核心思想：**把这 N 份冗余数据切成 N 份，每张卡只存 1/N**。

---

### ZeRO 三个阶段（核心创新）

#### Stage 1：优化器状态分片（Pos）

```
优化器状态（12 bytes/参数）按卡分片：
  卡0 只存参数 0 ~ Ψ/N 的优化器状态
  卡1 只存参数 Ψ/N ~ 2Ψ/N 的优化器状态
  ...

参数和梯度各卡仍然保留完整副本

显存节省：优化器状态从 12 → 12/N bytes/参数
整体约 4× 节省
```

通信量：**和 DDP 完全相同**（2Ψ AllReduce），零额外开销。

---

#### Stage 2：梯度分片（Pos+g）

```
在 Stage 1 基础上，梯度也分片：
  卡 i 只保留自己负责更新的那部分参数的梯度

backward 过程：
  各卡正常计算梯度
  Reduce-Scatter 替代 All-Reduce：
    每张卡最终只保留属于自己的那 1/N 梯度
  用本地梯度更新本地优化器状态

显存节省：梯度从 2 → 2/N bytes/参数
整体约 8× 节省
```

通信量：**和 DDP 完全相同**，用 Reduce-Scatter 替代 All-Reduce，总量不变。

---

#### Stage 3：参数分片（Pos+g+p）

```
连模型参数本身也分片：
  每张卡只常驻 1/N 的参数

Forward 时：
  需要某层参数 → All-Gather 临时聚合 → 计算 → 用完丢弃

Backward 时：
  同样 All-Gather 取参数 → 计算梯度 → 丢弃参数
  Reduce-Scatter 同步梯度

显存节省：参数从 2 → 2/N bytes/参数
整体：16/N bytes/参数，与 N 线性相关
```

通信量：**1.5× DDP**（多了 forward 的 All-Gather），这是唯一的代价。

---

### 三阶段对比总结

| 阶段 | 分片内容 | 显存节省 | 额外通信开销 |
|------|---------|---------|------------|
| **ZeRO-1** | 优化器状态 | ~4× | 无 |
| **ZeRO-2** | + 梯度 | ~8× | 无 |
| **ZeRO-3** | + 模型参数 | ~N× | 1.5× |

ZeRO-1/2 是**免费的午餐**：不增加任何通信，直接省显存。

---

### ZeRO-R：残余显存优化

除了模型状态，还有其他显存消耗：

**激活值（Activation）**：
- 传统做法：模型并行时每张卡都复制一份激活值
- ZeRO-R：激活值也按模型并行维度分片，用完再 All-Gather

**临时缓冲区（Temporary Buffers）**：
- 固定大小缓冲区，防止 AllReduce 时动态分配大块内存导致碎片

**显存碎片整理（Memory Defragmentation）**：
- 提前为激活值 checkpoint 和梯度预分配连续内存

---

### 通信原语的变化

理解 ZeRO 需要区分两个操作：

```
All-Reduce = Reduce-Scatter + All-Gather

DDP：      All-Reduce（每张卡最终有完整梯度）
ZeRO-2：   Reduce-Scatter（每张卡只有 1/N 梯度，够用了）
ZeRO-3：   Reduce-Scatter（梯度） + All-Gather（参数，按需取）
```

---

### 实验结论

- **ZeRO-2**：在 400 张 V100 上训练 **170B 参数**模型，吞吐 38+ TFLOPs/GPU
- **ZeRO-3**：理论上 1024 张 GPU 可训练 **1 万亿参数**模型
- ZeRO-2 相比 baseline 显存节省 8×，**通信量不变**

---

### 论文结构速览

```
Section 1: Introduction       ← 显存墙问题，ZeRO 的目标
Section 2: Background         ← 混合精度训练显存分布，模型并行局限
Section 3: ZeRO-DP            ← 三阶段核心设计（重点）
Section 4: ZeRO-R             ← 残余显存优化
Section 5: Communication      ← 通信量分析，证明开销可接受
Section 6: Evaluation         ← 实验结果
Section 7: Discussion         ← 与模型并行的对比
```

---

### 核心主线

> ZeRO 的本质是：**数据并行中 N 张卡存了 N 份冗余，把冗余切掉，每张卡只存 1/N**。
> Stage 1/2 切优化器状态和梯度，通信量不增加，是免费的优化。
> Stage 3 连参数也切，代价是 forward/backward 时按需 All-Gather，通信量增加 1.5×。

---

### 后续工作

| 论文 | 创新点 |
|------|--------|
| **ZeRO-Offload**（2021） | 优化器状态和梯度 offload 到 CPU 内存 |
| **ZeRO-Infinity**（2021） | 进一步 offload 到 NVMe SSD，突破 CPU 内存限制 |
| **ZeRO++** | 用量化压缩 All-Gather 的通信量，减少 Stage 3 的 1.5× 开销 |

---

## 3. ZeRO-Offload

**论文**：ZeRO-Offload: Democratizing Billion-Scale Model Training
**作者**：Jie Ren, Samyam Rajbhandari et al.（Microsoft Research）
**发表**：USENIX ATC 2021
**链接**：https://arxiv.org/abs/2101.06840
**实现**：DeepSpeed 库

---

### 背景：大模型训练的门槛问题

ZeRO 原论文解决了多卡场景下的显存问题，但要求有足够多的 GPU。

```
训练 10B 参数模型：
  原始需求：~16 张 V100（约 $10 万硬件成本）
  ZeRO-Offload 目标：1 张 GPU 搞定

"Democratizing"（民主化）的含义：
  让普通研究者用单卡也能训练十亿级参数模型
```

核心思路：**把 CPU 内存当成 GPU 显存的扩展**，把部分数据和计算卸载到 CPU。

---

### 关键问题：卸载什么？不卸载什么？

不能什么都卸载到 CPU——forward/backward 的矩阵乘法卸到 CPU 会慢几十倍。
论文通过三条原则推导出**唯一最优的卸载策略**：

**原则 1：CPU 计算量要尽量少**
- forward/backward 计算量是 O(M×B)（M=参数量，B=batch size），不能卸载
- 优化器更新计算量是 O(M)，和 batch size 无关，可以接受

**原则 2：通信量要最小**
- GPU↔CPU 之间的 PCIe 带宽是瓶颈（~16GB/s，远小于 NVLink）
- 卸载策略要让每次迭代的 CPU-GPU 传输量最小

**原则 3：在满足前两条的前提下，卸载尽量多的数据**

---

### 推导出的最优卸载方案

```
留在 GPU：
  FP16 参数          （forward/backward 需要频繁访问）
  forward/backward 计算

卸载到 CPU：
  FP16 梯度          （backward 算完就传到 CPU，不再需要）
  FP32 优化器状态     （momentum + variance + FP32参数备份）
  优化器更新计算      （Adam 在 CPU 上执行）
```

内存节省对比：

| 存储位置 | 内容 | bytes/参数 |
|---------|------|-----------|
| GPU 显存 | FP16 参数 | 2 |
| CPU 内存 | FP16 梯度 + FP32 优化器状态 | 2 + 12 = 14 |
| **合计GPU显存** | **原来16→现在2** | **8× 节省** |

论文证明这个方案是**唯一最优的**——没有其他分法能在满足前两条原则的同时卸载更多数据。

---

### 单卡训练流程

```
Step N 时间线：

GPU: [Forward(FP16参数)] → [Backward] → [梯度传CPU] → [Forward(N+1)]
                                              ↓
CPU:                               [Adam更新(FP32)] → [FP32→FP16] → [传回GPU]
```

**关键细节**：梯度在 backward 计算的同时就流式传输到 CPU（overlap），不是等 backward 全部完成再传，减少等待。

---

### CPU Adam 优化

CPU 执行 Adam 比 GPU 慢，论文做了专门优化：

| 优化手段 | 说明 |
|---------|------|
| **SIMD 向量化** | 用 AVX-512 指令，一条指令处理 16 个 float |
| **循环展开** | 减少循环控制开销 |
| **多线程** | 并行处理不同参数分片 |

结果：CPU Adam 比现有最优实现快 **6×**，使得 CPU 计算不成为瓶颈。

---

### Delayed Parameter Update（DPU）：进一步隐藏 CPU 计算延迟

即使 CPU Adam 优化了，CPU 更新参数仍需要时间。DPU 的思路：

```
不等 CPU 更新完再开始下一个 step，而是：

Step N:   GPU backward → 梯度传CPU
Step N+1: GPU 用旧参数跑 forward/backward，同时 CPU 在更新参数
Step N+2: GPU 用 CPU 更新好的参数（延迟了一步）
```

**代价**：用的是 N-1 步的梯度更新参数，引入一步延迟。
**实验结论**：训练初期不建议开（不稳定），稳定后开启**不影响收敛精度**。

---

### 多卡场景：与 ZeRO-2 结合

ZeRO-Offload 不只适用于单卡，多卡时与 ZeRO-2 结合：

```
ZeRO-2（多卡）：梯度和优化器状态在多卡间分片
ZeRO-Offload（单卡）：梯度和优化器状态卸载到 CPU

两者结合：
  先按 ZeRO-2 在多卡间分片
  每张卡再把自己负责的那部分卸载到 CPU

关键性质：
  N 张卡时，每张卡的 CPU-GPU 通信量 = 单卡的 1/N
  → 总 CPU-GPU 通信量不随卡数增加
  → 128 卡近线性加速
```

---

### 性能结论

| 场景 | 可训练模型规模 |
|------|--------------|
| PyTorch 单卡 V100 | 1.4B 参数 |
| **ZeRO-Offload 单卡 V100** | **13B 参数（提升 ~9×）** |
| ZeRO-Offload + 模型并行 DGX-2 | 70B 参数 |

吞吐：10B 模型在 V100 上达到 **40 TFLOPs/GPU**，接近硬件峰值。

---

### 论文结构速览

```
Section 1: Introduction      ← 民主化训练的动机
Section 2: Background        ← ZeRO-2 回顾，CPU-GPU 带宽分析
Section 3: Offload Strategy  ← 最优卸载方案的推导（重点）
Section 4: Implementation    ← CPU Adam、DPU、流水线重叠
Section 5: Multi-GPU         ← 与 ZeRO-2 结合的分析
Section 6: Evaluation        ← 单卡/多卡实验结果
```

---

### 核心主线

> ZeRO-Offload 的本质是：**把不需要频繁 GPU 算力的数据（优化器状态、梯度）和计算（Adam更新）卸到 CPU，GPU 只保留 FP16 参数专注 forward/backward**。
> 卸载方案是数学推导出的唯一最优解，不是经验选择。
> 配合 CPU Adam 优化和 DPU 流水线，CPU 计算延迟基本被隐藏。

---

### 与 ZeRO 原论文的关系

```
ZeRO（原论文）：多卡之间分片，解决多卡场景显存问题
ZeRO-Offload：  GPU→CPU 卸载，解决单卡/少卡场景显存问题
ZeRO-Infinity： 进一步卸载到 NVMe SSD，突破 CPU 内存限制（下一篇）
```

---

## 4. ZeRO-Infinity

**论文**：ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning
**作者**：Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He（Microsoft Research）
**发表**：SC'21（超算顶会）
**链接**：https://arxiv.org/abs/2104.07857
**实现**：DeepSpeed 库

---

### ZeRO 系列回顾

读这篇论文前，先明确三篇论文各自解决什么问题：

```
ZeRO（原论文）：   多卡之间分片显存，解决多卡冗余问题
ZeRO-Offload：    GPU→CPU 卸载，单卡也能训 10B+ 模型
ZeRO-Infinity：   GPU→CPU→NVMe 三层卸载，突破 CPU 内存限制，训练万亿参数
```

---

### 背景：GPU 内存墙

```
GPT-3（175B 参数）：混合精度训练需要 ~2.8 TB 显存
1T 参数模型：       需要 ~32 TB 显存

单卡 A100：         80 GB
512 张 A100：       40 TB（接近，但模型并行开销大）
```

ZeRO-Offload 把优化器状态卸到 CPU，但 CPU 内存也是有限的：

```
单机 CPU 内存：约 1~2 TB
1T 模型仅参数就需要 2 TB（FP16），还不算梯度和优化器状态
```

**ZeRO-Infinity 的答案**：把 NVMe SSD 也纳入内存体系。

---

### 三层内存体系

```
GPU HBM      40~80 GB/卡    带宽 ~2 TB/s    ← 最快，最贵，最小
CPU DRAM     1~2 TB/节点    带宽 ~100 GB/s  ← 中等
NVMe SSD     10~100 TB/节点 带宽 ~25 GB/s   ← 最慢，最大，最便宜
                            （多盘聚合可达 200 GB/s）
```

ZeRO-Infinity 让模型数据在这三层之间自动流动，GPU 只保留当前计算需要的部分。

---

### 五大核心设计

#### 1. Infinity Offload Engine（核心）

统一管理三层内存的数据搬运，对用户透明：

```
参数存在 NVMe → 需要时自动搬到 CPU → 再搬到 GPU 计算 → 计算完丢弃
梯度计算完    → 自动写回 CPU/NVMe
优化器状态    → 常驻 CPU/NVMe，更新时搬上来
```

与 ZeRO-Offload 的区别：ZeRO-Offload 只到 CPU，ZeRO-Infinity 打通到 NVMe。

---

#### 2. Memory-Centric Tiling（单层过大的问题）

**问题**：如果单个 Linear 层就有 50GB，怎么放进 80GB 的 GPU？

Megatron 的解法：张量并行，把权重矩阵切到多卡。
ZeRO-Infinity 的解法：Tiling，把大矩阵切成小块，逐块传入 GPU 计算：

```
Linear(50GB 权重) 分成 10 个 tile，每个 5GB：
  tile_0 → 传入 GPU → 计算对应输出 → 丢弃
  tile_1 → 传入 GPU → 计算对应输出 → 丢弃
  ...
  结果拼接 → 完整输出
```

**好处**：不需要改模型代码，不需要张量并行，自动处理任意大小的层。

---

#### 3. Bandwidth-Centric Partitioning（NVMe 带宽聚合）

单块 NVMe 带宽 ~25 GB/s，远低于 GPU 计算需求。

解法：把数据分散存到多块 NVMe，并行读取：

```
1 块 NVMe：25 GB/s
8 块 NVMe：~200 GB/s（接近线性扩展）
```

数据分片时按带宽最优方式分配，而不是按逻辑顺序，充分利用多盘并行 IO。

---

#### 4. Overlap-Centric Design（隐藏 IO 延迟）

NVMe 比 GPU 慢很多，如果串行等待 IO，GPU 大量空闲。

解法：**预取 + 双缓冲**：

```
时间线：
  GPU 计算 layer_N
  同时：CPU 从 NVMe 预取 layer_N+1 的参数
        GPU 从 CPU 预取 layer_N+1 的参数到 GPU

GPU 计算 layer_N+1（参数已经在 GPU 上）
  同时：预取 layer_N+2 ...
```

三层流水线同时运转，GPU 几乎不等 IO。

---

#### 5. Ease of Use（零代码修改）

与 Megatron 张量并行对比：

| 方案 | 模型代码改动 | 通信开销 | 可训练规模 |
|------|------------|---------|----------|
| 张量并行（Megatron） | 大（需重写每层） | 高 | 受 GPU 数量限制 |
| 流水线并行 | 中 | 中 | 受 GPU 数量限制 |
| **ZeRO-Infinity** | **无** | **低** | **几乎无限** |

用户只需修改 DeepSpeed 的 JSON 配置：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/local_nvme"
    },
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme"
    }
  }
}
```

---

### 训练规模对比

| 硬件配置 | 标准训练 | ZeRO-Infinity |
|---------|---------|--------------|
| 单卡 A100（80GB） | ~2B 参数 | **~1T 参数** |
| 单机 8 卡 + NVMe | ~16B 参数 | **~1T 参数** |
| 512 卡 + NVMe | ~几十B | **>100T 参数** |

---

### 性能代价

不是免费的午餐，代价是吞吐量下降：

```
纯 GPU 训练（模型能装下）：最快
ZeRO-3（纯 GPU 分片）：   略慢（AllGather 开销）
ZeRO-Offload（CPU）：      更慢（PCIe 带宽限制）
ZeRO-Infinity（NVMe）：    最慢（NVMe IO 限制）

但：ZeRO-Infinity 能训练其他方案根本无法训练的模型规模
```

论文实验：1T 参数模型在 512 张 V100 上达到 **>40% 硬件峰值利用率**，约 25 PetaFLOPs。

---

### 论文结构速览

```
Section 1: Introduction         ← GPU 内存墙问题，三层内存体系动机
Section 2: Background           ← ZeRO 系列回顾，NVMe 带宽分析
Section 3: Infinity Offload     ← 核心引擎设计（重点）
Section 4: Memory-Centric Tiling← 单层过大的解法
Section 5: Bandwidth Partitioning← NVMe 多盘聚合
Section 6: Overlap Design       ← 三层流水线
Section 7: Evaluation           ← 1T 模型训练实验
Section 8: Discussion           ← 与模型并行对比
```

---

### 核心主线

> ZeRO-Infinity 的本质是：**把内存体系从单层（GPU）扩展到三层（GPU + CPU + NVMe），用预取流水线隐藏 IO 延迟，用多盘聚合提升 NVMe 带宽，用 Tiling 解决单层过大问题**。
> 代价是训练吞吐下降，收益是几乎无限的模型规模扩展能力，且不需要修改模型代码。

---

### ZeRO 系列全景

```
ZeRO-1：  优化器状态分片（N卡，4× 显存节省，零通信开销）
ZeRO-2：  + 梯度分片（8× 显存节省，零通信开销）
ZeRO-3：  + 参数分片（N× 显存节省，1.5× 通信开销）
ZeRO-Offload：  梯度+优化器状态卸到 CPU（单卡训 13B）
ZeRO-Infinity： 全部卸到 CPU+NVMe（单机训 1T）
ZeRO++：        量化压缩 AllGather，减少 ZeRO-3 的通信开销
```

---

## 5. Megatron-LM（张量并行）

**论文**：Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
**作者**：Mohammad Shoeybi, Mostofa Patwary, Raul Puri et al.（NVIDIA Research）
**发表**：2019（arXiv:1909.08053）
**链接**：https://arxiv.org/abs/1909.08053
**代码**：https://github.com/NVIDIA/Megatron-LM

---

### 背景：为什么需要模型并行

```
DDP（数据并行）的前提：单卡能装下完整模型
模型越来越大：GPT-2（1.5B）已经接近单卡极限

Megatron 要解决：模型本身装不进单卡，怎么训练
```

ZeRO 的思路是"分片显存"，Megatron 的思路是"切开模型，每张卡算一部分"。
两者解决同一问题，方向不同：

```
ZeRO：     模型完整，参数分散存储，需要时通信取回
Megatron：  模型切开，每张卡只有一部分，各自负责各自的计算
```

---

### 核心创新：张量并行（Tensor Parallelism）

论文的核心贡献是**层内切分（intra-layer splitting）**——把单个矩阵乘法拆到多卡并行。

关键设计目标：**最小化通信次数**，每个 Transformer 层只需要 2 次 AllReduce。

---

#### MLP 块的切分

Transformer 的 FFN 由两个 Linear 组成：
```
Y = GeLU(X × A)     A: [H, 4H]
Z = Y × B           B: [4H, H]
```

**Column Parallel（第一个 Linear 按列切）**：

```
GPU_0 持有 A 的前 4H/N 列 → 算出 Y 的前 4H/N 列
GPU_1 持有 A 的后 4H/N 列 → 算出 Y 的后 4H/N 列

X 需要广播到所有卡（或各卡本来就有相同的 X）
各卡独立计算，无需通信
```

**Row Parallel（第二个 Linear 按行切）**：

```
GPU_0 持有 B 的前 4H/N 行 → 用自己的 Y 片段算出 Z 的部分和
GPU_1 持有 B 的后 4H/N 行 → 用自己的 Y 片段算出 Z 的部分和

Z = 各卡结果求和 → 一次 AllReduce
```

完整 FFN 只需要 **1 次 AllReduce**（在第二个 Linear 之后）。

---

#### Self-Attention 块的切分

多头注意力天然适合按 head 切分：

```
8 个 attention head，4 张卡：
  GPU_0：head 0, 1
  GPU_1：head 2, 3
  GPU_2：head 4, 5
  GPU_3：head 6, 7
```

具体实现：
- **Q/K/V 投影**：按列切（每张卡算自己负责的 head 的 QKV）
- **各 head 独立计算**：attention(Q, K, V) 无需跨卡通信
- **输出投影**：按行切，最后 AllReduce 合并

Self-Attention 块同样只需要 **1 次 AllReduce**。

---

#### 每层通信总量

```
一个 Transformer 层 = Self-Attention + MLP
通信次数：1（Attention）+ 1（MLP）= 2 次 AllReduce

通信量：2 × 2M bytes（前向 + 反向各一次）
其中 M = 序列长度 × batch size × hidden size
```

与 Data Parallel 的 AllReduce（同步梯度）不同：
- Megatron 的 AllReduce 在**前向/反向传播过程中**发生
- 通信量和模型大小无关，只和激活值大小有关

---

### 数据并行 + 模型并行组合

两种并行正交，可以叠加：

```
总 GPU 数 = 模型并行数(m) × 数据并行数(d)

例：512 张 GPU
  m=8（8 张卡共同计算一个模型）
  d=64（64 路数据并行）
  → 8 × 64 = 512 张 GPU

通信拓扑：
  同一模型并行组内（8卡）：NVLink 高带宽，频繁 AllReduce
  不同数据并行组间（64组）：InfiniBand，梯度同步 AllReduce
```

模型并行组内必须用高带宽互联（NVLink），这是 Megatron 对硬件的要求。

---

### 工程优化

#### Fused CUDA Kernels

把多个小算子融合成一个 kernel，减少 kernel launch 开销和显存读写：

| 融合算子 | 组合内容 |
|---------|---------|
| Fused Softmax | scale + mask + softmax |
| Fused Bias + GeLU | bias add + GeLU 激活 |
| Fused Bias + Dropout + Residual | 三个操作合一 |

#### Activation Checkpointing

反向传播需要前向的激活值，但存所有激活值太占显存：

```
普通方式：存所有层的激活值，反向时直接用
Checkpointing：只存部分层的激活值，其余在反向时重新计算（用时间换空间）
```

Megatron 在 Transformer 层边界做 checkpoint，每层只重算一次，显存节省约 **√层数** 倍。

#### Vocabulary Embedding 并行

词表 Embedding 矩阵 [V, H] 可能很大（词表 50K × 隐层 3072 = 600M 参数），同样按列切分到多卡。

---

### 训练规模与性能

| 模型 | 参数量 | 层数 | 隐层 |
|------|-------|------|------|
| Megatron-Large | 1.2B | 24 | 1536 |
| Megatron-XL | 2.5B | 32 | 1920 |
| Megatron-XXL | **8.3B** | 72 | 3072 |

8.3B 是当时（2019年）最大的 Transformer 语言模型。

**扩展效率**：

| GPU 数 | 模型并行度 | 效率 |
|--------|----------|------|
| 1 | 1 | 基准 |
| 8 | 8 | **99%** |
| 512 | 8×64 | **76%** |

8 卡内（NVLink）效率 99%，跨节点效率下降到 76%，原因是 InfiniBand 带宽低于 NVLink。

---

### 论文结构速览

```
Section 1: Introduction        ← 模型并行动机，与数据并行的关系
Section 2: Transformer Background ← 标准 Transformer 结构回顾
Section 3: Model Parallel Transformer ← 核心：MLP/Attention 切分方案（重点）
Section 4: Setup               ← 硬件、数据集、训练配置
Section 5: Experiments         ← 扩展效率、下游任务结果
Section 6: Conclusion          ← 总结与展望
```

---

### 核心主线

> Megatron 的本质是：**把矩阵乘法按行/列切分到多卡，每卡算部分结果，最后 AllReduce 合并**。
> 精心设计切分方案，使得每个 Transformer 层只需要 2 次 AllReduce，通信开销极小。
> 前提条件：模型并行组内的卡必须用 NVLink 高带宽互联，否则通信成瓶颈。

---

### 与 ZeRO 的本质区别

```
ZeRO：     每张卡有完整的计算图，只是参数/梯度/优化器状态分开存
           通信发生在：梯度同步（DP 维度）

Megatron：  每张卡只有模型的一部分，计算图被切开
           通信发生在：前向/反向传播中间（激活值 AllReduce）

ZeRO 易用（不改模型代码），Megatron 高效（通信量更小但要求高带宽互联）
```

---

### 后续工作

| 论文 | 新增内容 |
|------|---------|
| **Megatron-v2**（2021，arXiv:2104.04473） | 加入流水线并行（PP），TP+PP+DP 三维并行 |
| **Megatron-Turing NLG 530B**（2022） | TP+PP+DP 联合 DeepSpeed ZeRO，训练 530B 模型 |
| **Megatron-Core**（2023） | 重构为独立库，与框架解耦 |

---

## 6. GPipe

**论文**：GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism
**作者**：Yanping Huang et al.（Google Brain）
**发表**：NeurIPS 2019（arXiv:1811.06965）
**链接**：https://arxiv.org/abs/1811.06965

---

### 定位：流水线并行的奠基工作

GPipe 是 Megatron v2 论文里 "Default Schedule" 的原型，理解 GPipe 是理解后续所有流水线并行优化（PipeDream、Interleaved Schedule）的基础。

```
GPipe（本篇）：      提出 micro-batch 流水线 + re-materialization
PipeDream：         改进调度顺序，解决 GPipe 显存大的问题（1F1B）
Megatron Interleaved：进一步降低 bubble
```

---

### 朴素模型并行的问题

不用 micro-batch 时，朴素流水线并行只有一个 micro-batch 在流动：

```
时间线（K=4 个 stage）：

stage 0: [████][ 空闲 ][ 空闲 ][ 空闲 ][梯度]
stage 1: [ 等 ][████ ][ 空闲 ][ 空闲 ][ 等 ][梯度]
stage 2: [ 等 ][ 等  ][ ████ ][ 空闲 ][ 等 ][ 等 ][梯度]
stage 3: [ 等 ][ 等  ][ 等   ][ ████ ][ 等 ][ 等 ][ 等 ][梯度]

结论：同一时刻只有 1 张卡在计算，其余全部空闲
GPU 利用率 = 1/K
```

K=4 时只有 25% 的利用率，完全不可接受。

---

### GPipe 的解法：Mini-Batch 切分为 M 个 Micro-Batch

把一个 mini-batch（大小 N）切成 M 个 micro-batch（每个大小 N/M），让多个 micro-batch 同时在流水线中流动：

```
K=4 个 stage，M=8 个 micro-batch，时间线：

stage 0: [F1][F2][F3][F4][F5][F6][F7][F8][  bubble  ][B8][B7][B6][B5][B4][B3][B2][B1]
stage 1:     [F1][F2][F3][F4][F5][F6][F7][F8][ bubble ][B8][B7][B6][B5][B4][B3][B2][B1]
stage 2:         [F1][F2][F3][F4][F5][F6][F7][F8][bub][B8][B7][B6][B5][B4][B3][B2][B1]
stage 3:             [F1][F2][F3][F4][F5][F6][F7][F8][B8][B7][B6][B5][B4][B3][B2][B1]
                                                        ↑ flush 点（所有 F 完成后才开始 B）
```

**关键特征（Default / F-then-B Schedule）**：
- 所有 micro-batch 先全部 forward，再全部 backward
- flush 点之前积累所有梯度，optimizer 一次性更新

---

### Bubble 比例公式

```
bubble_fraction = (K - 1) / (M + K - 1) ≈ (K - 1) / M

K = stage 数（GPU 数）
M = micro-batch 数

当 M >> K 时，bubble → 0
推荐：M ≥ 4K
```

**举例**：
```
K=4，M=8：  bubble = 3/11 ≈ 27%
K=4，M=32： bubble = 3/35 ≈ 9%
K=4，M=128：bubble = 3/131 ≈ 2%（几乎可忽略）
```

增大 M 能压低 bubble，但代价是显存占用增大（要同时保存 M 个 micro-batch 的激活值）。

---

### Re-Materialization（核心技术，又称 Gradient Checkpointing）

**问题**：GPipe 的 F-then-B 调度需要保存所有 M 个 micro-batch 所有层的激活值，显存开销是：

```
普通存储：O(N × L)
  N = mini-batch size，L = 总层数，每层激活值都要存
```

**解法**：只在 stage 边界存 checkpoint，反向时**重新计算**（re-materialize）本 stage 内的激活值：

```
存储内容：
  每个 stage 的输入激活值（stage 边界，共 K 个检查点）

不存储：
  stage 内部各层的中间激活值

反向传播时：
  需要某层激活值 → 从该 stage 的输入重新跑一遍 forward → 得到所需激活值
```

**显存分析**：

```
不用 re-materialization：
  显存 = O(N × L)（存所有层激活值）

用 re-materialization：
  显存 = O(N + (L/K) × (N/M))
         ↑          ↑
    stage边界     单个stage内
    checkpoint   临时重算激活值

实验结论：re-materialization 使单卡可训练模型大 2.7×
```

**计算代价**：每个 stage 的 forward 多跑一次（重计算），整体计算量增加约 **1/K**（可接受）。

---

### 与后续工作的关系

GPipe 的调度方式被称为 **F-then-B（先全部 Forward 再全部 Backward）**，这是它的主要缺点：

```
GPipe（F-then-B）：
  优点：实现简单，梯度精确（所有 microbatch 用同一版本参数）
  缺点：需要保存 M 个 microbatch 的激活值，显存是 O(M × stage大小)

PipeDream-Flush（1F1B）：
  改进：交替执行 F 和 B，同时只需保存 p 个 microbatch 的激活值
  显存从 O(M) 降到 O(p)，Megatron v2 采用此方案

Interleaved Schedule：
  进一步改进：每张卡负责多个非连续 chunk，bubble 再降低 v 倍
```

---

### 实验结果

**AmoebaNet（图像分类）**：
- 单卡：限于显存，最大只能训练 ~82M 参数的 AmoebaNet
- 8 卡 GPipe：训练 **1.8B 参数**的 AmoebaNet-D（提升 20×）
- ImageNet top-1 准确率：84.4%（当时 SOTA）

**Transformer（多语言翻译）**：
- 4 卡 GPipe：3.5× 加速
- 83.9B 参数 Transformer（128 个分区 + re-materialization）

**扩展效率**：M ≥ 4K 时接近线性扩展。

---

### 论文结构速览

```
Section 1: Introduction          ← 模型并行动机，GPipe 定位
Section 2: The GPipe Library     ← 接口设计，用户如何使用
Section 3: Performance Analysis  ← Bubble 公式，re-materialization 分析（重点）
Section 4: Image Classification  ← AmoebaNet 实验
Section 5: Multilingual NMT      ← Transformer 实验
Section 6: Related Work          ← 与模型并行、数据并行的对比
```

---

### 核心主线

> GPipe 的核心是两件事：
> 1. **Micro-batch 流水线**：把 mini-batch 切成 M 个 micro-batch，让多个 micro-batch 同时在流水线中流动，bubble 比例降到 (K-1)/M，M 越大 bubble 越小。
> 2. **Re-Materialization**：只在 stage 边界存 checkpoint，反向时重算 stage 内激活值，显存节省 ~2.7×，使得大 M 成为可能。

两者相互配合：大 M 减少 bubble，re-materialization 让大 M 的显存可接受。

---

## 7. PipeDream

**论文**：PipeDream: Fast and Efficient Pipeline Parallel DNN Training
**作者**：Aaron Harlap, Deepak Narayanan, Amar Phanishayee et al.（CMU + Microsoft Research）
**发表**：SOSP 2019（arXiv:1806.03377）
**链接**：https://arxiv.org/abs/1806.03377
**代码**：https://github.com/msr-fiddle/pipedream

---

### 定位：解决 GPipe 的两个问题

GPipe 引入了 micro-batch 流水线，但存在两个缺点：

```
问题 1：显存大
  GPipe 的 F-then-B 调度需要同时保存所有 M 个 micro-batch 的激活值
  M 越大（bubble 越小），显存越大，两者相互矛盾

问题 2：All-then-Backward 导致 GPU 空闲
  前 K-1 个 micro-batch 填充流水线时，后面的 stage 必须等待
  最后 K-1 个 micro-batch 排空流水线时，前面的 stage 必须等待
```

PipeDream 的核心贡献：**1F1B 调度**，同时解决这两个问题。

---

### 1F1B 调度（One-Forward-One-Backward）

**核心思想**：不等所有 micro-batch 全部 forward 完，而是尽早开始 backward。

**调度规则**（稳态）：
- 每个 stage 在稳态时交替执行一次 forward 和一次 backward
- 优先处理编号小的 micro-batch 的 backward（尽早释放激活值）

```
时间线（K=4 个 stage，M=8 个 micro-batch）：

         时钟周期→  1    2    3    4    5    6    7    8    9    10   11
stage 0:          [F1] [F2] [F3] [F4] [B1] [F5] [B2] [F6] [B3] [F7] [B4]...
stage 1:               [F1] [F2] [F3] [F4] [B1] [F5] [B2] [F6] [B3] [F7]...
stage 2:                    [F1] [F2] [F3] [F4] [B1] [F5] [B2] [F6] [B3]...
stage 3:                         [F1] [F2] [F3] [F4] [B1] [F5] [B2] [F6]...

稳态（周期 5 之后）：每个 stage 每个时钟周期都在工作，无空闲
```

**与 GPipe 的对比**：

| | GPipe（F-then-B） | PipeDream（1F1B） |
|--|-----------------|-----------------|
| 调度 | 全部 F → 全部 B | F 和 B 交替 |
| 激活值显存 | O(M × stage大小) | O(K × stage大小) |
| 稳态 GPU 利用率 | 有明显 bubble | 稳态接近 100% |
| 实现复杂度 | 简单 | 较复杂（需 weight stashing） |

显存从 O(M) 降到 O(K)，由于通常 M >> K，**显存大幅节省**。

---

### Weight Stashing（权重缓存）：核心难题

1F1B 引入了一个新问题——**权重不一致**。

**问题描述**：

```
stage 0 在时钟周期 1 用权重 W(t=0) 做 micro-batch 1 的 forward
stage 0 在时钟周期 5 收到 micro-batch 1 的 backward
  但此时 stage 0 的权重已经被 micro-batch 2/3/4 的梯度更新为 W(t=3)

问题：forward 用 W(t=0)，backward 用 W(t=3)
  → 违反链式法则，梯度计算错误
```

**PipeDream 的解法：Weight Stashing（权重缓存）**

每个 stage 为流水线中每个活跃的 micro-batch 保存一份权重快照：

```
stage 0 的权重缓存：
  micro-batch 1 的 forward：用 W(t=0)，存一份 W(t=0)
  micro-batch 2 的 forward：用 W(t=1)，存一份 W(t=1)
  ...

micro-batch 1 的 backward 到达时：
  取出之前存的 W(t=0) → 用它计算梯度 → 保证 F/B 权重一致
  计算完 → 释放 W(t=0) 的缓存
```

**显存代价**：同时缓存 K 份权重（K = stage 数），每个 stage 需要 K 倍的权重显存。

---

### 流水线的填充与排空（Startup & Drain）

每个 batch 结束时，PipeDream 执行一次 **Pipeline Flush**：

```
停止注入新的 micro-batch
等待所有在途的 micro-batch 完成 forward 和 backward
所有 stage 做一次参数同步（optimizer step）
然后开始下一个 batch
```

Flush 的意义：保证**跨 batch 的权重一致性**，避免梯度累积误差无限放大。

```
bubble_fraction = (K - 1) / (M + K - 1) ≈ (K - 1) / M
```

---

### 自动分区算法

PipeDream 包含**自动决定如何切分模型**的动态规划算法：

```
目标：min(max(stage_0_time, stage_1_time, ..., stage_K_time))
约束：每个 stage 的显存不超过 GPU 显存上限
```

**为什么要平衡**：流水线吞吐由最慢的 stage 决定，若 stage 间计算量不均，快的 stage 会空等慢的 stage。

---

### 论文结构速览

```
Section 1: Introduction         ← 流水线并行动机，朴素模型并行的低效
Section 2: Background           ← DNN 训练流程，模型并行基础
Section 3: PipeDream Overview   ← 1F1B 调度总览
Section 4: Work Scheduling      ← 填充/稳态/排空 三阶段调度细节（重点）
Section 5: Weight Stashing      ← 权重缓存机制（重点）
Section 6: Partitioning         ← 自动分区动态规划算法
Section 7: Vertical Sync        ← 跨 stage 权重一致性（可选）
Section 8: Evaluation           ← VGG/ResNet/LM 实验
Section 9: Related Work         ← 与 GPipe、数据并行对比
```

---

### 核心主线

> PipeDream 的核心是 **1F1B 调度**：不等所有 micro-batch 全部 forward 完，而是尽早开始 backward，稳态时每个 stage 每个时钟周期都满负荷运行。
> 代价是引入了权重不一致问题，用 **Weight Stashing** 解决——为每个在途 micro-batch 缓存一份权重快照，保证 F/B 使用相同权重。
> 每个 batch 结束时 Pipeline Flush 保证跨 batch 收敛正确性。

---

### 在流水线并行演进中的位置

```
GPipe（06）：    F-then-B 调度，定义 bubble 问题，显存 O(M)
PipeDream（本篇）：1F1B 调度，显存降到 O(K)，引入 weight stashing
Megatron v2（08）：PipeDream-Flush（1F1B + 去掉 weight stashing）
                  + Interleaved Schedule（进一步降 bubble）
```

---

## 8. Megatron-LM v2（3D 并行）

**论文**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM
**作者**：Deepak Narayanan, Mohammad Shoeybi, Jared Casper et al.（NVIDIA Research）
**发表**：SC'21（arXiv:2104.04473）
**链接**：https://arxiv.org/abs/2104.04473
**代码**：https://github.com/NVIDIA/Megatron-LM

---

### 与上一篇论文的关系

上一篇 Megatron-LM（第 5 篇）只有**张量并行（TP）**。
这篇论文在 TP 基础上加入**流水线并行（PP）**，并提出三维并行（TP + PP + DP）的完整组合方案。

```
Megatron v1（2019）：TP，解决单层装不下单卡的问题
Megatron v2（2021）：TP + PP + DP，解决千卡规模的训练效率问题
```

---

### 流水线并行基础

**核心思想**：把模型的层按顺序分配到不同 GPU，数据像工厂流水线一样依次通过各个 stage。

```
4 个 stage，每个 stage 负责 6 层（共 24 层）：

GPU_0（stage 0）：layer  1~6
GPU_1（stage 1）：layer  7~12
GPU_2（stage 2）：layer 13~18
GPU_3（stage 3）：layer 19~24

数据流：输入 → GPU_0 → GPU_1 → GPU_2 → GPU_3 → 输出
```

与张量并行的区别：
- **TP**：同一层切到多卡，卡间频繁通信（AllReduce）
- **PP**：不同层在不同卡，卡间只传激活值（Send/Recv，通信量小）

---

### Pipeline Bubble（核心问题）

**Bubble 比例公式**：

```
bubble_fraction = (p - 1) / m

p = pipeline stage 数
m = 每个 batch 的 microbatch 数

例：p=4，m=8 → bubble = 3/8 = 37.5%
```

---

### 两种调度方案对比

#### 方案一：Default Schedule（GPipe）

所有 microbatch 先全部做 forward，再全部做 backward。问题：需要存所有 m 个 microbatch 的激活值，显存占用大。

#### 方案二：1F1B Schedule（PipeDream-Flush）

交替执行 forward 和 backward，显存只需保留 p 个 microbatch 的激活值，bubble 比例不变，但显存大幅节省。

#### 方案三：Interleaved Schedule（论文核心创新）

每张卡不连续负责多个 chunk（虚拟 stage）：

```
模型 24 层，4 张卡，每卡 2 个 chunk：
  GPU_0（chunk 0 + chunk 4）：layer 1~3,  layer 13~15
  GPU_1（chunk 1 + chunk 5）：layer 4~6,  layer 16~18
  GPU_2（chunk 2 + chunk 6）：layer 7~9,  layer 19~21
  GPU_3（chunk 3 + chunk 7）：layer 10~12, layer 22~24
```

**Bubble 比例**：

```
interleaved_bubble_fraction = (1/v) × (p - 1) / m

v = 每张卡的 chunk 数

例：p=4，m=8，v=2 → bubble = (1/2) × 3/8 = 18.75%（减半！）
```

**代价**：通信量增加 v 倍。**实验结论**：吞吐提升约 **10%**。

---

### Sequence Parallelism（序列并行）

TP 把矩阵乘法切到多卡，但 LayerNorm 和 Dropout 无法被 TP 切分，每张卡都要存完整序列长度的激活值。

解法：在 TP 的基础上，把序列维度也切分：

```
TP 负责：Linear 层的切分（hidden 维度）
SP 负责：LayerNorm/Dropout 的切分（sequence 维度）
```

**通信原语变化**：
- 原 TP：All-Reduce
- TP + SP：All-Gather + Reduce-Scatter（通信量相同，但消除了激活值冗余）

---

### 3D 并行的配置原则

**原则 1：TP 优先用在节点内**
```
TP 的 AllReduce 通信频繁且延迟敏感 → 必须走 NVLink（节点内高带宽）
PP 的 Send/Recv 通信量小、延迟不敏感 → 可以走 InfiniBand（跨节点）
```

**原则 2：先用模型并行装下模型，再用 DP 提吞吐**
```
总 GPU = TP × PP × DP
step 1：确定 TP × PP（使模型能装进显存）
step 2：剩余 GPU 全部用于 DP（提高吞吐量）
```

**原则 3：PP 的 stage 数影响 bubble，需要保证 m >> p**

---

### 实验规模与性能

| 配置 | 结果 |
|------|------|
| 3072 张 A100，1T 参数模型 | 163 TFLOPs/GPU（峰值 52%） |
| 聚合算力 | 502 PetaFLOPs |
| Interleaved vs Default | 吞吐提升 ~10% |

---

### 论文结构速览

```
Section 1: Introduction          ← 三种并行的动机和挑战
Section 2: Tensor Parallelism    ← 回顾 Megatron v1 的 TP（简短）
Section 3: Pipeline Parallelism  ← Default / 1F1B / Interleaved 三种调度（重点）
Section 4: Sequence Parallelism  ← SP 与 TP 的配合
Section 5: 3D Parallelism        ← TP+PP+DP 组合原则（重点）
Section 6: Microbatch Size       ← microbatch 选取对性能的影响
Section 7: Experiments           ← 1T 模型训练实验
Section 8: Related Work          ← 与 ZeRO、PipeDream 等方案对比
```

---

### 核心主线

> 本文的核心是两件事：
> 1. **Interleaved Pipeline Schedule**：把每张卡负责的层做成非连续的多个 chunk，bubble 比例降低 v 倍，代价是通信量增加 v 倍，实测吞吐提升 10%。
> 2. **3D 并行配置原则**：TP 用节点内 NVLink，PP 用跨节点 InfiniBand，DP 用剩余 GPU 提吞吐，三者正交组合实现千卡规模的高效训练。

---

### Megatron 两篇论文对比

| | Megatron v1（第5篇） | Megatron v2（本篇） |
|--|-----------------|-------------------|
| 并行方式 | TP | TP + PP + DP |
| 核心创新 | 层内矩阵切分 | Interleaved Pipeline Schedule |
| 最大规模 | 8.3B（512 V100） | 1T（3072 A100） |
| 通信优化 | 每层 2 次 AllReduce | Scatter-Gather 多网卡并行 |
| 激活显存 | Checkpointing | Selective Recomputation |

---

## 9. Reducing Activation Recomputation

**论文**：Reducing Activation Recomputation in Large Transformer Models
**作者**：Vijay Korthikanti, Jared Casper, Sangkug Lym et al.（NVIDIA Research）
**发表**：MLSys 2023（arXiv:2205.05198）
**链接**：https://arxiv.org/abs/2205.05198
**实现**：Megatron-LM / NeMo-Megatron

---

### 定位：解决激活值显存的精细化问题

前几篇论文的激活值处理策略只有两种极端：

```
Full Recomputation（GPipe/Megatron v1）：
  不存任何激活值，反向时全部重算
  优点：显存最省
  缺点：计算量增加 ~33%（每层 forward 多跑一次）

Full Storage（不做 Checkpointing）：
  存所有层的所有激活值
  优点：计算量最小
  缺点：显存爆炸，无法训练大模型
```

本论文的目标：**找到中间地带——只重算那些"便宜但占地方"的激活值，保留那些"贵但不大"的激活值**。

---

### 背景：激活值显存的构成

Transformer 一层的激活值显存（混合精度，FP16）：

```
每层激活值大小 ≈ sbh(34 + 5as/h) bytes

其中：
  s = 序列长度（sequence length）
  b = batch size
  h = hidden dimension
  a = attention head 数

各部分来源：
  线性层输入/输出：~10·sbh
  Attention 中间值：~5·as²b（attention map，随序列长度平方增长！）
  LayerNorm 输入：  ~4·sbh
  Dropout mask：    ~sbh
```

**关键发现**：attention 相关的激活值随序列长度平方增长，是显存的主要占用者——但 attention 的计算量是可重算代价最小的部分。

---

### 技术一：Selective Activation Recomputation（选择性重计算）

**不重算全部，只重算 attention 内部的激活值**：

```
保留（不重算）：
  线性层（QKV 投影、FFN）的输入激活值
  LayerNorm 的输入
  → 这些计算量大，重算代价高

重算（不保存）：
  Q·K^T 的结果（attention score）
  Softmax 的输出
  Attention 加权 V 的结果
  → 这些占显存大（O(as²b)），但计算量小
```

**显存节省**：

```
有张量并行（TP=t）时，每层激活值：

Full Storage：    sbh(10 + 24/t + 5as/ht)
Selective Recomp：sbh(10 + 24/t)            ← 去掉 5as/ht 项

对大模型（s=2048，a=96，h=12288）：
  节省约 70%（GPT-3 规格）

计算开销：仅增加 2.7%（只重算 attention 内部，非全层）
```

---

### 技术二：Sequence Parallelism（序列并行）

**问题：张量并行（TP）的盲区**

TP 把 Linear 层切分到多卡，但 LayerNorm、Dropout 这些算子无法被 TP 切分：

```
→ TP=t 时，这部分激活值有 t 份冗余副本
```

**解法：把这些算子按序列维度切分**

```
每张卡负责序列的 1/t：
  卡 0 处理 token 0 ~ s/t
  卡 1 处理 token s/t ~ 2s/t

LayerNorm 和 Dropout 在各自的序列片段上独立计算
→ 激活值从 [s, b, h] 降到 [s/t, b, h]，节省 t 倍
```

**通信原语的变化**：

```
进入 Linear 层前（SP→TP 切换）：
  All-Gather：[s/t, b, h] × t 卡 → [s, b, h]

Linear 层内：TP 正常工作

离开 Linear 层后（TP→SP 切换）：
  Reduce-Scatter：[s, b, h] → [s/t, b, h] × t 卡

关键性质：All-Gather + Reduce-Scatter 的通信量 = All-Reduce 的通信量
         → 引入 SP 不增加任何通信量！
```

---

### 两种技术的显存效果叠加

```
每层激活值大小（bytes）：

不用任何技术：              sbh(34 + 5as/h)
加 TP（t路并行）：          sbh(10 + 24/t + 5as/ht)
加 TP + SP：               sbh(34/t + 5as/ht)
加 TP + SP + Selective：    sbh(34/t)              ← 最优

综合效果（GPT-3 规格，t=8）：
  不优化：sbh × ~130
  TP only：sbh × ~18
  TP + SP + Selective：sbh × ~4.25

总体节省：约 5× vs 纯 TP
```

---

### 与 Full Recomputation 的对比

```
Full Recomputation：显存最省，但计算多 ~33%
TP + SP + Selective：显存接近 Full Recomputation，但计算只多 ~2.7%
```

**结论**：本文方案几乎免费地达到 Full Recomputation 的显存效果，无需接受 33% 的计算代价。

---

### 实验结果

在 2240 张 A100 上，测试 22B 到 1T 参数的模型：

| 指标 | Full Recomputation | 本文方案（SP + Selective） |
|------|------------------|------------------------|
| 额外计算开销 | ~33% | **~2.7%** |
| 训练吞吐提升 | 基准 | **+29~32%** |
| MFU（1T 模型） | 42.1% | **56.3%** |

---

### 论文结构速览

```
Section 1: Introduction          ← 激活值显存是训练瓶颈，提出两种技术
Section 2: Background            ← Transformer 激活值构成分析
Section 3: Sequence Parallelism  ← SP 设计，All-Gather + Reduce-Scatter（重点）
Section 4: Selective Recomputation ← 哪些算子重算，哪些保留（重点）
Section 5: Memory Analysis       ← 精确的显存公式推导
Section 6: Experiments           ← 22B~1T 参数规模实验
Section 7: Related Work          ← 与 GPipe、ZeRO、Full Recomp 对比
```

---

### 核心主线

> 本文的核心是两个互补的技术：
>
> 1. **Sequence Parallelism**：让 TP 无法切分的算子（LayerNorm/Dropout）也按序列维度分布到多卡，消除 t 份冗余副本，通信量不增加。
>
> 2. **Selective Recomputation**：只重算 attention 内部的激活值（占显存大但计算量小），保留 Linear 层的激活值，计算开销从 Full Recomputation 的 33% 降到 2.7%。
>
> 两者结合，在几乎不增加计算开销的前提下，达到接近 Full Recomputation 的显存效果。

---

### 在整体脉络中的位置

```
Megatron v1（05）：提出 TP，层内矩阵切分
Megatron v2（08）：加入 PP，简述 Selective Recomputation
本篇（09）：       Selective Recomputation + Sequence Parallelism 的完整技术报告
```

---

*全文完*
