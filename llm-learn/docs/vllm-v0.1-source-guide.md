# vLLM v0.1.0 源码精读

> **版本**：v0.1.0（tag: v0.1.0，对应论文 *Efficient Memory Management for Large Language Model Serving with PagedAttention*，SOSP 2023）
> **代码量**：48 个 Python 文件，约 3000 行，极为精简
> **核心价值**：这是 vLLM 最原始的形态，架构清晰，没有任何工程积累的复杂性，是理解 PagedAttention 设计思想的最佳版本

---

## 一、整体文件结构

```
vllm/
├── block.py                          # 逻辑块 / 物理块定义（PagedAttention 的基石）
├── sequence.py                       # 请求的核心数据结构（Sequence / SequenceGroup）
├── sampling_params.py                # 采样参数
├── outputs.py                        # 用户侧输出格式
├── config.py                         # 所有配置（模型/缓存/调度/并行）
├── utils.py                          # 工具函数
│
├── core/
│   ├── block_manager.py              # KV Cache 物理块分配器（PagedAttention 的核心）
│   ├── scheduler.py                  # 调度器：三队列调度 + preemption
│   └── policy.py                     # 调度策略（FCFS）
│
├── engine/
│   ├── llm_engine.py                 # 主引擎（控制核心，step() 主循环）
│   ├── async_llm_engine.py           # 异步封装（在线服务用）
│   ├── arg_utils.py                  # 命令行参数解析
│   ├── tokenizer_utils.py            # 增量 detokenize
│   └── ray_utils.py                  # 多 GPU / Ray 工具
│
├── worker/
│   ├── worker.py                     # GPU Worker（单卡执行实体）
│   └── cache_engine.py               # GPU/CPU KV Cache 内存管理
│
├── model_executor/
│   ├── input_metadata.py             # 传给模型的元数据（slot_mapping / block_tables）
│   ├── model_loader.py               # 模型加载
│   ├── layers/
│   │   ├── attention.py              # PagedAttention 算子（最核心）
│   │   ├── sampler.py                # 采样层（temperature / top-p / top-k / beam search）
│   │   ├── activation.py             # 激活函数
│   │   └── layernorm.py              # RMSNorm
│   └── models/
│       ├── llama.py                  # LLaMA 模型实现
│       ├── opt.py                    # OPT 模型实现
│       ├── gpt2.py                   # GPT-2 模型实现
│       └── gpt_neox.py               # GPT-NeoX 模型实现
│
└── entrypoints/
    ├── llm.py                        # 离线批量推理接口（LLM 类）
    └── openai/
        ├── api_server.py             # OpenAI 兼容 HTTP 服务
        └── protocol.py               # OpenAI 协议类型定义
```

---

## 二、完整请求调用链

```
用户调用 LLM.generate(prompts, sampling_params)
    │
    ▼
【① 离线入口】
vllm/entrypoints/llm.py
  LLM.generate()
    └─ _run_engine()         # 同步循环直到所有请求完成
         └─ engine.step()    # 反复调用 LLMEngine.step()

  _add_request()             # 将每个 prompt 加入引擎
    ├─ tokenizer.encode(prompt)        # tokenize
    ├─ 创建 Sequence（建立逻辑块）
    ├─ 创建 SequenceGroup
    └─ scheduler.add_seq_group()       # 加入 waiting 队列
    │
    ▼
【② LLMEngine.step()：一次完整的推理循环】
vllm/engine/llm_engine.py
  step()
    ├─ scheduler.schedule()            # ① 调度决策
    ├─ _run_workers("execute_model")   # ② GPU 执行
    ├─ scheduler.update(output)        # ③ 追加新 token
    ├─ _decode_sequences()             # ④ detokenize
    ├─ _stop_sequences()               # ⑤ 检查停止条件
    ├─ scheduler.free_finished_seq_groups()  # ⑥ 清理完成请求
    └─ 返回 List[RequestOutput]
    │
    ▼
【③ Scheduler.schedule()：三队列调度】
vllm/core/scheduler.py
  schedule()
    └─ _schedule()
         ├─ 1. 优先处理 running 队列（为下一个 token 分配 KV 块）
         │       → 若内存不足，触发 preemption（swap out 或 recompute）
         ├─ 2. 尝试将 swapped 队列 swap in 回 GPU
         └─ 3. 从 waiting 队列调度新请求（prefill）
    │
    ▼
【④ Worker.execute_model()：GPU 单步执行】
vllm/worker/worker.py
  execute_model()
    ├─ cache_engine.swap_in/swap_out/copy()  # 异步执行 KV Cache 搬运
    └─ _prepare_inputs()                     # 构建 input_ids、slot_mapping、block_tables
         └─ model(input_ids, positions, kv_caches, input_metadata, cache_events)
              │
              ▼
【⑤ PagedAttention.forward()：核心算子】
vllm/model_executor/layers/attention.py
  forward()
    ├─ 对 prompt tokens：multi_query_kv_attention()     # xformers 实现
    ├─ 等待 cache_event（KV 搬运完成）
    ├─ cache_ops.reshape_and_cache()                   # 将新 KV 写入物理块
    └─ 对 generation tokens：single_query_cached_kv_attention()  # CUDA kernel

  → 返回 hidden_states → Sampler.forward()
    │
    ▼
【⑥ Sampler.forward()：采样】
vllm/model_executor/layers/sampler.py
  forward()
    ├─ _prune_hidden_states()    # 只取每个序列最后一个 token 的 hidden state
    ├─ logits = hidden @ embedding.T   # 投影到词表
    ├─ _apply_penalties()        # presence / frequency penalty
    ├─ logits.div_(temperature)  # 温度缩放
    ├─ _apply_top_p_top_k()      # nucleus / top-k 截断
    └─ _sample()                 # greedy / multinomial / beam search
         → 返回 Dict[seq_id, SequenceOutputs]
```

---

## 三、核心数据结构精读

### 3.1 物理块与逻辑块（`vllm/block.py`）

PagedAttention 的两个基础抽象：

```python
class LogicalTokenBlock:
    """从序列视角看的连续 token 块（虚拟地址）。
    每个 Sequence 维护一个 LogicalTokenBlock 列表，顺序编号（0、1、2...）。
    """
    block_number: int      # 在序列内的编号（0 起）
    block_size: int        # 块大小（默认 16 tokens）
    token_ids: List[int]   # 存储的 token ID（-1 表示空槽）
    num_tokens: int        # 已填入的 token 数

class PhysicalTokenBlock:
    """GPU/CPU 显存中的真实内存块（物理地址）。
    BlockAllocator 统一管理，通过引用计数支持共享（beam search 的 Copy-on-Write）。
    """
    device: Device         # GPU 或 CPU
    block_number: int      # 物理块编号（= 在 KV cache tensor 中的 index）
    block_size: int
    ref_count: int         # 引用计数（>1 表示被多个序列共享）
```

**关键映射**：`LogicalTokenBlock[i]` → `PhysicalTokenBlock`，由 `BlockSpaceManager` 的 `block_tables` 维护。

### 3.2 序列与请求（`vllm/sequence.py`）

```python
class SequenceData:
    """纯数据层：只存 token IDs 和 logprob。不涉及内存管理。"""
    prompt_token_ids: List[int]   # 输入 token
    output_token_ids: List[int]   # 已生成 token
    cumulative_logprob: float     # 累计 log 概率（beam search 排序用）

class Sequence:
    """单条生成序列。持有逻辑块列表，管理 token 追加。"""
    seq_id: int
    data: SequenceData
    logical_token_blocks: List[LogicalTokenBlock]  # 逻辑块列表
    status: SequenceStatus         # WAITING / RUNNING / SWAPPED / FINISHED_*
    output_tokens: List[str]       # 已解码的文本 token（增量 detokenize 用）
    output_text: str               # 已解码的完整输出文本

class SequenceGroup:
    """一个用户请求对应一个 SequenceGroup。
    普通采样（n=1）：包含 1 个 Sequence；
    并行采样（n>1）或 beam search（best_of>1）：包含多个 Sequence，共享 prompt 的物理块。
    """
    request_id: str
    seqs: List[Sequence]
    sampling_params: SamplingParams
    arrival_time: float

class SequenceGroupMetadata:
    """Scheduler → Worker 的传递载体。每步调度后生成，含 block_tables。"""
    request_id: str
    is_prompt: bool              # True=prefill 阶段，False=decode 阶段
    seq_data: Dict[int, SequenceData]    # seq_id → 数据
    sampling_params: SamplingParams
    block_tables: Dict[int, List[int]]  # seq_id → [物理块编号列表]

class SequenceOutputs:
    """Worker → Scheduler 的采样结果。每个序列一个。"""
    seq_id: int
    parent_seq_id: int           # beam search 时可能指向不同父序列
    output_token: int            # 本步采样的 token
    logprobs: Dict[int, float]   # token_id → log probability
```

### 3.3 调度器输出（`vllm/core/scheduler.py`）

```python
class SchedulerOutputs:
    """调度决策的执行指令，由 Worker 直接执行。"""
    blocks_to_swap_in: Dict[int, int]      # CPU块号 → GPU块号（swap in 操作）
    blocks_to_swap_out: Dict[int, int]     # GPU块号 → CPU块号（swap out 操作）
    blocks_to_copy: Dict[int, List[int]]   # 源块号 → 目标块号列表（CoW 操作）
    # 注意：swap_in 和 swap_out 不会同时发生
```

### 3.4 KV Cache 内存布局（`vllm/worker/cache_engine.py`）

```python
# 每层 KV Cache 的存储格式：
#
# key_cache:   Tensor[num_gpu_blocks, num_heads, head_size/x, block_size, x]
#   x = 16 // element_size（FP16 时 x=8，目的是对齐 CUDA warp，提高内存带宽利用率）
#
# value_cache: Tensor[num_gpu_blocks, num_heads, head_size, block_size]
#
# 一个物理块（block_number=k）对应：
#   key_cache[k]   → 该块所有 token 的 K 向量
#   value_cache[k] → 该块所有 token 的 V 向量
#
# gpu_cache: List[Tuple[key_cache, value_cache]]，长度 = num_layers
```

### 3.5 传给 PagedAttention 的元数据（`vllm/model_executor/input_metadata.py`）

```python
class InputMetadata:
    """Worker._prepare_inputs() 构建，传递给每层 PagedAttention.forward()。"""
    seq_groups: List[Tuple[List[int], SamplingParams]]  # [(seq_ids, params)]
    seq_data: Dict[int, SequenceData]
    prompt_lens: List[int]           # 各 prefill 请求的 prompt 长度

    # PagedAttention 的核心索引
    slot_mapping: IntTensor          # [num_tokens] 每个 token 写入哪个物理槽
                                     # slot = block_number * block_size + block_offset
    context_lens: IntTensor          # [num_generation_tokens] 每个 decode token 的历史长度
    max_context_len: int
    block_tables: IntTensor          # [num_generation_seqs, max_blocks_per_seq] 物理块表
```

---

## 四、关键机制深读

### 4.1 PagedAttention：两阶段 forward

`vllm/model_executor/layers/attention.py` 的 `PagedAttention.forward()` 是整个系统最核心的算子，清晰地展示了 prefill 和 decode 的不同处理方式：

```
输入 batch 的 token 排布（1D 展平）：
|<---- prompt_0 ---->|<-- prompt_1 -->|...|<gen_0>|<gen_1>|...|<padding>|
                                          ↑ num_prompt_tokens

Step 1 [Prefill]：对所有 prompt token 做标准 attention（xformers）
  multi_query_kv_attention(output[:num_prompt_tokens], q, k, v, attn_bias)
  → attn_bias 是下三角因果掩码，确保每个 token 只看到自己及之前的 token

Step 2 [等待异步 KV 搬运完成]：
  cache_event.wait()
  → cache_engine 在独立的 CUDA stream 上做 swap_in/swap_out/copy，这里同步

Step 3 [写入 KV Cache]：
  cache_ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
  → 将所有有效 token（prompt + generation）的 KV 按 slot_mapping 写入物理块
  → slot_mapping[i] = block_number * block_size + block_offset

Step 4 [Decode]：对每个 generation token 做 PagedAttention 查询
  single_query_cached_kv_attention(output, query, key_cache, value_cache, input_metadata)
  → CUDA kernel：按 block_tables[seq_id] 找到该序列的所有历史物理块
  → 在这些块上计算 attention（跨不连续内存块的 attention）
```

### 4.2 BlockSpaceManager：页表管理

`vllm/core/block_manager.py` 实现了类操作系统的页表：

```python
# 核心数据结构
block_tables: Dict[int, BlockTable]
# seq_id → [PhysicalTokenBlock, PhysicalTokenBlock, ...]
# 逻辑块 i → block_tables[seq_id][i]（物理块对象）

# 获取 block_table 传给 Worker
get_block_table(seq) → List[int]  # 返回物理块编号列表（整数），供 CUDA kernel 使用
```

**四个核心操作**：

```
allocate(seq_group)：
  → 为 seq_group 的所有 prompt token 分配物理块
  → 若 n > 1（并行采样），同一 prompt 的物理块被所有 seq 共享
  → block.ref_count = seq_group.num_seqs()（引用计数）

append_slot(seq)：
  → decode 阶段：每步为新 token 申请槽位
  → 若当前最后一块未满：直接写入，无需新块
  → 若当前最后一块已满：分配新物理块
  → Copy-on-Write：若最后一块被多个 seq 共享（ref_count > 1）
       → 分配新块，返回 (旧块号, 新块号) → Scheduler 记录到 blocks_to_copy

fork(parent_seq, child_seq)：
  → beam search 分叉：child 直接引用 parent 的所有物理块（ref_count += 1）
  → 不分配新内存，O(1) 操作

free(seq)：
  → 逐块 ref_count -= 1，降为 0 则归还到 free_blocks 列表
```

### 4.3 三队列调度（`vllm/core/scheduler.py`）

调度器维护三个队列，每步 `_schedule()` 按固定优先级处理：

```
waiting  → 新请求，等待首次 prefill
running  → 正在 decode，每步需要新 KV 块
swapped  → 被换出到 CPU，等待 swap in
```

**调度优先级（从高到低）**：

```
优先级 1：保障 running 队列继续运行
  → 为每个 running seq_group 分配下一步的 KV 块
  → 若 GPU 内存不足（can_append_slot 返回 False）：
       触发 preemption，踢出 running 末尾（最低优先级）的 seq_group
       → 单序列：RECOMPUTE 模式（直接丢弃 GPU 块，重新放回 waiting 头部）
       → 多序列（beam search）：SWAP 模式（KV 块搬到 CPU，放入 swapped）

优先级 2：swap in（若无 swap out 压力时）
  → 将 swapped 头部 seq_group 换回 GPU（需要满足内存条件）
  → swapped 优先于 waiting（防止 CPU 内存无限增长）

优先级 3：加入 waiting 中的新请求（prefill）
  → 仅当 swapped 为空时才考虑
  → 限制条件：
       · can_allocate()：GPU 有足够空闲块（含 watermark 余量 1%）
       · num_batched_tokens：不超过 max_num_batched_tokens
       · num_seqs：不超过 max_num_seqs
```

**关键约束**：swap_in 和 swap_out 不会在同一步发生（代码中有 assert）。

### 4.4 Sampler 采样流水线

`vllm/model_executor/layers/sampler.py` 的 `Sampler.forward()` 步骤：

```
Step 1：_prune_hidden_states()
  → 从展平的 hidden_states 中，只取每个序列最后一个 token 的状态
  → prompt 取最后一个位置，generation token 只有一个位置

Step 2：logits = hidden_states @ embedding.T
  → 投影到词表维度（vocab_size）
  → 若 tensor parallel，gather_from_tensor_model_parallel_region() 合并结果

Step 3：_apply_penalties()
  → 使用 np.bincount 统计各 token 的出现频次
  → logits -= frequency_penalty * count  （频率惩罚）
  → logits -= presence_penalty * (count > 0)  （存在惩罚）

Step 4：logits.div_(temperature)  （温度缩放，原地）

Step 5：probs = softmax(logits, float32)
        logprobs = log(probs)

Step 6：_apply_top_p_top_k()
  → 排序 → cumsum → mask 掉概率累计超过 top_p 的 token
  → mask 掉 rank > top_k 的 token

Step 7：_sample()
  → greedy（temperature=0）：argmax
  → random（temperature>0）：multinomial
  → beam search：topk on (logprobs + cumulative_logprob)

→ 返回 Dict[seq_id, SequenceOutputs]（每个序列一个采样结果）
```

### 4.5 KV Cache 初始化：动态探测可用内存

`vllm/engine/llm_engine.py` 的 `_init_cache()` 使用一个巧妙的方法确定 KV Cache 大小：

```python
def _init_cache():
    # 1. 用最大 batch 做一次假前向，测量 GPU 峰值显存
    profile_num_available_blocks(block_size, gpu_memory_utilization, cpu_swap_space)
        → 用 max_num_seqs 个长度为 max_num_batched_tokens/max_num_seqs 的假序列
        → model.forward(kv_caches=[(None, None)] * num_layers)  # KV cache 传 None
        → peak_memory = torch.cuda.max_memory_allocated()
        → num_gpu_blocks = (total_gpu * utilization - peak_memory) // block_size_bytes

    # 2. 取所有 GPU worker 中的最小值（保证所有卡都能容纳）
    num_gpu_blocks = min(b[0] for b in num_blocks)  # 所有 worker 取 min

    # 3. 用确定的块数初始化真正的 KV cache tensor
    init_cache_engine(cache_config)
        → CacheEngine.allocate_gpu_cache()
        → 为每层分配 (num_gpu_blocks, ...) 的 key/value tensor
```

---

## 五、Prefill vs Decode 的区别

这是理解 vLLM 最重要的概念之一。同一个 `model.forward()` 调用可能同时包含两种请求：

| | Prefill（首次处理 prompt） | Decode（逐 token 生成） |
|--|--|--|
| **输入 token 数** | 完整 prompt（可能数百个） | 每序列恰好 1 个 |
| **Attention 计算** | 全序列自注意力（xformers） | 对历史全部 KV 做查询（CUDA kernel） |
| **KV Cache 操作** | 写入 prompt 所有 token 的 KV | 读取历史 KV + 写入新 token KV |
| **is_prompt 标志** | `True` | `False` |
| **block_tables** | 传 None（调度时才分配） | 传完整物理块映射表 |
| **slot_mapping** | 按顺序映射每个 prompt token | 只映射当前新 token 的槽位 |

在 `worker._prepare_inputs()` 中，batch 的内存排布是：

```
input_tokens = [prompt_0_tokens..., prompt_1_tokens..., gen_0_token, gen_1_token, ...]
              |<------------- prefill 部分 ------------->|<---- decode 部分 ---->|
```

`input_metadata.num_prompt_tokens` 记录分界点，`PagedAttention.forward()` 据此分两段处理。

---

## 六、异步 KV Cache 搬运机制

`CacheEngine` 使用独立的 CUDA stream 和 Event 实现异步搬运：

```python
# CacheEngine.__init__:
self.cache_stream = torch.cuda.Stream()       # 独立 stream
self.events = [torch.cuda.Event() for _ in range(num_layers)]  # 每层一个 event

# swap_in / swap_out / copy：
with torch.cuda.stream(self.cache_stream):    # 在独立 stream 上执行
    for layer_i in range(num_layers):
        cache_ops.swap_blocks(src[i], dst[i], mapping)
        event[i].record(stream=self.cache_stream)  # 记录完成 event

# PagedAttention.forward() 中：
if cache_event is not None:
    cache_event.wait()    # 当前计算 stream 等待 cache stream 完成
                          # 保证 KV 搬运完成后再写入新 token 的 KV
```

**时序**：
```
compute stream: [prompt attention] → wait(event[0]) → [reshape & cache] → [gen attention]
cache stream:   [swap ops layer 0]  record(event[0])
                [swap ops layer 1]  record(event[1])
                ...
```

---

## 七、核心参数说明（`vllm/config.py`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `block_size` | 16 | 每个物理块存储的 token 数 |
| `gpu_memory_utilization` | 0.9 | GPU 显存利用率上限（留 10% 余量） |
| `swap_space` | 4 GB | CPU swap 空间大小 |
| `max_num_batched_tokens` | 2560 | 每步 batch 的最大总 token 数（prefill + decode） |
| `max_num_seqs` | 256 | 同时运行的最大序列数 |
| `tensor_parallel_size` | 1 | 张量并行数量（跨 GPU 切分模型） |

---

## 八、目录结构与功能一览

| 文件 | 功能 | 核心类/函数 |
|------|------|------------|
| `block.py` | 逻辑/物理块定义 | `LogicalTokenBlock`, `PhysicalTokenBlock` |
| `sequence.py` | 请求数据结构 | `Sequence`, `SequenceGroup`, `SequenceGroupMetadata`, `SequenceOutputs` |
| `sampling_params.py` | 采样参数 | `SamplingParams` |
| `core/block_manager.py` | KV Cache 页表管理 | `BlockSpaceManager.allocate/append_slot/fork/free/swap_in/swap_out` |
| `core/scheduler.py` | 三队列调度 | `Scheduler.schedule()`, `_preempt_by_recompute/swap()` |
| `core/policy.py` | 调度策略 | `FCFSPolicy.sort_by_priority()` |
| `engine/llm_engine.py` | 主引擎 | `LLMEngine.step()`, `add_request()`, `_init_cache()` |
| `engine/async_llm_engine.py` | 异步封装 | `AsyncLLMEngine.generate()` |
| `worker/cache_engine.py` | GPU/CPU KV Cache 内存 | `CacheEngine.allocate_gpu_cache/swap_in/swap_out/copy` |
| `worker/worker.py` | GPU 执行单元 | `Worker.execute_model()`, `_prepare_inputs()`, `profile_num_available_blocks()` |
| `model_executor/input_metadata.py` | 模型输入元数据 | `InputMetadata`（slot_mapping, block_tables） |
| `model_executor/layers/attention.py` | PagedAttention 算子 | `PagedAttention.forward()`, `multi_query_kv_attention()`, `single_query_cached_kv_attention()` |
| `model_executor/layers/sampler.py` | 采样层 | `Sampler.forward()`, `_apply_penalties/temperature/top_p_top_k()`, `_sample()` |
| `model_executor/models/llama.py` | LLaMA 模型 | `LlamaForCausalLM.forward()` |
| `entrypoints/llm.py` | 离线推理入口 | `LLM.generate()` |

---

## 九、从 v0.1.0 到 v0.18.0 的演进方向

读完 v0.1.0 之后，理解 v0.18.0 时可以关注这些演进点：

| 方向 | v0.1.0 | v0.18.0 |
|------|--------|---------|
| **引擎架构** | 单进程，LLMEngine 直接调用 Worker | 多进程：API Server + EngineCore + GPU Worker 三进程，ZMQ 通信 |
| **Scheduler** | `core/scheduler.py`，三队列 + 简单 FCFS | `v1/core/sched/scheduler.py`，支持优先级、前缀缓存、结构化输出 |
| **KV Cache** | `BlockSpaceManager`，简单哈希无前缀缓存 | `KVCacheManager` + `BlockHashToBlockMap`，支持前缀缓存（prefix caching） |
| **Worker** | 同步 `Worker.execute_model()` | 异步 `AsyncGPUWorker`，`execute_model` + `sample_tokens` 分离 |
| **PagedAttention** | 直接调用 xformers + 自定义 CUDA kernel | FlashAttention 后端 + 多种 attention backend 抽象 |
| **数据结构** | `SequenceGroup`（含多个 Sequence） | `Request`（更扁平，直接持有 token ids 和 block_hashes） |
| **Preemption** | recompute（丢弃重算）或 swap（CPU 换出） | 同，但增加 `resumable` 断点续传能力 |
| **采样** | `Sampler` 在 `model_executor/layers/` | `Sampler` 在 `v1/sample/`，独立 RPC 调用 |
| **支持模型数** | 4 个（LLaMA、OPT、GPT-2、GPT-NeoX） | 数十个 |
