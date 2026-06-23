# vLLM 源码导读

> **版本**：v0.9.x（2025年6月）
> **GitHub**：https://github.com/vllm-project/vllm
> **阅读策略**：分两块——先串流程（忽略硬件适配细节），再读关键组件。

---

## 一、整体架构分层

```
┌─────────────────────────────────────────────────────────┐
│                   Entrypoint 层                          │
│  vllm/entrypoints/openai/api_server.py  (FastAPI HTTP)  │
│  vllm/entrypoints/llm.py               (离线 LLM 类)    │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP / ZMQ IPC
┌────────────────────────▼────────────────────────────────┐
│                  Engine 层（API Server 进程）             │
│  vllm/v1/engine/async_llm.py        AsyncLLM            │
│  vllm/v1/engine/llm_engine.py       LLMEngine（同步）   │
│  vllm/v1/engine/input_processor.py  输入预处理           │
│  vllm/v1/engine/output_processor.py 输出后处理           │
│  vllm/v1/engine/detokenizer.py      反 Tokenize         │
│  vllm/v1/engine/core_client.py      与 EngineCore 通信  │
└────────────────────────┬────────────────────────────────┘
                         │ ZMQ IPC（protobuf 消息）
┌────────────────────────▼────────────────────────────────┐
│                EngineCore 层（独立进程）                   │
│  vllm/v1/engine/core.py             EngineCore 主循环   │
│  vllm/v1/core/sched/scheduler.py    Scheduler 调度器    │
│  vllm/v1/core/kv_cache_manager.py   KV Cache 管理       │
│  vllm/v1/core/block_pool.py         物理块池             │
└────────────────────────┬────────────────────────────────┘
                         │ 共享内存 / MessageQueue
┌────────────────────────▼────────────────────────────────┐
│                 Executor / Worker 层                     │
│  vllm/v1/executor/multiproc_executor.py  多进程执行器   │
│  vllm/v1/worker/gpu_worker.py            GPU Worker     │
└────────────────────────┬────────────────────────────────┘
                         │ CUDA / NCCL
┌────────────────────────▼────────────────────────────────┐
│                 ModelRunner / Model 层                   │
│  vllm/v1/worker/gpu_model_runner.py      GPUModelRunner │
│  vllm/model_executor/models/             模型实现        │
│  vllm/v1/sample/sampler.py              Sampler         │
└─────────────────────────────────────────────────────────┘
```

### 多进程架构（V1 核心设计）

vLLM V1 将系统拆成多个进程，各司其职：

| 进程 | 数量 | 职责 |
|------|------|------|
| API Server | 1 | HTTP 接收、Tokenize、输出流 |
| EngineCore | 1 | 调度、KV Cache 管理、协调 GPU |
| GPU Worker | TP 并行度个 | 加载权重、执行模型前向 |

进程间通过 **ZMQ IPC**（API Server ↔ EngineCore）和 **共享内存 MessageQueue**（EngineCore ↔ Worker）通信。

---

## 二、完整请求调用链

### 第一块：流程串联（先读这里）

下面是一个请求从进入到输出的完整路径，**忽略多硬件适配、LoRA、多模态等旁支**，只看主干。

```
客户端 POST /v1/chat/completions
    │
    ▼
【① Entrypoint — API Server 进程】
vllm/entrypoints/openai/serving_chat.py
  OpenAIServingChat.create_chat_completion()
    ├─ 解析 ChatCompletionRequest
    ├─ 调用 tokenizer 处理消息
    └─ 调用 self.engine_client.generate()
    │
    ▼
vllm/v1/engine/async_llm.py
  AsyncLLM.generate()
    ├─ _add_request()
    │    ├─ InputProcessor.process_inputs()    # tokenize、构建 EngineCoreRequest
    │    └─ EngineCoreClient.add_request()     # ZMQ 发送给 EngineCore 进程
    └─ 持续 yield RequestOutput（流式返回）
    │
    │  [ZMQ IPC 跨进程]
    ▼
【② EngineCore — 独立进程，主调度循环】
vllm/v1/engine/core.py
  EngineCoreProc.run_engine_core()            # busy loop
    └─ EngineCore.step()
         ├─ Scheduler.schedule()              # 调度决策
         ├─ Executor.execute_model(sched_output)  # 执行模型
         └─ Scheduler.update_from_output()    # 更新状态
    │
    ▼
vllm/v1/core/sched/scheduler.py
  Scheduler.schedule()
    ├─ 遍历 running 请求，为新 token 分配 KV Cache slots
    ├─ KVCacheManager.allocate_slots()        # 分配物理块
    ├─ 从 waiting 队列调度新请求
    ├─ 必要时触发 preemption（抢占 running 请求）
    └─ 返回 SchedulerOutput（含 block_ids、token counts）
    │
    │  [共享内存]
    ▼
【③ Executor/Worker — GPU 进程】
vllm/v1/executor/multiproc_executor.py
  MultiprocExecutor.execute_model(scheduler_output)
    └─ collective_rpc("execute_model")        # 广播到所有 Worker
    │
    ▼
vllm/v1/worker/gpu_worker.py
  Worker.execute_model(scheduler_output)
    └─ model_runner.execute_model(scheduler_output)
    │
    ▼
【④ GPUModelRunner — 核心执行层】
vllm/v1/worker/gpu_model_runner.py
  GPUModelRunner.execute_model(scheduler_output)
    ├─ _preprocess()
    │    ├─ 构建 input_ids、position_ids
    │    └─ 构建 attention_metadata（block_table、slot_mapping）
    ├─ _model_forward()
    │    └─ model.forward(input_ids, positions, attn_metadata)
    │         # model 是 torch.nn.Module（如 LlamaForCausalLM）
    │         # PagedAttention 在此按 block_table 读写 KV Cache
    └─ _sample()
         └─ Sampler.forward(logits, sampling_metadata)
    │
    ▼
【⑤ Sampler】
vllm/v1/sample/sampler.py
  Sampler.forward(logits, sampling_metadata)
    ├─ apply_penalties()          # repetition/frequency penalty
    ├─ apply_temperature()        # logits /= temperature
    ├─ sample()
    │    ├─ greedy → argmax(dim=-1)
    │    └─ random → TopKTopP → multinomial
    └─ 返回 SamplerOutput（sampled_token_ids）
    │
    │  [结果逐层返回]
    ▼
【⑥ 输出处理 — API Server 进程】
vllm/v1/engine/output_processor.py
  OutputProcessor.process_outputs(engine_core_outputs)
    ├─ IncrementalDetokenizer.update()    # 增量 detokenize（只解码新 token）
    ├─ 检查 stop strings / stop token ids
    ├─ 检查 max_tokens 限制
    └─ 生成 RequestOutput

vllm/entrypoints/openai/serving_chat.py
    └─ 格式化为 ChatCompletionStreamResponse → 客户端
```

### 离线批量推理入口

```
vllm/entrypoints/llm.py
  LLM.generate(prompts, sampling_params)
    └─ _run_engine()           # 同步循环直到所有请求完成
         └─ engine.step()      # 反复调用

vllm/v1/engine/llm_engine.py  # 实际是 V1LLMEngine 的别名
  LLMEngine.step()
    ├─ engine_core_client.get_output()
    └─ output_processor.process_outputs()
```

---

## 三、关键源码文件速查

### Engine 层

| 组件 | 文件路径 | 核心函数 |
|------|----------|----------|
| 同步引擎 | `vllm/v1/engine/llm_engine.py` | `step()`, `add_request()` |
| 异步引擎 | `vllm/v1/engine/async_llm.py` | `generate()`, `_add_request()` |
| 引擎核心 | `vllm/v1/engine/core.py` | `EngineCore.step()` |
| 核心客户端 | `vllm/v1/engine/core_client.py` | `add_request()` |
| 输入处理 | `vllm/v1/engine/input_processor.py` | `process_inputs()` |
| 输出处理 | `vllm/v1/engine/output_processor.py` | `process_outputs()` |
| 反 Tokenize | `vllm/v1/engine/detokenizer.py` | `FastIncrementalDetokenizer.update()` |

> `vllm/engine/llm_engine.py` 是老路径，现在只是 V1 的别名，阅读 V1 版本即可。

### Scheduler 层

| 组件 | 文件路径 | 核心函数 |
|------|----------|----------|
| 主调度器 | `vllm/v1/core/sched/scheduler.py` | `schedule()`, `update_from_output()` |
| 调度输出 | `vllm/v1/core/sched/output.py` | `SchedulerOutput`, `NewRequestData` |
| 请求队列 | `vllm/v1/core/sched/request_queue.py` | FCFS / 优先级队列 |

### KV Cache 层

| 组件 | 文件路径 | 核心函数 |
|------|----------|----------|
| KV Cache 管理 | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots()`, `free()`, `get_computed_blocks()` |
| 块池 | `vllm/v1/core/block_pool.py` | `BlockPool`（物理块分配器） |
| GPU 侧块表 | `vllm/v1/worker/block_table.py` | 块表管理 |

### Executor / Worker 层

| 组件 | 文件路径 | 核心函数 |
|------|----------|----------|
| 多进程执行器 | `vllm/v1/executor/multiproc_executor.py` | `execute_model()`, `WorkerProc.worker_busy_loop()` |
| GPU Worker | `vllm/v1/worker/gpu_worker.py` | `execute_model()`, `init_device()`, `load_model()` |

### ModelRunner / Sampler 层

| 组件 | 文件路径 | 核心函数 |
|------|----------|----------|
| GPU Model Runner | `vllm/v1/worker/gpu_model_runner.py` | `execute_model()`, `_preprocess()`, `_model_forward()`, `_sample()` |
| 输入批次管理 | `vllm/v1/worker/gpu_input_batch.py` | `InputBatch` |
| 主采样器 | `vllm/v1/sample/sampler.py` | `Sampler.forward()`, `sample()` |

### 模型实现

| 组件 | 文件路径 |
|------|----------|
| 模型目录 | `vllm/model_executor/models/` |
| Llama 实现 | `vllm/model_executor/models/llama.py` |
| 算子层 | `vllm/model_executor/layers/` |
| CUDA 算子 | `csrc/` |

---

## 四、核心数据结构

### Request
**文件**：`vllm/v1/request.py`

```python
class Request:
    request_id: str              # 唯一请求 ID
    sampling_params: SamplingParams | None
    prompt_token_ids: list[int]  # 输入 token IDs
    max_tokens: int              # 最大输出长度
    arrival_time: float          # 到达时间戳
    status: RequestStatus        # WAITING / RUNNING / FINISHED_*
    output_token_ids: list[int]  # 已生成 token
    block_hashes: list[...]      # KV Cache 块哈希（前缀缓存用）

class RequestStatus(enum.Enum):
    WAITING
    RUNNING
    PREEMPTED
    FINISHED_STOPPED
    FINISHED_LENGTH_CAPPED
    FINISHED_ABORTED
```

### SamplingParams
**文件**：`vllm/sampling_params.py`

```python
class SamplingParams(msgspec.Struct):
    n: int = 1                   # 并行生成序列数
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 16
    stop: list[str]              # 停止字符串
    stop_token_ids: list[int]
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logprobs: int | None = None  # 返回 top-N logprobs
    logit_bias: dict[int, float]
```

### SchedulerOutput
**文件**：`vllm/v1/core/sched/output.py`

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]  # 首次调度的新请求
    scheduled_cached_reqs: CachedRequestData  # 已运行请求的增量更新
    num_scheduled_tokens: dict[str, int]      # req_id → token 数量映射
    total_num_scheduled_tokens: int           # 本次总 token 数
    finished_req_ids: set[str]                # 本次完成的请求
    preempted_req_ids: set[str]               # 被抢占的请求

@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: list[list[int]]    # 分配的 KV Cache 物理块 ID
```

### ModelRunnerOutput
**文件**：`vllm/v1/outputs.py`

```python
@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    sampled_token_ids: list[list[int]]   # [num_reqs, num_generated_tokens]
    logprobs: LogprobsLists | None
```

---

## 五、关键机制说明

### 5.1 PagedAttention / KV Cache 管理

```
物理内存块（Block）：固定大小（默认 16 tokens/block）
    ↓
BlockPool（block_pool.py）：维护所有空闲/已用物理块
    ↓
KVCacheManager.allocate_slots()：
    1. get_computed_blocks() → 前缀缓存命中，复用已有块（Prefix Caching）
    2. 分配新块给新 token
    3. 返回 block_ids 列表
    ↓
block_ids → SchedulerOutput → Worker → BlockTable（GPU 侧）
    ↓
gpu_model_runner._preprocess() 构建：
    slot_mapping：每个 token → 物理内存位置
    block_table：每个 seq → 物理块列表
    ↓
PagedAttention kernel（csrc/）：按 block_table 读写 KV Cache
```

### 5.2 EngineCore 主循环

```python
# vllm/v1/engine/core.py: EngineCoreProc.run_engine_core()
while not shutdown:
    process_input_sockets()         # 处理来自 API Server 的新请求（ZMQ）
    outputs = engine_core.step()
    process_output_sockets(outputs) # 将结果发回 API Server（ZMQ）

# EngineCore.step()
def step():
    scheduler_output = scheduler.schedule()       # ① 调度
    model_output = executor.execute_model(        # ② GPU 执行
        scheduler_output)
    scheduler.update_from_output(                 # ③ 更新状态
        scheduler_output, model_output)
    return engine_core_outputs
```

### 5.3 Sampler 采样流水线

```python
# vllm/v1/sample/sampler.py: Sampler.forward()
def forward(logits, sampling_metadata):
    logits = logits.float()
    apply_logits_processors(logits)    # 自定义 logits 处理器
    apply_penalties(logits)            # 惩罚项（repetition/frequency/presence）
    apply_temperature(logits)          # logits.div_(temperature)  原地操作
    sampled_ids = sample(logits)
        # greedy:  logits.argmax(dim=-1)
        # random:  TopKTopP 过滤 → multinomial 采样
    return SamplerOutput(sampled_ids, logprobs)
```

### 5.4 增量 Detokenize

每次只解码新增 token，避免重复解码历史：

```python
# vllm/v1/engine/detokenizer.py
class FastIncrementalDetokenizer:
    def update(self, new_token_ids):
        self.token_ids.extend(new_token_ids)
        return self.decode_next(new_token_ids)  # 只解码新增部分

    def get_next_output_text(self, delta=True):
        # delta=True：仅返回新增文本（流式输出）
        # delta=False：返回完整文本
```

### 5.5 CUDA Graph 优化

decode 阶段（每步只生成 1 个 token）可以捕获为 CUDA Graph，消除每次 kernel launch 的开销：

```python
# vllm/v1/worker/gpu_model_runner.py
_maybe_capture_cudagraph()  # 首次运行捕获 graph
# 之后 decode 时直接 graph.replay()，而非重新 launch kernels
```

---

## 六、目录结构全景

```
vllm/
├── entrypoints/               # 对外入口
│   ├── openai/
│   │   ├── api_server.py      # FastAPI HTTP 服务（build_app、init_app_state）
│   │   ├── serving_chat.py    # /v1/chat/completions
│   │   └── serving_*.py       # 其他端点（completion、embedding 等）
│   └── llm.py                 # 离线 LLM.generate()
│
├── v1/                        # V1 引擎（当前主要实现）
│   ├── engine/
│   │   ├── core.py            # EngineCore 主循环（调度 + 协调）
│   │   ├── core_client.py     # API Server 到 EngineCore 的 ZMQ 客户端
│   │   ├── async_llm.py       # AsyncLLM（在线服务入口）
│   │   ├── llm_engine.py      # 同步 LLMEngine（离线推理入口）
│   │   ├── input_processor.py # tokenize + 构建 EngineCoreRequest
│   │   ├── output_processor.py# detokenize + stop 判断 + RequestOutput 生成
│   │   └── detokenizer.py     # FastIncrementalDetokenizer
│   │
│   ├── core/
│   │   ├── sched/
│   │   │   ├── scheduler.py   # 核心调度器（FCFS + 抢占 + 前缀缓存）
│   │   │   ├── output.py      # SchedulerOutput、NewRequestData 等数据结构
│   │   │   └── request_queue.py  # 请求队列
│   │   ├── kv_cache_manager.py   # KV Cache 分配（调用 BlockPool）
│   │   └── block_pool.py         # 物理块池
│   │
│   ├── executor/
│   │   ├── multiproc_executor.py # 多进程执行器（WorkerProc 管理）
│   │   ├── uniproc_executor.py   # 单进程执行器（调试用）
│   │   └── ray_executor.py       # Ray 分布式执行器
│   │
│   ├── worker/
│   │   ├── gpu_worker.py         # GPU Worker（加载模型、执行前向）
│   │   ├── gpu_model_runner.py   # GPUModelRunner（最核心）
│   │   ├── gpu_input_batch.py    # InputBatch（输入张量批次管理）
│   │   └── block_table.py        # GPU 侧块表（block_ids → slot_mapping）
│   │
│   ├── sample/
│   │   └── sampler.py            # Sampler（从 logits 采样 token）
│   │
│   ├── attention/                # Attention 后端（FlashAttention 等）
│   ├── request.py                # Request 数据结构
│   ├── outputs.py                # ModelRunnerOutput、SamplerOutput
│   └── kv_cache_interface.py     # KV Cache 抽象接口
│
├── engine/                    # 旧路径（现为 V1 的别名，可忽略）
│   ├── llm_engine.py          # → v1.engine.llm_engine
│   └── async_llm_engine.py    # → v1.engine.async_llm
│
├── model_executor/
│   ├── models/                # 各模型实现（llama.py、qwen2.py 等）
│   ├── layers/                # 通用算子层（attention、linear、sampler 等）
│   └── model_loader/          # 权重加载（safetensors、GGUF 等）
│
├── sampling_params.py         # SamplingParams（用户侧采样参数）
├── outputs.py                 # RequestOutput（用户侧输出）
├── config/                    # VllmConfig 及各子配置
└── csrc/                      # CUDA/C++ 算子（PagedAttention kernel 等）
```

---

## 七、第二块：关键组件深读建议

在串完流程后，按以下顺序深入各组件：

| 优先级 | 组件 | 文件 | 重点关注 |
|--------|------|------|----------|
| ★★★ | **Scheduler** | `v1/core/sched/scheduler.py` | `schedule()` 的完整逻辑：FCFS、KV Cache 分配、preemption 触发条件 |
| ★★★ | **KVCacheManager** | `v1/core/kv_cache_manager.py` | `allocate_slots()`、`get_computed_blocks()`（前缀缓存命中逻辑） |
| ★★★ | **GPUModelRunner** | `v1/worker/gpu_model_runner.py` | `_preprocess()`（block_table/slot_mapping 构建）、`_model_forward()`、`_sample()` |
| ★★☆ | **Sampler** | `v1/sample/sampler.py` | 各种惩罚项计算、TopKTopP 实现、multinomial 采样 |
| ★★☆ | **EngineCore 循环** | `v1/engine/core.py` | busy loop 的结构、ZMQ 收发、step 的三步骤 |
| ★★☆ | **AsyncLLM** | `v1/engine/async_llm.py` | `generate()` 的异步生成器、请求生命周期管理 |
| ★☆☆ | **OutputProcessor** | `v1/engine/output_processor.py` | stop 判断逻辑、增量 detokenize 的触发时机 |
| ★☆☆ | **模型实现** | `model_executor/models/llama.py` | 了解 vLLM 对标准 HuggingFace 模型的适配方式（主要是 Attention 层替换） |
| ★☆☆ | **CUDA Graph** | `v1/worker/gpu_model_runner.py` | `_maybe_capture_cudagraph()`：何时捕获、何时 replay |

### 阅读顺序推荐

```
Step 1：通读流程（离线推理，最简单）
    entrypoints/llm.py → LLM.generate()

Step 2：理解调度（最核心）
    v1/core/sched/scheduler.py     # schedule()，重点读这 ~300 行
    v1/core/kv_cache_manager.py    # allocate_slots()，PagedAttention 的灵魂

Step 3：理解执行（GPU 上发生了什么）
    v1/worker/gpu_model_runner.py  # execute_model()

Step 4：理解采样
    v1/sample/sampler.py           # forward()，logits → token 全过程

Step 5：理解在线服务并发
    v1/engine/core.py              # busy loop 主循环
    v1/executor/multiproc_executor.py  # 多进程通信机制
```

### 首轮可跳过的部分

| 目录/文件 | 原因 |
|-----------|------|
| `vllm/v1/executor/ray_executor.py` | Ray 分布式，硬件适配 |
| `vllm/v1/worker/cpu_worker.py` / `xpu_worker.py` | 非 GPU 适配 |
| `vllm/v1/engine/coordinator.py` | 数据并行协调，DP > 1 才用 |
| `vllm/model_executor/layers/` 下量化相关 | AWQ、GPTQ kernel 等 |
| `csrc/` | CUDA kernel 实现，先看 Python 层接口 |
| `vllm/multimodal/` | 多模态处理 |
| `vllm/spec_decode/` | 投机解码，理解主流程后再看 |

---

## 八、重要设计决策备注

| 设计 | 说明 |
|------|------|
| **V1 架构** | 0.6+ 版本全面迁移到 `vllm/v1/`，`vllm/engine/` 中的文件大多是别名，直接读 V1 |
| **ZMQ IPC** | API Server 和 EngineCore 之间通过 ZMQ socket 通信，用 protobuf 序列化，支持跨机器 |
| **busy loop** | EngineCore 不用 asyncio event loop，而是 busy loop 轮询，减少调度延迟 |
| **CUDA Graph** | decode 阶段（batch size 固定）强制走 CUDA Graph，prefill 不走 |
| **前缀缓存** | `get_computed_blocks()` 通过哈希匹配已有 KV 块，命中则直接复用，无需重新计算 |
| **增量 detokenize** | 每步只 decode 新增 token，避免 O(n²) 的重复解码开销 |
| **连续批处理** | Scheduler 每步都可以加入新请求、移除完成请求，不需要等一个 batch 全部完成 |

