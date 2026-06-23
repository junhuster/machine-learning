# vLLM v0.18.0 源码导读

> **版本**：v0.18.0（releases/v0.18.0 分支）
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
│  vllm/v1/engine/detokenizer.py      增量反 Tokenize     │
│  vllm/v1/engine/core_client.py      与 EngineCore 通信  │
└────────────────────────┬────────────────────────────────┘
                         │ ZMQ IPC（msgpack 序列化）
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
│  vllm/v1/executor/ray_executor.py        Ray 执行器     │
│  vllm/v1/worker/gpu_worker.py            AsyncGPUWorker │
└────────────────────────┬────────────────────────────────┘
                         │ CUDA / NCCL
┌────────────────────────▼────────────────────────────────┐
│                 ModelRunner / Model 层                   │
│  vllm/v1/worker/gpu_model_runner.py      GPUModelRunner │
│  vllm/model_executor/models/             模型实现        │
│  vllm/v1/sample/sampler.py              Sampler         │
└─────────────────────────────────────────────────────────┘
```

### 多进程架构

vLLM V1 将系统拆分为多个进程，进程间通过 **ZMQ IPC + msgpack**（API Server ↔ EngineCore）和**共享内存 MessageQueue**（EngineCore ↔ Worker）通信：

| 进程 | 数量 | 职责 |
|------|------|------|
| API Server | 1 | HTTP 接收、Tokenize、输出流管理 |
| EngineCore | 1（DP 时多个） | 调度、KV Cache 管理、协调 GPU |
| GPU Worker | TP 并行度个 | 加载权重、执行模型前向 |

> **v0.18.0 新增**：GPU Worker 改为 `AsyncGPUWorker`，支持异步执行；同时新增 `DPCoordinator` 支持 Data Parallel 模式。

---

## 二、完整请求调用链

### 第一块：流程串联（先读这里）

以下是一个请求从进入到输出的完整路径，**忽略多硬件适配、LoRA、多模态、投机解码等旁支**，只看主干。

```
客户端 POST /v1/chat/completions
    │
    ▼
【① Entrypoint — API Server 进程】
vllm/entrypoints/openai/api_server.py
  build_app()                              # 构建 FastAPI App
  init_app_state()                         # 初始化 engine_client 等状态
  build_async_engine_client()              # 创建 AsyncLLM 实例

vllm/entrypoints/openai/engine/serving.py
  OpenAIServing.chat_completion()          # 处理 chat 请求
    ├─ 解析 ChatCompletionRequest
    ├─ tokenize（通过 engine_client.tokenize）
    └─ 调用 engine_client.generate()
    │
    ▼
vllm/v1/engine/async_llm.py
  AsyncLLM.generate()                      # 主异步生成器
    ├─ _add_request()
    │    ├─ InputProcessor.process_inputs()  # tokenize、构建 EngineCoreRequest
    │    └─ EngineCoreClient.add_request()   # ZMQ 发送给 EngineCore 进程
    └─ 持续 yield RequestOutput（流式返回）
    │
    │  [ZMQ IPC 跨进程，msgpack 序列化]
    ▼
【② EngineCore — 独立进程，主调度循环】
vllm/v1/engine/core.py
  EngineCoreProc.run_engine_core()         # busy loop 主循环
    └─ EngineCore.step()
         ├─ Scheduler.schedule()           # 调度决策
         ├─ Executor.execute_model(sched_output)   # 执行模型（异步）
         ├─ Executor.sample_tokens(...)    # 采样（v0.18 新增独立步骤）
         └─ Scheduler.update_from_output() # 更新状态
    │
    ▼
vllm/v1/core/sched/scheduler.py
  Scheduler.schedule()
    ├─ 遍历 running 请求，为新 token 分配 KV Cache slots
    ├─ KVCacheManager.allocate_slots()     # 分配物理块
    ├─ 从 waiting 队列调度新请求
    ├─ 必要时触发 preemption（抢占 running 请求）
    └─ 返回 SchedulerOutput（含 block_ids、token counts 等）
    │
    │  [共享内存 MessageQueue]
    ▼
【③ Executor/Worker — GPU 进程】
vllm/v1/executor/multiproc_executor.py
  MultiprocExecutor.execute_model(scheduler_output)
    └─ collective_rpc("execute_model")     # 广播到所有 Worker

vllm/v1/executor/multiproc_executor.py
  WorkerProc.worker_busy_loop()            # Worker 进程事件循环
    └─ worker.execute_model()

vllm/v1/worker/gpu_worker.py
  AsyncGPUWorker.execute_model(scheduler_output)
    └─ model_runner.execute_model(scheduler_output)
    │
    ▼
【④ GPUModelRunner — 核心执行层】
vllm/v1/worker/gpu_model_runner.py
  GPUModelRunner.execute_model(scheduler_output)
    ├─ _update_states()                    # 更新请求状态缓存
    ├─ _prepare_inputs()                   # 构建输入 tensor
    │    ├─ 构建 input_ids、position_ids
    │    └─ _build_attention_metadata()    # block_table、slot_mapping
    ├─ _model_forward()                    # 神经网络前向传播
    │    └─ model.forward(input_ids, positions, attn_metadata, ...)
    │         # PagedAttention 按 block_table 读写 KV Cache
    └─ 返回 ModelRunnerOutput（logits 等，采样由独立步骤完成）

  # v0.18.0：采样被拆成独立步骤
  GPUModelRunner.sample_tokens(sampler, sampler_output, scheduler_output)
    └─ Sampler.forward(logits, sampling_metadata)
    │
    ▼
【⑤ Sampler】
vllm/v1/sample/sampler.py
  Sampler.forward(logits, sampling_metadata)
    ├─ apply_logits_processors()           # 自定义处理器
    ├─ apply_penalties()                   # repetition/frequency/presence penalty
    ├─ apply_temperature()                 # logits.div_(temperature)
    ├─ sample()
    │    ├─ greedy → greedy_sample() → argmax(dim=-1)
    │    └─ random → TopKTopPSampler → multinomial
    └─ 返回 SamplerOutput（sampled_token_ids, logprobs_tensors）
    │
    │  [结果逐层返回]
    ▼
【⑥ 输出处理 — API Server 进程】
vllm/v1/engine/output_processor.py
  OutputProcessor.process_request_output(engine_core_outputs)
    ├─ RequestOutputCollector.put()        # 收集各请求的新 token
    ├─ IncrementalDetokenizer.update()     # 增量 detokenize（只解码新 token）
    ├─ 检查 stop strings / stop token ids
    ├─ 检查 max_tokens 限制
    └─ 生成 RequestOutput → yield 给 AsyncLLM.generate()

vllm/entrypoints/openai/engine/serving.py
    └─ 格式化为 ChatCompletionStreamResponse → SSE 流 → 客户端
```

### 离线批量推理入口

```
vllm/entrypoints/llm.py
  LLM.generate(prompts, sampling_params)
    └─ _run_engine()        # 同步循环直到所有请求完成
         └─ engine.step()   # 反复调用

vllm/v1/engine/llm_engine.py  # LLMEngine 类（向后兼容封装）
  LLMEngine.step()
    ├─ engine_core_client.get_output()
    └─ output_processor.process_request_output()
```

---

## 三、关键源码文件速查

### Engine 层

| 组件 | 文件路径 | 核心类/函数 |
|------|----------|------------|
| 同步引擎 | `vllm/v1/engine/llm_engine.py` | `LLMEngine`（向后兼容封装） |
| 异步引擎 | `vllm/v1/engine/async_llm.py` | `AsyncLLM.generate()`, `_add_request()` |
| 引擎核心 | `vllm/v1/engine/core.py` | `EngineCore.step()`, `EngineCoreProc.run_engine_core()` |
| 核心客户端 | `vllm/v1/engine/core_client.py` | `EngineCoreClient`、`InprocClient`、`MPClient` |
| 输入处理 | `vllm/v1/engine/input_processor.py` | `InputProcessor.process_inputs()` |
| 输出处理 | `vllm/v1/engine/output_processor.py` | `OutputProcessor.process_request_output()`, `RequestOutputCollector` |
| 增量反 Tokenize | `vllm/v1/engine/detokenizer.py` | `FastIncrementalDetokenizer.update()` |
| 并行采样 | `vllm/v1/engine/parallel_sampling.py` | `ParentRequest`（n>1 时使用） |
| DP 协调器 | `vllm/v1/engine/coordinator.py` | `DPCoordinator`, `DPCoordinatorProc` |

> `vllm/engine/llm_engine.py` 和 `vllm/engine/async_llm_engine.py` 都是 8 行的别名文件，直接指向 V1 实现。

### Scheduler 层

| 组件 | 文件路径 | 核心类/函数 |
|------|----------|------------|
| 主调度器 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule()`, `update_from_output()` |
| 调度器接口 | `vllm/v1/core/sched/interface.py` | `SchedulerInterface`（抽象基类） |
| 调度输出 | `vllm/v1/core/sched/output.py` | `SchedulerOutput`, `NewRequestData`, `CachedRequestData` |
| 请求队列 | `vllm/v1/core/sched/request_queue.py` | FCFS / 优先级队列 |
| 异步调度器 | `vllm/v1/core/sched/async_scheduler.py` | `AsyncScheduler` |

### KV Cache 层

| 组件 | 文件路径 | 核心类/函数 |
|------|----------|------------|
| KV Cache 管理 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots()`, `free()`, `get_computed_blocks()`, `cache_blocks()` |
| KV Cache 协调 | `vllm/v1/core/kv_cache_coordinator.py` | `KVCacheCoordinator` |
| 单类型管理器 | `vllm/v1/core/single_type_kv_cache_manager.py` | 统一块类型的管理器 |
| 块池 | `vllm/v1/core/block_pool.py` | `BlockHashToBlockMap`（前缀缓存哈希映射） |
| GPU 侧块表 | `vllm/v1/worker/block_table.py` | GPU 侧块表管理 |
| KV Cache 工具 | `vllm/v1/core/kv_cache_utils.py` | 哈希计算、块操作工具 |

### Executor / Worker 层

| 组件 | 文件路径 | 核心类/函数 |
|------|----------|------------|
| 抽象基类 | `vllm/v1/executor/abstract.py` | `Executor`（ABC） |
| 多进程执行器 | `vllm/v1/executor/multiproc_executor.py` | `MultiprocExecutor.execute_model()`, `WorkerProc.worker_busy_loop()` |
| 单进程执行器 | `vllm/v1/executor/uniproc_executor.py` | `UniProcExecutor`（调试用） |
| Ray 执行器 | `vllm/v1/executor/ray_executor.py` | `RayDistributedExecutor`（多机） |
| GPU Worker | `vllm/v1/worker/gpu_worker.py` | `AsyncGPUWorker.execute_model()` |
| Worker 基类 | `vllm/v1/worker/worker_base.py` | `WorkerBase` |

### ModelRunner / Sampler 层

| 组件 | 文件路径 | 核心类/函数 |
|------|----------|------------|
| GPU Model Runner | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model()`, `sample_tokens()`, `_build_attention_metadata()`, `_model_forward()` |
| 输入批次 | `vllm/v1/worker/gpu_input_batch.py` | `InputBatch`（输入张量批次管理） |
| 微批次包装 | `vllm/v1/worker/gpu_ubatch_wrapper.py` | `GpuUbatchWrapper`（流水线并行用） |
| 主采样器 | `vllm/v1/sample/sampler.py` | `Sampler.forward()`, `greedy_sample()`, `sample()` |
| 拒绝采样 | `vllm/v1/sample/rejection_sampler.py` | `RejectionSampler`（投机解码用） |

### 模型实现

| 组件 | 文件路径 |
|------|----------|
| 模型目录 | `vllm/model_executor/models/` |
| Llama 实现 | `vllm/model_executor/models/llama.py` |
| 算子层 | `vllm/model_executor/layers/` |
| 模型加载 | `vllm/model_executor/model_loader/` |
| CUDA 算子 | `csrc/` |

---

## 四、核心数据结构

### Request
**文件**：`vllm/v1/request.py`

```python
class Request:
    request_id: str
    client_index: int
    priority: int
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    lora_request: LoRARequest | None
    arrival_time: float
    status: RequestStatus              # 请求当前状态
    max_tokens: int                    # 最大输出长度
    prompt_token_ids: list[int] | None # 输入 token IDs
    prompt_embeds: torch.Tensor | None # 预计算嵌入（可选）
    _output_token_ids: list[int]       # 已生成 token（内部）
    _all_token_ids: list[int]          # prompt + output 全部 token
    num_computed_tokens: int           # 已完成计算的 token 数
    block_hashes: list[BlockHash]      # KV Cache 块哈希（前缀缓存用）
    num_cached_tokens: int             # 前缀缓存命中的 token 数
    num_preemptions: int               # 被抢占次数
    resumable: bool                    # 是否支持断点续传

class RequestStatus(enum.IntEnum):
    WAITING                    # 等待调度
    WAITING_FOR_FSM            # 等待结构化输出状态机
    WAITING_FOR_REMOTE_KVS     # 等待远程 KV 传输
    WAITING_FOR_STREAMING_REQ  # 等待流式输入
    RUNNING                    # 正在执行
    PREEMPTED                  # 被抢占
    FINISHED_STOPPED           # 遇到 stop string/token 结束
    FINISHED_LENGTH_CAPPED     # 达到 max_tokens 结束
    FINISHED_ABORTED           # 被中止
    FINISHED_IGNORED           # 被忽略（如全是特殊 token）
    FINISHED_ERROR             # 出错
    FINISHED_REPETITION        # 触发重复检测结束
```

### EngineCoreRequest
**文件**：`vllm/v1/engine/__init__.py`（通过 ZMQ 在进程间传递）

```python
class EngineCoreRequest(msgspec.Struct):
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None
    client_index: int = 0
    current_wave: int = 0
    priority: int = 0
    resumable: bool = False
    external_req_id: str | None = None
```

### SamplingParams
**文件**：`vllm/sampling_params.py`

```python
class SamplingParams(msgspec.Struct, PydanticMsgspecMixin):
    n: int = 1                          # 并行生成序列数
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    max_tokens: int | None = 16
    min_tokens: int = 0
    stop: str | list[str] | None = None  # 停止字符串
    stop_token_ids: list[int] | None = None
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logprobs: int | None = None          # 返回 top-N logprobs
    prompt_logprobs: int | None = None
    logit_bias: dict[int, float] | None = None
    allowed_token_ids: list[int] | None = None
    structured_outputs: StructuredOutputsParams | None = None
    repetition_detection: RepetitionDetectionParams | None = None
    output_kind: RequestOutputKind = CUMULATIVE  # 累积/增量/仅最终
    bad_words: list[str] | None = None
```

### SchedulerOutput
**文件**：`vllm/v1/core/sched/output.py`

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]   # 首次调度的新请求
    scheduled_cached_reqs: CachedRequestData   # 已运行请求的增量更新
    num_scheduled_tokens: dict[str, int]       # req_id → 本次 token 数
    total_num_scheduled_tokens: int            # 本次总 token 数
    scheduled_spec_decode_tokens: dict[str, list[int]]  # 投机解码 token
    scheduled_encoder_inputs: dict[str, list[int]]      # 视觉 encoder 输入
    num_common_prefix_blocks: list[int]        # 公共前缀块数（DP 用）
    finished_req_ids: set[str]                 # 本次完成的请求
    preempted_req_ids: set[str] | None         # 被抢占的请求
    has_structured_output_requests: bool       # 是否有结构化输出请求
    new_block_ids_to_zero: list[int] | None    # 需要清零的新块 ID

@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]    # 分配的 KV Cache 物理块 ID（按层）
    num_computed_tokens: int            # 前缀缓存命中的 token 数
    lora_request: LoRARequest | None
    prompt_embeds: torch.Tensor | None

@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_req_ids: set[str]           # 从 PREEMPTED 恢复的请求
    new_token_ids: list[list[int]]      # 新增 token（投机解码用）
    all_token_ids: dict[str, list[int]] # 所有 token（含 prompt）
    new_block_ids: list[tuple[list[int], ...] | None]  # 新分配块
    num_computed_tokens: list[int]      # 已计算 token 数
    num_output_tokens: list[int]        # 已输出 token 数
```

### ModelRunnerOutput / SamplerOutput
**文件**：`vllm/v1/outputs.py`

```python
@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]        # [num_reqs, num_generated_tokens]
    logprobs: LogprobsLists | None
    prompt_logprobs_dict: dict[str, LogprobsTensors | None]
    pooler_output: list[torch.Tensor | None] | None  # embedding 任务输出
    kv_connector_output: KVConnectorOutput | None    # 分离式推理用
    num_nans_in_logits: dict[str, int] | None
    cudagraph_stats: CUDAGraphStat | None

@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor    # GPU 上的采样结果
    logprobs_tensors: LogprobsTensors | None

@dataclass
class DraftTokenIds:
    req_ids: list[str]
    draft_token_ids: list[list[int]]   # 投机解码的 draft token
```

### EngineCoreOutput（进程间通信）
**文件**：`vllm/v1/engine/__init__.py`

```python
class EngineCoreOutput(msgspec.Struct):
    request_id: str
    new_token_ids: list[int]           # 本次新生成的 token
    new_logprobs: LogprobsLists | None
    new_prompt_logprobs_tensors: LogprobsTensors | None
    finish_reason: FinishReason | None # STOP/LENGTH/ABORT/ERROR/REPETITION
    stop_reason: int | str | None      # 具体的 stop token 或 stop string

class FinishReason(enum.IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4
```

---

## 五、关键机制说明

### 5.1 PagedAttention / KV Cache 管理

```
物理内存块（Block）：固定大小（默认 16 tokens/block）
    ↓
BlockPool（block_pool.py）：维护所有空闲/已用物理块
BlockHashToBlockMap：哈希 → 物理块（前缀缓存查找）
    ↓
KVCacheManager.allocate_slots()：
    1. get_computed_blocks()  → 前缀缓存命中检查（哈希匹配）
    2. 分配新块给新 token
    3. 返回 block_ids，注入 SchedulerOutput
    ↓
block_ids → Worker → block_table.py（GPU 侧块表）
    ↓
GPUModelRunner._build_attention_metadata()：
    slot_mapping：每个 token → 物理内存槽位
    block_table：每个 seq → 物理块列表
    ↓
PagedAttention kernel（csrc/）：按 block_table 读写 KV Cache
    ↓
new_block_ids_to_zero：新分配的块在使用前清零（防止 NaN）
```

### 5.2 EngineCore 主循环

```python
# vllm/v1/engine/core.py: EngineCoreProc.run_engine_core()
while not shutdown:
    process_input_sockets()         # 处理来自 API Server 的新请求/中止（ZMQ）
    outputs, model_executed = engine_core.step()
    post_step(model_executed)       # 调度后处理
    process_output_sockets(outputs) # 将结果发回 API Server（ZMQ）

# EngineCore.step()
def step() -> tuple[dict[int, EngineCoreOutputs], bool]:
    scheduler_output = scheduler.schedule()          # ① 调度
    model_output = executor.execute_model(           # ② GPU 前向（异步）
        scheduler_output)
    executor.sample_tokens(sampler, model_output)    # ③ 采样（独立步骤）
    scheduler.update_from_output(                    # ④ 更新调度状态
        scheduler_output, model_output)
    return engine_core_outputs, model_executed
```

> **v0.18.0 变化**：`execute_model` 和 `sample_tokens` 被拆分为两个独立 RPC 调用，支持 Pipeline Parallel 场景下 prefill/decode 阶段的分离。

### 5.3 Sampler 采样流水线

```python
# vllm/v1/sample/sampler.py: Sampler.forward()
def forward(logits, sampling_metadata,
            predict_bonus_token=False,
            logprobs_mode_override=None) -> SamplerOutput:
    apply_logits_processors(logits)    # 自定义 logits 处理器
    apply_penalties(logits)            # presence/frequency/repetition penalty
    apply_temperature(logits)          # logits.div_(temperature)  原地操作
    sampled_ids = sample(logits)
        # greedy:  greedy_sample() → argmax(dim=-1)
        # random:  TopKTopPSampler → multinomial
    logprobs = gather_logprobs(...)    # 收集 logprobs（可选）
    return SamplerOutput(sampled_ids, logprobs_tensors)
```

### 5.4 增量 Detokenize

```python
# vllm/v1/engine/detokenizer.py
class FastIncrementalDetokenizer:
    def update(self, new_token_ids: list[int]) -> str | None:
        # 只 decode 新增 token，避免 O(n²) 的重复解码开销
        self.token_ids.extend(new_token_ids)
        return self.decode_next(new_token_ids)
```

### 5.5 CUDA Graph 优化

```python
# vllm/v1/worker/gpu_model_runner.py
_maybe_capture_cudagraph()   # decode 阶段：首次运行时捕获 CUDA Graph
# 后续 decode：graph.replay() 而非重新 launch kernels，消除 CPU 开销

# vllm/v1/cudagraph_dispatcher.py
class CudagraphDispatcher:   # v0.18.0 新增：统一管理不同 batch size 的 CUDA Graph
    # 根据实际 batch size 选择合适的 graph replay
```

### 5.6 EngineCoreClient 选择逻辑

```python
# vllm/v1/engine/core_client.py: make_client()
if 单进程模式:
    return InprocClient(engine_core)    # 直接调用，无 IPC 开销（调试/单卡）
elif 多进程模式:
    return MPClient(...)                # ZMQ + 独立 EngineCore 进程（生产用）
```

---

## 六、目录结构全景

```
vllm/
├── entrypoints/                     # 对外入口
│   ├── openai/
│   │   ├── api_server.py            # FastAPI HTTP 服务（build_app、init_app_state）
│   │   ├── engine/
│   │   │   └── serving.py          # OpenAIServing 基类（ServeContext）
│   │   ├── chat_completion/         # /v1/chat/completions
│   │   ├── completion/              # /v1/completions
│   │   └── serve/                   # 扩展服务（disagg、RLHF 等）
│   ├── anthropic/                   # Anthropic API 兼容
│   ├── mcp/                         # MCP 工具服务器
│   └── llm.py                       # 离线 LLM.generate()
│
├── v1/                              # V1 引擎（核心实现）
│   ├── engine/
│   │   ├── core.py                  # EngineCore、EngineCoreProc（最核心，2000+行）
│   │   ├── core_client.py           # EngineCoreClient、InprocClient、MPClient
│   │   ├── async_llm.py             # AsyncLLM（在线服务入口）
│   │   ├── llm_engine.py            # LLMEngine（离线入口、向后兼容）
│   │   ├── input_processor.py       # tokenize + 构建 EngineCoreRequest
│   │   ├── output_processor.py      # RequestOutputCollector + OutputProcessor
│   │   ├── detokenizer.py           # Fast/SlowIncrementalDetokenizer
│   │   ├── logprobs.py              # LogprobsProcessor
│   │   ├── coordinator.py           # DPCoordinator（Data Parallel 协调）
│   │   ├── parallel_sampling.py     # ParentRequest（n>1 并行采样）
│   │   ├── utils.py                 # CoreEngineProcManager、EngineZmqAddresses
│   │   └── __init__.py              # EngineCoreRequest/Output/FinishReason 等定义
│   │
│   ├── core/
│   │   ├── sched/
│   │   │   ├── scheduler.py         # 核心调度器（FCFS + 抢占 + 前缀缓存）
│   │   │   ├── async_scheduler.py   # 异步调度器
│   │   │   ├── interface.py         # SchedulerInterface（抽象基类）
│   │   │   ├── output.py            # SchedulerOutput、NewRequestData、CachedRequestData
│   │   │   └── request_queue.py     # 请求队列（优先级支持）
│   │   ├── kv_cache_manager.py      # KVCacheManager（allocate_slots、cache_blocks）
│   │   ├── kv_cache_coordinator.py  # KVCacheCoordinator（多类型协调）
│   │   ├── single_type_kv_cache_manager.py  # 单类型 KV 缓存管理器
│   │   ├── block_pool.py            # BlockHashToBlockMap（前缀缓存哈希查找）
│   │   ├── encoder_cache_manager.py # 视觉 encoder 缓存管理
│   │   └── kv_cache_utils.py        # 块哈希计算、工具函数（66KB）
│   │
│   ├── executor/
│   │   ├── abstract.py              # Executor ABC（定义接口）
│   │   ├── multiproc_executor.py    # MultiprocExecutor + WorkerProc
│   │   ├── uniproc_executor.py      # UniProcExecutor（调试/单卡）
│   │   └── ray_executor.py          # RayDistributedExecutor（多机）
│   │
│   ├── worker/
│   │   ├── gpu_worker.py            # AsyncGPUWorker（加载模型、执行前向）
│   │   ├── gpu_model_runner.py      # GPUModelRunner（最核心，295KB！）
│   │   ├── gpu_input_batch.py       # InputBatch（输入张量批次管理）
│   │   ├── gpu_ubatch_wrapper.py    # GpuUbatchWrapper（PP 微批次）
│   │   ├── block_table.py           # GPU 侧块表（block_ids → slot_mapping）
│   │   ├── worker_base.py           # WorkerBase（基类）
│   │   └── workspace.py             # 工作区内存管理
│   │
│   ├── sample/
│   │   ├── sampler.py               # Sampler（greedy/topk/topp）
│   │   ├── rejection_sampler.py     # RejectionSampler（投机解码用）
│   │   ├── metadata.py              # SamplingMetadata
│   │   ├── ops/                     # TopKTopP 等采样操作
│   │   └── logits_processor/        # Logits 处理器
│   │
│   ├── attention/                   # Attention 后端（FlashAttention 等）
│   ├── cudagraph_dispatcher.py      # CudagraphDispatcher（v0.18.0 新增）
│   ├── request.py                   # Request、RequestStatus
│   ├── outputs.py                   # ModelRunnerOutput、SamplerOutput、DraftTokenIds
│   ├── kv_cache_interface.py        # KVCacheSpec 等 KV Cache 抽象接口
│   └── serial_utils.py              # MsgpackEncoder/Decoder（ZMQ 序列化）
│
├── engine/                          # 旧路径（别名，可忽略）
│   ├── llm_engine.py                # → v1.engine.llm_engine（8 行别名）
│   ├── async_llm_engine.py          # → v1.engine.async_llm（8 行别名）
│   ├── protocol.py                  # EngineClient 协议定义
│   └── arg_utils.py                 # EngineArgs / AsyncEngineArgs（2200+行）
│
├── model_executor/
│   ├── models/                      # 各模型实现（llama.py、qwen2.py 等）
│   ├── layers/                      # 通用算子层（attention、linear 等）
│   └── model_loader/                # 权重加载（safetensors、GGUF 等）
│
├── sampling_params.py               # SamplingParams、StructuredOutputsParams
├── outputs.py                       # CompletionOutput、RequestOutput（用户侧）
├── sequence.py                      # IntermediateTensors（PP 中间张量）
├── config/                          # VllmConfig 及各子配置
└── csrc/                            # CUDA/C++ 算子（PagedAttention kernel 等）
```

---

## 七、第二块：关键组件深读建议

### 按优先级排序

| 优先级 | 组件 | 文件 | 重点关注 |
|--------|------|------|----------|
| ★★★ | **Scheduler** | `v1/core/sched/scheduler.py` | `schedule()` 的完整逻辑：FCFS、KV Cache 分配、preemption 触发、`update_from_output()` |
| ★★★ | **KVCacheManager** | `v1/core/kv_cache_manager.py` | `allocate_slots()`（分配物理块）、`get_computed_blocks()`（前缀缓存命中）、`cache_blocks()`（写入哈希） |
| ★★★ | **GPUModelRunner** | `v1/worker/gpu_model_runner.py` | `_update_states()`、`_build_attention_metadata()`（block_table/slot_mapping）、`_model_forward()`、`sample_tokens()` |
| ★★☆ | **EngineCore** | `v1/engine/core.py` | `step()` 的四步结构、busy loop、ZMQ 收发、`execute_model` + `sample_tokens` 的拆分 |
| ★★☆ | **Sampler** | `v1/sample/sampler.py` | `forward()`、惩罚项计算、TopKTopP 实现、`greedy_sample()` |
| ★★☆ | **AsyncLLM** | `v1/engine/async_llm.py` | `generate()` 异步生成器、请求生命周期、`_add_request()` |
| ★★☆ | **OutputProcessor** | `v1/engine/output_processor.py` | `RequestOutputCollector.put()/get()`、stop 判断、增量 detokenize 触发 |
| ★☆☆ | **EngineCoreClient** | `v1/engine/core_client.py` | `InprocClient`（单进程路径）vs `MPClient`（ZMQ 路径）的选择逻辑 |
| ★☆☆ | **模型实现** | `model_executor/models/llama.py` | vLLM 对 HuggingFace 模型的适配方式（主要是 Attention 层替换） |
| ★☆☆ | **CUDA Graph** | `v1/cudagraph_dispatcher.py` | v0.18.0 新增的 CUDA Graph 分发器，管理不同 batch size 的 graph |

### 阅读顺序推荐

```
Step 1：通读流程（离线推理，最简单）
    entrypoints/llm.py → LLM.generate()    # 5 分钟了解整体

Step 2：理解调度（最核心）
    v1/core/sched/scheduler.py             # schedule()，读 ~300 行核心逻辑
    v1/core/kv_cache_manager.py            # allocate_slots()，PagedAttention 的灵魂

Step 3：理解执行（GPU 上发生了什么）
    v1/worker/gpu_model_runner.py          # execute_model() + sample_tokens()

Step 4：理解采样
    v1/sample/sampler.py                   # forward()，logits → token 全过程

Step 5：理解并发与进程间通信
    v1/engine/core.py                      # busy loop 主循环
    v1/executor/multiproc_executor.py      # WorkerProc 多进程机制
```

### 首轮可跳过的部分

| 目录/文件 | 原因 |
|-----------|------|
| `vllm/v1/executor/ray_executor.py` | Ray 分布式，多机场景才用 |
| `vllm/v1/worker/cpu_worker.py` | 非 GPU 适配 |
| `vllm/v1/engine/coordinator.py` | Data Parallel 协调，DP > 1 才用 |
| `vllm/v1/sample/rejection_sampler.py` | 投机解码拒绝采样，理解主流程后再看 |
| `vllm/model_executor/layers/` 下量化相关 | AWQ、GPTQ kernel 等 |
| `vllm/multimodal/` | 多模态处理 |
| `csrc/` | CUDA kernel 实现，先看 Python 层接口 |
| `vllm/entrypoints/anthropic/` | Anthropic API 兼容，非核心路径 |

---

## 八、v0.18.0 相对 v0.9.x 的主要变化

| 变化点 | 旧版（v0.9.x） | v0.18.0 |
|--------|--------------|---------|
| **Worker 类** | `AsyncGPUWorker`（基本） | `AsyncGPUWorker`（增强，支持异步流水线） |
| **采样步骤** | `execute_model` 内含采样 | `execute_model` + `sample_tokens` 拆分（独立 RPC） |
| **CUDA Graph** | `_maybe_capture_cudagraph` | 新增 `CudagraphDispatcher` 统一管理 |
| **请求状态** | 较少状态 | 增加 `WAITING_FOR_FSM`、`WAITING_FOR_STREAMING_REQ`、`FINISHED_REPETITION` 等 |
| **KV Cache** | 基础块管理 | 新增 `KVCacheCoordinator`、`new_block_ids_to_zero`（安全清零） |
| **重复检测** | 无 | `RepetitionDetectionParams`（检测生成重复） |
| **分离式推理** | 无 | `KVConnectorOutput`、`ECConnectorOutput`（prefill/decode 分离） |
| **DP 支持** | 基础 | `DPCoordinator`、`DPEngineCoreProc`、`DPMoEEngineCoreActor` |
| **Entrypoint** | `serving_chat.py`（直接） | `OpenAIServing`（基类）+ `ServeContext`（泛型上下文） |

---

## 九、重要设计决策备注

| 设计 | 说明 |
|------|------|
| **V1 架构** | 代码主体在 `vllm/v1/`，`vllm/engine/` 只是 8 行别名，直接读 V1 |
| **ZMQ IPC** | API Server 和 EngineCore 通过 ZMQ + msgpack 通信，`MsgpackEncoder/Decoder` 在 `v1/serial_utils.py` |
| **busy loop** | EngineCore 不用 asyncio event loop，而是 busy loop 轮询，减少调度延迟 |
| **execute + sample 拆分** | v0.18.0 起将前向传播和采样分为两个独立 RPC，支持 PP 场景的流水线 |
| **CUDA Graph** | decode 阶段（batch size 固定）走 CUDA Graph replay，prefill 不走；`CudagraphDispatcher` 管理多个 graph |
| **前缀缓存** | `get_computed_blocks()` 通过 `BlockHashToBlockMap` 哈希查找，命中则直接复用，减少重复计算 |
| **新块清零** | `new_block_ids_to_zero`：新分配的块在填入 KV 前清零，防止上一个请求残留数据引起 NaN |
| **增量 detokenize** | 每步只 decode 新增 token，避免 O(n²) 开销；`FastIncrementalDetokenizer` 是主路径 |
| **连续批处理** | Scheduler 每步可加入新请求、移除完成请求，不需要等一个 batch 全部完成 |
| **resumable 请求** | 支持断点续传：被抢占的请求可以从已完成的 token 处恢复，不需要从头 prefill |
