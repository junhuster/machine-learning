# OneTrans 推荐模型 — 性能分析工程

字节跳动单 Transformer 精排模型：**用一个 Transformer 同时做特征交互 + 序列建模**，替代 DIN（序列建模）+ DCN（特征交叉）的分离范式。

所有特征（用户、物品、上下文、历史行为序列）都被 token 化，加上 type embedding 和 position embedding，送入单个 Transformer Encoder，最终用 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 输出做 CTR 预测。

本工程包含：fake数据生成、模型训练、推理压测、PyTorch Profiler、Nsight System、Nsight Compute 全套性能分析流程。

---

## 目录结构

```
02_onetrans/
├── config.yaml          # 模型/训练/压测/性能分析配置（调参数改这里）
├── fake_data.py         # fake数据生成（1000条样本）
├── model.py             # OneTrans 单Transformer网络结构
├── train.py             # 训练脚本（NVTX标记、显式H2D、GPU利用率、ETA日志、模型保存）
├── inference.py         # 推理压测（端到端延迟拆解、显式H2D/D2H、CPU预处理模拟、batch sweep）
├── benchmark_op.py      # 微基准：torch.utils.benchmark测各类算子+Roofline分析
├── profile_torch.py     # PyTorch Profiler 分析
├── profile_ncu.py       # Nsight Compute 专用短脚本
├── requirements.txt     # Python依赖
├── README.md            # 本文档
│
│ 【运行后自动生成的日志文件（所有脚本只写日志文件，不输出到控制台）】
├── fake_data.log        # fake数据生成日志
├── train.log            # 训练日志（loss/step/耗时/ETA）
├── inference.log        # 推理压测日志（QPS/延迟/各阶段占比）
├── benchmark.log        # 微基准日志（算子耗时/Roofline）
├── profile_torch.log    # torch.profiler日志
├── profile_ncu.log      # ncu短脚本日志
├── fake_data.pkl        # fake训练数据（1000条）
├── model.pt             # 训练保存的模型文件
├── prof_log/            # torch.profiler的TensorBoard输出
├── trace_chrome.json    # torch.profiler的Chrome trace输出
├── nsys_*.nsys-rep      # Nsight System采集报告
└── ncu_*.ncu-rep        # Nsight Compute采集报告
```

> **日志说明**：所有脚本的输出都写入对应 `.log` 文件（append模式，多次运行会追加），控制台不会打印任何日志。查看日志用 `tail -f train.log` 或 `cat train.log`。

---

## 性能分析点对照清单（对照《推荐模型演进与性能优化精讲》PART 2）

> 用性能分析工具导出数据后，对着下表逐项对照分析。每个点都能在本工程中找到对应的功能/命令。

| 文档分析点 | 对应模型功能 | 分析命令/位置 |
|---|---|---|
| **①服务端宏观：端到端延迟拆解** | inference.py 分6阶段计时：数据生成→CPU预处理→H2D→NN→D2H→后处理 | `python3 inference.py` 输出 "End-to-End Latency Breakdown" |
| **①GPU利用率 <60% → CPU bottleneck** | train.py 日志含 gpu_util；inference.py 可开启 cpu_preprocess 制造GPU等CPU | 训练日志看 `gpu_util=XX%`；nsys看GPU timeline是否有大段空闲 |
| **①P50 vs P99 尾延迟** | inference.py 输出 NN延迟和总延迟的 p50/p90/p99/max | `python3 inference.py` 输出延迟统计 |
| **②框架级：torch.profiler 算子耗时** | profile_torch.py 采集CPU+CUDA算子，导出TensorBoard/Chrome trace | `python3 profile_torch.py` → `tensorboard --logdir=./prof_log` |
| **②不同batch/并发下表现** | inference.py batch_sweep模式测[1,4,16,64,256] | config设 `batch_sweep: true` → `python3 inference.py` |
| **③Kernel级：nsys CPU/GPU交互** | train.py/inference.py 内置NVTX range：data_load_cpu/h2d_transfer/forward/backward/nn_inference/d2h_transfer | `nsys profile --trace=cuda,nvtx,osrt python3 train.py` |
| **③Kernel级：H2D/D2H Memcpy占比** | train.py 显式CPU创建tensor再`.to(device)`；inference.py 显式H2D/D2H，nsys可观测cudaMemcpyHtoD/DtoH | `nsys profile --trace=cuda python3 inference.py` → 看Memcpy行 |
| **③小kernel间空闲间隙 → launch overhead** | Transformer每层有8+个小kernel（QKV投影/Attention/FFN/LayerNorm），nsys可观察 | nsys timeline看kernel间gap；batch_sweep对比不同batch |
| **③GPU长时间空闲等CPU** | config `enable_cpu_preprocess: true` + `cpu_preprocess_ms: 5` 可放大CPU bottleneck | 调大cpu_preprocess_ms后nsys看GPU空闲段是否变长 |
| **③Memcpy穿插在kernel之间** | 显式H2D/D2H + CPU预处理，数据依赖导致无法重叠 | nsys timeline看Memcpy与kernel的交错 |
| **③ncu单kernel硬件指标** | profile_ncu.py 跑1步（ncu_batch_size=64），ncu采集SM利用率/内存带宽/占用率/Warp stall | `ncu --set speedoflight --launch-skip 5 --launch-count 30 -o report python3 profile_ncu.py` |
| **③ncu memory-bound vs compute-bound** | benchmark_op.py 计算每个算子的AI=FLOPs/Bytes，对照T4平衡点~25 | `python3 benchmark_op.py` 输出每个算子的AI和瓶颈类型 |
| **④微基准：torch.utils.benchmark** | benchmark_op.py 精确测6类算子：Embedding/GEMM/Element-wise/Attention/LayerNorm/Concat | `python3 benchmark_op.py` |
| **Roofline判断** | benchmark_op.py 计算每个算子的计算强度AI，判断memory/compute-bound | `python3 benchmark_op.py` 汇总表 |
| **特征拉取占30-50%（行业参考）** | inference.py 的 cpu_preprocess 阶段模拟特征拉取，可调耗时占比 | 看Latency Breakdown中cpu_preprocess占比 |

---

---

## 0. 环境准备

```bash
cd /path/to/02_onetrans

# 安装依赖
pip install -r requirements.txt

# 验证GPU（T4）
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 验证Nsight工具已安装
which nsys && nsys --version
which ncu && ncu --version
```

> T4 是 Turing 架构（compute capability 7.5），Nsight System 和 Nsight Compute 均支持。
> T4 支持 TF32（Ampere及以上）不支持，但支持 FP16 混合精度。

---

## 1. 生成 Fake 数据

```bash
cd /path/to/02_onetrans
python3 fake_data.py
```

产出 `fake_data.pkl`，包含 1000 条样本，每条样本字段：
- `user_id`: 用户ID
- `item_id`: 候选物品ID
- `cate_id`: 物品类目ID
- `history`: 用户历史行为序列（长度20，不足padding）
- `history_mask`: 序列有效位掩码
- `numeric`: 3维数值特征
- `label`: 0/1 标签

---

## 2. 训练模型

```bash
cd /path/to/02_onetrans
python3 train.py
```

训练逻辑：
- 对 1000 条 fake 样本**无限循环随机采样**，直到达到 `config.yaml` 中的 `max_steps`
- 每 `log_every` 步打印日志：`loss / step / 已耗时 / 平均每步耗时 / 剩余时间(ETA)`
- 训练结束自动保存模型到 `onetrans_model.pt`

### 控制训练时长

修改 `config.yaml`：
```yaml
max_steps: 500          # 迭代次数，越大训练越久
batch_size: 256         # batch大小
num_layers: 3            # Transformer层数（层数越多越慢）
embedding_dim: 32        # d_model（维度越大越慢）
log_every: 10            # 日志打印间隔
```

### 典型输出

> 以下输出写入 `train.log`，用 `cat train.log` 或 `tail -f train.log` 查看。

```
[INFO] Device: cuda
[INFO] GPU: Tesla T4
[INFO] Loaded 1000 samples
[INFO] Model params: xxx,xxx
============================================================
[Step    10/500] loss=0.6931  elapsed=    3.5s  avg= 350.2ms/step  ETA=  171.5s
[Step    20/500] loss=0.6880  elapsed=    6.8s  avg= 340.1ms/step  ETA=  163.2s
...
============================================================
[DONE] Training finished in 170.5s
[DONE] Model saved to ./onetrans_model.pt
```

---

## 3. 推理压测

```bash
cd /path/to/02_onetrans
python3 inference.py
```

### 3.1 固定QPS压测模式（默认）

```bash
cd /path/to/02_onetrans
python3 inference.py
```

压测逻辑（端到端6阶段，每阶段独立计时）：
1. **数据生成**（CPU）：生成fake batch数据
2. **CPU预处理**（模拟特征拉取）：`enable_cpu_preprocess: true` 时模拟PS拉取+int化+拼接，耗时由 `cpu_preprocess_ms` 控制
3. **H2D搬运**：`enable_h2d_d2h: true` 时显式 `.to(device)`，nsys可观测cudaMemcpyHtoD
4. **NN推理**：Transformer前向计算
5. **D2H搬运**：显式 `.cpu()`，nsys可观测cudaMemcpyDtoH
6. **后处理**：sigmoid → numpy

- 以 `target_qps` 速率发送请求，达到 `target_requests` 后停止
- 输出：实际QPS、吞吐量、**端到端延迟拆解（各阶段占比）**、NN延迟p50/p90/p99、总延迟p50/p90/p99

### 3.2 Batch Sweep 模式（不同batch下延迟吞吐对比）

```bash
# 修改config.yaml: batch_sweep: true
python3 inference.py
```

自动测试 batch_size = [1, 4, 16, 64, 256]，输出每个batch的 avg/p50/p99延迟、吞吐量、GPU利用率。
Transformer模型对batch更敏感：小batch时kernel launch overhead极高（每层8+个kernel），大batch时GPU利用率提升明显。

### 控制压测强度 & 性能分析开关

修改 `config.yaml`：
```yaml
# 压测
infer_batch_size: 64      # 每个请求的样本数
target_qps: 100           # 目标请求数/秒
target_requests: 10000    # 总请求数（达到后停止）
warmup_requests: 100      # 预热次数

# 性能分析开关
enable_cpu_preprocess: true   # 模拟CPU特征拉取（制造GPU等CPU场景）
cpu_preprocess_ms: 2          # 模拟CPU预处理耗时(ms)，调大可放大CPU bottleneck
enable_h2d_d2h: true          # 显式H2D/D2H数据搬运（nsys可观测Memcpy）
enable_latency_breakdown: true # 输出端到端延迟拆解
batch_sweep: false             # true=batch sweep模式，false=固定QPS模式
```

### 典型输出（含延迟拆解）

> 以下输出写入 `inference.log`，用 `cat inference.log` 查看。

```
===== End-to-End Latency Breakdown =====
  cpu_preprocess    : avg=  2.045ms  ( 25.1%)
  h2d               : avg=  0.035ms  (  0.4%)
  nn                : avg=  5.980ms  ( 73.4%)
  d2h               : avg=  0.018ms  (  0.2%)
  postprocess       : avg=  0.075ms  (  0.9%)
  total             : avg=  8.153ms  (100.0%)
```

> OneTrans的NN占比(73.4%)通常比DCN-DIN高，因为Transformer自注意力计算量更大；对比两个模型的延迟拆解可以直观看到经典模型vs Transformer模型的计算特性差异。

---

## 3b. 微基准测试（算子级 + Roofline分析）

```bash
cd /path/to/02_onetrans
python3 benchmark_op.py
```

用 `torch.utils.benchmark` 精确测量6类关键算子，每个算子计算 **计算强度 AI = FLOPs/Bytes**，对照T4的Roofline平衡点（~25 FLOPs/byte）判断 memory-bound 还是 compute-bound：

| 算子 | 典型瓶颈 | OneTrans中的位置 |
|---|---|---|
| Embedding Lookup | memory-bound | 所有token的embedding查找（user/item/cate/history） |
| 小GEMM (M=batch) | memory-bound | QKV投影、输出投影、FFN（d_model=32较小） |
| 大GEMM (1024³) | compute-bound | （参考基准，OneTrans中没有这么大的GEMM） |
| Element-wise | memory-bound | 残差连接、GELU激活、position/type embedding相加 |
| Attention (短序列) | memory-bound | 自注意力QK^T+softmax+AV^T，token数=25较短 |
| LayerNorm | memory-bound | Pre-LN每层2个，3层=6个+最终1个 |
| Concat | memory-bound | token拼接（<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>/use/item/cate/numeric/history） |

> 对应文档1.3节Roofline模型。OneTrans的LayerNorm和Element-wise kernel数量比DCN-DIN多，nsys中更容易看到大量细碎小kernel。

---

## 4. PyTorch Profiler 分析（框架级定位）

torch.profiler 是 PyTorch 内置的性能分析工具，用来定位**哪个算子（op）最耗时**，是框架级定位的首选工具。它比nsys更聚焦于PyTorch层面的算子，比ncu更轻量（不需要深入硬件计数器）。

### 4.1 运行采集

```bash
cd /path/to/02_onetrans
python3 profile_torch.py
```

**profile_torch.py 的采集逻辑（初学者了解）：**
1. 先跑5步warmup（不采集）
2. 然后用 `torch.profiler.profile` 采集5步（wait=1, warmup=1, active=3, repeat=1）
3. 采集的活动：CPU算子 + CUDA kernel
4. 同时记录：算子输入形状(`record_shapes`)、内存占用(`profile_memory`)、Python调用栈(`with_stack`)、FLOPs(`with_flops`)

**产出文件：**
- `prof_log/` 目录：TensorBoard格式的trace文件
- `trace_chrome.json`：Chrome Trace格式的文件

### 4.2 用 TensorBoard 查看（推荐，可视化最好）

```bash
# 启动TensorBoard
tensorboard --logdir=./prof_log --port=6006

# 然后在浏览器打开 http://localhost:6006
```

> 如果提示`tensorboard: command not found`，先安装：`pip install tensorboard torch-tb-profiler`

#### TensorBoard 界面操作指南（初学者必读）

打开浏览器后，在顶部导航栏找到 **"PyTorch Profiler"** 标签页（可能在"INACTIVE"下拉菜单里，需要点一下激活）。

进入后，左侧有多个子标签页，从上到下依次是：

| 子标签页 | 看什么 | 对应分析点 |
|---|---|---|
| **Overview** | 总览：step时间、GPU利用率、TOP算子、性能建议 | 第一眼了解整体情况 |
| **Operator** | 所有PyTorch算子的表格，按CPU/CUDA时间排序 | **哪个算子最耗时**（核心！） |
| **Kernel** | 所有GPU kernel的表格，按CUDA时间排序 | 哪个GPU kernel最耗时 |
| **Trace** | 时间线视图（类似nsys和chrome trace） | 看算子执行顺序、kernel间隙、CPU/GPU交互 |
| **Memory** | 显存分配时间线 | 峰值显存、内存泄漏 |
| **Module** | 按模型模块（nn.Module）分组的耗时 | 模型哪个部分最耗时 |

#### 各子标签页操作要点

**Overview（总览）：**
- 顶部显示：Average Step Time（平均每步耗时）、GPU Utilization（GPU利用率）、Est. SM Efficiency（SM效率）
- 中间"Performance Recommendations"：PyTorch自动给出的优化建议（比如"GPU is underutilized"、"Large cudaMemcpy"等）
- 底部"Top Operators"：耗时TOP10的算子

**Operator（算子表格，最常用）：**
- 表格列包括：Name（算子名）、Calls（调用次数）、CPU Time（CPU总耗时）、CUDA Time（CUDA总耗时）、% of total（占比）
- **点击列标题可以排序**：点"CUDA Time"按GPU耗时排序，点"CPU Time"按CPU耗时排序
- **看什么**：
  - 按CUDA Time排序 → 哪些算子在GPU上最耗时（OneTrans通常是：bmm/matmul注意力GEMM、layer_norm、softmax、embedding）
  - 按CPU Time排序 → 哪些算子在CPU上最耗时（通常是数据处理、Python开销、attention mask构建）
  - 看"% of total"列 → 占比最高的几个算子就是优化重点
- **点击某个算子名**可以展开看调用栈（Call Stack），知道这个算子是代码里哪一行调用的

**Trace（时间线）：**
- 类似nsys的时间线，从上到下是CPU线程、CUDA Runtime、GPU kernel
- **操作**：鼠标滚轮缩放，拖拽平移，点击事件看详情
- **看什么**：
  - GPU kernel行是否有大段空白（GPU利用率低）— OneTrans小batch时常见
  - CPU和GPU是否重叠（CPU在准备数据时GPU是否在计算）
  - 算子的执行顺序和依赖关系

**Memory（内存）：**
- 显存分配/释放的时间线
- 看峰值显存（Peak Memory）、是否有内存只增不减（可能泄漏）

### 4.3 用 Perfetto 查看（推荐，加载速度快）

```bash
# profile_torch.py 会自动导出 trace_chrome.json
# 浏览器打开 https://ui.perfetto.dev/
# 点击左上角 "Open trace file" → 选择 trace_chrome.json
```

> **Perfetto 是 chrome://tracing 的官方替代工具**，加载大 trace 文件快 5~10 倍，UI 更现代，还支持用 SQL 查询 trace 数据。torch.profiler 导出的 Chrome Trace 格式和 Perfetto 完全兼容，直接打开即可。

#### Perfetto 操作指南

打开后界面是一个时间线：
- **缩放**：鼠标滚轮（以光标位置为中心），或键盘 `W`/`S`
- **平移**：键盘 `A`/`D`，或按住 Shift + 鼠标拖拽
- **选中事件**：鼠标左键点击某个色块
- **查看详情**：选中后下方面板显示事件名、耗时、调用栈、输入形状等
- **搜索**：顶部搜索框输入算子名（如"matmul"、"layer_norm"、"bmm"）高亮匹配的事件
- **切换线程**：左侧显示线程名，可以折叠/展开
- **SQL 查询**：点击底部 "Query (SQL)" 标签，可以用 SQL 查 trace 数据（比如查所有耗时>1ms的kernel）

**看什么**：
- 时间线上方是CPU线程（Python主线程），下方是GPU stream（CUDA kernel）
- 看GPU kernel之间是否有空白（launch overhead或CPU bottleneck）— OneTrans每层8+个kernel，小batch时间隙明显
- 点击某个kernel看它的名称和耗时

> **备选工具：chrome://tracing**（旧工具，大文件加载慢）
> Chrome 浏览器地址栏输入 `chrome://tracing` → 左上角 "Load" 按钮 → 选择 `trace_chrome.json`。操作方式类似，但 trace 文件大时加载可能卡顿，推荐用 Perfetto。

### 4.4 终端表格（脚本运行结束自动打印）

profile_torch.py 运行结束后会自动在终端打印两个表格：
- **CPU Operators TOP20**：按CPU时间排序的算子
- **CUDA Kernels TOP20**：按CUDA时间排序的kernel

适合快速看一眼，不需要打开浏览器。

### 4.5 重点关注（OneTrans模型）

| 指标 | 在TensorBoard哪里看 | 说明 |
|---|---|---|
| **哪个算子最耗时** | Operator标签页 → 按CUDA Time排序 | OneTrans通常是：bmm/matmul(注意力GEMM)、layer_norm、softmax、embedding、elementwise(残差/GELU) |
| **GPU利用率** | Overview → GPU Utilization | OneTrans小batch时通常<60%（每层8+个小kernel，launch overhead大） |
| **CPU vs GPU时间占比** | Operator表格 → CPU Time vs CUDA Time列 | CPU时间远大于GPU时间 → CPU bottleneck（attention mask构建、数据准备） |
| **kernel间隙** | Trace标签页 → GPU kernel行 | OneTrans小batch时间隙明显（kernel launch overhead） |
| **峰值显存** | Memory标签页 | 看是否超出T4的16GB |
| **自动优化建议** | Overview → Performance Recommendations | PyTorch自动给出的建议，参考即可 |
| **`aten::bmm`/`aten::matmul`** | Operator搜索"bmm"或"matmul" | 注意力QK^T和AV^T的批量矩阵乘，Transformer主力算子 |
| **`aten::layer_norm`** | Operator搜索"layer_norm" | Pre-LN每层2个，3层=6个+最终1个，数量多 |

---

## 5. Nsight System 分析（框架级 + GPU timeline）

Nsight Systems（简称 nsys）看**整体时间线**：CPU端操作、CUDA API调用、GPU kernel执行、NVTX标记范围、**H2D/D2H内存拷贝**、kernel之间的空闲间隙等。它是性能分析的"全景图"，用来定位"时间花在哪了"。

### 5.1 采集训练过程

```bash
cd /path/to/02_onetrans

# 建议先把config.yaml的max_steps改小（如50~100），避免采集文件太大
nsys profile \
  --output=nsys_train \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  python3 train.py
```

**每个参数的含义（初学者必读）：**

| 参数 | 含义 | 为什么需要 |
|---|---|---|
| `profile` | nsys的子命令，表示开始采集 | 固定写法 |
| `--output=nsys_train` | 输出文件名前缀，最终生成 `nsys_train.nsys-rep` | 方便区分训练/推理的报告 |
| `--force-overwrite=true` | 如果同名文件已存在则覆盖 | 反复调试时不用手动删旧文件 |
| `--trace=cuda,nvtx,osrt,cudnn,cublas` | 要追踪的事件类型，逗号分隔 | 见下方详解 |
| `--cuda-memory-usage=true` | 追踪CUDA内存分配/释放事件 | 看显存占用变化、是否有内存碎片 |

**`--trace` 各事件类型详解：**
- `cuda`：追踪CUDA API调用（cudaLaunchKernel、cudaMemcpy、cudaMalloc等）和GPU kernel执行。**最核心，必选**
- `nvtx`：追踪代码中用NVTX标记的范围（我们在train.py里埋了step_N、data_load_cpu、h2d_transfer、forward、backward等标记）。**必选，否则看不到我们埋的点**
- `osrt`：追踪OS运行时事件（线程创建、互斥锁、系统调用等）。可选，帮助排查CPU侧的线程同步问题
- `cudnn`：追踪cuDNN库调用（我们模型用得少，可选）
- `cublas`：追踪cuBLAS库调用（Transformer的GEMM走cuBLAS，可选但有用）

**产出文件：** `nsys_train.nsys-rep`（二进制报告文件，需要用nsys-ui打开）

### 5.2 采集推理过程

```bash
cd /path/to/02_onetrans

# 建议先把config.yaml的target_requests改小（如300~500），避免采集太久
nsys profile \
  --output=nsys_infer \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  python3 inference.py
```

推理采集不需要cudnn/cublas（推理kernel更少），trace范围小一点采集更快。

> **初学者提示**：nsys采集会让程序变慢约2~5倍，所以训练/推理的实际耗时会比正常长，这是正常的。采集到的时间线比例关系仍然准确。

### 5.3 用 Nsight Systems GUI 打开报告（初学者操作指南）

```bash
# 在有图形界面的机器上执行（T4服务器如果是纯命令行，需要把.nsys-rep文件下载到本地打开）
nsys-ui nsys_train.nsys-rep
```

**如果T4服务器没有图形界面：**
```bash
# 方法1：把文件scp到本地，用本地的nsys-ui打开
scp user@server:/path/to/nsys_train.nsys-rep ~/Downloads/
# 然后本地执行 nsys-ui ~/Downloads/nsys_train.nsys-rep

# 方法2：用命令行统计（见5.4），不需要GUI
```

#### 打开后界面布局（从上到下）

打开nsys-ui后，主界面是一个**时间线（Timeline）视图**，从上到下依次是：

```
┌─────────────────────────────────────────────────────┐
│  工具栏：缩放、过滤、导出按钮                          │
├─────────────────────────────────────────────────────┤
│  ① CPU线程行（Threads）                               │
│     Python主线程、数据加载线程等，显示函数调用栈       │
├─────────────────────────────────────────────────────┤
│  ② NVTX 行（NVTX）                                   │
│     显示我们代码里埋的标记：step_N、forward、backward  │
│     ★ 看这个行可以快速定位每个step的各个阶段           │
├─────────────────────────────────────────────────────┤
│  ③ CUDA API 行（CUDA API）                           │
│     显示cudaLaunchKernel、cudaMemcpy等API调用        │
│     ★ 看CPU侧发起了哪些CUDA操作                       │
├─────────────────────────────────────────────────────┤
│  ④ CUDA内存操作行（CUDA Memory Operations）           │
│     显示H2D/D2H内存拷贝（Memcpy HtoD/DtoH）          │
│     ★★★ 看H2D/D2H数据搬运就看这一行！                 │
├─────────────────────────────────────────────────────┤
│  ⑤ GPU Kernel 行（CUDA Kernels）                     │
│     显示每个GPU kernel的名称、起止时间、持续时长       │
│     ★★★ 看kernel执行、kernel间间隙、GPU利用率看这一行  │
│     OneTrans每层8+个kernel，这一行会很密集             │
└─────────────────────────────────────────────────────┘
```

#### 基本操作（初学者必读）

| 操作 | 方法 |
|---|---|
| **放大/缩小** | 鼠标滚轮（光标位置为中心缩放）；或工具栏 +/- 按钮 |
| **左右平移** | 按住Shift + 鼠标拖拽；或横向滚动条 |
| **选中某个事件** | 鼠标左键点击时间线上的色块 |
| **查看事件详情** | 选中后，底部面板（Properties）显示该事件的名称、起止时间、持续时长、线程等 |
| **过滤** | 工具栏有个过滤框（Filter），输入关键词如"Memcpy"只看内存拷贝 |
| **跳到某个时间点** | 底部时间轴可以拖拽 |

#### 对照分析点：看什么、在哪里看

| 你想看的分析点 | 在nsys-ui里怎么看 |
|---|---|
| **H2D/D2H数据搬运占比** | 看第④行"CUDA Memory Operations"，找橙色/红色的"Memcpy HtoD"（H2D）和"Memcpy DtoH"（D2H）色块。选中后看Properties里的Duration。和第⑤行kernel总时长对比，就知道搬运占比 |
| **GPU kernel之间的空闲间隙** | 看第⑤行"CUDA Kernels"，如果kernel色块之间有大量白色空白，说明GPU在等CPU（launch overhead或数据准备慢）。OneTrans小batch时间隙特别明显（每层8+个小kernel） |
| **NVTX各阶段占比** | 看第②行"NVTX"，每个step_N包含data_load_cpu→h2d_transfer→forward→backward几个子段。选中某个子段看Duration，就能知道forward和backward各占多少 |
| **GPU利用率** | 看第⑤行kernel色块是否"铺满"时间线。如果大部分时间都有kernel在跑，利用率高；如果大片空白，利用率低。OneTrans小batch时通常利用率低 |
| **Memcpy穿插在kernel之间** | 同时看第④行和第⑤行，如果Memcpy色块和kernel色块在时间上交错出现，说明数据依赖导致无法重叠（理想情况是Memcpy和kernel并行） |
| **CPU侧瓶颈** | 看第①行CPU线程，如果CPU函数调用很长而GPU kernel行有空白，说明CPU在忙而GPU在等 |
| **Transformer每层的kernel序列** | 放大第⑤行某个forward段，可以看到每层的kernel顺序：QKV投影GEMM→QK^T BMM→softmax→AV^T BMM→输出投影GEMM→LayerNorm→FFN GEMM→GELU→FFN GEMM→LayerNorm→残差 |
| **某个kernel的具体耗时** | 在第⑤行点击该kernel色块，底部Properties显示Kernel Name、Duration、Grid/Block尺寸等 |

#### 常用过滤技巧

在工具栏的Filter框输入：
- `Memcpy`：只显示内存拷贝相关事件（快速定位H2D/D2H）
- `kernel`：只显示kernel相关
- `nvtx`：只显示NVTX标记
- `cudaLaunchKernel`：只显示kernel launch API调用
- `gemm|matmul|bmm`：只显示矩阵乘相关kernel（OneTrans主力算子）

### 5.4 用命令行统计（不需要GUI）

如果T4服务器没有图形界面，可以用`nsys stats`直接输出统计表格：

```bash
# 1. 按GPU kernel耗时排序（看哪些kernel最耗时）
nsys stats --report gpu_kernel_time_nvtx_sum nsys_train.nsys-rep

# 2. 按CUDA API耗时排序（看CPU侧哪些CUDA调用最耗时）
nsys stats --report cuda_api_sum nsys_train.nsys-rep

# 3. NVTX范围耗时统计（看forward/backward各占多少）
nsys stats --report nvtx_sum nsys_train.nsys-rep

# 4. 内存拷贝统计（看H2D/D2H总量和耗时）
nsys stats --report cuda_memcpy_sum nsys_train.nsys-rep

# 5. 导出所有统计为CSV文件（方便用Excel打开分析）
nsys stats --format csv --output . nsys_train.nsys-rep
# 会在当前目录生成多个.csv文件
```

**`nsys stats` 各report含义：**
| report名 | 输出内容 | 对应分析点 |
|---|---|---|
| `gpu_kernel_time_nvtx_sum` | 每个GPU kernel的总耗时、调用次数、平均耗时 | 哪些kernel最耗时 |
| `cuda_api_sum` | 每个CUDA API的总耗时、调用次数 | CPU侧CUDA调用瓶颈 |
| `nvtx_sum` | 每个NVTX范围的总耗时、平均耗时 | forward/backward/data_load各占多少 |
| `cuda_memcpy_sum` | 内存拷贝的总量、耗时、按方向(H2D/D2H)分类 | H2D/D2H数据搬运占比 |

### 5.5 重点关注（OneTrans模型）

| 观察项 | 在nsys里怎么看 | 说明 | 优化方向 |
|---|---|---|---|
| **H2D搬运占比** | 第④行Memcpy HtoD总时长 / 第⑤行kernel总时长 | 我们在train.py里显式做了CPU→GPU搬运，正常应该占比很小（<5%）。如果占比高说明batch太大或数据准备慢 | 用pin_memory、non_blocking=True、减小batch |
| **kernel间空闲间隙** | 第⑤行kernel色块之间的空白 | OneTrans每层8+个kernel（QKV投影/Attention/FFN/LayerNorm），小batch时间隙特别明显，launch overhead大 | 增大batch、CUDA Graph、算子融合（Flash Attention） |
| **forward vs backward占比** | 第②行NVTX里forward和backward段的时长对比 | OneTrans的forward比DCN-DIN重（自注意力计算量大），正常backward约为forward的2倍 | — |
| **Transformer每层kernel序列** | 放大第⑤行某个forward段 | 可以看到每层的kernel顺序：GEMM→BMM→softmax→BMM→GEMM→LayerNorm→GEMM→GELU→GEMM→LayerNorm | 看是否有可以融合的相邻kernel |
| **LayerNorm kernel数量** | 第⑤行找名称含"layer_norm"的kernel，数数量 | Pre-LN每层2个，3层=6个+最终1个=7个，数量多且单个很短，launch overhead占比高 | 融合LayerNorm+残差+激活、Fused Layer Norm |
| **Element-wise kernel是否细碎** | 第⑤行找名称含"elementwise"的kernel | 残差连接、GELU激活、position/type embedding相加都是element-wise，数量多 | torch.compile算子融合 |
| **GPU利用率** | 第⑤行kernel是否铺满时间线 | OneTrans小batch时通常<60%（kernel多但每个都小，launch overhead大）；大batch时利用率提升 | 增大batch、CUDA Graph、算子融合 |

---

## 6. Nsight Compute 分析（GPU Kernel级深度定位）

Nsight Compute（简称 ncu）深入**单个GPU kernel的硬件级指标**：SM算力利用率、显存带宽利用率、占用率(Occupancy)、Warp调度效率、指令吞吐、L1/L2缓存命中率等。它是性能分析的"显微镜"，用来回答"这个kernel为什么慢"。

> **初学者提示**：ncu采集会让程序变慢10~50倍（因为每个kernel都要采集大量硬件计数器）。`profile_ncu.py`已优化为轻量版：用`ncu_batch_size=64`（比推理batch小很多，kernel类型完全一样）、预生成数据、只跑1个推理step。配合下面的`--set speedoflight --launch-count 30`，几分钟内就能跑完。

### 6.1 快速采集（推荐，几分钟跑完）

```bash
cd /path/to/02_onetrans

# 推荐：speedoflight模式（2次pass/kernel）+ 跳过前5个kernel + 只采30个
ncu \
  --set speedoflight \
  --launch-skip 5 \
  --launch-count 30 \
  -o ncu_report \
  --force-overwrite \
  python3 profile_ncu.py
```

采集最核心的两个指标：**SM算力利用率**和**显存带宽利用率**，足够判断 memory-bound 还是 compute-bound。

### 6.2 完整采集（所有kernel，慢，深度分析用）

```bash
# 对特定kernel类型用 full 模式深度分析（先用nsys找到top kernel，再针对性采集）
ncu \
  --set full \
  -o ncu_report_full \
  --force-overwrite \
  python3 profile_ncu.py
```

**每个参数的含义：**

| 参数 | 含义 | 为什么需要 |
|---|---|---|
| `--set speedoflight` | 只采集Speed of Light（算力+带宽利用率），2次pass/kernel | 最快，第一次分析用这个，判断bound类型 |
| `--set full` | 采集所有可用的硬件计数器（最全，也最慢，30+次pass/kernel） | 对top kernel做深度分析时用 |
| `--launch-skip 5` | 跳过前5个kernel launch（模型加载/cuBLAS初始化） | 跳过warmup，只采集正常推理的kernel |
| `--launch-count 30` | 只采集30个kernel launch | 限制采集数量，大幅加速 |
| `-o ncu_report` | 输出文件名前缀，最终生成 `ncu_report.ncu-rep` | 方便区分不同报告 |
| `--force-overwrite` | 同名文件已存在则覆盖 | 反复调试时不用手动删 |
| `python3 profile_ncu.py` | 要采集的程序 | 推理模式轻量版脚本，ncu_batch_size=64，1个推理step |

**`--set` 可选值（初学者了解）：**
| 值 | 采集内容 | 每kernel pass数 | 速度 | 适用场景 |
|---|---|---|---|---|
| `speedoflight` | 算力+带宽利用率 | 2 | 最快 | 快速判断bound类型（推荐先用这个） |
| `basic` | 基础指标（耗时、SM利用率、内存带宽） | ~5 | 快 | 快速定位 |
| `detailed` | 大部分计数器 | ~15 | 中等 | 常规分析 |
| `full` | 所有硬件计数器 | 30+ | 最慢 | 对top kernel深度分析 |

**产出文件：** `ncu_report.ncu-rep`

### 6.3 只采集特定类型的kernel（更快、更聚焦）

完整采集所有kernel会很慢（可能10分钟以上）。如果已经通过nsys知道哪类kernel最耗时，可以只采集那类：

```bash
# 只采集GEMM/BMM矩阵乘kernel（Transformer主力：QKV投影、注意力QK^T/AV^T、FFN）
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:.*gemm.*|.*matmul.*|.*linear.*|.*bmm.*|.*sgemm.*" \
  --set full \
  -o ncu_gemm \
  --force-overwrite \
  python3 profile_ncu.py

# 只采集Attention相关kernel（QK^T BMM + softmax + AV^T BMM）
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:.*bmm.*|.*softmax.*|.*attn.*" \
  --set full \
  -o ncu_attention \
  --force-overwrite \
  python3 profile_ncu.py

# 只采集Softmax kernel（注意力权重归一化）
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:.*softmax.*" \
  --set full \
  -o ncu_softmax \
  --force-overwrite \
  python3 profile_ncu.py

# 只采集LayerNorm kernel（Pre-LN每层2个，数量多）
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:.*layer_norm.*|.*layernorm.*|.*ln.*" \
  --set full \
  -o ncu_layernorm \
  --force-overwrite \
  python3 profile_ncu.py

# 只采集Embedding/gather kernel
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:.*embedding.*|.*gather.*|.*index.*|.*vectorized.*" \
  --set full \
  -o ncu_embedding \
  --force-overwrite \
  python3 profile_ncu.py

# 只采集Element-wise kernel（GELU激活、残差连接、position/type embedding相加）
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:.*elementwise.*|.*mul.*|.*add.*|.*gelu.*|.*relu.*" \
  --set full \
  -o ncu_elementwise \
  --force-overwrite \
  python3 profile_ncu.py
```

**参数含义：**
- `--kernel-name-base demangled`：用C++函数名（demangled，人类可读的名字）匹配kernel，而不是编译后的mangled名字
- `--kernel-name "regex:..."`：用正则表达式匹配kernel名称，只采集匹配的kernel。`regex:`前缀表示后面是正则

### 6.4 跳过前N个kernel launch（跳过warmup）

前几个kernel launch通常是warmup（cuBLAS初始化、显存分配等），不是正常计算。可以跳过：

```bash
# 跳过前200个kernel launch（Transformer kernel多，前200可能是warmup），只采集后面的100个
ncu \
  --launch-skip 200 \
  --launch-count 100 \
  --set full \
  -o ncu_skip \
  --force-overwrite \
  python3 profile_ncu.py
```

- `--launch-skip 200`：跳过前200次kernel launch
- `--launch-count 100`：只采集100次kernel launch

### 6.5 用 Nsight Compute GUI 打开报告（初学者操作指南）

```bash
# 在有图形界面的机器上执行
ncu-ui ncu_report.ncu-rep
```

**如果T4服务器没有图形界面：** 把`.ncu-rep`文件scp到本地，用本地的ncu-ui打开。

#### 打开后界面布局

ncu-ui的界面分为三个主要区域：

```
┌──────────────────┬──────────────────────────────────┬──────────────┐
│  ① Kernel列表    │  ② 详情面板（选中kernel的指标）    │  ③ 优化建议   │
│  （左侧）        │  （中间，可滚动）                  │  （右侧）     │
│                  │                                  │              │
│  kernel_1  1.2ms│  ★ Speed of Light (SOL)          │  "This kernel│
│  kernel_2  0.8ms│    SM Utilization: 45.2%        │   is memory  │
│  kernel_3  0.5ms│    Memory Utilization: 72.3%     │   bound"      │
│  ...            │                                  │  ...         │
│                  │  ★ Occupancy                      │              │
│  （按耗时排序）  │    Achieved: 32.5%               │              │
│                  │    Theoretical: 50.0%            │              │
│                  │    Registers/thread: 48          │              │
│                  │                                  │              │
│                  │  ★ Warp State Statistics          │              │
│                  │    (饼图: Active/Stall Long Sb/  │              │
│                  │     Stall Short Sb/...)           │              │
│                  │                                  │              │
│                  │  ★ Memory Workload Analysis       │              │
│                  │    DRAM Throughput: 210 GB/s     │              │
│                  │    L2 Hit Rate: 65%              │              │
│                  │    L1 Hit Rate: 40%              │              │
│                  │                                  │              │
│                  │  ★ Compute Workload Analysis      │              │
│                  │    FP32 Instructions: 1.2M       │              │
│                  │    Tensor Core Util: 0%           │              │
│                  │                                  │              │
│                  │  ★ Roofline Analysis              │              │
│                  │    (散点图: 每个kernel在roofline  │              │
│                  │     上的位置，判断bound类型)      │              │
└──────────────────┴──────────────────────────────────┴──────────────┘
```

#### 操作步骤（初学者必读）

**Step 1：在左侧①选一个kernel**
- 左侧列表按耗时从高到低排序，先点最耗时的那个
- 列表显示：kernel名称、总耗时、调用次数、平均耗时

**Step 2：在中间②看关键指标（按重要性排序）**

**第一眼看：Speed of Light（SOL）**
- 这是最核心的指标，两个百分比：
  - **SM Utilization（算力利用率）**：SM（流多处理器）的计算单元有多忙。>70%算高，<30%算低
  - **Memory Utilization（显存带宽利用率）**：显存带宽用了多少。>60%算高
- **判断方法**：
  - Memory Utilization高（>60%）+ SM Utilization低（<50%）→ **memory-bound**（内存瓶颈，算得少等数据）
  - SM Utilization高（>70%）+ Memory Utilization低 → **compute-bound**（算力瓶颈，数据够了算不过来）
  - 两个都低 → **latency-bound**（延迟瓶颈，通常是kernel太小、launch overhead大或warp stall多）— OneTrans小kernel常见

**第二眼看：Occupancy（占用率）**
- **Achieved Occupancy（实际占用率）**：实际有多少warp在SM上活跃。>50%算好，<25%算低
- **Theoretical Occupancy（理论占用率）**：硬件能达到的上限
- 如果Achieved远低于Theoretical，看原因：
  - **Registers/thread（每个线程用的寄存器数）**：用太多寄存器会限制每个SM能跑的warp数。T4每个SM最多64K寄存器，每个warp 32线程
  - **Shared Memory/block（每个block用的共享内存）**：用太多共享内存也会限制占用率

**第三眼看：Warp State Statistics（Warp状态统计）**
- 饼图显示warp在各种状态的时间占比：
  - **Active（活跃）**：warp正在执行指令，越高越好
  - **Stall Long Scoreboard（等内存访问）**：warp在等显存/L2数据回来。占比高（>30%）说明内存延迟大，是memory-bound的典型表现
  - **Stall Short Scoreboard（等计算依赖）**：warp在等前面的计算结果。占比高说明指令依赖链长
  - **Stall Fetch（取指令）**、**Stall Dispatch（派发）**等：其他stall原因
- **怎么看**：Active占比越高越好；如果某个Stall占比特别高，那就是瓶颈所在

**第四眼看：Memory Workload Analysis（内存工作负载）**
- **DRAM Throughput（显存带宽）**：实际用了多少GB/s。T4峰值320GB/s，用到200+算高
- **L2 Cache Hit Rate（L2缓存命中率）**：>50%算好，低说明数据访问没有局部性
- **L1/TEX Hit Rate（L1缓存命中率）**
- **DRAM Bytes/Transaction（每次内存事务的字节数）**：接近32B或128B说明内存访问是合并的(coalesced)，低说明非合并访问（效率低）

**第五眼看：Compute Workload Analysis（计算工作负载）**
- **FP32/FP16 Instructions**：各精度指令数
- **Tensor Core Utilization（Tensor Core利用率）**：用了多少Tensor Core。OneTrans的小GEMM（d_model=32）通常Tensor Core利用率低（因为矩阵太小用不上Tensor Core）
- **FLOPs**：总浮点运算量

**第六眼看：Roofline Analysis（Roofline分析）**
- 散点图，每个点是一个kernel
- 横轴：计算强度 AI = FLOPs/Bytes（FLOPs/byte）
- 纵轴：实际性能（FLOPs/s）
- 图上有两条线：斜线（内存带宽上限）和水平线（算力上限）
- **点在斜线下方** → memory-bound（增加计算强度或减少内存访问可以提升性能）
- **点在水平线附近** → compute-bound（已经到算力上限，需要Tensor Core或量化）
- T4的Roofline平衡点约为 8.1TFLOPS / 320GB/s ≈ 25 FLOPs/byte

**Step 3：看右侧③的优化建议**
- Nsight Compute会自动给出优化建议，比如：
  - "This kernel is memory-bound" → 建议减少内存访问、算子融合
  - "Low occupancy due to high register usage" → 建议减少寄存器使用
  - "Uncoalesced memory access" → 建议优化数据布局
- 这些建议是参考，需要结合具体kernel判断

#### 对照分析点：看什么、点哪里

| 你想看的分析点 | 在ncu-ui里点哪里 |
|---|---|
| **这个kernel是memory-bound还是compute-bound** | 中间面板 → Speed of Light → 看SM Utilization和Memory Utilization谁高；或看Roofline Analysis散点图 |
| **SM算力利用率** | Speed of Light → SM Utilization百分比 |
| **显存带宽利用率** | Speed of Light → Memory Utilization百分比；或Memory Workload Analysis → DRAM Throughput |
| **占用率低的原因** | Occupancy → 看Achieved vs Theoretical，再看Registers/thread和Shared Memory/block |
| **Warp在等什么（stall原因）** | Warp State Statistics → 饼图，看哪个Stall占比最高 |
| **L2缓存命中率** | Memory Workload Analysis → L2 Cache Hit Rate |
| **内存访问是否合并** | Memory Workload Analysis → DRAM Bytes/Transaction，接近32B/128B说明合并 |
| **Tensor Core用了多少** | Compute Workload Analysis → Tensor Core Utilization |
| **Roofline上的位置** | Roofline Analysis → 散点图 |
| **自动优化建议** | 右侧面板 → Guidance/Recommendations |

### 6.6 命令行查看摘要（不需要GUI）

```bash
# 打印所有kernel的关键指标摘要（一行一个kernel，含SM利用率、内存带宽、占用率等）
ncu --import ncu_report.ncu-rep --print-summary per-kernel

# 打印某个指定kernel的完整详细指标
ncu --import ncu_report.ncu-rep --kernel-name "kernel名称" --print-details

# 只打印Speed of Light指标（快速判断bound类型）
ncu --import ncu_report.ncu-rep --print-summary per-kernel --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed
```

### 6.7 重点关注的硬件指标速查表

| 指标 | 在ncu-ui哪里 | 含义 | 健康值 | 低/高的原因和优化 |
|---|---|---|---|---|
| **SM Utilization** | Speed of Light | 算力单元利用率 | >70% | 低→kernel太小/launch overhead，增大batch/融合算子 |
| **Memory Utilization** | Speed of Light | 显存带宽利用率 | >60% | 高→memory-bound，优化数据复用/融合/量化；低→compute-bound |
| **Achieved Occupancy** | Occupancy | 实际占用率 | >50% | 低→寄存器/共享内存用太多，减少并行度限制 |
| **Warp Stall: Long Scoreboard** | Warp State | 等内存访问的stall占比 | <30% | 高→内存延迟大，优化coalescing/cache/数据局部性 |
| **Warp Stall: Short Scoreboard** | Warp State | 等计算依赖的stall占比 | <20% | 高→指令依赖链长，优化算法减少依赖 |
| **L2 Cache Hit Rate** | Memory Workload | L2缓存命中率 | >50% | 低→数据访问无局部性，优化访问模式/数据布局 |
| **DRAM Bytes/Transaction** | Memory Workload | 内存合并度 | 接近32B/128B | 低→非合并访问，优化数据布局/padding对齐 |
| **Registers/thread** | Occupancy | 每线程寄存器数 | <64(T4) | 高→限制占用率，可尝试减少临时变量/用--maxrregcount |

### 6.8 OneTrans 模型典型kernel分析要点

1. **GEMM / BMM (矩阵乘)**:
   - Transformer中最多的算子：QKV投影(3个Linear)、注意力QK^T和AV^T(2个BMM)、输出投影、FFN(2个Linear)
   - 每层约 8 个矩阵乘，3层就是 24 个
   - 我们的d_model=32，GEMM尺寸较小，可能是**memory-bound**（M小K大）
   - 在ncu-ui看：Speed of Light判断bound类型；看Tensor Core Utilization（小GEMM通常<10%，因为矩阵太小用不上Tensor Core）
   - 优化：FP16混合精度（T4支持FP16 Tensor Core）、确保矩阵维度是8的倍数、增大batch让M变大、Flash Attention融合QK^T+softmax+AV^T

2. **Attention (QK^T + softmax + AV^T)**:
   - 自注意力的三个核心操作，token数=5+20=25，序列较短
   - **典型memory-bound**（多次小GEMM+softmax，launch overhead大）
   - 在ncu-ui看：Warp Stall Long Scoreboard占比（等内存访问）；看kernel是否很小（微秒级）
   - 优化：Flash Attention（融合scale+mask+softmax+dropout，减少内存读写和kernel launch）

3. **Softmax (注意力归一化)**:
   - 每层1个，对注意力分数做softmax
   - token数=25较短，kernel很小，launch overhead占比高
   - 在ncu-ui看：Duration很短（微秒级），SM Utilization低
   - 优化：融合到Flash Attention中、在线softmax

4. **LayerNorm**:
   - Pre-LN结构：每层2个（attention前+FFN前），3层=6个，加最终1个=7个
   - **典型memory-bound**（读-算-写，计算量小，AI≈2.3）
   - 在ncu-ui看：Memory Utilization高，SM Utilization低；数量多且单个很短
   - 优化：融合LayerNorm+残差+激活、Fused Layer Norm kernel（torch.nn.functional.layer_norm已经是fused kernel）

5. **Element-wise (GELU、残差连接、Dropout)**:
   - FFN中的GELU激活、残差连接、dropout、position/type embedding相加
   - **典型memory-bound**（每元素只做几次运算但要多次读写HBM）
   - 在ncu-ui看：kernel名称通常含"elementwise"，Duration很短（微秒级），launch overhead占比高
   - 优化：torch.compile算子融合（把多个element-wise合并成一个kernel）、手动融合

6. **Embedding lookup (gather/vectorized)**:
   - 所有token的embedding查找：user、item、cate、history(20个)
   - **典型memory-bound**（随机gather，计算量极小，AI=0）
   - 在ncu-ui看：Speed of Light里Memory Utilization应该很高（>60%），SM Utilization低；看Memory Workload Analysis里的L2 Hit Rate（随机访问命中率低是正常的）
   - 优化：减小embedding维度、FP16/INT8量化、GPU embedding cache（NVIDIA Merlin HPS）

---

## 7. OneTrans vs DCN-DIN 性能对比分析要点

用同样的工具分析两个模型后，可以对比：

| 维度 | DCN-DIN | OneTrans |
|---|---|---|
| 主力算子 | Embedding gather + GEMM(MLP) + Element-wise(Cross) | GEMM(注意力+FFN) + Softmax + LayerNorm |
| Kernel数量 | 较少，结构简单 | 较多，每层Transformer有8+个kernel |
| 算力bound程度 | 中等，MLP的GEMM | 较高，自注意力的GEMM密集 |
| 内存bound程度 | Embedding gather占比较高 | LayerNorm、Softmax、残差连接占比 |
| Kernel launch overhead | 较低 | 较高（层数多、kernel细碎） |
| 优化方向 | 融合Cross层element-wise、优化embedding | Flash Attention、融合LN+激活、CUDA Graph |

---

## 8. 完整分析流程建议（对照文档四层定位方法论）

```
第一层（服务端宏观）:
  Step 1: 跑推理压测 python3 inference.py → 看端到端延迟拆解，哪阶段占比最高
          （cpu_preprocess/h2d/nn/d2h/postprocess 各占多少%）
          对比DCN-DIN的延迟拆解，看Transformer模型NN占比是否更高
  Step 2: 看GPU利用率（训练日志gpu_util、推理batch_sweep的gpu_util列）
          GPU<60% → CPU bottleneck；GPU~100%但延迟高 → 计算瓶颈

第二层（框架级）:
  Step 3: torch profiler python3 profile_torch.py → 看CPU/CUDA算子TOP耗时
          定位是哪类算子慢（Attention/GEMM/LayerNorm/Element-wise）
  Step 4: batch_sweep模式 → 不同batch下延迟吞吐对比
          Transformer对batch更敏感，小batch时launch overhead占比极高

第三层（Kernel级）:
  Step 5: nsys profile → 看timeline：GPU是否有gap、kernel是否细碎（Transformer每层8+个）、
          H2D/D2H Memcpy占比、NVTX各阶段占比、Memcpy是否穿插在kernel间
  Step 6: ncu --set full → 对TOP耗时kernel做硬件级分析
          看SM利用率/内存带宽/占用率/Warp stall，判断memory-bound还是compute-bound

第四层（微基准）:
  Step 7: python3 benchmark_op.py → 用torch.utils.benchmark精确测6类算子
          计算每个算子的AI=FLOPs/Bytes，对照Roofline判断瓶颈类型

优化与验证:
  Step 8: 根据分析结果优化（混合精度、Flash Attention、算子融合、CUDA Graph、增大batch等）
  Step 9: 重新跑以上步骤验证优化效果，对比DCN-DIN和OneTrans的性能差异
```

---

## 9. 常见问题

**Q: nsys/ncu 命令找不到？**
A: 确保 CUDA Toolkit 已安装且在 PATH 中。Nsight System 通常在 `/usr/local/cuda/nsight-systems/bin/nsys`，Nsight Compute 在 `/usr/local/cuda/nsight-compute/ncu`。

**Q: ncu 报错 "you don't have permission to profile"？**
A: 需要设置 `/proc/sys/kernel/perf_event_paranoid` 为 <=2，或用 sudo 运行。T4云服务器上通常需要 `sudo sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'`。

**Q: torch profiler 报 CUDA 错误？**
A: 确保 PyTorch 版本与 CUDA 驱动匹配。`python3 -c "import torch; print(torch.version.cuda)"` 查看。

**Q: 训练时loss不下降？**
A: fake数据是随机的，loss不下降是正常的。本工程目的是性能分析，不是模型收敛。

**Q: OneTrans训练比DCN-DIN慢很多正常吗？**
A: 正常。Transformer的自注意力计算量更大，且kernel更多更细碎。可以通过减小 `num_layers`、`embedding_dim`、`ffn_dim` 来加速。
