# DCN + DIN 推荐模型 — 性能分析工程

经典推荐精排模型：**DIN（目标感知注意力，序列建模）+ DCNv2（Cross网络显式高阶特征交叉 + Deep MLP）**。

本工程包含：fake数据生成、模型训练、推理压测、PyTorch Profiler、Nsight System、Nsight Compute 全套性能分析流程。

---

## 快速执行命令（复制即用）

> 所有命令在模型目录下执行。日志只写文件不打控制台，用 `tail -f xxx.log` 查看实时输出。

```bash
# ===== 0. 环境准备 =====
pip install -r requirements.txt
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ===== 1. 生成 Fake 数据（1000条） =====
python3 fake_data.py

# ===== 2. 训练模型（迭代次数在 config.yaml 的 max_steps 控制） =====
python3 train.py

# ===== 3. 推理压测（QPS、请求数在 config.yaml 控制） =====
python3 inference.py

# ===== 4. 算子微基准 + Roofline =====
python3 benchmark_op.py

# ===== 5. torch.profiler 框架级分析（导出 Chrome trace） =====
python3 profile_torch.py
# 用 Perfetto 打开（推荐，加载快）：浏览器访问 https://ui.perfetto.dev/ → Open trace file → 选 trace_chrome.json
# 备选：Chrome 打开 chrome://tracing → Load → 选 trace_chrome.json

# ===== 6. Nsight System 时间线分析 =====
nsys profile --trace=cuda,nvtx,osrt -o nsys_train --force-overwrite true python3 train.py
nsys profile --trace=cuda,nvtx,osrt -o nsys_infer --force-overwrite true python3 inference.py
# 用 nsys-ui 打开 .nsys-rep 文件

# ===== 7. ncu Kernel级分析 =====
# 推荐：speedoflight模式 + 跳过warmup + 只采30个kernel，几分钟内跑完
ncu --set speedoflight --launch-skip 5 --launch-count 30 -o ncu_report --force-overwrite python3 profile_ncu.py
# 深度分析：对top kernel用 --set full --kernel-name "regex:..." 单独采集
# 用 ncu-ui ncu_report.ncu-rep 打开（或下载到本地用 Nsight Compute GUI 打开）
# CSV 用 Excel 打开，按 Total Time 降序排列
```

---

## 目录结构

```
01_dcn_din/
├── config.yaml          # 模型/训练/压测/性能分析配置（调参数改这里）
├── fake_data.py         # fake数据生成（1000条样本）
├── model.py             # DCNv2 + DIN 网络结构
├── train.py             # 训练脚本（NVTX标记、显式H2D、GPU利用率、ETA日志、模型保存）
├── inference.py         # 推理压测（端到端延迟拆解、显式H2D/D2H、CPU预处理模拟、batch sweep）
├── benchmark_op.py      # 微基准：torch.utils.benchmark测各类算子+Roofline分析
├── profile_torch.py     # PyTorch Profiler 分析
├── profile_ncu.py       # Nsight Compute 专用短脚本（推理模式，加载训练好的模型跑推理）
├── requirements.txt     # Python依赖
├── README.md            # 本文档
│
│ 【运行后自动生成的日志文件（所有脚本只写日志文件，不输出到控制台）】
├── fake_data.log        # fake数据生成日志
├── train.log            # 训练日志（loss/step/耗时/ETA）
├── inference.log        # 推理压测日志（QPS/延迟/各阶段占比）
├── benchmark.log        # 微基准日志（算子耗时/Roofline）
├── profile_torch.log    # torch.profiler日志
├── profile_ncu.log      # kernel分析短脚本日志
├── fake_data.pkl        # fake训练数据（1000条）
├── model.pt             # 训练保存的模型文件
├── prof_log/            # torch.profiler的TensorBoard输出
├── trace_chrome.json    # torch.profiler的Chrome trace输出
├── nsys_*.nsys-rep      # Nsight System采集报告
└── ncu_*.ncu-rep        # Nsight Compute采集报告（ncu-ui打开）
```

> **日志说明**：所有脚本的输出都写入对应 `.log` 文件（append模式，多次运行会追加），控制台不会打印任何日志。查看日志用 `tail -f train.log` 或 `cat train.log`。

---

## 性能分析点对照清单（对照《推荐模型演进与性能优化精讲》PART 2）

> 用性能分析工具导出数据后，对着下表逐项对照分析。每个点都能在本工程中找到对应的功能/命令。

| 文档分析点                                | 对应模型功能                                                                                                   | 分析命令/位置                                                       |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **①服务端宏观：端到端延迟拆解**                   | inference.py 分6阶段计时：数据生成→CPU预处理→H2D→NN→D2H→后处理                                                           | `python3 inference.py` 输出 "End-to-End Latency Breakdown"       |
| **①GPU利用率 <60% → CPU bottleneck**    | train.py 日志含 gpu_util；inference.py 可开启 cpu_preprocess 制造GPU等CPU                                          | 训练日志看 `gpu_util=XX%`；nsys看GPU timeline是否有大段空闲                 |
| **①P50 vs P99 尾延迟**                  | inference.py 输出 NN延迟和总延迟的 p50/p90/p99/max                                                                | `python3 inference.py` 输出延迟统计                                  |
| **②框架级：torch.profiler 算子耗时**         | profile_torch.py 采集CPU+CUDA算子，导出TensorBoard/Chrome trace                                                 | `python3 profile_torch.py` → `tensorboard --logdir=./prof_log` |
| **②不同batch/并发下表现**                   | inference.py batch_sweep模式测[1,4,16,64,256]                                                               | config设 `batch_sweep: true` → `python3 inference.py`           |
| **③Kernel级：nsys CPU/GPU交互**          | train.py/inference.py 内置NVTX range：data_load_cpu/h2d_transfer/forward/backward/nn_inference/d2h_transfer | `nsys profile --trace=cuda,nvtx,osrt python3 train.py`         |
| **③Kernel级：H2D/D2H Memcpy占比**        | train.py 显式CPU创建tensor再`.to(device)`；inference.py 显式H2D/D2H，nsys可观测cudaMemcpyHtoD/DtoH                   | `nsys profile --trace=cuda python3 inference.py` → 看Memcpy行    |
| **③小kernel间空闲间隙 → launch overhead**  | 模型本身有大量小kernel（Embedding lookup、Element-wise、LayerNorm），nsys可观察                                          | nsys timeline看kernel间gap；batch_sweep对比不同batch                 |
| **③GPU长时间空闲等CPU**                    | config `enable_cpu_preprocess: true` + `cpu_preprocess_ms: 5` 可放大CPU bottleneck                          | 调大cpu_preprocess_ms后nsys看GPU空闲段是否变长                           |
| **③Memcpy穿插在kernel之间**               | 显式H2D/D2H + CPU预处理，数据依赖导致无法重叠                                                                            | nsys timeline看Memcpy与kernel的交错                                |
| **③ncu单kernel硬件指标**               | profile_ncu.py 跑3步，ncu采集SM利用率/内存带宽/占用率/Warp stall                                                     | `ncu --set full -o report python3 profile_ncu.py` → ncu-ui打开     |
| **③ncu memory-bound vs compute-bound判断** | benchmark_op.py 计算每个算子的AI=FLOPs/Bytes，对照T4平衡点~25；ncu看Speed of Light的SM/Memory利用率                 | `python3 benchmark_op.py` + `ncu_report.ncu-rep`               |
| **④微基准：torch.utils.benchmark**       | benchmark_op.py 精确测6类算子：Embedding/GEMM/Element-wise/Attention/LayerNorm/Concat                           | `python3 benchmark_op.py`                                      |
| **Roofline判断**                       | benchmark_op.py 计算每个算子的计算强度AI，判断memory/compute-bound                                                     | `python3 benchmark_op.py` 汇总表                                  |
| **特征拉取占30-50%（行业参考）**                | inference.py 的 cpu_preprocess 阶段模拟特征拉取，可调耗时占比                                                            | 看Latency Breakdown中cpu_preprocess占比                           |

---

## 0. 环境准备

```bash
cd /path/to/01_dcn_din

# 安装依赖
pip install -r requirements.txt

# 验证GPU（T4）
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 验证Nsight工具已安装
which nsys && nsys --version
which ncu && ncu --version
```

> T4 是 Turing 架构（compute capability 7.5），Nsight System 和 Nsight Compute 均支持。

---

## 1. 生成 Fake 数据

```bash
cd /path/to/01_dcn_din
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
cd /path/to/01_dcn_din
python3 train.py
```

训练逻辑：

- 对 1000 条 fake 样本**无限循环随机采样**，直到达到 `config.yaml` 中的 `max_steps`
- 每 `log_every` 步打印日志：`loss / step / 已耗时 / 平均每步耗时 / 剩余时间(ETA)`
- 训练结束自动保存模型到 `dcn_din_model.pt`

### 控制训练时长

修改 `config.yaml`：

```yaml
max_steps: 500      # 迭代次数，越大训练越久
batch_size: 256     # batch大小
log_every: 10       # 日志打印间隔
```

### 典型输出

> 以下输出写入 `train.log`，用 `cat train.log` 或 `tail -f train.log` 查看。

```
[INFO] Device: cuda
[INFO] GPU: Tesla T4
[INFO] Loaded 1000 samples
[INFO] Model params: 523,xxx
============================================================
[Step    10/500] loss=0.6931  elapsed=    2.1s  avg= 210.5ms/step  ETA=  103.0s
[Step    20/500] loss=0.6850  elapsed=    4.0s  avg= 200.2ms/step  ETA=   96.1s
...
============================================================
[DONE] Training finished in 100.2s
[DONE] Model saved to ./dcn_din_model.pt
```

---

## 3. 推理压测

```bash
cd /path/to/01_dcn_din
python3 inference.py
```

### 3.1 固定QPS压测模式（默认）

```bash
cd /path/to/01_dcn_din
python3 inference.py
```

压测逻辑（端到端6阶段，每阶段独立计时）：

1. **数据生成**（CPU）：生成fake batch数据
2. **CPU预处理**（模拟特征拉取）：`enable_cpu_preprocess: true` 时模拟PS拉取+int化+拼接，耗时由 `cpu_preprocess_ms` 控制
3. **H2D搬运**：`enable_h2d_d2h: true` 时显式 `.to(device)`，nsys可观测cudaMemcpyHtoD
4. **NN推理**：模型前向计算
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
用于分析：小batch时kernel launch overhead占比、大batch时GPU利用率提升、找到延迟吞吐平衡点。

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
  cpu_preprocess    : avg=  2.045ms  ( 38.2%)
  h2d               : avg=  0.032ms  (  0.6%)
  nn                : avg=  3.120ms  ( 58.3%)
  d2h               : avg=  0.015ms  (  0.3%)
  postprocess       : avg=  0.142ms  (  2.7%)
  total             : avg=  5.354ms  (100.0%)

===== NN Inference Latency =====
  avg  : 3.120 ms
  p50  : 2.950 ms
  p99  : 8.200 ms
```

> 分析要点：cpu_preprocess占38.2%说明CPU侧是瓶颈（对应文档"特征拉取占30-50%"）；nn占58.3%是GPU计算；H2D/D2H占比很低说明数据搬运不是瓶颈（可通过调大batch或cpu_preprocess_ms改变占比）。

---

## 3b. 微基准测试（算子级 + Roofline分析）

```bash
cd /path/to/01_dcn_din
python3 benchmark_op.py
```

用 `torch.utils.benchmark` 精确测量6类关键算子，每个算子计算 **计算强度 AI = FLOPs/Bytes**，对照T4的Roofline平衡点（~25 FLOPs/byte）判断 memory-bound 还是 compute-bound：

| 算子                     | 典型瓶颈          | 原因                              |
| ---------------------- | ------------- | ------------------------------- |
| Embedding Lookup       | memory-bound  | 随机gather，AI=0，纯内存访问             |
| 小GEMM (M=batch)        | memory-bound  | M小K大，读写占比高                      |
| 大GEMM (1024³)          | compute-bound | M*N*K都大，AI高                     |
| Element-wise (Mul+Add) | memory-bound  | 每元素2次运算但4次读写                    |
| Attention (短序列)        | memory-bound  | 多次小GEMM+softmax，launch overhead |
| LayerNorm              | memory-bound  | 7次运算但3次读写，AI≈2.3                |
| Concat                 | memory-bound  | 纯数据搬运，AI=0                      |

> 对应文档1.3节Roofline模型和"精排模型典型算子瓶颈类型"表。

---

## 4. PyTorch Profiler 分析（框架级定位）

torch.profiler 是 PyTorch 内置的性能分析工具，用来定位**哪个算子（op）最耗时**，是框架级定位的首选工具。它比nsys更聚焦于PyTorch层面的算子，比ncu更轻量（不需要深入硬件计数器）。

### 4.1 运行采集

```bash
cd /path/to/01_dcn_din
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

| 子标签页         | 看什么                          | 对应分析点                      |
| ------------ | ---------------------------- | -------------------------- |
| **Overview** | 总览：step时间、GPU利用率、TOP算子、性能建议  | 第一眼了解整体情况                  |
| **Operator** | 所有PyTorch算子的表格，按CPU/CUDA时间排序 | **哪个算子最耗时**（核心！）           |
| **Kernel**   | 所有GPU kernel的表格，按CUDA时间排序    | 哪个GPU kernel最耗时            |
| **Trace**    | 时间线视图（类似nsys和chrome trace）   | 看算子执行顺序、kernel间隙、CPU/GPU交互 |
| **Memory**   | 显存分配时间线                      | 峰值显存、内存泄漏                  |
| **Module**   | 按模型模块（nn.Module）分组的耗时        | 模型哪个部分最耗时                  |

#### 各子标签页操作要点

**Overview（总览）：**

- 顶部显示：Average Step Time（平均每步耗时）、GPU Utilization（GPU利用率）、Est. SM Efficiency（SM效率）
- 中间"Performance Recommendations"：PyTorch自动给出的优化建议（比如"GPU is underutilized"、"Large cudaMemcpy"等）
- 底部"Top Operators"：耗时TOP10的算子

**Operator（算子表格，最常用）：**

- 表格列包括：Name（算子名）、Calls（调用次数）、CPU Time（CPU总耗时）、CUDA Time（CUDA总耗时）、% of total（占比）
- **点击列标题可以排序**：点"CUDA Time"按GPU耗时排序，点"CPU Time"按CPU耗时排序
- **看什么**：
  - 按CUDA Time排序 → 哪些算子在GPU上最耗时（通常是GEMM、Embedding、Element-wise）
  - 按CPU Time排序 → 哪些算子在CPU上最耗时（通常是数据处理、Python开销）
  - 看"% of total"列 → 占比最高的几个算子就是优化重点
- **点击某个算子名**可以展开看调用栈（Call Stack），知道这个算子是代码里哪一行调用的

**Trace（时间线）：**

- 类似nsys的时间线，从上到下是CPU线程、CUDA Runtime、GPU kernel
- **操作**：鼠标滚轮缩放，拖拽平移，点击事件看详情
- **看什么**：
  - GPU kernel行是否有大段空白（GPU利用率低）
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
- **搜索**：顶部搜索框输入算子名（如"matmul"、"embedding"）高亮匹配的事件
- **切换线程**：左侧显示线程名，可以折叠/展开
- **SQL 查询**：点击底部 "Query (SQL)" 标签，可以用 SQL 查 trace 数据（比如查所有耗时>1ms的kernel）

**看什么**：

- 时间线上方是CPU线程（Python主线程），下方是GPU stream（CUDA kernel）
- 看GPU kernel之间是否有空白（launch overhead或CPU bottleneck）
- 点击某个kernel看它的名称和耗时

> **备选工具：chrome://tracing**（旧工具，大文件加载慢）
> Chrome 浏览器地址栏输入 `chrome://tracing` → 左上角 "Load" 按钮 → 选择 `trace_chrome.json`。操作方式类似，但 trace 文件大时加载可能卡顿，推荐用 Perfetto。

### 4.4 终端表格（脚本运行结束自动打印）

profile_torch.py 运行结束后会自动在终端打印两个表格：

- **CPU Operators TOP20**：按CPU时间排序的算子
- **CUDA Kernels TOP20**：按CUDA时间排序的kernel

适合快速看一眼，不需要打开浏览器。

### 4.5 重点关注（DCN-DIN模型）

| 指标                 | 在TensorBoard哪里看                        | 说明                                                            |
| ------------------ | -------------------------------------- | ------------------------------------------------------------- |
| **哪个算子最耗时**        | Operator标签页 → 按CUDA Time排序             | DCN-DIN通常是：Embedding lookup、GEMM(Linear)、Element-wise(Cross层) |
| **GPU利用率**         | Overview → GPU Utilization             | <60%说明CPU bottleneck（数据准备慢或kernel launch overhead）            |
| **CPU vs GPU时间占比** | Operator表格 → CPU Time vs CUDA Time列    | CPU时间远大于GPU时间 → CPU bottleneck                                |
| **kernel间隙**       | Trace标签页 → GPU kernel行                 | 大段空白说明GPU在等CPU                                                |
| **峰值显存**           | Memory标签页                              | 看是否超出T4的16GB                                                  |
| **自动优化建议**         | Overview → Performance Recommendations | PyTorch自动给出的建议，参考即可                                           |

---

## 5. Nsight System 分析（框架级 + GPU timeline）

Nsight Systems（简称 nsys）看**整体时间线**：CPU端操作、CUDA API调用、GPU kernel执行、NVTX标记范围、**H2D/D2H内存拷贝**、kernel之间的空闲间隙等。它是性能分析的"全景图"，用来定位"时间花在哪了"。

### 5.1 采集训练过程

```bash
cd /path/to/01_dcn_din

# 建议先把config.yaml的max_steps改小（如50~100），避免采集文件太大
nsys profile \
  --output=nsys_train \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  python3 train.py
```

**每个参数的含义（初学者必读）：**

| 参数                                    | 含义                                 | 为什么需要           |
| ------------------------------------- | ---------------------------------- | --------------- |
| `profile`                             | nsys的子命令，表示开始采集                    | 固定写法            |
| `--output=nsys_train`                 | 输出文件名前缀，最终生成 `nsys_train.nsys-rep` | 方便区分训练/推理的报告    |
| `--force-overwrite=true`              | 如果同名文件已存在则覆盖                       | 反复调试时不用手动删旧文件   |
| `--trace=cuda,nvtx,osrt,cudnn,cublas` | 要追踪的事件类型，逗号分隔                      | 见下方详解           |
| `--cuda-memory-usage=true`            | 追踪CUDA内存分配/释放事件                    | 看显存占用变化、是否有内存碎片 |

**`--trace` 各事件类型详解：**

- `cuda`：追踪CUDA API调用（cudaLaunchKernel、cudaMemcpy、cudaMalloc等）和GPU kernel执行。**最核心，必选**
- `nvtx`：追踪代码中用NVTX标记的范围（我们在train.py里埋了step_N、data_load_cpu、h2d_transfer、forward、backward等标记）。**必选，否则看不到我们埋的点**
- `osrt`：追踪OS运行时事件（线程创建、互斥锁、系统调用等）。可选，帮助排查CPU侧的线程同步问题
- `cudnn`：追踪cuDNN库调用（我们模型用得少，可选）
- `cublas`：追踪cuBLAS库调用（GEMM矩阵乘走cuBLAS，可选但有用）

**产出文件：** `nsys_train.nsys-rep`（二进制报告文件，需要用nsys-ui打开）

### 5.2 采集推理过程

```bash
cd /path/to/01_dcn_din

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
└─────────────────────────────────────────────────────┘
```

#### 基本操作（初学者必读）

| 操作          | 方法                                         |
| ----------- | ------------------------------------------ |
| **放大/缩小**   | 鼠标滚轮（光标位置为中心缩放）；或工具栏 +/- 按钮                |
| **左右平移**    | 按住Shift + 鼠标拖拽；或横向滚动条                      |
| **选中某个事件**  | 鼠标左键点击时间线上的色块                              |
| **查看事件详情**  | 选中后，底部面板（Properties）显示该事件的名称、起止时间、持续时长、线程等 |
| **过滤**      | 工具栏有个过滤框（Filter），输入关键词如"Memcpy"只看内存拷贝      |
| **跳到某个时间点** | 底部时间轴可以拖拽                                  |

#### 对照分析点：看什么、在哪里看

| 你想看的分析点               | 在nsys-ui里怎么看                                                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **H2D/D2H数据搬运占比**     | 看第④行"CUDA Memory Operations"，找橙色/红色的"Memcpy HtoD"（H2D）和"Memcpy DtoH"（D2H）色块。选中后看Properties里的Duration。和第⑤行kernel总时长对比，就知道搬运占比 |
| **GPU kernel之间的空闲间隙** | 看第⑤行"CUDA Kernels"，如果kernel色块之间有大量白色空白，说明GPU在等CPU（launch overhead或数据准备慢）。空白越多，GPU利用率越低                                       |
| **NVTX各阶段占比**         | 看第②行"NVTX"，每个step_N包含data_load_cpu→h2d_transfer→forward→backward几个子段。选中某个子段看Duration，就能知道forward和backward各占多少                |
| **GPU利用率**            | 看第⑤行kernel色块是否"铺满"时间线。如果大部分时间都有kernel在跑，利用率高；如果大片空白，利用率低。也可以用nsys stats统计                                                    |
| **Memcpy穿插在kernel之间** | 同时看第④行和第⑤行，如果Memcpy色块和kernel色块在时间上交错出现，说明数据依赖导致无法重叠（理想情况是Memcpy和kernel并行）                                                    |
| **CPU侧瓶颈**            | 看第①行CPU线程，如果CPU函数调用很长而GPU kernel行有空白，说明CPU在忙而GPU在等                                                                           |
| **某个kernel的具体耗时**     | 在第⑤行点击该kernel色块，底部Properties显示Kernel Name、Duration、Grid/Block尺寸等                                                             |

#### 常用过滤技巧

在工具栏的Filter框输入：

- `Memcpy`：只显示内存拷贝相关事件（快速定位H2D/D2H）
- `kernel`：只显示kernel相关
- `nvtx`：只显示NVTX标记
- `cudaLaunchKernel`：只显示kernel launch API调用

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

### 5.5 重点关注（DCN-DIN模型）

| 观察项                         | 在nsys里怎么看                            | 说明                                                                             | 优化方向                                  |
| --------------------------- | ------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------- |
| **H2D搬运占比**                 | 第④行Memcpy HtoD总时长 / 第⑤行kernel总时长     | 我们在train.py里显式做了CPU→GPU搬运，正常应该占比很小（<5%）。如果占比高说明batch太大或数据准备慢                   | 用pin_memory、non_blocking=True、减小batch |
| **kernel间空闲间隙**             | 第⑤行kernel色块之间的空白                     | DCN-DIN的kernel不多（Embedding+GEMM+Element-wise），间隙应该较小。如果间隙大说明CPU侧data_load或h2d慢 | 优化数据pipeline、CUDA Graph               |
| **forward vs backward占比**   | 第②行NVTX里forward和backward段的时长对比       | 正常backward约为forward的2倍（反向传播计算量更大）。如果比例异常需要排查                                   | —                                     |
| **Element-wise kernel是否细碎** | 第⑤行找名称含"elementwise"的kernel，看数量和单个时长 | DCN的Cross层有多个逐元素乘加kernel，如果数量多且单个很短，launch overhead大                           | torch.compile算子融合                     |
| **GPU利用率**                  | 第⑤行kernel是否铺满时间线                     | 训练时应该>80%；如果低说明CPU bottleneck或batch太小                                          | 增大batch、优化数据pipeline                  |

---

## 6. Nsight Compute 分析（GPU Kernel级深度定位）

Nsight Compute（简称 ncu）深入**单个GPU kernel的硬件级指标**：SM算力利用率、显存带宽利用率、占用率(Occupancy)、Warp调度效率、指令吞吐、L1/L2缓存命中率等。它是性能分析的"显微镜"，用来回答"这个kernel为什么慢"。

> **初学者提示**：ncu采集会让程序变慢10~50倍（因为每个kernel都要采集大量硬件计数器）。`profile_ncu.py`已优化为轻量版：用`ncu_batch_size=64`（比推理batch 512小很多，kernel类型完全一样）、预生成数据、只跑1个推理step。配合下面的`--set speedoflight --launch-count 30`，几分钟内就能跑完。

### 6.1 快速采集（推荐，几分钟跑完）

```bash
cd /path/to/01_dcn_din

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
# 只采集GEMM/矩阵乘kernel（DCN的低秩矩阵乘、Deep MLP全连接层）
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:.*gemm.*|.*matmul.*|.*linear.*|.*sgemm.*" \
  --set full \
  -o ncu_gemm \
  --force-overwrite \
  python3 profile_ncu.py

# 只采集Embedding lookup/gather kernel
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:.*embedding.*|.*gather.*|.*index.*|.*vectorized.*" \
  --set full \
  -o ncu_embedding \
  --force-overwrite \
  python3 profile_ncu.py

# 只采集Element-wise kernel（Cross层的逐元素乘、PReLU激活）
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:.*elementwise.*|.*mul.*|.*add.*|.*prelu.*|.*relu.*" \
  --set full \
  -o ncu_elementwise \
  --force-overwrite \
  python3 profile_ncu.py

# 只采集Softmax kernel（DIN注意力的softmax）
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:.*softmax.*" \
  --set full \
  -o ncu_softmax \
  --force-overwrite \
  python3 profile_ncu.py
```

**参数含义：**
- `--kernel-name-base demangled`：用C++函数名（demangled，人类可读的名字）匹配kernel，而不是编译后的mangled名字
- `--kernel-name "regex:..."`：用正则表达式匹配kernel名称，只采集匹配的kernel。`regex:`前缀表示后面是正则

### 6.4 跳过前N个kernel launch（跳过warmup）

前几个kernel launch通常是warmup（cuBLAS初始化、显存分配等），不是正常计算。可以跳过：

```bash
# 跳过前100个kernel launch，只采集后面的50个
ncu \
  --launch-skip 100 \
  --launch-count 50 \
  --set full \
  -o ncu_skip \
  --force-overwrite \
  python3 profile_ncu.py
```

- `--launch-skip 100`：跳过前100次kernel launch
- `--launch-count 50`：只采集50次kernel launch

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
  - 两个都低 → **latency-bound**（延迟瓶颈，通常是kernel太小、launch overhead大或warp stall多）

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
- **Tensor Core Utilization（Tensor Core利用率）**：用了多少Tensor Core。推荐模型的小GEMM通常Tensor Core利用率低（因为矩阵太小）
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

### 6.8 DCN-DIN 模型典型kernel分析要点

1. **Embedding lookup (gather/vectorized kernel)**:
   - 推荐模型的embedding lookup是**典型memory-bound**算子（随机gather，计算量极小）
   - 在ncu-ui看：Speed of Light里Memory Utilization应该很高（>60%），SM Utilization低
   - 看Memory Workload Analysis里的L2 Hit Rate（随机访问命中率低是正常的）
   - 优化：减小embedding维度、FP16/INT8量化、GPU embedding cache（NVIDIA Merlin HPS）

2. **GEMM (矩阵乘, cublas_sgemm)**:
   - DCN的低秩矩阵乘（U和V）、Deep MLP的全连接层都是GEMM
   - 模型参数较大（embedding_dim=256, deep_hidden=[4096,2048,1024,512]），GEMM尺寸中等
   - 在ncu-ui看：Speed of Light判断bound类型；看Tensor Core Utilization（小GEMM通常<10%，因为矩阵太小用不上Tensor Core）
   - 优化：FP16混合精度（T4支持FP16 Tensor Core）、确保矩阵维度是8的倍数、增大batch让M变大

3. **Element-wise (Cross层逐元素乘、PReLU)**:
   - Cross层的 `x0 * transformed + bias + xl` 是多个element-wise操作
   - **典型memory-bound**（每元素只做几次运算但要多次读写HBM）
   - 在ncu-ui看：kernel名称通常含"elementwise"，Duration很短（微秒级），launch overhead占比高
   - 优化：torch.compile算子融合（把多个element-wise合并成一个kernel）、手动融合

4. **Softmax (DIN注意力)**:
   - DIN的attention softmax，序列长度500，token数较多
   - kernel较小，可能launch overhead大
   - 优化：融合scale+mask+softmax（Flash Attention思路），但短序列收益有限

---

## 7. 完整分析流程建议（对照文档四层定位方法论）

```
第一层（服务端宏观）:
  Step 1: 跑推理压测 python3 inference.py → 看端到端延迟拆解，哪阶段占比最高
          （cpu_preprocess/h2d/nn/d2h/postprocess 各占多少%）
  Step 2: 看GPU利用率（训练日志gpu_util、推理batch_sweep的gpu_util列）
          GPU<60% → CPU bottleneck；GPU~100%但延迟高 → 计算瓶颈

第二层（框架级）:
  Step 3: torch profiler python3 profile_torch.py → 看CPU/CUDA算子TOP耗时
          定位是哪类算子慢（Embedding/GEMM/Element-wise/Attention）
  Step 4: batch_sweep模式 → 不同batch下延迟吞吐对比，找launch overhead和平衡点

第三层（Kernel级）:
  Step 5: nsys profile → 看timeline：GPU是否有gap、kernel是否细碎、
          H2D/D2H Memcpy占比、NVTX各阶段占比、Memcpy是否穿插在kernel间
  Step 6: ncu --set full → 对TOP耗时kernel做硬件级分析
          看SM利用率/内存带宽/占用率/Warp stall，判断memory-bound还是compute-bound（Excel打开CSV）

第四层（微基准）:
  Step 7: python3 benchmark_op.py → 用torch.utils.benchmark精确测6类算子
          计算每个算子的AI=FLOPs/Bytes，对照Roofline判断瓶颈类型

优化与验证:
  Step 8: 根据分析结果优化（混合精度、算子融合、增大batch、减少H2D/D2H、CPU预处理上移等）
  Step 9: 重新跑以上步骤验证优化效果，对比优化前后的延迟拆解和算子耗时
```

---

## 8. 常见问题

**Q: nsys/ncu 命令找不到？**
A: 确保 CUDA Toolkit 已安装且在 PATH 中。Nsight System 通常在 `/usr/lib/nsight-systems/bin/nsys`，Nsight Compute 在 `/opt/nvidia/nsight-compute/<版本>/ncu`。找不到时建软链接：`sudo ln -sf /实际路径/nsys /usr/local/bin/nsys`。

**Q: ncu 报 "permission" 或 "ERR_NVGPUCTRPERM"？**
A: 采集硬件计数器需要 root 权限，加 `sudo` 运行。或持久修复：`sudo tee /etc/modprobe.d/nvidia-profiling.conf <<< 'options nvidia NVreg_RestrictProfilingToAdminUsers=0'` 然后重启。

**Q: ncu 报 "Cuda driver is not compatible"？**
A: ncu 版本与 CUDA 驱动不匹配。T4 + 驱动570(CUDA 12.8) 推荐安装 ncu 2024.3.0 中间版本（从 NVIDIA 官网下载 .deb），2021版太老报Error 36，2025版太新报驱动不兼容。

**Q: torch profiler 报 CUDA 错误？**
A: 确保 PyTorch 版本与 CUDA 驱动匹配。`python3 -c "import torch; print(torch.version.cuda)"` 查看。

**Q: 训练时loss不下降？**
A: fake数据是随机的，loss不下降是正常的。本工程目的是性能分析，不是模型收敛。
