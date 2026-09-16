# 03_rank_simulate —— 精排打分模型（故意埋两个性能坑）

> **学习目的**：这是一个**刻意造出两个性能问题**的精排推理工程，供你练习用 torch.profiler / nsys / ncu 定位瓶颈。模型结构是推荐精排场景的"单塔打分"（一个用户 + N 个候选 item 逐个打分），但 `model.py` 里**故意**埋了两个坑——你的任务是通过 perf 分析把这两个坑各自找出来，并得出对应的优化结论。

---

## 一、两个故意埋的坑（先看清楚坑在哪，再去看 perf）

### 坑 2：user 特征冗余计算 N 次（N = 候选 item 数）

**场景**：精排里，一个用户对 N 个候选 item 打分。用户侧的特征（历史序列、用户画像）对**所有候选 item 是共享的**，本该只算一次。

**代码位置**（`model.py` 的 `forward`）：

```python
for i in range(item_ids.shape[0]):        # N 个候选 item 逐个打分
    u = self.user_tower(hist)             # ★ 冗余：user_tower 对每个 item 都重算一遍
    v = self.item_tower(self.item_emb(item_ids[i]))
    s = self.score_head(torch.cat([u, v]))
```

`user_tower(hist)` 的结果与 `i` 无关（输入 `hist` 不变），却放在了循环里，被重复计算了 N 次。`user_tower` 内部有历史序列投影 GEMM（`[hist_len, emb_dim] → [hist_len, 128]`）+ 均值池化 + 两层 MLP，是**有明确耗时**的，重复 N 次就是白烧 GPU。

**预期结论**（你应该通过分析得出）：

> user_tower（用户侧计算）对 N 个候选完全重复，应该**提出循环外、只算一次，缓存复用**。这是精排"多候选共享用户侧计算"的经典优化点。

**怎么在 perf 里发现**（详见第四节）：
- Perfetto 里搜 `user_tower`（NVTX 区间），看到它重复 N 次；或看 `user_mlp` 的 GEMM kernel（`aten::addmm` / `linear`）的 `# of Calls` = N × 步数
- `profile_torch.log` 里 key_averages 表，`aten::linear` 的调用次数约等于 N 的倍数

### 坑 1：launch overhead（kernel 碎片化，大量微小算子）

**场景**：item 侧特征处理里塞了一堆独立的微小 element-wise 算子，每个 kernel 在 GPU 上只跑 1~2µs，但 CPU 下发 + Python 调度的 launch 开销比执行时间还长，导致 GPU 时间轴上 kernel 之间出现大量空隙（GPU 空等 CPU 喂 kernel）。

**代码位置**（`model.py` 的 `item_tower`）：

```python
for i in range(self.small_w.shape[0]):     # n_small_ops 次（默认 30）
    x = x * self.small_w[i] + self.small_b[i]   # 1 个 element-wise kernel
    x = torch.relu(x)                           # 又 1 个 kernel
```

30 次循环 = 60 个独立 kernel，每个都极快，launch 开销主导。

**预期结论**（你应该通过分析得出）：

> item_tower 里有几十个微小 kernel，launch 开销占比高，应该**算子融合**（几十个小 op 合成一个 kernel）或上 **CUDA Graph**。

**怎么在 perf 里发现**（详见第四节）：
- Perfetto 的 GPU 泳道里，`item_tower` 区间内是一堆 1~2µs 的 `elementwise_kernel` / `mul` / `relu` 小块，之间有明显空隙
- 用 SQL 统计：每个 kernel 平均耗时极小（~1µs），但 kernel 数量巨大（30 × 2 × N 个）

---

## 二、目录结构

```
03_rank_simulate/
├── config.yaml          # 模型/训练/性能分析参数（调这里）
├── fake_data.py         # fake 数据生成（1 用户 + N 候选 item）
├── model.py             # RankModel（★ 两个坑都埋在这里）
├── train.py             # 训练脚本（NVTX 标记）
├── inference.py         # 纯推理脚本（供 nsys 采集，无 torch.profiler）
├── profile_torch.py     # torch.profiler 采集（推理模式）
├── profile_ncu.py       # Nsight Compute 采集
├── fix_trace_time.py    # trace 时间戳修复（复用 01_dcn_din）
├── requirements.txt     # 依赖
└── README.md            # 本文档
```

---

## 三、快速开始（服务器上执行）

```bash
# 1. 生成 fake 数据 + 训练
python3 fake_data.py
python3 train.py                 # 训练完得到 rank_model.pt，日志在 train.log

# 2. torch.profiler 采集（Perfetto 打开）
python3 profile_torch.py         # → trace_chrome.json + profile_torch.log
#    浏览器打开 https://ui.perfetto.dev/ → Open trace file → trace_chrome.json

# 3. nsys 采集（Nsight Systems 打开）
nsys profile --trace=cuda,nvtx,osrt \
  --force-overwrite=true -o nsys_rank python3 inference.py
#    注意：nsys 只能包 inference.py（纯推理、无 torch.profiler）。
#    千万别包 profile_torch.py —— 它内部用了 torch.profiler（自己订阅 CUPTI），
#    会报 CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED，nsys 采不到 GPU 活动（报告是空壳）。

# 4. ncu 采集（Nsight Compute，必须 sudo）
sudo ncu --set basic --launch-skip 5 --launch-count 100 \
  -o ncu_report --force-overwrite python3 profile_ncu.py
```

---

## 四、两个坑的分析指引（对着 perf 图看）

### 4.1 分析坑 2（user 冗余）：在 Perfetto 里数重复

1. 打开 `trace_chrome.json`，`Ctrl+F` 搜 `user_tower`
2. 你会看到 **N 个（默认 32）几乎一样的 `user_tower` 区间**，一个接一个排开——每个区间里的 kernel 都一样（`user_proj` 的 GEMM、`mean`、`user_mlp` 的 GEMM）
3. 或者看 `profile_torch.log` 里 CUDA Kernels 表的 `# of Calls`：`user_mlp` / `linear` 相关的 kernel 调用次数 ≈ N × 3 步
4. **结论**：`user_tower` 对 N 个候选是同一份计算，重复了 N 次 → 提出循环外只算一次

**修复对照**（`model.py`）：

```python
# 修复前：user_tower 在循环里，重复 N 次
for i in range(N):
    u = self.user_tower(hist)
    ...

# 修复后：user_tower 提出循环，只算一次
u = self.user_tower(hist)          # ← 移到循环外
for i in range(N):
    v = self.item_tower(self.item_emb(item_ids[i]))
    s = self.score_head(torch.cat([u, v]))
    ...
```

### 4.2 分析坑 1（launch overhead）：在 Perfetto 里看空隙

1. 打开 `trace_chrome.json`，搜 `item_tower` 或直接看 GPU 泳道（`Stream` 行）
2. 你会看到 **一堆微秒级的小方块**（`elementwise_kernel`、`mul`、`relu`），**方块之间有明显空隙**——GPU 大部分时间在等下一个 kernel，而不是在算
3. 用 SQL 量化（Perfetto 左下角 Query）：

```sql
-- 看 elementwise 类 kernel 的数量和平均耗时
select name, count(*) as calls, avg(dur) as avg_us, sum(dur)/1000.0 as total_ms
from slice
where category = 'kernel' and name like '%elementwise%'
group by name
order by total_ms desc;
```

4. **结论**：每个 kernel 平均只有 ~1µs，但数量巨大 → launch 开销主导 → 算子融合 / CUDA Graph

**修复对照**（`model.py` 的 `item_tower`）：

```python
# 修复前：30 次循环 = 60 个独立 kernel
for i in range(n):
    x = x * self.small_w[i] + self.small_b[i]
    x = torch.relu(x)

# 修复后（任选其一）：
# 方案 A：向量化融合 —— 把 30 次标量乘加合成一次向量操作
x = torch.relu(x * self.small_w + self.small_b)   # 1~2 个 kernel

# 方案 B：torch.compile 自动融合
# 方案 C：CUDA Graph 消除 launch 开销
```

### 4.3 用 nsys / ncu 交叉验证

- **nsys**：`nsys_rank.nsys-rep`（用 `inference.py` 采集）里，NVTX 行会直接显示 `user_tower` / `item_tower` 区间（各自重复 N 次），`item_tower` 区间内能明显看到大量 kernel 之间的 gap
- **ncu**：`profile_ncu.py` 用 `ncu_candidates=8`，采集的 kernel 里会有一大堆 `elementwise_kernel`（坑1 的铁证）；`--set basic` 看这些 elementwise kernel 的 SOL，会发现 SM/Memory 都很低（因为每个 kernel 太小，纯 latency/launch bound）

---

## 五、预期能看到的"修复收益"

| 坑 | 修复前 | 修复后 | 关键变化 |
|---|---|---|---|
| 坑2（user 冗余） | user_tower 跑 N 次 | user_tower 跑 1 次 | `user_tower` 相关 kernel 的 `# of Calls` 从 N×步数 降到 步数 |
| 坑1（launch overhead） | 60 个小 kernel | 1~2 个融合 kernel | kernel 总数骤降，GPU 时间轴空隙消失 |

> **练习目标**：先自己 perf 分析，得出两个结论，再改 `model.py` 修复，重跑 perf 对比 kernel 数量和总耗时变化。

---

## 六、常见问题

**Q: 训练很慢？**
A: 正常——坑1 故意造了 N×60 个小 kernel，每个都有 launch 开销，训练必然慢。这正是"没优化的模型"该有的样子。`max_steps` 默认 30，`n_small_ops` 默认 30，已经压得比较小。

**Q: 想让某个坑更明显？**
A: 改 `config.yaml`：`n_small_ops` 调大 → 坑1 更明显；`num_candidates` 调大 → 坑2 更明显。

**Q: ncu 报 `ERR_NVGPUCTRPERM` / 报告只有 kernel 名单没有指标？**
A: 没加 `sudo`。GPU 计数器是 admin-only，加 `sudo ncu` 重新采集。

**Q: trace 打开后时间轴显示成"年"？**
A: `profile_torch.py` 已自动调用 `fix_trace_time.py` 归一化。单独修：`python3 fix_trace_time.py trace_chrome.json --in-place`。

**Q: nsys 报 `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED (39)` / `CUDA profiler activities will be missing`？**
A: 你把 nsys 套在了带 `torch.profiler` 的脚本（`profile_torch.py`）外面。torch.profiler 自己订阅了 CUPTI，nsys 也订阅 CUPTI，同一进程只能有一个订阅者。**nsys 只能包纯推理脚本 `inference.py`**（无 profiler）。torch.profiler 和 nsys 是两条独立分析路径，分开跑，不能叠加。
