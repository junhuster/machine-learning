# 任务2：DQRM 简化版（DLRM 结构）+ PyTorch 官方工具做 PTQ

复现论文 **DQRM: Deep Quantized Recommendation Models**（Zhou et al., 2024, arXiv:2410.20046，
Meta / Berkeley / Intel）的**模型结构**，用 **PyTorch 官方 `torch.ao.quantization` 做训练后量化（PTQ）**，
导出带 Q/DQ 的 ONNX，再用 TensorRT 建 INT8 engine，最后压测 **FP32 vs INT8**。

> **本项目只做训练后量化（PTQ），不做量化感知训练（QAT）。**
> 原论文用的是 QAT（还把量化当正则化），但这里按需求只做 PTQ。

---

## 1. 环境要求

| 组件 | 必需 | 说明 |
|---|---|---|
| GPU | ✅ | 建议 T4（sm_75，支持 INT8 Tensor Core） |
| `torch >= 2.0` | ✅ | 用到 `torch.ao.quantization.observer` / `torch.onnx.export` |
| `onnx` / `onnxruntime` | ✅ | onnxruntime 用于导出自检（**必须**，见第 5 节） |
| `tensorrt`（Python 包） | ✅ | `build_engine.py` / `infer.py` |
| numpy / pyyaml | ✅ | 数据与配置 |

`torch.ao.quantization` 是 torch 自带的，**不需要额外装包**。

先跑一次环境探测：`python3 ../common/check_env.py`

> ⚠️ 与任务1 的区别：任务1 用 NVIDIA 的 `pytorch-quantization`；本任务用 **PyTorch 官方**。

---

## 2. 模型：DQRM 简化版（DLRM 结构）

```
dense 特征 [B,128] ──▶ bottom MLP ──▶ d [B,256]
sparse 特征 [B,26] ──▶ 26 张 Embedding 表 ──▶ e_1..e_26 [B,256]
                                  │
      interaction: <d, e_i> 逐列点积 ──▶ [B,26]
                                  │
      concat([d, interaction]) ──▶ top MLP [4096,2048,1024] ──▶ logit
```

**与论文的差异（重要）**

| | 原论文 DQRM | 本项目 |
|---|---|---|
| 量化对象 | **embedding 表**（INT4，为了省显存 / 上端侧） | **dense 部分（bottom/top MLP）**（为了在线推理提速） |
| 精度 | INT4 | **INT8**（T4 没有 INT4 推理加速，做 INT4 只能看精度不能看速度） |
| 训练方式 | QAT + 分布式梯度压缩 + Lazy Quantization | **只做 PTQ** |
| 结论 | embedding 对低精度鲁棒、MLP 更敏感 | 同向印证（见第 6 节） |

> 论文的 Lazy Quantization 与梯度压缩属于**分布式训练**优化，不属于本次推理量化的范围。

**参数分布**（`python3 model.py` 可打印）：dense 约 12M，embedding 约 53M。
**注意**：embedding 占了参数和访存的大头，但它**不会被量化**（见第 4 节），这直接决定了 INT8 的收益上限。

---

## 3. 操作步骤

```bash
cd 02_dqrm_int8

# ① 造数据（固定 seed，切 train/val/calib）
python3 gen_data.py
#   预期: train=56000 val=16000 calib=8000 | 正样本率≈0.5 | outputs/fake_data.npz

# ② 训练 FP32 基线
python3 train.py
#   预期: 打印 dense/embedding 参数量；每 epoch 的 loss 与 val_auc
#         outputs/model.pt

# ③ 导出 FP32 ONNX
python3 export_onnx.py --check
#   预期: onnxruntime vs torch 误差 ~1e-08

# ④ 逐层敏感度分析（torch.ao 观察器校准一次后逐层开关，很快）
python3 sensitivity.py
#   预期: 逐层打印 ΔAUC（单独量化）与「回血」（leave-one-out）
#         outputs/sensitivity.csv + outputs/sensitive_layers.txt

# ⑤ PTQ 量化 + 导出 Q/DQ ONNX
python3 quantize_qdq.py
#   预期: 打印每层的 act/weight scale 与 zero_point；
#         "ONNX 校验通过: QuantizeLinear=N, DequantizeLinear=N"；
#         三个自检数字（见第 5 节）
#         outputs/model_quant_qdq.onnx
#   对照: python3 quantize_qdq.py --all-layers   # 不跳过敏感层

# ⑥ 建 TensorRT engine
python3 build_engine.py
#   预期: outputs/model_fp32.plan 与 outputs/model_int8.plan
#   看哪些层真的走了 INT8: python3 build_engine.py --only int8 --verbose 2>&1 | grep -i int8

# ⑦ 压测对比
python3 bench_compare.py
#   预期: 各 batch 的 mean/p99/吞吐 + 加速比；末尾给结论
#         outputs/bench_report.md + outputs/bench_stats.csv

# ⑧ 单条推理
python3 infer.py --engine outputs/model_int8.plan --batch 32 --compare
```

一键跑完：`bash run_all.sh`

---

## 4. torch.ao 在本项目里具体怎么用（面试可讲）

### 用它的哪一部分

**量化参数由 `torch.ao.quantization.observer.MinMaxObserver` 求出** ——
这正是 `torch.ao.quantization.quantize_fx` 在 FX graph mode PTQ 内部用的**同一套观察器与
`calculate_qparams()` 算法**：

```python
from torch.ao.quantization.observer import MinMaxObserver

# 激活：对称 per-tensor int8
act_obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
# 权重：对称 per-tensor int8
w_obs   = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)

act_obs(x_calib);  scale_a, zp_a = act_obs.calculate_qparams()
w_obs(weight);     scale_w, zp_w = w_obs.calculate_qparams()
```

求出参数后，用标准 Q/DQ 写进 ONNX：

```python
q = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)   # → QuantizeLinear
x_dq = torch.dequantize(q)                                        # → DequantizeLinear
```

`ao_quant.py` 里的 `compute_qparams()` 用 forward hook 把每层输入喂给观察器，
`QDQLinear` 负责按开关做 Q/DQ，`set_enabled()` 支持逐层开关（敏感度分析就靠它）。

### 为什么不直接导出 torch.ao 的 FX 量化模型

在 torch 2.x 上，把 FX 量化图直接 `torch.onnx.export` 有**三处硬限制**（本项目实测）：

| 现象 | 报错 |
|---|---|
| 量化张量的 `Concat` | `ONNX symbolic expected the output of onnx::Concat to be a quantized tensor` |
| 权重 per-channel | `aten::quantize_per_channel` **没有 ONNX symbolic**，直接 `UnsupportedOperatorError` |
| `convert_fx(..., is_reference=True)` 的 reference 模型 | 直接导出报 `RuntimeError: ... Got int`；先 `torch.jit.trace` 再导出会得到**数值错误**的图（实测 reference vs ORT 差 0.28） |

所以本项目**分层处理**：torch.ao 负责「算量化参数」，标准 Q/DQ 负责「写进 ONNX」。
这样既用到了 torch 官方工具，又能保证导出的图正确（并有自检兜底）。

### 一个必须知道的坑：激活别用 uint8 非对称

torch 2.2 导出 `quint8`（非对称，zero_point≠0）的激活 Q/DQ 时，会在 `QuantizeLinear`
（输出 uint8）之后插一个 `Cast`。实测当 `zero_point=0` 时数值会出错：
**误差从 5e-08 飙升到 0.26**（多层累积）。

→ 所以本项目**激活与权重统一用对称 int8**（`qint8` + `per_tensor_symmetric`，zero_point=0），
此时图里只有 `INT8` 一种 Cast。

### 为什么 embedding 不量化

`torch.ao`（以及 TensorRT 的显式量化路径）**默认都不会量化 `nn.Embedding`**：
它对应 ONNX 的 `Gather`，是稀疏随机访存，没有 INT8 加速收益，而且真实场景里 embedding 表在
CPU / 分布式参数服务器上。所以 `list_quantizable_names()` 只收集 `nn.Linear`。

**这正是本任务要展示的核心结论：推荐精排模型的量化收益，取决于 dense 与 sparse 的比例。**

---

## 5. 导出自检：三个数字怎么读

`quantize_qdq.py` 跑完会打印：

```
自检 ① logit 量级 |max| = 1.114
自检 ② FP32 -> INT8 差异 = 1.838e-02   (量化本身的代价)
自检 ③ INT8 -> ONNX 差异 = 1.215e-02   (导出是否忠实)
判读：③ 应显著小于 ②；若 ③ 与 ② 同量级，导出可能有问题
```

- **①** 给个参照物：量化差异要和 logit 本身量级比才有意义。
- **②** 是量化必然带来的偏差（越靠近 0，说明模型越抗量化）。
- **③** 是「torch 算的 INT8」与「导出的 ONNX 在 ONNX Runtime 上跑的 INT8」之间的差。
  **③ ≈ 1e-07 说明导出完全忠实**；本项目实测逐层都是 ~1.5e-07，误差随层数线性累积
  （6 层全量化约 7e-03），属于 torch 的 `quantize_per_tensor` 与 ONNX 的 `QuantizeLinear`
  在**舍入边界上的实现差异**，不是结构错误。

> ⚠️ **自检必须把 onnxruntime 的图优化关到 BASIC**：
> ```python
> so = ort.SessionOptions()
> so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
> ```
> 用默认的 `ORT_ENABLE_ALL` 时，ORT 会**错误地处理本图的 Q→DQ 模式**，
> 让自检报出 2.4e-01 的假阳性（实测：全开 2.4e-01 / 仅基础 3.6e-07 / 关闭 3.6e-07）。
> 导出的图本身是对的，TensorRT 按显式量化规范解析，不受此影响。

**权威精度以 `bench_compare.py` 里 TRT engine 实测的 AUC 为准**（自检只是流程校验）。

---

## 6. 敏感度分析（面试可讲）

用 `QConfigMapping`-style 的逐层开关（`set_enabled`），两条互补证据：

| 方法 | 做法 | 含义 |
|---|---|---|
| **A. per-layer-only** | 只开启第 i 层量化，其余 FP32 → ΔAUC | 该层单独就扛不住 INT8 |
| **B. leave-one-out** | 全量化后关掉第 i 层 → AUC 回血 | 该层被量化坑得最狠 |

判定：任一 ΔAUC > `sensitive_auc_drop`（默认 0.002）→ 敏感层 → 不量化，写入
`outputs/sensitive_layers.txt`，`quantize_qdq.py` 自动读取。

> 效率点：因为量化参数**只校准一次**，之后逐层只是开关，所以整个分析很快
> （对比：如果每层都重新 prepare/校准/导出，几十层要跑很久）。

**与论文的印证**：DQRM 的结论是「embedding 对低精度鲁棒、MLP 更敏感」。
本项目的敏感层名单（dense MLP 里靠前/靠后的层）会给出同方向的经验证据。

---

## 7. 产物一览（`outputs/`）

| 文件 | 说明 |
|---|---|
| `fake_data.npz` | fake 训练/验证/校准数据 |
| `model.pt` | 训练好的 FP32 模型 |
| `model.onnx` | 未量化 ONNX（FP32 engine 输入） |
| `model_quant_qdq.onnx` | 带 Q/DQ 的 ONNX（INT8 engine 输入） |
| `sensitivity.csv` / `sensitive_layers.txt` | 逐层敏感度明细 / 敏感层名单 |
| `model_fp32.plan` / `model_int8.plan` | TensorRT engine |
| `bench_report.md` / `bench_stats.csv` | 压测报告与明细 |

---

## 8. 常见报错

| 报错 | 原因 / 解决 |
|---|---|
| 自检 ③ 与 ② 同量级（如 2e-01） | onnxruntime 用了默认全量优化。改成 `ORT_ENABLE_BASIC` |
| `导出的 ONNX 里没有 QuantizeLinear 节点` | 没有层被 enabled（敏感层全被跳过，或 `--all-layers` 没给对） |
| `minimum of empty tensor` / 校准报错 | 校准集为空。调大 `num_samples` 或减小 `calib_ratio` |
| `build_serialized_network 返回 None` | 加 `--verbose` 看 TRT 日志；检查输入形状与 opset |
| INT8 engine 里还有 FP32 层 | 正常。TRT 会把收益低/精度敏感的层留在 FP32 |
| 显存不够 | 调小 `data.num_samples` / `model.top_mlp` / `train.batch_size` |

---

## 9. 与 `01_transformer_deploy/` 的对照

| | 任务1 | 任务2（本目录） |
|---|---|---|
| 模型 | Transformer 精排（dense 重） | DLRM 结构（embedding 重） |
| 量化工具 | NVIDIA 系（transformer-deploy / pytorch-quantization） | **PyTorch 官方** `torch.ao` 观察器 |
| 论文 | — | DQRM（Meta 2024）结构简化版 |
| 权重/激活粒度 | per-tensor 对称 int8 | per-tensor 对称 int8 |
| 预期结论 | INT8 加速明显 | **INT8 加速有限**——大头是 embedding 查表，量化不了 |

> 两个任务的差异主要在「用哪套工具求量化参数」与「模型结构」；
> Q/DQ 写进 ONNX 的方式相同（都用标准 `QuantizeLinear`/`DequantizeLinear`，
> 原因见第 4 节的三个限制）。
