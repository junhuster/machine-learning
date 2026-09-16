# QDQ 量化：推荐精排模型 INT8 加速实验

两套「torch → ONNX → TensorRT」+「Q/DQ 量化」的端到端实验，用于对比 **量化前 / 量化后** 的 TensorRT 推理性能，并给出压测结论。

| 目录 | 模型 | 量化工具链 | 说明 |
|---|---|---|---|
| `01_transformer_deploy/` | 推荐场景 **Transformer 精排**模型 | HF `transformer-deploy`（底层 = NVIDIA `pytorch-quantization`） | 序列 + 多头注意力 + 大 MLP，dense 占比高，量化收益明显 |
| `02_dqrm_int8/` | **DQRM 简化版**（DLRM 结构） | PyTorch 官方 `torch.ao.quantization`（观察器 PTQ） | 复现 Meta《Deep Quantized Recommendation Models》结构，落点 INT8 |

**两个任务都只做训练后量化（PTQ），不做 QAT。**

两套共用 `common/`（AUC、TRT 构建/推理封装、压测框架），避免重复造轮子。

---

## 0. 先跑环境探测（**必做**）

```bash
python3 common/check_env.py
```

它会打印 GPU、驱动、CUDA、以及 `torch / onnx / onnxruntime / tensorrt / pytorch_quantization / transformers` 是否就绪。
**把输出贴出来再往下走** —— 缺哪个装哪个，两个任务对工具的要求不同（见各自 README「环境要求」）。

---

## 1. 术语先对齐（避开面试/沟通坑）

| 名词 | 是什么 | 在本项目里的角色 |
|---|---|---|
| **Q/DQ** | QuantizeLinear / DequantizeLinear 节点，插在图上标记「这里按 INT8 算」 | 量化的载体；有它之后的 ONNX 才能被 TRT 按 INT8 解析 |
| **PTQ**（Post-Training Quantization） | 训练后量化：拿少量校准数据算 amax/scale，不重训 | **两个任务都走这条路** |
| **QAT**（Quantization-Aware Training） | 量化感知训练：训练时就带 fake-quant，让模型适应低精度 | ❌ **本项目不做**（虽然 DQRM 原论文是 QAT） |
| **显式量化 vs 隐式 PTQ** | 显式=图里带 Q/DQ（本方案）；隐式=TRT 自己校准（`--calib`） | **两者不要混用**，会报 explicit quantization 冲突 |
| **向量量化 / 残差量化** | 用 codebook 压缩特征序列（DMQN、TARQ 用的是这个） | ⚠️ **不是数值精度量化**，与本项目目标不同，别混淆 |

> ⚠️ **T4 硬件上限（本项目前提）**：Turing sm_75 支持 **FP16 / INT8 Tensor Core**，但**不支持 TF32（Ampere+）、INT4 / FP8**。
> 所以：INT8 有真实硬件加速；INT4 只省显存不提速。本实验**只做 FP32 vs INT8**。

---

## 2. 端到端流程（两个任务结构一致）

```
gen_data.py            ① 生成 fake 训练数据 → 落盘 outputs/fake_data.npz（固定 seed，切 train/calib/val）
      ↓
train.py               ② 训练模型 → outputs/model.pt（打印 AUC）
      ↓
export_onnx.py         ③ torch → ONNX（动态 batch）→ outputs/model.onnx
      ↓
quantize_qdq.py        ④ 插 Q/DQ 节点 → outputs/model_quant.onnx（+ 敏感层名单）
      ↓
build_engine.py        ③ onnx → TRT engine（fp32 / int8）→ outputs/*.plan
      ↓
infer.py               ⑤ 加载 .plan 推理（单条 / 批量），可与 torch 模型对拍
      ↓
bench_compare.py       ⑥ FP32 vs INT8 压测 → outputs/bench_report.md（加速比 + ΔAUC 结论）
```

`sensitivity.py` 是第 ④ 步的**前置分析**：逐层量化看 ΔAUC，挑出敏感层不量化。
`run_all.sh` 把上面串起来一键跑完。

---

## 3. 快速开始

```bash
# 任务1：Transformer + NVIDIA 系 QDQ
cd 01_transformer_deploy
python3 gen_data.py && python3 train.py
python3 export_onnx.py
python3 sensitivity.py            # 先看哪些层敏感
python3 quantize_qdq.py           # 按敏感度插 Q/DQ
python3 build_engine.py           # 建 fp32 / int8 engine
python3 bench_compare.py          # 压测出结论
python3 infer.py --engine outputs/model_int8.plan --batch 32

# 任务2：DQRM 简化版 + torch.ao（换成对应脚本）
cd ../02_dqrm_int8 && bash run_all.sh
```

每个子目录的 README 有**逐条命令 + 每步预期输出 + 常见报错**。

---

## 4. 目录约定

- 所有产物落在各自 `outputs/`（`.npz / .pt / .onnx / .plan / .csv / .md`）。
- 脚本内部一律用 `Path(__file__).resolve().parent` 定位路径，**从任意目录执行都能跑**。
- 不做 `torch.set_default_device` 之类的隐式全局改动；设备由 `--device` 或 config 决定。

---

## 5. 踩坑记录（都是实测踩出来的，先看能省时间）

### 坑 1：ONNX Runtime 的「全量图优化」会算错 Q→DQ 图
用 `ORT_ENABLE_ALL` 跑含 Q/DQ 的 ONNX，结果与 torch 差 **2.4e-01**；
改成 `ORT_ENABLE_BASIC` 或关闭优化，立刻回到 **3.6e-07**。
→ **导出自检必须设 `so.graph_optimization_level = ORT_ENABLE_BASIC`**，否则会看到假阳性。
（导出的图本身是对的，TensorRT 不受影响。）

### 坑 2：torch.ao 的 FX 量化图不能直接导出 ONNX
三处硬限制（详见 `02_dqrm_int8/README.md` 第 4 节）：
量化 `Concat` 无 symbolic、per-channel 权重 `quantize_per_channel` 无 symbolic、
`convert_fx(is_reference=True)` 的模型直接导出报 `Got int`（先 `torch.jit.trace` 再导出会得到**数值错误**的图）。
→ 结论：**torch.ao 负责算量化参数，标准 Q/DQ 负责写进 ONNX**。

### 坑 3：激活别用 uint8 非对称
`torch.quantize_per_tensor(..., torch.quint8)` + `zero_point=0` 导出时，
Q 与 DQ 之间会被插一个 `Cast`，实测误差从 5e-08 飙到 **0.26**。
→ **激活与权重统一用对称 int8**（`qint8` + `per_tensor_symmetric`）。

### 坑 4：「三个工具」其实是两条路线
`transformer-deploy` 插 Q/DQ 用的就是 NVIDIA 的 `pytorch-quantization`，
所以只有 **NVIDIA 系** 与 **PyTorch 系（torch.ao）** 两条独立路线 —— 正好对应两个任务。

### 坑 5：embedding 量化不了
`nn.Embedding` 对应 ONNX `Gather`，稀疏随机访存，torch.ao 与 TensorRT 的显式量化都不量化它。
→ **推荐精排模型（embedding 重）的 INT8 收益天生有限**，这正是任务2 要展示的结论。

---

## 6. 关键结论会是什么（预期，实测后替换）

1. **INT8 对 dense 部分（大 GEMM / attention）加速明显**，对 embedding 查表（Gather）几乎无效 —— 因为 Gather 无 Tensor Core 加速且常 fallback FP32。
2. **所以模型的 dense/sparse 比例决定量化收益**：任务1（Transformer）收益 > 任务2（DLRM，embedding 重）。
3. **敏感层通常是**：第一层/最后一层的 Linear、LayerNorm 附近、attention 输出投影。
4. **FP16 在本实验里不参与对比**（按需求只做 FP32 vs INT8），但 INT8 engine 里被 TRT 判定为敏感的层会**留在 FP32**，这本身是 TRT 的自动行为。
