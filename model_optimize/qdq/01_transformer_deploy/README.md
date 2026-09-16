# 任务1：Transformer 精排模型 + NVIDIA 系 Q/DQ 量化

用 **HF `transformer-deploy` 生态（底层 = NVIDIA `pytorch-quantization`）** 的思路，把一个推荐场景的
Transformer 精排模型做 PTQ 量化，导出带 Q/DQ 的 ONNX，再用 TensorRT 建 INT8 engine，
最后压测对比 **FP32 vs INT8** 的延迟与吞吐。

---

## 1. 环境要求

| 组件 | 必需 | 说明 |
|---|---|---|
| GPU | ✅ | 建议 T4（sm_75）。Turing 支持 FP16/INT8 Tensor Core |
| `torch >= 2.0` | ✅ | 需带 ONNX 导出（`torch.onnx.export`） |
| `onnx` / `onnxruntime` | ✅ | 导出校验 + 可用于对拍 |
| `tensorrt`（Python 包） | ✅ | `build_engine.py` / `infer.py` 用 |
| `pytorch-quantization` | ⭕ 可选 | 装了就走 NVIDIA 官方路径；没装走内置等价实现（见下） |
| `transformer-deploy` | ⭕ 可选 | 它的 `onnx_to_tensorrt` 可以替代本项目的 `build_engine.py` |

先跑一次：

```bash
python3 ../common/check_env.py
```

### 关于「到底用哪个工具」

```
HF transformer-deploy
   └─ 它插入 Q/DQ 的那一步，import 的就是 ──▶ NVIDIA pytorch-quantization
                                              (tensorrt.pytorch_quantization)

PyTorch 官方 torch.ao.quantization  ← 与本任务无关（见 02_dqrm_int8/）
```

所以本任务的量化器只有一种来源：**pytorch-quantization**。
但 `pytorch-quantization` 的自动替换（`quant_modules.initialize()`）对**自定义模型**不总是生效，
因此本项目提供 `quant.backend` 三档：

| backend | 行为 |
|---|---|
| `pytorch_quantization` | 强制用 NVIDIA 工具（装了才行） |
| `torch_fake` | 用内置实现：自己算 scale，用 `torch.quantize_per_tensor` 表达 Q/DQ |
| `auto`（默认） | 能 import 到 `pytorch_quantization` 就用它，否则用内置 |

**两者产出的 ONNX 是一样的**（都带 `QuantizeLinear`/`DequantizeLinear` 节点），
差别只在「谁算 scale、谁插节点」。这是刻意的设计——保证链路一定能跑通，
不会因为某个第三方包没装就卡住。

> ⚠️ **T4 不支持 TF32 / INT4 / FP8**，所以本实验**只做 FP32 vs INT8**，不做 INT4。

---

## 2. 模型与数据

**模型**（`model.py`）：推荐场景 Transformer 精排

```
user/item/cate Embedding ─┐
dense 特征 → Linear      ─┼→ candidate token + 历史序列 token
历史 item Embedding      ─┘        ↓  位置编码
                        N × TransformerBlock（Pre-LN, 手写 MHSA + FFN）
                                   ↓
                  [user+cate, candidate, 序列mean] → MLP head → logit
```

- 每个 `Linear` 都是**独立子模块**（attention 手写 qkv/proj），方便按名定位、逐层量化。
- 规模刻意放大：`embed_dim=256, layers=4, ffn=1024, head=[1024,512]`，
  加上 5 万/2 万词表 → 参数量约 **30M+**，FP32 单请求耗时足够高，量化才有意义。

**数据**（`gen_data.py`）：fake 特征 + **有学习信号的 label**

用固定隐因子模型造 label（`<U[user],V[item]>` + dense 线性项 + 类目偏置 + 历史偏好），
所以 AUC 能稳定在 **0.75 左右**——这一点很重要，因为如果 label 是随机的，AUC 恒为 0.5，
量化前后的 ΔAUC 全是噪声，根本判断不了敏感层。

落盘：`outputs/fake_data.npz`，含 `train_/val_/calib_` 三套（calib 从训练集切出，**不能和验证集混**）。

---

## 3. 操作步骤

```bash
cd 01_transformer_deploy

# ① 造数据
python3 gen_data.py
#   预期: train=84000 val=24000 calib=12000 | 正样本率≈0.5x
#         outputs/fake_data.npz

# ② 训练
python3 train.py
#   预期: 每 epoch 打印 loss 与 val_auc；5 个 epoch 后 val_auc ≈ 0.75
#         outputs/model.pt

# ③ 导出 FP32 ONNX
python3 export_onnx.py --check
#   预期: 打印 5 个输入 + 1 个输出的形状([batch,...])；
#         onnxruntime vs torch 最大绝对误差 < 1e-4

# ④ 逐层敏感度分析（★ 决定哪些层不量化）
python3 sensitivity.py
#   预期: 先打印基线 AUC(FP32)，然后逐层打印 AUC 与 ΔAUC；
#         再打印 leave-one-out 的「回血」；
#         outputs/sensitivity.csv + outputs/sensitive_layers.txt
#   省时间: python3 sensitivity.py --max-val 4096 --skip-loo

# ⑤ 插 Q/DQ 导出量化 ONNX
python3 quantize_qdq.py
#   预期: 打印跳过了几个敏感层、AUC(INT8 伪量化) 与 ΔAUC；
#         最后打印 "ONNX 校验通过: QuantizeLinear=N, DequantizeLinear=N"
#         outputs/model_quant_qdq.onnx
#   对照实验: python3 quantize_qdq.py --all-layers   # 不跳敏感层，看 ΔAUC 有多大

# ⑥ 建 TensorRT engine
python3 build_engine.py
#   预期: outputs/model_fp32.plan 与 outputs/model_int8.plan
#   想看哪些层真的走了 INT8: python3 build_engine.py --only int8 --verbose 2>&1 | grep -i int8

# ⑦ 压测对比
python3 bench_compare.py
#   预期: 打印各 batch 的 mean/p99/吞吐 + 加速比；末尾给结论
#         outputs/bench_report.md + outputs/bench_stats.csv

# ⑧ 单条/批量推理（可与 torch 对拍）
python3 infer.py --engine outputs/model_int8.plan --batch 32 --compare
```

一键跑完：`bash run_all.sh`

---

## 4. 敏感度分析是怎么做的（面试可讲）

**两条互补证据**（`sensitivity.py`）：

| 方法 | 做法 | 含义 |
|---|---|---|
| **A. per-layer-only** | 每次只把第 i 层打开量化，其余 FP32，算 AUC → ΔAUC | ΔAUC 大 = 这层单独就扛不住 INT8 |
| **B. leave-one-out** | 先全量化，再逐层把第 i 层恢复 FP32，看 AUC 回血 | 回血大 = 这层被量化坑得最狠 |

为什么两条都要：**单独量化的敏感度 ≠ 层间组合的敏感度**（层之间有交互）。
A 和 B 结论一致 → 可信；不一致 → 以 **B 为主**（因为线上跑的就是「全量化态」）。

判定规则：`ΔAUC > quant.sensitive_auc_drop`（默认 0.002）→ 判为敏感层 → **不量化**，
名字写入 `outputs/sensitive_layers.txt`，`quantize_qdq.py` 会自动读取。

> 💡 关键效率点：整个分析**只在 PyTorch 里跑伪量化，不建任何 TRT engine**。
> 如果每层都重导 ONNX + 重建 engine，50 层要跑几小时；现在只要几分钟。

---

## 6. 导出自检：三个数字怎么读

`quantize_qdq.py` 跑完会打印：

```
自检 ① logit 量级 |max| = 1.896
自检 ② FP32 -> INT8 差异 = 3.551e-02   (量化本身的代价)
自检 ③ INT8 -> ONNX 差异 = 2.384e-07   (导出是否忠实)
判读：③ 应显著小于 ②；若 ③ 与 ② 同量级，导出可能有问题
```

**③ 是导出忠实度的硬指标**：本项目实测 **2.4e-07**（11 层全量化），说明
`torch.onnx.export` 把 `torch.quantize_per_tensor` / `torch.dequantize` 转成
ONNX 的 `QuantizeLinear` / `DequantizeLinear` 是**逐位忠实**的。

> ⚠️ **自检必须把 onnxruntime 的图优化关到 BASIC**：
> ```python
> so = ort.SessionOptions()
> so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
> ```
> 用默认的 `ORT_ENABLE_ALL` 时，ORT 会**错误处理本图的 Q→DQ 模式**，让自检报出
> 2.4e-01 的**假阳性**（实测：全开 2.4e-01 / 仅基础 3.6e-07 / 关闭 3.6e-07）。
> 导出的图本身是对的，TensorRT 按显式量化规范解析，不受影响。
> 排查这个问题花了不少时间，记在这里避免重复踩坑。

**权威精度以 `bench_compare.py` 里 TRT engine 实测的 AUC 为准**（自检只是流程校验）。

---

## 7. 产物一览（`outputs/`）

| 文件 | 说明 |
|---|---|
| `fake_data.npz` | fake 训练/验证/校准数据 |
| `model.pt` | 训练好的 FP32 模型 |
| `model.onnx` | 未量化 ONNX（FP32 engine 的输入） |
| `model_quant_qdq.onnx` | 带 Q/DQ 的 ONNX（INT8 engine 的输入） |
| `sensitivity.csv` | 逐层敏感度明细（两种证据 + 判定） |
| `sensitive_layers.txt` | 敏感层名单（不量化） |
| `model_fp32.plan` / `model_int8.plan` | TensorRT engine |
| `bench_report.md` / `bench_stats.csv` | 压测报告与明细 |

---

## 8. 常见报错

| 报错 | 原因 / 解决 |
|---|---|
| 自检 ③ 与 ② 同量级（如 2e-01） | onnxruntime 用了默认全量优化。改成 `ORT_ENABLE_BASIC`（见第 6 节） |
| `导出的 ONNX 里没有 QuantizeLinear 节点` | torch 没把 `quantize_per_tensor` 映射成 Q/DQ。把 `quant.backend` 改成 `pytorch_quantization`，或升级 torch 到 ≥2.0、opset ≥13 |
| `build_serialized_network 返回 None` | 加 `--verbose` 看 TRT 日志；常见原因：输入形状写错、opset 过高、TRT 版本不匹配 |
| INT8 engine 里很多层仍是 FP32 | 正常。TRT 会把「量化收益低或精度敏感」的层留在 FP32，这正是显式量化的行为；用 `--verbose` 可看到哪些层被 INT8 化 |
| `tensorrt` import 失败 | 装 TensorRT 的 Python 包（`pip install tensorrt`，或从 NVIDIA tar 包的 `python/` 目录 `pip install`） |
| 敏感度分析太慢 | `--max-val 4096 --skip-loo`，或 `--layers blocks.0,head` 只看部分层 |
| 显存不够 | 调小 `data.num_samples` / `model.embed_dim` / `train.batch_size` |
| 显式量化与隐式 PTQ 冲突 | 不要给 `build_engine.py` 传校准缓存；带 Q/DQ 的 ONNX 直接 `--int8` 即可 |

---

## 9. 与 `02_dqrm_int8/` 的对照

| | 任务1（本目录） | 任务2 |
|---|---|---|
| 模型 | Transformer 精排（dense 重） | DLRM 结构（embedding 重） |
| 量化工具 | NVIDIA 系（transformer-deploy / pytorch-quantization） | PyTorch 官方 `torch.ao`（FX QDQ + QAT） |
| 量化方式 | PTQ（训练后校准） | QAT（量化感知训练） |
| 预期结论 | INT8 加速明显 | INT8 加速有限——因为大头是 embedding 查表 |
