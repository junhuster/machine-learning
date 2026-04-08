# 大语言模型推理优化技术汇总

> 整理时间：2026-04-08

---

## 一、优化技术概览

### 1. 模型层面

| 技术 | 原理 | 效果 |
|------|------|------|
| **量化** | INT8/INT4/FP8 降低精度 | 显存减少 2-4x，速度提升 |
| **剪枝** | 移除不重要的权重/注意力头 | 模型变小，速度提升 |
| **知识蒸馏** | 大模型教小模型 | 小模型保持性能 |
| **稀疏化** | 稀疏注意力/稀疏 MLP | 计算量减少 |

### 2. 解码策略

| 技术 | 原理 | 适用场景 |
|------|------|----------|
| **KV Cache** | 缓存已计算的 K/V | 所有自回归模型必备 |
| **PagedAttention** | 分页管理 KV Cache | 多并发、长序列 |
| **Speculative Decoding** | 小模型猜测+大模型验证 | 加速 2-3x |
| **Parallel Decoding** | 多 token 并行生成 | 批量生成 |

### 3. 计算优化

| 技术 | 说明 |
|------|------|
| **Flash Attention** | 融合注意力计算，减少显存访问 |
| **Flash Attention V2** | 进一步优化并行性和分区 |
| **Flash Decoding** | 优化解码阶段的 attention |
| **算子融合** | 多个 kernel 合并，减少启动开销 |
| **Tensor Parallelism** | 多 GPU 并行计算 |

### 4. 系统层面

| 技术 | 说明 |
|------|------|
| **Continuous Batching** | 动态批处理，提高 GPU 利用率 |
| **Prefix Caching** | 缓存公共前缀的 KV |
| **模型并行** | Tensor/Sequence/Pipeline Parallel |
| **Offloading** | CPU-GPU 内存交换（显存不足时） |

### 5. 常用推理框架

| 框架 | 特点 |
|------|------|
| **vLLM** | PagedAttention，高吞吐 |
| **TensorRT-LLM** | NVIDIA 优化，FlashAttention |
| **TGI** | HuggingFace，生产级 |
| **llama.cpp** | CPU/Apple Silicon 友好 |
| **DeepSpeed-Inference** | 微软，支持多种优化 |
| **ONNX Runtime** | 通用部署 |

### 6. 量化方法

| 方法 | 特点 |
|------|------|
| **GPTQ** | 训练后量化，4bit 效果好 |
| **AWQ** | 保护重要权重，精度损失小 |
| **GGUF** | llama.cpp 格式，CPU 友好 |
| **FP8** | 新硬件支持（H100） |

### 7. 推荐组合

```
高吞吐服务：vLLM + FlashAttention + Continuous Batching
低延迟场景：TensorRT-LLM + INT4 量化
显存有限：4bit 量化 + KV Cache 优化
```

---

## 二、推荐书籍

### 1. 系统与推理优化

| 书籍 | 涵盖内容 | 特点 |
|------|----------|------|
| **《Designing Machine Learning Systems》** | 模型部署、推理优化、系统设计 | 实践导向，覆盖面广 |
| **《Machine Learning Engineering》** - Andriy Burkov | MLOps、模型压缩、部署 | 简洁实用 |
| **《机器学习系统：设计与实现》** | 系统架构、分布式训练推理 | 中文，开源免费 |

### 2. 深度学习基础与优化

| 书籍 | 涵盖内容 |
|------|----------|
| **《Deep Learning》** - Goodfellow | 量化、剪枝理论基础 |
| **《动手学深度学习》** | 中文，含PyTorch实现 |

### 3. Transformer与大模型

| 书籍 | 涵盖内容 |
|------|----------|
| **《Natural Language Processing with Transformers》** | HuggingFace官方，Transformer架构 |
| **《Build a Large Language Model (From Scratch)》** - Sebastian Raschka | 从零实现，含推理优化章节 |
| **《Hands-On Large Language Models》** | LLM架构、量化、部署 |

### 4. 并行计算与系统

| 书籍 | 涵盖内容 |
|------|----------|
| **《Deep Learning Systems》** - CMU课程教材 | 分布式训练、模型并行、推理系统 |
| **《Programming Massively Parallel Processors》** | CUDA编程、GPU优化原理 |

### 5. 中文专门书籍

| 书籍 | 内容 |
|------|------|
| **《大语言模型：原理与工程实践》** | 模型架构、微调、推理部署 |
| **《大规模语言模型：从理论到实践》** | 预训练、微调、推理优化 |
| **《深度学习推理引擎》** | TVM、TensorRT、推理优化 |

### 6. 推荐阅读顺序

```
1. 《Build a Large Language Model (From Scratch)》
   → 理解LLM架构和基础推理

2. 《Designing Machine Learning Systems》
   → 系统层面理解部署优化

3. 论文 + 开源项目文档
   → FlashAttention、vLLM、量化等最新技术
```

---

## 三、论文资源

### 1. 注意力优化

| 技术 | 论文 | 链接 |
|------|------|------|
| **FlashAttention** | FlashAttention: Fast and Memory-Efficient Exact Attention | https://arxiv.org/abs/2205.14135 |
| **FlashAttention-2** | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | https://arxiv.org/abs/2307.08691 |
| **FlashAttention-3** | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | https://arxiv.org/abs/2407.08608 |
| **PagedAttention** | Efficient Memory Management for Large Language Model Serving with PagedAttention | https://arxiv.org/abs/2309.06180 |

### 2. 量化

| 技术 | 论文 | 链接 |
|------|------|------|
| **GPTQ** | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | https://arxiv.org/abs/2210.17323 |
| **AWQ** | AWQ: Activation-aware Weight Quantization for LLM Compression | https://arxiv.org/abs/2306.00978 |
| **QLoRA** | QLoRA: Efficient Finetuning of Quantized LLMs | https://arxiv.org/abs/2305.14314 |

### 3. 解码优化

| 技术 | 论文 | 链接 |
|------|------|------|
| **Speculative Decoding** | Fast Inference from Transformers via Speculative Decoding | https://arxiv.org/abs/2211.17192 |
| **SpecTr** | SpecTr: Fast Speculative Decoding via Optimal Transport | https://arxiv.org/abs/2401.11734 |
| **Medusa** | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | https://arxiv.org/abs/2401.10774 |

### 4. KV Cache 优化

| 技术 | 论文 | 链接 |
|------|------|------|
| **FlexGen** | FlexGen: High-Throughput Generative Inference of Large Language Models | https://arxiv.org/abs/2303.06865 |
| **H2O** | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models | https://arxiv.org/abs/2306.14048 |

---

## 四、GitHub 项目

### 1. 推理框架

| 框架 | GitHub | 文档 |
|------|--------|------|
| **vLLM** | https://github.com/vllm-project/vllm | https://vllm.readthedocs.io/ |
| **TensorRT-LLM** | https://github.com/NVIDIA/TensorRT-LLM | https://nvidia.github.io/TensorRT-LLM/ |
| **TGI** | https://github.com/huggingface/text-generation-inference | https://huggingface.co/docs/text-generation-inference/ |
| **llama.cpp** | https://github.com/ggerganov/llama.cpp | README 即文档 |
| **DeepSpeed** | https://github.com/microsoft/DeepSpeed | https://www.deepspeed.ai/ |
| **ONNX Runtime** | https://github.com/microsoft/onnxruntime | https://onnxruntime.ai/docs/ |
| **Triton Inference Server** | https://github.com/triton-inference-server/server | https://docs.nvidia.com/deeplearning/triton/ |

### 2. 注意力优化

| 项目 | GitHub |
|------|--------|
| **FlashAttention** | https://github.com/Dao-AILab/flash-attention |

### 3. 量化工具

| 项目 | GitHub |
|------|--------|
| **AutoGPTQ** | https://github.com/AutoGPTQ/AutoGPTQ |
| **AWQ** | https://github.com/mit-han-lab/llm-awq |
| **llama.cpp (GGUF)** | https://github.com/ggerganov/llama.cpp |

### 4. 解码优化

| 项目 | GitHub |
|------|--------|
| **Medusa** | https://github.com/FasterDecoding/Medusa |

### 5. 论文合集

| 项目 | GitHub |
|------|--------|
| **Awesome LLM Inference** | https://github.com/DefTruth/Awesome-LLM-Inference |
| **LLM Inference Papers** | https://github.com/hijkzzz/Awesome-LLM-Inference-Papers |

---

## 五、技术博客

### 1. HuggingFace Blog

| 主题 | 链接 |
|------|------|
| 主页 | https://huggingface.co/blog |
| LLM 推理 | https://huggingface.co/blog/llm-inference |
| 量化指南 | https://huggingface.co/blog/4bit-transformers-bitsandbytes |

### 2. 其他优质博客

| 来源 | 链接 | 内容 |
|------|------|------|
| **vLLM Blog** | https://vllm.ai/blog/ | PagedAttention 原理解读 |
| **NVIDIA Blog** | https://developer.nvidia.com/blog/ | TensorRT-LLM、GPU优化 |
| **Lilian Weng Blog** | https://lilianweng.github.io/ | LLM 结构化解读 |
| **Jay Alammar Blog** | https://jalammar.github.io/ | 可视化解释 Transformer |

---

## 六、课程与教程

| 来源 | 链接 |
|------|------|
| **CMU Deep Learning Systems** | https://dlsyscourse.org/ |
| **Stanford CS224N** | https://web.stanford.edu/class/cs224n/ |
| **HuggingFace NLP Course** | https://huggingface.co/learn/nlp-course/ |
| **Andrej Karpathy - Let's build GPT** | https://www.youtube.com/watch?v=kCc8FmEb1nY |

---

## 七、学习路径建议

```
1. 入门
   → HuggingFace Blog（推理基础）
   → Karpathy YouTube（从零构建GPT）

2. 进阶
   → FlashAttention 论文
   → vLLM 论文 + 文档
   → GPTQ/AWQ 论文

3. 实践
   → vLLM 文档动手实践
   → TensorRT-LLM 示例
```

---

## 八、常用命令速查

### vLLM 启动服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --max-model-len 4096
```

### llama.cpp 量化

```bash
# 转换为 GGUF
python convert.py /path/to/model --outtype q4_0 --outfile model.gguf

# 运行
./main -m model.gguf -p "Hello"
```

### TensorRT-LLM 构建

```bash
python build.py --model_dir /path/to/model \
    --dtype float16 \
    --use_gpt_attention_plugin \
    --output_dir /trt_model
```

---

## 九、性能对比参考

| 框架 | 吞吐 | 延迟 | 显存效率 | 易用性 |
|------|------|------|----------|--------|
| vLLM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| TensorRT-LLM | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| TGI | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| llama.cpp | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

> 注：实际性能因模型、硬件、配置而异
