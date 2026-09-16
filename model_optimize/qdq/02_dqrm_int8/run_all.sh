#!/usr/bin/env bash
# 一键跑完任务2 全流程（需在 GPU 服务器上执行）
set -euo pipefail
cd "$(dirname "$0")"

echo "==================== 0. 环境探测 ===================="
python3 ../common/check_env.py || true

echo "==================== 1. 生成 fake 数据 ===================="
python3 gen_data.py

echo "==================== 2. 训练（FP32 基线） ===================="
python3 train.py

echo "==================== 3. 导出 ONNX（FP32） ===================="
python3 export_onnx.py --check

echo "==================== 4. 逐层敏感度分析 ===================="
python3 sensitivity.py

echo "==================== 5. torch.ao 量化（PTQ/QAT）并导出 Q/DQ ONNX ===================="
python3 quantize_qdq.py

echo "==================== 6. 建 TensorRT engine ===================="
python3 build_engine.py

echo "==================== 7. 压测对比 ===================="
python3 bench_compare.py

echo "==================== 8. 单条推理示例 ===================="
python3 infer.py --engine outputs/model_int8.plan --batch 32 --compare

echo
echo "全部完成，结果见 outputs/bench_report.md"
