"""
纯推理脚本（精排打分）—— 专供 nsys profile 采集用。

和 profile_torch.py 的区别（重要）：
  - profile_torch.py 内部用了 torch.profiler（自己订阅 CUPTI），**不能再被 nsys 包住跑**
    （否则报 CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED，nsys 采不到 GPU 活动）
  - 本脚本不做任何 profiler 订阅，只跑纯推理前向，专供 nsys 采集
  - NVTX 区间（user_tower / item_tower / score_head）已埋进 model.py，nsys 里直接可见

使用方法:
  nsys profile --trace=cuda,nvtx,osrt \
    --force-overwrite=true -o nsys_rank python3 inference.py
"""
import os
import yaml
import torch

from model import RankModel
from fake_data import generate_batch

# 日志配置：只写文件，不输出到控制台
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference.log')
    _fh = logging.FileHandler(_log_path, mode='a', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(_fh)
    _logger.propagate = False

# NVTX：外层推理步区间（model.py 内部已埋 user_tower/item_tower/score_head）
HAS_NVTX = False
nvtx = None
try:
    import torch.cuda.nvtx as _nvtx
    _nvtx.range_push("nvtx_probe")
    _nvtx.range_pop()
    nvtx = _nvtx
    HAS_NVTX = True
except Exception:
    HAS_NVTX = False


def load_config():
    path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.info(f"[INFO] Device: {device}")

    # 加载训练好的模型
    save_path = config.get('save_path', './rank_model.pt')
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model = RankModel(ckpt['config']).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        _logger.info(f"[INFO] Loaded trained model from {save_path}")
    else:
        _logger.warning(f"[WARN] Model file {save_path} not found, using fresh model")
        model = RankModel(config).to(device)

    model.eval()
    batch = generate_batch(config, device=device)
    steps = config.get('nsys_infer_steps', 5)

    # 预热（nsys 报告里这段会被前面的 warmup 占据，正式推理步才是分析对象）
    with torch.no_grad():
        for _ in range(3):
            _ = model(batch['user_id'], batch['hist_ids'], batch['item_ids'])
    torch.cuda.synchronize()

    # 正式推理（nsys 采集范围，只跑前向，无 profiler 订阅）
    _logger.info(f"[INFO] Running {steps} inference steps for nsys...")
    with torch.no_grad():
        for i in range(steps):
            if HAS_NVTX:
                nvtx.range_push(f"infer_step_{i}")
            _ = model(batch['user_id'], batch['hist_ids'], batch['item_ids'])
            if HAS_NVTX:
                nvtx.range_pop()

    torch.cuda.synchronize()
    _logger.info("[DONE] inference finished")


if __name__ == '__main__':
    main()
