"""
Nsight Compute (ncu) 专用短脚本 —— 推理模式（精排打分）
- 加载训练好的模型，1 个用户对小批量候选打分（只跑 1 步）
- 配合 ncu 命令做 kernel 级分析
- 用 ncu_candidates（默认 8）比 num_candidates 小很多，加速 ncu 采集，kernel 类型一样

用法:
  sudo ncu --set basic --launch-skip 5 --launch-count 50 \
      -o ncu_report --force-overwrite python3 profile_ncu.py
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
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_ncu.log')
    _fh = logging.FileHandler(_log_path, mode='a', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(_fh)
    _logger.propagate = False


def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', "Nsight Compute requires CUDA GPU"

    # 加载训练好的模型
    save_path = config.get('save_path', './rank_model.pt')
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model = RankModel(ckpt['config']).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        _logger.info(f"[NCU] Loaded trained model from {save_path}")
    else:
        _logger.warning(f"[NCU] Model file {save_path} not found, using fresh model")
        model = RankModel(config).to(device)

    model.eval()

    # ncu 专用小候选数（默认 8），加速采集
    ncu_candidates = config.get('ncu_candidates', 8)

    # 预生成 1 个用户 + 小批量候选
    _logger.info(f"[NCU] Pre-generating batch (num_candidates={ncu_candidates})...")
    batch = generate_batch(config, num_candidates=ncu_candidates, device=device)

    torch.cuda.synchronize()

    # 只跑 1 个推理 step（推理重复性，1 步包含所有 kernel 类型）
    _logger.info(f"[NCU] Running 1 inference step (N={ncu_candidates}) for kernel profiling...")

    with torch.no_grad():
        logits = model(batch['user_id'], batch['hist_ids'], batch['item_ids'])

    torch.cuda.synchronize()
    _logger.info(f"[NCU] Done. output shape={tuple(logits.shape)}")


if __name__ == '__main__':
    main()
