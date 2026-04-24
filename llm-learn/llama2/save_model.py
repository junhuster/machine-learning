import os
import glob
import sys
import time
from pathlib import Path
import torch
import logging as log
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent/'util'))
import logger
logger.init_logger("/home/ubuntu/work/logs/llama2-pre-train.log")

def save_checkpoint(model, save_dir, dim, n_layers, vocab_size, step, max_checkpoints=3):
    """
    保存模型检查点，自动添加 step + 时间后缀，并只保留最新的N个
    """
    # 1. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 生成时间后缀：年月日_小时
    time_suffix = time.strftime("%Y%m%d_%H", time.localtime())
    
    # 3. 构建 checkpoint 文件名（加入 step）
    ckpt_filename = f"pretrain_{dim}_{n_layers}_{vocab_size}_step{step}_{time_suffix}.pth"
    ckpt_path = os.path.join(save_dir, ckpt_filename)
    
    # 4. 保存模型（多卡兼容）
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state_dict, ckpt_path)
    log.info(f"✅ 模型已保存：{ckpt_path}")

    # 5. 获取同类型检查点，按修改时间排序
    pattern = os.path.join(save_dir, f"pretrain_{dim}_{n_layers}_{vocab_size}_step*_*.pth")
    ckpt_list = sorted(glob.glob(pattern), key=os.path.getmtime)
    
    # 6. 保留最新3个，删除最老的
    if len(ckpt_list) > max_checkpoints:
        for old_ckpt in ckpt_list[:-max_checkpoints]:
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
                log.info(f"🗑️ 删除旧检查点：{old_ckpt}")
