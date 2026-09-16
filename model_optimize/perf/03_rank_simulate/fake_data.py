"""
Fake 数据生成模块（精排场景）

精排场景的样本结构：一个用户 + N 个候选 item（N = num_candidates）。
每个样本包含：
    - user_id:  标量，用户 ID
    - hist_ids: [hist_len] 用户历史行为序列（历史里也是 item id）
    - item_ids: [N] 候选 item id
    - labels:   [N] 每个候选的标签（0/1，模拟点击）

两个函数：
    - generate_fake_data: 生成 num_samples 条样本并保存到 pkl
    - generate_batch:     实时生成 1 个用户 + N 个候选（用于推理/profile）
"""
import os
import pickle
import numpy as np

# 日志配置：只写文件，不输出到控制台
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_data.log')
    _fh = logging.FileHandler(_log_path, mode='a', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(_fh)
    _logger.propagate = False


def generate_one_sample(config):
    """生成一条精排样本：1 个用户 + N 个候选 item"""
    num_users = config['num_users']
    num_items = config['num_items']
    hist_len = config['hist_len']
    num_candidates = config['num_candidates']

    user_id = np.random.randint(0, num_users)
    # 历史行为序列：固定长度，随机 item id
    hist_ids = np.random.randint(0, num_items, size=hist_len).astype(np.int64)
    # 候选 item：不重复采样 N 个
    item_ids = np.random.choice(num_items, size=num_candidates, replace=False).astype(np.int64)
    # 标签：简单模拟，正样本稀疏（约 20% 正例）
    labels = (np.random.rand(num_candidates) > 0.8).astype(np.float32)

    return {
        'user_id': user_id,
        'hist_ids': hist_ids,
        'item_ids': item_ids,
        'labels': labels,
    }


def generate_fake_data(config, save_path=None, verbose=True):
    """生成 num_samples 条样本，可选保存为 pickle"""
    n = config['num_samples']
    if verbose:
        _logger.info(f"Generating {n} fake samples...")
    samples = [generate_one_sample(config) for _ in range(n)]

    data = {
        'user_id': np.array([s['user_id'] for s in samples], dtype=np.int64),
        'hist_ids': np.stack([s['hist_ids'] for s in samples]),
        'item_ids': np.stack([s['item_ids'] for s in samples]),
        'labels': np.stack([s['labels'] for s in samples]),
    }

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        if verbose:
            _logger.info(f"Fake data saved to {save_path}")
            _logger.info(f"  user_id shape: {data['user_id'].shape}")
            _logger.info(f"  hist_ids shape: {data['hist_ids'].shape}")
            _logger.info(f"  item_ids shape: {data['item_ids'].shape}")
            _logger.info(f"  positive ratio: {data['labels'].mean():.3f}")
    return data


def generate_batch(config, num_candidates=None, device='cpu'):
    """实时生成 1 个用户 + N 个候选 item（用于推理/profile，不保存）"""
    import torch
    tmp = dict(config)
    if num_candidates is not None:
        tmp['num_candidates'] = num_candidates
    s = generate_one_sample(tmp)
    return {
        'user_id': torch.tensor(s['user_id'], device=device),
        'hist_ids': torch.tensor(s['hist_ids'], device=device),
        'item_ids': torch.tensor(s['item_ids'], device=device),
        'labels': torch.tensor(s['labels'], device=device),
    }


if __name__ == '__main__':
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    generate_fake_data(config, config['data_path'])
