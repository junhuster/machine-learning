"""
Fake 数据生成模块
- generate_fake_data: 生成指定数量的fake样本并保存到文件
- generate_batch: 实时生成一个batch的fake数据（用于推理压测）
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
    """生成单条fake样本"""
    num_users = config['num_users']
    num_items = config['num_items']
    num_categories = config['num_categories']
    history_len = config['history_len']
    num_numeric = config['num_numeric']

    user_id = np.random.randint(0, num_users)
    item_id = np.random.randint(0, num_items)
    cate_id = np.random.randint(0, num_categories)

    # 历史行为序列：随机实际长度，不足部分padding
    actual_len = np.random.randint(1, history_len + 1)
    history = np.random.randint(0, num_items, size=history_len).astype(np.int64)
    history_mask = np.zeros(history_len, dtype=np.float32)
    history_mask[:actual_len] = 1.0

    # 数值特征
    numeric = np.random.randn(num_numeric).astype(np.float32)

    # 标签：简单模拟，item_id为偶数时更可能点击
    label = 1 if (item_id % 2 == 0 and np.random.rand() > 0.3) else 0

    return {
        'user_id': user_id,
        'item_id': item_id,
        'cate_id': cate_id,
        'history': history,
        'history_mask': history_mask,
        'numeric': numeric,
        'label': label,
    }


def generate_fake_data(config, save_path=None, verbose=True):
    """生成num_samples条fake数据，可选保存为pickle"""
    n = config['num_samples']
    if verbose:
        _logger.info(f"Generating {n} fake samples...")
    samples = [generate_one_sample(config) for _ in range(n)]

    data = {
        'user_id': np.array([s['user_id'] for s in samples], dtype=np.int64),
        'item_id': np.array([s['item_id'] for s in samples], dtype=np.int64),
        'cate_id': np.array([s['cate_id'] for s in samples], dtype=np.int64),
        'history': np.stack([s['history'] for s in samples]),
        'history_mask': np.stack([s['history_mask'] for s in samples]),
        'numeric': np.stack([s['numeric'] for s in samples]),
        'label': np.array([s['label'] for s in samples], dtype=np.float32),
    }

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        if verbose:
            _logger.info(f"Fake data saved to {save_path}")
            _logger.info(f"  user_id shape: {data['user_id'].shape}")
            _logger.info(f"  history shape: {data['history'].shape}")
            _logger.info(f"  numeric shape: {data['numeric'].shape}")
            _logger.info(f"  positive ratio: {data['label'].mean():.3f}")
    return data


def generate_batch(config, batch_size, device='cpu'):
    """实时生成一个batch的fake数据（用于推理压测，不保存）"""
    import torch
    tmp_config = dict(config)
    tmp_config['num_samples'] = batch_size
    data = generate_fake_data(tmp_config, save_path=None, verbose=False)
    batch = {
        'user_id': torch.tensor(data['user_id'], device=device),
        'item_id': torch.tensor(data['item_id'], device=device),
        'cate_id': torch.tensor(data['cate_id'], device=device),
        'history': torch.tensor(data['history'], device=device),
        'history_mask': torch.tensor(data['history_mask'], device=device),
        'numeric': torch.tensor(data['numeric'], device=device),
    }
    return batch


if __name__ == '__main__':
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    generate_fake_data(config, config['data_path'])
