"""
dataset.py — DeepSeek-Mini数据集类

包含：
  - PretrainDataset: 预训练数据集，每行 {"text": "..."}
  - SFTDataset: 指令微调数据集，每行为messages列表（chat格式）
  - collate_pretrain: 预训练用collate_fn
  - collate_sft: SFT用collate_fn（需要处理loss_mask）
"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset


# ===========================================================================
# 预训练数据集
# ===========================================================================

class PretrainDataset(Dataset):
    """
    预训练数据集，读取JSONL格式文件。
    每行格式：{"text": "中文语料..."}

    内存高效实现：
    - __init__ 阶段只扫描文件，记录每行的字节偏移量（offsets），不加载文本内容
    - __getitem__ 按需用 seek+readline 读取单行，再解析JSON
    - 对大文件友好，内存占用约等于存储偏移量数组（每条8字节）
    """

    def __init__(self, path: str, tokenizer, max_seq_len: int):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 扫描文件，记录每条有效行的字节起始偏移
        self.offsets = []
        with open(path, "rb") as f:
            offset = 0
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    self.offsets.append(offset)
                offset += len(raw_line)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            raw_line = f.readline()

        obj = json.loads(raw_line.decode("utf-8"))
        text = obj.get("text", "")

        ids = self.tokenizer.encode(text, add_special_tokens=True)

        # 目标长度 = max_seq_len + 1（input取[:max_seq_len]，target取[1:max_seq_len+1]）
        target_len = self.max_seq_len + 1
        if len(ids) > target_len:
            ids = ids[:target_len]
        else:
            ids = ids + [self.pad_id] * (target_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)


def collate_pretrain(batch):
    return torch.stack(batch, dim=0)


# ===========================================================================
# 指令微调数据集
# ===========================================================================

class SFTDataset(Dataset):
    """
    指令微调数据集，读取JSONL格式文件。
    每行格式：messages列表，例如：
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    只对assistant回复部分计算loss（通过loss_mask实现）。

    内存高效实现：同PretrainDataset，使用字节偏移按需读取。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0

        # 扫描文件，记录每条有效行的字节起始偏移
        self._offsets = []
        with open(data_path, "rb") as f:
            self._offsets.append(0)
            while f.readline():
                self._offsets.append(f.tell())
        self._total_lines = len(self._offsets) - 1

    def __len__(self):
        return self._total_lines

    def generate_loss_mask(self, input_ids):
        """
        生成loss mask，0表示不计算损失，1表示计算损失。
        只对assistant回复部分（<|im_start|>assistant\n 到 eos_token 之间）标记为1。
        """
        mask = [0] * len(input_ids)
        a_sequence = self.tokenizer("<|im_start|>assistant\n")["input_ids"]
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0

        while i <= n - a_length:
            # 检查当前位置是否匹配 <|im_start|>assistant\n
            match = all(input_ids[i + k] == a_sequence[k] for k in range(a_length))
            if match:
                # 从子序列结束位置开始，找第一个eos_token_id
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == self.tokenizer.eos_token_id:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j  # 包含eos
                    for pos in range(start, end + 1):
                        if pos < len(mask):
                            mask[pos] = 1
                i += a_length
            else:
                i += 1

        return mask

    def __getitem__(self, index: int):
        with open(self.data_path, "rb") as f:
            f.seek(self._offsets[index])
            line = f.readline().decode("utf-8")

        sample = json.loads(line)
        text = self.tokenizer.apply_chat_template(
            sample, tokenize=False, add_generation_prompt=False
        )
        input_id = self.tokenizer(text).data["input_ids"][: self.max_length]
        text_len = len(input_id)

        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        return (
            torch.from_numpy(X),
            torch.from_numpy(Y),
            torch.from_numpy(loss_mask),
        )


def collate_sft(batch):
    X = torch.stack([item[0] for item in batch])
    Y = torch.stack([item[1] for item in batch])
    loss_mask = torch.stack([item[2] for item in batch])
    return X, Y, loss_mask
