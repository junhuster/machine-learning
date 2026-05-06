import os
import re
import time
import torch
import json
from torch.utils.data import Dataset, DataLoader

# ---------------------
# 1. 模型保存 + 自动删除旧模型
# ---------------------
def save_model(model, step, save_dir, max_save):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"deepseek_step_{step}.pth")
    torch.save(model.state_dict(), path)

    files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("deepseek_step_")],
        key=lambda x: int(re.findall(r"\d+", x)[0])
    )
    while len(files) > max_save:
        oldest = files.pop(0)
        os.remove(os.path.join(save_dir, oldest))

# ---------------------
# 2. 加载最新模型 + 获取step
# ---------------------
def get_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir):
        return None, 0
    files = [f for f in os.listdir(save_dir) if f.startswith("deepseek_step_")]
    if not files:
        return None, 0
    files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    latest = files[-1]
    step = int(re.findall(r"\d+", latest)[0])
    return os.path.join(save_dir, latest), step

# ---------------------
# 3. 预训练数据集
# ---------------------
class PretrainDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.data = open(path, encoding="utf-8").read().splitlines()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        txt = self.data[idx]
        tok = self.tokenizer(
            txt, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": tok["input_ids"].squeeze(),
            "attention_mask": tok["attention_mask"].squeeze()
        }
# ===================== 新增：10GB超大中文预训练语料 分块懒加载 =====================
class LargePretrainDataset(Dataset):
    """
    适配10GB超大txt语料，不一次性载入内存，懒加载逐行读取
    每行一篇短文/一段语料，自动截断、padding
    """
    def __init__(self, file_path, tokenizer, max_len=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 预统计总行数
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.line_count = sum(1 for _ in f)

    def __len__(self):
        return self.line_count

    def __getitem__(self, idx):
        # 跳到指定行读取，懒加载不占内存
        with open(self.file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    text = line.strip()
                    break
        # 分词
        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": tok["input_ids"].squeeze(),
            "attention_mask": tok["attention_mask"].squeeze()
        }
# ---------------------
# 4. 指令微调数据集
# ---------------------
class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.data = json.load(open(path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"用户：{item['instruction']}\n助手：{item['output']}"
        tok = self.tokenizer(
            prompt, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": tok["input_ids"].squeeze(),
            "attention_mask": tok["attention_mask"].squeeze()
        }

# ---------------------
# 5. 计时器 + 剩余时间计算
# ---------------------
class StepTimer:
    def __init__(self):
        self.start = None
        self.count = 0

    def reset(self):
        self.start = time.time()
        self.count = 0

    def log(self, current_step, total_steps, loss, epoch, total_epoch):
        self.count += 1
        used = time.time() - self.start
        step_time = used / self.count
        remain = (total_steps - current_step) * step_time / 60
        print(f"Epoch {epoch}/{total_epoch} | Step {current_step}/{total_steps} | Loss: {loss:.4f} | Remain: {remain:.1f} min")