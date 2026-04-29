from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import tiktoken
import time
import sys
import numpy as np
import torch
from transformers import AutoTokenizer
import os
import json
from pathlib import Path
from datetime import datetime
import logging as log
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent/'util'))
import logger
logger.init_logger("/home/ubuntu/work/logs/gpt2-zh-pre-train.log")
import generate_text as Gtext
# ===================== 配置区域 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_JSON_PATH = "/home/ubuntu/work/data/llm-data/train_data/zh/monkey/pretrain_data/monkey_pretrain_8G.jsonl"        # 10GB 训练数据
VAL_JSON_PATH = "/home/ubuntu/work/data/llm-data/train_data/zh/monkey/pretrain_data/monkey_pretrain_val_34M.jsonl"          # 验证集（建议自己切分小文件）
SAVE_DIR = "/home/ubuntu/work/data/llm-data/pretrained_model/gpt2/124M_zh/"
VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
SAVE_STEP = 1000
NUM_EPOCH = 1
EVAL_STEP = 500
BATCH_SIZE = 32
LOG_STEP = 100
# GPT2 配置
GPT_CONFIG_124M = {
    "vocab_size": VOCAB_SIZE,
    "context_length": CONTEXT_LENGTH,
    "emb_dim": 768,
    "n_heads": 8,
    "n_layers": 8,
    "drop_rate": 0.1,
    "qkv_bias": False
}
# ====================================================

# ------------------- 模型组件 -------------------
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"], num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# ------------------- 损失函数 -------------------
def calc_loss_batch(input_batch, target_batch, loss_mask, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    # 逐 token 计算损失（向量）
    loss_per_token = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), reduction="none")
    # 只保留 mask=1 的部分，然后求平均
    loss_mask = loss_mask.to(device)
    loss = (loss_per_token * loss_mask.view(-1)).sum() / loss_mask.sum().clamp(min=1e-8)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = min(num_batches, len(data_loader)) if num_batches else len(data_loader)
    for i, (input_batch, target_batch, loss_mask) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, loss_mask, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# ------------------- 评估与生成 -------------------
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# ------------------- 模型保存 + 自动清理 -------------------
def save_model_with_cleanup(model, optimizer, global_step, save_dir=SAVE_DIR, max_keep=2):
    os.makedirs(save_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y_%m_%d")
    model_name = f"gpt2_pretrain_step{global_step}_{date_str}.pth"
    save_path = os.path.join(save_dir, model_name)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": global_step,
        "config": GPT_CONFIG_124M
    }, save_path)
    print(f"\n✅ 模型已保存：{model_name}")

    model_files = [f for f in os.listdir(save_dir) if f.endswith(".pth") and "gpt2_pretrain" in f]
    model_files.sort(key=lambda x: os.path.getctime(os.path.join(save_dir, x)))

    if len(model_files) > max_keep:
        oldest_file = model_files[0]
        os.remove(os.path.join(save_dir, oldest_file))
        print(f"🗑️ 已删除最老模型：{oldest_file}")

# ------------------- 训练主函数 -------------------
def train_model_simple(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    train_data_len = len(train_loader)
    start_time = time.time() 
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch, loss_mask in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, loss_mask, model, device)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % LOG_STEP == 0:
                spend_time = time.time() - start_time
                current_lr = optimizer.param_groups[0]["lr"]
                log.info((f"Ep {epoch+1:02d} | Step {global_step:06d} | "
                      f"Train {loss:.3f} | LR {current_lr:.6f} "
                      f"epoch_Time:{spend_time / (global_step + 1) * train_data_len // 60 - spend_time // 60}min;"))

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                current_lr = optimizer.param_groups[0]["lr"]
                log.info(f"Ep {epoch+1:02d} | Step {global_step:06d} | "
                      f"Train {train_loss:.3f} | Val {val_loss:.3f} | LR {current_lr:.6f}")
            if global_step % SAVE_STEP == 0:
                save_model_with_cleanup(model, optimizer, global_step)
                Gtext.generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

# ==============================================================================================
#
#  🔥 核心：超大文件（10GB）流式读取，不占内存，绝对不会OOM
#
# ==============================================================================================
class StreamingJsonDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.json_path = json_path
        self.padding = 0
        # 🔥 关键：只建立行偏移表，不加载全文！
        self.line_offsets = []
        with open(json_path, "r", encoding="utf-8") as f:
            self.line_offsets.append(0)
            while f.readline():
                self.line_offsets.append(f.tell())
            self._total_lines = len(self.line_offsets) - 1

        log.info(f"✅ 数据加载完成：共 {self._total_lines} 条文本，内存占用极低！")

    def __len__(self):
        return self._total_lines

    def __getitem__(self, idx):
        # 🔥 动态读取单行，用完即丢，不会爆内存
        with open(self.json_path, "rb") as f:
            f.seek(self.line_offsets[idx])
            line = f.readline().decode('utf-8')
            data = json.loads(line)
            text = data.get("text", "")

        text = f"{self.tokenizer.bos_token}{text}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)

def create_streaming_dataloader(json_path, tokenizer, batch_size=2, max_length=256, stride=256, shuffle=True):
    dataset = StreamingJsonDataset(json_path, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0  # 大文件必须设为0
    )

# ------------------- 主训练 -------------------
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('/home/ubuntu/work/data/llm-data/pretrained_model/llama2/tokenizer/')
    train_loader = create_streaming_dataloader(
        TRAIN_JSON_PATH, tokenizer, batch_size=BATCH_SIZE,
        max_length=CONTEXT_LENGTH, shuffle=True
    )
    val_loader = create_streaming_dataloader(
        VAL_JSON_PATH, tokenizer, batch_size=BATCH_SIZE,
        max_length=CONTEXT_LENGTH, shuffle=False
    )

    model = GPTModel(GPT_CONFIG_124M)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    total_steps = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    log.info(f"✅ 设备：{DEVICE} | 参数量：{sum(p.numel() for p in model.parameters()):,}")
    log.info("=" * 80)

    start_text = "我是一个中文语言模型"
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, scheduler, DEVICE,
        num_epochs=NUM_EPOCH, eval_freq=EVAL_STEP, eval_iter=5,
        start_context=start_text, tokenizer=tokenizer
    )