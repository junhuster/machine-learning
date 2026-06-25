"""
train_sft.py — DeepSeek-V4-Mini 指令微调（SFT）训练脚本

功能：
  - 读取 JSONL 格式 SFT 数据（每行为 messages 列表，chat 格式）
  - 只对 assistant 回复部分计算 loss（loss_mask）
  - 从预训练 checkpoint 加载权重，在此基础上继续训练
  - 余弦退火学习率调度
  - 支持断点续训
  - 日志按天轮转（只写文件）

用法：
    python train_sft.py \\
        --data_path ./data/sft.jsonl \\
        --pretrain_ckpt_dir ./checkpoints \\
        --model_dir ./checkpoints_sft \\
        --tokenizer deepseek-ai/DeepSeek-V3 \\
        --config_path ./config_mini.json
"""

import argparse
import json
import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import ModelArgs, Transformer
from inference import generate_chat, _find_latest_checkpoint, _ckpt_step
from dataset import SFTDataset, collate_sft


# ===========================================================================
# 训练超参数默认值
# ===========================================================================

DEFAULTS = dict(
    data_path="/home/ubuntu/work/data/llm-data/train_data/zh/monkey/sft_data/BelleGroup_sft_small.jsonl",
    max_seq_len=512,

    config_path="./config_mini.json",
    tokenizer="/home/ubuntu/work/data/llm-data/pretrained_model/llama2/tokenizer/",

    pretrain_ckpt_dir="/home/ubuntu/work/data/llm-data/pretrained_model/deepseek-v4-mini/32G/p_model/",
    model_dir="/home/ubuntu/work/data/llm-data/pretrained_model/deepseek-v4-mini/32G/sft_model/",
    save_steps=500,
    max_ckpts=2,

    num_epochs=1,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    lr_min=1e-6,
    weight_decay=0.01,
    grad_clip=1.0,

    log_steps=100,
    log_dir="/home/ubuntu/work/logs/deepseek-v4-mini/",
    gen_max_tokens=100,
    gen_temperature=0.7,
)


# ===========================================================================
# Logger
# ===========================================================================

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("deepseek_v4_mini_sft")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_file = os.path.join(log_dir, "train_sft.log")
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=0, encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


# ===========================================================================
# Checkpoint 保存/加载
# ===========================================================================

def ckpt_filename(step: int) -> str:
    return f"ckpt_step{step:08d}.pt"


def save_checkpoint(
    model, optimizer, scheduler, scaler,
    step, epoch, model_dir, max_ckpts, logger,
):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, ckpt_filename(step))
    torch.save(
        {
            "step":                 step,
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
        },
        path,
    )
    logger.info(f"Checkpoint 已保存: {path}")

    ckpts = sorted(Path(model_dir).glob("ckpt_step*.pt"), key=_ckpt_step)
    while len(ckpts) > max_ckpts:
        old = ckpts.pop(0)
        old.unlink()
        logger.info(f"已删除旧 checkpoint: {old}")


def load_pretrain_weights(
    pretrain_ckpt_dir: str,
    model: Transformer,
    device: torch.device,
    logger: logging.Logger,
):
    """从预训练 checkpoint 加载权重（strict=False，允许部分加载）。"""
    ckpt_path = _find_latest_checkpoint(pretrain_ckpt_dir)
    if ckpt_path is None:
        logger.info("未找到预训练 checkpoint，使用随机初始化权重。")
        return
    logger.info(f"从预训练 checkpoint 加载权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)


def load_sft_checkpoint(
    model_dir, model, optimizer, scheduler, scaler, device, logger,
) -> tuple:
    ckpt_path = _find_latest_checkpoint(model_dir)
    if ckpt_path is None:
        logger.info("未找到 SFT checkpoint，从预训练权重开始。")
        return 0, 0
    logger.info(f"从 SFT checkpoint 续训: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("step", 0), ckpt.get("epoch", 0)


# ===========================================================================
# 主训练函数
# ===========================================================================

def train(args):
    logger = setup_logger(args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"训练设备: {device}")

    logger.info(f"加载分词器: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    with open(args.config_path) as f:
        cfg = json.load(f)
    cfg["max_seq_len"]    = args.max_seq_len
    cfg["max_batch_size"] = args.batch_size

    model_args = ModelArgs(**cfg)
    model      = Transformer(model_args).to(device=device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    dataset = SFTDataset(args.data_path, tokenizer, args.max_seq_len)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sft,
        drop_last=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    steps_per_epoch = len(loader)
    total_steps     = steps_per_epoch * args.num_epochs // args.grad_accum_steps
    logger.info(
        f"数据集: {len(dataset)} 样本 | steps/epoch: {steps_per_epoch} | "
        f"total_optimizer_steps: {total_steps}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr_min,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 优先从 SFT checkpoint 续训，否则从预训练权重初始化
    start_step, start_epoch = load_sft_checkpoint(
        args.model_dir, model, optimizer, scheduler, scaler, device, logger,
    )
    if start_step == 0:
        load_pretrain_weights(args.pretrain_ckpt_dir, model, device, logger)

    global_step    = start_step
    run_start_time = time.time()
    step_time_accum = 0.0
    steps_this_run  = 0
    accum_loss      = 0.0

    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        step_t0 = time.time()
        for batch_idx, (X, Y, loss_mask) in enumerate(loader):
            if batch_idx + epoch * steps_per_epoch < start_step * args.grad_accum_steps:
                continue

            is_first_in_accum = (batch_idx % args.grad_accum_steps == 0)
            is_last_in_accum  = (batch_idx % args.grad_accum_steps == args.grad_accum_steps - 1)

            if is_first_in_accum:
                optimizer.zero_grad()
                accum_loss = 0.0

            X, Y, loss_mask = X.to(device), Y.to(device), loss_mask.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                logits = model(X, start_pos=0, use_cache=False)  # (B, T, V)
                # 只对 assistant 部分计算 loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    Y.reshape(-1),
                    reduction="none",
                )
                loss = (loss * loss_mask.reshape(-1)).sum() / (loss_mask.sum() + 1e-9)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if is_last_in_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                global_step += 1
                step_time_accum = time.time() - step_t0
                steps_this_run  += 1

                if global_step % args.log_steps == 0:
                    avg_step_time   = step_time_accum / steps_this_run
                    remaining_steps = total_steps - global_step
                    eta_minutes     = avg_step_time * remaining_steps / 60.0
                    current_lr      = scheduler.get_last_lr()[0]
                    exe_tm = (time.time() - run_start_time) / 60.0
                    logger.info(
                        f"epoch={epoch + 1}/{args.num_epochs} | "
                        f"step={global_step}/{total_steps} | "
                        f"loss={accum_loss:.4f} | "
                        f"lr={current_lr:.2e} | "
                        f"ExeTime={exe_tm:.1f}min |"
                        f"ETA={eta_minutes:.1f}min"
                    )

                    model.eval()
                    prompt=[{"你好，介绍一下自己"}]
                    try:
                        sample = generate_chat(
                            model=model, tokenizer=tokenizer,
                            messages=prompt,
                            max_new_tokens=args.gen_max_tokens,
                            temperature=args.gen_temperature,
                            device=device,
                        )
                        logger.info(f"[prompt] {prompt} -> Ans:{sample}")
                    except Exception as e:
                        logger.warning(f"[sample] 生成失败: {e}")
                    model.train()

                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        global_step, epoch,
                        args.model_dir, args.max_ckpts, logger,
                    )

    save_checkpoint(
        model, optimizer, scheduler, scaler,
        global_step, args.num_epochs - 1,
        args.model_dir, args.max_ckpts, logger,
    )
    total_time = (time.time() - run_start_time) / 60.0
    logger.info(f"SFT 训练完成。总时间: {total_time:.1f}min，最终 step: {global_step}")


# ===========================================================================
# 命令行参数
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Mini SFT 训练")
    for key, default in DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", action="store_true", default=default)
        else:
            parser.add_argument(f"--{key}", type=type(default), default=default)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
