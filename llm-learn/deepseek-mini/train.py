"""
train.py — DeepSeek-Mini在单张T4 GPU上的fp16训练脚本

功能：
  - 读取JSONL训练数据（每行 {"text": "中文语料"}，已裁剪为256字符）
  - 使用DeepSeek-V3开源分词器（不需要自己训练）
  - 余弦退火学习率调度
  - 每SAVE_STEPS保存一次checkpoint（torch.save），最多保留最近2个
  - 支持断点续训（自动查找最新checkpoint）
  - 每LOG_STEPS打印：步数、loss、lr、剩余时间（分钟）、生成样例
  - 日志按天轮转，旧日志文件名带日期后缀

用法：
    python train.py \\
        --data_path ./data/train.jsonl \\
        --model_dir ./checkpoints \\
        --tokenizer deepseek-ai/DeepSeek-V3 \\
        --config_path ./config_mini.json
"""

import argparse
import json
import logging
import math
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
from inference import generate, _find_latest_checkpoint, _ckpt_step
from dataset import PretrainDataset, collate_pretrain


# ===========================================================================
# 训练超参数默认值（可通过命令行参数覆盖）
# ===========================================================================

DEFAULTS = dict(
    # 数据
    data_path="./data/train.jsonl",
    max_seq_len=256,                    # 训练序列长度（和数据裁剪长度对齐）

    # 模型配置
    config_path="./config_mini.json",

    # 分词器
    tokenizer="deepseek-ai/DeepSeek-V3",

    # Checkpoint
    model_dir="./checkpoints",
    save_steps=500,                     # 每N步保存一次
    max_ckpts=2,                        # 最多保留N个最新checkpoint

    # 训练
    num_epochs=3,
    batch_size=8,
    lr=3e-4,                            # 余弦退火起始lr
    lr_min=1e-5,                        # 余弦退火最低lr
    weight_decay=0.01,
    grad_clip=1.0,                      # 梯度裁剪阈值

    # 日志
    log_steps=100,                      # 每N步打印一次日志
    log_dir="./logs",
    start_text="人工智能的发展历史",    # 周期性推理用的提示词
    gen_max_tokens=50,                  # 日志推理最多生成token数
    gen_temperature=0.8,                # 日志推理采样温度
)


# ===========================================================================
# Logger（按天轮转）
# ===========================================================================

def setup_logger(log_dir: str) -> logging.Logger:
    """
    配置日志：
    - 写入log_dir/train.log，每天午夜轮转
    - 旧日志文件名格式：train.log.YYYY-MM-DD
    - 同时输出到控制台
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("deepseek_mini_train")
    logger.setLevel(logging.DEBUG)

    # 避免重复添加handler（被train()多次调用时）
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 每天轮转，后缀精确到天（如 train.log.2025-05-26）
    log_file = os.path.join(log_dir, "train.log")
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=0,          # 保留所有旧日志
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# ===========================================================================
# Checkpoint保存/加载
# ===========================================================================

def ckpt_filename(step: int) -> str:
    """生成checkpoint文件名，步数补零到8位，便于字典序排序"""
    return f"ckpt_step{step:08d}.pt"


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    epoch: int,
    model_dir: str,
    max_ckpts: int,
    logger: logging.Logger,
):
    """
    保存模型checkpoint，并删除超出max_ckpts限制的最老checkpoint。

    保存内容（使用torch.save）：
    - step, epoch: 训练进度
    - model_state_dict: 模型权重
    - optimizer_state_dict: 优化器状态（断点续训需要）
    - scheduler_state_dict: 学习率调度器状态
    - scaler_state_dict: fp16 GradScaler状态
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, ckpt_filename(step))

    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        path,
    )
    logger.info(f"Checkpoint已保存: {path}")

    # 删除超出限制的最旧checkpoint
    ckpts = sorted(Path(model_dir).glob("ckpt_step*.pt"), key=_ckpt_step)
    while len(ckpts) > max_ckpts:
        old = ckpts.pop(0)
        old.unlink()
        logger.info(f"已删除旧checkpoint: {old}")


def load_checkpoint(
    model_dir: str,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    logger: logging.Logger,
) -> tuple:
    """
    加载最新checkpoint。若无checkpoint则从头训练。

    Returns:
        (start_step, start_epoch): 续训的起始步数和epoch数
    """
    ckpt_path = _find_latest_checkpoint(model_dir)
    if ckpt_path is None:
        logger.info("未找到checkpoint，从头开始训练。")
        return 0, 0

    logger.info(f"从checkpoint续训: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_step = ckpt.get("step", 0)
    start_epoch = ckpt.get("epoch", 0)
    logger.info(f"续训起点: step={start_step}, epoch={start_epoch}")
    return start_step, start_epoch


# ===========================================================================
# 主训练函数
# ===========================================================================

def train(args):
    logger = setup_logger(args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"训练设备: {device}")

    # ---- 分词器 ----
    logger.info(f"加载分词器: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # ---- 模型 ----
    logger.info(f"加载模型配置: {args.config_path}")
    with open(args.config_path) as f:
        cfg = json.load(f)

    # 用训练序列长度覆盖config中的max_seq_len
    cfg["max_seq_len"] = args.max_seq_len + 1  # +1确保cache能容纳完整序列
    cfg["max_batch_size"] = args.batch_size

    model_args = ModelArgs(**cfg)
    model = Transformer(model_args).to(device=device, dtype=torch.float16)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count / 1e6:.2f}M")

    # ---- 数据集 ----
    logger.info(f"加载数据: {args.data_path}")
    dataset = PretrainDataset(args.data_path, tokenizer, args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pretrain,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.num_epochs
    logger.info(
        f"数据集大小: {len(dataset)} 样本, "
        f"steps/epoch: {steps_per_epoch}, "
        f"total_steps: {total_steps}"
    )

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # ---- 余弦退火学习率 ----
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr_min,
    )

    # ---- fp16 GradScaler ----
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ---- 断点续训 ----
    start_step, start_epoch = load_checkpoint(
        args.model_dir, model, optimizer, scheduler, scaler, device, logger
    )

    # ---- 训练循环 ----
    global_step = start_step
    run_start_time = time.time()    # 本次运行开始时间
    step_time_accum = 0.0           # 本次运行累计步时间
    steps_this_run = 0              # 本次运行已执行的步数
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        for batch_idx, batch in enumerate(loader):
            # 跳过已完成的步（续训时）
            steps_done_before_this_epoch = epoch * steps_per_epoch
            current_global_step_if_processed = steps_done_before_this_epoch + batch_idx + 1
            if current_global_step_if_processed <= start_step:
                continue

            step_t0 = time.time()

            batch = batch.to(device)        # (B, max_seq_len+1)
            input_ids = batch[:, :-1]       # (B, max_seq_len)  输入
            target_ids = batch[:, 1:]       # (B, max_seq_len)  目标（移位一位）

            # fp16自动混合精度
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                logits = model(input_ids, use_cache=False)   # (B, T, V)
                # 交叉熵loss，忽略pad位置
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=pad_id,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            step_duration = time.time() - step_t0
            step_time_accum += step_duration
            steps_this_run += 1

            # ---- 打印日志 ----
            if global_step % args.log_steps == 0:
                avg_step_time = step_time_accum / steps_this_run
                remaining_steps = total_steps - global_step
                eta_minutes = avg_step_time * remaining_steps / 60.0
                current_lr = scheduler.get_last_lr()[0]

                logger.info(
                    f"epoch={epoch + 1}/{args.num_epochs} | "
                    f"step={global_step}/{total_steps} | "
                    f"loss={loss.item():.4f} | "
                    f"lr={current_lr:.2e} | "
                    f"ETA={eta_minutes:.1f}min"
                )

                # 用start_text做一次推理，观察生成质量
                model.eval()
                try:
                    sample_text = generate(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=args.start_text,
                        max_new_tokens=args.gen_max_tokens,
                        temperature=args.gen_temperature,
                        top_p=0.9,
                        device=device,
                    )
                    logger.info(
                        f"[sample] prompt='{args.start_text}' => '{sample_text}'"
                    )
                except Exception as e:
                    logger.warning(f"[sample] 生成失败: {e}")
                model.train()

            # ---- 保存checkpoint ----
            if global_step % args.save_steps == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    global_step, epoch,
                    args.model_dir, args.max_ckpts, logger,
                )

    # ---- 训练结束，保存最终checkpoint ----
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        global_step, args.num_epochs - 1,
        args.model_dir, args.max_ckpts, logger,
    )
    total_time = (time.time() - run_start_time) / 60.0
    logger.info(f"训练完成。总时间: {total_time:.1f}min, 最终step: {global_step}")


# ===========================================================================
# 命令行参数解析
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="训练DeepSeek-Mini（单张T4 GPU）")

    for key, default in DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", action="store_true", default=default)
        else:
            parser.add_argument(f"--{key}", type=type(default), default=default)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
