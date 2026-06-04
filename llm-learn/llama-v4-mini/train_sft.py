"""
train_sft.py — Llama4-Mini 指令微调（SFT）训练脚本

功能：
  - 读取 JSONL 格式 SFT 数据（每行为 messages 列表，chat 格式）
  - 只对 assistant 回复部分计算 loss（loss_mask）
  - 从预训练 checkpoint 加载权重，在此基础上继续训练
  - 余弦退火学习率调度
  - 每 SAVE_STEPS 保存一次 checkpoint，最多保留最近 max_ckpts 个
  - 支持断点续训（自动查找最新 checkpoint）
  - 每 LOG_STEPS 打印：步数、loss、lr、剩余时间、生成样例

用法：
    python train_sft.py \\
        --data_path ./data/sft.jsonl \\
        --pretrain_ckpt_dir ./checkpoints \\
        --model_dir ./checkpoints_sft \\
        --tokenizer meta-llama/Llama-4-Scout-17B-16E \\
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
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM

from inference import _find_latest_checkpoint, _ckpt_step, generate_chat
from dataset import SFTDataset, collate_sft


# ===========================================================================
# 训练超参数默认值
# ===========================================================================

DEFAULTS = dict(
    # 数据
    data_path="./data/sft.jsonl",
    max_seq_len=512,

    # 模型配置
    config_path="./config_mini.json",

    # 分词器
    tokenizer="meta-llama/Llama-4-Scout-17B-16E",

    # 预训练权重目录（用于初始化，若为空则随机初始化）
    pretrain_ckpt_dir="./checkpoints",

    # SFT Checkpoint 保存目录
    model_dir="./checkpoints_sft",
    save_steps=500,
    max_ckpts=2,

    # 训练
    num_epochs=1,
    batch_size=4,
    grad_accum_steps=1,                 # 梯度累积步数；1=不累积，>1=累积N步后更新一次参数
    lr=1e-4,                            # SFT 通常比预训练 lr 小
    lr_min=1e-6,
    weight_decay=0.01,
    grad_clip=1.0,

    # 日志
    log_steps=100,
    log_dir="./logs_sft",
    gen_max_tokens=100,
    gen_temperature=0.7,
)


# ===========================================================================
# Logger
# ===========================================================================

def setup_logger(log_dir: str) -> logging.Logger:
    """
    配置日志：写入 log_dir/train_sft.log，每天午夜轮转，同时输出到控制台。
    避免重复添加 handler（train() 被多次调用时）。
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("llama4_mini_sft")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file = os.path.join(log_dir, "train_sft.log")
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=0, encoding="utf-8"
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# ===========================================================================
# Checkpoint 保存/加载
# ===========================================================================

def ckpt_filename(step: int) -> str:
    """生成 checkpoint 文件名，步数补零到 8 位，便于字典序排序"""
    return f"ckpt_step{step:08d}.pt"


def save_checkpoint(
    model: Llama4ForCausalLM,
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
    保存 SFT checkpoint，并删除超出 max_ckpts 限制的最老 checkpoint。

    保存内容（使用 torch.save）：
    - step, epoch: 训练进度
    - model_state_dict: 模型权重
    - optimizer_state_dict: 优化器状态（断点续训需要）
    - scheduler_state_dict: 学习率调度器状态
    - scaler_state_dict: fp16 GradScaler 状态
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
    logger.info(f"Checkpoint 已保存: {path}")

    ckpts = sorted(Path(model_dir).glob("ckpt_step*.pt"), key=_ckpt_step)
    while len(ckpts) > max_ckpts:
        old = ckpts.pop(0)
        old.unlink()
        logger.info(f"已删除旧 checkpoint: {old}")


def load_checkpoint(
    model_dir: str,
    model: Llama4ForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    logger: logging.Logger,
) -> tuple:
    """加载 SFT 自己的 checkpoint（断点续训）"""
    ckpt_path = _find_latest_checkpoint(model_dir)
    if ckpt_path is None:
        logger.info("未找到 SFT checkpoint，从头开始 SFT 训练。")
        return 0, 0

    logger.info(f"从 SFT checkpoint 续训: {ckpt_path}")
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


def load_pretrain_weights(
    pretrain_ckpt_dir: str,
    model: Llama4ForCausalLM,
    device: torch.device,
    logger: logging.Logger,
):
    """从预训练 checkpoint 加载模型权重（仅权重，不加载 optimizer 等状态）"""
    ckpt_path = _find_latest_checkpoint(pretrain_ckpt_dir)
    if ckpt_path is None:
        logger.warning(f"未找到预训练 checkpoint（{pretrain_ckpt_dir}），使用随机初始化权重。")
        return
    logger.info(f"加载预训练权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # strict=False：允许预训练权重与当前模型结构存在细微差异
    # （例如配置变动导致部分层名称不匹配），缺失的键使用随机初始化值


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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- 模型 ----
    logger.info(f"加载模型配置: {args.config_path}")
    with open(args.config_path) as f:
        cfg = json.load(f)

    text_config = Llama4TextConfig(**cfg)
    model = Llama4ForCausalLM(text_config)
    model = model.to(device=device, dtype=torch.float16)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count / 1e6:.2f}M")

    # ---- 数据集 ----
    logger.info(f"加载 SFT 数据: {args.data_path}")
    dataset = SFTDataset(args.data_path, tokenizer, args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sft,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    steps_per_epoch = len(loader)
    # optimizer 实际更新次数 = batch 总数 / grad_accum_steps
    total_steps = steps_per_epoch * args.num_epochs // args.grad_accum_steps
    logger.info(
        f"数据集大小: {len(dataset)} 样本, "
        f"steps/epoch: {steps_per_epoch}, "
        f"grad_accum_steps: {args.grad_accum_steps}, "
        f"total_optimizer_steps: {total_steps}"
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

    # ---- 初始化权重：优先从 SFT checkpoint 续训，否则加载预训练权重 ----
    sft_ckpt_exists = _find_latest_checkpoint(args.model_dir) is not None
    if sft_ckpt_exists:
        start_step, start_epoch = load_checkpoint(
            args.model_dir, model, optimizer, scheduler, scaler, device, logger
        )
    else:
        load_pretrain_weights(args.pretrain_ckpt_dir, model, device, logger)
        start_step, start_epoch = 0, 0

    # ---- 训练循环 ----
    global_step = start_step
    run_start_time = time.time()
    step_time_accum = 0.0
    steps_this_run = 0
    accum_loss = 0.0                # 当前累积周期内的累计 loss（用于日志展示）

    sample_messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]

    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        for batch_idx, (X, Y, loss_mask) in enumerate(loader):
            # 断点续训跳步：global_step 是 optimizer 更新次数，
            # 对应的 batch 起始位置是 start_step * grad_accum_steps
            if batch_idx + epoch * steps_per_epoch < start_step * args.grad_accum_steps:
                continue

            # 每个累积周期的第一个 batch 时清零梯度、记录计时起点
            is_first_in_accum = (batch_idx % args.grad_accum_steps == 0)
            is_last_in_accum = (batch_idx % args.grad_accum_steps == args.grad_accum_steps - 1)

            if is_first_in_accum:
                optimizer.zero_grad()
                step_t0 = time.time()
                accum_loss = 0.0

            X = X.to(device)                            # (B, seq_len)
            Y = Y.to(device)                            # (B, seq_len)
            loss_mask = loss_mask.to(device).float()    # (B, seq_len)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                # 用 logits only 模式：不传 labels，手动计算 masked loss
                outputs = model(input_ids=X, use_cache=False)
                logits = outputs.logits                 # (B, T, V)

                loss_flat = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    Y.reshape(-1),
                    reduction="none",
                )
                loss_flat = loss_flat * loss_mask.reshape(-1)
                # clamp(min=1.0) 防止 loss_mask 全为 0 时出现除零错误
                # （理论上不应发生，但数据存在全非 assistant 行时做兜底保护）
                denom = loss_mask.sum().clamp(min=1.0)
                # 梯度累积：loss 除以累积步数，使多步梯度之和等价于一次大 batch 的梯度
                loss = loss_flat.sum() / denom / args.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            # 仅在累积满 grad_accum_steps 步时执行参数更新
            if is_last_in_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                global_step += 1
                step_time_accum += time.time() - step_t0
                steps_this_run += 1

                # ---- 打印日志 ----
                if global_step % args.log_steps == 0:
                    avg_step_time = step_time_accum / steps_this_run
                    eta_minutes = avg_step_time * (total_steps - global_step) / 60.0
                    current_lr = scheduler.get_last_lr()[0]

                    logger.info(
                        f"epoch={epoch + 1}/{args.num_epochs} | "
                        f"step={global_step}/{total_steps} | "
                        f"loss={accum_loss:.4f} | "
                        f"lr={current_lr:.2e} | "
                        f"ETA={eta_minutes:.1f}min"
                    )

                    model.eval()
                    try:
                        sample_text = generate_chat(
                            model=model,
                            tokenizer=tokenizer,
                            messages=sample_messages,
                            max_new_tokens=args.gen_max_tokens,
                            temperature=args.gen_temperature,
                            top_p=0.9,
                            device=device,
                        )
                        logger.info(f"[sample] {sample_messages[0]['content']} => {sample_text}")
                    except Exception as e:
                        logger.warning(f"[sample] 生成失败: {e}")
                    model.train()

                # ---- 保存 checkpoint ----
                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        global_step, epoch,
                        args.model_dir, args.max_ckpts, logger,
                    )

    # ---- 训练结束，保存最终 checkpoint ----
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        global_step, args.num_epochs - 1,
        args.model_dir, args.max_ckpts, logger,
    )
    total_time = (time.time() - run_start_time) / 60.0
    logger.info(f"SFT 训练完成。总时间: {total_time:.1f}min, 最终 step: {global_step}")


# ===========================================================================
# 命令行参数解析
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Llama4-Mini 指令微调（SFT）")

    for key, default in DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", action="store_true", default=default)
        else:
            parser.add_argument(f"--{key}", type=type(default), default=default)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
