"""
train.py — DeepSeek-V4-Mini 预训练脚本

功能：
  - 读取 JSONL 训练数据（每行 {"text": "..."}）
  - 使用 DeepSeek-V3 分词器
  - 主干 next-token prediction loss + MTP 辅助 loss
  - 余弦退火学习率调度
  - 每 SAVE_STEPS 保存一次 checkpoint，最多保留最近 N 个
  - 支持断点续训
  - 日志按天轮转（只写文件，不输出控制台）

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
# 训练超参数默认值
# ===========================================================================

DEFAULTS = dict(
    # 数据
    data_path="/home/ubuntu/work/data/llm-data/train_data/zh/monkey/pretrain_data/monkey_pretrain_310M.jsonl",
    max_seq_len=256,

    # 模型配置
    config_path="./config_mini.json",

    # 分词器
    tokenizer="/home/ubuntu/work/data/llm-data/pretrained_model/llama2/tokenizer/",

    # Checkpoint
    model_dir="/home/ubuntu/work/data/llm-data/pretrained_model/deepseek-v4-mini/32G/p_model/",
    save_steps=50,
    max_ckpts=2,

    # 训练
    num_epochs=1,
    batch_size=4,
    grad_accum_steps=8,
    lr=5e-5,
    lr_min=1e-5,
    weight_decay=0.01,
    grad_clip=1.0,

    # MTP 辅助 loss 权重（0=不使用 MTP loss）
    mtp_loss_weight=0.1,

    # 日志
    log_steps=50,
    log_dir="/home/ubuntu/work/logs/deepseek-v4-mini/",
    start_text="人工智能的发展历史",
    gen_max_tokens=50,
    gen_temperature=0.8,
)


# ===========================================================================
# Logger
# ===========================================================================

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("deepseek_v4_mini_train")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_file = os.path.join(log_dir, "train.log")
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
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, ckpt_filename(step))
    torch.save(
        {
            "step":               step,
            "epoch":              epoch,
            "model_state_dict":   model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":  scaler.state_dict(),
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
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: torch.cuda.amp.GradScaler,
        device: torch.device,
        logger: logging.Logger,
) -> tuple:
    ckpt_path = _find_latest_checkpoint(model_dir)
    if ckpt_path is None:
        logger.info("未找到 checkpoint，从头开始训练。")
        return 0, 0

    logger.info(f"从 checkpoint 续训: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_step  = ckpt.get("step", 0)
    start_epoch = ckpt.get("epoch", 0)
    logger.info(f"续训起点: step={start_step}, epoch={start_epoch}")
    return start_step, start_epoch


# ===========================================================================
# MTP Loss
# ===========================================================================

def compute_mtp_loss(
        model: Transformer,
        h: torch.Tensor,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        pad_id: int,
        grad_accum_steps: int,
) -> torch.Tensor:
    """
    计算所有 MTP 层的辅助 loss。

    Args:
        h:              主干最后一层的 HC 隐层状态 (B, T, hc_mult, dim)
        input_ids:      (B, T) 当前步输入
        target_ids:     (B, T) 目标 token（右移一位）
        pad_id:         padding token id
        grad_accum_steps: 梯度累积步数（用于 loss 缩放）
    Returns:
        累加的 MTP loss（已除以 grad_accum_steps）
    """
    mtp_loss_total = torch.tensor(0.0, device=h.device, dtype=torch.float32)
    for k, mtp_block in enumerate(model.mtp):
        # MTP 层预测的是比主干再往后一个位置的 token
        # input_ids 右移 k+1 位：mtp 用 [t+k+1] 处的 embed 融合 [t] 处的隐层
        mtp_input = torch.roll(input_ids, -(k + 1), dims=1)  # (B, T)
        mtp_target = torch.roll(target_ids, -(k + 1), dims=1)  # (B, T)

        mtp_logits = mtp_block(h.detach(), start_pos=0, input_ids=mtp_input)
        # (B, T, vocab_size)

        mtp_loss = F.cross_entropy(
            mtp_logits.view(-1, mtp_logits.size(-1)),
            mtp_target.reshape(-1),
            ignore_index=pad_id,
        ) / grad_accum_steps
        mtp_loss_total = mtp_loss_total + mtp_loss

    return mtp_loss_total


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
    pad_id    = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ---- 模型 ----
    logger.info(f"加载模型配置: {args.config_path}")
    with open(args.config_path) as f:
        cfg = json.load(f)

    cfg["max_seq_len"]   = args.max_seq_len + 1
    cfg["max_batch_size"] = args.batch_size

    model_args = ModelArgs(**cfg)
    model      = Transformer(model_args).to(device=device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count / 1e6:.2f}M")

    # ---- 数据集 ----
    logger.info(f"加载数据: {args.data_path}")
    dataset = PretrainDataset(args.data_path, tokenizer, args.max_seq_len)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pretrain,
        drop_last=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    steps_per_epoch = len(loader)
    total_steps     = steps_per_epoch * args.num_epochs // args.grad_accum_steps
    logger.info(
        f"数据集: {len(dataset)} 样本 | steps/epoch: {steps_per_epoch} | "
        f"grad_accum: {args.grad_accum_steps} | total_optimizer_steps: {total_steps}"
    )

    # ---- 优化器 ----
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

    # ---- 断点续训 ----
    start_step, start_epoch = load_checkpoint(
        args.model_dir, model, optimizer, scheduler, scaler, device, logger,
    )

    # ---- 训练循环 ----
    global_step    = start_step
    run_start_time = time.time()
    step_time_accum = 0.0
    steps_this_run  = 0
    accum_loss      = 0.0

    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        step_t0 = time.time()
        for batch_idx, batch in enumerate(loader):
            if batch_idx + epoch * steps_per_epoch < start_step * args.grad_accum_steps:
                continue

            is_first_in_accum = (batch_idx % args.grad_accum_steps == 0)
            is_last_in_accum  = (batch_idx % args.grad_accum_steps == args.grad_accum_steps - 1)

            if is_first_in_accum:
                optimizer.zero_grad()
                accum_loss = 0.0

            batch      = batch.to(device)             # (B, max_seq_len+1)
            input_ids  = batch[:, :-1]                # (B, max_seq_len)
            target_ids = batch[:, 1:]                 # (B, max_seq_len)

            skip_batch = False
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                if args.mtp_loss_weight > 0 and len(model.mtp) > 0:
                    logits, h = model(input_ids, start_pos=0, use_cache=False, return_hidden=True)
                else:
                    logits = model(input_ids, start_pos=0, use_cache=False)

                main_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=pad_id,
                ) / args.grad_accum_steps

                # ---- MTP 辅助 loss ----
                if args.mtp_loss_weight > 0 and len(model.mtp) > 0:
                    mtp_loss = compute_mtp_loss(
                        model, h, input_ids, target_ids, pad_id, args.grad_accum_steps,
                    )
                    loss = main_loss + args.mtp_loss_weight * mtp_loss
                else:
                    loss = main_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"batch={batch_idx} loss nan/inf，跳过")
                    skip_batch = True
                else:
                    scaler.scale(loss).backward()

            if skip_batch:
                optimizer.zero_grad()
                continue

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
                    avg_step_time  = step_time_accum / steps_this_run
                    remaining_steps = total_steps - global_step
                    eta_minutes    = avg_step_time * remaining_steps / 60.0
                    current_lr     = scheduler.get_last_lr()[0]
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
                        logger.info(f"[sample] prompt='{args.start_text}' => '{sample_text}'")
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
    logger.info(f"训练完成。总时间: {total_time:.1f}min，最终 step: {global_step}")


# ===========================================================================
# 命令行参数
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Mini 预训练")
    for key, default in DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", action="store_true", default=default)
        else:
            parser.add_argument(f"--{key}", type=type(default), default=default)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
