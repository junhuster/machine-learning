"""
train.py — Llama4-Mini 在单张 T4 GPU 上的 fp16 训练脚本

功能：
  - 读取 JSONL 训练数据（每行 {"text": "语料"}）
  - 使用 meta-llama/Llama-4-Scout-17B-16E 分词器（不需要自己训练）
  - 仅使用文本语言模型部分（Llama4ForCausalLM），无视觉/多模态组件
  - 余弦退火学习率调度
  - 每 SAVE_STEPS 保存一次 checkpoint（torch.save），最多保留最近 max_ckpts 个
  - 支持断点续训（自动查找最新 checkpoint）
  - 每 LOG_STEPS 打印：步数、loss、lr、剩余时间（分钟）、生成样例
  - 日志按天轮转，旧日志文件名带日期后缀

用法：
    # 全部使用默认值
    python train.py

    # 指定数据路径和 checkpoint 目录
    python train.py --data_path ./data/train.jsonl --model_dir ./checkpoints

    # 常用参数（其余参数有默认值，可省略）：
    #   --data_path      ./data/train.jsonl
    #   --config_path    ./config_mini.json
    #   --tokenizer      meta-llama/Llama-4-Scout-17B-16E
    #   --model_dir      ./checkpoints
    #   --num_epochs     3
    #   --batch_size     4
    #   --max_seq_len    256
    #   --lr             3e-4
    #   --save_steps     500
    #   --log_steps      100
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

from inference import _find_latest_checkpoint, _ckpt_step, generate
from dataset import PretrainDataset, collate_pretrain


# ===========================================================================
# 训练超参数默认值（可通过命令行参数覆盖）
# ===========================================================================

DEFAULTS = dict(
    # 数据
    data_path="./data/train.jsonl",
    max_seq_len=256,                    # 训练序列长度

    # 模型配置
    config_path="./config_mini.json",

    # 分词器
    tokenizer="meta-llama/Llama-4-Scout-17B-16E",

    # Checkpoint
    model_dir="./checkpoints",
    save_steps=500,                     # 每 N 步保存一次
    max_ckpts=2,                        # 最多保留 N 个最新 checkpoint

    # 训练
    num_epochs=1,
    batch_size=4,
    grad_accum_steps=1,                 # 梯度累积步数；1=不累积，>1=累积N步后更新一次参数
    lr=3e-4,                            # 余弦退火起始 lr
    lr_min=1e-5,                        # 余弦退火最低 lr
    weight_decay=0.01,
    grad_clip=1.0,                      # 梯度裁剪阈值

    # 日志
    log_steps=100,                      # 每 N 步打印一次日志
    log_dir="./logs",
    start_text="人工智能的发展历史",    # 周期性推理用的提示词
    gen_max_tokens=50,                  # 日志推理最多生成 token 数
    gen_temperature=0.8,                # 日志推理采样温度
)


# ===========================================================================
# Logger（按天轮转）
# ===========================================================================

def setup_logger(log_dir: str) -> logging.Logger:
    """
    配置日志：
    - 写入 log_dir/train.log，每天午夜轮转
    - 旧日志文件名格式：train.log.YYYY-MM-DD
    - 同时输出到控制台
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("llama4_mini_train")
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler（被 train() 多次调用时）
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 每天轮转，后缀精确到天（如 train.log.2025-05-28）
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
    保存模型 checkpoint，并删除超出 max_ckpts 限制的最老 checkpoint。

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

    # 删除超出限制的最旧 checkpoint
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
    """
    加载最新 checkpoint。若无 checkpoint 则从头训练。

    Returns:
        (start_step, start_epoch): 续训的起始步数和 epoch 数
    """
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
    # llama4 tokenizer 可能没有 pad_token，用 eos_token 代替
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
    # optimizer 实际更新次数 = batch 总数 / grad_accum_steps
    total_steps = steps_per_epoch * args.num_epochs // args.grad_accum_steps
    logger.info(
        f"数据集大小: {len(dataset)} 样本, "
        f"steps/epoch: {steps_per_epoch}, "
        f"grad_accum_steps: {args.grad_accum_steps}, "
        f"total_optimizer_steps: {total_steps}"
    )

    # ---- 优化器 ----
    # AdamW：Adam + 权重衰减解耦（weight_decay 不作用于 bias/norm 参数）
    # betas=(0.9, 0.95)：LLM 训练常用设置，beta2 比默认的 0.999 小，对梯度二阶矩更新更快
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # ---- 余弦退火学习率 ----
    # lr 从 args.lr 开始，按余弦曲线平滑下降到 args.lr_min
    # T_max=total_steps：在整个训练周期内完成一个完整的余弦周期
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr_min,
    )

    # ---- fp16 GradScaler ----
    # T4 不支持 bf16，使用 fp16 + GradScaler 防止梯度下溢
    # GradScaler 动态调整 loss 缩放因子：梯度出现 inf/nan 时自动缩小 scale，稳定后逐渐增大
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ---- 断点续训 ----
    start_step, start_epoch = load_checkpoint(
        args.model_dir, model, optimizer, scheduler, scaler, device, logger
    )

    # ---- 训练循环 ----
    global_step = start_step
    run_start_time = time.time()    # 本次运行开始时间
    step_time_accum = 0.0           # 本次运行累计步时间（按 optimizer step 计）
    steps_this_run = 0              # 本次运行已完成的 optimizer step 数
    pad_id = tokenizer.pad_token_id
    accum_loss = 0.0                # 当前累积周期内的累计 loss（用于日志展示）

    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        for batch_idx, batch in enumerate(loader):
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

            batch = batch.to(device)        # (B, max_seq_len+1)
            input_ids = batch[:, :-1]       # (B, max_seq_len)  输入：去掉最后一个 token
            target_ids = batch[:, 1:]       # (B, max_seq_len)  目标：去掉第一个 token（右移一位）

            # 将 pad 位置设为 -100，Llama4ForCausalLM 内置 loss_function 会忽略这些位置
            labels = target_ids.clone()
            labels[labels == pad_id] = -100

            # fp16 自动混合精度：前向计算在 fp16 下进行，降低显存并加速
            # use_cache=False：训练时不需要 KV cache（每步都是全序列前向）
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    use_cache=False,
                )
                # 梯度累积：loss 除以累积步数，使多步梯度之和等价于一次大 batch 的梯度
                loss = outputs.loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            # 仅在累积满 grad_accum_steps 步时执行参数更新
            if is_last_in_accum:
                # 标准 fp16 训练步骤：
                # 1. scaler.unscale_：将梯度缩放回原始尺度，才能正确裁剪
                # 2. clip_grad_norm_：梯度裁剪，防止梯度爆炸
                # 3. scaler.step：如果梯度有 inf/nan 则跳过此步
                # 4. scaler.update：根据本步是否出现 inf/nan 动态调整 scale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # 每个 optimizer step 更新 lr（余弦退火按 optimizer step 计）

                global_step += 1
                step_time_accum += time.time() - step_t0
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
                        f"loss={accum_loss:.4f} | "
                        f"lr={current_lr:.2e} | "
                        f"ETA={eta_minutes:.1f}min"
                    )

                    # 用 start_text 做一次推理，观察生成质量
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
    logger.info(f"训练完成。总时间: {total_time:.1f}min, 最终 step: {global_step}")


# ===========================================================================
# 命令行参数解析
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="训练 Llama4-Mini（单张 T4 GPU）")

    # 从 DEFAULTS 字典动态生成所有命令行参数，避免重复维护
    # bool 类型用 action="store_true"，其他类型保持原始 type
    for key, default in DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", action="store_true", default=default)
        else:
            parser.add_argument(f"--{key}", type=type(default), default=default)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
