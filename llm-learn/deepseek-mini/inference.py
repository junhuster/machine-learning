"""
inference.py — DeepSeek-Mini独立推理/文本生成脚本

可被训练脚本导入，也可单独运行：

    python inference.py \\
        --model_dir ./checkpoints \\
        --tokenizer deepseek-ai/DeepSeek-V3 \\
        --prompt "你好，请介绍一下人工智能" \\
        --max_new_tokens 200 \\
        --temperature 0.8
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from model import ModelArgs, Transformer


# ---------------------------------------------------------------------------
# Checkpoint helpers（供train.py和本脚本共用）
# ---------------------------------------------------------------------------

def _ckpt_step(path: Path) -> int:
    """从 ckpt_step00012345.pt 文件名中解析出步数整数，解析失败返回-1"""
    try:
        return int(path.stem.replace("ckpt_step", ""))
    except ValueError:
        return -1


def _find_latest_checkpoint(model_dir: str) -> Optional[str]:
    """在model_dir中找步数最大的checkpoint，返回路径字符串，没有则返回None"""
    p = Path(model_dir)
    if not p.exists():
        return None
    ckpts = sorted(p.glob("ckpt_step*.pt"), key=_ckpt_step)
    return str(ckpts[-1]) if ckpts else None


def load_model(
    config_path: str,
    model_dir: str,
    device: torch.device,
) -> Transformer:
    """
    从config文件 + 最新checkpoint加载模型，返回eval模式的Transformer。

    Args:
        config_path: config_mini.json路径
        model_dir: checkpoint目录（存放ckpt_step*.pt）
        device: 目标设备
    Returns:
        加载好权重的Transformer，已设置为eval()模式
    """
    with open(config_path) as f:
        cfg = json.load(f)
    args = ModelArgs(**cfg)

    model = Transformer(args).to(device=device, dtype=torch.float16)

    ckpt_path = _find_latest_checkpoint(model_dir)
    if ckpt_path is not None:
        print(f"[inference] 加载checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        print("[inference] 未找到checkpoint，使用随机初始化权重")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> int:
    """
    从logits中采样下一个token的id。

    Args:
        logits: (vocab_size,) 或 (1, vocab_size)
        temperature: 采样温度，越低越确定性；<=0则贪心
        top_p: nucleus sampling阈值
    Returns:
        token id（Python int）
    """
    logits = logits.view(-1).float()

    if temperature <= 0.0:
        return int(logits.argmax().item())

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # Top-p nucleus filtering
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # 去掉累积概率超过top_p的token（保留第一个超过的，确保至少1个token）
    remove_mask = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-9)

    next_idx = torch.multinomial(sorted_probs, num_samples=1)
    return int(sorted_indices[next_idx].item())


# ---------------------------------------------------------------------------
# 核心生成函数（训练脚本和独立推理都调用此函数）
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> str:
    """
    给定prompt，自回归生成文本。

    Args:
        model: 已加载的Transformer，应处于eval()模式
        tokenizer: 与DeepSeek词表兼容的HuggingFace tokenizer
        prompt: 输入文本字符串
        max_new_tokens: 最多生成的新token数
        temperature: 采样温度（<=0为贪心）
        top_p: nucleus sampling阈值
        device: 目标设备；None则自动检测
    Returns:
        生成的文本字符串（不含prompt）
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    # 编码prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    max_seq_len = model.args.max_seq_len

    # 如果prompt太长则截断，保留后段
    if len(input_ids) >= max_seq_len:
        input_ids = input_ids[-(max_seq_len - 1):]

    prompt_len = len(input_ids)
    prompt_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, prompt_len)

    # 清零KV cache
    model.reset_kv_cache()

    # Prefill阶段：处理整个prompt，获取第一个生成token的logits
    logits = model(prompt_tensor, start_pos=0, use_cache=True)  # (1, vocab_size)

    generated_ids = []
    eos_id = tokenizer.eos_token_id

    # 从prefill的logits采样第一个token
    next_token_id = _sample(logits[0], temperature=temperature, top_p=top_p)
    if next_token_id == eos_id:
        if was_training:
            model.train()
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    generated_ids.append(next_token_id)

    # 自回归生成剩余token
    for step in range(1, max_new_tokens):
        cur_pos = prompt_len + step  # 当前要写入cache的位置
        if cur_pos >= max_seq_len:
            break

        cur_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        # start_pos = cur_pos - 1：当前token在序列中的位置
        logits = model(cur_token, start_pos=cur_pos - 1, use_cache=True)  # (1, vocab_size)

        next_token_id = _sample(logits[0], temperature=temperature, top_p=top_p)
        if next_token_id == eos_id:
            break
        generated_ids.append(next_token_id)

    if was_training:
        model.train()

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


@torch.no_grad()
def generate_chat(
    model: Transformer,
    tokenizer,
    messages: list,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> str:
    """
    基于chat messages格式进行对话推理（SFT模型使用）。

    Args:
        model: 已加载的Transformer，应处于eval()模式
        tokenizer: 与DeepSeek词表兼容的HuggingFace tokenizer
        messages: 对话列表，格式如 [{"role": "user", "content": "..."}]
        max_new_tokens: 最多生成的新token数
        temperature: 采样温度（<=0为贪心）
        top_p: nucleus sampling阈值
        device: 目标设备；None则自动检测
    Returns:
        assistant回复的文本字符串
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )


# ---------------------------------------------------------------------------
# CLI入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-Mini文本生成")
    parser.add_argument("--model_dir", type=str, default="./checkpoints",
                        help="checkpoint目录（含ckpt_step*.pt和config_mini.json）")
    parser.add_argument("--config", type=str, default=None,
                        help="config JSON路径（默认：model_dir/config_mini.json或脚本同级目录）")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/DeepSeek-V3",
                        help="HuggingFace tokenizer名称或本地路径")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下人工智能",
                        help="生成提示词")
    parser.add_argument("--chat", action="store_true",
                        help="使用chat模式（SFT模型），将prompt作为user消息")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] device={device}")

    # 查找config文件
    config_path = args.config
    if config_path is None:
        candidate = os.path.join(args.model_dir, "config_mini.json")
        if os.path.exists(candidate):
            config_path = candidate
        else:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_mini.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到config文件: {config_path}")

    print(f"[inference] config={config_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = load_model(config_path, args.model_dir, device)

    print("\n===== Prompt =====")
    print(args.prompt)
    print("\n===== Generated =====")

    if args.chat:
        messages = [{"role": "user", "content": args.prompt}]
        result = generate_chat(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
    else:
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

    print(result)


if __name__ == "__main__":
    main()
