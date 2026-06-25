"""
inference.py — DeepSeek-V4-Mini 推理脚本

用法：
    python inference.py \\
        --model_dir ./checkpoints \\
        --config ./config_mini.json \\
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
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_step(path: Path) -> int:
    """从 ckpt_step00012345.pt 文件名中解析步数，解析失败返回 -1。"""
    try:
        return int(path.stem.replace("ckpt_step", ""))
    except ValueError:
        return -1


def _find_latest_checkpoint(model_dir: str) -> Optional[str]:
    """在 model_dir 中找步数最大的 checkpoint，返回路径字符串，没有则返回 None。"""
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
    """从 config + 最新 checkpoint 加载模型，返回 eval 模式的 Transformer。"""
    with open(config_path) as f:
        cfg = json.load(f)
    args = ModelArgs(**cfg)
    model = Transformer(args).to(device=device, dtype=torch.float16)

    ckpt_path = _find_latest_checkpoint(model_dir)
    if ckpt_path is not None:
        print(f"[inference] 加载 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        print("[inference] 未找到 checkpoint，使用随机初始化权重")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> int:
    """从 logits 采样下一个 token id。"""
    logits = logits.view(-1).float()

    if temperature <= 0.0:
        return int(logits.argmax().item())

    logits = logits / temperature
    probs  = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative  = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-9)

    next_idx = torch.multinomial(sorted_probs, num_samples=1)
    return int(sorted_indices[next_idx].item())


# ---------------------------------------------------------------------------
# 生成函数
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
    """给定 prompt，自回归生成文本。推理时不调用 MTP 层。"""
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    input_ids   = tokenizer.encode(prompt, add_special_tokens=True)
    max_seq_len = model.args.max_seq_len

    if len(input_ids) >= max_seq_len:
        input_ids = input_ids[-(max_seq_len - 1):]

    prompt_len    = len(input_ids)
    prompt_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    model.reset_kv_cache()

    # Prefill
    logits = model(prompt_tensor, start_pos=0, use_cache=True)  # (1, vocab_size)

    generated_ids = []
    eos_id        = tokenizer.eos_token_id

    next_token_id = _sample(logits[0], temperature=temperature, top_p=top_p)
    if next_token_id == eos_id:
        if was_training:
            model.train()
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    generated_ids.append(next_token_id)

    # Decode
    for step in range(1, max_new_tokens):
        cur_pos = prompt_len + step
        if cur_pos >= max_seq_len:
            break

        cur_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        logits    = model(cur_token, start_pos=cur_pos - 1, use_cache=True)  # (1, vocab_size)

        next_token_id = _sample(logits[0], temperature=temperature, top_p=top_p)
        if next_token_id == eos_id:
            break
        generated_ids.append(next_token_id)

    if was_training:
        model.train()

    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def chat_template(tokenizer, prompt):
        message = [
            {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

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

    """基于 chat messages 格式进行对话推理（SFT 模型使用）。"""
    prompt = chat_template(tokenizer, messages)
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
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Mini 文本生成")
    parser.add_argument("--model_dir",      type=str, default="./checkpoints")
    parser.add_argument("--config",         type=str, default=None)
    parser.add_argument("--tokenizer",      type=str, default="/home/ubuntu/work/data/llm-data/pretrained_model/llama2/tokenizer/")
    parser.add_argument("--prompt",         type=str, default="你好，请介绍一下人工智能")
    parser.add_argument("--chat",           action="store_true")
    parser.add_argument("--max_new_tokens", type=int,   default=200)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top_p",          type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] device={device}")

    config_path = args.config
    if config_path is None:
        candidate = os.path.join(args.model_dir, "config_mini.json")
        config_path = candidate if os.path.exists(candidate) else \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_mini.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到 config 文件: {config_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model     = load_model(config_path, args.model_dir, device)

    print("\n===== Prompt =====")
    print(args.prompt)
    print("\n===== Generated =====")

    if args.chat:
        result = generate_chat(
            model=model, tokenizer=tokenizer,
            messages=[{"role": "user", "content": args.prompt}],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, device=device,
        )
    else:
        result = generate(
            model=model, tokenizer=tokenizer, prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, device=device,
        )

    print(result)


if __name__ == "__main__":
    main()
