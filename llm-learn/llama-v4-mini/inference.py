"""
inference.py — Llama4-Mini 独立推理/文本生成脚本

可被训练脚本导入，也可单独运行：

    # 全部使用默认值
    python inference.py

    # 指定 checkpoint 目录和提示词
    python inference.py --model_dir ./checkpoints --prompt "请介绍大语言模型"

    # 常用参数（其余参数有默认值，可省略）：
    #   --model_dir       ./checkpoints
    #   --config          ./config_mini.json（脚本同级目录）
    #   --tokenizer       meta-llama/Llama-4-Scout-17B-16E
    #   --prompt          你好，请介绍一下人工智能
    #   --max_new_tokens  200
    #   --temperature     0.8
    #   --top_p           0.9
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM


# ---------------------------------------------------------------------------
# Checkpoint helpers（供 train.py 和本脚本共用）
# ---------------------------------------------------------------------------

def _ckpt_step(path: Path) -> int:
    """从 ckpt_step00012345.pt 文件名中解析出步数整数，解析失败返回 -1"""
    try:
        return int(path.stem.replace("ckpt_step", ""))
    except ValueError:
        return -1


def _find_latest_checkpoint(model_dir: str) -> Optional[str]:
    """在 model_dir 中找步数最大的 checkpoint，返回路径字符串，没有则返回 None"""
    p = Path(model_dir)
    if not p.exists():
        return None
    ckpts = sorted(p.glob("ckpt_step*.pt"), key=_ckpt_step)
    return str(ckpts[-1]) if ckpts else None


def load_model(
    config_path: str,
    model_dir: str,
    device: torch.device,
) -> Llama4ForCausalLM:
    """
    从 config 文件 + 最新 checkpoint 加载模型，返回 eval 模式的 Llama4ForCausalLM。

    Args:
        config_path: config_mini.json 路径
        model_dir: checkpoint 目录（存放 ckpt_step*.pt）
        device: 目标设备
    Returns:
        加载好权重的 Llama4ForCausalLM，已设置为 fp16 + eval() 模式
    """
    with open(config_path) as f:
        cfg = json.load(f)

    text_config = Llama4TextConfig(**cfg)
    model = Llama4ForCausalLM(text_config)
    model = model.to(device=device, dtype=torch.float16)

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
    """
    从 logits 中采样下一个 token 的 id。

    Args:
        logits: (vocab_size,) 或 (1, vocab_size)
        temperature: 采样温度，越低越确定性；<=0 则贪心
        top_p: nucleus sampling 阈值
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
    # 去掉累积概率超过 top_p 的 token（保留第一个超过的，确保至少 1 个 token）
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
    model: Llama4ForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> str:
    """
    给定 prompt，自回归生成文本。

    使用 Llama4ForCausalLM 继承的 GenerationMixin.generate() 接口。

    Args:
        model: 已加载的 Llama4ForCausalLM，应处于 eval() 模式
        tokenizer: HuggingFace tokenizer
        prompt: 输入文本字符串
        max_new_tokens: 最多生成的新 token 数
        temperature: 采样温度（<=0 为贪心）
        top_p: nucleus sampling 阈值
        device: 目标设备；None 则自动检测
    Returns:
        生成的文本字符串（不含 prompt）
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    # 编码 prompt
    encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoding["input_ids"].to(device)

    max_position_embeddings = model.config.max_position_embeddings
    # 截断过长的 prompt
    if input_ids.shape[1] >= max_position_embeddings:
        input_ids = input_ids[:, -(max_position_embeddings - 1):]

    # 构建生成参数
    # do_sample=True 时启用随机采样（temperature/top_p 生效）
    # do_sample=False 时退化为贪心搜索（temperature<=0 时）
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    # GenerationMixin.generate() 内部自动管理 KV cache，逐 token 自回归生成
    output_ids = model.generate(input_ids, **gen_kwargs)

    # output_ids 包含 prompt + 生成内容，只取新生成的部分
    prompt_len = input_ids.shape[1]
    new_ids = output_ids[0, prompt_len:]
    result = tokenizer.decode(new_ids, skip_special_tokens=True)

    if was_training:
        model.train()

    return result


@torch.no_grad()
def generate_chat(
    model: Llama4ForCausalLM,
    tokenizer,
    messages: list,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> str:
    """
    基于 chat messages 格式进行对话推理（SFT 模型使用）。

    Args:
        model: 已加载的 Llama4ForCausalLM，应处于 eval() 模式
        tokenizer: HuggingFace tokenizer
        messages: 对话列表，格式如 [{"role": "user", "content": "..."}]
        max_new_tokens: 最多生成的新 token 数
        temperature: 采样温度（<=0 为贪心）
        top_p: nucleus sampling 阈值
        device: 目标设备；None 则自动检测
    Returns:
        assistant 回复的文本字符串
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
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Llama4-Mini 文本生成")
    parser.add_argument("--model_dir", type=str, default="./checkpoints",
                        help="checkpoint 目录（含 ckpt_step*.pt）")
    parser.add_argument("--config", type=str, default=None,
                        help="config JSON 路径（默认：脚本同级目录下的 config_mini.json）")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-4-Scout-17B-16E",
                        help="HuggingFace tokenizer 名称或本地路径")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下人工智能",
                        help="生成提示词")
    parser.add_argument("--chat", action="store_true",
                        help="使用 chat 模式（SFT 模型），将 prompt 作为 user 消息")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] device={device}")

    # 查找 config 文件
    config_path = args.config
    if config_path is None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_mini.json")
        if os.path.exists(candidate):
            config_path = candidate
        else:
            raise FileNotFoundError("找不到 config_mini.json，请用 --config 指定路径")

    print(f"[inference] config={config_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
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
