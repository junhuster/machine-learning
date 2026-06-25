##TODO 训练词表改用llama2的
"""
inference_v2.py — DeepSeek-Mini推理脚本（增强版）

在 inference.py 基础上新增：
  - top_k 截断
  - repetition_penalty 重复惩罚
  - beam search（默认关闭，use_beam_search=True 启用）

========== 执行过程概览 ==========

【普通采样模式（默认）】

  1. 编码 prompt
       用 tokenizer 把输入文本转成 token id 列表。

  2. Prefill
       把整个 prompt 一次性喂给模型，得到第一个生成位置的 logits。
       同时初始化 KV cache，后续每步只需计算新 token。

  3. 采样第一个 token（_sample）
       a. 重复惩罚：已出现的 token logit 按 repetition_penalty 压低
       b. temperature 缩放：调节分布的"平坦/尖锐"程度
       c. top_k 截断：只保留概率最高的 k 个 token
       d. top_p 截断：按累计概率再截一刀（nucleus sampling）
       e. multinomial 随机采样（temperature=0 时直接 argmax）

  4. 自回归解码循环
       每步把上一步采样到的 token 喂给模型，得到新 logits，
       重复步骤 3，直到遇到 eos 或达到 max_new_tokens。

  5. 解码输出
       把生成的 token id 列表用 tokenizer 还原成文本字符串。

【Beam Search 模式（use_beam_search=True）】

  1. 编码 prompt，同上。

  2. Prefill，得到初始 logits，取 top-num_beams 个 token，
     初始化 num_beams 条候选路径（beam），每条记录：
       (累计 log 概率, 已生成的 token 列表)

  3. 逐步扩展
     每步对每条存活的 beam：
       a. 把 prompt + 该 beam 的已生成序列拼接，重新 forward 一次
          （不使用增量 KV cache，每步重算，实现简洁但速度较慢）
       b. 施加重复惩罚
       c. 取 top-num_beams 个候选 token，各自计算新的累计得分
     从所有 beam × num_beams 个候选中，选得分最高的 num_beams 条继续。
     遇到 eos 的 beam 移入"已完成"列表。

  4. 选最优结果
     所有已完成和存活的 beam，按"累计 log 概率 / 序列长度"归一化排序，
     取得分最高的那条作为最终输出。

========================================

可被训练脚本导入，也可单独运行：

    python inference_v2.py \\
        --model_dir ./checkpoints \\
        --tokenizer deepseek-ai/DeepSeek-V3 \\
        --prompt "你好，请介绍一下人工智能" \\
        --max_new_tokens 200 \\
        --temperature 0.8 \\
        --top_k 50 \\
        --repetition_penalty 1.2

    # 启用 beam search
    python inference_v2.py \\
        --prompt "你好" \\
        --use_beam_search \\
        --num_beams 4
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from model import ModelArgs, Transformer
from inference import _ckpt_step, _find_latest_checkpoint, load_model_with_path


# ---------------------------------------------------------------------------
# Sampling（增强版）
# ---------------------------------------------------------------------------

def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    repetition_penalty: float,
) -> torch.Tensor:
    """
    对已生成过的 token 施加重复惩罚。

    做法：已出现的 token，其 logit 若为正则除以 penalty，若为负则乘以 penalty。
    penalty > 1 时压低已出现 token 的概率，penalty = 1 时无效果。
    """
    if repetition_penalty == 1.0 or not generated_ids:
        return logits

    ids = torch.tensor(generated_ids, dtype=torch.long, device=logits.device)
    score = logits[ids]
    # 正值除以 penalty（降低），负值乘以 penalty（更负，也是降低）
    score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
    logits = logits.clone()
    logits[ids] = score
    return logits


def _apply_top_k(probs: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    保留概率最高的 top_k 个 token，其余置 0。
    top_k <= 0 表示不启用。
    """
    if top_k <= 0 or top_k >= probs.size(-1):
        return probs
    # kth_val：第 top_k 大的概率值
    kth_val = torch.topk(probs, top_k).values[..., -1]
    probs = probs.clone()
    probs[probs < kth_val] = 0.0
    return probs


def _sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    generated_ids: Optional[List[int]] = None,
) -> int:
    """
    从 logits 中采样下一个 token id。

    Args:
        logits: (vocab_size,) 或 (1, vocab_size)
        temperature: 采样温度，<=0 则贪心
        top_p: nucleus sampling 阈值
        top_k: 保留最高概率的 k 个 token，<=0 不启用
        repetition_penalty: 重复惩罚系数，>1 压低已出现 token，1.0 无效果
        generated_ids: 已生成的 token id 列表，用于重复惩罚
    Returns:
        token id（Python int）
    """
    logits = logits.view(-1).float()

    # 重复惩罚（在 temperature 缩放之前施加，作用于原始 logits）
    if generated_ids:
        logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    if temperature <= 0.0:
        return int(logits.argmax().item())

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # top_k 截断
    probs = _apply_top_k(probs, top_k)

    # top_p nucleus 截断
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)

    # 重新归一化
    total = sorted_probs.sum()
    if total <= 0:
        # 极端情况：所有候选都被截掉，回退到 argmax
        return int(logits.argmax().item())
    sorted_probs = sorted_probs / total

    next_idx = torch.multinomial(sorted_probs, num_samples=1)
    return int(sorted_indices[next_idx].item())


# ---------------------------------------------------------------------------
# 核心生成函数（普通自回归）
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> str:
    """
    给定 prompt，自回归生成文本。

    Args:
        model: 已加载的 Transformer，应处于 eval() 模式
        tokenizer: HuggingFace tokenizer
        prompt: 输入文本字符串
        max_new_tokens: 最多生成的新 token 数
        temperature: 采样温度（<=0 为贪心）
        top_p: nucleus sampling 阈值
        top_k: 保留最高概率的 k 个 token，<=0 不启用
        repetition_penalty: 重复惩罚系数（1.0 无效果，推荐 1.1~1.3）
        device: 目标设备；None 则自动检测
    Returns:
        生成的文本字符串（不含 prompt）
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    max_seq_len = model.args.max_seq_len

    if len(input_ids) >= max_seq_len:
        input_ids = input_ids[-(max_seq_len - 1):]

    prompt_len = len(input_ids)
    prompt_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    model.reset_kv_cache()
    logits = model(prompt_tensor, start_pos=0, use_cache=True)  # (1, vocab_size)

    generated_ids: List[int] = []
    eos_id = tokenizer.eos_token_id

    next_token_id = _sample(
        logits[0], temperature=temperature, top_p=top_p, top_k=top_k,
        repetition_penalty=repetition_penalty, generated_ids=generated_ids,
    )
    if next_token_id == eos_id:
        if was_training:
            model.train()
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    generated_ids.append(next_token_id)

    for step in range(1, max_new_tokens):
        cur_pos = prompt_len + step
        if cur_pos >= max_seq_len:
            break

        cur_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        logits = model(cur_token, start_pos=cur_pos - 1, use_cache=True)  # (1, vocab_size)

        next_token_id = _sample(
            logits[0], temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, generated_ids=generated_ids,
        )
        if next_token_id == eos_id:
            break
        generated_ids.append(next_token_id)

    if was_training:
        model.train()

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Beam Search 生成函数
# ---------------------------------------------------------------------------
#
# 基本思路：
#   维护 num_beams 条候选路径（beam），每步对每条 beam 展开 vocab，
#   取全局 top-num_beams 的 (beam_id, token_id) 组合继续。
#
# KV cache 处理：
#   beam search 需要每条 beam 独立的 KV cache 状态。
#   这里采用"每步重算"的简单策略：
#     - 不使用增量 KV cache
#     - 每步把每条 beam 的完整序列重新 forward 一次
#   代价：速度比增量 cache 慢（O(len^2) vs O(len)），但实现简洁，适合学习理解原理。
#
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_beam_search(
    model: Transformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    num_beams: int = 4,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> str:
    """
    用 beam search 生成文本。

    每步保留 num_beams 条得分最高的候选序列，最终返回得分最高的那条。

    Args:
        model: 已加载的 Transformer，应处于 eval() 模式
        tokenizer: HuggingFace tokenizer
        prompt: 输入文本字符串
        max_new_tokens: 最多生成的新 token 数
        num_beams: beam 数量（并行保留的候选路径数）
        repetition_penalty: 重复惩罚系数
        device: 目标设备；None 则自动检测
    Returns:
        得分最高的生成文本字符串（不含 prompt）
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    max_seq_len = model.args.max_seq_len
    if len(input_ids) >= max_seq_len:
        input_ids = input_ids[-(max_seq_len - 1):]

    prompt_len = len(input_ids)
    eos_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Step 1：Prefill，用完整 prompt 跑一次 forward，得到初始 logits
    # ------------------------------------------------------------------
    prompt_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    model.reset_kv_cache()
    # beam search 不使用增量 cache，每步重算，所以 use_cache=False
    logits = model(prompt_tensor, start_pos=0, use_cache=False)  # (1, vocab_size)
    logits = logits.float().squeeze(0)  # (vocab_size,)

    # 对第一步的 logits 施加重复惩罚（此时 generated 为空，无效果，保持一致性）
    logits = _apply_repetition_penalty(logits, [], repetition_penalty)
    log_probs = torch.log_softmax(logits, dim=-1)  # (vocab_size,)

    # 取 top-num_beams 个 token 作为初始 beam
    topk_log_probs, topk_ids = torch.topk(log_probs, num_beams)  # (num_beams,)

    # 每条 beam 用 (累计log_prob, token序列) 表示
    # beams: List of (cumulative_log_prob: float, generated_ids: List[int])
    beams: List[Tuple[float, List[int]]] = [
        (topk_log_probs[i].item(), [topk_ids[i].item()])
        for i in range(num_beams)
    ]

    # 已完成的 beam（遇到 eos 就移入这里）
    completed: List[Tuple[float, List[int]]] = []

    # ------------------------------------------------------------------
    # Step 2：逐步扩展，每步对所有存活 beam 各跑一次 forward
    # ------------------------------------------------------------------
    for step in range(1, max_new_tokens):
        if not beams:
            break

        all_candidates: List[Tuple[float, List[int]]] = []

        for cum_log_prob, gen_ids in beams:
            last_token_id = gen_ids[-1]

            # 检查序列长度上限
            cur_len = prompt_len + len(gen_ids)
            if cur_len >= max_seq_len:
                completed.append((cum_log_prob, gen_ids))
                continue

            # 把 prompt + 已生成序列拼起来，重新 forward
            full_ids = input_ids + gen_ids
            full_tensor = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
            model.reset_kv_cache()
            logits = model(full_tensor, start_pos=0, use_cache=False)  # (1, vocab_size)
            logits = logits.float().squeeze(0)  # (vocab_size,)

            # 对已生成序列施加重复惩罚
            logits = _apply_repetition_penalty(logits, gen_ids, repetition_penalty)
            log_probs = torch.log_softmax(logits, dim=-1)  # (vocab_size,)

            # 展开：取 top-num_beams 个候选 token
            topk_log_probs, topk_ids = torch.topk(log_probs, num_beams)

            for i in range(num_beams):
                token_id = topk_ids[i].item()
                new_cum = cum_log_prob + topk_log_probs[i].item()
                new_gen = gen_ids + [token_id]

                if token_id == eos_id:
                    completed.append((new_cum, gen_ids))  # 不含 eos
                else:
                    all_candidates.append((new_cum, new_gen))

        if not all_candidates:
            break

        # 从所有候选中选出得分最高的 num_beams 条继续
        # 用序列长度做归一化，避免短序列占优
        all_candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
        beams = all_candidates[:num_beams]

    # 把还活着的 beam 也加入候选
    completed.extend(beams)

    if was_training:
        model.train()

    if not completed:
        return ""

    # 按归一化得分取最优
    best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
    return tokenizer.decode(best[1], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 统一入口（根据开关选择普通生成或 beam search）
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_options(
    model: Transformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    use_beam_search: bool = False,   # 默认关闭
    num_beams: int = 4,
    device: Optional[torch.device] = None,
) -> str:
    """
    统一生成入口。

    use_beam_search=False（默认）：走普通自回归采样（支持 temperature/top_p/top_k/repetition_penalty）
    use_beam_search=True：走 beam search（temperature/top_p/top_k 不生效，由 num_beams 控制）
    """
    if use_beam_search:
        return generate_beam_search(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            device=device,
        )
    else:
        return generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            device=device,
        )


@torch.no_grad()
def generate_chat(
    model: Transformer,
    tokenizer,
    messages: list,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    use_beam_search: bool = False,
    num_beams: int = 4,
    device: Optional[torch.device] = None,
) -> str:
    """
    基于 chat messages 格式进行对话推理（SFT 模型使用）。
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate_with_options(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        use_beam_search=use_beam_search,
        num_beams=num_beams,
        device=device,
    )


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
pretrain_prompt_datas = [
    '你好呀',
    "中国的首都是哪里？",
    "刘备和关羽什么关系？",
    "宋徽宗怎么样?",
    "应天门在哪里？",
    "介绍下人工智能"
]
def main():
    parser = argparse.ArgumentParser(description="DeepSeek-Mini文本生成（增强版）")
    parser.add_argument("--model_dir", type=str, default="/home/ubuntu/work/data/llm-data/pretrained_model/deepseek-v3-mini/32G/release/")
    parser.add_argument("--model_path", type=str, default="/home/ubuntu/work/data/llm-data/pretrained_model/deepseek-v3-mini/32G/release/deepseek-v3-mini_400M_sft.pt")    
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--prompt", type=str, default="中国的首都是")
    parser.add_argument("--chat", action="store_true", help="使用 chat 模式")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    # 普通采样参数
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0, help="top_k 截断，0 表示不启用")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="重复惩罚系数，>1 压低已出现 token，推荐 1.1~1.3")
    # beam search 参数
    parser.add_argument("--use_beam_search", action="store_true", default=True,
                        help="启用 beam search（默认关闭）")
    parser.add_argument("--num_beams", type=int, default=4, help="beam 数量")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference_v2] device={device}")

    config_path = args.config
    if config_path is None:
        candidate = os.path.join(args.model_dir, "config_mini.json")
        if os.path.exists(candidate):
            config_path = candidate
        else:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_mini.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到config文件: {config_path}")
    model_file_name = Path(args.model_path).name    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = load_model_with_path(config_path, args.model_path, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.use_beam_search:
        print(f"\nmodel file:{model_file_name} param_num:{num_params / 1e6:.3f} 模式=beam search, num_beams={args.num_beams}")
    else:
        print(f"\nmodel file:{model_file_name} param_num:{num_params / 1e6:.3f} 模式=采样, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, repetition_penalty={args.repetition_penalty}")

    if args.chat:
        for i in range(len(pretrain_prompt_datas)):
            start = time.time()
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
            elaps = time.time() - start
            print(f"\nQA: {pretrain_prompt_datas[i]} => infer_cost: {elaps:.3f} sec\nAI answer: {result}\n")
    else:
        for i in range(len(pretrain_prompt_datas)):
            start = time.time()
            result = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=pretrain_prompt_datas[i],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            elaps = time.time() - start
            print(f"\nQA: {pretrain_prompt_datas[i]} => infer_cost: {elaps:.3f} sec\nAI answer: {result}\n")

    print(result)


if __name__ == "__main__":
    main()
