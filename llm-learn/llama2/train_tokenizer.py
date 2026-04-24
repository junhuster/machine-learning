import random
import json
import os
import gc
import sys 
from tqdm import tqdm
import logging as log
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent/'util'))
import logger
logger.init_logger("/home/ubuntu/work/logs/llama2-token-train.log")
os.environ["RAYON_NUM_THREADS"] = "5"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator

random.seed(42)
global_id = 0

data_path = "/home/ubuntu/work/data/llm-data/train_data/zh/monkey/pretrain_data/monkey_pretrain_310M.jsonl"
save_dir = "/home/ubuntu/work/data/llm-data/pretrained_model/llama2/tokenizer"

def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """读取JSONL文件并安全提取文本数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        global global_id
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data['text']
                global_id += 1
                if global_id % 100 == 0:
                    gc.collect()
                    log.info(f"glob_id:{global_id} triggler gc")
            except json.JSONDecodeError:
                log.info(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                log.info(e)
                continue

def create_tokenizer_config(save_dir: str) -> None:
    """创建完整的tokenizer配置文件"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 创建special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    import random
    """训练并保存自定义tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()  # 添加文本规范化
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 配置特殊token
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|im_start|>", 
        "<|im_end|>"
    ]

    # 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=6,  # 提高低频词过滤
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # ===== 采样训练 =====
    sample_ratio = 0.9  # 20% 数据足够
    #random.seed(42)

    log.info(f"Training with {sample_ratio*100}% sampled data")

    # 先统计总行数用于进度条
    log.info("Counting lines...")
    with open(data_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in tqdm(f, desc="Counting"))
    sample_lines = int(total_lines * sample_ratio)
    # 采样生成器
    def sampled_texts():
        global global_id
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if random.random() < sample_ratio:
                    try:
                        yield json.loads(line)['text']
                        global_id += 1
                        #if global_id % 1000 == 0:
                        #    log.info(f"glob_id:{global_id}")
                    except (json.JSONDecodeError, KeyError):
                        continue


    # 训练tokenizer
    log.info(f"Training tokenizer with data from {data_path}")
    tokenizer.train_from_iterator(sampled_texts(), trainer=trainer, length=sample_lines)
    #tokenizer.train([data_path], trainer=trainer)

    # 验证特殊token映射
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        log.info("Special tokens mapping error:", e)
        raise

    # 保存tokenizer文件
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    # 创建配置文件
    create_tokenizer_config(save_dir)
    log.info(f"Tokenizer saved to {save_dir}")

def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        log.info(f"Error loading tokenizer: {e}")
        return
    # 测试基本属性
    log.info("\n=== Tokenizer基本信息 ===")
    log.info(f"Vocab size: {len(tokenizer)}")
    log.info(f"Special tokens: {tokenizer.all_special_tokens}")
    log.info(f"Special token IDs: {tokenizer.all_special_ids}")
    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    
    log.info("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        # add_generation_prompt=True
    )
    log.info(f"Generated prompt:\n {prompt}")

    # 测试编码解码
    log.info("\n=== 编码解码测试 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    log.info(f"Decoded text matches original:{ decoded == prompt}")

    # 测试特殊token处理
    log.info("\n=== 特殊token处理 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    log.info(f"Original: {test_text}")
    log.info(f"Decoded:  {decoded}")
    log.info(f"Special tokens preserved:{decoded == test_text}")

def main():
    log.info(f"begin to train token")
    # 训练tokenizer
    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=10000
    )
    log.info(f"end to train token and begin eval token")
    # 评估tokenizer
    eval_tokenizer(save_dir)
    log.info(f"env eval token")

if __name__ == '__main__':
    #main()
    eval_tokenizer(save_dir)
