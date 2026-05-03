import json
import sys
from tqdm import tqdm


def extract_text(input_file: str, output_file: str) -> None:
    """从JSONL文件中提取text字段，写入输出文件"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Processing"):
            try:
                data = json.loads(line)
                text = data.get('text', '')
                f_out.write(text + '\n')
            except json.JSONDecodeError:
                continue
if __name__ == '__main__':
    src_file = "/home/ubuntu/work/data/llm-data/train_data/zh/monkey/pretrain_data/monkey_pretrain_996M.jsonl"
    dst_file = "/home/ubuntu/work/data/llm-data/train_data/zh/monkey/pretrain_data/monkey_pretrain_996M.txt"
    extract_text(
        input_file=src_file,
        output_file=dst_file
    )