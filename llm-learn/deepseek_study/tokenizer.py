from transformers import AutoTokenizer

def load_deepseek_tokenizer():
    # 直接使用DeepSeek官方开源分词器（本地学习用）
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer