import torch
from transformers import AutoTokenizer
import os
from train_gpt2_chinese import GPTModel, GPT_CONFIG_124M  # 直接从训练代码导入！
import generate_text as Gtext
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 完全禁用 TF 日志
# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/ubuntu/work/data/llm-data/pretrained_model/gpt2/124M_zh/gpt2_pretrain_test.pth"

# ===================== 加载模型 =====================
def load_model(model_path):
    model = GPTModel(GPT_CONFIG_124M)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"✅ 模型加载成功：{os.path.basename(model_path)}, 参数量:{sum(p.numel() for p in model.parameters())}")
    return model

# ===================== 主程序 =====================
if __name__ == "__main__":
    start_list = [
        "我是一个中文语言模型",
        "今天天气不错"
    ]
    start_context = "我是一个中文语言模型"
    tokenizer = AutoTokenizer.from_pretrained('/home/ubuntu/work/data/llm-data/pretrained_model/llama2/tokenizer/')
    model = load_model(MODEL_PATH)
    for i in range(len(start_list)):
        Gtext.generate_and_print_sample(model=model, tokenizer=tokenizer, device=DEVICE, start_context=start_list[i], temperature=1.0, top_k=None, max_new_tokens=50, index=i)

    