import torch
from transformers import AutoTokenizer
import os
from train_gpt2_chinese import GPTModel, GPT_CONFIG_124M  # 直接从训练代码导入！
import generate_text as Gtext
# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./saved_models.pth"

# ===================== 加载模型 =====================
def load_model(model_path):
    model = GPTModel(GPT_CONFIG_124M)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"✅ 模型加载成功：{os.path.basename(model_path)}")
    return model

# ===================== 主程序 =====================
if __name__ == "__main__":
    start_context = "你好"
    tokenizer = AutoTokenizer.from_pretrained('/home/ubuntu/work/data/llm-data/pretrained_model/llama2/tokenizer/')
    model = load_model(model_path)
    Gtext.generate_and_print_sample(model, tokenizer, DEVICE, start_context)

    