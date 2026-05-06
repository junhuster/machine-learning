import torch
from modeling_deepseek import DeepSeek
from tokenizer import load_deepseek_tokenizer
from utils import get_latest_checkpoint
from config import *

def generate():
    tokenizer = load_deepseek_tokenizer()
    model = DeepSeek(
        vocab_size=VOCAB_SIZE, dim=DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
        hidden_dim=HIDDEN_DIM, max_seq_len=MAX_SEQ_LEN, use_moe=USE_MOE
    ).to(DEVICE)

    ckpt, _ = get_latest_checkpoint(PRETRAIN_SAVE_DIR)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    prompt = input("输入：")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        for _ in range(200):
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            next_token = logits.argmax(-1)[:, -1:]
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(next_token)], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    print(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))

if __name__ == "__main__":
    generate()