import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modeling_deepseek import DeepSeek
from tokenizer import load_deepseek_tokenizer
from utils import save_model, get_latest_checkpoint, PretrainDataset, StepTimer
from config import *

def main():
    tokenizer = load_deepseek_tokenizer()
    model = DeepSeek(
        vocab_size=VOCAB_SIZE, dim=DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
        hidden_dim=HIDDEN_DIM, max_seq_len=MAX_SEQ_LEN, use_moe=USE_MOE
    ).to(DEVICE)

    # 断点续训
    ckpt, start_step = get_latest_checkpoint(PRETRAIN_SAVE_DIR)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"✅ 加载模型：{ckpt}，从 step {start_step} 开始")

    # 数据
    ds = PretrainDataset(PRETRAIN_DATA, tokenizer, MAX_SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=FP16)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    timer = StepTimer()
    timer.reset()

    total_steps = len(loader) * EPOCHS
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in loader:
            if global_step < start_step:
                global_step += 1
                continue

            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)

            with torch.cuda.amp.autocast(enabled=FP16):
                logits = model(ids, mask)
                loss = criterion(
                    logits[:, :-1].reshape(-1, VOCAB_SIZE),
                    ids[:, 1:].reshape(-1)
                )

            scaler.scale(loss).backward()
            if (global_step + 1) % GRADIENT_ACCUMULATION == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1

            if global_step % LOG_STEP == 0:
                timer.log(global_step, total_steps, loss.item(), epoch+1, EPOCHS)

            if global_step % SAVE_STEP == 0:
                save_model(model, global_step, PRETRAIN_SAVE_DIR, MAX_SAVE_CKPT)

if __name__ == "__main__":
    main()