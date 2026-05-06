import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型参数（小模型，T4 16G可跑）
VOCAB_SIZE = 65024
DIM = 512
N_HEADS = 8
N_LAYERS = 8
HIDDEN_DIM = 1536
MAX_SEQ_LEN = 512
USE_MOE = True  # 打开MOE学习，显存仍足够

# 训练
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LR = 2e-5
EPOCHS = 2
FP16 = True

# 日志 & 保存
LOG_STEP = 10
SAVE_STEP = 200
MAX_SAVE_CKPT = 3

# 路径
PRETRAIN_DATA = "pretrain_data.txt"
SFT_DATA = "sft_data.json"
PRETRAIN_SAVE_DIR = "ckpt_pretrain"
SFT_SAVE_DIR = "ckpt_sft"