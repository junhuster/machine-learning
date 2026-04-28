import sys 
import os
import tiktoken
import torch
import logging as log
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
sys.path.append(str(Path(__file__).parent.parent/'util'))
import logger
logger.init_logger("/home/ubuntu/work/logs/gpt2-train.log")
import gpt_2 as GPT2
import argparse

start_text = "Every effort moves you"
start_text1 = "I promised to"
train_file_path = "/home/ubuntu/work/data/llm-data/train_data/merged_output.txt"
load_local_model = True
local_model_path = "/home/ubuntu/work/data/llm-data/local_model/gpt2/model.pth"
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": True       # Query-Key-Value bias
}

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

if __name__ == "__main__":
    GPT2.clear_memory()
    parser = argparse.ArgumentParser()
    parser.add_argument('-ft',
                        '--fine_tune',
                        type=int,
                        required=False,
                        default=0,
                        help='train from stratch or from a foundatiton model')

    parser.add_argument('-mpath',
                        '--model_path',
                        type=str,
                        required=False,
                        default='/home/ubuntu/work/data/llm-data/pretrained_model/gpt2/124M_torch/model.pth',
                        help='')
    
    args = parser.parse_args()

    file_path = train_file_path
    #url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    tokenizer = tiktoken.get_encoding("gpt2")
    #torch.manual_seed(123)

    train_loader = GPT2.create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = GPT2.create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    if args.fine_tune == 1:
        print(f"\n train from a foundatiton model")
        print(f"\n foundatiton model path:{args.model_path}\n")
        checkpoint = torch.load(args.model_path)
        model = GPT2.GPTModel(GPT_CONFIG_124M)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"\n train from scratch\n")
        model = GPT2.GPTModel(GPT_CONFIG_124M)
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, tokens_seen = GPT2.train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context= start_text, tokenizer=tokenizer
    )

    GPT2.generate_and_print_sample(model, tokenizer, device, start_text)
    GPT2.generate_and_print_sample(model, tokenizer, device, start_text1)

    #epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    #vi plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    GPT2.clear_memory()

