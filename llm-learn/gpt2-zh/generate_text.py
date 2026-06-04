import logging as log
import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent/'util'))
import logger
#logger.init_logger("/home/ubuntu/work/logs/gpt2-zh-pre-train.log")

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    """
    Generate text using the model with temperature and top-k sampling.

    Args:
        model: The language model
        idx: Input token indices (batch, n_tokens)
        max_new_tokens: Number of new tokens to generate
        context_size: Maximum context size the model can handle
        temperature: Sampling temperature (higher = more random, lower = more deterministic)
                    - temperature > 1.0: more random/diverse
                    - temperature = 1.0: normal sampling
                    - temperature < 1.0: more conservative/focused
                    - temperature -> 0: approaches greedy (argmax)
        top_k: If set, only sample from the top k most likely tokens. None means no restriction.

    Returns:
        Generated token indices (batch, n_tokens + max_new_tokens)
    """
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            # Get the top k logits and their indices
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            # Create a mask for logits not in top-k
            logits = torch.full_like(logits, float('-inf'))
            # Scatter the top-k logits back to their original positions
            logits.scatter_(-1, top_k_indices, top_k_logits)

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Sample from the probability distribution
        idx_next = torch.multinomial(probas, num_samples=1)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate_and_print_sample(model, tokenizer, device, start_context, temperature=1.0, top_k=None, max_new_tokens=50, index=0):
    """
    Generate and print a sample from the model.

    Args:
        model: The language model
        tokenizer: Tokenizer for encoding/decoding text
        device: Device to run on (cpu/cuda)
        start_context: Initial text prompt
        temperature: Sampling temperature (default: 1.0)
        top_k: Top-k sampling parameter (default: None)
        max_new_tokens: Maximum number of new tokens to generate (default: 50)
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
    log.info(f"output text:{decoded_text}")
    print(f"\ninput text {index}: {start_context}\n")
    print(f"\noutput_text {index}: {decoded_text}\n")
    model.train()

def text_to_token_ids(text, tokenizer):
    """Convert text to token IDs."""
    encoded = tokenizer(text).data['input_ids']
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Convert token IDs back to text."""
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
