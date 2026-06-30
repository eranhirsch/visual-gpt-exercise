import os
import pickle

import numpy as np
import torch

from model import (
    GPT,
    GPTConfig,
)

# meta.pkl gives vocab_size and seq_len
data_dir = os.path.join("data", "mnist")
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
seq_len = meta["seq_len"]

block_size = seq_len - 1  # predict pixel t+1 from pixels 0..t
model = GPT(
    GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
    )
)
# ... optimizer + training loop: same shape as your char-GPT ...


def get_batch(split):
    # TODO: return x, y  (each shape (batch_size, block_size), dtype long)
    ...
