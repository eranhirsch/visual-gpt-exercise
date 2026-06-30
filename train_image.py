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

# the whole token stream, one flat 1-D array of ids (uint16, written by prepare.py)
train_data = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
val_data = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)

batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    # The obvious (text-style) batcher: pick random offsets into the stream,
    # take block_size tokens as x and the same window shifted by one as y.
    data = train_data if split == "train" else val_data

    # batch_size random start offsets; -block_size keeps x and the +1 y in bounds
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # x[:, t] -> token at offset; y[:, t] -> the next token (the training target)
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )

    return x.to(device), y.to(device)
