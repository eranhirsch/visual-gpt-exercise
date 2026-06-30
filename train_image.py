import dataclasses
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
model.to(device)

# training knobs (a few hundred iters is enough to see the naive batcher fail)
max_iters = 300
eval_interval = 50
eval_iters = 50
learning_rate = 1e-3

optimizer = model.configure_optimizers(
    weight_decay=1e-1,
    learning_rate=learning_rate,
    betas=(0.9, 0.99),
    device_type="cuda" if device == "cuda" else "cpu",
)


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
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )

    return x.to(device), y.to(device)


def get_better_batch(split):
    # The fix: for an image, position == spatial location. Position 0 is the
    # top-left pixel, ALWAYS. So every example must be one whole image aligned
    # to position 0 -- never a random offset into the stream.
    data = train_data if split == "train" else val_data

    # one row per image; sample whole rows (whole images)
    images = data.reshape(-1, seq_len)  # (num_images, seq_len)
    ix = torch.randint(images.shape[0], (batch_size,))
    batch = torch.from_numpy(images[ix.numpy()].astype(np.int64))  # (batch, seq_len)

    # within an image: predict pixel t+1 from pixels 0..t.
    # slicing makes these views non-contiguous, and model.py does targets.view(-1)
    # -> .contiguous() here, or the view() blows up (the bug is in the data layer).
    x = batch[:, :-1].contiguous()  # pixels 0 .. seq_len-2
    y = batch[:, 1:].contiguous()  # pixels 1 .. seq_len-1 (the targets)

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    # average loss over a few random batches, for train and val
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_better_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


if __name__ == "__main__":
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"iter {it:4d} | train {losses['train']:.4f} | val {losses['val']:.4f}"
            )

        x, y = get_better_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # save a checkpoint sample_image.py can load
    ckpt = {
        "model": model.state_dict(),
        "config": dataclasses.asdict(model.config),
        "meta": meta,
    }
    torch.save(ckpt, os.path.join(data_dir, "ckpt_image.pt"))
    print(f"saved checkpoint -> {os.path.join(data_dir, 'ckpt_image.pt')}")
