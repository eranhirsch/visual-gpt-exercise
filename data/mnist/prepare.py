"""
Prepare MNIST for an image-GPT.

The idea: an image is just a sequence of tokens. We (optionally) shrink each
digit, flatten it row-major, binarize every pixel to {0, 1}, and write one long stream
of tokens -- exactly like nanoGPT's char datasets, except the "characters" are
black/white pixels and the vocabulary has size 2.

Why downscale? Self-attention is O(seq_len^2) and autoregressive sampling is ~O(seq_len^3),
so image resolution is the main lever on CPU cost. 28x28 = 784 tokens is slow to
sample on CPU; 14x14 = 196 tokens (the default, --downscale=2) is much cheaper, and MNIST
digits stay clearly legible. Use --downscale=1 for full resolution.

Outputs (next to this file): train.bin, val.bin (uint16 token streams, nanoGPT memmap
format) and meta.pkl (vocab_size + image geometry, for model init and decoding).
"""

import argparse
import gzip
import os
import pickle

import numpy as np
import requests

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "raw")
os.makedirs(RAW, exist_ok=True)

p = argparse.ArgumentParser()
p.add_argument(
    "--downscale", type=int, default=2, help="average-pool factor; 1 = full 28x28"
)
p.add_argument(
    "--threshold", type=int, default=128, help="pixel >= threshold -> 1 (ink)"
)
args = p.parse_args()

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"  # torchvision's mirror, no auth
IMAGES = {"train": "train-images-idx3-ubyte.gz", "val": "t10k-images-idx3-ubyte.gz"}


def read_images(fname):
    path = os.path.join(RAW, fname)
    if not os.path.exists(path):
        print(f"downloading {fname}")
        r = requests.get(BASE + fname, timeout=60)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic, n, rows, cols = np.frombuffer(buf, dtype=">u4", count=4)  # 16-byte header
    assert magic == 2051, f"bad image magic {magic}"
    return np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(n, rows, cols)


def downscale(imgs, f):
    if f == 1:
        return imgs
    n, H, W = imgs.shape
    assert H % f == 0 and W % f == 0, f"{H}x{W} not divisible by {f}"
    return imgs.reshape(n, H // f, f, W // f, f).mean(
        axis=(2, 4)
    )  # average-pool f x f blocks


def to_tokens(imgs):
    return (
        (imgs >= args.threshold).astype(np.uint16).reshape(-1)
    )  # binarize, flatten, concat


for split, fname in IMAGES.items():
    imgs = downscale(read_images(fname), args.downscale)
    _, H, W = imgs.shape
    to_tokens(imgs).tofile(os.path.join(HERE, f"{split}.bin"))
    print(f"{split}: {len(imgs)} images -> {len(imgs) * H * W} tokens ({H}x{W})")

seq_len = H * W
meta = {
    "vocab_size": 2,
    "img_h": int(H),
    "img_w": int(W),
    "seq_len": int(seq_len),
    "downscale": args.downscale,
    "threshold": args.threshold,
}
pickle.dump(meta, open(os.path.join(HERE, "meta.pkl"), "wb"))
print(
    f"vocab_size=2, seq_len={seq_len}, downscale={args.downscale}, threshold={args.threshold}"
)

# round-trip proof: decode the first training image straight from train.bin
art = np.fromfile(
    os.path.join(HERE, "train.bin"), dtype=np.uint16, count=seq_len
).reshape(H, W)
print("\nfirst training digit, decoded from train.bin:")
for row in art:
    print("".join("##" if v else ".." for v in row))
