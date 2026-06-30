import os

import torch

from model import GPT, GPTConfig

data_dir = os.path.join("data", "mnist")
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(os.path.join(data_dir, "ckpt_image.pt"), weights_only=True)
meta = ckpt["meta"]
H, W = meta["img_h"], meta["img_w"]
seq_len = meta["seq_len"]

model = GPT(GPTConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

temperature = 0.8
n_samples = 4

for _ in range(n_samples):
    # seed with the top-left pixel: in MNIST that corner is virtually always background (0)
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    # we already have pixel 0, so ask for the remaining seq_len - 1 pixels
    out = model.generate(idx, max_new_tokens=seq_len - 1, temperature=temperature)
    img = out[0].view(H, W).tolist()

    print("\n".join("".join("#" if p else " " for p in row) for row in img))
    print("-" * W)
