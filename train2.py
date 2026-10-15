"""
Train the Subway Surfers model (behavior cloning).

Input : dataset.npz made by prepare.py (grayscale frames, key labels, per-file ids).
Output: model2.pt (the epoch with the best average per-class recall).

Pipeline:
  1. Label widening - the EXTRA frames before every key press get the same label,
     so the model learns to act slightly ahead of the human.
  2. Trimming       - drop the first/last frames of every gameplay segment
     (start animation, crash just before game over).
  3. Split          - random 150-frame chunks of every file go to validation;
     training frames right next to validation chunks are excluded (no leakage).
  4. Balancing      - every epoch samples the same number of frames for each key
     and only a few "no key" frames (about 95% of the data is "no key").
  5. Augmentation   - horizontal flip (left/right labels swapped), brightness, small shift.
  6. Metric         - per-class recall (plain accuracy is useless with this imbalance).
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

# ============================== CONFIG ==============================
OFFS = [0]            # frame offsets used as input ([0] = single frame); must match the bot
EPOCHS = 6
BATCH = 32
NOOP_MULT = 2.0       # "no key" samples per epoch = NOOP_MULT * (samples per key class)
DROP_END = 22         # frames dropped at the end of each gameplay segment
DROP_START = 38       # frames dropped at the start of each gameplay segment
CHUNK = 150           # validation split granularity (frames)
VAL_FRACTION = 0.2    # share of chunks used for validation
EXTRA = 4             # frames before a key press that receive the same label
SMALL = 400           # files shorter than this are neither trimmed nor used for validation
DATASET_FILE = "dataset.npz"
MODEL_FILE = "model2.pt"
NAMES = ["none", "left", "right", "up", "down"]
# =====================================================================

here = os.path.dirname(os.path.abspath(__file__))

# ------------------------------ device ------------------------------
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    try:  # make sure the GPU really works, not just that it exists
        nn.Conv2d(1, 1, 3).to("cuda")(torch.zeros(1, 1, 8, 8, device="cuda"))
        print("Device: GPU ->", torch.cuda.get_device_name(0))
    except Exception as e:
        print("GPU not usable, falling back to CPU. Error:", e)
        dev = "cpu"
else:
    print("Device: CPU (no GPU found)")

# ------------------------------ data ------------------------------
d = np.load(os.path.join(here, DATASET_FILE))
x, y, keep, fid = d["x"], d["y"].astype(np.int64), d["keep"], d["file_id"]
n = len(x)

# 1. Label widening: copy each key label onto the EXTRA frames before it
y0 = y.copy()
for dd in range(1, EXTRA + 1):
    src = np.where(y0 > 0)[0]
    tgt = src - dd
    ok = tgt >= 0
    src, tgt = src[ok], tgt[ok]
    ok = (fid[tgt] == fid[src]) & (y[tgt] == 0)    # same file, and not already labeled
    y[tgt[ok]] = y0[src[ok]]
print("Class counts after label widening (none/left/right/up/down):",
      np.bincount(y[keep], minlength=5))

# 2 + 3. Trim segment edges and split into train / validation per file
pos = np.zeros(n, int)               # position of the frame inside its file
drop = np.zeros(n, bool)             # trimmed frames
is_val = np.zeros(n, bool)           # validation frames
near_val = np.zeros(n, bool)         # frames close to a validation frame
rng = np.random.default_rng(1)
for f in np.unique(fid):
    idx = np.where(fid == f)[0]
    m = len(idx)
    pos[idx] = np.arange(m)
    k = keep[idx]

    dr = np.zeros(m, bool)
    ends = np.where(k & ~np.append(k[1:], False))[0]        # last frame of each segment
    for e in ends:
        dr[max(0, e - DROP_END): e + 1] = True
    starts = np.where(k & ~np.insert(k[:-1], 0, False))[0]  # first frame of each segment
    for s in starts:
        dr[s: s + DROP_START] = True

    n_chunks = (m + CHUNK - 1) // CHUNK
    val_chunks = rng.choice(n_chunks, size=max(1, int(round(n_chunks * VAL_FRACTION))),
                            replace=False)
    v = np.isin(np.arange(m) // CHUNK, val_chunks)
    if m < SMALL:
        dr[:] = False
        v[:] = False
    drop[idx] = dr
    is_val[idx] = v
    near_val[idx] = np.convolve(v.astype(float), np.ones(17), "same") > 0

valid = keep & ~drop & (pos >= max(OFFS))
tr = np.where(valid & ~near_val)[0]
va = np.where(valid & is_val)[0]
print("train samples:", len(tr), np.bincount(y[tr], minlength=5))
print("val   samples:", len(va), np.bincount(y[va], minlength=5))

# 4. Balanced sampling: indices per class
cls = [tr[y[tr] == c] for c in range(5)]
for c in range(5):
    if len(cls[c]) == 0:
        raise SystemExit(f"No training data for class '{NAMES[c]}'.")
per = int(np.mean([len(cls[c]) for c in range(1, 5)]))
n_noop = min(int(NOOP_MULT * per), len(cls[0]))
print(f"Per epoch: {per} samples for each key, {n_noop} for 'none'")
rs = np.random.default_rng(2)


def make_epoch():
    """One shuffled epoch of sample indices with balanced classes."""
    parts = [rs.choice(cls[c], size=per, replace=len(cls[c]) < per) for c in range(1, 5)]
    parts.append(rs.choice(cls[0], size=n_noop, replace=False))
    e = np.concatenate(parts)
    rs.shuffle(e)
    return e


# The full dataset stays on the CPU; only the current batch is moved to the GPU
# (keeping everything on a 4 GB GPU caused out-of-memory errors).
X = torch.from_numpy(x)
Y = torch.from_numpy(y).to(dev)
offs = torch.tensor(OFFS)


def get_batch(idx, aug=False):
    """Build a batch (frames scaled to 0..1, labels). Optionally augment."""
    idx = torch.as_tensor(idx)
    xb = X[idx[:, None] - offs[None, :]].to(dev).float() / 255.0
    yb = Y[idx.to(dev)].clone()
    if aug:
        # 5. Augmentation: mirror half of the batch and swap left/right labels
        flip = torch.rand(len(idx), device=dev) < 0.5
        was_left = flip & (yb == 1)
        was_right = flip & (yb == 2)
        xb[flip] = xb[flip].flip(-1)
        yb[was_left] = 2
        yb[was_right] = 1
        # random brightness
        xb = (xb * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, 1, device=dev))).clamp(0, 1)
        # small random shift (same for the whole batch)
        dy, dx = np.random.randint(-4, 5, size=2)
        xb = torch.roll(xb, shifts=(int(dy), int(dx)), dims=(2, 3))
    return xb, yb


# ------------------------------ model ------------------------------
class Net(nn.Module):
    """ResNet18 on a single grayscale frame -> 5 classes (none/left/right/up/down)."""

    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet18(weights=w)   # ImageNet weights (downloaded once)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 5))
        self.m = m

    def forward(self, x):
        x = (x - 0.45) / 0.225
        return self.m(x.expand(-1, 3, -1, -1))       # gray -> 3 channels


# ------------------------------ training ------------------------------
if __name__ == "__main__":
    model = Net().to(dev)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    def evaluate():
        """Per-class recall and confusion matrix on the validation frames."""
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(va), 128):
                xb, _ = get_batch(va[i:i + 128])
                preds.append(model(xb).argmax(1).cpu().numpy())
        model.train()
        p = np.concatenate(preds)
        t = y[va]
        conf = np.zeros((5, 5), int)
        np.add.at(conf, (t, p), 1)
        rec = np.array([conf[c, c] / conf[c].sum() if conf[c].sum() else np.nan
                        for c in range(5)])
        return rec, conf

    best, best_conf = -1, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        perm = make_epoch()
        total, n_batches = 0, 0
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            xb, yb = get_batch(perm[i:i + BATCH], aug=True)
            loss = lossf(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        rec, conf = evaluate()
        score = np.nanmean(rec)
        print(f"epoch {ep:2d} | loss {total / n_batches:.3f} | "
              f"recall none/left/right/up/down:", np.round(rec, 2), f"| avg {score:.2f}")
        if score > best:                      # keep the best epoch
            best, best_conf = score, conf
            torch.save({"model": model.state_dict(), "arch": "resnet18"},
                       os.path.join(here, MODEL_FILE))

    print(f"\nTraining done ({time.time() - t0:.0f} s). Best average recall: {best:.2f}")
    print("\nConfusion table (row = true action, column = model prediction)")
    print("          none left right   up down")
    for c in range(5):
        print(f"{NAMES[c]:>6}", best_conf[c])
    print(f"{MODEL_FILE} saved.")