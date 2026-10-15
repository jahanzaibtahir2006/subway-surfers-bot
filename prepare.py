"""
Build dataset.npz from the recorded sessions in data/session_*.npz.

Each session file (made by record.py) contains:
    x : (N, 128, 128) uint8 grayscale frames
    y : (N,)          labels, 0=none 1=left 2=right 3=up 4=down

Menu / result / loading screens barely move, so frames are marked as "gameplay"
only where the smoothed frame-to-frame change is above MOTION_THRESH.
Nothing is deleted: the mask is stored in the output and train2.py uses it.

Output:
    dataset.npz      x, y, keep (gameplay mask), file_id (source session of each frame)
    filter_check.png random KEPT (top row) and DROPPED (bottom row) frames to check the filter
"""
import glob
import os

import cv2
import numpy as np

# ============================== CONFIG ==============================
MOTION_THRESH = 3.0       # mean pixel change below this = menu/result screen
SMOOTH = 9                # frames used to smooth the motion signal
SKIP = set()              # session file names to ignore, e.g. {"session_123.npz"} for a bad recording
DATA_DIR = "data"
OUT_FILE = "dataset.npz"
# =====================================================================

here = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(here, DATA_DIR, "session_*.npz")))
files = [f for f in files if os.path.basename(f) not in SKIP]
if not files:
    raise SystemExit(f"No session files found in the '{DATA_DIR}' folder.")

X, Y, K, F, M = [], [], [], [], []
for i, f in enumerate(files):
    d = np.load(f)
    x, y = d["x"], d["y"]

    # motion = mean absolute difference between consecutive frames, then smoothed
    diff = np.abs(np.diff(x.astype(np.int16), axis=0)).mean(axis=(1, 2))
    diff = np.concatenate([[diff[0]], diff])
    pad = SMOOTH // 2
    sm = np.convolve(np.pad(diff, pad, mode="edge"), np.ones(SMOOTH) / SMOOTH, mode="valid")
    keep = sm > MOTION_THRESH

    print(os.path.basename(f), "| frames:", len(x), "| gameplay:", int(keep.sum()))
    X.append(x)
    Y.append(y)
    K.append(keep)
    F.append(np.full(len(x), i))
    M.append(sm)

X = np.concatenate(X)
Y = np.concatenate(Y)
K = np.concatenate(K)
F = np.concatenate(F)
M = np.concatenate(M)

print("\nMotion percentiles (10/25/50/75/90):", np.percentile(M, [10, 25, 50, 75, 90]).round(1))
print("Total frames:", len(X), "| kept:", int(K.sum()), "| dropped:", int((~K).sum()))
print("Kept frames per class (none/left/right/up/down):", np.bincount(Y[K], minlength=5))

np.savez_compressed(os.path.join(here, OUT_FILE), x=X, y=Y, keep=K, file_id=F)

# Visual check of the filter: top row = kept frames, bottom row = dropped frames
rng = np.random.default_rng(0)


def pick(mask, n=6):
    """Concatenate n random frames (or black tiles if there are fewer) side by side."""
    idx = np.where(mask)[0]
    imgs = []
    if len(idx):
        imgs = [X[i] for i in rng.choice(idx, size=min(n, len(idx)), replace=False)]
    while len(imgs) < n:
        imgs.append(np.zeros((128, 128), np.uint8))
    return np.hstack(imgs)


sheet = cv2.cvtColor(np.vstack([pick(K), pick(~K)]), cv2.COLOR_GRAY2BGR)
cv2.putText(sheet, "KEPT", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.putText(sheet, "DROPPED", (5, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
sheet = cv2.resize(sheet, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(here, "filter_check.png"), sheet)
print(f"\nSaved filter_check.png and {OUT_FILE}")