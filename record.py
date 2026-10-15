import ctypes, os, time
ctypes.windll.shcore.SetProcessDpiAwareness(2)

import cv2, mss, numpy as np, pygetwindow as gw
from pynput import keyboard

FPS = 15
SIZE = 128
LEAD = 0.15
SAVE_EVERY = 180   # har 3 minute baad khud save
ACTIONS = {keyboard.Key.left: 1, keyboard.Key.right: 2,
           keyboard.Key.up: 3, keyboard.Key.down: 4}   # 0 = kuch nahi

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

wins = [x for x in gw.getAllWindows()
        if x.title.startswith("BlueStacks") and x.visible and x.width > 200 and x.height > 200]
if not wins:
    raise SystemExit("BlueStacks window nahi mili. Pehle game kholo.")
ov = [x for x in wins if "Overlay" in x.title]
w = ov[0] if ov else wins[0]
print("Window:", w.title)
region = {"left": w.left, "top": w.top, "width": w.width, "height": w.height}
print("Capture region:", region)

state = {"quit": False}
events = []
counts = np.zeros(5, dtype=int)

def on_press(key):
    if key == keyboard.Key.esc:
        state["quit"] = True
    elif key in ACTIONS:
        events.append((time.time(), ACTIONS[key]))
        counts[ACTIONS[key]] += 1

keyboard.Listener(on_press=on_press).start()

frames, times = [], []

def save_chunk():
    if len(frames) < 30:
        return
    t = np.array(times)
    labels = np.zeros(len(frames), dtype=np.int8)
    for ts, a in events:
        labels[(t >= ts - LEAD) & (t <= ts)] = a
    os.makedirs(OUT_DIR, exist_ok=True)
    name = os.path.join(OUT_DIR, f"session_{int(time.time())}.npz")
    np.savez_compressed(name, x=np.array(frames, dtype=np.uint8), y=labels)
    print("SAVED:", name, "| frames:", len(frames),
          "| none/left/right/up/down:", np.bincount(labels, minlength=5))
    frames.clear(); times.clear(); events.clear()

print("Recording SHURU. Games khelo. Khatam karne ke liye Esc dabao.")
dt = 1 / FPS
with mss.MSS() as sct:
    nxt = time.time()
    last_print = time.time()
    last_save = time.time()
    while not state["quit"]:
        img = np.array(sct.grab(region))
        g = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        frames.append(g)
        times.append(time.time())
        if time.time() - last_print > 10:
            print("frames:", len(frames), "| keys (left/right/up/down):", counts[1:])
            last_print = time.time()
        if time.time() - last_save > SAVE_EVERY:
            save_chunk()
            last_save = time.time()
        nxt += dt
        time.sleep(max(0, nxt - time.time()))

save_chunk()
print("Khatam. Data folder mein save ho gaya.")