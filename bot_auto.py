"""
Subway Surfers auto-player (BlueStacks, Windows).

A ResNet18 trained by behavior cloning looks at the screen (grayscale 128x128)
and decides: none / left / right / up / down. Keys are sent with pynput.
When the "Continue?" screen is detected, the bot restarts the game by itself
(reward screens and ads are handled with simple image checks and clicks).

Controls:
    `    (backtick)  toggle the bot ON/OFF manually
    Esc              stop everything (also works while the bot is restarting)
"""
import ctypes
import os
import threading
import time
from collections import deque

ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Windows only: exact pixel capture/clicks

import cv2
import mss
import numpy as np
import pygetwindow as gw
import torch
import torch.nn as nn
import torchvision
from pynput import keyboard, mouse

# ============================== CONFIG ==============================
# --- Model input ---
FPS = 15                  # capture / decision rate
SIZE = 128                # frames are resized to SIZE x SIZE
OFFS = [0]                # frame offsets fed to the model ([0] = current frame only)
MODEL_FILE = "model2.pt"

# --- Decision rules (index: 1=left, 2=right, 3=up, 4=down) ---
THRESH = {1: 0.50, 2: 0.60, 3: 0.50, 4: 0.45}   # min probability to act
DELAY = {1: 0.35, 2: 0.30, 3: 0.40, 4: 0.40}    # seconds between decision and key press
REFRACT = {3: 0.3, 4: 1.0}                      # min seconds before repeating jump / roll
LANE_COOLDOWN = 0.8       # min seconds between two lane changes
LR_SUM = 0.65             # if P(left)+P(right) >= this, move toward the likelier side
NO_DOWN_AFTER_JUMP = 0.8  # do not roll right after a jump
MOTION = 3.0              # below this screen motion the game is treated as paused/menu

# --- Auto restart ---
AUTO = True               # restart automatically after game over
MAX_GAMES = 100           # stop after this many games
CLICK_WAIT = 2.0          # seconds between restart steps
START_WAIT = 3.0          # seconds to wait after PLAY before the bot turns ON
OUT_THR = 0.8             # template score above which "Continue?" screen is detected
PLAY_THR = 0.8            # template score above which the PLAY screen is detected
BLUE_THR = 0.35           # blue pixel fraction above which it is the reward screen
SAVE_GAME_DATA = True     # save frames + actions of every game to GAME_DIR
GAME_DIR = "bot_games"

# --- Screen layout (pixels, relative to the BlueStacks window; machine specific) ---
CONTINUE_CROP = (220, 272, 100, 25)    # center x, center y, half width, half height
PLAY_CROP = (712, 760, 285, 415)       # y1, y2, x1, x2 of the PLAY button in start_tpl.png
PLAY_BTN = (339, 712)                  # fallback click position of the PLAY button
BOX_1 = (279, 365)                     # reward screen: first click (open the box)
BOX_2 = (312, 368)                     # reward screen: second click (1 s later)
AD_SMALL = (395, 79)                   # ad close button, normal window
AD_BIG = (1334, 64)                    # ad close button, expanded (full screen) window
BACK_BTN = (209, -14)                  # window "back" button (above the game area)
BACK_AFTER = 5                         # ad attempts before pressing the back button
BACK_WAIT = 10.0                       # seconds to wait after pressing back
WIN_PREFIX = "BlueStacks"              # window title must start with this
WIN_PREFER = "Overlay"                 # prefer the window whose title contains this
# =====================================================================

here = os.path.dirname(os.path.abspath(__file__))
KEYS = {1: keyboard.Key.left, 2: keyboard.Key.right, 3: keyboard.Key.up, 4: keyboard.Key.down}
LET = {1: "L", 2: "R", 3: "U", 4: "D"}

# "Continue?" template (grayscale crop of the game-over popup)
tpl = cv2.imread(os.path.join(here, "continue_tpl.png"), cv2.IMREAD_GRAYSCALE)
if tpl is None:
    raise SystemExit("continue_tpl.png not found.")

# PLAY button template, cut from a full screenshot of the start screen
_start = cv2.imread(os.path.join(here, "start_tpl.png"))
if _start is None:
    raise SystemExit("start_tpl.png not found.")
_y1, _y2, _x1, _x2 = PLAY_CROP
PLAY_TPL = cv2.cvtColor(_start[_y1:_y2, _x1:_x2], cv2.COLOR_BGR2GRAY)


# ============================== Model ===============================
class Net(nn.Module):
    """ResNet18 on a single grayscale frame -> 5 classes (none/left/right/up/down)."""

    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 5))
        self.m = m

    def forward(self, x):
        x = (x - 0.45) / 0.225
        return self.m(x.expand(-1, 3, -1, -1))  # gray -> 3 channels


dev = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(dev)
ck = torch.load(os.path.join(here, MODEL_FILE), map_location=dev, weights_only=True)
model.load_state_dict(ck["model"])
model.eval()


# ============================== Window ==============================
def find_win():
    """Return the BlueStacks window (preferring the overlay window), or None."""
    ws = [x for x in gw.getAllWindows()
          if x.title.startswith(WIN_PREFIX) and x.visible and x.width > 200 and x.height > 200]
    pref = [x for x in ws if WIN_PREFER in x.title]
    return (pref or ws or [None])[0]


def region_of(win):
    return {"left": win.left, "top": win.top, "width": win.width, "height": win.height}


w = find_win()
if w is None:
    raise SystemExit("BlueStacks window not found. Start the game first.")
region = region_of(w)
print("Window:", w.title)
print("Capture region:", region, "| Device:", dev)
print("Delay jump/roll:", DELAY[3], DELAY[4], "| Refract jump/roll:", REFRACT[3], REFRACT[4])


# ============================== State ===============================
state = {"active": False, "quit": False, "start": 0.0, "dump": False, "clear": False,
         "lane": 1, "last_a": 0, "last_t": 0.0, "last_move": 0.0}
runs = []          # duration (seconds) of every finished game

kb = keyboard.Controller()
ms = mouse.Controller()


def toggle_bot():
    """Switch the bot ON/OFF. Switching OFF also logs the game that just ended."""
    state["active"] = not state["active"]
    if state["active"]:
        state["start"] = time.time()
        state["lane"] = 1
        state["clear"] = True
        print(">>> BOT ON")
        return
    state["dump"] = True
    end = state["last_move"] if state["last_move"] > state["start"] else time.time()
    dur = end - state["start"]
    la = state["last_a"]
    last_txt = f"{LET[la]}, {end - state['last_t']:.1f} s before out" if la else "no key"
    runs.append(dur)
    print(f">>> OUT: {dur:.0f} s | last key: {last_txt}")
    print(f"    Game #{len(runs)} | average {sum(runs) / len(runs):.1f} s | "
          f"median {sorted(runs)[len(runs) // 2]:.0f} s")
    with open(os.path.join(here, "runs_log.csv"), "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')},{dur:.1f},{last_txt}\n")
    state["last_a"] = 0


def on_press(key):
    if key == keyboard.Key.esc:
        state["quit"] = True
    elif key == keyboard.Key.f8 or key == keyboard.KeyCode.from_char("`"):
        toggle_bot()


keyboard.Listener(on_press=on_press).start()


# ============================ Input helpers ============================
def press(k):
    kb.press(k)
    time.sleep(0.03)
    kb.release(k)


def click_win(win, x, y):
    """Left click at (x, y) relative to the given window."""
    ms.position = (win.left + x, win.top + y)
    time.sleep(0.1)
    ms.click(mouse.Button.left, 1)


def nap(sec):
    """Sleep, but wake up early if Esc was pressed."""
    t_end = time.time() + sec
    while time.time() < t_end and not state["quit"]:
        time.sleep(0.1)


def blue_frac(im):
    """Fraction of strongly blue pixels (the reward screen is mostly blue)."""
    hsv = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2HSV)
    m = (hsv[:, :, 0] > 95) & (hsv[:, :, 0] < 125) & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 100)
    return float(m.mean())


# ============================ Auto restart ============================
def finish_restart(sct, home):
    """
    After the first Continue click, look at the screen every CLICK_WAIT seconds:
      - PLAY button visible -> click it and finish
      - mostly blue         -> reward screen, click the box twice
      - anything else       -> assume an ad, click its close button
    After BACK_AFTER failed ad attempts the window back button is pressed.
    `home` is the original game window (BACK_BTN is relative to it).
    """
    ad_n = 0     # consecutive ad attempts
    miss = 0     # consecutive times the window was not found
    for n in range(1, 301):
        nap(CLICK_WAIT)
        if state["quit"]:
            print("    Esc pressed: restart stopped")
            return
        cw = find_win()
        if cw is None:
            miss += 1
            print(f"    [{n}] window not found ({miss})")
            if miss >= 10:
                print("    BlueStacks seems closed: stopping the bot")
                state["quit"] = True
                return
            continue
        miss = 0

        im = np.array(sct.grab(region_of(cw)))
        big = cw.width > 800            # expanded window = full screen ad
        bl = blue_frac(im)
        play = 0.0
        px, py = PLAY_BTN
        if not big:
            gray = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
            if gray.shape[0] >= PLAY_TPL.shape[0] and gray.shape[1] >= PLAY_TPL.shape[1]:
                res = cv2.matchTemplate(gray, PLAY_TPL, cv2.TM_CCOEFF_NORMED)
                _, play, _, loc = cv2.minMaxLoc(res)
                px = loc[0] + PLAY_TPL.shape[1] // 2    # click the button where it is found
                py = loc[1] + PLAY_TPL.shape[0] // 2

        if big:
            ad_n += 1
            print(f"    [{n}] big window ({cw.width}) -> close ad (big), attempt {ad_n}")
            click_win(cw, *AD_BIG)
        elif play > PLAY_THR:
            print(f"    [{n}] PLAY screen (match={play:.2f}) -> start")
            click_win(cw, px, py)
            return
        elif bl > BLUE_THR:
            ad_n = 0
            print(f"    [{n}] reward screen (blue={bl:.2f}) -> open box")
            click_win(cw, *BOX_1)
            time.sleep(1.0)
            click_win(cw, *BOX_2)
        else:
            ad_n += 1
            print(f"    [{n}] unknown screen, assuming ad (blue={bl:.2f}, play={play:.2f}) "
                  f"-> close ad, attempt {ad_n}")
            click_win(cw, *AD_SMALL)

        if ad_n >= BACK_AFTER:
            print(f"    [{n}] ad still there after {ad_n} attempts -> back button, "
                  f"waiting {BACK_WAIT:.0f} s")
            click_win(home, *BACK_BTN)
            nap(BACK_WAIT)
            ad_n = 0
        if n % 30 == 0:
            cv2.imwrite(os.path.join(here, f"stuck_{int(time.time())}.png"),
                        cv2.cvtColor(im, cv2.COLOR_BGRA2BGR))
            print("    stuck screenshot saved")


# ============================== Debug sheet ==============================
buf = deque(maxlen=70)      # last ~5 s of (frame, probabilities, key pressed)
debug_n = 0


def save_debug_sheet():
    """Save a 4x4 grid of the last frames before game over, with the model's
    probabilities (L/R/U/D) and the keys it pressed (red frame)."""
    global debug_n
    h = list(buf)
    if len(h) < 20:
        print("    (debug sheet: not enough frames)")
        return
    idxs = list(range(len(h) - 1, -1, -4))[:16][::-1]
    tiles = []
    for j in idxs:
        g, p, _ = h[j]
        pressed = 0
        for k in range(j, max(j - 4, -1), -1):
            if h[k][2]:
                pressed = h[k][2]
                break
        img = cv2.cvtColor(cv2.resize(g, (192, 192)), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (0, 0), (192, 16), (0, 0, 0), -1)
        cv2.rectangle(img, (0, 160), (192, 192), (0, 0, 0), -1)
        cv2.putText(img, f"-{(len(h) - 1 - j) / FPS:.1f}s", (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(img, f"L{p[1]:.2f} R{p[2]:.2f}", (3, 174),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(img, f"U{p[3]:.2f} D{p[4]:.2f}", (3, 188),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        if pressed:
            cv2.rectangle(img, (0, 0), (191, 191), (0, 0, 255), 4)
            cv2.putText(img, LET[pressed], (150, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1.8, (0, 0, 255), 4)
        tiles.append(img)
    while len(tiles) < 16:
        tiles.insert(0, np.zeros((192, 192, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4)])
    debug_n += 1
    name = f"debug_{debug_n:02d}.png"
    cv2.imwrite(os.path.join(here, name), sheet)
    print(f"    debug sheet saved: {name} (red = key press; U/D keys fire after DELAY)")


# ============================== Main loop ==============================
print("Click on the BlueStacks window, then press ` to turn the bot ON/OFF.")
print("Esc = quit.")

CX, CY, HW, HH = CONTINUE_CROP
game_x, game_a = [], []                  # frames and actions of the current game
hist = deque(maxlen=max(OFFS) + 1)       # most recent frames for the model
prev = None
motion = 10.0                            # smoothed screen-change measure
last_lane = 0.0
last_jump = 0.0
refract_until = {3: 0.0, 4: 0.0}
counts = np.zeros(5, dtype=int)          # keys pressed so far (index = action)
last_print = time.time()
loops = 0
dt = 1 / FPS
nxt = time.time()

with mss.MSS() as sct:
    while not state["quit"]:
        if state["clear"]:
            buf.clear()
            refract_until[3] = refract_until[4] = 0.0
            state["clear"] = False
        if state["dump"]:
            save_debug_sheet()
            state["dump"] = False

        # --- capture ---
        img = np.array(sct.grab(region))
        g = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        score = cv2.matchTemplate(g[CY - HH:CY + HH, CX - HW:CX + HW], tpl,
                                  cv2.TM_CCOEFF_NORMED)[0][0]      # "Continue?" match
        g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if prev is not None:
            diff = float(np.abs(g.astype(np.int16) - prev.astype(np.int16)).mean())
            motion = 0.8 * motion + 0.2 * diff
        prev = g
        hist.append(g)
        loops += 1

        # --- game over: log, save data, restart ---
        if AUTO and state["active"] and score > OUT_THR:
            state["last_move"] = time.time()
            toggle_bot()                                  # turn OFF and log the game
            save_debug_sheet()
            state["dump"] = False
            if SAVE_GAME_DATA:
                os.makedirs(os.path.join(here, GAME_DIR), exist_ok=True)
                np.savez_compressed(os.path.join(here, GAME_DIR, f"game_{int(time.time())}.npz"),
                                    x=np.array(game_x, dtype=np.uint8),
                                    a=np.array(game_a, dtype=np.int8), dur=runs[-1])
            game_x.clear()
            game_a.clear()
            if len(runs) >= MAX_GAMES:
                print(f">>> {MAX_GAMES} games done, bot stopped")
                state["quit"] = True
                continue
            nap(CLICK_WAIT)
            if state["quit"]:
                continue
            click_win(w, *PLAY_BTN)                       # first click: leave "Continue?"
            finish_restart(sct, w)
            if state["quit"]:
                continue
            nw = find_win()                               # window may have moved/resized
            if nw is not None:
                w = nw
                region = region_of(w)
            nap(START_WAIT)
            toggle_bot()                                  # turn ON for the new game
            prev = None
            nxt = time.time()
            continue

        # --- play ---
        if state["active"] and motion > MOTION:
            state["last_move"] = time.time()
            with torch.no_grad():
                fr = [hist[max(0, len(hist) - 1 - o)] for o in OFFS]
                xb = torch.from_numpy(np.stack(fr)).float().div(255)[None].to(dev)
                p = torch.softmax(model(xb), 1)[0].cpu().numpy()
            now = time.time()

            ratio = np.array([p[c] / THRESH[c] for c in range(1, 5)])   # >= 1 means "act"
            if p[1] + p[2] >= LR_SUM:                     # confidence split between L and R
                s = 0 if p[1] >= p[2] else 1
                ratio[s] = max(ratio[s], 1.0)
            if now - last_lane < LANE_COOLDOWN:
                ratio[0] = ratio[1] = 0
            if now < refract_until[3]:
                ratio[2] = 0
            if now < refract_until[4] or now - last_jump < NO_DOWN_AFTER_JUMP:
                ratio[3] = 0

            a = int(np.argmax(ratio)) + 1
            did = 0
            if ratio[a - 1] >= 1:
                did = a
                counts[a] += 1
                state["last_a"] = a
                state["last_t"] = now + DELAY[a]
                threading.Timer(DELAY[a], press, args=(KEYS[a],)).start()
                if a in (1, 2):
                    last_lane = now
                    state["lane"] = max(0, state["lane"] - 1) if a == 1 else min(2, state["lane"] + 1)
                else:
                    refract_until[a] = now + REFRACT[a]
                    if a == 3:
                        last_jump = now
            buf.append((g, p, did))
            game_x.append(g)
            game_a.append(did)

        # --- status line every 5 s ---
        if time.time() - last_print > 5:
            fps = loops / (time.time() - last_print)
            print("ON" if state["active"] else "PAUSED",
                  "| keys left/right/up/down:", counts[1:],
                  "| lane:", state["lane"],
                  "| FPS:", round(fps, 1), "| motion:", round(motion, 1))
            loops = 0
            last_print = time.time()

        # --- keep a steady frame rate ---
        nxt += dt
        if nxt < time.time() - 1:
            nxt = time.time()
        time.sleep(max(0, nxt - time.time()))

print("Done. Total keys left/right/up/down:", counts[1:])