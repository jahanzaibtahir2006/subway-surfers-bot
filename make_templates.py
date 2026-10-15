"""
Capture the two template images the bot needs, from your own BlueStacks window.

    python make_templates.py continue   -> continue_tpl.png (the "Continue?" game-over popup)
    python make_templates.py start      -> start_tpl.png    (the score screen with the PLAY button)

Usage: run the command, then within 5 seconds bring the game to the matching screen.
Do not move the BlueStacks window between capturing and running the bot.
CONTINUE_CROP must be the same as in bot_auto.py.
"""
import ctypes
import os
import sys
import time

ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Windows only: exact pixel capture

import cv2
import mss
import numpy as np
import pygetwindow as gw

WIN_PREFIX = "BlueStacks"
WIN_PREFER = "Overlay"
CONTINUE_CROP = (220, 272, 100, 25)    # center x, center y, half width, half height

if len(sys.argv) != 2 or sys.argv[1] not in ("continue", "start"):
    raise SystemExit("Usage: python make_templates.py continue|start")
which = sys.argv[1]

wins = [x for x in gw.getAllWindows()
        if x.title.startswith(WIN_PREFIX) and x.visible and x.width > 200 and x.height > 200]
if not wins:
    raise SystemExit("BlueStacks window not found. Start the game first.")
pref = [x for x in wins if WIN_PREFER in x.title]
w = pref[0] if pref else wins[0]
region = {"left": w.left, "top": w.top, "width": w.width, "height": w.height}

print(f"Capturing in 5 seconds. Show the '{which}' screen now...")
time.sleep(5)
with mss.MSS() as sct:
    im = np.array(sct.grab(region))

here = os.path.dirname(os.path.abspath(__file__))
if which == "start":
    cv2.imwrite(os.path.join(here, "start_tpl.png"), cv2.cvtColor(im, cv2.COLOR_BGRA2BGR))
    print("Saved start_tpl.png", im.shape[:2])
else:
    cx, cy, hw, hh = CONTINUE_CROP
    gray = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
    cv2.imwrite(os.path.join(here, "continue_tpl.png"), gray[cy - hh:cy + hh, cx - hw:cx + hw])
    print("Saved continue_tpl.png")