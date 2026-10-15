"""
Print the position of every mouse click, relative to the BlueStacks window.

Use it to find the coordinates for the CONFIG section of bot_auto.py
(PLAY_BTN, BOX_1, BOX_2, AD_SMALL, AD_BIG, BACK_BTN, CONTINUE_CROP):
run this script, click the button in the game, and copy the
"inside window" numbers.

Press Esc to quit.
"""
import ctypes

ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Windows only: same pixel units as the bot

import pygetwindow as gw
from pynput import keyboard, mouse

WIN_PREFIX = "BlueStacks"      # window title must start with this
WIN_PREFER = "Overlay"         # prefer the window whose title contains this

wins = [x for x in gw.getAllWindows()
        if x.title.startswith(WIN_PREFIX) and x.visible and x.width > 200 and x.height > 200]
if not wins:
    raise SystemExit("BlueStacks window not found. Start the game first.")
pref = [x for x in wins if WIN_PREFER in x.title]
w = pref[0] if pref else wins[0]
print("Window:", w.title, "| left/top/width/height:", w.left, w.top, w.width, w.height)
print("Now click on a button in the game. Esc = quit")


def on_click(x, y, button, pressed):
    if pressed:
        print(f"Click: screen=({x},{y}) | inside window=({x - w.left},{y - w.top})")


def on_press(key):
    if key == keyboard.Key.esc:
        return False       # stops the keyboard listener, which ends the script


mouse.Listener(on_click=on_click).start()
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()