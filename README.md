# Subway Surfers AI Bot

A bot that plays Subway Surfers (BlueStacks, Windows) by looking at the screen.
A ResNet18 trained on my own gameplay decides every frame whether to go left, right,
jump, roll, or do nothing. When the game ends, the bot restarts it by itself.

**Demo video:** (https://lnkd.in/p/dPXWcrCS)

> [!TIP]
> **To avoid the ad hassle, turn the internet OFF** (on the PC or inside BlueStacks) before running the bot.
> With the internet on, ads and extra popups appear at random places and the bot can get stuck.
> With it off, the screens stay in the same place and the auto-restart works reliably.

## How it works

**Method: behavior cloning (supervised imitation learning).** This is *not* reinforcement
learning: there is no reward signal. The model only copies what a human did.

1. `record.py` captures the game window at 15 FPS (grayscale, 128x128) while I play
   with the arrow keys. Frames in the 0.15 s before a key press get that key as label,
   everything else is "none".
2. `prepare.py` builds `dataset.npz` and marks frames with almost no motion
   (menus, result screens) so they are not used for training.
3. `train2.py` trains ResNet18 (ImageNet weights, new 5-class head) and keeps the epoch
   with the best average per-class recall.
4. `bot_auto.py` captures the screen, runs the model, and presses keys with a small
   delay (so the character reaches the obstacle first).

Training details that mattered:
- About 95% of frames are "no key", so each epoch is balanced (equal samples per key,
  a few "none" samples).
- The 4 frames before a key press also get its label, so the model acts slightly early.
- The first/last frames of every run are dropped (start animation, the crash itself).
- Validation uses random 150-frame chunks; neighbouring training frames are excluded
  to avoid leakage.
- Augmentation: horizontal flip (left/right labels swapped), brightness, small shift.

## Results

- Final model: ~72k frames of my own gameplay, 6 epochs, ~9 minutes on a GTX 1050 Ti.
  Average per-class recall **0.71** (none 0.64, left 0.65, right 0.64, up 0.72, down 0.91).
- Bot over 100 consecutive games, fully unattended: **average 20.3 s, median 18 s** per run.

Decision rules on top of the model (all in the CONFIG block of `bot_auto.py`):
per-key probability thresholds, per-key delays, cooldowns between lane changes,
and a rule that moves toward the likelier side when the model splits its
confidence between left and right (it did this near shiny obstacles and then did nothing).

## Auto restart

- The "Continue?" game-over popup is found with template matching.
- Reward screens (mostly blue) are opened with two clicks.
- The PLAY button is found with template matching, so it works even if the layout shifts.
- Unknown screens are treated as ads: click the close button, and press the
  window back button if the ad does not go away.
- `Esc` stops the bot at any time, also during a restart.

## What did not work

- Retraining on the bot's own long games (41 games, ~27k frames): recall dropped
  (0.71 -> 0.68), so the final model uses human data only.
- Collecting 29% more human data did not improve average recall either; the
  single-frame model seems to have reached its ceiling.
- Detecting the chasing guard by dark pixels failed: the black train looks the same.

> [!NOTE]
> This is a research/learning project, not a plug-and-play tool. The model, templates and
> click coordinates are tuned to one BlueStacks window size (441x784). On another setup you
> will need to adjust the CONFIG values and, most likely, record your own data and retrain.

## Setup (Windows only)

```
pip install -r requirements.txt
```

1. Run BlueStacks with the game visible. The scripts look for a window whose title
   starts with `BlueStacks` (they prefer one containing `Overlay`).
2. Create the two templates from your own screen:
   ```
   python make_templates.py continue
   python make_templates.py start
   ```
3. Find your click coordinates with `python mouse_pos.py` and put them in the CONFIG
   block of `bot_auto.py` (`PLAY_BTN`, `BOX_1`, `BOX_2`, `AD_SMALL`, `AD_BIG`,
   `BACK_BTN`). My values are for a 441x784 window and will not match yours.
4. Run the bot: `python bot_auto.py` (needs `model2.pt`).
   Press `` ` `` (backtick) when the character starts running. `Esc` quits.

To train on your own gameplay:

```
python record.py      # play the game, Esc to stop (data/ folder)
python prepare.py     # builds dataset.npz
python train2.py      # writes model2.pt
```

## Limitations

- Windows only; coordinates and templates depend on the window size.
- Keep the BlueStacks window uncovered and visible, since the bot reads the screen.
- Frame-level recall is ~0.71, so the bot still crashes after about 20 seconds on average.
- It may get stuck on screens it does not know (it saves `stuck_*.png` to help debugging).
- Ads and popups can still appear with the internet on (see the tip at the top). Turn it off for stable runs.
- Use only on a game and device you own.

## Files

| File | Purpose |
|---|---|
| `record.py` | record gameplay and key presses |
| `prepare.py` | build and filter the dataset |
| `train2.py` | train the model |
| `bot_auto.py` | play and auto-restart |
| `make_templates.py` | capture the two template images |
| `mouse_pos.py` | print click coordinates |
