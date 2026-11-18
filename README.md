# Gesture Cam

Opens your camera and overlays images based on what you do:

- **Hand gestures** (thumbs up, V sign, fist…) → image in the **center**.
- **Body movements** (arms up, T-pose, wave, jump…) → image at the **top**, with your **skeleton** drawn over the video.

Both run at the same time: a hand image and a body image can be on screen together.

## Run

```bash
uv sync
uv run gesture-cam
```

On first run it downloads the MediaPipe models to `models/` and creates sample images in `images/`.

### Options

| Option | Effect |
|---|---|
| `--camera 1` | Use another camera |
| `--windowed` | Start in a window instead of fullscreen |
| `--min-score 0.7` | Minimum confidence for hand gestures |
| `--size 0.5` | Hand image height (relative to the video) |
| `--body-size 0.3` | Body image height |
| `--no-hands` / `--no-pose` | Turn off one of the two detections (if it runs slowly) |
| `--images other/folder` | Use a different images folder |

### Keys

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `F` | Toggle fullscreen |
| `S` | Show/hide the skeleton |
| `D` | Debug mode (hand points and raw hand and body detections) |

## Images

Put your images in `images/` with these names (`.png`, `.webp`, `.jpg` or `.jpeg`; a PNG with transparency looks best). If you delete one, a sample is regenerated on the next start.

**Hand (center)**

| Gesture | File |
|---|---|
| 👍 Thumbs up | `thumb_up.png` |
| 👎 Thumbs down | `thumb_down.png` |
| ✌️ Victory / V sign | `victory.png` |
| ✋ Open palm | `open_palm.png` |
| ✊ Fist | `closed_fist.png` |
| ☝️ Pointing up | `pointing_up.png` |
| 🤟 I love you | `iloveyou.png` |

**Body (top)**

| Movement | File |
|---|---|
| 🙌 Both arms up | `arms_up.png` |
| ✝️ T-pose | `t_pose.png` |
| 🧍 Hands on hips | `hands_on_hips.png` |
| 🏋️ Squat | `squat.png` |
| 👋 Wave (hand moving above the shoulder) | `wave.png` |
| 🦘 Jump | `jump.png` |

## How it works

Every camera frame (mirrored) goes through two pretrained [MediaPipe](https://ai.google.dev/edge/mediapipe) models that run locally:

1. **Gesture Recognizer** (`recognizer.py`): finds up to 2 hands (21 points each) and classifies the hand shape into 7 gestures with a confidence score. The highest-confidence gesture above `--min-score` wins.
2. **Pose Landmarker** (`pose.py`): returns 33 body points (nose, shoulders, elbows, wrists, hips, knees, ankles…) with their visibility. These are used to draw the skeleton.

Body movements (`movements.py`) are **geometric rules** on those points, measured in "torso lengths" so they work whether you are close to the camera or far away:

- **Static poses** (single frame): wrists above the nose → arms up; wrists at shoulder height and far out → T-pose; wrists by the hips with elbows out → hands on hips; hip-knee-ankle angle < 120° → squat.
- **Motion** (about 1.5 s of history): a wrist above the shoulder that changes direction 3+ times → wave; hips and feet rising suddenly compared to their recent position → jump (standing up from a squat doesn't count because the feet don't rise).
- Priority: jump > wave > static poses.

To avoid flicker, each channel (hand and body) has its own **stabilizer**: a result has to repeat for a few frames in a row to activate, and stays for a moment after it disappears. Images are alpha-blended over the video (`overlay.py`).

## Tests

```bash
uv run pytest
```

They check the movement rules with synthetic skeletons, no camera needed.
