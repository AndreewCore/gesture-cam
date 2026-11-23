# Gesture Cam

Opens your camera with a menu to pick one of two modes:

| Key | Mode | What it does |
|---|---|---|
| `1` | **Gestos** | Overlays images for hand gestures (center) and body movements (top) |
| `2` | **Polnareff** | Plays a video in step with a gesture sequence: the video advances while you repeat, in order, the gestures it shows |

Pick a mode by clicking its card or pressing its number. `M` goes back to the menu from either mode. `--mode gestos` or `--mode polnareff` skips the menu.

## Polnareff mode

Put a video in `videos/` (the first one alphabetically is used, or pass `--video PATH`). Videos aren't committed to the repo; without one the Polnareff card shows as unavailable. The sequence follows the first 4 s of the *Manly Handshake* clip:

| # | Gesture | Video stretch | How it's detected |
|---|---|---|---|
| 1 | ✋✋ Both hands open | 0.00 – 0.25 s | Both hands classified as `Open_Palm` |
| 2 | 🙏 Hands pressed together at the chest | 0.25 – 0.85 s | Pose: wrists together, between nose and waist |
| 3 | ✌️ V sign | 0.85 – 1.45 s | `Victory` from the hand model |
| 4 | 👌 OK sign | 1.45 – 2.20 s | Hand points: thumb and index tips touching, other three fingers extended |
| 5 | 🫡 Salute (flat hand at the forehead) | 2.20 – 4.00 s | Pose: index finger above the nose and near the head, elbow raised |

- The video appears in a corner (picture-in-picture) when you start step 1 and plays in real time while you hold the current gesture (or move on to the next one).
- If you stop, or the detector loses the gesture, the frame **freezes** ("Frozen 1.2 s"). Pick the gesture back up within **3 s** and it continues; otherwise the video hides and the sequence starts over.
- Holding a gesture after its stretch has played counts as stopping: you have to move on to the next one.
- After the salute, the rest of the video plays on its own (no sound: OpenCV doesn't play audio) and then hides.
- `D` (debug) shows the steps detected each frame and the sequence state.

The steps and timings live in `STEPS` in `sequence.py`; the rules in `sequence_rules.py`.

## Gestos mode

Overlays images based on what you do:

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
| `--mode gestos` | Start straight in a mode (`gestos` / `polnareff`) instead of the menu |
| `--camera 1` | Use another camera |
| `--windowed` | Start in a window instead of fullscreen |
| `--min-score 0.7` | Minimum confidence for hand gestures |
| `--size 0.5` | Hand image height, relative to the video (Gestos) |
| `--body-size 0.3` | Body image height (Gestos) |
| `--images other/folder` | Use a different images folder (Gestos) |
| `--video path.mp4` | Sequence video (default: first video in `videos/`) |
| `--video-size 0.4` | Video window height (relative to the camera) |
| `--no-hands` / `--no-pose` | Turn off one of the two detections (some sequence steps stop working) |

### Keys

| Key | Action |
|---|---|
| `1` / `2` | Pick a mode (in the menu) |
| `M` | Back to the menu |
| `Q` / `Esc` | Quit |
| `F` | Toggle fullscreen |
| `S` | Show/hide the skeleton |
| `D` | Debug mode (hand points, raw detections; in Polnareff also the detected steps and sequence state) |

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

They check the movement rules, the sequence step rules, the sequence state machine and the menu with synthetic skeletons, hands and a simulated clock, no camera needed.
