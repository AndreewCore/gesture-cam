# Gesture Cam

Opens your camera and plays a video **in step with a gesture sequence**: the video advances while you repeat, in order, the gestures it shows.

> **Note:** the hand-gesture and body-movement images described below are temporarily **disabled** (commented out in `main.py` with the `DISABLED(video-sequence)` marker) so they don't overlap with the sequence. The skeleton is still drawn.

## Video sequence

Put a video in `videos/` (the first one alphabetically is used, or pass `--video PATH`). Videos aren't committed to the repo. The sequence follows the first 4 s of the *Manly Handshake* clip:

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

## Image overlays (disabled)

The app can also overlay images based on what you do:

- **Hand gestures** (thumbs up, V sign, fist…) → image in the **center**.
- **Body movements** (arms up, T-pose, wave, jump…) → image at the **top**, with your **skeleton** drawn over the video.

Both run at the same time: a hand image and a body image can be on screen together.

## Run

```bash
uv sync
uv run gesture-cam
```

On first run it downloads the MediaPipe models to `models/`.

### Options

| Option | Effect |
|---|---|
| `--camera 1` | Use another camera |
| `--windowed` | Start in a window instead of fullscreen |
| `--min-score 0.7` | Minimum confidence for hand gestures |
| `--video path.mp4` | Sequence video (default: first video in `videos/`) |
| `--video-size 0.4` | Video window height (relative to the camera) |
| `--no-hands` / `--no-pose` | Turn off one of the two detections (some sequence steps stop working) |

### Keys

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `F` | Toggle fullscreen |
| `S` | Show/hide the skeleton |
| `D` | Debug mode (hand points, raw hand gesture, detected steps and sequence state) |

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

They check the movement rules, the sequence step rules and the sequence state machine with synthetic skeletons, hands and a simulated clock, no camera needed.
