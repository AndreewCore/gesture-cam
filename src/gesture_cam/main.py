"""Entry point: a menu to pick the camera mode, then the capture → detection → overlay loop.

Modes:
- Gestos: images for hand gestures (center) and body movements (top).
- Polnareff: plays the video in step with its gesture sequence.
- Snake: a snake that follows the main hand's index finger, over the camera.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .menu import Menu, MenuOption
from .movements import BODY_MOVES, BodyMoveClassifier
from .overlay import blend, draw_skeleton, fit_height, load_gesture_images
from .placeholders import ensure_placeholders
from .pose import POSE_CONNECTIONS, PoseDetector
from .recognizer import HAND_GESTURES, GestureDetector
from .sequence import FROZEN, STEPS, VideoSequence
from .sequence_rules import detect_steps
from .snake_mode import SnakeMode
from .video import VideoPlayer, find_video

WINDOW = "Gesture Cam"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FADE_SECONDS = 0.2

GESTOS, POLNAREFF, SNAKE = "gestos", "polnareff", "snake"
MENU_OPTIONS = [
    MenuOption(GESTOS, "Gestos", "Hand & body images", "1"),
    MenuOption(POLNAREFF, "Polnareff", "Follow the video's gestures", "2"),
    MenuOption(SNAKE, "Snake", "Point with your index finger", "3"),
]


class GestureStabilizer:
    """Filters flicker: needs several frames in a row to activate, and holds briefly after losing it."""

    def __init__(self, frames_to_activate: int = 5, hold_seconds: float = 0.4):
        """Set how many frames confirm a gesture and how long it is held once lost."""
        self.frames_to_activate = frames_to_activate
        self.hold_seconds = hold_seconds
        self._candidate: str | None = None
        self._count = 0
        self.active: str | None = None
        self.active_since = 0.0
        self._last_seen = 0.0

    def update(self, gesture: str | None, now: float) -> str | None:
        """Take the frame's raw gesture and return the stable gesture to display."""
        if gesture == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = gesture, 1

        if gesture is not None and gesture == self.active:
            self._last_seen = now
        elif gesture is not None and self._count >= self.frames_to_activate:
            self.active, self.active_since, self._last_seen = gesture, now, now
        elif self.active is not None and now - self._last_seen > self.hold_seconds:
            self.active = None
        return self.active


class ImageLayer:
    """One image layer (hand or body): scales, caches and draws the active image."""

    def __init__(self, images: dict[str, np.ndarray], rel_height: float, anchor: str):
        """`anchor` is 'center' or 'top'; `rel_height` is the height relative to the frame."""
        self.images = images
        self.rel_height = rel_height
        self.anchor = anchor
        # Resizing the same image on every frame is wasted work.
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def draw(self, frame: np.ndarray, name: str | None, since: float, now: float) -> None:
        """Draw the image for `name` (if any), fading in from `since`."""
        if name not in self.images:
            return
        fh, fw = frame.shape[:2]
        target_h = int(fh * self.rel_height)
        key = (name, target_h)
        if key not in self._cache:
            self._cache[key] = fit_height(self.images[name], target_h)
        img = self._cache[key]
        ih, iw = img.shape[:2]
        x = (fw - iw) // 2
        y = (fh - ih) // 2 if self.anchor == "center" else int(fh * 0.03)
        blend(frame, img, x, y, min(1.0, (now - since) / FADE_SECONDS))


class GesturesMode:
    """'Gestos' mode: an image for the current hand gesture and another for the body movement."""

    uses_pose = True

    def __init__(self, images: dict[str, np.ndarray], hand_size: float, body_size: float):
        """Split the images into the hand layer (center) and the body layer (top)."""
        self.hand_layer = ImageLayer({k: v for k, v in images.items() if k in HAND_GESTURES}, hand_size, "center")
        self.body_layer = ImageLayer({k: v for k, v in images.items() if k in BODY_MOVES}, body_size, "top")
        self.reset()

    def reset(self) -> None:
        """Start clean so a gesture from before switching modes doesn't linger."""
        self.hand_stab = GestureStabilizer()
        # Fewer frames to activate: motion events are already held by the classifier.
        self.body_stab = GestureStabilizer(frames_to_activate=3, hold_seconds=0.4)
        self.classifier = BodyMoveClassifier()

    def process(self, frame: np.ndarray, hand_det, landmarks, now: float) -> tuple[np.ndarray, list[str]]:
        """Update the stabilizers, draw the images and return the frame and the debug lines."""
        fh, fw = frame.shape[:2]
        hand = self.hand_stab.update(hand_det.gesture if hand_det else None, now)
        raw_move = self.classifier.classify(landmarks, now, aspect=fw / fh)
        body = self.body_stab.update(raw_move, now)
        self.body_layer.draw(frame, body, self.body_stab.active_since, now)
        self.hand_layer.draw(frame, hand, self.hand_stab.active_since, now)
        return frame, [f"body: {raw_move or '-'}"]


class PolnareffMode:
    """'Polnareff' mode: the video plays in a corner while the person repeats its gestures."""

    uses_pose = True

    def __init__(self, player: VideoPlayer, video_size: float):
        """`video_size` is the video window's height relative to the camera frame."""
        self.player = player
        self.video_size = video_size
        self.sequence = VideoSequence(video_length=player.duration)

    def reset(self) -> None:
        """Hide the video and wait for the first gesture again."""
        self.sequence.reset()

    def process(self, frame: np.ndarray, hand_det, landmarks, now: float) -> tuple[np.ndarray, list[str]]:
        """Advance the sequence, draw the video and return the frame and the debug lines."""
        fh, fw = frame.shape[:2]
        steps = detect_steps(hand_det, landmarks, aspect=fw / fh)
        playhead = self.sequence.update(steps, now)
        if playhead is not None:
            label = f"Frozen {self.sequence.frozen_for(now):.1f} s" if self.sequence.state == FROZEN else None
            self.player.draw(frame, playhead, self.video_size, label)
        seq = self.sequence
        return frame, [
            f"steps: {', '.join(sorted(steps)) or '-'}",
            f"sequence: {seq.state} - step {seq.step_index + 1}/{len(STEPS)} ({STEPS[seq.step_index].name})",
        ]

    def close(self) -> None:
        """Release the video file."""
        self.player.close()


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description="Camera with gesture modes: Gestos (images), Polnareff (video), Snake.")
    p.add_argument("--mode", choices=(GESTOS, POLNAREFF, SNAKE), help="Start in this mode instead of the menu")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--images", type=Path, default=PROJECT_ROOT / "images", help="Images folder (Gestos)")
    p.add_argument("--size", type=float, default=0.35, help="Hand image height relative to the video (Gestos)")
    p.add_argument("--body-size", type=float, default=0.25, help="Body image height relative to the video (Gestos)")
    p.add_argument("--video", type=Path, default=None, help="Polnareff video (default: first video in videos/)")
    p.add_argument("--video-size", type=float, default=0.35, help="Video window height relative to the camera")
    p.add_argument("--snake-speed", type=float, default=8.0, help="Snake speed in cells per second")
    p.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "gesture_recognizer.task")
    p.add_argument("--pose-model", type=Path, default=PROJECT_ROOT / "models" / "pose_landmarker_full.task")
    p.add_argument("--min-score", type=float, default=0.6, help="Minimum gesture confidence (0-1)")
    p.add_argument("--no-hands", action="store_true", help="Disable hand detection")
    p.add_argument("--no-pose", action="store_true", help="Disable skeleton and body detection")
    p.add_argument("--windowed", action="store_true", help="Start windowed instead of fullscreen")
    return p.parse_args()


def set_fullscreen(on: bool) -> None:
    """Switch the window between fullscreen and windowed."""
    mode = cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, mode)


def draw_debug(frame, hand_detection, mode_lines: list[str]) -> None:
    """Draw the hand points, the raw hand gesture and the active mode's own debug lines."""
    h, w = frame.shape[:2]
    lines = []
    if hand_detection is not None:
        for hand in hand_detection.hands:
            for x, y in hand:
                cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1, cv2.LINE_AA)
        lines.append(f"hand: {hand_detection.gesture or '-'} ({hand_detection.score:.2f})")
    lines += mode_lines
    for i, text in enumerate(lines):
        org = (20, h - 30 - 35 * (len(lines) - 1 - i))
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)


def window_size() -> tuple[int, int] | None:
    """Current size of the window's drawing area, or None if the backend can't report it."""
    try:
        _, _, w, h = cv2.getWindowImageRect(WINDOW)
    except cv2.error:
        return None
    return (w, h) if w > 0 and h > 0 else None


def build_modes(args) -> tuple[dict[str, GesturesMode | PolnareffMode | SnakeMode], dict[str, str]]:
    """Create every mode that can run; return them plus the reason each missing one is unavailable."""
    names = HAND_GESTURES + BODY_MOVES
    created = ensure_placeholders(args.images, names)
    if created:
        print(f"Sample images created in {args.images}: {', '.join(created)}")
    modes: dict[str, GesturesMode | PolnareffMode | SnakeMode] = {
        GESTOS: GesturesMode(load_gesture_images(args.images, names), args.size, args.body_size),
        SNAKE: SnakeMode(window_size, args.snake_speed),
    }
    unavailable: dict[str, str] = {}

    video = args.video or find_video(PROJECT_ROOT / "videos")
    if video is None:
        unavailable[POLNAREFF] = "No video in videos/"
    else:
        try:
            modes[POLNAREFF] = PolnareffMode(VideoPlayer(video), args.video_size)
        except ValueError as exc:
            print(exc)
            unavailable[POLNAREFF] = "Could not open the video"
    return modes, unavailable


def main() -> None:
    """Open the camera, show the mode menu and run the loop for the chosen mode."""
    args = parse_args()
    if args.no_hands and args.no_pose:
        raise SystemExit("--no-hands and --no-pose together leave the app with nothing to detect.")
    if args.no_hands or args.no_pose:
        # Polnareff mixes hand steps (V, OK) with pose steps (hands together, salute).
        print("Warning: with --no-hands or --no-pose some Polnareff steps can't be detected.")

    modes, unavailable = build_modes(args)
    if args.no_hands:
        unavailable[SNAKE] = "Needs hand detection"
    if args.mode in unavailable:
        raise SystemExit(f"Mode {args.mode} is unavailable: {unavailable[args.mode]}")
    menu = Menu(MENU_OPTIONS)
    menu.disabled = unavailable

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hand_detector = None if args.no_hands else GestureDetector(args.model, min_score=args.min_score)
    pose_detector = None if args.no_pose else PoseDetector(args.pose_model)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, menu.on_mouse)
    fullscreen = not args.windowed
    set_fullscreen(fullscreen)
    debug, show_skeleton = False, True
    active: str | None = args.mode  # None = showing the menu
    start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("The camera stopped sending frames.")
                break
            # Mirror the image so moving feels natural, like a real mirror.
            frame = cv2.flip(frame, 1)
            now = time.monotonic()

            if active is None:
                # Detection is skipped in the menu: nothing uses it and it costs most of the frame time.
                menu.draw(frame)
            else:
                mode = modes[active]
                ts = int((now - start) * 1000)
                hand_det = hand_detector.detect(frame, ts) if hand_detector else None
                landmarks = pose_detector.detect(frame, ts) if pose_detector and mode.uses_pose else None
                # The skeleton goes under the images and the video so it doesn't cover them.
                if show_skeleton and landmarks is not None:
                    draw_skeleton(frame, landmarks, POSE_CONNECTIONS)
                # A mode may return a new image (Snake draws on a canvas sized to the window).
                frame, lines = mode.process(frame, hand_det, landmarks, now)
                if debug:
                    sides = ", ".join(hand_det.handedness) if hand_det and hand_det.handedness else "-"
                    draw_debug(frame, hand_det if mode.uses_pose else None, [f"mode: {active}  hands: {sides}", *lines])

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if active is None:
                picked = menu.choice(key)
                if picked is not None:
                    modes[picked].reset()
                    active = picked
            elif key == ord("m"):
                active = None
            if key == ord("f"):
                fullscreen = not fullscreen
                set_fullscreen(fullscreen)
            elif key == ord("d"):
                debug = not debug
            elif key == ord("s"):
                show_skeleton = not show_skeleton
            # Closing the window with its X button should also end the program.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if hand_detector:
            hand_detector.close()
        if pose_detector:
            pose_detector.close()
        cap.release()
        if POLNAREFF in modes:
            modes[POLNAREFF].close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
