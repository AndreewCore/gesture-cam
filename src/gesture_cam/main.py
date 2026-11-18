"""Entry point: shows the camera and overlays images for hand gestures and body movements."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .movements import BODY_MOVES, BodyMoveClassifier
from .overlay import blend, draw_skeleton, fit_height, load_gesture_images
from .placeholders import ensure_placeholders
from .pose import POSE_CONNECTIONS, PoseDetector
from .recognizer import HAND_GESTURES, GestureDetector

WINDOW = "Gesture Cam"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FADE_SECONDS = 0.2


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


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description="Camera that shows images based on your gestures and movements.")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--images", type=Path, default=PROJECT_ROOT / "images", help="Images folder")
    p.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "gesture_recognizer.task")
    p.add_argument("--pose-model", type=Path, default=PROJECT_ROOT / "models" / "pose_landmarker_full.task")
    p.add_argument("--min-score", type=float, default=0.6, help="Minimum gesture confidence (0-1)")
    p.add_argument("--size", type=float, default=0.35, help="Hand image height relative to the video")
    p.add_argument("--body-size", type=float, default=0.25, help="Body image height relative to the video")
    p.add_argument("--no-hands", action="store_true", help="Disable hand gestures")
    p.add_argument("--no-pose", action="store_true", help="Disable skeleton and body movements")
    p.add_argument("--windowed", action="store_true", help="Start windowed instead of fullscreen")
    return p.parse_args()


def set_fullscreen(on: bool) -> None:
    """Switch the window between fullscreen and windowed."""
    mode = cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, mode)


def draw_debug(frame, hand_detection, body_move: str | None) -> None:
    """Draw the hand points and the raw hand and body results."""
    h, w = frame.shape[:2]
    lines = []
    if hand_detection is not None:
        for hand in hand_detection.hands:
            for x, y in hand:
                cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1, cv2.LINE_AA)
        lines.append(f"hand: {hand_detection.gesture or '-'} ({hand_detection.score:.2f})")
    lines.append(f"body: {body_move or '-'}")
    for i, text in enumerate(lines):
        org = (20, h - 30 - 35 * (len(lines) - 1 - i))
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    """Open the camera and run the capture → detection (hand + body) → overlay loop."""
    args = parse_args()
    if args.no_hands and args.no_pose:
        raise SystemExit("--no-hands and --no-pose together leave the app with nothing to detect.")

    names = HAND_GESTURES + BODY_MOVES
    created = ensure_placeholders(args.images, names)
    if created:
        print(f"Sample images created in {args.images}: {', '.join(created)}")
    images = load_gesture_images(args.images, names)
    hand_layer = ImageLayer({k: v for k, v in images.items() if k in HAND_GESTURES}, args.size, "center")
    body_layer = ImageLayer({k: v for k, v in images.items() if k in BODY_MOVES}, args.body_size, "top")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hand_detector = None if args.no_hands else GestureDetector(args.model, min_score=args.min_score)
    pose_detector = None if args.no_pose else PoseDetector(args.pose_model)
    hand_stab = GestureStabilizer()
    # Fewer frames to activate: motion events are already held by the classifier.
    body_stab = GestureStabilizer(frames_to_activate=3, hold_seconds=0.4)
    classifier = BodyMoveClassifier()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fullscreen = not args.windowed
    set_fullscreen(fullscreen)
    debug, show_skeleton = False, True
    start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("The camera stopped sending frames.")
                break
            # Mirror the image so moving feels natural, like a real mirror.
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            now = time.monotonic()
            ts = int((now - start) * 1000)

            hand_det = hand_detector.detect(frame, ts) if hand_detector else None
            hand = hand_stab.update(hand_det.gesture if hand_det else None, now)

            landmarks = pose_detector.detect(frame, ts) if pose_detector else None
            raw_move = classifier.classify(landmarks, now, aspect=fw / fh) if pose_detector else None
            body = body_stab.update(raw_move, now)

            # The skeleton goes under the images so it doesn't cover them.
            if show_skeleton and landmarks is not None:
                draw_skeleton(frame, landmarks, POSE_CONNECTIONS)
            body_layer.draw(frame, body, body_stab.active_since, now)
            hand_layer.draw(frame, hand, hand_stab.active_since, now)

            if debug:
                draw_debug(frame, hand_det, raw_move)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
