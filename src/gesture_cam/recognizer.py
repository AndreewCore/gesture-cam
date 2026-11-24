"""Wrapper around MediaPipe's Gesture Recognizer, plus model download."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)

# Gestures the pretrained model can recognize ("None" is excluded on purpose).
HAND_GESTURES = (
    "Thumb_Up",
    "Thumb_Down",
    "Victory",
    "Open_Palm",
    "Closed_Fist",
    "Pointing_Up",
    "ILoveYou",
)


@dataclass
class Detection:
    """One frame's result: winning gesture, its confidence, and landmarks and gesture for each hand."""

    gesture: str | None
    score: float
    hands: list[list[tuple[float, float]]]
    # Aligned with `hands`; rules like "both palms open" need each hand's gesture, not just the best one.
    hand_gestures: list[str | None] = field(default_factory=list)
    # "Left"/"Right" per hand, aligned with `hands`; picks the main hand in Snake.
    handedness: list[str] = field(default_factory=list)


def ensure_model(path: Path, url: str) -> Path:
    """Download the model from `url` to `path` if it doesn't exist yet, and return the path."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        # Renaming only at the end avoids leaving a corrupt model if the download is interrupted.
        tmp.rename(path)
    return path


class GestureDetector:
    """Detects hand gestures frame by frame using MediaPipe in VIDEO mode."""

    def __init__(self, model_path: Path, min_score: float = 0.6, num_hands: int = 2):
        """Load the model; `min_score` discards low-confidence gestures."""
        self.min_score = min_score
        options = vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model(model_path, HAND_MODEL_URL))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Detection:
        """Analyze a BGR frame and return the highest-confidence gesture across all hands."""
        # MediaPipe requires strictly increasing timestamps in VIDEO mode.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(image, timestamp_ms)

        best, best_score = None, 0.0
        per_hand: list[str | None] = []
        for categories in result.gestures:
            top = categories[0] if categories else None
            valid = top is not None and top.category_name in HAND_GESTURES
            per_hand.append(top.category_name if valid and top.score >= self.min_score else None)
            if valid and top.score > best_score:
                best, best_score = top.category_name, top.score

        hands = [[(lm.x, lm.y) for lm in hand] for hand in result.hand_landmarks]
        # MediaPipe labels hands assuming a mirrored selfie image, which is what we feed it.
        sides = [h[0].category_name if h else "" for h in result.handedness]
        if best_score < self.min_score:
            best = None
        return Detection(best, best_score, hands, per_hand, sides)

    def close(self) -> None:
        """Release the recognizer's native resources."""
        self._recognizer.close()
