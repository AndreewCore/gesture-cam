"""Wrapper around MediaPipe's Pose Landmarker: 33 body points per frame."""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .recognizer import ensure_model

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# (a, b) landmark index pairs that form the bones of the skeleton.
POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]

# Each landmark is (x, y, visibility), with x/y normalized to 0-1.
Landmark = tuple[float, float, float]


class PoseDetector:
    """Detects one person's skeleton frame by frame using MediaPipe in VIDEO mode."""

    def __init__(self, model_path: Path):
        """Load (and download if needed) the pose model."""
        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(ensure_model(model_path, POSE_MODEL_URL))
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Landmark] | None:
        """Return the 33 landmarks of the detected person, or None if nobody is in frame."""
        # MediaPipe requires strictly increasing timestamps in VIDEO mode.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        return [(lm.x, lm.y, lm.visibility or 0.0) for lm in result.pose_landmarks[0]]

    def close(self) -> None:
        """Release the detector's native resources."""
        self._landmarker.close()
