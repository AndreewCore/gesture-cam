"""Envoltorio del Pose Landmarker de MediaPipe: 33 puntos del cuerpo por frame."""

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

# Pares (a, b) de índices de landmarks que forman los huesos del esqueleto.
POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]

# Cada landmark se representa como (x, y, visibilidad), con x/y normalizados a 0-1.
Landmark = tuple[float, float, float]


class PoseDetector:
    """Detecta el esqueleto de una persona frame a frame usando MediaPipe en modo VIDEO."""

    def __init__(self, model_path: Path):
        """Carga (y descarga si hace falta) el modelo de pose."""
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
        """Devuelve los 33 landmarks de la persona detectada, o None si no hay nadie."""
        # MediaPipe exige timestamps estrictamente crecientes en modo VIDEO.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        return [(lm.x, lm.y, lm.visibility or 0.0) for lm in result.pose_landmarks[0]]

    def close(self) -> None:
        """Libera los recursos nativos del detector."""
        self._landmarker.close()
