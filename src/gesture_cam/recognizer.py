"""Envoltorio del Gesture Recognizer de MediaPipe y descarga de su modelo."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)

# Gestos que el modelo preentrenado sabe reconocer ("None" se excluye a propósito).
GESTURES = (
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
    """Resultado de un frame: gesto ganador, su confianza y landmarks de cada mano."""

    gesture: str | None
    score: float
    hands: list[list[tuple[float, float]]]


def ensure_model(path: Path) -> Path:
    """Descarga el modelo a `path` si todavía no existe y devuelve la ruta."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando modelo de gestos en {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        # Renombrar al final evita dejar un modelo corrupto si se corta la descarga.
        tmp.rename(path)
    return path


class GestureDetector:
    """Detecta gestos de mano frame a frame usando MediaPipe en modo VIDEO."""

    def __init__(self, model_path: Path, min_score: float = 0.6, num_hands: int = 2):
        """Carga el modelo; `min_score` descarta gestos con poca confianza."""
        self.min_score = min_score
        options = vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Detection:
        """Analiza un frame BGR y devuelve el gesto con mayor confianza de todas las manos."""
        # MediaPipe exige timestamps estrictamente crecientes en modo VIDEO.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(image, timestamp_ms)

        best, best_score = None, 0.0
        for categories in result.gestures:
            if not categories:
                continue
            top = categories[0]
            if top.category_name in GESTURES and top.score > best_score:
                best, best_score = top.category_name, top.score

        hands = [[(lm.x, lm.y) for lm in hand] for hand in result.hand_landmarks]
        if best_score < self.min_score:
            best = None
        return Detection(best, best_score, hands)

    def close(self) -> None:
        """Libera los recursos nativos del reconocedor."""
        self._recognizer.close()
