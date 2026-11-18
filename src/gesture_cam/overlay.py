"""Carga de imágenes por gesto y mezcla con transparencia sobre el video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")


def gesture_filename(gesture: str) -> str:
    """Nombre base del archivo para un gesto (`Thumb_Up` → `thumb_up`)."""
    return gesture.lower()


def load_image(path: Path) -> np.ndarray:
    """Lee una imagen y la normaliza a BGRA para poder mezclarla siempre igual."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def load_gesture_images(images_dir: Path, gestures: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Devuelve {gesto: imagen BGRA} con la primera extensión encontrada por gesto."""
    images = {}
    for gesture in gestures:
        stem = gesture_filename(gesture)
        for ext in IMAGE_EXTENSIONS:
            path = images_dir / f"{stem}{ext}"
            if path.exists():
                try:
                    images[gesture] = load_image(path)
                except ValueError as exc:
                    print(exc)
                break
    return images


def fit_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Escala la imagen manteniendo proporción para que mida `target_h` de alto."""
    h, w = img.shape[:2]
    target_w = max(1, round(w * target_h / h))
    interp = cv2.INTER_AREA if target_h < h else cv2.INTER_LINEAR
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def blend(frame: np.ndarray, img_bgra: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """Pinta `img_bgra` sobre `frame` (in-place) en (x, y), recortando lo que salga del borde."""
    fh, fw = frame.shape[:2]
    ih, iw = img_bgra.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + iw, fw), min(y + ih, fh)
    if x0 >= x1 or y0 >= y1:
        return

    src = img_bgra[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = (src[:, :, 3:4].astype(np.float32) / 255.0) * opacity
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (src[:, :, :3] * alpha + roi * (1.0 - alpha)).astype(np.uint8)
