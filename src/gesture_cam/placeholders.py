"""Genera imágenes de ejemplo para los gestos que aún no tienen imagen propia."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .overlay import IMAGE_EXTENSIONS, gesture_filename

# Color BGR y etiqueta legible por gesto.
_STYLE = {
    "Thumb_Up": ((80, 200, 60), "LIKE!"),
    "Thumb_Down": ((60, 60, 220), "NOPE"),
    "Victory": ((220, 160, 40), "PEACE"),
    "Open_Palm": ((40, 200, 230), "HOLA!"),
    "Closed_Fist": ((50, 50, 50), "POW!"),
    "Pointing_Up": ((200, 80, 180), "IDEA!"),
    "ILoveYou": ((150, 60, 230), "LOVE"),
    # Movimientos del cuerpo.
    "arms_up": ((0, 140, 255), "ARMS UP"),
    "t_pose": ((180, 120, 0), "T-POSE"),
    "hands_on_hips": ((120, 0, 120), "HIPS"),
    "squat": ((0, 100, 200), "SQUAT"),
    "wave": ((0, 200, 160), "HI!"),
    "jump": ((30, 30, 200), "JUMP!"),
}


def _render(color: tuple[int, int, int], label: str, size: int = 512) -> np.ndarray:
    """Dibuja un círculo con texto sobre fondo transparente (BGRA)."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center, radius = (size // 2, size // 2), size // 2 - 12
    cv2.circle(img, center, radius, (*color, 255), -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, (255, 255, 255, 255), 10, cv2.LINE_AA)

    font, thickness = cv2.FONT_HERSHEY_DUPLEX, 6
    scale = 3.0
    # Reducimos la escala hasta que el texto quepa dentro del círculo.
    while cv2.getTextSize(label, font, scale, thickness)[0][0] > size * 0.75:
        scale -= 0.1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    org = ((size - tw) // 2, (size + th) // 2)
    cv2.putText(img, label, org, font, scale, (255, 255, 255, 255), thickness, cv2.LINE_AA)
    return img


def ensure_placeholders(images_dir: Path, gestures: tuple[str, ...]) -> list[str]:
    """Crea un PNG de ejemplo por cada gesto sin imagen; devuelve los gestos generados."""
    images_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for gesture in gestures:
        stem = gesture_filename(gesture)
        if any((images_dir / f"{stem}{ext}").exists() for ext in IMAGE_EXTENSIONS):
            continue
        color, label = _STYLE.get(gesture, ((128, 128, 128), gesture))
        cv2.imwrite(str(images_dir / f"{stem}.png"), _render(color, label))
        created.append(gesture)
    return created
