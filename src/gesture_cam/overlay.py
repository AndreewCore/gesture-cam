"""Loads per-gesture images and blends them, with transparency, over the video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")


def gesture_filename(gesture: str) -> str:
    """Base file name for a gesture (`Thumb_Up` → `thumb_up`)."""
    return gesture.lower()


def load_image(path: Path) -> np.ndarray:
    """Read an image and normalize it to BGRA so every image blends the same way."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def load_gesture_images(images_dir: Path, gestures: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Return {gesture: BGRA image}, using the first matching extension for each gesture."""
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
    """Scale the image, keeping its aspect ratio, so it is `target_h` pixels tall."""
    h, w = img.shape[:2]
    target_w = max(1, round(w * target_h / h))
    interp = cv2.INTER_AREA if target_h < h else cv2.INTER_LINEAR
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def blend(frame: np.ndarray, img_bgra: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """Paint `img_bgra` onto `frame` (in place) at (x, y), clipping whatever falls off the edge."""
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


def draw_skeleton(frame: np.ndarray, landmarks, connections, min_visibility: float = 0.5) -> None:
    """Draw the skeleton's bones and joints (in place), skipping barely visible points."""
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) if v > min_visibility else None for x, y, v in landmarks]
    for a, b in connections:
        if pts[a] and pts[b]:
            # Dark outline underneath so the lines stay visible on light backgrounds.
            cv2.line(frame, pts[a], pts[b], (40, 40, 40), 7, cv2.LINE_AA)
            cv2.line(frame, pts[a], pts[b], (255, 230, 80), 3, cv2.LINE_AA)
    for p in pts:
        if p:
            cv2.circle(frame, p, 5, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 3, (80, 120, 255), -1, cv2.LINE_AA)
