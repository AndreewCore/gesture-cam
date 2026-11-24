"""'Snake' mode: the snake follows the main hand's index finger, drawn over the live camera."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from .snake import GAME_OVER, SnakeGame, grid_for

INDEX_TIP = 8  # MediaPipe hand landmark
POINTER_SMOOTHING = 0.5  # EMA weight of the new sample: the fingertip jitters a few pixels per frame.
GRID_OPACITY = 0.15
GAME_OPACITY = 0.85
FONT = cv2.FONT_HERSHEY_DUPLEX


def cover(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    """Scale `frame` to fill width×height without distortion, cropping the overflow.

    Returns the canvas plus the scale and crop offsets, to map points from the frame onto it.
    """
    fh, fw = frame.shape[:2]
    s = max(width / fw, height / fh)
    rw, rh = max(width, round(fw * s)), max(height, round(fh * s))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
    ox, oy = (rw - width) // 2, (rh - height) // 2
    return resized[oy : oy + height, ox : ox + width].copy(), s, ox, oy


def main_hand(hand_det) -> list[tuple[float, float]] | None:
    """Points of the right hand if visible, otherwise of the first hand, otherwise None."""
    if hand_det is None or not hand_det.hands:
        return None
    for points, side in zip(hand_det.hands, hand_det.handedness):
        if side == "Right":
            return points
    return hand_det.hands[0]


class SnakeMode:
    """Runs a `SnakeGame` on a canvas sized to the window, with the camera as background."""

    uses_pose = False  # Only the hand matters here; skipping pose saves most of the frame time.

    def __init__(self, window_size: Callable[[], tuple[int, int] | None], speed: float = 8.0):
        """`window_size` returns the window's current (width, height), or None if unknown."""
        self.window_size = window_size
        self.speed = speed
        self.game: SnakeGame | None = None
        self.reset()

    def reset(self) -> None:
        """Start a new game; the grid is created on the next frame, once the canvas size is known."""
        self.game = None
        self._pointer: tuple[float, float] | None = None

    def process(self, frame: np.ndarray, hand_det, landmarks, now: float) -> tuple[np.ndarray, list[str]]:
        """Move the snake toward the fingertip and return the drawn canvas and the debug lines."""
        fh, fw = frame.shape[:2]
        width, height = self.window_size() or (fw, fh)
        canvas, s, ox, oy = cover(frame, width, height)

        cols, rows, cell = grid_for(width, height)
        if self.game is None:
            self.game = SnakeGame(cols, rows, self.speed)
        else:
            self.game.resize(cols, rows)
        # The grid rarely fills the window exactly: center it and leave the leftover as a margin.
        gx, gy = (width - cols * cell) // 2, (height - rows * cell) // 2

        points = main_hand(hand_det)
        if points is None:
            self._pointer = None
        else:
            x, y = points[INDEX_TIP]
            raw = (x * fw * s - ox, y * fh * s - oy)
            p = self._pointer
            self._pointer = raw if p is None else tuple(
                p[i] + POINTER_SMOOTHING * (raw[i] - p[i]) for i in range(2)
            )
        target = None
        if self._pointer is not None:
            target = (int((self._pointer[0] - gx) // cell), int((self._pointer[1] - gy) // cell))
        self.game.update(target, now)

        self._draw(canvas, gx, gy, cell, target)
        g = self.game
        return canvas, [
            f"snake: {g.state} - length {len(g.body)} - score {g.score}",
            f"target: {target or '-'}  grid {g.cols}x{g.rows} @ {cell}px",
        ]

    def _draw(self, canvas: np.ndarray, gx: int, gy: int, cell: int, target) -> None:
        """Paint grid, apple, snake, pointer and score over the camera (in place)."""
        g = self.game
        h, w = canvas.shape[:2]

        # Every layer starts as a copy of the canvas, so blending only changes what was drawn on it.
        layer = canvas.copy()
        for c in range(g.cols + 1):
            cv2.line(layer, (gx + c * cell, gy), (gx + c * cell, gy + g.rows * cell), (255, 255, 255), 1)
        for r in range(g.rows + 1):
            cv2.line(layer, (gx, gy + r * cell), (gx + g.cols * cell, gy + r * cell), (255, 255, 255), 1)
        cv2.addWeighted(layer, GRID_OPACITY, canvas, 1 - GRID_OPACITY, 0, dst=canvas)

        layer = canvas.copy()
        if g.apple is not None:
            self._draw_apple(layer, gx + g.apple[0] * cell, gy + g.apple[1] * cell, cell)
        n = len(g.body)
        for i, (c, r) in reversed(list(enumerate(g.body))):
            x, y = gx + c * cell, gy + r * cell
            # Fades from a bright head to a darker tail so the direction reads at a glance.
            k = i / max(n - 1, 1)
            color = (int(60 - 20 * k), int(220 - 90 * k), int(60 - 20 * k))
            pad = max(1, cell // 12)
            cv2.rectangle(layer, (x + pad, y + pad), (x + cell - pad, y + cell - pad), color, -1, cv2.LINE_AA)
        self._draw_eyes(layer, gx + g.head[0] * cell, gy + g.head[1] * cell, cell, g.direction)
        cv2.addWeighted(layer, GAME_OPACITY, canvas, 1 - GAME_OPACITY, 0, dst=canvas)

        if target is not None and self._pointer is not None:
            tc = (min(max(target[0], 0), g.cols - 1), min(max(target[1], 0), g.rows - 1))
            x, y = gx + tc[0] * cell, gy + tc[1] * cell
            cv2.rectangle(canvas, (x, y), (x + cell, y + cell), (0, 220, 255), 2, cv2.LINE_AA)
            px, py = int(self._pointer[0]), int(self._pointer[1])
            cv2.circle(canvas, (px, py), max(6, cell // 4), (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), max(6, cell // 4), (0, 220, 255), 2, cv2.LINE_AA)

        _text(canvas, f"Score {g.score}", (20, 50), 1.2, 2)
        if g.state == GAME_OVER:
            band = canvas.copy()
            cv2.rectangle(band, (0, h // 2 - 70), (w, h // 2 + 70), (0, 0, 0), -1)
            cv2.addWeighted(band, 0.6, canvas, 0.4, 0, dst=canvas)
            _text_centered(canvas, f"Game over  -  score {g.score}", w // 2, h // 2, 1.6, 3)

    @staticmethod
    def _draw_apple(img: np.ndarray, x: int, y: int, cell: int) -> None:
        """Red apple with a stem and a leaf filling cell (x, y)."""
        cx, cy, r = x + cell // 2, y + cell // 2 + cell // 16, max(4, int(cell * 0.38))
        cv2.circle(img, (cx, cy), r, (40, 40, 220), -1, cv2.LINE_AA)
        cv2.circle(img, (cx - r // 3, cy - r // 3), max(2, r // 4), (140, 140, 255), -1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - r), (cx + r // 4, cy - r - r // 2), (30, 60, 90), max(2, cell // 14), cv2.LINE_AA)
        cv2.ellipse(img, (cx + r // 2, cy - r - r // 4), (max(3, r // 2), max(2, r // 4)), -30, 0, 360,
                    (60, 180, 60), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_eyes(img: np.ndarray, x: int, y: int, cell: int, direction: tuple[int, int]) -> None:
        """Two eyes on the head cell, shifted toward where it's heading."""
        dx, dy = direction
        cx, cy = x + cell // 2 + dx * cell // 5, y + cell // 2 + dy * cell // 5
        # Eyes sit side by side, perpendicular to the direction of travel.
        off = (-dy * cell // 5, dx * cell // 5)
        r = max(2, cell // 8)
        for sgn in (1, -1):
            ex, ey = cx + sgn * off[0], cy + sgn * off[1]
            cv2.circle(img, (ex, ey), r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (ex + dx * r // 2, ey + dy * r // 2), max(1, r // 2), (0, 0, 0), -1, cv2.LINE_AA)


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, thickness: int) -> None:
    """White text with a dark outline so it reads over any camera background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _text_centered(img: np.ndarray, text: str, cx: int, cy: int, scale: float, thickness: int) -> None:
    """`_text` centered on (cx, cy)."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    _text(img, text, (cx - tw // 2, cy + th // 2), scale, thickness)
