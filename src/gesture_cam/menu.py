"""Mode-selection menu drawn with OpenCV over the dimmed camera image."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX


@dataclass(frozen=True)
class MenuOption:
    """One selectable mode: its id, the name shown on the card, a subtitle and the key that picks it."""

    mode: str
    title: str
    subtitle: str
    key: str


class Menu:
    """Draws the option cards and turns clicks or number keys into a chosen mode."""

    def __init__(self, options: list[MenuOption]):
        """`options` are laid out left to right in the given order."""
        self.options = options
        self.disabled: dict[str, str] = {}  # mode -> reason it can't be picked
        self._rects: list[tuple[int, int, int, int]] = []
        self._hover: int | None = None
        self._clicked: str | None = None

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """OpenCV mouse callback: track the hovered card and remember clicks."""
        self._hover = self._hit(x, y)
        if event == cv2.EVENT_LBUTTONUP and self._hover is not None:
            self._clicked = self.options[self._hover].mode

    def choice(self, key: int) -> str | None:
        """Return the mode picked this frame (by key or by a pending click), if it's available."""
        picked = next((o.mode for o in self.options if key == ord(o.key)), None)
        if picked is None:
            picked, self._clicked = self._clicked, None
        return picked if picked is not None and picked not in self.disabled else None

    def _hit(self, x: int, y: int) -> int | None:
        """Index of the card under (x, y), if any."""
        for i, (x0, y0, x1, y1) in enumerate(self._rects):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
        return None

    def draw(self, frame: np.ndarray) -> None:
        """Dim `frame` and paint the title, the cards and the key hints on it (in place)."""
        fh, fw = frame.shape[:2]
        frame[:] = (frame * 0.35).astype(np.uint8)
        _text_centered(frame, "Gesture Cam", fw // 2, int(fh * 0.16), 1.6, 3, (255, 255, 255))
        _text_centered(frame, "Choose a mode", fw // 2, int(fh * 0.24), 0.8, 1, (200, 200, 200))

        n = len(self.options)
        gap = int(fw * 0.04)
        card_w = min(int(fw * 0.34), (fw - gap * (n + 1)) // n)
        card_h = int(fh * 0.42)
        x = (fw - (card_w * n + gap * (n - 1))) // 2
        y = int(fh * 0.32)
        # Rects are recomputed every frame so they follow the camera's resolution.
        self._rects = []
        for i, opt in enumerate(self.options):
            rect = (x, y, x + card_w, y + card_h)
            self._rects.append(rect)
            self._draw_card(frame, opt, rect, hovered=i == self._hover)
            x += card_w + gap

        hint = "Click a card or press " + " / ".join(o.key for o in self.options) + "   |   M: menu   Q: quit"
        _text_centered(frame, hint, fw // 2, int(fh * 0.86), 0.7, 1, (220, 220, 220))

    def _draw_card(self, frame: np.ndarray, opt: MenuOption, rect, hovered: bool) -> None:
        """Paint one option card; disabled cards are greyed out with the reason."""
        x0, y0, x1, y1 = rect
        disabled = opt.mode in self.disabled
        fill = (60, 60, 60) if disabled else (120, 70, 30) if hovered else (80, 45, 20)
        border = (120, 120, 120) if disabled else (80, 220, 255) if hovered else (255, 255, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), fill, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y0), (x1, y1), border, 4 if hovered else 2, cv2.LINE_AA)

        cx, h = (x0 + x1) // 2, y1 - y0
        text = (150, 150, 150) if disabled else (255, 255, 255)
        cv2.circle(frame, (cx, y0 + int(h * 0.24)), int(h * 0.11), border, 2, cv2.LINE_AA)
        _text_centered(frame, opt.key, cx, y0 + int(h * 0.24), 1.2, 2, text)
        _text_centered(frame, opt.title, cx, y0 + int(h * 0.55), _fit(opt.title, x1 - x0, 1.6, 3), 3, text)
        sub = self.disabled.get(opt.mode, opt.subtitle)
        sub_color = (90, 90, 230) if disabled else (210, 210, 210)
        _text_centered(frame, sub, cx, y0 + int(h * 0.75), _fit(sub, x1 - x0, 0.7, 1), 1, sub_color)


def _fit(text: str, width: int, scale: float, thickness: int) -> float:
    """Largest scale (up to `scale`) at which `text` fits in 85% of `width`."""
    while scale > 0.3 and cv2.getTextSize(text, FONT, scale, thickness)[0][0] > width * 0.85:
        scale -= 0.05
    return scale


def _text_centered(frame, text: str, cx: int, cy: int, scale: float, thickness: int, color) -> None:
    """Draw `text` centered on (cx, cy) with a dark outline so it reads on any background."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    org = (cx - tw // 2, cy + th // 2)
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)
