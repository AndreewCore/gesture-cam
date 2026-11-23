"""Reads the sequence video by timestamp and draws it as a picture-in-picture over the camera."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .overlay import fit_height

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def find_video(folder: Path) -> Path | None:
    """Return the first video in `folder` (alphabetically), or None if there isn't one."""
    videos = sorted(p for p in folder.glob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    return videos[0] if videos else None


class VideoPlayer:
    """Returns the frame for any second of the video, decoding only what's needed."""

    def __init__(self, path: Path):
        """Open the video and read its fps and length."""
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        self._index = -1  # Index of the last decoded frame.
        self._frame: np.ndarray | None = None
        self._scaled: tuple[int, int, np.ndarray] | None = None  # (index, height, image)

    def frame_at(self, t: float) -> np.ndarray | None:
        """Return the frame shown at second `t` (the cached one if it hasn't changed)."""
        target = min(max(int(t * self.fps), 0), max(self.frame_count - 1, 0))
        if target == self._index:
            return self._frame
        if target < self._index or target - self._index > self.fps:
            # Going back (restart) or a big jump: seeking beats decoding everything in between.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self._index = target - 1
        # grab() skips decoding the frames we only need to step over.
        while self._index < target - 1:
            if not self._cap.grab():
                return self._frame
            self._index += 1
        ok, frame = self._cap.read()
        if ok:
            self._frame, self._index = frame, target
        return self._frame

    def draw(self, frame: np.ndarray, t: float, rel_height: float = 0.35, label: str | None = None) -> None:
        """Paint the video frame for second `t` in the bottom-right corner of `frame` (in place)."""
        src = self.frame_at(t)
        if src is None:
            return
        fh, fw = frame.shape[:2]
        target_h = int(fh * rel_height)
        if self._scaled is None or self._scaled[:2] != (self._index, target_h):
            self._scaled = (self._index, target_h, fit_height(src, target_h))
        img = self._scaled[2]
        ih, iw = img.shape[:2]
        iw = min(iw, fw - 40)
        margin, border = 20, 3
        x, y = fw - iw - margin, fh - ih - margin
        cv2.rectangle(frame, (x - border, y - border), (x + iw + border - 1, y + ih + border - 1), (255, 255, 255), -1)
        frame[y : y + ih, x : x + iw] = img[:, :iw]
        if label:
            org = (x + 10, y + 35)
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

    def close(self) -> None:
        """Release the video file."""
        self._cap.release()
