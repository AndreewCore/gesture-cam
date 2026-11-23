"""State machine that plays the video in step with a gesture sequence.

The video advances in real time only while the person is doing the gesture for the current
stretch of the video (or the next one). If they stop, the frame freezes; if they don't resume
within `FREEZE_TIMEOUT`, the video hides and the sequence starts over.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    """A gesture and the stretch of video (in seconds) that plays while it is held."""

    name: str
    start: float
    end: float


# Measured on the first 4 s of the video (frames sampled every 0.25 s).
STEPS = (
    Step("open_palms", 0.00, 0.25),
    Step("hands_together", 0.25, 0.85),
    Step("victory", 0.85, 1.45),
    Step("ok_sign", 1.45, 2.20),
    Step("salute", 2.20, 4.00),
)

HIDDEN, PLAYING, FROZEN, FINISHED = "hidden", "playing", "frozen", "finished"


class VideoSequence:
    """Turns the per-frame detected steps into a video playhead (or None when hidden)."""

    FREEZE_TIMEOUT = 3.0  # s without progress before giving up and hiding the video.
    GRACE = 0.25  # s: the detectors flicker; shorter gaps shouldn't show as a freeze.
    MAX_DT = 0.1  # s: a slow frame shouldn't make the video skip ahead.

    def __init__(self, steps: tuple[Step, ...] = STEPS, video_length: float | None = None):
        """`video_length` is how long to keep playing after the sequence ends (None = stop there)."""
        self.steps = steps
        self.video_length = video_length
        self.reset()

    def reset(self) -> None:
        """Hide the video and wait for the first gesture again."""
        self.state = HIDDEN
        self.playhead = 0.0
        self.step_index = 0
        self._last_now: float | None = None
        self._last_progress = 0.0

    def frozen_for(self, now: float) -> float:
        """Seconds the video has been stopped (0 unless frozen)."""
        return now - self._last_progress if self.state == FROZEN else 0.0

    def update(self, detected: set[str], now: float) -> float | None:
        """Take the frame's detected steps and return the video second to show, or None to hide it."""
        dt = 0.0 if self._last_now is None else min(max(now - self._last_now, 0.0), self.MAX_DT)
        self._last_now = now

        if self.state == HIDDEN:
            if self.steps[0].name in detected:
                self.state, self.playhead, self.step_index = PLAYING, 0.0, 0
                self._last_progress = now
                return self.playhead
            return None

        if self.state == FINISHED:
            # The sequence is done: the rest of the video plays on its own.
            self.playhead += dt
            if self.video_length is None or self.playhead >= self.video_length:
                self.reset()
                return None
            return self.playhead

        if self._advance(detected, dt):
            self._last_progress = now
            if self.state != FINISHED:
                self.state = PLAYING
        elif now - self._last_progress > self.FREEZE_TIMEOUT:
            self.reset()
            return None
        elif now - self._last_progress > self.GRACE:
            self.state = FROZEN
        return self.playhead

    def _advance(self, detected: set[str], dt: float) -> bool:
        """Move the playhead if the right gesture is held; return whether it moved."""
        step = self.steps[self.step_index]
        nxt = self.steps[self.step_index + 1] if self.step_index + 1 < len(self.steps) else None
        # Accepting the next gesture too covers the transition between two steps.
        if step.name not in detected and (nxt is None or nxt.name not in detected):
            return False

        new = self.playhead + dt
        if new < step.end:
            self.playhead = new
            return dt > 0
        if nxt is None:
            self.playhead, self.state = step.end, FINISHED
            return True
        if nxt.name in detected:
            self.playhead, self.step_index = new, self.step_index + 1
            return True
        # Still holding the old gesture at the end of its stretch: wait at the boundary.
        # Holding it isn't progress, so the freeze timer keeps running.
        moved = self.playhead < step.end
        self.playhead = step.end
        return moved
