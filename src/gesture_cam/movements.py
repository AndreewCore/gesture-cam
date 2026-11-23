"""Recognizes body movements from pose landmarks using geometric rules.

It doesn't depend on MediaPipe: it takes lists of normalized (x, y, visibility), so it
can be tested with synthetic landmarks.
"""

from __future__ import annotations

import math
import statistics
from collections import deque

BODY_MOVES = ("arms_up", "t_pose", "hands_on_hips", "squat", "wave", "jump")

# Landmark indices of the MediaPipe pose model (33 landmarks).
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

MIN_VISIBILITY = 0.5

Point = tuple[float, float]


class Body:
    """One frame's landmarks in comparable units, with visibility helpers."""

    def __init__(self, landmarks, aspect: float = 1.0):
        """`aspect` = frame width/height; corrects for x and y being normalized differently."""
        self.pts: list[Point] = [(x * aspect, y) for x, y, _ in landmarks]
        self.vis = [v for _, _, v in landmarks]

    def visible(self, *idx: int) -> bool:
        """True if all the given landmarks are visible enough."""
        return all(self.vis[i] > MIN_VISIBILITY for i in idx)

    def __getitem__(self, i: int) -> Point:
        """Direct access to point `i`."""
        return self.pts[i]

    def mid(self, a: int, b: int) -> Point:
        """Midpoint between two landmarks."""
        (ax, ay), (bx, by) = self.pts[a], self.pts[b]
        return ((ax + bx) / 2, (ay + by) / 2)

    def torso_len(self) -> float:
        """Shoulders-to-hips distance: the ruler every other distance is measured with."""
        return max(_dist(self.mid(L_SHOULDER, R_SHOULDER), self.mid(L_HIP, R_HIP)), 1e-6)


def _dist(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a: Point, b: Point, c: Point) -> float:
    """Angle in degrees at vertex `b` formed by a-b-c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    if n == 0:
        return 180.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
    return math.degrees(math.acos(cos))


def _torso_ok(b: Body) -> bool:
    """Without visible shoulders and hips there is no reliable scale for the rules."""
    return b.visible(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)


def is_arms_up(b: Body) -> bool:
    """Both wrists above the nose."""
    if not b.visible(NOSE, L_WRIST, R_WRIST):
        return False
    # In image coordinates y grows downward, so "above" means a smaller y.
    return b[L_WRIST][1] < b[NOSE][1] and b[R_WRIST][1] < b[NOSE][1]


def is_t_pose(b: Body) -> bool:
    """Arms stretched out horizontally: wrists and elbows at shoulder height and far from the body."""
    if not (_torso_ok(b) and b.visible(L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)):
        return False
    t = b.torso_len()
    for s, e, w in ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)):
        sx, sy = b[s]
        ex, ey = b[e]
        wx, wy = b[w]
        if abs(wy - sy) > 0.35 * t or abs(ey - sy) > 0.35 * t:
            return False
        if abs(wx - sx) < 0.9 * t or abs(wx - sx) < abs(ex - sx):
            return False
    return True


def is_hands_on_hips(b: Body) -> bool:
    """Wrists next to the hips with the elbows pointing outward."""
    if not (_torso_ok(b) and b.visible(L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)):
        return False
    t = b.torso_len()
    cx = b.mid(L_SHOULDER, R_SHOULDER)[0]
    half_shoulders = abs(b[L_SHOULDER][0] - b[R_SHOULDER][0]) / 2
    for w, h, e in ((L_WRIST, L_HIP, L_ELBOW), (R_WRIST, R_HIP, R_ELBOW)):
        if _dist(b[w], b[h]) > 0.45 * t:
            return False
        # With arms hanging the elbows stay close to the body; with hands on hips they stick out.
        if abs(b[e][0] - cx) < half_shoulders + 0.15 * t:
            return False
    return True


def is_squat(b: Body) -> bool:
    """Both knees bent (hip-knee-ankle angle under ~120°)."""
    if not b.visible(L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE):
        return False
    return (
        _angle(b[L_HIP], b[L_KNEE], b[L_ANKLE]) < 120
        and _angle(b[R_HIP], b[R_KNEE], b[R_ANKLE]) < 120
    )


class MotionTracker:
    """Detects movements that need history (wave and jump) across several frames."""

    WAVE_WINDOW = 1.5  # s
    WAVE_REVERSALS = 3
    WAVE_AMPLITUDE = 0.15  # in torso lengths
    JUMP_WINDOW = 1.5  # s
    JUMP_RISE = 0.25  # in torso lengths
    FEET_RISE = 0.15

    def __init__(self):
        """Start with empty wrist and hip histories."""
        self._wrists = {L_WRIST: deque(), R_WRIST: deque()}
        self._hips: deque[tuple[float, float, float | None]] = deque()  # (t, hip_y, feet_y)

    def update(self, b: Body, now: float) -> str | None:
        """Record the frame and return 'jump', 'wave' or None."""
        if not _torso_ok(b):
            self.reset()
            return None
        t = b.torso_len()
        jump = self._update_jump(b, t, now)
        wave = self._update_wave(b, t, now)
        return "jump" if jump else "wave" if wave else None

    def reset(self) -> None:
        """Forget the history, e.g. when the person leaves the frame."""
        for d in self._wrists.values():
            d.clear()
        self._hips.clear()

    def _update_wave(self, b: Body, t: float, now: float) -> bool:
        """Wave: a wrist above its shoulder that changes x direction several times."""
        cx = b.mid(L_SHOULDER, R_SHOULDER)[0]
        waving = False
        for w, s in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
            hist = self._wrists[w]
            if not b.visible(w) or b[w][1] > b[s][1]:
                hist.clear()
                continue
            # Position relative to the body and in torso lengths: unaffected by stepping closer or sideways.
            hist.append((now, (b[w][0] - cx) / t))
            while hist and now - hist[0][0] > self.WAVE_WINDOW:
                hist.popleft()
            if _count_reversals([x for _, x in hist], self.WAVE_AMPLITUDE) >= self.WAVE_REVERSALS:
                waving = True
        return waving

    def _update_jump(self, b: Body, t: float, now: float) -> bool:
        """Jump: hips (and feet, if visible) rise quickly compared to their recent position."""
        # Store raw y and divide by the current torso when comparing: dividing each sample by
        # its own torso distorts the rise if the torso's apparent size changes.
        hip_y = b.mid(L_HIP, R_HIP)[1]
        feet_y = b.mid(L_ANKLE, R_ANKLE)[1] if b.visible(L_ANKLE, R_ANKLE) else None
        self._hips.append((now, hip_y, feet_y))
        while self._hips and now - self._hips[0][0] > self.JUMP_WINDOW:
            self._hips.popleft()
        if len(self._hips) < 5:
            return False

        base_hip = statistics.median(h for _, h, _ in self._hips)
        if (base_hip - hip_y) / t < self.JUMP_RISE:
            return False
        # Standing up from a squat also raises the hips; requiring the feet to rise tells them apart.
        feet = [f for _, _, f in self._hips if f is not None]
        if feet_y is not None and len(feet) >= 5:
            return (statistics.median(feet) - feet_y) / t >= self.FEET_RISE
        return True


def _count_reversals(xs: list[float], amplitude: float) -> int:
    """Count direction changes in `xs` larger than `amplitude` (ignores jitter)."""
    if len(xs) < 3:
        return 0
    reversals, direction = 0, 0
    lo = hi = extreme = xs[0]
    for x in xs[1:]:
        if direction == 0:
            # No direction yet: wait for the first clear move to either side.
            lo, hi = min(lo, x), max(hi, x)
            if x - lo >= amplitude:
                direction, extreme = 1, x
            elif hi - x >= amplitude:
                direction, extreme = -1, x
        elif direction == 1:
            if x > extreme:
                extreme = x
            elif extreme - x >= amplitude:
                reversals, direction, extreme = reversals + 1, -1, x
        else:
            if x < extreme:
                extreme = x
            elif x - extreme >= amplitude:
                reversals, direction, extreme = reversals + 1, 1, x
    return reversals


class BodyMoveClassifier:
    """Combines static and motion rules into a single result per frame."""

    MOTION_HOLD = 0.6  # s: a jump lasts an instant, so it is held long enough for the image to show.

    def __init__(self):
        """Create the motion tracker and the last-event state."""
        self.motion = MotionTracker()
        self._event: str | None = None
        self._event_time = -math.inf

    def classify(self, landmarks, now: float, aspect: float = 1.0) -> str | None:
        """Return the frame's movement with priority jump > wave > static poses."""
        if landmarks is None:
            self.motion.reset()
            return None
        b = Body(landmarks, aspect)

        event = self.motion.update(b, now)
        if event is not None:
            self._event, self._event_time = event, now
        if self._event is not None and now - self._event_time <= self.MOTION_HOLD:
            return self._event

        # Squat goes before hands on hips: when crouching both can match.
        for name, rule in (
            ("arms_up", is_arms_up),
            ("t_pose", is_t_pose),
            ("squat", is_squat),
            ("hands_on_hips", is_hands_on_hips),
        ):
            if rule(b):
                return name
        return None
