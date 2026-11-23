"""Per-frame rules for the steps of the video gesture sequence.

Like `movements.py`, it doesn't depend on MediaPipe: it takes hand points (x, y), each hand's
gesture name and the 33 pose landmarks, so it can be tested with synthetic data.
"""

from __future__ import annotations

from .movements import (
    L_ELBOW,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ELBOW,
    R_SHOULDER,
    R_WRIST,
    Body,
    _dist,
)

# Pose landmarks for the index finger: they follow the hand better than the wrist for the salute.
L_INDEX, R_INDEX = 19, 20

# Hand landmark indices (MediaPipe Hands, 21 points per hand).
H_WRIST = 0
H_THUMB_TIP = 4
H_INDEX_TIP = 8
H_MIDDLE_MCP = 9
# (tip, PIP) of the three fingers that stay extended in the OK sign.
H_OTHER_FINGERS = ((12, 10), (16, 14), (20, 18))

HandPoints = list[tuple[float, float]]


def _shoulder_width(b: Body) -> float | None:
    """Ruler for the pose rules; shoulders stay in frame at a desk even when the hips don't."""
    if not b.visible(L_SHOULDER, R_SHOULDER):
        return None
    return max(_dist(b[L_SHOULDER], b[R_SHOULDER]), 1e-6)


def _hand(points: HandPoints, aspect: float) -> HandPoints:
    """Scale x by the frame's aspect ratio so distances within the hand are comparable."""
    return [(x * aspect, y) for x, y in points]


def _hand_size(h: HandPoints) -> float:
    """Wrist to middle-finger knuckle: stable whether the fingers are open or closed."""
    return max(_dist(h[H_WRIST], h[H_MIDDLE_MCP]), 1e-6)


def is_open_palms(hand_gestures: list[str | None]) -> bool:
    """Both hands detected and open."""
    return len(hand_gestures) >= 2 and all(g == "Open_Palm" for g in hand_gestures[:2])


def is_hands_together(b: Body | None, hands: list[HandPoints], aspect: float = 1.0) -> bool:
    """Hands pressed together in front of the chest or face (praying hands)."""
    if b is not None and b.visible(NOSE, L_WRIST, R_WRIST):
        sw = _shoulder_width(b)
        if sw is not None:
            close = _dist(b[L_WRIST], b[R_WRIST]) < 0.5 * sw
            # Below the nose and not hanging down by the waist.
            height_ok = all(b[NOSE][1] < b[w][1] < b[L_SHOULDER][1] + 2.0 * sw for w in (L_WRIST, R_WRIST))
            if close and height_ok:
                return True
    # Fallback for when the pose loses the wrists but the hand model still sees both hands.
    if len(hands) >= 2:
        h1, h2 = _hand(hands[0], aspect), _hand(hands[1], aspect)
        c1 = _centroid(h1)
        c2 = _centroid(h2)
        return _dist(c1, c2) < 1.0 * max(_hand_size(h1), _hand_size(h2))
    return False


def _centroid(h: HandPoints) -> tuple[float, float]:
    """Mean point of a hand."""
    return (sum(x for x, _ in h) / len(h), sum(y for _, y in h) / len(h))


def is_ok_sign(points: HandPoints, aspect: float = 1.0) -> bool:
    """Thumb and index tips touching in a ring, with the other three fingers extended."""
    h = _hand(points, aspect)
    size = _hand_size(h)
    if _dist(h[H_THUMB_TIP], h[H_INDEX_TIP]) > 0.35 * size:
        return False
    wrist = h[H_WRIST]
    # An extended finger has its tip farther from the wrist than its middle joint; a curled one doesn't.
    return all(_dist(wrist, h[tip]) > 1.1 * _dist(wrist, h[pip]) for tip, pip in H_OTHER_FINGERS)


def is_salute(b: Body | None) -> bool:
    """A hand at the forehead with its elbow raised out to the side, like a military salute."""
    if b is None or not b.visible(NOSE):
        return False
    sw = _shoulder_width(b)
    if sw is None:
        return False
    nx, ny = b[NOSE]
    for idx, elbow, shoulder in ((L_INDEX, L_ELBOW, L_SHOULDER), (R_INDEX, R_ELBOW, R_SHOULDER)):
        if not b.visible(idx, elbow):
            continue
        ix, iy = b[idx]
        # The elbow check separates the salute from a V or OK held up by the face with the elbow down.
        if iy < ny and abs(ix - nx) < 1.0 * sw and b[elbow][1] < b[shoulder][1] + 0.3 * sw:
            return True
    return False


def detect_steps(hand_det, landmarks, aspect: float = 1.0) -> set[str]:
    """Return every sequence step that matches this frame (several can match at once).

    `hand_det` is a `recognizer.Detection` (or None) and `landmarks` the 33 pose points (or None).
    """
    found: set[str] = set()
    hands: list[HandPoints] = hand_det.hands if hand_det is not None else []
    gestures: list[str | None] = hand_det.hand_gestures if hand_det is not None else []
    body = Body(landmarks, aspect) if landmarks is not None else None

    if is_open_palms(gestures):
        found.add("open_palms")
    if is_hands_together(body, hands, aspect):
        found.add("hands_together")
    if "Victory" in gestures:
        found.add("victory")
    if any(len(h) == 21 and is_ok_sign(h, aspect) for h in hands):
        found.add("ok_sign")
    if is_salute(body):
        found.add("salute")
    return found

