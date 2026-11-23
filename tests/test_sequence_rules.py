"""Tests for the sequence step rules with synthetic hands and skeletons."""

from gesture_cam import movements as m
from gesture_cam import sequence_rules as r
from gesture_cam.movements import Body
from gesture_cam.recognizer import Detection

from test_movements import standing


def pose(overrides):
    """Standing skeleton with some points moved; `overrides` = {index: (x, y)}."""
    lm = standing()
    for i, (x, y) in overrides.items():
        lm[i] = (x, y, 0.99)
    return lm


def hand(**overrides):
    """21-point hand, upright, with three fingers extended and an OK ring; `overrides` = {index: (x, y)}."""
    pts = {
        0: (0.5, 0.8), 9: (0.5, 0.7),  # wrist, middle knuckle → hand size 0.1
        4: (0.45, 0.65), 8: (0.46, 0.65),  # thumb and index tips touching
        10: (0.5, 0.62), 12: (0.5, 0.55),
        14: (0.53, 0.63), 16: (0.54, 0.56),
        18: (0.56, 0.66), 20: (0.58, 0.6),
    }
    pts.update({int(k[1:]): v for k, v in overrides.items()})
    return [pts.get(i, (0.5, 0.72)) for i in range(21)]


def test_ok_sign():
    """Ring with thumb and index, other fingers extended → OK."""
    assert r.is_ok_sign(hand())


def test_ok_sign_rejects_open_ring():
    """Thumb and index apart is not an OK."""
    assert not r.is_ok_sign(hand(p4=(0.38, 0.68)))


def test_ok_sign_rejects_curled_fingers():
    """With the other fingers curled it's a pinch, not an OK."""
    assert not r.is_ok_sign(hand(p12=(0.5, 0.68), p16=(0.53, 0.69), p20=(0.55, 0.7)))


def test_open_palms_needs_both_hands():
    """Both hands must be open palms."""
    assert r.is_open_palms(["Open_Palm", "Open_Palm"])
    assert not r.is_open_palms(["Open_Palm"])
    assert not r.is_open_palms(["Open_Palm", "Victory"])


def test_hands_together_from_pose():
    """Wrists together in front of the chest."""
    b = Body(pose({m.L_WRIST: (0.49, 0.38), m.R_WRIST: (0.51, 0.38)}))
    assert r.is_hands_together(b, [])


def test_hands_apart_is_not_together():
    """Arms down by the sides."""
    assert not r.is_hands_together(Body(standing()), [])


def test_hands_together_above_head_is_not_together():
    """Hands together above the head is a different pose."""
    b = Body(pose({m.L_WRIST: (0.49, 0.05), m.R_WRIST: (0.51, 0.05)}))
    assert not r.is_hands_together(b, [])


def test_salute():
    """Index finger at the forehead, elbow up at shoulder height."""
    b = Body(pose({r.R_INDEX: (0.52, 0.1), m.R_WRIST: (0.56, 0.12), m.R_ELBOW: (0.72, 0.3)}))
    assert r.is_salute(b)


def test_hand_by_face_with_elbow_down_is_not_salute():
    """A V or OK held up by the face keeps the elbow low."""
    b = Body(pose({r.R_INDEX: (0.6, 0.1), m.R_WRIST: (0.6, 0.18), m.R_ELBOW: (0.62, 0.45)}))
    assert not r.is_salute(b)


def test_detect_steps_combines_hand_and_pose():
    """Victory from the hand model plus hands together from the pose in one frame."""
    det = Detection("Victory", 0.9, [hand(p4=(0.38, 0.68))], ["Victory"])
    lm = pose({m.L_WRIST: (0.49, 0.38), m.R_WRIST: (0.51, 0.38)})
    assert r.detect_steps(det, lm) == {"victory", "hands_together"}


def test_detect_steps_with_nothing():
    """No hands and no person → no steps."""
    assert r.detect_steps(None, None) == set()
