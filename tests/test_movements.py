"""Tests for the movement rules using synthetic skeletons (no camera or MediaPipe)."""

from gesture_cam import movements as m
from gesture_cam.movements import BodyMoveClassifier

FPS = 30
DT = 1 / FPS


def standing(**overrides):
    """33-point standing skeleton with arms down; `overrides` = {index: (x, y)}."""
    pts = [(0.5, 0.5)] * 33
    base = {
        m.NOSE: (0.5, 0.15),
        m.L_SHOULDER: (0.4, 0.3), m.R_SHOULDER: (0.6, 0.3),
        m.L_ELBOW: (0.38, 0.45), m.R_ELBOW: (0.62, 0.45),
        m.L_WRIST: (0.37, 0.6), m.R_WRIST: (0.63, 0.6),
        m.L_HIP: (0.43, 0.6), m.R_HIP: (0.57, 0.6),
        m.L_KNEE: (0.43, 0.78), m.R_KNEE: (0.57, 0.78),
        m.L_ANKLE: (0.43, 0.95), m.R_ANKLE: (0.57, 0.95),
    }
    base.update(overrides)
    for i, p in base.items():
        pts[i] = p
    return [(x, y, 0.99) for x, y in pts]


def shifted(landmarks, dy):
    """Shift the whole skeleton vertically (negative dy = up)."""
    return [(x, y + dy, v) for x, y, v in landmarks]


def squatting():
    """Squatting skeleton: lower trunk and knees pointing out."""
    lm = standing()
    upper = {m.NOSE, m.L_SHOULDER, m.R_SHOULDER, m.L_ELBOW, m.R_ELBOW, m.L_WRIST, m.R_WRIST, m.L_HIP, m.R_HIP}
    lm = [(x, y + 0.15, v) if i in upper else (x, y, v) for i, (x, y, v) in enumerate(lm)]
    lm[m.L_KNEE] = (0.33, 0.8, 0.99)
    lm[m.R_KNEE] = (0.67, 0.8, 0.99)
    return lm


def run(frames, start=0.0):
    """Run a sequence of frames through a fresh classifier and return every result."""
    clf = BodyMoveClassifier()
    return [clf.classify(lm, start + i * DT) for i, lm in enumerate(frames)]


def test_neutral_is_nothing():
    """Standing still must not trigger any movement."""
    assert run([standing()] * 30)[-1] is None


def test_no_person_is_nothing():
    """No person in frame means no movement."""
    assert BodyMoveClassifier().classify(None, 0.0) is None


def test_arms_up():
    """Both wrists above the nose."""
    lm = standing()
    lm[m.L_WRIST] = (0.4, 0.05, 0.99)
    lm[m.R_WRIST] = (0.6, 0.05, 0.99)
    assert run([lm])[-1] == "arms_up"


def test_t_pose():
    """Arms stretched out at shoulder height."""
    lm = standing()
    lm[m.L_ELBOW], lm[m.R_ELBOW] = (0.25, 0.3, 0.99), (0.75, 0.3, 0.99)
    lm[m.L_WRIST], lm[m.R_WRIST] = (0.1, 0.3, 0.99), (0.9, 0.3, 0.99)
    assert run([lm])[-1] == "t_pose"


def test_hands_on_hips():
    """Wrists on the hips with elbows out."""
    lm = standing()
    lm[m.L_ELBOW], lm[m.R_ELBOW] = (0.3, 0.45, 0.99), (0.7, 0.45, 0.99)
    lm[m.L_WRIST], lm[m.R_WRIST] = (0.44, 0.58, 0.99), (0.56, 0.58, 0.99)
    assert run([lm])[-1] == "hands_on_hips"


def test_squat():
    """Both knees bent."""
    assert run([squatting()])[-1] == "squat"


def test_low_visibility_ignored():
    """Barely visible points must not trigger rules."""
    lm = standing()
    lm[m.L_WRIST] = (0.4, 0.05, 0.1)
    lm[m.R_WRIST] = (0.6, 0.05, 0.1)
    assert run([lm])[-1] is None


def test_wave():
    """Hand above the shoulder moving side to side."""
    frames = []
    for i in range(int(1.2 * FPS)):
        lm = standing()
        # Wrist above the shoulder going back and forth every 0.2 s.
        x = 0.65 if (i // 6) % 2 == 0 else 0.8
        lm[m.R_WRIST] = (x, 0.2, 0.99)
        lm[m.R_ELBOW] = (0.66, 0.35, 0.99)
        frames.append(lm)
    assert "wave" in run(frames)


def test_raised_hand_still_is_not_wave():
    """A raised but still hand is not a wave."""
    lm = standing()
    lm[m.R_WRIST] = (0.7, 0.2, 0.99)
    assert "wave" not in run([lm] * 45)


def test_jump():
    """The whole body rises suddenly compared to its recent position."""
    frames = [standing()] * 20 + [shifted(standing(), -0.1)] * 3
    assert run(frames)[-1] == "jump"


def test_jump_is_held_briefly():
    """The jump keeps being reported for a moment after landing."""
    frames = [standing()] * 20 + [shifted(standing(), -0.1)] * 3 + [standing()] * 5
    assert run(frames)[-1] == "jump"


def test_standing_up_from_squat_is_not_jump():
    """Standing up from a squat raises the hips but not the feet: not a jump."""
    frames = [squatting()] * 20 + [standing()] * 5
    assert "jump" not in run(frames)
