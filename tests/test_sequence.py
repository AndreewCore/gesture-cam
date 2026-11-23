"""Tests for the video sequence state machine with a simulated clock."""

from gesture_cam.sequence import FINISHED, FROZEN, HIDDEN, PLAYING, STEPS, VideoSequence

DT = 1 / 30


def feed(seq, gestures, seconds, start):
    """Feed `gestures` every frame for `seconds`; return the time after the last frame and the last result."""
    t, out = start, None
    for _ in range(round(seconds / DT)):
        out = seq.update(set(gestures), t)
        t += DT
    return t, out


def perform_steps(seq, t=0.0, upto=len(STEPS)):
    """Hold each step's gesture for exactly its stretch of video."""
    for step in STEPS[:upto]:
        t, _ = feed(seq, {step.name}, step.end - step.start + 2 * DT, t)
    return t


def test_hidden_until_first_gesture():
    """Other gestures don't start the video."""
    seq = VideoSequence()
    _, out = feed(seq, {"victory"}, 1.0, 0.0)
    assert out is None and seq.state == HIDDEN


def test_full_sequence_finishes_and_plays_the_rest():
    """All five steps in order → finished, then plays until the end of the video and hides."""
    seq = VideoSequence(video_length=6.0)
    t = perform_steps(seq)
    assert seq.state == FINISHED
    t, out = feed(seq, set(), 1.0, t)
    assert seq.state == FINISHED and out > 4.0
    feed(seq, set(), 1.5, t)
    assert seq.state == HIDDEN


def test_video_advances_with_gesture():
    """The playhead follows real time while the right gesture is held."""
    seq = VideoSequence()
    _, out = feed(seq, {"open_palms"}, 0.2, 0.0)
    assert 0.15 < out < 0.25 and seq.state == PLAYING


def test_stopping_freezes_then_resumes():
    """Pausing mid-step freezes the frame; picking the gesture back up continues from there."""
    seq = VideoSequence()
    t = perform_steps(seq, upto=2)
    t, _ = feed(seq, {"victory"}, 0.2, t)
    t, frozen_at = feed(seq, set(), 1.0, t)
    assert seq.state == FROZEN
    t, still = feed(seq, set(), 1.0, t)
    assert still == frozen_at
    _, out = feed(seq, {"victory"}, 0.1, t)
    assert seq.state == PLAYING and out > frozen_at


def test_freeze_timeout_hides():
    """More than 3 s without resuming hides the video and resets."""
    seq = VideoSequence()
    t = perform_steps(seq, upto=2)
    _, out = feed(seq, set(), 3.2, t)
    assert out is None and seq.state == HIDDEN and seq.step_index == 0


def test_holding_old_gesture_waits_at_boundary_and_times_out():
    """Holding open palms forever doesn't advance past its stretch, and still times out."""
    seq = VideoSequence()
    t, out = feed(seq, {"open_palms"}, 1.0, 0.0)
    assert out == STEPS[0].end and seq.state == FROZEN
    while seq.update({"open_palms"}, t) is not None:
        t += DT
    assert seq.state == HIDDEN and 3.0 < t < 3.5
    t += DT
    # The first gesture is still held, so the sequence starts over on the next frame.
    assert seq.update({"open_palms"}, t) == 0.0


def test_skipping_a_step_does_not_advance():
    """Jumping from open palms straight to V doesn't count as the hands-together step."""
    seq = VideoSequence()
    t, _ = feed(seq, {"open_palms"}, 0.3, 0.0)
    _, out = feed(seq, {"victory"}, 1.0, t)
    assert out == STEPS[0].end


def test_short_flicker_does_not_freeze():
    """A gap shorter than GRACE keeps the state as playing."""
    seq = VideoSequence()
    t = perform_steps(seq, upto=2)
    t, _ = feed(seq, {"victory"}, 0.1, t)
    t, _ = feed(seq, set(), 0.1, t)
    assert seq.state == PLAYING
