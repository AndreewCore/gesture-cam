"""Tests for the Snake game logic (grid, steering, apples, collisions, resizing)."""

import random

from gesture_cam.snake import DOWN, GAME_OVER, LEFT, PLAYING, RIGHT, UP, SnakeGame, grid_for

STEP = 1 / 8  # one move per update at the default speed


def game(cols=20, rows=12, seed=0):
    """Game with deterministic apples."""
    return SnakeGame(cols, rows, rng=random.Random(seed))


def run(g, target, steps, start=0.0):
    """Call update `steps` times, one move interval apart; return the time after the last call."""
    t = start
    for _ in range(steps):
        g.update(target, t)
        t += STEP
    return t


def test_grid_grows_with_window():
    """A bigger window gets more cells; cells stay square and at least 24 px."""
    small = grid_for(640, 360)
    big = grid_for(1920, 1080)
    assert big[0] >= small[0] and big[1] >= small[1]
    assert small[2] >= 24
    cols, rows, cell = grid_for(1280, 720)
    assert cols * cell <= 1280 and rows * cell <= 720


def test_does_not_move_without_target():
    """No hand → the snake stays put."""
    g = game()
    body = list(g.body)
    run(g, None, 10)
    assert g.body == body


def test_moves_to_target_and_stops():
    """It walks to the target cell and then stays there."""
    g = game()
    g.apple = (0, 0)  # out of the way
    target = (g.head[0] + 4, g.head[1] + 2)
    run(g, target, 6)
    assert g.head == target
    body = list(g.body)
    run(g, target, 5, start=10.0)
    assert g.body == body


def test_respects_speed():
    """Updates faster than the speed don't add extra moves."""
    g = game()
    g.apple = (0, 0)
    head = g.head
    for i in range(4):
        g.update((19, head[1]), i * 0.01)
    assert g.head == (head[0] + 1, head[1])


def test_never_turns_back_into_neck():
    """A target straight behind makes it turn to the side, never reverse."""
    g = game()
    g.apple = (19, 0)
    hx, hy = g.head
    run(g, (hx - 5, hy), 1)
    assert g.direction in (UP, DOWN)
    assert g.head != g.body[2]


def test_eating_grows_and_scores():
    """Head on the apple → one segment longer, score +1, new apple not on the body."""
    g = game()
    hx, hy = g.head
    g.apple = (hx + 1, hy)
    length = len(g.body)
    run(g, (hx + 1, hy), 1)
    assert len(g.body) == length + 1 and g.score == 1
    assert g.apple is not None and g.apple not in g.body


def test_self_collision_is_game_over_then_restarts():
    """Hitting its own body ends the game; after 2 s a new one starts."""
    g = game()
    g.body = [(5, 5), (6, 5), (6, 6), (5, 6), (4, 6), (4, 5)]
    g.direction = LEFT
    g.apple = (0, 0)
    g.update((5, 7), 0.0)  # forced down into (5, 6)
    assert g.state == GAME_OVER
    g.update((10, 10), 1.0)
    assert g.state == GAME_OVER
    g.update((10, 10), 2.1)
    assert g.state == PLAYING and len(g.body) == SnakeGame.START_LENGTH and g.score == 0


def test_moving_into_the_tail_tip_is_allowed():
    """The tail tip leaves its cell on the same step, so following it is not a collision."""
    g = game()
    g.body = [(5, 5), (6, 5), (6, 6), (5, 6)]
    g.direction = LEFT
    g.apple = (0, 0)
    g.update((5, 9), 0.0)
    assert g.state == PLAYING and g.head == (5, 6)


def test_resize_keeps_body_inside_and_connected():
    """After shrinking the grid every segment is on it and adjacent to the next one."""
    g = game(cols=30, rows=20)
    g.body = [(28, 18), (27, 18), (26, 18), (25, 18)]
    g.direction = RIGHT
    g.resize(12, 8)
    assert all(0 <= c < 12 and 0 <= r < 8 for c, r in g.body)
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(g.body, g.body[1:]))
    assert g.apple not in g.body
