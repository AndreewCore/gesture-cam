"""Snake game logic on a grid, steered toward a target cell (no OpenCV, so it can be tested).

Unlike classic Snake it only moves while it has somewhere to go: it walks one cell at a time
toward the target and stops once its head is on it.
"""

from __future__ import annotations

import random

Cell = tuple[int, int]  # (col, row)

PLAYING, GAME_OVER = "playing", "game_over"
RIGHT, LEFT, DOWN, UP = (1, 0), (-1, 0), (0, 1), (0, -1)


def grid_for(width: int, height: int, cells_short_side: int = 18, min_cell: int = 24) -> tuple[int, int, int]:
    """Return (cols, rows, cell_px) for a window: square cells, more of them in a bigger window."""
    cell = max(min_cell, round(min(width, height) / cells_short_side))
    return max(width // cell, 4), max(height // cell, 4), cell


class SnakeGame:
    """State of one game: body (head first), apple, score and game-over timer."""

    START_LENGTH = 3
    GAME_OVER_SECONDS = 2.0

    def __init__(self, cols: int, rows: int, speed: float = 8.0, rng: random.Random | None = None):
        """`speed` is in cells per second; `rng` lets tests make the apples deterministic."""
        self.cols, self.rows = cols, rows
        self.speed = speed
        self.rng = rng or random.Random()
        self.reset()

    def reset(self) -> None:
        """New game: short snake in the middle heading right, score 0, fresh apple."""
        c, r = self.cols // 2, self.rows // 2
        self.body: list[Cell] = [(c - i, r) for i in range(self.START_LENGTH)]
        self.direction = RIGHT
        self.score = 0
        self.state = PLAYING
        self.over_at = 0.0
        self._last_move = -1e9
        self.apple = self._spawn_apple()

    @property
    def head(self) -> Cell:
        """The head's cell."""
        return self.body[0]

    def update(self, target: Cell | None, now: float) -> None:
        """Advance one cell toward `target` if it's time to; stay still without a target or on it."""
        if self.state == GAME_OVER:
            if now - self.over_at >= self.GAME_OVER_SECONDS:
                self.reset()
            return
        if target is None:
            return
        target = (min(max(target[0], 0), self.cols - 1), min(max(target[1], 0), self.rows - 1))
        if target == self.head or now - self._last_move < 1.0 / self.speed:
            return
        self._last_move = now

        d = self._choose_direction(target)
        new = (self.head[0] + d[0], self.head[1] + d[1])
        if not self._inside(new):
            return
        growing = new == self.apple
        # The tail tip moves out of the way this step, unless the snake is growing.
        obstacles = self.body if growing else self.body[:-1]
        if new in obstacles:
            self.state, self.over_at = GAME_OVER, now
            return
        self.direction = d
        self.body.insert(0, new)
        if growing:
            self.score += 1
            self.apple = self._spawn_apple()
        else:
            self.body.pop()

    def resize(self, cols: int, rows: int) -> None:
        """Fit the game to a new grid size without restarting it."""
        if (cols, rows) == (self.cols, self.rows):
            return
        old_cols, old_rows = self.cols, self.rows
        self.cols, self.rows = cols, rows

        def scale(cell: Cell) -> Cell:
            return self._clamp((cell[0] * cols // old_cols, cell[1] * rows // old_rows))

        # Scale only the head and keep the body's shape relative to it, so segments stay adjacent.
        hx, hy = self.head
        nx, ny = scale(self.head)
        body: list[Cell] = []
        for x, y in self.body:
            cell = self._clamp((nx + x - hx, ny + y - hy))
            if not body or body[-1] != cell:
                body.append(cell)
        self.body = body
        if self.apple is not None:
            self.apple = scale(self.apple)
            if self.apple in self.body:
                self.apple = self._spawn_apple()

    def _choose_direction(self, target: Cell) -> tuple[int, int]:
        """Step that brings the head closer to `target` without turning back into its neck."""
        dx, dy = target[0] - self.head[0], target[1] - self.head[1]
        step_x = (1 if dx > 0 else -1, 0) if dx else None
        step_y = (0, 1 if dy > 0 else -1) if dy else None
        ordered = [step_x, step_y] if abs(dx) >= abs(dy) else [step_y, step_x]
        reverse = (-self.direction[0], -self.direction[1])
        for d in ordered:
            if d is not None and d != reverse:
                return d
        # The target is straight behind: turn to whichever side is free, then come around.
        sides = [(self.direction[1], self.direction[0]), (-self.direction[1], -self.direction[0])]
        for d in sides:
            cell = (self.head[0] + d[0], self.head[1] + d[1])
            if self._inside(cell) and cell not in self.body:
                return d
        return next((d for d in sides if self._inside((self.head[0] + d[0], self.head[1] + d[1]))), sides[0])

    def _spawn_apple(self) -> Cell | None:
        """Random free cell for the apple, or None when the snake fills the grid."""
        taken = set(self.body)
        free = [(c, r) for c in range(self.cols) for r in range(self.rows) if (c, r) not in taken]
        return self.rng.choice(free) if free else None

    def _inside(self, cell: Cell) -> bool:
        """True if `cell` is on the grid."""
        return 0 <= cell[0] < self.cols and 0 <= cell[1] < self.rows

    def _clamp(self, cell: Cell) -> Cell:
        """Nearest cell on the grid."""
        return (min(max(cell[0], 0), self.cols - 1), min(max(cell[1], 0), self.rows - 1))
