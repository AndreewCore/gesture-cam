"""Tests for the mode menu's selection logic (drawing on a blank frame, no camera)."""

import cv2
import numpy as np

from gesture_cam.menu import Menu, MenuOption

OPTIONS = [MenuOption("a", "Alpha", "first", "1"), MenuOption("b", "Beta", "second", "2")]


def drawn_menu(disabled=None):
    """Menu already drawn once on a 1280x720 frame so its cards have positions."""
    menu = Menu(OPTIONS)
    menu.disabled = disabled or {}
    menu.draw(np.zeros((720, 1280, 3), np.uint8))
    return menu


def card_center(menu, i):
    """Pixel at the middle of card `i`."""
    x0, y0, x1, y1 = menu._rects[i]
    return (x0 + x1) // 2, (y0 + y1) // 2


def test_number_key_picks_mode():
    """Pressing an option's key selects it."""
    assert drawn_menu().choice(ord("2")) == "b"


def test_other_keys_pick_nothing():
    """Unrelated keys (or no key) don't select anything."""
    menu = drawn_menu()
    assert menu.choice(ord("x")) is None and menu.choice(255) is None


def test_click_on_card_picks_it_once():
    """A click selects the card under the mouse, and only for one frame."""
    menu = drawn_menu()
    menu.on_mouse(cv2.EVENT_LBUTTONUP, *card_center(menu, 0), 0, None)
    assert menu.choice(255) == "a"
    assert menu.choice(255) is None


def test_click_outside_cards_picks_nothing():
    """Clicking the background does nothing."""
    menu = drawn_menu()
    menu.on_mouse(cv2.EVENT_LBUTTONUP, 5, 5, 0, None)
    assert menu.choice(255) is None


def test_disabled_mode_cannot_be_picked():
    """A disabled card ignores both its key and clicks."""
    menu = drawn_menu({"b": "unavailable"})
    assert menu.choice(ord("2")) is None
    menu.on_mouse(cv2.EVENT_LBUTTONUP, *card_center(menu, 1), 0, None)
    assert menu.choice(255) is None
