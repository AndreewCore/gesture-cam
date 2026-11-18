"""Punto de entrada: muestra la cámara y superpone imágenes según gestos de mano y movimientos del cuerpo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .movements import BODY_MOVES, BodyMoveClassifier
from .overlay import blend, draw_skeleton, fit_height, load_gesture_images
from .placeholders import ensure_placeholders
from .pose import POSE_CONNECTIONS, PoseDetector
from .recognizer import HAND_GESTURES, GestureDetector

WINDOW = "Gesture Cam"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FADE_SECONDS = 0.2


class GestureStabilizer:
    """Filtra parpadeos: exige varios frames seguidos para activar y retiene tras perderlo."""

    def __init__(self, frames_to_activate: int = 5, hold_seconds: float = 0.4):
        """Configura cuántos frames confirman un gesto y cuánto se retiene al perderlo."""
        self.frames_to_activate = frames_to_activate
        self.hold_seconds = hold_seconds
        self._candidate: str | None = None
        self._count = 0
        self.active: str | None = None
        self.active_since = 0.0
        self._last_seen = 0.0

    def update(self, gesture: str | None, now: float) -> str | None:
        """Recibe el gesto crudo del frame y devuelve el gesto estable a mostrar."""
        if gesture == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = gesture, 1

        if gesture is not None and gesture == self.active:
            self._last_seen = now
        elif gesture is not None and self._count >= self.frames_to_activate:
            self.active, self.active_since, self._last_seen = gesture, now, now
        elif self.active is not None and now - self._last_seen > self.hold_seconds:
            self.active = None
        return self.active


class ImageLayer:
    """Una capa de imagen (mano o cuerpo): escala, cachea y dibuja la imagen activa."""

    def __init__(self, images: dict[str, np.ndarray], rel_height: float, anchor: str):
        """`anchor` es 'center' o 'top'; `rel_height` es el alto relativo al frame."""
        self.images = images
        self.rel_height = rel_height
        self.anchor = anchor
        # Redimensionar cada frame la misma imagen es trabajo inútil.
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def draw(self, frame: np.ndarray, name: str | None, since: float, now: float) -> None:
        """Dibuja la imagen de `name` (si existe) con fade-in desde `since`."""
        if name not in self.images:
            return
        fh, fw = frame.shape[:2]
        target_h = int(fh * self.rel_height)
        key = (name, target_h)
        if key not in self._cache:
            self._cache[key] = fit_height(self.images[name], target_h)
        img = self._cache[key]
        ih, iw = img.shape[:2]
        x = (fw - iw) // 2
        y = (fh - ih) // 2 if self.anchor == "center" else int(fh * 0.03)
        blend(frame, img, x, y, min(1.0, (now - since) / FADE_SECONDS))


def parse_args() -> argparse.Namespace:
    """Opciones de línea de comandos."""
    p = argparse.ArgumentParser(description="Cámara que muestra imágenes según tus gestos y movimientos.")
    p.add_argument("--camera", type=int, default=0, help="Índice de la cámara (default 0)")
    p.add_argument("--images", type=Path, default=PROJECT_ROOT / "images", help="Carpeta de imágenes")
    p.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "gesture_recognizer.task")
    p.add_argument("--pose-model", type=Path, default=PROJECT_ROOT / "models" / "pose_landmarker_full.task")
    p.add_argument("--min-score", type=float, default=0.6, help="Confianza mínima del gesto (0-1)")
    p.add_argument("--size", type=float, default=0.35, help="Alto de la imagen de mano relativo al video")
    p.add_argument("--body-size", type=float, default=0.25, help="Alto de la imagen de cuerpo relativo al video")
    p.add_argument("--no-hands", action="store_true", help="Desactivar gestos de mano")
    p.add_argument("--no-pose", action="store_true", help="Desactivar esqueleto y movimientos del cuerpo")
    p.add_argument("--windowed", action="store_true", help="Arrancar en ventana, no pantalla completa")
    return p.parse_args()


def set_fullscreen(on: bool) -> None:
    """Cambia la ventana entre pantalla completa y modo ventana."""
    mode = cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, mode)


def draw_debug(frame, hand_detection, body_move: str | None) -> None:
    """Dibuja los puntos de las manos y los resultados crudos de mano y cuerpo."""
    h, w = frame.shape[:2]
    lines = []
    if hand_detection is not None:
        for hand in hand_detection.hands:
            for x, y in hand:
                cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1, cv2.LINE_AA)
        lines.append(f"mano: {hand_detection.gesture or '-'} ({hand_detection.score:.2f})")
    lines.append(f"cuerpo: {body_move or '-'}")
    for i, text in enumerate(lines):
        org = (20, h - 30 - 35 * (len(lines) - 1 - i))
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    """Abre la cámara y ejecuta el bucle captura → detección (mano + cuerpo) → superposición."""
    args = parse_args()
    if args.no_hands and args.no_pose:
        raise SystemExit("--no-hands y --no-pose a la vez dejan la app sin nada que detectar.")

    names = HAND_GESTURES + BODY_MOVES
    created = ensure_placeholders(args.images, names)
    if created:
        print(f"Imágenes de ejemplo creadas en {args.images}: {', '.join(created)}")
    images = load_gesture_images(args.images, names)
    hand_layer = ImageLayer({k: v for k, v in images.items() if k in HAND_GESTURES}, args.size, "center")
    body_layer = ImageLayer({k: v for k, v in images.items() if k in BODY_MOVES}, args.body_size, "top")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hand_detector = None if args.no_hands else GestureDetector(args.model, min_score=args.min_score)
    pose_detector = None if args.no_pose else PoseDetector(args.pose_model)
    hand_stab = GestureStabilizer()
    # Menos frames para activar: los eventos de movimiento ya vienen retenidos por el clasificador.
    body_stab = GestureStabilizer(frames_to_activate=3, hold_seconds=0.4)
    classifier = BodyMoveClassifier()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fullscreen = not args.windowed
    set_fullscreen(fullscreen)
    debug, show_skeleton = False, True
    start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("La cámara dejó de enviar imágenes.")
                break
            # Espejo para que moverse se sienta natural, como en un espejo real.
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            now = time.monotonic()
            ts = int((now - start) * 1000)

            hand_det = hand_detector.detect(frame, ts) if hand_detector else None
            hand = hand_stab.update(hand_det.gesture if hand_det else None, now)

            landmarks = pose_detector.detect(frame, ts) if pose_detector else None
            raw_move = classifier.classify(landmarks, now, aspect=fw / fh) if pose_detector else None
            body = body_stab.update(raw_move, now)

            # El esqueleto va debajo de las imágenes para no taparlas.
            if show_skeleton and landmarks is not None:
                draw_skeleton(frame, landmarks, POSE_CONNECTIONS)
            body_layer.draw(frame, body, body_stab.active_since, now)
            hand_layer.draw(frame, hand, hand_stab.active_since, now)

            if debug:
                draw_debug(frame, hand_det, raw_move)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                set_fullscreen(fullscreen)
            elif key == ord("d"):
                debug = not debug
            elif key == ord("s"):
                show_skeleton = not show_skeleton
            # Cerrar con la X de la ventana también debe terminar el programa.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if hand_detector:
            hand_detector.close()
        if pose_detector:
            pose_detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
