"""Punto de entrada: muestra la cámara y superpone una imagen según el gesto detectado."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .overlay import blend, fit_height, load_gesture_images
from .placeholders import ensure_placeholders
from .recognizer import GESTURES, GestureDetector

WINDOW = "Gesture Cam"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def parse_args() -> argparse.Namespace:
    """Opciones de línea de comandos."""
    p = argparse.ArgumentParser(description="Cámara que muestra imágenes según tus gestos.")
    p.add_argument("--camera", type=int, default=0, help="Índice de la cámara (default 0)")
    p.add_argument("--images", type=Path, default=PROJECT_ROOT / "images", help="Carpeta de imágenes")
    p.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "gesture_recognizer.task")
    p.add_argument("--min-score", type=float, default=0.6, help="Confianza mínima del gesto (0-1)")
    p.add_argument("--size", type=float, default=0.35, help="Alto de la imagen relativo al video")
    p.add_argument("--windowed", action="store_true", help="Arrancar en ventana, no pantalla completa")
    return p.parse_args()


def set_fullscreen(on: bool) -> None:
    """Cambia la ventana entre pantalla completa y modo ventana."""
    mode = cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, mode)


def draw_debug(frame, detection) -> None:
    """Dibuja los puntos de las manos y el gesto crudo con su confianza."""
    h, w = frame.shape[:2]
    for hand in detection.hands:
        for x, y in hand:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1, cv2.LINE_AA)
    label = f"{detection.gesture or '-'} ({detection.score:.2f})"
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    """Abre la cámara y ejecuta el bucle captura → detección → superposición."""
    args = parse_args()

    created = ensure_placeholders(args.images, GESTURES)
    if created:
        print(f"Imágenes de ejemplo creadas en {args.images}: {', '.join(created)}")
    images = load_gesture_images(args.images, GESTURES)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = GestureDetector(args.model, min_score=args.min_score)
    stabilizer = GestureStabilizer()
    # Cache del escalado: redimensionar cada frame la misma imagen es trabajo inútil.
    scaled: dict[tuple[str, int], object] = {}

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fullscreen = not args.windowed
    set_fullscreen(fullscreen)
    debug = False
    start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("La cámara dejó de enviar imágenes.")
                break
            # Espejo para que moverse se sienta natural, como en un espejo real.
            frame = cv2.flip(frame, 1)

            now = time.monotonic()
            detection = detector.detect(frame, int((now - start) * 1000))
            gesture = stabilizer.update(detection.gesture, now)

            if gesture in images:
                fh, fw = frame.shape[:2]
                target_h = int(fh * args.size)
                key = (gesture, target_h)
                if key not in scaled:
                    scaled[key] = fit_height(images[gesture], target_h)
                img = scaled[key]
                ih, iw = img.shape[:2]
                opacity = min(1.0, (now - stabilizer.active_since) / 0.2)
                blend(frame, img, (fw - iw) // 2, (fh - ih) // 2, opacity)

            if debug:
                draw_debug(frame, detection)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                set_fullscreen(fullscreen)
            elif key == ord("d"):
                debug = not debug
            # Cerrar con la X de la ventana también debe terminar el programa.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
