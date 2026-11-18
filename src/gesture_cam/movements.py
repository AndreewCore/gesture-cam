"""Reconoce movimientos del cuerpo a partir de los landmarks de pose, con reglas geométricas.

No depende de MediaPipe: recibe listas de (x, y, visibilidad) normalizadas, así que se
puede probar con landmarks sintéticos.
"""

from __future__ import annotations

import math
import statistics
from collections import deque

BODY_MOVES = ("arms_up", "t_pose", "hands_on_hips", "squat", "wave", "jump")

# Índices del modelo de pose de MediaPipe (33 landmarks).
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
    """Landmarks de un frame en unidades comparables, con helpers de visibilidad."""

    def __init__(self, landmarks, aspect: float = 1.0):
        """`aspect` = ancho/alto del frame; corrige que x e y estén normalizados distinto."""
        self.pts: list[Point] = [(x * aspect, y) for x, y, _ in landmarks]
        self.vis = [v for _, _, v in landmarks]

    def visible(self, *idx: int) -> bool:
        """True si todos los landmarks indicados tienen visibilidad suficiente."""
        return all(self.vis[i] > MIN_VISIBILITY for i in idx)

    def __getitem__(self, i: int) -> Point:
        """Acceso directo al punto `i`."""
        return self.pts[i]

    def mid(self, a: int, b: int) -> Point:
        """Punto medio entre dos landmarks."""
        (ax, ay), (bx, by) = self.pts[a], self.pts[b]
        return ((ax + bx) / 2, (ay + by) / 2)

    def torso_len(self) -> float:
        """Distancia hombros-caderas: la regla con la que se miden todas las demás distancias."""
        return max(_dist(self.mid(L_SHOULDER, R_SHOULDER), self.mid(L_HIP, R_HIP)), 1e-6)


def _dist(a: Point, b: Point) -> float:
    """Distancia euclidiana entre dos puntos."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a: Point, b: Point, c: Point) -> float:
    """Ángulo en grados en el vértice `b` formado por a-b-c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    if n == 0:
        return 180.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
    return math.degrees(math.acos(cos))


def _torso_ok(b: Body) -> bool:
    """Sin hombros y caderas visibles no hay escala fiable para las reglas."""
    return b.visible(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)


def is_arms_up(b: Body) -> bool:
    """Ambas muñecas por encima de la nariz."""
    if not b.visible(NOSE, L_WRIST, R_WRIST):
        return False
    # En imagen, y crece hacia abajo: "arriba" es y menor.
    return b[L_WRIST][1] < b[NOSE][1] and b[R_WRIST][1] < b[NOSE][1]


def is_t_pose(b: Body) -> bool:
    """Brazos extendidos en horizontal: muñecas y codos a la altura de los hombros y lejos del cuerpo."""
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
    """Muñecas junto a las caderas con los codos abiertos hacia fuera."""
    if not (_torso_ok(b) and b.visible(L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)):
        return False
    t = b.torso_len()
    cx = b.mid(L_SHOULDER, R_SHOULDER)[0]
    half_shoulders = abs(b[L_SHOULDER][0] - b[R_SHOULDER][0]) / 2
    for w, h, e in ((L_WRIST, L_HIP, L_ELBOW), (R_WRIST, R_HIP, R_ELBOW)):
        if _dist(b[w], b[h]) > 0.45 * t:
            return False
        # Con los brazos colgando los codos quedan pegados al cuerpo; en jarra salen hacia fuera.
        if abs(b[e][0] - cx) < half_shoulders + 0.15 * t:
            return False
    return True


def is_squat(b: Body) -> bool:
    """Ambas rodillas dobladas (ángulo cadera-rodilla-tobillo menor a ~120°)."""
    if not b.visible(L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE):
        return False
    return (
        _angle(b[L_HIP], b[L_KNEE], b[L_ANKLE]) < 120
        and _angle(b[R_HIP], b[R_KNEE], b[R_ANKLE]) < 120
    )


class MotionTracker:
    """Detecta movimientos que necesitan historia (saludo y salto) a partir de varios frames."""

    WAVE_WINDOW = 1.5  # s
    WAVE_REVERSALS = 3
    WAVE_AMPLITUDE = 0.15  # en largos de torso
    JUMP_WINDOW = 1.5  # s
    JUMP_RISE = 0.25  # en largos de torso
    FEET_RISE = 0.15

    def __init__(self):
        """Inicializa el historial vacío de muñecas y caderas."""
        self._wrists = {L_WRIST: deque(), R_WRIST: deque()}
        self._hips: deque[tuple[float, float, float | None]] = deque()  # (t, cadera_y, pies_y)

    def update(self, b: Body, now: float) -> str | None:
        """Registra el frame y devuelve 'jump', 'wave' o None."""
        if not _torso_ok(b):
            self.reset()
            return None
        t = b.torso_len()
        jump = self._update_jump(b, t, now)
        wave = self._update_wave(b, t, now)
        return "jump" if jump else "wave" if wave else None

    def reset(self) -> None:
        """Olvida la historia, p. ej. cuando la persona sale de cuadro."""
        for d in self._wrists.values():
            d.clear()
        self._hips.clear()

    def _update_wave(self, b: Body, t: float, now: float) -> bool:
        """Saludo: una muñeca sobre su hombro que cambia de dirección en x varias veces."""
        cx = b.mid(L_SHOULDER, R_SHOULDER)[0]
        waving = False
        for w, s in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
            hist = self._wrists[w]
            if not b.visible(w) or b[w][1] > b[s][1]:
                hist.clear()
                continue
            # Posición relativa al cuerpo y en largos de torso: inmune a acercarse o moverse de lado.
            hist.append((now, (b[w][0] - cx) / t))
            while hist and now - hist[0][0] > self.WAVE_WINDOW:
                hist.popleft()
            if _count_reversals([x for _, x in hist], self.WAVE_AMPLITUDE) >= self.WAVE_REVERSALS:
                waving = True
        return waving

    def _update_jump(self, b: Body, t: float, now: float) -> bool:
        """Salto: caderas (y pies, si se ven) suben rápido respecto a su posición reciente."""
        # Se guarda y crudo y se divide por el torso actual al comparar: dividir cada muestra
        # por su propio torso falsea la subida si el torso cambia de tamaño aparente.
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
        # Levantarse de una sentadilla también sube la cadera; exigir que suban los pies lo distingue.
        feet = [f for _, _, f in self._hips if f is not None]
        if feet_y is not None and len(feet) >= 5:
            return (statistics.median(feet) - feet_y) / t >= self.FEET_RISE
        return True


def _count_reversals(xs: list[float], amplitude: float) -> int:
    """Cuenta cambios de dirección en `xs` que superan `amplitude` (ignora temblores)."""
    if len(xs) < 3:
        return 0
    reversals, direction = 0, 0
    lo = hi = extreme = xs[0]
    for x in xs[1:]:
        if direction == 0:
            # Aún sin dirección: esperamos el primer desplazamiento claro hacia algún lado.
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
    """Combina reglas estáticas y de movimiento en un único resultado por frame."""

    MOTION_HOLD = 0.6  # s: un salto dura un instante, se retiene para que se vea la imagen.

    def __init__(self):
        """Crea el tracker de movimiento y el estado del último evento."""
        self.motion = MotionTracker()
        self._event: str | None = None
        self._event_time = -math.inf

    def classify(self, landmarks, now: float, aspect: float = 1.0) -> str | None:
        """Devuelve el movimiento del frame con prioridad salto > saludo > poses estáticas."""
        if landmarks is None:
            self.motion.reset()
            return None
        b = Body(landmarks, aspect)

        event = self.motion.update(b, now)
        if event is not None:
            self._event, self._event_time = event, now
        if self._event is not None and now - self._event_time <= self.MOTION_HOLD:
            return self._event

        # La sentadilla va antes que manos en caderas: al agacharse se pueden cumplir ambas.
        for name, rule in (
            ("arms_up", is_arms_up),
            ("t_pose", is_t_pose),
            ("squat", is_squat),
            ("hands_on_hips", is_hands_on_hips),
        ):
            if rule(b):
                return name
        return None
