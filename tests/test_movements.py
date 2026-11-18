"""Pruebas de las reglas de movimiento con esqueletos sintéticos (sin cámara ni MediaPipe)."""

from gesture_cam import movements as m
from gesture_cam.movements import BodyMoveClassifier

FPS = 30
DT = 1 / FPS


def standing(**overrides):
    """Esqueleto de 33 puntos de pie, brazos caídos; `overrides` = {índice: (x, y)}."""
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
    """Desplaza todo el esqueleto verticalmente (dy negativo = sube)."""
    return [(x, y + dy, v) for x, y, v in landmarks]


def squatting():
    """Esqueleto en sentadilla: tronco más bajo y rodillas hacia fuera."""
    lm = standing()
    upper = {m.NOSE, m.L_SHOULDER, m.R_SHOULDER, m.L_ELBOW, m.R_ELBOW, m.L_WRIST, m.R_WRIST, m.L_HIP, m.R_HIP}
    lm = [(x, y + 0.15, v) if i in upper else (x, y, v) for i, (x, y, v) in enumerate(lm)]
    lm[m.L_KNEE] = (0.33, 0.8, 0.99)
    lm[m.R_KNEE] = (0.67, 0.8, 0.99)
    return lm


def run(frames, start=0.0):
    """Pasa una secuencia de frames por un clasificador nuevo y devuelve todos los resultados."""
    clf = BodyMoveClassifier()
    return [clf.classify(lm, start + i * DT) for i, lm in enumerate(frames)]


def test_neutral_is_nothing():
    """De pie y quieto no debe disparar ningún movimiento."""
    assert run([standing()] * 30)[-1] is None


def test_no_person_is_nothing():
    """Sin persona en cuadro no hay movimiento."""
    assert BodyMoveClassifier().classify(None, 0.0) is None


def test_arms_up():
    """Ambas muñecas sobre la nariz."""
    lm = standing()
    lm[m.L_WRIST] = (0.4, 0.05, 0.99)
    lm[m.R_WRIST] = (0.6, 0.05, 0.99)
    assert run([lm])[-1] == "arms_up"


def test_t_pose():
    """Brazos extendidos a la altura de los hombros."""
    lm = standing()
    lm[m.L_ELBOW], lm[m.R_ELBOW] = (0.25, 0.3, 0.99), (0.75, 0.3, 0.99)
    lm[m.L_WRIST], lm[m.R_WRIST] = (0.1, 0.3, 0.99), (0.9, 0.3, 0.99)
    assert run([lm])[-1] == "t_pose"


def test_hands_on_hips():
    """Muñecas en las caderas con codos abiertos."""
    lm = standing()
    lm[m.L_ELBOW], lm[m.R_ELBOW] = (0.3, 0.45, 0.99), (0.7, 0.45, 0.99)
    lm[m.L_WRIST], lm[m.R_WRIST] = (0.44, 0.58, 0.99), (0.56, 0.58, 0.99)
    assert run([lm])[-1] == "hands_on_hips"


def test_squat():
    """Rodillas dobladas en ambas piernas."""
    assert run([squatting()])[-1] == "squat"


def test_low_visibility_ignored():
    """Puntos poco visibles no deben activar reglas."""
    lm = standing()
    lm[m.L_WRIST] = (0.4, 0.05, 0.1)
    lm[m.R_WRIST] = (0.6, 0.05, 0.1)
    assert run([lm])[-1] is None


def test_wave():
    """Mano sobre el hombro moviéndose de lado a lado."""
    frames = []
    for i in range(int(1.2 * FPS)):
        lm = standing()
        # Muñeca sobre el hombro yendo y viniendo cada 0.2 s.
        x = 0.65 if (i // 6) % 2 == 0 else 0.8
        lm[m.R_WRIST] = (x, 0.2, 0.99)
        lm[m.R_ELBOW] = (0.66, 0.35, 0.99)
        frames.append(lm)
    assert "wave" in run(frames)


def test_raised_hand_still_is_not_wave():
    """Una mano levantada pero quieta no es un saludo."""
    lm = standing()
    lm[m.R_WRIST] = (0.7, 0.2, 0.99)
    assert "wave" not in run([lm] * 45)


def test_jump():
    """Todo el cuerpo sube de golpe respecto a su posición reciente."""
    frames = [standing()] * 20 + [shifted(standing(), -0.1)] * 3
    assert run(frames)[-1] == "jump"


def test_jump_is_held_briefly():
    """El salto se sigue reportando un instante tras aterrizar."""
    frames = [standing()] * 20 + [shifted(standing(), -0.1)] * 3 + [standing()] * 5
    assert run(frames)[-1] == "jump"


def test_standing_up_from_squat_is_not_jump():
    """Levantarse de una sentadilla sube la cadera pero no los pies: no es salto."""
    frames = [squatting()] * 20 + [standing()] * 5
    assert "jump" not in run(frames)
