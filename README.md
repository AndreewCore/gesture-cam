# Gesture Cam

Abre tu cámara y superpone imágenes según lo que hagas:

- **Gestos de mano** (pulgar arriba, V, puño…) → imagen en el **centro**.
- **Movimientos del cuerpo** (brazos arriba, T, saludo, salto…) → imagen **arriba**, y tu **esqueleto** dibujado sobre el video.

Ambos funcionan a la vez: puedes tener una imagen de mano y otra de cuerpo en pantalla al mismo tiempo.

## Ejecutar

```bash
uv sync
uv run gesture-cam
```

La primera vez descarga los modelos de MediaPipe a `models/` y crea imágenes de ejemplo en `images/`.

### Opciones

| Opción | Efecto |
|---|---|
| `--camera 1` | Usar otra cámara |
| `--windowed` | Arrancar en ventana |
| `--min-score 0.7` | Confianza mínima para gestos de mano |
| `--size 0.5` | Alto de la imagen de mano (relativo al video) |
| `--body-size 0.3` | Alto de la imagen de cuerpo |
| `--no-hands` / `--no-pose` | Desactivar una de las dos detecciones (si va lento) |
| `--images otra/carpeta` | Carpeta de imágenes alternativa |

### Teclas

| Tecla | Acción |
|---|---|
| `Q` / `Esc` | Salir |
| `F` | Pantalla completa on/off |
| `S` | Mostrar/ocultar esqueleto |
| `D` | Modo debug (puntos de la mano y detección cruda de mano y cuerpo) |

## Imágenes

Pon tus imágenes en `images/` con estos nombres (`.png`, `.webp`, `.jpg` o `.jpeg`; PNG con transparencia queda mejor). Si borras una, se regenera un ejemplo al arrancar.

**Mano (centro)**

| Gesto | Archivo |
|---|---|
| 👍 Pulgar arriba | `thumb_up.png` |
| 👎 Pulgar abajo | `thumb_down.png` |
| ✌️ Victoria / V | `victory.png` |
| ✋ Palma abierta | `open_palm.png` |
| ✊ Puño | `closed_fist.png` |
| ☝️ Índice arriba | `pointing_up.png` |
| 🤟 Te quiero | `iloveyou.png` |

**Cuerpo (arriba)**

| Movimiento | Archivo |
|---|---|
| 🙌 Ambos brazos arriba | `arms_up.png` |
| ✝️ Pose en T | `t_pose.png` |
| 🧍 Manos en la cintura | `hands_on_hips.png` |
| 🏋️ Sentadilla | `squat.png` |
| 👋 Saludo (mano moviéndose sobre el hombro) | `wave.png` |
| 🦘 Salto | `jump.png` |

## Cómo funciona

Cada frame de la cámara (en espejo) pasa por dos modelos preentrenados de [MediaPipe](https://ai.google.dev/edge/mediapipe), que se ejecutan localmente:

1. **Gesture Recognizer** (`recognizer.py`): localiza hasta 2 manos (21 puntos cada una) y clasifica su forma en 7 gestos con una confianza. Gana el gesto con mayor confianza por encima de `--min-score`.
2. **Pose Landmarker** (`pose.py`): devuelve 33 puntos del cuerpo (nariz, hombros, codos, muñecas, caderas, rodillas, tobillos…) con su visibilidad. Con ellos se dibuja el esqueleto.

Los movimientos del cuerpo (`movements.py`) son **reglas geométricas** sobre esos puntos, medidas en "largos de torso" para que funcionen cerca o lejos de la cámara:

- **Poses estáticas** (un frame): muñecas sobre la nariz → brazos arriba; muñecas a la altura de los hombros y lejos → T; muñecas junto a la cadera con codos abiertos → cintura; ángulo cadera-rodilla-tobillo < 120° → sentadilla.
- **Movimientos** (historial de ~1.5 s): una muñeca sobre el hombro que cambia de dirección 3+ veces → saludo; caderas y pies que suben de golpe respecto a su posición reciente → salto (levantarse de una sentadilla no cuenta porque los pies no suben).
- Prioridad: salto > saludo > poses estáticas.

Para evitar parpadeos, cada canal (mano y cuerpo) tiene su propio **estabilizador**: el resultado debe repetirse unos frames seguidos para activarse y se mantiene un instante tras desaparecer. Las imágenes se mezclan con transparencia sobre el video (`overlay.py`).

## Tests

```bash
uv run pytest
```

Prueban las reglas de movimiento con esqueletos sintéticos, sin cámara.
