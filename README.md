# Gesture Cam

Abre tu cámara y, cuando haces un gesto con la mano, muestra una imagen encima del video.

## Ejecutar

```bash
uv sync
uv run gesture-cam
```

La primera vez descarga el modelo de MediaPipe a `models/` y crea imágenes de ejemplo en `images/`.

Opciones: `--camera 1`, `--windowed`, `--min-score 0.7`, `--size 0.5`, `--images otra/carpeta`.

### Teclas

| Tecla | Acción |
|---|---|
| `Q` / `Esc` | Salir |
| `F` | Pantalla completa on/off |
| `D` | Modo debug (puntos de la mano y gesto detectado) |

## Gestos e imágenes

Pon tus imágenes en `images/` con estos nombres (`.png`, `.webp`, `.jpg` o `.jpeg`; PNG con transparencia queda mejor):

| Gesto | Archivo |
|---|---|
| 👍 Pulgar arriba | `thumb_up.png` |
| 👎 Pulgar abajo | `thumb_down.png` |
| ✌️ Victoria / V | `victory.png` |
| ✋ Palma abierta | `open_palm.png` |
| ✊ Puño | `closed_fist.png` |
| ☝️ Índice arriba | `pointing_up.png` |
| 🤟 Te quiero | `iloveyou.png` |

Si borras una imagen, se regenera un ejemplo al volver a arrancar.
