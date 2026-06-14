# Análisis táctico de fútbol amateur con cámara VEO

Pipeline de visión por computador que extrae métricas tácticas y físicas de partidos de fútbol amateur a partir del vídeo panorámico de la cámara VEO. El sistema cubre toda la cadena: corrección geométrica del vídeo → detección y seguimiento de jugadores → clasificación por equipo → métricas tácticas → dashboard interactivo.

Trabajo de Fin de Grado — Xavier Algarra Pérez, Escuela de Ingeniería, 2026.

---

## Requisitos

- Python 3.10+
- GPU con CUDA recomendada (funciona en CPU pero lento)
- Vídeo VEO en formato **panorámico** (2048×2048 px, no el estándar 1080p)

```bash
pip install -r requirements.txt
```

---

## Estructura del proyecto

```
├── pipeline_core.py          # Motor principal: detección, clasificación, tracking, métricas
├── veo_pipeline.py           # Corrección geométrica específica de VEO (elastic + undistort)
├── veo_app.py                # Aplicación Streamlit (interfaz principal)
├── homografia_interactiva.py # Herramienta de calibración manual de homografía
├── generar_prototipos_v3.py  # Generación semi-manual de prototipos de color por equipo
├── finetune_players_panoramic.py  # Fine-tuning del detector YOLO
├── prepare_dataset.py        # Descarga y prepara el dataset desde Roboflow
├── marcar_tiempos.py         # Herramienta para marcar inicio/fin de cada tiempo
└── notebooks/                # Análisis y diagnóstico
```

---

## Descargar vídeos VEO

Los vídeos se descargan con [`yt-dlp`](https://github.com/yt-dlp/yt-dlp). Instálalo con `pip install yt-dlp`.

### Vídeo panorámico (recomendado para este pipeline)

El formato panorámico solo está disponible durante los **90 días siguientes** a la grabación del partido.

```bash
yt-dlp -f panoramico-2048p -o "data/videos/partido.mp4" "https://app.veo.co/matches/ID-DEL-PARTIDO/"
```

### Vídeo estándar (siempre disponible)

```bash
yt-dlp -f standard-1080p -o "data/videos/partido_std.mp4" "https://app.veo.co/matches/ID-DEL-PARTIDO/"
```

> Sustituye la URL por la del partido en cuestión. 
---

## Uso rápido

### 1. Calibrar la homografía

Antes de procesar un partido hay que calibrar la homografía para el campo concreto. Ejecuta:

```bash
python homografia_interactiva.py --video ruta/al/video.mp4
```

Se abre una ventana con el frame. Haz clic sobre las marcas del campo visibles (líneas, esquinas, círculo central) siguiendo el orden que indica la interfaz. El resultado se guarda como `homografia.npy`.

> Necesitas seleccionar al menos 8 puntos de referencia. Cuantos más marques, menor error de reproyección.

### 2. Generar prototipos de color

Los prototipos definen la firma cromática de cada equipo. Solo hay que generarlos una vez por partido:

```bash
python generar_prototipos_v3.py --video ruta/al/video.mp4
```

El script hace clustering automático de los colores detectados y te muestra los grupos para que valides cuál corresponde a cada equipo. El resultado se guarda como `prototypes_v3.pkl` (y versiones `_day` / `_night` si detecta cambio de iluminación).

### 3. Lanzar el dashboard

```bash
streamlit run veo_app.py
```

Desde la interfaz puedes cargar el vídeo, seleccionar los ficheros de homografía y prototipos, ajustar parámetros y lanzar el procesado. Los resultados se muestran en pestañas: mapa táctico, heatmaps, métricas físicas, análisis por IA y exportación.

### 4. Marcar tiempos del partido

Si solo quieres procesar el primer tiempo (o un intervalo concreto):

```bash
python marcar_tiempos.py --video ruta/al/video.mp4
```

Guarda `tiempos_partido.json` que la app usa automáticamente para recortar el procesado.

---

## Entrenar tu propio detector

El detector está basado en YOLOv8. Los datasets de entrenamiento están disponibles en Roboflow:

- **Dataset de jugadores** (vídeo estándar VEO): [analisis-tactico-futbol](https://universe.roboflow.com/javiers-workspace-osouu/analisis-tactico-futbol)
- **Dataset panorámico** (frames corregidos): [panoramic-cam](https://universe.roboflow.com/javiers-workspace-osouu/panoramic-cam)

Para descargar y preparar el dataset:

```bash
python prepare_dataset.py
```

Para lanzar el fine-tuning desde pesos preentrenados:

```bash
python finetune_players_panoramic.py
```

Edita las rutas y parámetros de entrenamiento al principio del script antes de ejecutarlo.

---

## Notebooks

| Notebook | Descripción |
|----------|-------------|
| `veo_banyoles_pipeline.ipynb` | Demo completa del pipeline sobre el partido de Banyoles |
| `analisis_formacion.ipynb` | Extracción de formación táctica (4-3-3, etc.) desde CSV de tracking |
| `analisis_ids.ipynb` | Diagnóstico de fragmentación de IDs y parámetros del tracker |
| `analisis_pkl.ipynb` | Inspector y tuner de prototipos de color |
| `homography_field.ipynb` | Verificación de la homografía con overlay del modelo FIFA |
| `clasificacion_pipeline.ipynb` | Pipeline de clasificación con análisis de márgenes y unknowns |
| `diagnostico_tracking.ipynb` | Diagnóstico de trayectorias, gaps y duración de pistas |
| `detections_viewer.ipynb` | Visor frame a frame de detecciones y heatmaps |
| `comparacion_trackers.ipynb` | Comparación ByteTrack vs StrongKalmanTrackerV3 |

---

## Modelos entrenados

El modelo de balón (`runs/detect/modelo_ball_v33/weights/best.pt`) está incluido en el repositorio (20 MB).

El modelo de jugadores (`modelo_players_v24_panoramic2`) pesa 126 MB y supera el límite de GitHub. Para obtenerlo:
- Entrénalo desde cero con los datasets de Roboflow y el script `finetune_players_panoramic.py`
- O contacta con el autor para recibirlo directamente

---

## Cómo funciona

```
Vídeo VEO panorámico (2048×2048)
        │
        ▼
Corrección elástica del codec VEO
        │
        ▼
Undistorsión racional (8 coeficientes, extraídos del shader GLSL)
        │
        ▼
Detección YOLOv8 en cada semi-frame (cam A y cam B)
        │
        ▼
Proyección a coordenadas métricas del campo (homografía)
        │
        ▼
Fusión bicámara + NMS global en espacio métrico
        │
        ▼
Clasificación por equipo (histograma de tono HSV + distancia Hellinger)
        │
        ▼
Tracking StrongKalmanTrackerV3 en coordenadas métricas
        │
        ▼
Métricas tácticas y físicas → Dashboard Streamlit + informe IA
```

---

## Licencia

MIT
