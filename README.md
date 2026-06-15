# Análisis táctico de fútbol amateur con cámara VEO

Pipeline de visión por computador que extrae métricas tácticas y físicas de partidos de fútbol amateur a partir del vídeo panorámico de la cámara VEO. El sistema cubre toda la cadena: corrección geométrica del vídeo → detección y seguimiento de jugadores → clasificación por equipo → métricas tácticas → dashboard interactivo.

Trabajo de Fin de Grado — Xavier Algarra Pérez, Escuela de Ingeniería, 2026.

---

## Contenidos

1. [Requisitos e instalación](#requisitos-e-instalación)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Flujo completo del pipeline](#flujo-completo-del-pipeline)
4. [Paso 0 — Descargar los vídeos VEO](#paso-0--descargar-los-vídeos-veo)
5. [Paso 1 — Marcar tiempos del partido](#paso-1--marcar-tiempos-del-partido)
6. [Paso 2 — Calibración de la homografía](#paso-2--calibración-de-la-homografía)
7. [Paso 3 — Generar prototipos de color](#paso-3--generar-prototipos-de-color)
8. [Paso 4 — Dashboard y procesado](#paso-4--dashboard-y-procesado)
9. [Adaptación a campos de distintas dimensiones](#adaptación-a-campos-de-distintas-dimensiones)
10. [Entrenar el detector YOLO](#entrenar-el-detector-yolo)
11. [Modelos incluidos y disponibles](#modelos-incluidos-y-disponibles)
12. [Notebooks de análisis y diagnóstico](#notebooks-de-análisis-y-diagnóstico)
13. [Referencia de argumentos por script](#referencia-de-argumentos-por-script)
14. [Cómo funciona (arquitectura interna)](#cómo-funciona-arquitectura-interna)

---

## Requisitos e instalación

**Requisitos del sistema:**

- Python 3.10 o superior
- GPU con CUDA recomendada (funciona en CPU pero el procesado es 8–10× más lento)
- Vídeo VEO en formato **panorámico** (2048×2048 px); el formato estándar 1080p no es compatible con el pipeline de corrección geométrica

**Instalación:**

```bash
pip install -r requirements.txt
```

Para CUDA (si tienes GPU NVIDIA), sustituye la línea de torch por:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Dependencia opcional — análisis narrativo con IA:**

El informe táctico automático requiere la API de Anthropic:

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."   # Linux/Mac
set ANTHROPIC_API_KEY=sk-ant-...        # Windows CMD
```

**Dependencia opcional — SAM2 (segmentación para prototipos de color):**

SAM2 mejora la calidad de los prototipos de color al aislar la silueta del jugador. No es obligatorio:

```bash
# Descargar el modelo tiny (más rápido, suficiente para prototipos):
# https://github.com/ultralytics/assets/releases — sam2_t.pt (~38 MB)
```

---

## Estructura del proyecto

```
├── pipeline_core.py          # Motor principal: detección, clasificación, tracking, métricas
├── veo_pipeline.py           # Corrección geométrica específica de VEO (elastic + undistort)
├── veo_app.py                # Aplicación Streamlit (interfaz principal)
├── homografia_interactiva.py # Herramienta de calibración manual de homografía
├── generar_prototipos_v3.py  # Generación semi-manual de prototipos de color por equipo
├── marcar_tiempos.py         # Herramienta para marcar inicio/fin de cada tiempo
├── finetune_players_panoramic.py  # Fine-tuning del detector YOLO
├── prepare_dataset.py        # Descarga y prepara el dataset desde Roboflow
├── requirements.txt
├── runs/
│   └── detect/
│       └── modelo_ball_v33/weights/best.pt   # modelo de balón incluido (20 MB)
└── notebooks/
    ├── veo_banyoles_pipeline.ipynb
    ├── analisis_formacion.ipynb
    ├── analisis_ids.ipynb
    ├── analisis_pkl.ipynb
    ├── homography_field.ipynb
    ├── clasificacion_pipeline.ipynb
    ├── diagnostico_tracking.ipynb
    ├── detections_viewer.ipynb
    └── comparacion_trackers.ipynb
```

---

## Flujo completo del pipeline

```
Vídeo VEO panorámico (2048×2048)
        │
        ▼
[Paso 0] Descarga con yt-dlp
        │
        ▼
[Paso 1] marcar_tiempos.py  →  tiempos_partido.json
        │
        ▼
[Paso 2] homografia_interactiva.py (×2, cam A y cam B)
        │  →  data/homography/H_veo_top.npy
        │  →  data/homography/H_veo_bottom.npy
        ▼
[Paso 3] generar_prototipos_v3.py
        │  →  prototypes_v3.pkl  (o _day.pkl / _night.pkl)
        ▼
[Paso 4] streamlit run veo_app.py
        │
        ▼
Corrección geométrica del codec VEO
  ├── Descompresión elástica (AWS ElasticTranscoder inverso)
  └── Undistorsión racional 8 coeficientes (shader GLSL extraído con Spector.js)
        │
        ▼
Detección YOLO en cada semi-frame (cam A y cam B separadas)
        │
        ▼
Proyección píxel → metros en el campo (homografía H_a / H_b)
        │
        ▼
Fusión bicámara + NMS global en espacio métrico
        │
        ▼
Clasificación por equipo (histograma de tono HSV, distancia Hellinger)
        │
        ▼
Tracking StrongKalmanTrackerV3 en coordenadas métricas
        │
        ▼
Métricas tácticas y físicas → Dashboard Streamlit + informe IA
```

---

## Paso 0 — Descargar los vídeos VEO

Los vídeos se descargan con [`yt-dlp`](https://github.com/yt-dlp/yt-dlp).

### Vídeo panorámico (necesario para el pipeline)

El formato panorámico solo está disponible durante los **90 días siguientes** a la grabación.

```bash
yt-dlp -f panorama-2048p -o "data/videos/partido_pan.mp4" "https://app.veo.co/matches/ID-DEL-PARTIDO/"
```

### Vídeo estándar (para marcar tiempos visualmente)

```bash
yt-dlp -f standard-1080p -o "data/videos/partido_std.mp4" "https://app.veo.co/matches/ID-DEL-PARTIDO/"
```

> Necesitas tener acceso al partido en tu cuenta VEO. Sustituye la URL por la del partido concreto.

---

## Paso 1 — Marcar tiempos del partido

Esta herramienta permite identificar exactamente en qué frame empieza y termina cada tiempo, para que el pipeline procese solo los momentos de juego real y excluya el calentamiento y el descanso.

```bash
python marcar_tiempos.py ruta/al/video_standard.mp4
```

Se abre una ventana con el vídeo. Controles:

| Tecla | Acción |
|-------|--------|
| `←` / `→` | Retroceder / avanzar 1 frame |
| `A` / `D` | Retroceder / avanzar ~1 segundo |
| `Q` / `E` | Retroceder / avanzar ~10 segundos |
| Trackbar | Saltar a cualquier frame |
| `1` | Marcar inicio del 1er tiempo |
| `2` | Marcar fin del 1er tiempo |
| `3` | Marcar inicio del 2º tiempo |
| `4` | Marcar fin del 2º tiempo |
| `S` | Guardar en `tiempos_partido.json` |
| `ESC` | Salir |

**Salida:** `tiempos_partido.json` con los frames e instantes en minutos de cada marca. La app Streamlit lo carga automáticamente.

---

## Paso 2 — Calibración de la homografía

La homografía transforma píxeles de la imagen corregida a metros en el campo. Hay que calibrar **dos homografías**: una para la cámara superior (cam A, mitad izquierda del campo) y otra para la inferior (cam B, mitad derecha).

### Calibración desde el vídeo VEO directamente

```bash
# Calibrar la cámara superior (top, campo izquierdo)
python homografia_interactiva.py --half top --frame 3000

# Calibrar la cámara inferior (bottom, campo derecho)
python homografia_interactiva.py --half bottom --frame 3000
```

### Calibración desde una imagen exportada (flujo recomendado)

La app Streamlit puede exportar frames ya corregidos en la resolución exacta del pipeline (paso de calibración dentro de la pestaña "Calibración & Colores"). Una vez descargados:

```bash
python homografia_interactiva.py --imagen camA-partido.png
python homografia_interactiva.py --imagen camB-partido.png
```

### Especificar dimensiones del campo

Si el campo tiene dimensiones distintas a las predeterminadas (105×70,94 m), indícalas con `--longitud` y `--anchura`. Si no se especifican, el script pregunta interactivamente al inicio:

```bash
python homografia_interactiva.py --half top --frame 3000 --longitud 98.0 --anchura 61.0
```

> **Importante:** Las dimensiones pasadas aquí determinan el sistema de referencia del mapa de campo. Deben coincidir exactamente con las que configures en la app Streamlit y en `generar_prototipos_v3.py`.

### Referencia completa de argumentos

| Argumento | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `--imagen` | str | — | Ruta a una imagen PNG/JPG exportada (alternativa al vídeo) |
| `--half` | `top`/`bottom` | `top` | Qué mitad del campo calibrar (se infiere del nombre del archivo si se usa `--imagen`) |
| `--frame` | int | `3000` | Frame del vídeo a usar como referencia |
| `--alpha` | float | `0.45` | Opacidad inicial del overlay del campo |
| `--longitud` | float | — | Longitud del campo en metros (si se omite, pregunta interactivamente) |
| `--anchura` | float | — | Anchura del campo en metros (si se omite, pregunta interactivamente) |

### Controles de la ventana interactiva

| Acción | Control |
|--------|---------|
| Mover punto de control | Click + arrastrar |
| Teleportar punto seleccionado | Ctrl + click |
| Paneo de la vista | Click derecho + arrastrar |
| Zoom | Rueda del ratón |
| Centrar en punto seleccionado | Espacio |
| Seleccionar punto (1-9, 0, a-z) | Teclado numérico/letras |
| Nudge 1 px | Flechas |
| Nudge 10 px | Shift + Flechas |
| Guardar homografía | `S` |
| Reiniciar puntos | `R` |
| Ajustar opacidad overlay | `A` (más) / `Z` (menos) |
| Toggle overlay campo | `F` |
| Toggle rejilla de metros | `G` |
| Salir | `Q` / `ESC` |

**Salida:** el archivo `.npy` se guarda en `data/homography/H_veo_top.npy` (o `_bottom.npy`). Si se usa una imagen como entrada, se guarda junto a ella con el sufijo `_homography.npy`.

**Error de reproyección objetivo:** por debajo de 1 m de error medio es aceptable; por debajo de 0,5 m es bueno. La herramienta lo imprime al guardar.

---

## Paso 3 — Generar prototipos de color

Los prototipos definen la firma cromática de cada clase (equipo local, visitante, porteros, árbitro). Solo hay que generarlos una vez por partido.

### Uso básico

```bash
python generar_prototipos_v3.py \
    --video  data/videos/partido_pan.mp4 \
    --model  runs/detect/modelo_players_v24_panoramic2/weights/best.pt \
    --output prototypes_v3.pkl
```

### Con homografías (permite detectar porteros automáticamente por posición)

```bash
python generar_prototipos_v3.py \
    --video  data/videos/partido_pan.mp4 \
    --model  runs/detect/modelo_players_v24_panoramic2/weights/best.pt \
    --output prototypes_v3.pkl \
    --H_a    data/homography/H_veo_top.npy \
    --H_b    data/homography/H_veo_bottom.npy \
    --field_length 98.0 \
    --field_width  61.0
```

### Con intervalos específicos (evitar calentemiento y descanso)

El argumento `--intervals` acepta el formato `MM:SS-MM:SS` (o `H:MM:SS-H:MM:SS`). Se pueden indicar varios intervalos separados por comas. El sufijo opcional `:L` o `:R` indica en qué dirección ataca el equipo local en ese intervalo:

```bash
python generar_prototipos_v3.py \
    --video  data/videos/partido_pan.mp4 \
    --model  runs/detect/modelo_players_v24_panoramic2/weights/best.pt \
    --output prototypes_v3.pkl \
    --H_a    data/homography/H_veo_top.npy \
    --H_b    data/homography/H_veo_bottom.npy \
    --field_length 98.0 --field_width 61.0 \
    --intervals "17:11-1:06:00:L,1:07:30-1:52:00:R" \
    --conf 0.45 --k 3
```

### Flujo interactivo del script

1. **Pre-scan de iluminación** (sin YOLO, rápido): detecta si hay cambio entre condiciones de día/noche. Si lo hay, genera dos PKLs automáticamente (`_day.pkl` y `_night.pkl`).
2. **Muestreo del vídeo** con YOLO (o carga desde caché `rows_cache.pkl`).
3. **Clustering KMeans** sobre las firmas de tono. El script muestra una galería visual de los clusters y pide etiquetar cada uno: `player_home`, `player_away`, `referee`, o `skip`.
4. **Pools de outliers**: los jugadores que no encajan en ningún cluster principal (porteros, árbitros) se muestran en galerías separadas por posición para etiquetarlos como `gk_home`, `gk_away`, etc.
5. **Validación**: clasifica todas las detecciones con el PKL generado y muestra la distribución de clases.

### Referencia completa de argumentos

| Argumento | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `--video` | str | **obligatorio** | Ruta al vídeo panorámico |
| `--model` | str | **obligatorio** | Ruta al modelo YOLO de jugadores |
| `--output` | str | `prototypes_v3.pkl` | Ruta del PKL de salida |
| `--cache` | str | `rows_cache.pkl` | Ruta del caché de detecciones (reutilizable) |
| `--no_cache` | flag | — | Ignorar caché existente y resampler desde cero |
| `--sample_sec` | int | `180` | Segundos de vídeo a muestrear (si no se usan `--intervals`) |
| `--step` | float | `2.5` | Intervalo entre frames muestreados (segundos) |
| `--conf` | float | `0.45` | Umbral de confianza YOLO para el muestreo |
| `--k` | int | — | Número de clusters forzado. Si se omite, el script sugiere automáticamente el mejor k |
| `--n_top` | int | `20` | Muestras más centrales de cada cluster para calcular el prototipo |
| `--intervals` | str | — | Intervalos de juego. Formato: `START-END[:L\|:R][,...]` |
| `--H_a` | str | — | Homografía cámara A (.npy). Activa split posicional de porteros |
| `--H_b` | str | — | Homografía cámara B (.npy) |
| `--field_length` | float | `105.0` | Longitud del campo en metros |
| `--field_width` | float | `68.0` | Anchura del campo en metros |
| `--home_left` | flag | activado | El equipo local ataca hacia la izquierda del campo (x < field_length/2) en el 1T. Puede sobreescribirse por intervalo con `:L`/`:R` |
| `--sam` | str | `sam2_t.pt` | Modelo SAM2 para segmentación. Cadena vacía para desactivar |

---

## Paso 4 — Dashboard y procesado

```bash
streamlit run veo_app.py
```

Abre el navegador en `http://localhost:8501`. La app guía el proceso en cuatro pasos:

### Paso 1 — Descarga

- Introduce la URL del partido VEO para descargar automáticamente los dos formatos con `yt-dlp`
- O bien carga manualmente los archivos ya descargados

### Paso 2 — Calibración

**Pestaña "Calibración & Colores":**

- Carga las homografías de cada cámara (`.npy`). La app busca automáticamente en `data/calib_alpha1/calib_camA_alpha1_homography.npy` y `calib_camB_alpha1_homography.npy`
- Carga el PKL de prototipos de color (día y, opcionalmente, noche)
- La "costura" entre cámaras se calcula automáticamente desde las homografías
- El botón "Exportar frames para el calibrador" genera y descarga los dos frames corregidos al resolución exacta del pipeline, listos para usar con `homografia_interactiva.py`

**Pestaña "Partido & Tracker":**

- Introduce los metadatos del partido (equipos, competición, fecha, estadio)
- Configura las dimensiones del campo (véase la sección siguiente)
- Elige el algoritmo de tracking y ajusta sus parámetros

### Paso 3 — Procesar

Configura el segmento a analizar, lanza el procesado y observa en tiempo real el mapa táctico y las detecciones activas. Parámetros principales:

| Parámetro | Descripción |
|-----------|-------------|
| Inicio / Duración | Frame de inicio y duración en minutos del segmento a procesar |
| Detección cada N frames | `1` = máxima precisión; `3` = equilibrio; `5` = máxima velocidad |
| Confianza YOLO | Umbral de confianza del detector (0,35–0,80) |
| Largo / Ancho del campo | Dimensiones reales del campo en metros |
| Umbral día/noche | Brillo medio del frame por encima del cual se usa el PKL de día |
| Local defiende portería izquierda en 1T | Orientación del equipo local para calcular correctamente las métricas direccionales |

### Paso 4 — Resultados

Pestañas disponibles:

- **Mapa táctico**: posición media de cada jugador sobre el campo, formación estimada (4-3-3, etc.) y dirección de ataque
- **Heatmaps**: mapa de calor de presencia por equipo, overlays sobre el campo
- **Físico**: distancia total, distancia por tercio (defensivo/medio/ataque), sprints, HSR, velocidad máxima y media por jugador
- **Táctico**: línea defensiva, compacidad, anchura de bloque, territorio, presión, posesión, mapa zonal 3×5, transiciones
- **Análisis IA**: informe táctico generado por Claude Sonnet utilizando todas las métricas calculadas (requiere `ANTHROPIC_API_KEY`)
- **Exportar**: descarga de CSV con registros por frame, estadísticas físicas y tácticas

---

## Adaptación a campos de distintas dimensiones

Este es el aspecto más importante para reutilizar el pipeline en un campo distinto al de Banyoles (98×61 m).

### Dónde se define el tamaño del campo

El tamaño del campo actúa como **sistema de referencia métrico** en el que se expresan todas las posiciones de jugadores y todas las métricas. Hay que configurarlo de forma consistente en tres lugares:

#### 1. Durante la calibración de la homografía

Al ejecutar `homografia_interactiva.py`, el catálogo de puntos de referencia (esquinas, áreas, círculo central) se dibuja en función de las dimensiones del campo. Si indicas las dimensiones incorrectas, la homografía proyectará los jugadores a posiciones erróneas.

```bash
# Campo de Banyoles: 98×61 m
python homografia_interactiva.py --half top --frame 3000 --longitud 98.0 --anchura 61.0

# Campo estándar FIFA: 105×68 m
python homografia_interactiva.py --half top --frame 3000 --longitud 105.0 --anchura 68.0

# Campo pequeño (fútbol base): 90×55 m
python homografia_interactiva.py --half top --frame 3000 --longitud 90.0 --anchura 55.0
```

Si no se pasan por CLI, el script pregunta las dimensiones de forma interactiva al inicio.

#### 2. Durante la generación de prototipos

En `generar_prototipos_v3.py`, las dimensiones del campo se usan para separar los pools de porteros por posición (área penal = últimos 16,5 m del campo). Si no coinciden con la homografía, los porteros se asignarán al pool incorrecto:

```bash
python generar_prototipos_v3.py \
    --field_length 98.0 \
    --field_width  61.0 \
    ...
```

#### 3. En la app Streamlit durante el procesado

En la pestaña **"Partido & Tracker"** (paso 2 de la app) o en el expander **"Segmento a procesar"** (paso 3), introduce el largo y el ancho del campo:

- **Largo (línea de banda)**: entre 90 y 120 m
- **Ancho (línea de fondo)**: entre 45 y 90 m

Estos valores se almacenan en `config['field_length']` y `config['field_width']` y se propagan a todas las funciones de métricas (`compute_tactical_stats`, `compute_physical_stats`, `compute_zone_stats`, `compute_lineup_and_formation`, etc.).

#### 4. En `pipeline_core.py` (solo si se usa el módulo directamente, sin la app)

Las constantes por defecto del módulo son:

```python
# pipeline_core.py, líneas 29-30
L_M = 98.0   # longitud por defecto (metros)
A_M = 61.0   # anchura por defecto (metros)
```

Si llamas a `process_segment()` directamente desde Python (sin pasar por la app), asegúrate de incluir las dimensiones en el diccionario `config`:

```python
config = {
    'field_length': 105.0,  # tu campo
    'field_width':  68.0,
    # ...resto de parámetros
}
core.process_segment(vid_pan, vid_std, H_a, H_b, protos_day, None, config, ...)
```

### Checklist para un campo nuevo

1. Mide el campo o consulta la ficha técnica del estadio
2. Calibra `H_veo_top.npy` pasando `--longitud X --anchura Y`
3. Calibra `H_veo_bottom.npy` con las mismas dimensiones
4. Genera prototipos pasando `--field_length X --field_width Y`
5. En la app, introduce las dimensiones en "Partido & Tracker" antes de procesar

> **Nota:** las dimensiones en `veo_pipeline.py` (`FIELD_LENGTH = 105.0`, `FIELD_WIDTH = 68.61`) corresponden al modelo matemático interno del shader GLSL de VEO y **no deben modificarse**. Son constantes del modelo de cámara, no del campo que se está analizando.

---

## Entrenar el detector YOLO

### 1. Descargar y preparar el dataset

Los datasets están en Roboflow:

- **Jugadores en panorámico**: [panoramic-cam](https://universe.roboflow.com/javiers-workspace-osouu/panoramic-cam)
- **Jugadores en vídeo estándar**: [analisis-tactico-futbol](https://universe.roboflow.com/javiers-workspace-osouu/analisis-tactico-futbol)

```bash
python prepare_dataset.py
```

Este script descarga el dataset, separa las anotaciones de jugadores y balón en dos datasets independientes, convierte las segmentaciones a bounding boxes, y genera los `data.yaml` necesarios para el entrenamiento. Requiere la clave de API de Roboflow (actualmente incluida en el script).

### 2. Lanzar el fine-tuning

```bash
# Entrenar jugadores y balón (ambos en secuencia)
python finetune_players_panoramic.py

# Solo jugadores
python finetune_players_panoramic.py --only players

# Solo balón
python finetune_players_panoramic.py --only ball

# Reanudar desde un checkpoint interrumpido
python finetune_players_panoramic.py --resume
```

**Edita las rutas** al inicio del script antes de ejecutar:

```python
# finetune_players_panoramic.py
BALL_YAML    = "panoramic-cam-1_ball/data.yaml"
PLAYERS_YAML = "panoramic-cam-1_players/data.yaml"
WEIGHTS_PLAYERS = "runs/detect/modelo_players_v23/weights/best.pt"
WEIGHTS_BALL    = "runs/detect/modelo_ball_v33/weights/best.pt"
```

**Parámetros de entrenamiento actuales** (jugadores):

- Arquitectura: YOLO v2.6, 80 épocas, imagen 1280 px, batch 4
- Optimizador: AdamW, lr0=1e-4, cosine annealing
- Augmentación: HSV, escala, mosaic, mixup=0,1
- Requiere GPU con al menos 8 GB VRAM

Los modelos entrenados se guardan en:
- `runs/detect/modelo_players_v24_panoramic/weights/best.pt`
- `runs/detect/modelo_ball_v34_panoramic/weights/best.pt`

---

## Modelos incluidos y disponibles

| Modelo | Tamaño | Disponibilidad |
|--------|--------|----------------|
| Balón `runs/detect/modelo_ball_v33/weights/best.pt` | 20 MB | Incluido en el repositorio |
| Jugadores panorámico `modelo_players_v24_panoramic2` | 126 MB | Supera el límite de GitHub. Entrena con `finetune_players_panoramic.py` o contacta con el autor |

La app busca el modelo de jugadores automáticamente en `runs/detect/modelo_players_v24_panoramic2/weights/best.pt`.

---

## Notebooks de análisis y diagnóstico

| Notebook | Descripción |
|----------|-------------|
| `veo_banyoles_pipeline.ipynb` | Demo completa del pipeline sobre el partido de Banyoles. Reproduce todo el flujo desde la corrección geométrica hasta las métricas tácticas |
| `analisis_formacion.ipynb` | Extracción y visualización de la formación táctica (4-3-3, etc.) desde el CSV de tracking. Incluye análisis por tiempo |
| `analisis_ids.ipynb` | Diagnóstico de fragmentación de IDs: histograma de duración de tracks, detección de fragmentos cortos, ajuste de parámetros del tracker |
| `analisis_pkl.ipynb` | Inspector y tuner de prototipos de color. Muestra las firmas hue de cada clase, permite comparar PKLs y ajustar umbrales manualmente |
| `homography_field.ipynb` | Verificación de la homografía: proyecta el modelo FIFA sobre el frame para comprobar el alineamiento visual |
| `clasificacion_pipeline.ipynb` | Pipeline de clasificación paso a paso con análisis de márgenes de decisión y tasa de unknowns |
| `diagnostico_tracking.ipynb` | Diagnóstico completo de trayectorias: gaps por ID, duración media, velocidades, comparación de trackers |
| `detections_viewer.ipynb` | Visor frame a frame de detecciones con overlay de bounding boxes y heatmaps interactivos |
| `comparacion_trackers.ipynb` | Comparación cuantitativa entre ByteTrack y StrongKalmanTrackerV3: fragmentación, switches de ID, cobertura temporal |

---

## Referencia de argumentos por script

### `marcar_tiempos.py`

```
python marcar_tiempos.py <video_standard.mp4>
```

| Argumento | Descripción |
|-----------|-------------|
| `video` (posicional) | Ruta al vídeo estándar VEO (1920×1080) |

**Salida:** `tiempos_partido.json` con los frames e instantes en minutos de inicio/fin de cada tiempo.

---

### `homografia_interactiva.py`

```
python homografia_interactiva.py [opciones]
```

| Argumento | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `--imagen` | str | — | Ruta a imagen PNG/JPG (alternativa al vídeo). Si se omite, lee del vídeo en `VIDEO_PATH` |
| `--half` | `top`/`bottom` | `top` | Mitad del campo a calibrar. Se infiere del nombre del archivo si se usa `--imagen` |
| `--frame` | int | `3000` | Frame del vídeo VEO a usar como referencia |
| `--alpha` | float | `0.45` | Opacidad inicial del overlay del campo |
| `--longitud` | float | — | Longitud del campo en metros. Si no se indica, pregunta interactivamente |
| `--anchura` | float | — | Anchura del campo en metros. Debe indicarse junto con `--longitud` |

**Salida:** archivo `.npy` con la homografía 3×3 (píxeles → metros). Por defecto en `data/homography/H_veo_{half}.npy`.

---

### `generar_prototipos_v3.py`

```
python generar_prototipos_v3.py --video <ruta> --model <ruta> [opciones]
```

| Argumento | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `--video` | str | **obligatorio** | Vídeo panorámico |
| `--model` | str | **obligatorio** | Pesos YOLO de jugadores |
| `--output` | str | `prototypes_v3.pkl` | Ruta del PKL de salida |
| `--cache` | str | `rows_cache.pkl` | Caché de detecciones (permite relanzar sin re-detectar) |
| `--no_cache` | flag | — | Ignorar caché y re-detectar |
| `--sample_sec` | int | `180` | Segundos de vídeo a muestrear (sin `--intervals`) |
| `--step` | float | `2.5` | Segundos entre frames muestreados |
| `--sam` | str | `sam2_t.pt` | Modelo SAM2. Cadena vacía para desactivar |
| `--k` | int | — | Clusters KMeans forzado (si se omite, se sugiere automáticamente) |
| `--n_top` | int | `20` | Top-N detecciones más centrales para calcular cada prototipo |
| `--conf` | float | `0.45` | Umbral de confianza para muestreo YOLO |
| `--intervals` | str | — | Intervalos de juego. Formato: `MM:SS-MM:SS[:L\|:R],...` |
| `--H_a` | str | — | Homografía cámara A (activa split posicional de porteros) |
| `--H_b` | str | — | Homografía cámara B |
| `--field_length` | float | `105.0` | Longitud del campo en metros |
| `--field_width` | float | `68.0` | Anchura del campo en metros |
| `--home_left` | flag | activado | Local ataca hacia la izquierda en el 1T |

**Salidas:** `prototypes_v3.pkl` (un único PKL) o `prototypes_v3_day.pkl` + `prototypes_v3_night.pkl` si se detecta cambio de iluminación durante el partido.

---

### `finetune_players_panoramic.py`

```
python finetune_players_panoramic.py [opciones]
```

| Argumento | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `--only` | `players`/`ball` | — | Entrenar solo un modelo. Si se omite, entrena ambos en secuencia |
| `--resume` | flag | — | Reanudar desde `last.pt` (entrenamiento interrumpido) |

---

### `prepare_dataset.py`

```
python prepare_dataset.py
```

Sin argumentos. Descarga el dataset `panoramic-cam v1` de Roboflow, separa jugadores y balón, convierte segmentaciones a bounding boxes y genera los YAML de entrenamiento.

---

### `veo_pipeline.py` (uso directo como módulo)

No tiene CLI propia, pero puede ejecutarse como script de demo:

```bash
python veo_pipeline.py [segundos]
```

Carga el frame en el segundo indicado (defecto: 40 s), aplica la corrección geométrica completa y guarda `veo_pipeline_demo.png`.

Funciones exportables principales:

| Función | Descripción |
|---------|-------------|
| `load_raw_halves(path, frame_sec)` | Carga un frame y lo divide en cam_top y cam_bot |
| `undistort_half(img, cam)` | Aplica elastic + undistorsión racional. Devuelve imagen corregida y K |
| `pixel_to_field(px, py, cam, new_K)` | Proyecta un píxel al plano del campo mediante ray-casting |
| `pixels_to_field_batch(pixels, cam, new_K)` | Versión vectorizada para N píxeles |
| `yolo_to_field_H(boxes, H)` | Convierte bboxes YOLO a metros usando homografía manual |
| `load_homographies(h_left, h_right)` | Carga las dos homografías desde disco |

---

## Cómo funciona (arquitectura interna)

### Corrección geométrica (`veo_pipeline.py`)

El vídeo panorámico VEO tiene dos distorsiones apiladas que deben deshacerse:

1. **Compresión elástica en Y** (`undo_elastic_compression`): el codec AWS ElasticTranscoder reescala las filas de forma no lineal. La zona superior (≤42% de altura) se estira ×1,41; la inferior se comprime ×0,70. Se invierte con `cv2.remap`.

2. **Undistorsión racional** (`undistort_half`): modelo OpenCV de 8 coeficientes extraído del shader GLSL del player web de VEO mediante Spector.js. Parámetros distintos para la lente izquierda (cam A) y derecha (cam B).

Los parámetros del shader (matrices de cámara K, coeficientes de distorsión, matrices de rotación lente-cámara-mundo) están fijados en el código para la calibración VEO versión 6.2 y no deben modificarse.

### Detección y clasificación (`pipeline_core.py`)

Cada frame (2048×2048) se divide en dos mitades (cam A superior, cam B inferior). Cada mitad pasa por:

1. `preprocess_half()`: corrección completa (elastic + undistorsión con alpha=1+crop)
2. YOLO v2.6 separado por cámara
3. Proyección bboxes → metros con la homografía (`pixel_to_field`)
4. Fusión de detecciones de ambas cámaras en la zona de costura (NMS en espacio métrico)
5. Clasificación por hue signature v3 (distancia Hellinger sobre 12 bins + fracciones blanco/oscuro)

### Tracking (`StrongKalmanTrackerV3`)

- Filtro de Kalman 2D en coordenadas métricas (estado: posición + velocidad)
- Matching húngaro en dos etapas: detecciones de alta confianza primero, luego las restantes contra tracks perdidos
- Buffer "lost" de 8 s para cubrir oclusiones, corners y saques de banda
- Galería Re-ID de 120 s con firma de tono acumulada para recuperar IDs tras ausencias prolongadas
- Límites por clase: máximo 12 jugadores de campo por equipo, 1 portero, 4 árbitros

### Métricas tácticas

Todas las métricas son relativas a la dirección de ataque de cada equipo en cada mitad del partido, usando `home_attacks_left` y `halftime_frame`:

| Métrica | Descripción |
|---------|-------------|
| **Línea defensiva** | Posición del 4.º jugador más retrasado (metros desde portería propia) |
| **Compacidad** | Desviación típica combinada de posiciones x e y del bloque |
| **Territorio** | % de frames en que el equipo tiene más jugadores en campo rival |
| **Presión** | Media de jugadores del equipo en el último tercio rival |
| **Posesión** | % de frames en que el jugador más cercano al balón es del equipo |
| **Pressing** | Media de los 3 jugadores más próximos al balón |
| **Transición** | Tiempo medio entre cruces del centroide por la línea de medio campo |
| **Mapa zonal** | 15 zonas (5 col × 3 fil); densidad relativa home/away en cada celda |
| **Formación** | Estimada por mapa de densidad 2D + supresión iterativa de picos |
| **HSR / Sprints** | Distancia a alta intensidad (>15 km/h) y sprints (>20 km/h) por jugador |

---

## Licencia

MIT
