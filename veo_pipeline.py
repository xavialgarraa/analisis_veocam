"""
veo_pipeline.py
===============
Pipeline de corrección geométrica para cámaras VEO.

Reproduce el shader GLSL del player web de VEO:
  1. Descompresión elástica Y       (función getImage del shader)
  2. Undistorsión rational model    (función distort/getUV)
  3. Proyección rayo → plano campo  (función main, Y=0 world plane)

Valores numéricos extraídos en vivo del WebGL (Spector.js).
Calibración version 6.2.

Uso rápido:
    from veo_pipeline import load_raw_halves, undistort_half, pixel_to_field
    top, bot = load_raw_halves("data/videos/veo_panoramico.mp4", frame_sec=40)
    top_rect = undistort_half(top, cam='left')
    x_m, z_m = pixel_to_field(px=800, py=600, cam='left')
"""

import cv2
import numpy as np
from functools import lru_cache
from typing import Literal, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES DEL SHADER  (extraídas en vivo del WebGL)
# ─────────────────────────────────────────────────────────────────────────────

# --- Compresión elástica (getImage del shader) ---
_TD = 19.0 / 45.0   # topDestRatio   = 0.42222  umbral en espacio sensor
_TS = 19.0 / 32.0   # topSourceRatio = 0.59375  umbral en espacio textura
_BD = 1.0 - _TD     # bottomDestRatio
_BS = 1.0 - _TS     # bottomSourceRatio

# --- Distorsión rational OpenCV 8-coef (idéntica en ambas lentes) ---
DIST_COEFFS = np.array([
    0.8003565073013306,     # k1
    0.10934139043092728,    # k2
    0.0,                    # p1
    0.0,                    # p2
    0.0012755499919876456,  # k3
    1.155151128768921,      # k4
    0.29691845178604126,    # k5
    0.012696989811956882,   # k6
], dtype=np.float64)

# --- Intrínsecos 4K (3840×2160) — GLSL mat3 column-major → .reshape(3,3).T ---
K_LEFT_4K = np.array([
    [2249.317626953125, 0.0,              1992.5738525390625],
    [0.0,              2249.317626953125, 1110.5870361328125],
    [0.0,              0.0,              1.0               ],
], dtype=np.float64)

K_RIGHT_4K = np.array([
    [2253.00390625, 0.0,          1983.9678955078125],
    [0.0,          2253.00390625, 1115.5120849609875],
    [0.0,          0.0,          1.0               ],
], dtype=np.float64)

# Resoluciones
SENSOR_W, SENSOR_H = 3840.0, 2160.0   # resolución para la que se calibró K
RAW_W,    RAW_H    = 2048,   1024      # resolución de cada mitad en el raw

# Intrínsecos escalados a la resolución raw (la que usamos en Python)
_sx = RAW_W / SENSOR_W
_sy = RAW_H / SENSOR_H
K_LEFT  = K_LEFT_4K  * np.array([[_sx, 1, _sx], [1, _sy, _sy], [1, 1, 1]])
K_RIGHT = K_RIGHT_4K * np.array([[_sx, 1, _sx], [1, _sy, _sy], [1, 1, 1]])

# --- Matrices 4×4 del shader (GLSL column-major → .reshape(4,4).T) ---
def _m4(flat):
    return np.array(flat, dtype=np.float64).reshape(4, 4).T

CAMERA2WORLD = _m4([
     0.9973753094673157,    0.003910509869456291,   0.07229968905448914,   0.0,
    -0.003475569887086749,  0.9999750852584839,    -0.0061406600289046764, 0.0,
    -0.07232190668582916,   0.005873260088264942,   0.9973640441894531,    0.0,
     0.33331966400146484,  -3.8922219276428223,    -40.07933044433594,     1.0,
])

CAMERA2LOGIC = _m4([
     0.999992311000824,      -0.003910577390342951,   1.2181130770727577e-8, 0.0,
     0.003910512663424015,    0.9999751448631287,      0.005873254965990782,  0.0,
    -2.2977852495387197e-5,  -0.005873214919120073,   0.9999827146530151,    0.0,
    -9.385230370639874e-10,  -2.399940228769992e-7,  -1.409581229516732e-9,  1.0,
])

LENS_LEFT2CAMERA = _m4([
     0.7041498422622681,    0.002415139926597476,   0.7100473642349243,    0.0,
     0.18273499608039856,   0.9656950235366821,    -0.18450191617012024,   0.0,
    -0.6861348152160645,    0.2596674859523773,     0.6795526742935181,    0.0,
    -0.03899998962879181,   3.456999911577441e-5,  -1.5139999959501438e-5, 1.0,
])

LENS_RIGHT2CAMERA = _m4([
     0.7044305801391602,    0.000531190016772598,  -0.7097727060317993,   0.0,
    -0.18178533017635345,   0.9667803645133972,    -0.1796935647726059,   0.0,
     0.6860988736152649,    0.2556079030036926,     0.6811262369155884,   0.0,
     0.039000000804662704,  0.0,                    0.0,                  1.0,
])

# Dimensiones del campo (metros)
FIELD_LENGTH = 105.0
FIELD_WIDTH  = 68.61029052734375

# Parámetros panorámica
TRI_HALF_BASE_RATIO = 0.065   # semibase triángulo negro / ancho total
TRI_APEX_Y_RATIO    = 0.60    # altura vértice / altura imagen


# ─────────────────────────────────────────────────────────────────────────────
#  1.  CARGA DE FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_halves(video_path: str, frame_sec: float = 0.0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae un frame del raw VEO (2048×2048) y lo divide en las dos mitades.

    Returns
    -------
    top : ndarray (1024, 2048, 3) RGB — cámara izquierda (campo izq)
    bot : ndarray (1024, 2048, 3) RGB — cámara derecha  (campo der)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_sec * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"No se pudo leer frame en t={frame_sec}s")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mid = frame_rgb.shape[0] // 2
    return frame_rgb[:mid].copy(), frame_rgb[mid:].copy()


# ─────────────────────────────────────────────────────────────────────────────
#  2.  DESCOMPRESIÓN ELÁSTICA  (inversa de getImage del shader)
# ─────────────────────────────────────────────────────────────────────────────

# Pre-calculamos el mapa de descompresión para 1024 filas (se reutiliza)
@lru_cache(maxsize=8)
def _elastic_remap(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea los mapas cv2.remap que deshacen la compresión elástica en Y.

    El shader getImage() transforma uv.y (coordenada sensor) así antes de
    hacer el lookup en la textura:
        if sensor_y < TD:  texture_y = sensor_y / TD * TS       (×1.406, estira)
        else:              texture_y = (sensor_y-TD)/BD * BS + TS  (×0.703, comprime)

    Para "descomprimir" el raw, usamos cv2.remap con ese mapa DIRECTO:
        output[y_sensor] = input[y_texture = forward(y_sensor)]
    """
    y_sensor = np.arange(H, dtype=np.float32) / H

    # Mapeo directo: sensor → textura (lo que hace el shader)
    y_tex = np.where(
        y_sensor < _TD,
        y_sensor / _TD * _TS,
        (y_sensor - _TD) / _BD * _BS + _TS,
    )

    map_y = (y_tex * H).astype(np.float32)
    map_y = np.broadcast_to(map_y[:, np.newaxis], (H, W)).copy()
    map_x = np.broadcast_to(
        np.arange(W, dtype=np.float32)[np.newaxis, :], (H, W)
    ).copy()
    return map_x, map_y


def undo_elastic_compression(img_half: np.ndarray) -> np.ndarray:
    """
    Deshace la compresión elástica en Y que VEO aplica al codificar el raw.

    El AWS ElasticTranscoder re-escala las filas de forma no-lineal:
      - Zona superior (≤42 % de la altura):  estirada ×1.41
      - Zona inferior (>42 %):               comprimida ×0.70
    Invierte ese mapeo para recuperar la imagen real del sensor.

    Parámetros
    ----------
    img_half : ndarray (H, W, C) — una mitad del raw (top o bot)

    Devuelve
    --------
    ndarray (H, W, C) — imagen con la compresión de codec deshecha
    """
    H, W = img_half.shape[:2]
    map_x, map_y = _elastic_remap(H, W)
    return cv2.remap(img_half, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


# ─────────────────────────────────────────────────────────────────────────────
#  3.  UNDISTORSIÓN RACIONAL  (modelo OpenCV 8-coef)
# ─────────────────────────────────────────────────────────────────────────────

_undist_cache: dict = {}

def _get_undist_maps(K: np.ndarray, dist: np.ndarray,
                     w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (tuple(K.flatten().tolist()), w, h)
    if key not in _undist_cache:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1)
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, new_K, (w, h), cv2.CV_32FC1)
        _undist_cache[key] = (map1, map2, new_K)
    return _undist_cache[key]


def undistort_half(img_half: np.ndarray,
                   cam: Literal['left', 'right'] = 'left',
                   elastic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline completo de corrección para una mitad del raw VEO:
        raw → undo_elastic → rational undistort → imagen rectilineal

    Parámetros
    ----------
    img_half : ndarray (H, W, C)
    cam      : 'left'  (= cam_top, mitad superior del raw)
               'right' (= cam_bot, mitad inferior del raw)
    elastic  : si True, aplica primero la descompresión elástica

    Devuelve
    --------
    undist   : ndarray (H, W, C)  imagen corregida
    new_K    : ndarray (3, 3)     matriz intrínseca de la imagen corregida
               (necesaria para pixel_to_field)
    """
    if elastic:
        img_half = undo_elastic_compression(img_half)

    K    = K_LEFT  if cam == 'left' else K_RIGHT
    dist = DIST_COEFFS
    h, w = img_half.shape[:2]
    map1, map2, new_K = _get_undist_maps(K, dist, w, h)
    undist = cv2.remap(img_half, map1, map2, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    return undist, new_K


# ─────────────────────────────────────────────────────────────────────────────
#  4.  PROYECCIÓN PÍXEL → PLANO DEL CAMPO  (ray-casting Y=0)
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_field(px: float, py: float,
                   cam: Literal['left', 'right'],
                   new_K: Optional[np.ndarray] = None
                   ) -> Optional[Tuple[float, float]]:
    """
    Proyecta un píxel de la imagen undistorted al plano del campo (Y=0 world).

    Ray-casting del shader main():
        píxel → rayo en espacio cámara → espacio world → intersección Y=0

    Parámetros
    ----------
    px, py  : coordenadas del píxel en la imagen undistorted
    cam     : 'left' o 'right'
    new_K   : matriz K de la imagen undistorted (devuelta por undistort_half).
              Si None, usa K_LEFT/K_RIGHT escalada (menos preciso).

    Devuelve
    --------
    (x_m, z_m) posición en metros en el plano del campo, o None si el rayo
    apunta hacia arriba y no intersecta.

    Nota sobre el sistema de coordenadas VEO
    ----------------------------------------
    - Origen: posición de la cámara en X=0 (no el centro del campo)
    - Y: eje vertical (Y=0 = plano del campo, Y<0 bajo el campo)
    - Z: profundidad desde la cámara hacia el campo
    El campo cubre aprox X ∈ [-FIELD_LENGTH/2, +FIELD_LENGTH/2]
                          Z ∈ [0, FIELD_WIDTH]
    """
    K_ref = new_K if new_K is not None else (K_LEFT if cam == 'left' else K_RIGHT)

    # 1. Píxel → dirección normalizada en espacio cámara-lente
    fx, fy = K_ref[0, 0], K_ref[1, 1]
    cx, cy = K_ref[0, 2], K_ref[1, 2]
    x_norm = (px - cx) / fx
    y_norm = (py - cy) / fy
    d_lens = np.array([x_norm, y_norm, 1.0, 0.0])  # w=0 → dirección

    # 2. Dirección en espacio world
    L2C = LENS_LEFT2CAMERA if cam == 'left' else LENS_RIGHT2CAMERA
    d_world = CAMERA2WORLD @ L2C @ d_lens          # (4,)

    # 3. Posición de la lente en espacio world
    origin_lens = CAMERA2WORLD @ L2C @ np.array([0.0, 0.0, 0.0, 1.0])  # (4,)

    # 4. Intersección con el plano Y=0
    #    ray: P(t) = origin + t * direction
    #    Y=0:  origin.y + t * dir.y = 0  →  t = -origin.y / dir.y
    dir_y = d_world[1]
    if abs(dir_y) < 1e-8:
        return None   # rayo casi horizontal

    t = -origin_lens[1] / dir_y
    if t < 0:
        return None   # intersección detrás de la cámara

    x_m = origin_lens[0] + t * d_world[0]
    z_m = origin_lens[2] + t * d_world[2]

    return float(x_m), float(z_m)


def pixels_to_field_batch(pixels: np.ndarray,
                          cam: Literal['left', 'right'],
                          new_K: Optional[np.ndarray] = None
                          ) -> np.ndarray:
    """
    Versión vectorizada de pixel_to_field para múltiples detecciones.

    Parámetros
    ----------
    pixels : ndarray (N, 2)  columnas [px, py]
    cam    : 'left' o 'right'
    new_K  : matriz K undistorted

    Devuelve
    --------
    field_pos : ndarray (N, 2)  columnas [x_m, z_m]
                NaN donde el rayo no intersecta el campo
    """
    K_ref = new_K if new_K is not None else (K_LEFT if cam == 'left' else K_RIGHT)
    fx, fy = K_ref[0, 0], K_ref[1, 1]
    cx, cy = K_ref[0, 2], K_ref[1, 2]

    N = len(pixels)
    x_norm = (pixels[:, 0] - cx) / fx
    y_norm = (pixels[:, 1] - cy) / fy
    ones   = np.ones(N)
    zeros  = np.zeros(N)
    # d_lens: (4, N)
    d_lens = np.stack([x_norm, y_norm, ones, zeros], axis=0)

    L2C = LENS_LEFT2CAMERA if cam == 'left' else LENS_RIGHT2CAMERA
    M   = CAMERA2WORLD @ L2C
    d_world  = M @ d_lens                                        # (4, N)
    origin_l = (CAMERA2WORLD @ L2C @ np.array([0, 0, 0, 1]))    # (4,)

    dir_y = d_world[1]   # (N,)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -origin_l[1] / dir_y
    valid = (np.abs(dir_y) > 1e-8) & (t > 0)

    field_pos = np.full((N, 2), np.nan)
    field_pos[valid, 0] = origin_l[0] + t[valid] * d_world[0, valid]
    field_pos[valid, 1] = origin_l[2] + t[valid] * d_world[2, valid]
    return field_pos


# ─────────────────────────────────────────────────────────────────────────────
#  5.  RECONSTRUCCIÓN PANORÁMICA
# ─────────────────────────────────────────────────────────────────────────────

def build_panoramic(top_rect: np.ndarray, bot_rect: np.ndarray) -> np.ndarray:
    """
    Une las dos imágenes rectificadas en el panorámico estilo VEO:
        [cam_left | cam_right]  →  4096×1024
    con el triángulo negro central (zona muerta de la montura).
    """
    panoramic = np.hstack([top_rect, bot_rect])
    H, W = panoramic.shape[:2]
    seam_x = W // 2

    mask = np.ones((H, W), dtype=np.uint8)
    tri_half = int(W * TRI_HALF_BASE_RATIO)
    tri_apex = int(H * TRI_APEX_Y_RATIO)
    pts = np.array([
        [seam_x - tri_half, H],
        [seam_x + tri_half, H],
        [seam_x,            tri_apex],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 0)
    return panoramic * mask[:, :, np.newaxis]


# ─────────────────────────────────────────────────────────────────────────────
#  6.  HOMOGRAFÍA MANUAL (calibrador.py)  — método más robusto para el TFG
# ─────────────────────────────────────────────────────────────────────────────

def load_homographies(h_left_path: str  = "data/H_left_undist.npy",
                      h_right_path: str = "data/H_right_undist.npy"
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga las homografías generadas por calibrador.py sobre imágenes undistorted.
    Ambas mapean píxeles de imagen undistorted → metros del campo (X, Y).
    """
    H_left  = np.load(h_left_path)
    H_right = np.load(h_right_path)
    return H_left, H_right


def pix2field_H(px: float, py: float, H: np.ndarray) -> Tuple[float, float]:
    """
    Proyecta un píxel al campo usando una homografía imagen→campo.

    Parámetros
    ----------
    px, py : coordenadas en la imagen undistorted
    H      : homografía 3×3 (imagen → campo, generada por calibrador.py)

    Devuelve
    --------
    (x_m, y_m) posición en metros en el campo
    """
    pt  = np.array([[[px, py]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H)
    return float(res[0, 0, 0]), float(res[0, 0, 1])


def yolo_to_field_H(boxes: np.ndarray, H: np.ndarray,
                    foot_ratio: float = 0.95) -> np.ndarray:
    """
    Convierte bboxes YOLO a posiciones campo usando homografía manual.

    Parámetros
    ----------
    boxes      : ndarray (N, 4)  formato [x1, y1, x2, y2]
    H          : homografía imagen→campo (de calibrador.py)
    foot_ratio : fracción de la bbox desde arriba para el punto de pie

    Devuelve
    --------
    ndarray (N, 2)  posiciones en metros [x_m, y_m]
    """
    cx_pix = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy_pix = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) * foot_ratio
    pts = np.stack([cx_pix, cy_pix], axis=1).astype(np.float32).reshape(-1, 1, 2)
    res = cv2.perspectiveTransform(pts, H)
    return res.reshape(-1, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  7.  UTILIDADES YOLO → CAMPO  (ray-casting — alternativa automática)
# ─────────────────────────────────────────────────────────────────────────────

def yolo_detections_to_field(boxes: np.ndarray,
                              cam: Literal['left', 'right'],
                              new_K: Optional[np.ndarray] = None,
                              foot_ratio: float = 0.95
                              ) -> np.ndarray:
    """
    Convierte bounding boxes de YOLO a posiciones en el campo.

    Parámetros
    ----------
    boxes      : ndarray (N, 4)  formato [x1, y1, x2, y2] en pixels
    cam        : 'left' o 'right'
    new_K      : K de la imagen undistorted
    foot_ratio : fracción de la bbox desde arriba para el punto de pie
                 (0.95 = casi el borde inferior de la caja)

    Devuelve
    --------
    ndarray (N, 2)  posiciones en metros [x_m, z_m] (NaN si fuera de campo)
    """
    # Punto de pie: centro horizontal, pie de la caja
    cx_pix = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy_pix = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) * foot_ratio
    pixels = np.stack([cx_pix, cy_pix], axis=1)
    return pixels_to_field_batch(pixels, cam, new_K)


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO  (ejecutar con: python veo_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    video = r"C:\Users\xavia\Documents\GitHub\tfg-cv\data\videos\veo_panoramico.mp4"
    sec   = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0

    print(f"Cargando frame t={sec}s …")
    top_raw, bot_raw = load_raw_halves(video, sec)

    print("Corrigiendo distorsión (elastic + rational model) …")
    top_rect, K_top = undistort_half(top_raw, cam='left')
    bot_rect, K_bot = undistort_half(bot_raw, cam='right')

    panoramic = build_panoramic(top_rect, bot_rect)

    # Test de proyección: esquinas de la imagen
    test_pts = [(0, 0), (1024, 512), (2047, 1023)]
    print("\nTest pixel_to_field (cam left):")
    for px, py in test_pts:
        result = pixel_to_field(px, py, cam='left', new_K=K_top)
        if result:
            print(f"  ({px:4d},{py:4d}) → ({result[0]:+6.1f} m, {result[1]:+6.1f} m)")
        else:
            print(f"  ({px:4d},{py:4d}) → rayo no intersecta el campo")

    # Visualización
    fig, axes = plt.subplots(3, 1, figsize=(22, 12))
    axes[0].imshow(top_raw);   axes[0].set_title('TOP raw'); axes[0].axis('off')
    axes[1].imshow(top_rect);  axes[1].set_title('TOP corregida (elastic + rational)'); axes[1].axis('off')
    axes[2].imshow(panoramic); axes[2].set_title('Panorámica VEO reconstruida'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig('veo_pipeline_demo.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("\nGuardado: veo_pipeline_demo.png")
