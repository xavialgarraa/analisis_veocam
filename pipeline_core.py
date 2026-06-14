"""
pipeline_core.py
================
Core pipeline VEO — alpha=1+crop
Basado en veo_tracking_alpha1.ipynb
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, cv2, pickle, threading
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.ndimage import gaussian_filter
import torch
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle, Arc

from veo_pipeline import undo_elastic_compression, K_LEFT, K_RIGHT, DIST_COEFFS


# ═══════════════════════════════════════════════════════════
#  1. CONSTANTES DE CAMPO Y DISPLAY
# ═══════════════════════════════════════════════════════════

L_M    = 98.0
A_M    = 61.0
ESCALA = 10

HALF_H = 1024   # mitad de la altura del frame VEO (RAW_H de veo_pipeline)
RAW_W  = 2048

CONF_THRESH        = 0.5
CONF_BALL          = 0.50
BRIGHT_THRESH      = 75.0
CLASSIFY_MAX_SCORE = 80.0
GLOBAL_NMS_M       = 2.5
MAX_DETS_FRAME     = 26
SEAM_MARGIN_M      = 6.0
SEAM_FUSION_M      = 2.5
TRACK_MAX_DIST_M   = 6.0
TRACK_WINDOW_S     = 7.0
TRACK_CLASS_W      = 8.0
CLASS_OVERRIDE_F   = 20  # frames consecutivos con clase distinta para reclasificar

CLASSES_ORDER = ['player_home', 'player_away', 'gk_home', 'gk_away', 'referee']

COLOR_MAP_BGR = {
    'player_home': (0,   0,   220),
    'player_away': (220, 60,  0),
    'gk_home':     (230, 0,   230),
    'gk_away':     (0,   200, 200),
    'referee':     (0,   220, 220),
}
COLOR_MAP_HEX = {
    'player_home': '#DC143C',
    'player_away': '#1E90FF',
    'gk_home':     '#FF00FF',
    'gk_away':     '#00CED1',
    'referee':     '#FFD700',
    'unknown':     '#888888',
}
TEAM_COLORS = COLOR_MAP_HEX
BALL_COLOR  = '#FF6B35'
DARK_BG     = '#0d1117'
GRASS_DARK  = '#1a472a'
GRASS_LIGHT = '#215732'
LINE_COLOR  = 'white'

GAP_COLOR_FRAMES     = 10
COLOR_WEIGHT_M       = 3.0
MIN_HITS_CLASS       = 8
MIN_CONF_NEW_UNKNOWN = 0.65
ALPHA_UNDIST         = 1.0

# ── v3 clasificación ──────────────────────────────────────
_HUE_BINS            = 12
CLASSIFY_V3_MAX_SCORE       = 0.38
CLASSIFY_V3_MARGIN          = 0.04
CLASSIFY_V3_CONFIDENT_MARGIN = 0.1  # below this → double-check with BGR Pass 2

# ── Área penal (para apply_gk_by_position) ────────────────
_PA_DEPTH_M = 13.0  # zona de detección GK por posición (área penal = 16.5m)
_PA_Y_MIN   = (A_M - 40.32) / 2        # ≈ 10.34
_PA_Y_MAX   = (A_M + 40.32) / 2        # ≈ 50.66


# ═══════════════════════════════════════════════════════════
#  2. MAPAS DE UNDISTORT (alpha=1+crop)
# ═══════════════════════════════════════════════════════════

_map1_a = _map2_a = _roi_a = None
_map1_b = _map2_b = _roi_b = None
_maps_initialized = False




def init_undistort_maps(half_h: int = HALF_H, vid_w: int = RAW_W):
    """Pre-computa los mapas de undistort para ambas cámaras."""
    global _map1_a, _map2_a, _roi_a, _map1_b, _map2_b, _roi_b, _maps_initialized
    _new_K_a, _roi_a = cv2.getOptimalNewCameraMatrix(
        K_LEFT,  DIST_COEFFS, (vid_w, half_h), alpha=ALPHA_UNDIST)
    _map1_a, _map2_a = cv2.initUndistortRectifyMap(
        K_LEFT,  DIST_COEFFS, None, _new_K_a, (vid_w, half_h), cv2.CV_32FC1)

    _new_K_b, _roi_b = cv2.getOptimalNewCameraMatrix(
        K_RIGHT, DIST_COEFFS, (vid_w, half_h), alpha=ALPHA_UNDIST)
    _map1_b, _map2_b = cv2.initUndistortRectifyMap(
        K_RIGHT, DIST_COEFFS, None, _new_K_b, (vid_w, half_h), cv2.CV_32FC1)
    _maps_initialized = True


def preprocess_half(half_bgr: np.ndarray, cam: str = 'left') -> np.ndarray:
    """
    Aplica alpha=1+crop a una mitad del frame VEO.
    Resultado: misma resolución que la entrada, sin bordes negros.
    """
    global _maps_initialized
    if not _maps_initialized:
        init_undistort_maps(half_bgr.shape[0], half_bgr.shape[1])

    prep = undo_elastic_compression(half_bgr)

    if cam == 'left':
        undist = cv2.remap(prep, _map1_a, _map2_a, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        roi = _roi_a
    else:
        undist = cv2.remap(prep, _map1_b, _map2_b, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        roi = _roi_b

    h_o, w_o = undist.shape[:2]
    x, y, rw, rh = roi
    crop = undist[y:y + rh, x:x + rw]
    return cv2.resize(crop, (w_o, h_o), interpolation=cv2.INTER_LINEAR)


# ═══════════════════════════════════════════════════════════
#  3. INFO DE VIDEO Y GEOMETRÍA
# ═══════════════════════════════════════════════════════════

def get_video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        'fps':        fps,
        'frames':     frames,
        'width':      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height':     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_s': frames / fps if fps > 0 else 0.0,
    }
    cap.release()
    return info


def compute_limite_x(H_a, H_b, vid_w: int, half_h: int) -> float:
    corners_a = esquinas_metros(H_a, vid_w, half_h)
    corners_b = esquinas_metros(H_b, vid_w, half_h)
    return float(np.mean([corners_a[:, 0].max(), corners_b[:, 0].min()]))


def compute_seam_margin(H_a, H_b, vid_w, half_h,
                        buffer_m=2.0,
                        overlap_ratio=0.15,
                        max_margin=8.0):

    corners_a = esquinas_metros(H_a, vid_w, half_h)
    corners_b = esquinas_metros(H_b, vid_w, half_h)

    a_min, a_max = corners_a[:,0].min(), corners_a[:,0].max()
    b_min, b_max = corners_b[:,0].min(), corners_b[:,0].max()

    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))

    margin = overlap * overlap_ratio + buffer_m

    return min(margin, max_margin)

# ═══════════════════════════════════════════════════════════
#  4. COLOR Y CLASIFICACIÓN (alpha=1+crop — BGR Euclidean)
# ═══════════════════════════════════════════════════════════

def get_torso_crop(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    return crop[int(h * 0.20):int(h * 0.60), int(w * 0.20):int(w * 0.80)]


def get_player_color(crop: np.ndarray, grass_hue=None):
    crop_rs = cv2.resize(crop, (64, 64))
    hsv = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    mask_cesped = (H > 35) & (H < 85) & (S > 50)
    if grass_hue is not None:
        lo = max(0, grass_hue - 15); hi = min(180, grass_hue + 15)
        mask_cesped |= (H > lo) & (H < hi) & (S > 40)
    mask_piel = (H < 20) & (S > 30) & (V > 90)
    mask      = ~(mask_cesped | mask_piel)
    pixels = crop_rs.reshape(-1, 3).astype(np.float32)
    valid  = pixels[mask.flatten()]
    gray   = np.array([128., 128., 128.])
    if len(valid) < 9:
        return gray, gray, [gray.copy(), gray.copy(), gray.copy()]
    median_c = np.median(valid, axis=0)
    mean_c   = np.mean(valid,   axis=0)
    k        = min(3, len(valid))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(valid, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order  = np.argsort(-counts)
    top3   = [centers[i] for i in order]
    while len(top3) < 3:
        top3.append(median_c.copy())
    return median_c, mean_c, top3


def color_distance(c1, c2) -> float:
    return float(np.linalg.norm(np.array(c1) - np.array(c2)))


def classify_player(crop: np.ndarray, class_prototypes: dict,
                    grass_hue=None, max_score=None):
    """Clasifica jugador por distancia BGR euclidea."""
    _threshold = max_score if max_score is not None else CLASSIFY_MAX_SCORE
    median_c, mean_c, top3 = get_player_color(crop, grass_hue)
    best_class, best_score = None, float('inf')
    for cls, proto in class_prototypes.items():
        d_median  = color_distance(median_c, proto['median'])
        d_mean    = color_distance(mean_c,   proto['mean'])
        d_cluster = (
            sum(min(color_distance(c, pc) for pc in proto['top3']) for c in top3) / len(top3)
            if proto.get('top3') else 999
        )
        score = 0.25 * d_median + 0.5 * d_mean + 0.25 * d_cluster
        if score < best_score:
            best_score = score
            best_class = cls
    if best_score > _threshold:
        return 'unknown', best_score
    return best_class, best_score


# ── v3: Hue signature ─────────────────────────────────────

def get_hue_signature(torso: np.ndarray, grass_hue=None) -> np.ndarray:
    """Array de _HUE_BINS bins normalizados + white_frac + dark_frac (len=14)."""
    if torso.size == 0:
        return np.zeros(_HUE_BINS + 2, dtype=np.float32)
    img = cv2.resize(torso, (64, 64))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H_ch = hsv[:, :, 0].astype(np.float32)
    S_ch = hsv[:, :, 1].astype(np.float32)
    V_ch = hsv[:, :, 2].astype(np.float32)
    mask_grass = (H_ch > 35) & (H_ch < 85) & (S_ch > 50)
    if grass_hue is not None:
        lo = max(0, grass_hue - 15); hi = min(180, grass_hue + 15)
        mask_grass |= (H_ch > lo) & (H_ch < hi) & (S_ch > 40)
    mask_skin  = (H_ch < 20) & (S_ch > 30) & (V_ch > 90)
    valid      = ~(mask_grass | mask_skin)
    n_valid    = int(valid.sum())
    if n_valid < 5:
        return np.zeros(_HUE_BINS + 2, dtype=np.float32)
    h_v = H_ch[valid]; s_v = S_ch[valid]; v_v = V_ch[valid]
    colored = s_v > 30
    if colored.sum() >= 3:
        hist, _ = np.histogram(h_v[colored], bins=_HUE_BINS, range=(0, 180))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
    else:
        hist = np.zeros(_HUE_BINS, dtype=np.float32)
    white_frac = float((v_v > 200).sum()) / n_valid
    dark_frac  = float((v_v < 60).sum())  / n_valid
    return np.concatenate([hist, [white_frac, dark_frac]]).astype(np.float32)


def _hue_sig_from_pixels(H_ch, S_ch, V_ch, pixel_mask, grass_hue=None):
    """
    Calcula la firma hue sobre los píxeles seleccionados por pixel_mask,
    excluyendo hierba y piel. Retorna array shape (14,).
    Función interna compartida por get_hue_signature_masked y la variante sin máscara.
    """
    mask_grass = (H_ch > 35) & (H_ch < 85) & (S_ch > 50)
    if grass_hue is not None:
        lo = max(0, grass_hue - 15); hi = min(180, grass_hue + 15)
        mask_grass |= (H_ch > lo) & (H_ch < hi) & (S_ch > 40)
    mask_skin = (H_ch < 20) & (S_ch > 30) & (V_ch > 90)
    valid = pixel_mask & ~(mask_grass | mask_skin)
    n_valid = int(valid.sum())
    if n_valid < 5:
        return None  # señal de que no hay píxeles suficientes
    h_v = H_ch[valid]; s_v = S_ch[valid]; v_v = V_ch[valid]
    colored = s_v > 30
    if colored.sum() >= 3:
        hist, _ = np.histogram(h_v[colored], bins=_HUE_BINS, range=(0, 180))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
    else:
        hist = np.zeros(_HUE_BINS, dtype=np.float32)
    white_frac = float((v_v > 200).sum()) / n_valid
    dark_frac  = float((v_v < 60).sum())  / n_valid
    return np.concatenate([hist, [white_frac, dark_frac]]).astype(np.float32)


def get_hue_signature_no_mask(crop: np.ndarray, grass_hue=None) -> np.ndarray:
    """
    Firma hue sin máscara SAM. Usa el torso crop (tercio central) con el mismo
    filtrado hierba/piel que get_hue_signature, manteniendo consistencia con
    los prototipos generados por generar_prototipos_v3.py sin SAM2.
    """
    torso = get_torso_crop(crop)
    if torso.size == 0:
        torso = crop
    return get_hue_signature(torso, grass_hue)


def get_hue_signature_masked(crop: np.ndarray, mask, grass_hue=None) -> np.ndarray:
    """Firma hue usando máscara SAM de silueta completa."""
    if crop.size == 0:
        return np.zeros(_HUE_BINS + 2, dtype=np.float32)
    if mask is None:
        return get_hue_signature_no_mask(crop, grass_hue)
    h, w = crop.shape[:2]
    mh, mw = mask.shape[:2]
    if mh != h or mw != w:
        mask_r = cv2.resize(mask.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask_r = mask.astype(bool)
    img  = cv2.resize(crop, (64, 64))
    m64  = cv2.resize(mask_r.astype(np.uint8), (64, 64),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H_ch = hsv[:, :, 0].astype(np.float32)
    S_ch = hsv[:, :, 1].astype(np.float32)
    V_ch = hsv[:, :, 2].astype(np.float32)
    sig  = _hue_sig_from_pixels(H_ch, S_ch, V_ch, m64, grass_hue)
    if sig is None:
        # máscara SAM vacía → fallback a crop completo filtrado
        return get_hue_signature_no_mask(crop, grass_hue)
    return sig


def hue_sig_distance(sig1, sig2) -> float:
    """Distancia Hellinger 70% (bins hue) + L1 30% (white/dark fracs)."""
    if sig1 is None or sig2 is None:
        return 1.0
    s1 = np.array(sig1, dtype=np.float32)
    s2 = np.array(sig2, dtype=np.float32)
    if s1.ndim == 0 or s2.ndim == 0 or len(s1) < _HUE_BINS or len(s2) < _HUE_BINS:
        return 1.0
    p  = np.clip(s1[:_HUE_BINS], 0, None)
    q  = np.clip(s2[:_HUE_BINS], 0, None)
    hell  = float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2))
    l1_wd = float(np.abs(s1[_HUE_BINS:_HUE_BINS + 2] - s2[_HUE_BINS:_HUE_BINS + 2]).sum() / 2.0)
    return 0.70 * hell + 0.30 * l1_wd


def hex_to_lab(hex_color: str):
    """Convierte color hex '#rrggbb' a array LAB OpenCV [L, a, b]."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0].tolist()


def _bgr_desc_to_lab(bgr_list) -> np.ndarray:
    """Convierte color_desc [B, G, R] a array LAB."""
    b, g, r = int(bgr_list[0]), int(bgr_list[1]), int(bgr_list[2])
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)


def bgr_similarity(bgr_desc, bgr_top3) -> float:
    """
    Score de similitud [0, 1] entre color_desc de una detección y bgr_top3 de un prototipo.
    bgr_top3: list of up to 3 [B,G,R] entries (colores dominantes del prototipo).
    Devuelve 1 − (distancia_CIE76_minima / 60), clampeado a [0, 1].
    """
    if bgr_desc is None or not bgr_top3:
        return 0.5
    det_lab = _bgr_desc_to_lab(bgr_desc)
    dists = [float(np.linalg.norm(det_lab - _bgr_desc_to_lab(c))) for c in bgr_top3]
    best = min(dists)
    return float(np.clip(1.0 - best / 60.0, 0.0, 1.0))


def classify_gk_score(det: dict,
                      gk_home_proto: dict,
                      gk_away_proto: dict,
                      home_attacks_left: bool,
                      field_l: float = L_M,
                      field_a: float = A_M,
                      pos_weight: float = 0.65,
                      color_weight: float = 0.35,
                      min_color_gate: float = 0.25,
                      min_combined: float = 0.55) -> str | None:
    """
    Clasifica una detección como gk_home / gk_away usando posición + color.

    Devuelve la clase ganadora si el score combinado supera min_combined,
    o None si la detección no parece un portero.

    Se usa cuando classify_player_v3 devuelve 'unknown' o un score > max_dist
    del prototipo.
    """
    pos     = det.get('pos_m')
    bgr     = det.get('color_desc')
    results = {}

    for side, proto in [('gk_home', gk_home_proto), ('gk_away', gk_away_proto)]:
        if proto is None:
            results[side] = 0.0
            continue

        # Goal x según qué equipo ataca por qué lado
        if side == 'gk_home':
            goal_x = field_l if home_attacks_left else 0.0
        else:
            goal_x = 0.0 if home_attacks_left else field_l

        # Position score
        if pos is not None:
            dx      = abs(pos[0] - goal_x)
            pa_y_ok = (field_a - 40.32) / 2 <= pos[1] <= (field_a + 40.32) / 2
            if dx <= _PA_DEPTH_M and pa_y_ok:
                pos_score = 1.0
            else:
                pos_score = 0.0
        else:
            pos_score = 0.0

        # Color score — proto usa 'top3' (PKL v3/v4) o 'bgr_top3' (nombre alternativo)
        top3 = proto.get('top3') or proto.get('bgr_top3') or []
        color_score = bgr_similarity(bgr, top3) if bgr is not None else 0.5

        if color_score < min_color_gate:
            results[side] = 0.0
            continue

        results[side] = pos_weight * pos_score + color_weight * color_score

    if not results:
        return None
    best = max(results, key=results.get)
    return best if results[best] >= min_combined else None


def classify_player_v3(crop: np.ndarray, proto_dict: dict,
                       grass_hue=None, mask=None,
                       max_score: float = CLASSIFY_V3_MAX_SCORE,
                       min_margin: float = CLASSIFY_V3_MARGIN,
                       double_check: bool = True):
    """
    Clasifica con hue signatures v3 (Hellinger distance).
    Returns (cls, score, margin, scores_dict).

    double_check=True  → si margen bajo/unknown, hace Pass 2 BGR (Option B).
    double_check=False → devuelve resultado puro hue (usado en generar_prototipos).
    """
    if mask is not None:
        sig = get_hue_signature_masked(crop, mask, grass_hue)
    else:
        sig = get_hue_signature_no_mask(crop, grass_hue)

    scores = {cls: hue_sig_distance(sig, np.asarray(proto['hue_sig']))
              for cls, proto in proto_dict.items()
              if 'hue_sig' in proto}

    if not scores:
        cls_v1, score_v1 = classify_player(crop, proto_dict, grass_hue)
        return cls_v1, score_v1, 0.0, {}

    sorted_s = sorted(scores.items(), key=lambda x: x[1])
    best_cls, best_score = sorted_s[0]
    second_score = sorted_s[1][1] if len(sorted_s) > 1 else best_score
    margin = second_score - best_score

    # Umbral adaptativo por proto (generado en generar_prototipos_v3.py como percentil 90)
    # Si el PKL no tiene max_dist (PKLs antiguos), usa el parámetro global como fallback
    proto_max_dist = proto_dict[best_cls].get('max_dist', max_score) if best_cls in proto_dict else max_score

    # Pass 1 result
    p1_uncertain = best_score > proto_max_dist or margin < min_margin
    # Relaxed: si el margen es alto y el score no supera mucho el umbral, asignar igual.
    # Cubre jugadores en condiciones de luz distintas al PKL sin cambiar los prototipos.
    if p1_uncertain and margin >= 0.10 and best_score <= proto_max_dist * 1.4:
        p1_uncertain = False
    p1_cls = 'unknown' if p1_uncertain else best_cls

    # Pass 2: BGR double-check cuando hay incertidumbre (Option B)
    if double_check and (p1_uncertain or margin < CLASSIFY_V3_CONFIDENT_MARGIN):
        # SAM mask + torso crop para consistencia con cómo se construyeron los prototipos
        if mask is not None:
            msk = cv2.resize(mask.astype(np.uint8),
                             (crop.shape[1], crop.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            crop_dc = crop.copy(); crop_dc[msk == 0] = 0
        else:
            crop_dc = crop
        cls_v1, score_v1 = classify_player(crop_dc, proto_dict, grass_hue)
        if cls_v1 != 'unknown':
            return cls_v1, score_v1, margin, scores

    return p1_cls, best_score, margin, scores


def extract_sam_masks_batch(cam_img: np.ndarray, bboxes_list: list,
                             sam_model, device: str = 'cpu') -> list:
    """
    Extrae máscaras SAM2 para una lista de bboxes.
    Devuelve lista de arrays bool o Nones si sam_model es None.
    """
    if sam_model is None or not bboxes_list:
        return [None] * len(bboxes_list)
    try:
        bboxes_np = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2 in bboxes_list],
                              dtype=np.float32)
        results = sam_model(cam_img, bboxes=bboxes_np, verbose=False, device=device)
        if not results or results[0].masks is None:
            return [None] * len(bboxes_list)
        masks_data = results[0].masks.data.cpu().numpy()
        h, w = cam_img.shape[:2]
        out = []
        for i in range(len(bboxes_list)):
            if i < len(masks_data):
                m = masks_data[i]
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    m = m.astype(bool)
                out.append(m)
            else:
                out.append(None)
        return out
    except Exception:
        return [None] * len(bboxes_list)


def apply_gk_by_position(dets: list,
                          home_attacks_left: bool = True,
                          gk_home_color_lab=None,   # list of LAB arrays (top3) or None
                          gk_away_color_lab=None,   # list of LAB arrays (top3) or None
                          gk_home_cls: str = 'gk_home',
                          gk_away_cls: str = 'gk_away',
                          field_l: float = L_M,
                          field_a: float = A_M) -> list:
    """
    Reclasifica porteros solo desde detecciones 'unknown'.

    Estrategia:
    - Con color de referencia: filtra unknowns por distancia CIE76 al color
      dado; la posición en el área penal suma -10 al score (bonus, no filtro).
      Se elige el unknown con menor score total.
    - Sin color de referencia: filtra unknowns dentro del área penal y elige
      el más cercano a la línea de gol.
    Solo actúa si el rol gk_* no está ya asignado en dets.

    home_attacks_left=True → portero home defiende lado derecho (x≈field_l).
    """
    has_gk_home = any(d.get('cls') == gk_home_cls for d in dets)
    has_gk_away = any(d.get('cls') == gk_away_cls for d in dets)
    if has_gk_home and has_gk_away:
        return dets

    result = [dict(d) for d in dets]

    _SA_DEPTH_M = 5.5   # área chica (6-yard box)
    _SA_Y_HALF  = 9.16 / 2  # mitad del ancho del área chica

    def _in_area(d, side):
        pos = d.get('pos_m')
        if pos is None:
            return False
        pa_y_min = (field_a - 40.32) / 2
        pa_y_max = (field_a + 40.32) / 2
        in_y = pa_y_min <= pos[1] <= pa_y_max
        in_x = (pos[0] <= _PA_DEPTH_M if side == 'left'
                else pos[0] >= field_l - _PA_DEPTH_M)
        return in_y and in_x

    def _in_small_area(d, side):
        pos = d.get('pos_m')
        if pos is None:
            return False
        cy = field_a / 2
        in_y = (cy - _SA_Y_HALF - 2) <= pos[1] <= (cy + _SA_Y_HALF + 2)
        in_x = (pos[0] <= _SA_DEPTH_M if side == 'left'
                else pos[0] >= field_l - _SA_DEPTH_M)
        return in_y and in_x

    # Promover unknowns + players del mismo equipo que estén en el área chica.
    # Solo se permite reclasificar player_home→gk_home y player_away→gk_away,
    # nunca cruzar equipo (evita que player_away sea absorbido como gk_home).
    def _promotable(d, goal_x, target_cls):
        cls = d.get('cls')
        if cls == 'unknown':
            return True
        side = 'left' if goal_x < field_l / 2 else 'right'
        if target_cls == gk_home_cls and cls == 'player_home':
            return _in_small_area(d, side)
        if target_cls == gk_away_cls and cls == 'player_away':
            return _in_small_area(d, side)
        return False

    def _pos_penalty(dist_from_goal: float) -> float:
        """Penalización cuadrática suave: 0 en la portería, crece hacia el centro.
        Escala: ~0 en área chica, ~40 en borde del área penal, ~115 a 22m."""
        return (dist_from_goal / _PA_DEPTH_M) ** 2 * 40.0

    def _find_by_color(color_top3_labs, side, goal_x, target_cls):
        # color_top3_labs: list of LAB arrays (one per top3 color)
        _pa_y_min = (field_a - 40.32) / 2
        _pa_y_max = (field_a + 40.32) / 2
        best_i, best_score = None, float('inf')
        for i, d in enumerate(result):
            if not _promotable(d, goal_x, target_cls) or d.get('color_desc') is None:
                continue
            pos = d.get('pos_m')
            if pos is None:
                continue
            dist_from_goal = abs(pos[0] - goal_x)
            if dist_from_goal > 16.5:  # límite = área penal real (x)
                continue
            if not (_pa_y_min <= pos[1] <= _pa_y_max):  # fuera del ancho del área penal
                continue
            det_lab = _bgr_desc_to_lab(d['color_desc'])
            color_dist = min(float(np.linalg.norm(det_lab - ref)) for ref in color_top3_labs)
            score = color_dist + _pos_penalty(dist_from_goal)
            if score < best_score:
                best_score, best_i = score, i
        return best_i

    def _find_by_position(side, goal_x, target_cls):
        _pa_y_min = (field_a - 40.32) / 2
        _pa_y_max = (field_a + 40.32) / 2
        candidates = [
            (i, d) for i, d in enumerate(result)
            if _promotable(d, goal_x, target_cls)
            and d.get('pos_m') is not None
            and abs(d['pos_m'][0] - goal_x) <= 16.5        # dentro del área penal (x)
            and _pa_y_min <= d['pos_m'][1] <= _pa_y_max    # dentro del ancho del área penal (y)
        ]
        if not candidates:
            return None
        best_i, _ = min(candidates,
                        key=lambda t: _pos_penalty(abs(t[1]['pos_m'][0] - goal_x)))
        return best_i

    if home_attacks_left:
        home_goal_x, home_side = field_l, 'right'
        away_goal_x, away_side = 0.0,     'left'
    else:
        home_goal_x, home_side = 0.0,     'left'
        away_goal_x, away_side = field_l, 'right'

    for cls_gk, side, goal_x, color_ref, do_it in [
        (gk_home_cls, home_side, home_goal_x, gk_home_color_lab, not has_gk_home),
        (gk_away_cls, away_side, away_goal_x, gk_away_color_lab, not has_gk_away),
    ]:
        if not do_it:
            continue
        best_i = None
        if color_ref is not None:
            best_i = _find_by_color(color_ref, side, goal_x, cls_gk)
        if best_i is None:  # sin color ref o sin candidatos con color_desc → posición
            best_i = _find_by_position(side, goal_x, cls_gk)
        if best_i is not None:
            result[best_i]['cls'] = cls_gk

    return result


# ═══════════════════════════════════════════════════════════
#  5. BRILLO Y SELECCIÓN DE PKL
# ═══════════════════════════════════════════════════════════

def get_grass_hue(frame: np.ndarray):
    hsv  = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85)
            & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40))
    if mask.sum() < 100:
        return None
    return float(np.median(hsv[:, :, 0][mask]))


def frame_brightness(frame: np.ndarray) -> float:
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


def detect_lighting(frame: np.ndarray, bright_thresh=None) -> str:
    thresh = bright_thresh if bright_thresh is not None else BRIGHT_THRESH
    return 'day' if frame_brightness(frame) > thresh else 'night'


def select_protos(frame: np.ndarray, protos_day, protos_night,
                  bright_thresh=None) -> dict:
    if protos_day is None and protos_night is None:
        raise ValueError('Ningun PKL cargado')
    if protos_day   is None: return protos_night
    if protos_night is None: return protos_day
    cond = detect_lighting(frame, bright_thresh)
    return protos_day if cond == 'day' else protos_night


# ═══════════════════════════════════════════════════════════
#  6. HOMOGRAFÍA
# ═══════════════════════════════════════════════════════════

def esquinas_metros(H, w_px: int, h_px: int) -> np.ndarray:
    pts = np.array([[[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]]], dtype=np.float32)
    return cv2.perspectiveTransform(pts, H)[0]


def pixel_to_field(x_px: float, y_px: float, H) -> tuple:
    pt = np.array([[[float(x_px), float(y_px)]]], dtype=np.float32)
    pm = cv2.perspectiveTransform(pt, H)[0][0]
    return float(pm[0]), float(pm[1])


# ═══════════════════════════════════════════════════════════
#  7. DIBUJO DEL CAMPO (matplotlib)
# ═══════════════════════════════════════════════════════════

def draw_pitch_mpl(ax, lm=None, am=None, limite_x=None, L=None, A=None):
    lm = lm if lm is not None else (L if L is not None else L_M)
    am = am if am is not None else (A if A is not None else A_M)
    ax.set_facecolor(GRASS_DARK)
    ax.set_xlim(-1, lm + 1)
    ax.set_ylim(am + 1, -1)
    ax.set_aspect('equal')
    ax.axis('off')

    stripe_w = lm / 12
    for i in range(12):
        c = GRASS_LIGHT if i % 2 == 0 else GRASS_DARK
        ax.add_patch(Rectangle((i * stripe_w, 0), stripe_w, am, color=c, zorder=0))

    lw = 1.4
    kw = dict(color=LINE_COLOR, lw=lw, zorder=2)

    def line(x0, y0, x1, y1): ax.plot([x0, x1], [y0, y1], **kw)
    def rect(x, y, w, h):     ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=LINE_COLOR, lw=lw, zorder=2))
    def circle(cx, cy, r):    ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=LINE_COLOR, lw=lw, zorder=2))
    def dot(cx, cy, r=0.4):   ax.add_patch(Circle((cx, cy), r, color=LINE_COLOR, zorder=3))
    def arc(cx, cy, r, t1, t2): ax.add_patch(Arc((cx, cy), r*2, r*2, angle=0, theta1=t1, theta2=t2, color=LINE_COLOR, lw=lw, zorder=2))

    rect(0, 0, lm, am)
    line(lm/2, 0, lm/2, am)
    circle(lm/2, am/2, 9.15)
    dot(lm/2, am/2, 0.35)

    for side, x0 in [('L', 0), ('R', lm)]:
        sign = 1 if side == 'L' else -1
        rect(x0 if side == 'L' else x0 - 16.5, am/2 - 40.32/2, 16.5, 40.32)
        rect(x0 if side == 'L' else x0 - 5.5,  am/2 - 18.32/2, 5.5,  18.32)
        px_ = 11 if side == 'L' else lm - 11
        dot(px_, am/2, 0.35)
        a1, a2 = (-53, 53) if side == 'L' else (127, 233)
        arc(px_, am/2, 9.15, a1, a2)

    for cx, cy, t1, t2 in [(0, 0, 0, 90), (lm, 0, 90, 180), (0, am, 270, 360), (lm, am, 180, 270)]:
        arc(cx, cy, 1, t1, t2)

    gw = 7.32 / 2
    for xg, xd in [(0, -2), (lm, lm + 2)]:
        ax.add_patch(Rectangle((min(xg, xd), am/2 - gw), 2, 2*gw,
                               fill=True, facecolor='#aaa', edgecolor=LINE_COLOR, lw=lw, zorder=2))

    if limite_x is not None:
        ax.axvline(limite_x, color='#4488ff', lw=1.5, ls='--', alpha=0.5, zorder=4)


# ═══════════════════════════════════════════════════════════
#  8. MINIMAP (render_minimap_frame + overlay_minimap)
# ═══════════════════════════════════════════════════════════

def render_minimap_frame(tracked: list, ball,
                         size_px: tuple = (400, 240),
                         L: float = 98.0, A: float = 61.0,
                         protos=None,
                         home_hex: str = '#e63946',
                         away_hex: str = '#1d70b8',
                         gk_home_hex: str = '#ffff00',
                         gk_away_hex: str = '#ff6600') -> np.ndarray:
    """Devuelve imagen BGR del mapa táctico con jugadores y balón."""
    w_px, h_px = size_px
    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor(DARK_BG)
    draw_pitch_mpl(ax, L, A)

    _cls_hex = {
        'player_home': home_hex,
        'player_away': away_hex,
        'gk_home':     gk_home_hex,
        'gk_away':     gk_away_hex,
        'referee':     '#FFD700',
        'unknown':     '#888888',
    }
    for d in tracked:
        pm = d.get('pos_m')
        if pm is None:
            continue
        xm, ym = pm
        if not (0 <= xm <= L and 0 <= ym <= A):
            continue
        cls   = d.get('cls', 'unknown')
        hex_c = _cls_hex.get(cls, '#888888')
        brightness = sum(int(hex_c[i:i+2], 16) for i in (1, 3, 5)) / 3
        edge_c = 'black' if brightness > 160 else 'white'
        # GKs: diamond marker to distinguish from outfield players
        marker = 'D' if cls in ('gk_home', 'gk_away') else 'o'
        ax.scatter(xm, ym, color=hex_c, s=60, zorder=5, marker=marker,
                   edgecolors=edge_c, linewidths=0.8)
        gid = d.get('gid')
        if gid is not None:
            ax.text(xm, ym - 1.5, f"#{gid}", color='white',
                    fontsize=5, ha='center', zorder=6, fontweight='bold')

    if ball is not None:
        bxm, bym = ball[0], ball[1]
        if 0 <= bxm <= L and 0 <= bym <= A:
            ax.scatter(bxm, bym, color=BALL_COLOR, s=80, marker='o',
                       zorder=7, edgecolors='white', linewidths=0.8)

    plt.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    out = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return cv2.resize(out, size_px)


def overlay_minimap(frame: np.ndarray, minimap: np.ndarray,
                    margin: int = 10, alpha: float = 0.85) -> np.ndarray:
    """Superpone minimap en la esquina inferior derecha del frame."""
    out = frame.copy()
    fh, fw = out.shape[:2]
    mh, mw = minimap.shape[:2]
    y0 = fh - mh - margin
    x0 = fw - mw - margin
    if y0 < 0 or x0 < 0:
        return out
    roi = out[y0:y0 + mh, x0:x0 + mw]
    cv2.addWeighted(minimap, alpha, roi, 1 - alpha, 0, roi)
    out[y0:y0 + mh, x0:x0 + mw] = roi
    return out


# ═══════════════════════════════════════════════════════════
#  9. TRACKER (KalmanFieldTracker v2 — idéntico al notebook)
# ═══════════════════════════════════════════════════════════

class _KFTrack:
    def __init__(self, gid, pos_m, cls, conf, frame_idx, color_desc=None, fps=30.0):
        self.gid        = gid
        self.conf       = conf
        self.last_frame = frame_idx
        self.created    = frame_idx
        self.hits       = 1
        self.fps        = fps
        self.class_history = defaultdict(int)
        self.class_history[cls] += 1
        self.cls        = cls
        self.color_desc = (np.array(color_desc, dtype=np.float32)
                           if color_desc is not None else None)
        self._color_alpha  = 0.15
        self._alt_cls_streak = 0  # frames consecutivos clasificando clase distinta

        dt = 1.0
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov   = np.diag([0.05, 0.05, 0.5, 0.5]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3
        self.kf.errorCovPost      = np.eye(4, dtype=np.float32)
        self.kf.statePost         = np.array([[pos_m[0]], [pos_m[1]], [0.0], [0.0]], dtype=np.float32)

    def predict(self):
        s = self.kf.predict()
        return np.array([float(s[0, 0]), float(s[1, 0])], dtype=np.float32)

    def correct(self, pos_m, cls, conf, frame_idx, color_desc=None):
        meas = np.array([[pos_m[0]], [pos_m[1]]], dtype=np.float32)
        self.kf.correct(meas)
        self.last_frame = frame_idx
        self.hits      += 1
        self.conf       = conf
        if cls != 'unknown':
            self.class_history[cls] += 1
            new_cls = max(self.class_history.items(), key=lambda kv: kv[1])[0]
            # Contundencia: N frames consecutivos contradiciendo la clase actual → override
            _player_cls = ('player_home', 'player_away')
            if cls in _player_cls and cls != self.cls and self.cls in _player_cls:
                self._alt_cls_streak += 1
                if self._alt_cls_streak >= CLASS_OVERRIDE_F:
                    self.cls = cls
                    self.class_history = defaultdict(int)
                    self.class_history[cls] = CLASS_OVERRIDE_F
                    self._alt_cls_streak = 0
            else:
                self._alt_cls_streak = 0
                self.cls = new_cls
        if color_desc is not None:
            c = np.array(color_desc, dtype=np.float32)
            if self.color_desc is None:
                self.color_desc = c
            else:
                self.color_desc = ((1 - self._color_alpha) * self.color_desc
                                   + self._color_alpha * c)

    def color_dist(self, color_desc) -> float:
        if self.color_desc is None or color_desc is None:
            return 0.0
        c = np.array(color_desc, dtype=np.float32)
        raw = float(np.linalg.norm(self.color_desc - c))
        return min(raw / 441.0 * COLOR_WEIGHT_M, COLOR_WEIGHT_M)

    @property
    def pos(self):
        return np.array([float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])], dtype=np.float32)

    @property
    def velocity(self):
        return np.array([float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])], dtype=np.float32)

    @property
    def speed_ms(self):
        return float(np.linalg.norm(self.velocity)) * self.fps


class KalmanFieldTracker:
    def __init__(self, fps=30.0, max_dist_m=TRACK_MAX_DIST_M,
                 window_s=TRACK_WINDOW_S, class_weight=TRACK_CLASS_W):
        self._next_id     = 1
        self._tracks      = {}
        self.fps          = fps
        self.max_dist     = max_dist_m
        self.window       = int(fps * window_s)
        self.class_weight = class_weight
        self.history      = defaultdict(list)

    def update(self, frame_idx: int, unified_dets: list) -> list:
        expired = [gid for gid, tr in self._tracks.items()
                   if frame_idx - tr.last_frame > self.window]
        for gid in expired:
            del self._tracks[gid]

        valid_dets = [d for d in unified_dets if d.get('pos_m') is not None]

        if not self._tracks or not valid_dets:
            result = []
            for d in valid_dets:
                if not self._should_create(d):
                    continue
                gid = self._new_track(frame_idx, d)
                result.append({**d, 'gid': gid,
                               'cls': self._tracks[gid].cls,
                               'speed_ms': 0.0})
            return result

        track_ids = list(self._tracks.keys())
        pred_pos  = {gid: self._tracks[gid].predict() for gid in track_ids}

        det_pos   = [np.array(d['pos_m'], dtype=np.float32) for d in valid_dets]
        det_cls   = [d['cls'] for d in valid_dets]
        det_color = [d.get('color_desc') for d in valid_dets]

        cost = np.full((len(valid_dets), len(track_ids)), np.inf)
        for i, (dp, dc, dcolor) in enumerate(zip(det_pos, det_cls, det_color)):
            for j, gid in enumerate(track_ids):
                tr   = self._tracks[gid]
                dist = float(np.linalg.norm(dp - pred_pos[gid]))
                if dist > self.max_dist:
                    continue
                class_pen = 0.0
                if dc != 'unknown' and dc != tr.cls and tr.hits >= MIN_HITS_CLASS:
                    class_pen = self.class_weight
                color_pen = 0.0
                gap = frame_idx - tr.last_frame
                if gap >= GAP_COLOR_FRAMES:
                    color_pen = tr.color_dist(dcolor)
                cost[i, j] = dist + class_pen + color_pen

        assigned_dets = {}
        assigned_ids  = set()
        while True:
            if cost.size == 0 or cost.min() == np.inf:
                break
            idx = np.unravel_index(np.argmin(cost), cost.shape)
            i, j = int(idx[0]), int(idx[1])
            if cost[i, j] == np.inf:
                break
            gid = track_ids[j]
            assigned_dets[i] = gid
            assigned_ids.add(j)
            cost[i, :] = np.inf
            cost[:, j] = np.inf

        result = []
        for i, d in enumerate(valid_dets):
            if i in assigned_dets:
                gid = assigned_dets[i]
                self._tracks[gid].correct(
                    d['pos_m'], d['cls'], d['conf'], frame_idx,
                    color_desc=d.get('color_desc'))
            else:
                if not self._should_create(d):
                    continue
                gid = self._new_track(frame_idx, d)
            voted_cls = self._tracks[gid].cls
            speed     = self._tracks[gid].speed_ms
            self._record(gid, frame_idx, d['pos_m'], voted_cls)
            result.append({**d, 'gid': gid, 'cls': voted_cls, 'speed_ms': speed})

        for j, gid in enumerate(track_ids):
            if j not in assigned_ids:
                self._tracks[gid].predict()

        return result

    def _should_create(self, d) -> bool:
        if d['cls'] != 'unknown':
            return True
        return d.get('conf', 0.0) >= MIN_CONF_NEW_UNKNOWN

    def _new_track(self, frame_idx: int, d: dict) -> int:
        gid = self._next_id
        self._next_id += 1
        self._tracks[gid] = _KFTrack(
            gid, d['pos_m'], d['cls'], d['conf'], frame_idx,
            color_desc=d.get('color_desc'), fps=self.fps)
        self._record(gid, frame_idx, d['pos_m'], self._tracks[gid].cls)
        return gid

    def _record(self, gid, frame_idx, pos_m, cls):
        speed = self._tracks[gid].speed_ms if gid in self._tracks else 0.0
        self.history[gid].append(
            (frame_idx, float(pos_m[0]), float(pos_m[1]), cls, round(speed, 2)))

    def get_trajectory(self, gid):
        return self.history.get(gid, [])

    def active_ids(self):
        return list(self._tracks.keys())

    def get_speed(self, gid):
        tr = self._tracks.get(gid)
        return tr.speed_ms if tr else 0.0


# ═══════════════════════════════════════════════════════════
#  9b. StrongKalmanTrackerV3
#  Kalman + Hungarian + Two-stage + Lost buffer + Class-cap
# ═══════════════════════════════════════════════════════════

# ── V3 defaults (sobreescribibles por config) ───────────────
V3_HIGH_CONF        = 0.50
V3_LOW_CONF         = 0.35
V3_LOST_S           = 8.0    # ampliado 4→8s: cubre corners, saques de banda y oclusiones largas
V3_GAP_HUE          = 5
V3_HUE_W            = 4.0
V3_MAX_TOTAL        = 35
V3_MAX_CLASS        = {'player_home': 12, 'player_away': 12,
                        'gk_home': 1, 'gk_away': 1, 'referee': 4}
V3_MIN_HITS_CONFIRM = 3      # 2→3: menos tracks falsos confirmados en galería
V3_GALLERY_TTL_S    = 120.0  # ampliado 30→120s: recupera jugadores tras ausencias prolongadas
V3_REID_HUE_THRESH  = 0.33   # distancia Hellinger máxima para Re-ID
V3_REID_MAX_DIST_M  = 25.0   # distancia espacial máxima para Re-ID (m)


class _KFTrackV2:
    def __init__(self, gid, pos_m, cls, conf, frame_idx,
                 color_desc=None, hue_sig=None, fps=30.0, cam='?'):
        self.gid           = gid
        self.conf          = conf
        self.last_frame    = frame_idx
        self.created       = frame_idx
        self.hits          = 1
        self.fps           = fps
        self.is_confirmed  = False
        self.class_history = defaultdict(int)
        self.class_history[cls] += 1
        self.cls    = cls
        self.cam    = cam   # última cámara que observó esta detección
        self._alpha = 0.25
        self.color_desc = (np.array(color_desc, dtype=np.float32)
                           if color_desc is not None else None)
        self.hue_sig = (np.array(hue_sig, dtype=np.float32)
                        if hue_sig is not None else None)
        dt = 1.0 / fps
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix    = np.array([[1,0,dt,0],[0,1,0,dt],
                                                 [0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        # Q en m/s: pos noise 0.05m, vel noise 0.5 m/s
        self.kf.processNoiseCov     = np.diag([0.05, 0.05, 0.5, 0.5]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.kf.statePost           = np.array([[pos_m[0]], [pos_m[1]],
                                                 [0.], [0.]], np.float32)

    def predict(self):
        s = self.kf.predict()
        return np.array([float(s[0, 0]), float(s[1, 0])], np.float32)

    def correct(self, pos_m, cls, conf, frame_idx, color_desc=None, hue_sig=None, cam='?'):
        self.kf.correct(np.array([[pos_m[0]], [pos_m[1]]], np.float32))
        self.last_frame = frame_idx
        self.hits      += 1
        self.conf       = conf
        if cam != '?':
            self.cam = cam
        if self.hits >= V3_MIN_HITS_CONFIRM:
            self.is_confirmed = True
        if cls != 'unknown':
            self.class_history[cls] += 1
            self.cls = max(self.class_history.items(), key=lambda kv: kv[1])[0]
        def _ema(cur, new):
            if new is None: return cur
            v = np.array(new, dtype=np.float32)
            return v if cur is None else (1 - self._alpha) * cur + self._alpha * v
        self.color_desc = _ema(self.color_desc, color_desc)
        self.hue_sig    = _ema(self.hue_sig, hue_sig)

    def hue_dist(self, hue_sig_det, w) -> float:
        if self.hue_sig is None or hue_sig_det is None: return 0.0
        return min(hue_sig_distance(self.hue_sig, hue_sig_det) * w, w)

    def color_dist_bgr(self, color_desc, w=COLOR_WEIGHT_M) -> float:
        if self.color_desc is None or color_desc is None: return 0.0
        raw = float(np.linalg.norm(
            self.color_desc - np.array(color_desc, dtype=np.float32)))
        return min(raw / 441.0 * w, w)

    @property
    def pos(self):
        return np.array([float(self.kf.statePost[0, 0]),
                         float(self.kf.statePost[1, 0])], np.float32)

    @property
    def speed_ms(self):
        vx = float(self.kf.statePost[2, 0])
        vy = float(self.kf.statePost[3, 0])
        return float(np.sqrt(vx ** 2 + vy ** 2))


class StrongKalmanTrackerV3:
    """
    Kalman 2D + Hungarian + two-stage matching + lost buffer + class-cap + Re-ID gallery.
    """
    def __init__(self, fps=30.0, max_dist_m=TRACK_MAX_DIST_M,
                 window_s=TRACK_WINDOW_S, class_weight=TRACK_CLASS_W,
                 lost_s=V3_LOST_S,
                 gallery_ttl_s=V3_GALLERY_TTL_S,
                 reid_hue_thresh=V3_REID_HUE_THRESH,
                 reid_max_dist_m=V3_REID_MAX_DIST_M):
        self._next_id       = 1
        self._tracked       = {}
        self._lost          = {}
        self._gallery       = {}   # gid -> {cls, pos, hue_sig, color_desc, last_frame}
        self.fps            = fps
        self.max_dist       = max_dist_m
        self.window         = int(fps * window_s)
        self.lost_win       = int(fps * lost_s)
        self.gallery_ttl    = int(fps * gallery_ttl_s)
        self.reid_hue_th    = reid_hue_thresh
        self.reid_max_dist  = reid_max_dist_m
        self.class_w        = class_weight
        self.history        = defaultdict(list)

    def _cost_matrix(self, tracks_dict, dets, frame_idx):
        gids = list(tracks_dict.keys())
        C = np.full((len(gids), len(dets)), 1e6, dtype=np.float32)
        for i, gid in enumerate(gids):
            tr   = tracks_dict[gid]
            pred = tr.pos
            for j, d in enumerate(dets):
                pm = d.get('pos_m')
                if pm is None or None in pm: continue
                dist = float(np.linalg.norm(np.array(pm, np.float32) - pred))
                if dist > self.max_dist: continue
                pen  = 0.0
                dc   = d.get('cls', 'unknown')
                if dc != 'unknown' and dc != tr.cls and tr.hits >= MIN_HITS_CLASS:
                    pen += self.class_w
                gap = frame_idx - tr.last_frame
                if gap >= GAP_COLOR_FRAMES:
                    pen += tr.color_dist_bgr(d.get('color_desc'))
                if gap >= V3_GAP_HUE:
                    pen += tr.hue_dist(d.get('hue_sig'), V3_HUE_W)
                C[i, j] = dist + pen
        return gids, C

    def _match(self, tracks_dict, dets, frame_idx):
        if not tracks_dict or not dets:
            return [], list(tracks_dict.keys()), list(range(len(dets)))
        gids, C = self._cost_matrix(tracks_dict, dets, frame_idx)
        ri, ci  = linear_sum_assignment(C)
        matched, used_t, used_d = [], set(), set()
        for r, c in zip(ri, ci):
            if C[r, c] < 1e5:
                matched.append((gids[r], c)); used_t.add(r); used_d.add(c)
        um_g = [gids[i] for i in range(len(gids)) if i not in used_t]
        um_d = [j       for j in range(len(dets))  if j not in used_d]
        return matched, um_g, um_d

    def _class_count(self, cls):
        return sum(1 for tr in list(self._tracked.values()) + list(self._lost.values())
                   if tr.cls == cls)

    def _should_create(self, d):
        cls = d.get('cls', 'unknown')
        if len(self._tracked) + len(self._lost) >= V3_MAX_TOTAL:
            return False
        if cls in V3_MAX_CLASS and self._class_count(cls) >= V3_MAX_CLASS[cls]:
            return False
        if cls == 'unknown':
            return d.get('conf', 0.0) >= MIN_CONF_NEW_UNKNOWN
        return True

    def _try_reid(self, d, frame_idx):
        """Busca en la galería el track más compatible. Devuelve gid o None."""
        if not self._gallery:
            return None
        pm = np.array(d['pos_m'], np.float32)
        dc = d.get('cls', 'unknown')
        hue_det = d.get('hue_sig')
        best_gid, best_cost = None, float('inf')
        for gid, e in self._gallery.items():
            # La clase debe coincidir (unknown es comodín)
            if dc != 'unknown' and e['cls'] != 'unknown' and dc != e['cls']:
                continue
            dist = float(np.linalg.norm(pm - np.array(e['pos'], np.float32)))
            if dist > self.reid_max_dist:
                continue
            # Distancia de tono (Hellinger); si no hay firma disponible, solo distancia
            hue_d = 0.0
            if hue_det is not None and e['hue_sig'] is not None:
                hue_d = hue_sig_distance(
                    np.array(e['hue_sig'], np.float32),
                    np.array(hue_det, np.float32))
                if hue_d > self.reid_hue_th:
                    continue
            cost = dist / self.reid_max_dist + hue_d
            if cost < best_cost:
                best_cost, best_gid = cost, gid
        return best_gid

    def _restore_from_gallery(self, old_gid, d, frame_idx):
        """Crea un track con el gid antiguo a partir de una entrada de galería."""
        e = self._gallery.pop(old_gid)
        tr = _KFTrackV2(old_gid, d['pos_m'], d.get('cls', e['cls']),
                        d.get('conf', 1.0), frame_idx,
                        color_desc=e['color_desc'],
                        hue_sig=e['hue_sig'], fps=self.fps,
                        cam=d.get('cam', '?'))
        # Restaurar firma acumulada y marcar como confirmado de inmediato
        tr.hits         = max(V3_MIN_HITS_CONFIRM, e.get('hits', V3_MIN_HITS_CONFIRM))
        tr.is_confirmed = True
        if e['hue_sig'] is not None:
            tr.hue_sig = np.array(e['hue_sig'], np.float32)
        if e['color_desc'] is not None:
            tr.color_desc = np.array(e['color_desc'], np.float32)
        return tr

    def update(self, frame_idx, unified_dets):
        # Expirar _lost → galería (en vez de borrar directamente)
        for gid in [g for g, t in self._lost.items()
                    if frame_idx - t.last_frame > self.lost_win]:
            tr = self._lost.pop(gid)
            if tr.is_confirmed:
                self._gallery[gid] = {
                    'cls':        tr.cls,
                    'pos':        tr.pos.tolist(),
                    'hue_sig':    tr.hue_sig.tolist() if tr.hue_sig is not None else None,
                    'color_desc': tr.color_desc.tolist() if tr.color_desc is not None else None,
                    'last_frame': tr.last_frame,
                    'hits':       tr.hits,
                }

        # Expirar _tracked (raro; normalmente pasan por _lost antes)
        self._tracked = {g: t for g, t in self._tracked.items()
                         if frame_idx - t.last_frame <= self.window}

        # Expirar galería
        for gid in [g for g, e in self._gallery.items()
                    if frame_idx - e['last_frame'] > self.gallery_ttl]:
            del self._gallery[gid]

        valid = [d for d in unified_dets
                 if d.get('pos_m') is not None and None not in d['pos_m']]
        high  = [d for d in valid if d.get('conf', 0) >= V3_HIGH_CONF]
        low   = [d for d in valid if V3_LOW_CONF <= d.get('conf', 0) < V3_HIGH_CONF]

        for tr in self._tracked.values(): tr.predict()
        for tr in self._lost.values():    tr.predict()

        # Stage 1: high-conf <-> tracked
        m1, um_t1, um_h1 = self._match(self._tracked, high, frame_idx)
        for gid, di in m1:
            d = high[di]
            self._tracked[gid].correct(d['pos_m'], d.get('cls', 'unknown'),
                                       d.get('conf', 1.0), frame_idx,
                                       color_desc=d.get('color_desc'),
                                       hue_sig=d.get('hue_sig'),
                                       cam=d.get('cam', '?'))
        for gid in um_t1:
            self._lost[gid] = self._tracked.pop(gid)

        # Stage 2: unmatched high + low <-> lost
        s2   = [high[j] for j in um_h1] + low
        n_hi = len(um_h1)
        m2, _, um_s2 = self._match(self._lost, s2, frame_idx)
        for gid, di in m2:
            d = s2[di]
            self._lost[gid].correct(d['pos_m'], d.get('cls', 'unknown'),
                                    d.get('conf', 1.0), frame_idx,
                                    color_desc=d.get('color_desc'),
                                    hue_sig=d.get('hue_sig'),
                                    cam=d.get('cam', '?'))
            self._tracked[gid] = self._lost.pop(gid)

        # Stage 3: todos los unmatched intentan Re-ID; solo high-conf crea tracks nuevos
        for di in um_s2:
            is_high = di < n_hi
            d = s2[di]
            if len(self._tracked) + len(self._lost) >= V3_MAX_TOTAL:
                break
            old_gid = self._try_reid(d, frame_idx)
            if old_gid is not None:
                tr = self._restore_from_gallery(old_gid, d, frame_idx)
                self._tracked[old_gid] = tr
            elif is_high and self._should_create(d):
                gid = self._next_id; self._next_id += 1
                tr  = _KFTrackV2(gid, d['pos_m'], d.get('cls', 'unknown'),
                                 d.get('conf', 1.0), frame_idx,
                                 color_desc=d.get('color_desc'),
                                 hue_sig=d.get('hue_sig'), fps=self.fps,
                                 cam=d.get('cam', '?'))
                self._tracked[gid] = tr

        result = []
        for gid, tr in self._tracked.items():
            if not tr.is_confirmed: continue
            self.history[gid].append(
                (frame_idx, float(tr.pos[0]), float(tr.pos[1]), tr.cls,
                 round(tr.speed_ms, 2)))
            result.append({'gid': gid, 'cls': tr.cls, 'speed_ms': tr.speed_ms,
                           'pos_m': tuple(tr.pos), 'conf': tr.conf,
                           'color_desc': tr.color_desc.tolist() if tr.color_desc is not None else None,
                           'cam': tr.cam})
        return result

    def active_ids(self):
        return set(self._tracked.keys())

    def get_trajectory(self, gid):
        return self.history.get(gid, [])


# ═══════════════════════════════════════════════════════════
#  9b. ByteFieldTracker (ByteTrack + Re-ID gallery en metros)
# ═══════════════════════════════════════════════════════════

# ByteTrack defaults (sobreescribibles por config)
BT_HIGH_THRESH    = 0.38
BT_LOW_THRESH     = 0.30
BT_MATCH_THRESH_M = 5.0
BT_CLASS_PENALTY  = 8.0
BT_MAX_LOST_S     = 3.0
BT_MIN_HITS       = 2
BT_GALLERY_TTL_S  = 20.0
BT_REID_HUE_THRESH= 0.33
BT_REID_MAX_DIST_M= 20.0
BT_HUE_EMA_ALPHA  = 0.20


class _BTrack:
    TRACKED = "tracked"; LOST = "lost"; REMOVED = "removed"

    def __init__(self, gid, det, frame_idx, fps):
        self.gid = gid; self.fps = fps
        self.state = self.TRACKED; self.hits = 1; self.age = 0
        self.lost_since = None; self.last_frame = frame_idx; self.cls_votes = {}
        self.cam = det.get("cam", "?")
        self._update_cls(det.get("cls", "unknown"))
        raw_sig = det.get("hue_sig")
        self.hue_sig = np.array(raw_sig, dtype=np.float32) if raw_sig is not None else None
        self.kf_x = np.array([det["pos_m"][0], det["pos_m"][1], 0.0, 0.0], dtype=np.float64)
        dt = 1.0 / fps
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        q = 0.5
        self.Q = np.eye(4) * q; self.Q[2,2] = q*4; self.Q[3,3] = q*4
        self.R = np.eye(2) * 0.5; self.P = np.eye(4) * 5.0

    def predict(self):
        self.kf_x = self.F @ self.kf_x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1

    def update(self, det, frame_idx):
        z = np.array([det["pos_m"][0], det["pos_m"][1]], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + K @ (z - self.H @ self.kf_x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.hits += 1; self.lost_since = None; self.state = self.TRACKED
        self.last_frame = frame_idx
        self._update_cls(det.get("cls", "unknown"))
        if "cam" in det:
            self.cam = det["cam"]
        raw_sig = det.get("hue_sig")
        if raw_sig is not None:
            new_sig = np.array(raw_sig, dtype=np.float32)
            self.hue_sig = new_sig if self.hue_sig is None else (
                (1 - BT_HUE_EMA_ALPHA) * self.hue_sig + BT_HUE_EMA_ALPHA * new_sig)

    def mark_lost(self, frame_idx):
        self.state = self.LOST
        if self.lost_since is None: self.lost_since = frame_idx

    def _update_cls(self, cls):
        if cls and cls != "unknown":
            self.cls_votes[cls] = self.cls_votes.get(cls, 0) + 1

    def to_gallery_entry(self, frame_idx):
        return {"hue_sig": self.hue_sig.copy() if self.hue_sig is not None else None,
                "cls": self.cls, "last_pos": self.pos,
                "last_frame": frame_idx, "n_hits": self.hits}

    @property
    def pos(self):
        return float(self.kf_x[0]), float(self.kf_x[1])

    @property
    def velocity(self):
        return float(self.kf_x[2]), float(self.kf_x[3])

    @property
    def speed_ms(self):
        vx, vy = self.velocity
        return float(np.sqrt(vx**2 + vy**2))

    @property
    def cls(self):
        return max(self.cls_votes, key=self.cls_votes.get) if self.cls_votes else "unknown"

    def lost_frames(self, frame_idx):
        return 0 if self.lost_since is None else frame_idx - self.lost_since


class ByteFieldTracker:
    """ByteTrack en coordenadas de campo (metros) + Re-ID gallery por hue_sig."""

    def __init__(self, fps=30.0,
                 high_thresh=BT_HIGH_THRESH, low_thresh=BT_LOW_THRESH,
                 match_thresh_m=BT_MATCH_THRESH_M, class_penalty=BT_CLASS_PENALTY,
                 max_lost_s=BT_MAX_LOST_S, min_hits=BT_MIN_HITS,
                 gallery_ttl_s=BT_GALLERY_TTL_S, reid_hue_thresh=BT_REID_HUE_THRESH,
                 reid_max_dist_m=BT_REID_MAX_DIST_M):
        self.fps = fps; self.high_thresh = high_thresh; self.low_thresh = low_thresh
        self.match_thresh = match_thresh_m; self.class_penalty = class_penalty
        self.max_lost_f = int(max_lost_s * fps); self.min_hits = min_hits
        self.gallery_ttl_f = int(gallery_ttl_s * fps)
        self.reid_hue_thresh = reid_hue_thresh; self.reid_max_dist_m = reid_max_dist_m
        self._next_id = 1; self._tracks = []; self._gallery = {}
        self.history = defaultdict(list)

    def _cost_matrix(self, tracks, dets):
        C = np.full((len(tracks), len(dets)), 1e9)
        for i, tr in enumerate(tracks):
            for j, det in enumerate(dets):
                pm = det.get("pos_m")
                if pm is None or None in pm: continue
                dist = float(np.sqrt((tr.pos[0]-pm[0])**2 + (tr.pos[1]-pm[1])**2))
                if det.get("cls","unknown") not in ("unknown",) and det.get("cls") != tr.cls:
                    dist += self.class_penalty
                C[i,j] = dist
        return C

    def _match(self, tracks, dets):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        C = self._cost_matrix(tracks, dets)
        ri, ci = linear_sum_assignment(C)
        matched, um_t, um_d = [], [], []
        mt, md = set(), set()
        for r, c in zip(ri, ci):
            if C[r,c] > self.match_thresh:
                um_t.append(r); um_d.append(c)
            else:
                matched.append((r,c)); mt.add(r); md.add(c)
        um_t += [i for i in range(len(tracks)) if i not in mt]
        um_d += [j for j in range(len(dets))   if j not in md]
        return matched, list(set(um_t)), list(set(um_d))

    def _reid(self, det, frame_idx):
        cls = det.get("cls", "unknown"); pm = det.get("pos_m")
        if pm is None or None in pm: return None
        sig = det.get("hue_sig")

        # Caso 1: clase conocida + firma hue → Re-ID completo
        if cls != "unknown" and sig is not None:
            sig_arr = np.array(sig, dtype=np.float32)
            best_gid = None; best_d = self.reid_hue_thresh
            for gid, e in self._gallery.items():
                if e["cls"] != cls or e["hue_sig"] is None: continue
                if frame_idx - e["last_frame"] > self.gallery_ttl_f: continue
                if e["n_hits"] < self.min_hits: continue
                ep = e["last_pos"]
                if np.sqrt((pm[0]-ep[0])**2 + (pm[1]-ep[1])**2) > self.reid_max_dist_m: continue
                d = hue_sig_distance(sig_arr, e["hue_sig"])
                if d < best_d: best_d, best_gid = d, gid
            return best_gid

        # Caso 2: clase unknown → Re-ID solo espacial con umbral estricto (3 m)
        best_gid = None; best_dist = 3.0
        for gid, e in self._gallery.items():
            if frame_idx - e["last_frame"] > self.gallery_ttl_f: continue
            if e["n_hits"] < self.min_hits: continue
            ep = e["last_pos"]
            dist = float(np.sqrt((pm[0]-ep[0])**2 + (pm[1]-ep[1])**2))
            if dist < best_dist: best_dist, best_gid = dist, gid
        return best_gid

    def _add_to_gallery(self, tr, frame_idx):
        e = tr.to_gallery_entry(frame_idx)
        if e["hue_sig"] is not None: self._gallery[tr.gid] = e

    def _clean_gallery(self, frame_idx):
        for g in [g for g,e in self._gallery.items()
                  if frame_idx - e["last_frame"] > self.gallery_ttl_f]:
            del self._gallery[g]

    def _should_create(self, det):
        if det.get("cls", "unknown") != "unknown":
            return True
        return det.get("conf", 0.0) >= MIN_CONF_NEW_UNKNOWN

    def _record(self, gid, frame_idx, pos_m, cls, speed):
        self.history[gid].append(
            (frame_idx, float(pos_m[0]), float(pos_m[1]), cls, round(speed, 2)))

    def update(self, frame_idx, dets):
        for tr in self._tracks: tr.predict()
        valid = [d for d in dets if d.get("pos_m") and None not in d["pos_m"]]
        high = [d for d in valid if d.get("conf", 0) >= self.high_thresh]
        low  = [d for d in valid if self.low_thresh <= d.get("conf", 0) < self.high_thresh]

        t_tracked = [tr for tr in self._tracks if tr.state == _BTrack.TRACKED]
        t_lost    = [tr for tr in self._tracks if tr.state == _BTrack.LOST]

        # Ronda 1: tracked ↔ high-conf
        m1, um_t1, um_d1 = self._match(t_tracked, high)
        for r, c in m1: t_tracked[r].update(high[c], frame_idx)
        for r in um_t1: t_tracked[r].mark_lost(frame_idx)

        # Ronda 2: lost ↔ (unmatched high + all low)
        n_high_um = len(um_d1)  # primeros n índices en s2 son high-conf
        s2 = [high[j] for j in um_d1] + low
        m2, um_t2, um_d2 = self._match(t_lost, s2)
        for r, c in m2: t_lost[r].update(s2[c], frame_idx)
        for r in um_t2:
            if t_lost[r].lost_frames(frame_idx) > self.max_lost_f:
                self._add_to_gallery(t_lost[r], frame_idx)
                t_lost[r].state = _BTrack.REMOVED

        # Nuevas pistas: SOLO desde unmatched high-conf (no low-conf)
        for j in um_d2:
            det = s2[j]
            old_gid = self._reid(det, frame_idx)
            if old_gid is not None:
                tr = _BTrack(old_gid, det, frame_idx, self.fps)
                tr.hits = self.min_hits; self._tracks.append(tr)
                del self._gallery[old_gid]
            elif j < n_high_um and self._should_create(det):
                tr = _BTrack(self._next_id, det, frame_idx, self.fps)
                self._next_id += 1; self._tracks.append(tr)

        self._tracks = [tr for tr in self._tracks if tr.state != _BTrack.REMOVED]
        self._clean_gallery(frame_idx)

        result = []
        for tr in self._tracks:
            if tr.state == _BTrack.TRACKED and tr.hits >= self.min_hits:
                self._record(tr.gid, frame_idx, tr.pos, tr.cls, tr.speed_ms)
                result.append({
                    "gid": tr.gid, "pos_m": tr.pos, "cls": tr.cls,
                    "conf": 1.0, "speed_ms": tr.speed_ms,
                    "cam": getattr(tr, 'cam', '?'),
                })
        return result

    def active_ids(self):
        return {tr.gid for tr in self._tracks if tr.state == _BTrack.TRACKED}

    def get_trajectory(self, gid):
        return self.history.get(gid, [])

    def get_speed(self, gid):
        tr = next((t for t in self._tracks if t.gid == gid), None)
        return tr.speed_ms if tr else 0.0


# ═══════════════════════════════════════════════════════════
#  10. SEAM DEDUP + GLOBAL NMS
# ═══════════════════════════════════════════════════════════

def _unify_seam(dets_a: list, dets_b: list, cut_x: float,
                margin: float = SEAM_MARGIN_M,
                fusion_thr: float = SEAM_FUSION_M) -> list:
    from scipy.spatial.distance import cdist as _cdist

    finales = []
    zone_a_only = [d for d in dets_a
                   if d['pos_m'] is not None and d['pos_m'][0] < cut_x - margin]
    zone_b_only = [d for d in dets_b
                   if d['pos_m'] is not None and d['pos_m'][0] > cut_x + margin]
    finales.extend(zone_a_only)
    finales.extend(zone_b_only)

    front_a = [d for d in dets_a
               if d['pos_m'] is not None and abs(d['pos_m'][0] - cut_x) <= margin]
    front_b = [d for d in dets_b
               if d['pos_m'] is not None and abs(d['pos_m'][0] - cut_x) <= margin]

    if not front_a and not front_b:
        return finales
    if not front_a:
        finales.extend(front_b); return finales
    if not front_b:
        finales.extend(front_a); return finales

    pts_a = np.array([d['pos_m'] for d in front_a], dtype=np.float32)
    pts_b = np.array([d['pos_m'] for d in front_b], dtype=np.float32)
    D     = _cdist(pts_a, pts_b)

    used_a, used_b = set(), set()
    while True:
        mask  = np.ones_like(D, dtype=bool)
        for i in used_a: mask[i, :] = False
        for j in used_b: mask[:, j] = False
        if not mask.any():
            break
        d_min = D[mask].min()
        if d_min >= fusion_thr:
            break
        coords = np.argwhere((D == d_min) & mask)
        i, j   = int(coords[0, 0]), int(coords[0, 1])
        used_a.add(i); used_b.add(j)
        da, db = front_a[i], front_b[j]
        px = (da['pos_m'][0] + db['pos_m'][0]) / 2.0
        py = (da['pos_m'][1] + db['pos_m'][1]) / 2.0
        merged = dict(da)
        merged['pos_m'] = (px, py)
        merged['cam']   = 'AB'
        if da['cls'] != db['cls']:
            merged['cls'] = da['cls'] if da['conf'] >= db['conf'] else db['cls']
        merged['conf'] = max(da['conf'], db['conf'])
        finales.append(merged)

    for i, d in enumerate(front_a):
        if i not in used_a: finales.append(d)
    for j, d in enumerate(front_b):
        if j not in used_b: finales.append(d)

    return finales


def _full_field_dedup(dets_a: list, dets_b: list,
                      fusion_thr: float = SEAM_FUSION_M) -> list:
    """
    Deduplicación global entre Cam A y Cam B sin límite X.

    Sustituye a _unify_seam para cámaras con overlap grande (p.ej. VEO gran
    angular donde el overlap puede ser >60 m).  En lugar de cortar el campo
    en dos zonas, busca parejas (det_A, det_B) a menos de fusion_thr metros
    en todo el campo y las fusiona; las dets sin pareja se mantienen.

    La diferencia clave con _unify_seam: ninguna detección se descarta por
    estar fuera de un margen — todas participan en la deduplicación.
    """
    from scipy.spatial.distance import cdist as _cdist

    valid_a = [d for d in dets_a if d.get('pos_m') and None not in d['pos_m']]
    valid_b = [d for d in dets_b if d.get('pos_m') and None not in d['pos_m']]
    no_pos  = [d for d in dets_a + dets_b
               if not (d.get('pos_m') and None not in d['pos_m'])]

    if not valid_a or not valid_b:
        return valid_a + valid_b + no_pos

    pts_a = np.array([d['pos_m'] for d in valid_a], dtype=np.float32)
    pts_b = np.array([d['pos_m'] for d in valid_b], dtype=np.float32)
    D     = _cdist(pts_a, pts_b)

    used_a, used_b = set(), set()
    merged = []

    while True:
        mask = np.ones_like(D, dtype=bool)
        for i in used_a: mask[i, :] = False
        for j in used_b: mask[:, j] = False
        if not mask.any():
            break
        d_min = float(D[mask].min())
        if d_min >= fusion_thr:
            break
        coords = np.argwhere((D == d_min) & mask)
        i, j   = int(coords[0, 0]), int(coords[0, 1])
        used_a.add(i); used_b.add(j)

        da, db  = valid_a[i], valid_b[j]
        px = (da['pos_m'][0] + db['pos_m'][0]) / 2.0
        py = (da['pos_m'][1] + db['pos_m'][1]) / 2.0
        best    = da if da['conf'] >= db['conf'] else db
        fused   = dict(best)
        fused['pos_m'] = (px, py)
        fused['cam']   = 'AB'
        if da['cls'] != db['cls']:
            fused['cls'] = da['cls'] if da['conf'] >= db['conf'] else db['cls']
        merged.append(fused)

    for i, d in enumerate(valid_a):
        if i not in used_a: merged.append(d)
    for j, d in enumerate(valid_b):
        if j not in used_b: merged.append(d)

    return merged + no_pos


def _global_nms(dets: list, lm: float = L_M, am: float = A_M,
                min_dist: float = GLOBAL_NMS_M,
                max_dets: int = MAX_DETS_FRAME) -> list:
    valid = [d for d in dets
             if d['pos_m'] is not None
             and 0.0 <= d['pos_m'][0] <= lm
             and 0.0 <= d['pos_m'][1] <= am]
    valid = sorted(valid, key=lambda d: -d['conf'])
    kept, kept_pos = [], []
    for d in valid:
        p = np.array(d['pos_m'], dtype=np.float32)
        if not any(np.linalg.norm(p - q) < min_dist for q in kept_pos):
            kept.append(d)
            kept_pos.append(p)
        if len(kept) >= max_dets:
            break
    return kept


# ═══════════════════════════════════════════════════════════
#  11. HEATMAPS Y ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════

def compute_heatmaps_smooth(heatmaps: dict, sigma_px: int = 8) -> dict:
    return {cls: gaussian_filter(hm.astype(np.float32), sigma=sigma_px)
            for cls, hm in heatmaps.items()}


def compute_stats(frame_records: list, ball_records: list, fps: float) -> pd.DataFrame:
    if not frame_records:
        return pd.DataFrame()
    df = pd.DataFrame(frame_records)

    def _total_distance(grp):
        grp = grp.sort_values('frame')
        dx  = grp['x_m'].diff().fillna(0)
        dy  = grp['y_m'].diff().fillna(0)
        dist = np.sqrt(dx**2 + dy**2)
        dist[dist > TRACK_MAX_DIST_M * 2] = 0
        return dist.sum()

    dist_df = (df.groupby(['gid', 'cls'])
                 .apply(_total_distance)
                 .reset_index(name='dist_m'))
    return dist_df.sort_values('dist_m', ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
#  11b. ESTADÍSTICAS AVANZADAS
# ═══════════════════════════════════════════════════════════

SPRINT_MS = 5.5    # m/s ≈ 20 km/h
HSR_MS    = 4.17   # m/s ≈ 15 km/h  (high-speed running)
_TELEPORT = 8.0    # m  — salto imposible entre frames, descartado

_HOME_CLS = {'player_home', 'gk_home'}
_AWAY_CLS = {'player_away', 'gk_away'}


def _home_attacks_high_x(frame_idx: int,
                          home_attacks_left: bool,
                          halftime_frame) -> bool:
    """
    Devuelve True si el equipo local ataca hacia x alto (x=field_length)
    en el frame dado.

    Convención en la app:
      home_attacks_left=True  →  local DEFIENDE portería izquierda (x=0) en 1T
                              →  local ATACA hacia x alto  en 1T
      En el 2T los equipos cambian de lado.
    """
    is_first_half = (halftime_frame is None or frame_idx < halftime_frame)
    return home_attacks_left if is_first_half else not home_attacks_left


def _x_norm(x_m: float, is_home: bool,
             frame_idx: int,
             home_attacks_left: bool,
             halftime_frame,
             field_length: float) -> float:
    """
    Convierte x absoluta → x relativa al equipo:
      0 = portería propia, field_length = portería contraria.
    """
    home_high = _home_attacks_high_x(frame_idx, home_attacks_left, halftime_frame)
    if is_home:
        # local ataca hacia high-x → su portería propia está en x=0 → x_norm = x
        return x_m if home_high else (field_length - x_m)
    else:
        # visitante ataca en dirección contraria
        return (field_length - x_m) if home_high else x_m


def _x_norm_series(x_series: 'pd.Series', frame_series: 'pd.Series',
                   is_home: bool,
                   home_attacks_left: bool,
                   halftime_frame,
                   field_length: float) -> np.ndarray:
    """Vectorizado sobre un grupo de filas."""
    home_high = np.array([
        _home_attacks_high_x(int(f), home_attacks_left, halftime_frame)
        for f in frame_series
    ])
    x = x_series.values
    if is_home:
        return np.where(home_high, x, field_length - x)
    else:
        return np.where(home_high, field_length - x, x)


def compute_physical_stats(frame_records: list, fps: float,
                           field_length: float = 105.0,
                           cfg: dict = None) -> pd.DataFrame:
    """
    Por jugador: distancia total, por tercio (relativo al equipo),
    sprints, HSR, velocidad.

    cfg puede incluir home_attacks_left y halftime_frame para calcular
    correctamente los tercios defensivo/medio/atacante por equipo y mitad.
    """
    if not frame_records:
        return pd.DataFrame()

    cfg = cfg or {}
    home_attacks_left = cfg.get('home_attacks_left', True)
    halftime_frame    = cfg.get('halftime_frame', None)

    df    = pd.DataFrame(frame_records).sort_values(['gid', 'frame'])
    third = field_length / 3.0
    results = []

    for gid, grp in df.groupby('gid'):
        grp    = grp.sort_values('frame').reset_index(drop=True)
        _mode  = grp['cls'].mode()
        cls    = _mode[0] if len(_mode) > 0 else 'unknown'
        is_home = cls in _HOME_CLS

        dx = grp['x_m'].diff().fillna(0.0).values
        dy = grp['y_m'].diff().fillna(0.0).values
        seg = np.sqrt(dx**2 + dy**2)

        frame_diff = grp['frame'].diff().fillna(1.0).values
        frame_dt   = np.maximum(frame_diff / fps, 1.0 / fps)

        # Paso esperado = mediana del frame_diff (= skip_frames del análisis)
        # Solo pasos de exactamente 1 skip se usan para velocidad (+ margen de 1 frame)
        _diffs = frame_diff[frame_diff > 0]
        expected_step = float(np.median(_diffs)) if len(_diffs) > 0 else 1.0
        gap_mask = frame_diff > expected_step + 1   # permite skip o skip+1, nada más
        seg[gap_mask] = 0.0

        # Filtro físico estricto: máximo realista en fútbol profesional ~34 km/h
        _MAX_SPEED_MS = 9.5
        seg[seg > _MAX_SPEED_MS * frame_dt] = 0.0

        # x relativa al equipo (0=portería propia, FL=portería rival)
        xn = _x_norm_series(grp['x_m'], grp['frame'],
                            is_home, home_attacks_left, halftime_frame, field_length)

        d        = seg.copy()
        dist_def = d[xn < third].sum()
        dist_mid = d[(xn >= third) & (xn < 2 * third)].sum()
        dist_att = d[xn >= 2 * third].sum()

        spd_raw = seg / frame_dt

        # Suavizar con media móvil centrada (ventana 5 frames ≈ 0.5s a skip=3/30fps)
        # Reduce el ruido de posición que hace oscilar la vel. instantánea ±3 m/s
        w = 5
        spd = pd.Series(spd_raw).rolling(w, center=True, min_periods=1).mean().values

        # Histéresis: sprint entra a SPRINT_MS, sale solo si baja de SPRINT_OFF_MS
        SPRINT_OFF_MS = SPRINT_MS - 1.0   # 4.5 m/s
        raw_above = np.zeros(len(spd), dtype=bool)
        _in_spr = False
        for _i, _s in enumerate(spd):
            if not _in_spr:
                if _s >= SPRINT_MS:
                    _in_spr = True
                    raw_above[_i] = True
            else:
                if _s < SPRINT_OFF_MS:
                    _in_spr = False
                else:
                    raw_above[_i] = True

        # Duración mínima: 3 frames consecutivos suavizados (~0.3s) para contar sprint
        is_sprint = np.zeros(len(spd), dtype=bool)
        _in_spr2, _streak = False, 0
        for _i, _v in enumerate(raw_above):
            if _v:
                _streak += 1
                if _streak >= 3:
                    is_sprint[_i] = True
                    if not _in_spr2 and _streak == 3:
                        is_sprint[_i - 2] = True
                        is_sprint[_i - 1] = True
                    _in_spr2 = True
            else:
                _streak = 0
                _in_spr2 = False

        is_hsr = (spd >= HSR_MS) & ~is_sprint

        sprint_bouts = int(np.diff(is_sprint.astype(int), prepend=0).clip(0).sum())

        peak_idx = int(np.argmax(spd))
        # Velocidad máxima: percentil 95 de momentos activos (descarta picos de ruido)
        spd_active = spd[spd > 1.0]
        max_spd = float(np.percentile(spd_active, 95)) if len(spd_active) >= 5 else float(spd.max())
        results.append({
            'gid':                gid,
            'cls':                cls,
            'dist_m':             round(float(seg.sum()), 1),
            'dist_def_m':         round(float(dist_def), 1),
            'dist_mid_m':         round(float(dist_mid), 1),
            'dist_att_m':         round(float(dist_att), 1),
            'sprint_count':       sprint_bouts,
            'sprint_dist_m':      round(float(d[is_sprint].sum()), 1),
            'sprint_time_s':      round(float(is_sprint.sum() / fps), 1),
            'hsr_dist_m':         round(float(d[is_hsr].sum()), 1),
            'max_speed_ms':       round(max_spd, 2),
            'avg_speed_ms':       round(float(spd[spd > 0.1].mean()) if (spd > 0.1).any() else 0.0, 2),
            'time_s':             round(float(len(grp) / fps), 1),
            'sprint_peak_frame':  int(grp['frame'].iloc[peak_idx]),
            'sprint_peak_x':      round(float(grp['x_m'].iloc[peak_idx]), 1),
            'sprint_peak_y':      round(float(grp['y_m'].iloc[peak_idx]), 1),
            'sprint_peak_cam':    grp['cam'].iloc[peak_idx] if 'cam' in grp.columns else 'A',
        })

    return (pd.DataFrame(results)
            .sort_values('dist_m', ascending=False)
            .reset_index(drop=True))


def compute_tactical_stats(frame_records: list, ball_records: list,
                           fps: float,
                           field_length: float = 105.0,
                           field_width: float = 68.0,
                           cfg: dict = None) -> dict:
    """
    Métricas tácticas por frame y agregadas.
    Usa cfg (home_attacks_left, halftime_frame) para calcular correctamente:
      - línea defensiva (siempre el 4.º jugador más cercano a su propia portería)
      - territorio (nº jugadores en campo rival según dirección real de ataque)
      - centroide normalizado (0=portería propia, FL=rival)
    """
    if not frame_records:
        return {}

    cfg               = cfg or {}
    home_attacks_left = cfg.get('home_attacks_left', True)
    halftime_frame    = cfg.get('halftime_frame', None)

    df     = pd.DataFrame(frame_records)
    frames = sorted(df['frame'].unique())

    rows = []
    for fr in frames:
        sub  = df[df['frame'] == fr]
        home = sub[sub['cls'].isin(_HOME_CLS)]
        away = sub[sub['cls'].isin(_AWAY_CLS)]
        row  = {'frame': fr}

        home_high = _home_attacks_high_x(int(fr), home_attacks_left, halftime_frame)

        for prefix, team, is_home in [('home', home, True), ('away', away, False)]:
            n = len(team)
            if n >= 2:
                row[f'{prefix}_cx'] = float(team['x_m'].mean())
                row[f'{prefix}_cy'] = float(team['y_m'].mean())
                row[f'{prefix}_compact'] = float(
                    np.sqrt(team['x_m'].std(ddof=0)**2 + team['y_m'].std(ddof=0)**2))

                # x relativa al equipo → línea defensiva = 4.º jugador con menor x_norm
                xn = np.array([
                    _x_norm(x, is_home, int(fr), home_attacks_left, halftime_frame, field_length)
                    for x in team['x_m']
                ])
                xn_sorted = np.sort(xn)
                row[f'{prefix}_def_line'] = float(xn_sorted[min(3, n - 1)])
                row[f'{prefix}_width']    = float(team['y_m'].max() - team['y_m'].min())

                # centroide normalizado para gráficas coherentes
                row[f'{prefix}_cx_norm'] = float(xn.mean())
            else:
                for k in ['cx', 'cy', 'compact', 'def_line', 'width', 'cx_norm']:
                    row[f'{prefix}_{k}'] = np.nan

        # Territorio: jugadores en campo rival según dirección real
        if home_high:
            # local ataca hacia x alto → su campo rival es x > FL/2
            row['home_in_att_half'] = int((home['x_m'] > field_length / 2).sum())
            row['away_in_att_half'] = int((away['x_m'] < field_length / 2).sum())
        else:
            row['home_in_att_half'] = int((home['x_m'] < field_length / 2).sum())
            row['away_in_att_half'] = int((away['x_m'] > field_length / 2).sum())

        rows.append(row)

    pf = pd.DataFrame(rows)

    # Pressing distance (avg de los 3 más cercanos al balón)
    if ball_records:
        ball_df  = pd.DataFrame(ball_records)
        ball_idx = ball_df.set_index('frame')[['x_m', 'y_m']].to_dict('index')
        press_rows = []
        for fr in frames:
            if fr not in ball_idx:
                continue
            bx, by = ball_idx[fr]['x_m'], ball_idx[fr]['y_m']
            sub = df[df['frame'] == fr]
            if len(sub):
                dists = np.sqrt((sub['x_m'] - bx)**2 + (sub['y_m'] - by)**2)
                press_rows.append({'frame': fr,
                                   'pressing_dist': float(dists.nsmallest(3).mean())})
        if press_rows:
            pf = pf.merge(pd.DataFrame(press_rows), on='frame', how='left')

    # Transición: tiempo medio entre cruces del centroide normalizado por la mitad (FL/2)
    def _transition_s(cx_norm_series):
        cx = cx_norm_series.dropna().values
        if len(cx) < 2:
            return None
        crossings = np.where(np.diff((cx > field_length / 2).astype(int)) != 0)[0]
        if len(crossings) < 2:
            return None
        return round(float(np.mean(np.diff(crossings)) / fps), 1)

    home_trans_s = _transition_s(pf.get('home_cx_norm', pd.Series(dtype=float)))
    away_trans_s = _transition_s(pf.get('away_cx_norm', pd.Series(dtype=float)))

    territory_home_pct = float(
        (pf['home_in_att_half'] > pf['away_in_att_half']).mean() * 100
    ) if 'home_in_att_half' in pf.columns else 50.0

    # Asimetría lateral: avg_y relativo al centro del campo (>0 = lado derecho)
    _yc = field_width / 2
    home_sub = df[df['cls'].isin(_HOME_CLS)]
    away_sub = df[df['cls'].isin(_AWAY_CLS)]
    home_avg_y_bias = round(float(home_sub['y_m'].mean() - _yc), 1) if not home_sub.empty else 0.0
    away_avg_y_bias = round(float(away_sub['y_m'].mean() - _yc), 1) if not away_sub.empty else 0.0

    # High press: % of frames where def_line > 50% of field (team pushed up past halfway)
    hp_high_press = float((pf['home_def_line'].dropna() > field_length * 0.5).mean() * 100) \
        if 'home_def_line' in pf.columns else 0.0
    ap_high_press = float((pf['away_def_line'].dropna() > field_length * 0.5).mean() * 100) \
        if 'away_def_line' in pf.columns else 0.0

    # Box occupancy: avg players in attacking final 16.5m zone — vectorized per frame
    box_depth = 16.5
    _home_high_vec = pd.Series(
        [_home_attacks_high_x(int(f), home_attacks_left, halftime_frame) for f in frames],
        index=frames, dtype=bool,
    )
    # Map each row in df to whether home attacks high that frame
    df_home_high = df['frame'].map(_home_high_vec)
    home_all   = df[df['cls'].isin(_HOME_CLS)].copy()
    away_all   = df[df['cls'].isin(_AWAY_CLS)].copy()
    home_hh    = home_all['frame'].map(_home_high_vec)
    away_hh    = away_all['frame'].map(_home_high_vec)
    home_in_box = ((home_hh & (home_all['x_m'] > field_length - box_depth)) |
                   (~home_hh & (home_all['x_m'] < box_depth)))
    away_in_box = ((away_hh & (away_all['x_m'] < box_depth)) |
                   (~away_hh & (away_all['x_m'] > field_length - box_depth)))
    home_box_per_frame = home_all[home_in_box].groupby('frame').size().reindex(frames, fill_value=0)
    away_box_per_frame = away_all[away_in_box].groupby('frame').size().reindex(frames, fill_value=0)

    # Defensive line stability: low std = compact block, high std = disorganized
    home_def_stability = round(float(pf['home_def_line'].dropna().std()), 1) \
        if 'home_def_line' in pf.columns else 0.0
    away_def_stability = round(float(pf['away_def_line'].dropna().std()), 1) \
        if 'away_def_line' in pf.columns else 0.0

    summary = {
        'per_frame':            pf,
        'home_avg_compact':     round(float(pf['home_compact'].dropna().mean()), 1),
        'away_avg_compact':     round(float(pf['away_compact'].dropna().mean()), 1),
        'home_avg_def_line':    round(float(pf['home_def_line'].dropna().mean()), 1),
        'away_avg_def_line':    round(float(pf['away_def_line'].dropna().mean()), 1),
        'home_avg_width':       round(float(pf['home_width'].dropna().mean()), 1),
        'away_avg_width':       round(float(pf['away_width'].dropna().mean()), 1),
        'territory_home_pct':   round(territory_home_pct, 1),
        'territory_away_pct':   round(100 - territory_home_pct, 1),
        'home_avg_pressure':    round(float(pf['home_in_att_half'].mean()), 1),
        'away_avg_pressure':    round(float(pf['away_in_att_half'].mean()), 1),
        'home_transition_s':    home_trans_s,
        'away_transition_s':    away_trans_s,
        'home_avg_y_bias':      home_avg_y_bias,
        'away_avg_y_bias':      away_avg_y_bias,
        'home_high_press_pct':  round(hp_high_press, 1),
        'away_high_press_pct':  round(ap_high_press, 1),
        'home_box_occ':         round(float(home_box_per_frame.mean()), 2),
        'away_box_occ':         round(float(away_box_per_frame.mean()), 2),
        'home_def_stability':   home_def_stability,
        'away_def_stability':   away_def_stability,
    }
    if 'pressing_dist' in pf.columns:
        summary['avg_pressing_dist'] = round(float(pf['pressing_dist'].dropna().mean()), 1)

    # Posesión: frame con jugador más cercano al balón → equipo con posesión
    if ball_records:
        ball_df  = pd.DataFrame(ball_records)
        ball_idx = ball_df.set_index('frame')[['x_m', 'y_m']].to_dict('index')
        poss_home = poss_away = 0
        for fr in frames:
            if fr not in ball_idx:
                continue
            bx, by = ball_idx[fr]['x_m'], ball_idx[fr]['y_m']
            sub = df[df['frame'] == fr]
            if sub.empty:
                continue
            dists = np.sqrt((sub['x_m'] - bx)**2 + (sub['y_m'] - by)**2)
            closest_cls = sub.loc[dists.idxmin(), 'cls']
            if closest_cls in _HOME_CLS:
                poss_home += 1
            elif closest_cls in _AWAY_CLS:
                poss_away += 1
        total_poss = poss_home + poss_away
        if total_poss > 0:
            summary['possession_home_pct'] = round(100 * poss_home / total_poss, 1)
            summary['possession_away_pct'] = round(100 * poss_away / total_poss, 1)

    return summary


def compute_zone_stats(frame_records: list,
                       field_length: float = 105.0,
                       field_width: float = 68.0,
                       cfg: dict = None) -> dict:
    """
    Divide el campo en 15 zonas (3 franjas × 5 columnas) en coordenadas reales
    del campo (x_m, y_m). No normaliza por perspectiva de equipo.

    Para cada zona devuelve:
      home / away        → conteos absolutos de detecciones
      home_density       → % de todas las detecciones HOME que caen aquí
      away_density       → % de todas las detecciones AWAY que caen aquí
      home_pct           → ratio relativo home/(home+away) (retrocompatibilidad)

    La orientación (qué lado es ataque/defensa) la comunica la UI con flechas,
    usando home_attacks_left del config.
    """
    if not frame_records:
        return {}

    df = pd.DataFrame(frame_records)

    # Cuadrícula 5 columnas × 3 franjas en coordenadas reales
    n_cols, n_rows = 5, 3
    col_w = field_length / n_cols
    row_h = field_width  / n_rows

    zone_defs = {}
    for r in range(n_rows):
        for c in range(n_cols):
            zone_defs[f'r{r}c{c}'] = (
                (c * col_w, (c + 1) * col_w),
                (r * row_h, (r + 1) * row_h),
            )

    total_home = int(df[df['cls'].isin(_HOME_CLS)].shape[0]) or 1
    total_away = int(df[df['cls'].isin(_AWAY_CLS)].shape[0]) or 1

    results = {}
    last_col = n_cols - 1
    for zname, ((x0, x1), (y0, y1)) in zone_defs.items():
        col_idx = int(zname[zname.index('c') + 1:])
        x_mask  = ((df['x_m'] >= x0) & (df['x_m'] <= x1)
                   if col_idx == last_col
                   else (df['x_m'] >= x0) & (df['x_m'] < x1))
        mask    = x_mask & (df['y_m'] >= y0) & (df['y_m'] < y1)
        zone_df = df[mask]
        h = int(zone_df[zone_df['cls'].isin(_HOME_CLS)].shape[0])
        a = int(zone_df[zone_df['cls'].isin(_AWAY_CLS)].shape[0])
        t = h + a
        results[zname] = {
            'home':         h,
            'away':         a,
            'home_density': round(100 * h / total_home, 2),
            'away_density': round(100 * a / total_away, 2),
            'home_pct':     round(100 * h / t, 1) if t else 50.0,
            'away_pct':     round(100 * a / t, 1) if t else 50.0,
        }
    return results


_KNOWN_FORMATIONS = [
    (4,4,2),(4,3,3),(4,2,3,1),(4,5,1),(4,1,4,1),(4,4,1,1),(4,3,2,1),
    (3,5,2),(3,4,3),(3,4,2,1),(3,5,1,1),(3,3,4),
    (5,3,2),(5,4,1),(5,2,3),(5,3,1,1),
]

def _snap_formation(lines_out: list) -> str:
    """Ajusta la formación calculada a la más cercana del catálogo de formaciones conocidas."""
    if not lines_out:
        return '?'
    counts = tuple(len(l) for l in lines_out)
    total  = sum(counts)
    if total < 6:
        return '?'

    # Si solo 2 líneas → intentar redistribuir la más grande en 2 mitades
    if len(counts) == 2:
        big, small = (counts[0], counts[1]) if counts[0] >= counts[1] else (counts[1], counts[0])
        h1, h2 = big // 2, big - big // 2
        # Probar ambas disposiciones (pequeña primero o al final)
        cand1 = (small, h1, h2)
        cand2 = (h1, h2, small)
        # Elegir la que más se parezca a formaciones conocidas de 3 líneas
        def _dist3(c):
            three = [f for f in _KNOWN_FORMATIONS if len(f) == 3]
            if not three: return 999
            return min(sum(abs(a-b) for a,b in zip(f,c)) for f in three)
        counts = cand1 if _dist3(cand1) <= _dist3(cand2) else cand2

    # Coincidencia exacta
    if counts in _KNOWN_FORMATIONS:
        return '-'.join(str(c) for c in counts)

    # Buscar formación conocida con mismo número de líneas y mínima diferencia
    same_lines = [f for f in _KNOWN_FORMATIONS if len(f) == len(counts)]
    if same_lines:
        best = min(same_lines, key=lambda f: sum(abs(a-b) for a,b in zip(f, counts)))
        return '-'.join(str(c) for c in best)

    # Fallback: mejor formación de 3 líneas del catálogo
    three = [f for f in _KNOWN_FORMATIONS if len(f) == 3]
    if three:
        # Redistribuir counts a 3 grupos más balanceados
        s = sorted(counts, reverse=True)
        c3 = (s[0], sum(s[1:]) // 2, (sum(s[1:]) + 1) // 2) if len(s) >= 2 else counts
        best = min(three, key=lambda f: sum(abs(a-b) for a,b in zip(f, c3)))
        return '-'.join(str(c) for c in best)

    return '?'


def compute_lineup_and_formation(frame_records: list, fps: float,
                                  field_length: float = 98.0,
                                  cfg: dict = None,
                                  min_time_s: float = 60.0) -> dict:
    """
    Identifica el 11 más representativo de cada equipo y estima su formación.

    Solo considera jugadores con >= min_time_s de aparición (elimina tracks
    fragmentados). Devuelve:
      {
        'home': {'players': [...], 'formation': '4-3-3', 'lines': [[...], ...]},
        'away': {...}
      }
    Cada jugador: {'gid', 'cls', 'avg_x_norm', 'avg_y', 'time_s', 'role'}
    """
    if not frame_records:
        return {}

    cfg               = cfg or {}
    home_attacks_left = cfg.get('home_attacks_left', True)
    halftime_frame    = cfg.get('halftime_frame', None)

    df = pd.DataFrame(frame_records)
    result = {}

    for team, cls_set, is_home in [
        ('home', _HOME_CLS, True),
        ('away', _AWAY_CLS, False),
    ]:
        sub = df[df['cls'].isin(cls_set)]
        if sub.empty:
            result[team] = {'players': [], 'formation': '?', 'lines': []}
            continue

        all_candidates = []
        for gid, grp in sub.groupby('gid'):
            grp = grp.sort_values('frame').reset_index(drop=True)
            _mode = grp['cls'].mode()
            cls = _mode[0] if len(_mode) > 0 else 'unknown'
            time_s = (grp['frame'].iloc[-1] - grp['frame'].iloc[0] + 1) / fps
            xn = _x_norm_series(grp['x_m'], grp['frame'],
                                 is_home, home_attacks_left, halftime_frame, field_length)
            all_candidates.append({
                'gid':        gid,
                'cls':        cls,
                'avg_x_norm': float(np.mean(xn)),
                'avg_y':      float(grp['y_m'].mean()),
                'time_s':     round(time_s, 1),
            })

        # Umbral adaptativo: reducir min_time_s hasta tener al menos 7 outfield
        gk_cls = 'gk_home' if is_home else 'gk_away'
        per_gid = []
        for _min_t in [min_time_s, min_time_s * 0.5, min_time_s * 0.25, 10.0]:
            per_gid = [p for p in all_candidates if p['time_s'] >= _min_t]
            per_gid.sort(key=lambda p: p['time_s'], reverse=True)
            _outfield_try = [p for p in per_gid if p['cls'] != gk_cls]
            if len(_outfield_try) >= 7:
                break

        # Separar GK y jugadores de campo, limitar a 1 GK + 10 outfield
        gks      = [p for p in per_gid if p['cls'] == gk_cls][:1]
        outfield = [p for p in per_gid if p['cls'] != gk_cls][:10]

        # Asignar rol 'gk'
        for p in gks:
            p['role'] = 'GK'

        # Inferir formación por análisis de huecos en x_norm
        formation = '?'
        lines_out = []
        if len(outfield) >= 6:
            of_sorted = sorted(outfield, key=lambda p: p['avg_x_norm'])
            xvals = [p['avg_x_norm'] for p in of_sorted]
            n = len(xvals)
            gaps = sorted(
                [(xvals[i+1] - xvals[i], i) for i in range(n - 1)],
                reverse=True
            )
            # Máximo 2 separadores (3 líneas): no permitir 4+ líneas
            n_sep = 2
            sep_indices = sorted(g[1] for g in gaps[:n_sep])

            prev = 0
            for sep in sep_indices:
                line = of_sorted[prev:sep + 1]
                if line:
                    lines_out.append(line)
                prev = sep + 1
            tail = of_sorted[prev:]
            if tail:
                lines_out.append(tail)

            # Solo eliminar líneas vacías; no fusionar las de 1 jugador
            # (4,5,1) es una formación válida en el catálogo — no destruirla
            lines_out = [l for l in lines_out if l]
            formation = _snap_formation(lines_out)

        # Asignar roles por línea
        role_names = {1: 'DEF', 2: 'MED', 3: 'DEL', 4: 'DEL'}
        for li, line in enumerate(lines_out):
            role = role_names.get(li + 1, 'MED')
            for p in line:
                p['role'] = role

        all_players = gks + outfield
        result[team] = {
            'players':   all_players,
            'formation': formation,
            'lines':     lines_out,
        }

    return result


def extract_formation_from_csv(df: 'pd.DataFrame',
                                team: str = 'home',
                                field_length: float = 98.0,
                                home_attacks_left: bool = True,
                                halftime_frame=None,
                                n_outfield: int = 10,
                                min_frames: int = 200) -> tuple:
    """
    Extrae la formación de un equipo usando KMeans 2D sobre posiciones medianas
    por GID. Robusto ante fragmentación de IDs: no selecciona los top-N GIDs
    sino que agrupa todos los fragmentos persistentes en n_outfield clusters
    posicionales, uno por slot de jugador de campo.

    Args:
        df            : DataFrame con columnas [frame, gid, cls, x_m, y_m].
        team          : 'home' o 'away'.
        field_length  : longitud del campo en metros.
        home_attacks_left : True si el local ataca hacia x bajas en la 1ª parte.
        halftime_frame: fotograma de inicio de la 2ª parte (None = ignorar).
        n_outfield    : slots de campo a encontrar (10 para fútbol 11).
        min_frames    : mínimo de apariciones para incluir un GID.

    Returns:
        (formation, centers_df) donde:
          formation  : str tipo '4-3-3'
          centers_df : DataFrame [x_norm, y_m, line, role] por slot
    """
    from sklearn.cluster import KMeans

    if team == 'home':
        player_cls, gk_cls, is_home = ['player_home', 'gk_home'], 'gk_home', True
    else:
        player_cls, gk_cls, is_home = ['player_away', 'gk_away'], 'gk_away', False

    outfield = df[df['cls'].isin(player_cls) & (df['cls'] != gk_cls)].copy()
    if outfield.empty:
        return '?', pd.DataFrame()

    # Filtrar GIDs demasiado cortos — descender umbral si hay pocos válidos
    gid_counts = outfield.groupby('gid')['frame'].nunique()
    for thresh in [min_frames, min_frames // 2, 50, 20]:
        valid_gids = gid_counts[gid_counts >= thresh].index
        if len(valid_gids) >= n_outfield:
            break
    outfield = outfield[outfield['gid'].isin(valid_gids)]
    if len(valid_gids) < n_outfield:
        return '?', pd.DataFrame()

    # Normalizar x al sistema de referencia del equipo (en metros, 0=portería propia)
    xn = _x_norm_series(outfield['x_m'], outfield['frame'],
                        is_home, home_attacks_left, halftime_frame, field_length)
    outfield = outfield.copy()
    outfield['x_norm_m'] = xn   # metros (0–field_length), misma escala que y_m

    # Posición mediana por GID → un punto por fragmento de trayectoria
    medians = (outfield.groupby('gid')
                       .agg(x_norm_m=('x_norm_m', 'median'),
                            y_m     =('y_m',       'median'))
                       .reset_index())

    # KMeans 2D en metros (x e y misma escala → clustering correcto)
    X  = medians[['x_norm_m', 'y_m']].values
    km = KMeans(n_clusters=n_outfield, n_init=20, random_state=42, max_iter=500)
    km.fit(X)
    centers = km.cluster_centers_   # (n_outfield, 2) → [x_norm_m, y_m]

    # Detectar líneas por los 2 mayores huecos en x (metros)
    x_sorted = np.sort(centers[:, 0])
    gaps     = x_sorted[1:] - x_sorted[:-1]
    sep_pos  = np.sort(np.argsort(gaps)[-2:])

    lines, prev = [], 0
    for s in sep_pos:
        lines.append(int(s - prev + 1))
        prev = s + 1
    lines.append(int(n_outfield - prev))

    formation = _snap_formation_from_tuple(tuple(lines))

    # Asignar etiqueta de línea a cada centro
    x_order     = np.argsort(centers[:, 0])
    line_labels = np.empty(n_outfield, dtype=int)
    li, cnt = 0, 0
    for rank in range(n_outfield):
        slot = x_order[rank]
        line_labels[slot] = li
        cnt += 1
        if li < len(lines) - 1 and cnt >= lines[li]:
            li += 1; cnt = 0

    role_map   = {0: 'DEF', 1: 'MED', 2: 'DEL', 3: 'DEL'}
    centers_df = pd.DataFrame({
        'x_norm': centers[:, 0] / field_length,  # 0–1 para visualización
        'y_m':    centers[:, 1],
        'line':   line_labels,
        'role':   [role_map.get(int(l), 'MED') for l in line_labels],
    }).sort_values('x_norm').reset_index(drop=True)

    return formation, centers_df


def extract_formation_density(df: 'pd.DataFrame',
                               team: str = 'home',
                               field_length: float = 98.0,
                               field_width: float = 61.0,
                               home_attacks_left: bool = True,
                               halftime_frame=None,
                               n_outfield: int = 10,
                               speed_max: float = 4.0,
                               bins: tuple = (60, 37),
                               sigma: float = 2.5,
                               suppress_radius_m: float = 8.0) -> tuple:
    """
    Extrae la formación mediante mapa de densidad 2D (heatmap peaks).

    En lugar de trabajar por GID, usa TODAS las posiciones de jugadores del
    equipo para construir un histograma 2D, lo suaviza con un filtro gaussiano
    y encuentra los n_outfield picos de densidad mediante supresión de máximos
    iterativa. Completamente robusto ante fragmentación de IDs.

    Args:
        speed_max       : filtrar frames con velocidad > speed_max (m/s),
                          elimina transiciones rápidas que distorsionan el mapa.
        bins            : resolución del histograma (x_bins, y_bins).
        sigma           : radio de suavizado gaussiano (en bins).
        suppress_radius_m: radio de supresión alrededor de cada pico encontrado.
    """
    if team == 'home':
        player_cls, is_home = ['player_home'], True
    else:
        player_cls, is_home = ['player_away'], False

    sub = df[df['cls'].isin(player_cls)].copy()
    if sub.empty:
        return '?', pd.DataFrame()

    # Filtrar por velocidad para quedarse con momentos posicionales
    if 'speed_ms' in sub.columns and speed_max > 0:
        sub = sub[sub['speed_ms'] <= speed_max]

    if len(sub) < 500:
        return '?', pd.DataFrame()

    # Normalizar x al sistema de referencia del equipo (metros, 0=portería propia)
    xn = _x_norm_series(sub['x_m'], sub['frame'],
                        is_home, home_attacks_left, halftime_frame, field_length)
    sub = sub.copy()
    sub['xn'] = xn

    # Histograma 2D (x_norm_m vs y_m)
    H, xedges, yedges = np.histogram2d(
        sub['xn'].clip(0, field_length),
        sub['y_m'].clip(0, field_width),
        bins=bins,
        range=[[0, field_length], [0, field_width]],
    )

    H_smooth = gaussian_filter(H.astype(float), sigma=sigma)

    # Centros de bins
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2

    # Supresión iterativa de máximos
    r_x = max(1, int(suppress_radius_m / (field_length / bins[0])))
    r_y = max(1, int(suppress_radius_m / (field_width  / bins[1])))

    H_work  = H_smooth.copy()
    centers = []
    for _ in range(n_outfield):
        idx = np.unravel_index(np.argmax(H_work), H_work.shape)
        centers.append([xc[idx[0]], yc[idx[1]]])
        x0 = max(0, idx[0] - r_x);  x1 = min(bins[0], idx[0] + r_x + 1)
        y0 = max(0, idx[1] - r_y);  y1 = min(bins[1], idx[1] + r_y + 1)
        H_work[x0:x1, y0:y1] = 0.0

    centers = np.array(centers)   # (n_outfield, 2) en metros

    # Detectar líneas por los 2 mayores huecos en x (metros)
    x_sorted = np.sort(centers[:, 0])
    gaps     = x_sorted[1:] - x_sorted[:-1]
    sep_pos  = np.sort(np.argsort(gaps)[-2:])

    lines, prev = [], 0
    for s in sep_pos:
        lines.append(int(s - prev + 1))
        prev = s + 1
    lines.append(int(n_outfield - prev))

    formation = _snap_formation_from_tuple(tuple(lines))

    # Asignar etiqueta de línea y rol a cada centro
    x_order     = np.argsort(centers[:, 0])
    line_labels = np.empty(n_outfield, dtype=int)
    li, cnt = 0, 0
    for rank in range(n_outfield):
        slot = x_order[rank]
        line_labels[slot] = li
        cnt += 1
        if li < len(lines) - 1 and cnt >= lines[li]:
            li += 1; cnt = 0

    role_map   = {0: 'DEF', 1: 'MED', 2: 'DEL', 3: 'DEL'}
    centers_df = pd.DataFrame({
        'x_norm': centers[:, 0] / field_length,   # 0–1 para visualización
        'y_m':    centers[:, 1],
        'line':   line_labels,
        'role':   [role_map.get(int(l), 'MED') for l in line_labels],
    }).sort_values('x_norm').reset_index(drop=True)

    return formation, centers_df


def _snap_formation_from_tuple(counts: tuple) -> str:
    """Como _snap_formation pero recibe tupla de enteros directamente."""
    total = sum(counts)
    if total < 6:
        return '?'
    if counts in _KNOWN_FORMATIONS:
        return '-'.join(str(c) for c in counts)
    same = [f for f in _KNOWN_FORMATIONS if len(f) == len(counts)]
    if same:
        best = min(same, key=lambda f: sum(abs(a - b) for a, b in zip(f, counts)))
        return '-'.join(str(c) for c in best)
    three = [f for f in _KNOWN_FORMATIONS if len(f) == 3]
    if three:
        s  = sorted(counts, reverse=True)
        c3 = (s[0], sum(s[1:]) // 2, (sum(s[1:]) + 1) // 2)
        best = min(three, key=lambda f: sum(abs(a - b) for a, b in zip(f, c3)))
        return '-'.join(str(c) for c in best)
    return '?'


def build_ai_prompt(physical_df: pd.DataFrame,
                    tactical: dict,
                    zone_stats: dict,
                    cfg: dict,
                    fps: float,
                    lineup_data: dict = None) -> str:
    """Construye el prompt estructurado para el LLM."""
    home_team = cfg.get('home_team', 'Equipo Local')
    away_team = cfg.get('away_team', 'Equipo Visitante')
    fl        = cfg.get('field_length', 98.0)
    fw        = cfg.get('field_width',  61.0)

    start_f   = cfg.get('start_frame', 0)
    n_f       = cfg.get('n_frames', 0)
    end_f     = start_f + n_f
    hal_frame = cfg.get('halftime_frame')

    # Calcular qué tiempos están cubiertos por el segmento analizado
    if hal_frame and start_f <= hal_frame <= end_f:
        t1_min = (hal_frame - start_f) / fps / 60
        t2_min = (end_f - hal_frame)   / fps / 60
        if t1_min > 2 and t2_min > 2:
            time_str   = f"AMBOS TIEMPOS — 1T: {t1_min:.0f} min + 2T: {t2_min:.0f} min (total: {t1_min+t2_min:.0f} min)"
            halves_tag = "ambos tiempos"
        elif t1_min >= t2_min:
            time_str   = f"PRINCIPALMENTE 1er TIEMPO — {t1_min:.0f} min analizados del 1T, {t2_min:.0f} min del 2T"
            halves_tag = "principalmente 1T"
        else:
            time_str   = f"PRINCIPALMENTE 2º TIEMPO — {t1_min:.0f} min del 1T, {t2_min:.0f} min analizados del 2T"
            halves_tag = "principalmente 2T"
    elif hal_frame and end_f <= hal_frame:
        t1_min   = n_f / fps / 60
        time_str = f"SOLO 1er TIEMPO — {t1_min:.0f} min analizados (el descanso aún no ocurre en este segmento)"
        halves_tag = "solo 1T"
    elif hal_frame and start_f >= hal_frame:
        t2_min   = n_f / fps / 60
        time_str = f"SOLO 2º TIEMPO — {t2_min:.0f} min analizados (el segmento empieza tras el descanso)"
        halves_tag = "solo 2T"
    else:
        dur_min  = n_f / fps / 60
        time_str = f"{dur_min:.0f} min analizados (sin detección de descanso — tiempo indeterminado)"
        halves_tag = "tiempo indeterminado"

    # Solo jugadores de campo (sin GK) para las métricas físicas por jugador (/10)
    hp_df = physical_df[physical_df['cls'] == 'player_home'] if len(physical_df) else pd.DataFrame()
    ap_df = physical_df[physical_df['cls'] == 'player_away'] if len(physical_df) else pd.DataFrame()

    h_form = lineup_data.get('home', {}).get('formation', '?') if lineup_data else '?'
    a_form = lineup_data.get('away', {}).get('formation', '?') if lineup_data else '?'

    lines = []

    # ── 1. Contexto ──────────────────────────────────────────
    lines += [
        "═══════════════════════════════════════════════════════",
        f"PARTIDO  : {home_team} (local)  vs  {away_team} (visitante)",
        f"Campo    : {fl:.0f}×{fw:.0f}m",
        f"Análisis : {time_str}",
        f"Sistema  : seguimiento automático Veo (cámara panorámica, fútbol amateur/semiprofesional)",
        "═══════════════════════════════════════════════════════",
        "",
    ]

    # ── 2. Formación estimada ─────────────────────────────────
    lines += [
        "── FORMACIÓN ESTIMADA (mapa de densidad posicional, precisión ~70%) ──",
        f"  {home_team}: {h_form}",
        f"  {away_team}: {a_form}",
        "  IMPORTANTE: la forma global del sistema es fiable; los roles individuales NO lo son",
        "  (el tracking fragmentado impide asignar jugadores concretos a posiciones).",
        "",
    ]

    # ── 3. Métricas posicionales (ALTA FIABILIDAD) ───────────
    if tactical:
        def _v(key):
            v = tactical.get(key, '?')
            return f"{v}" if v != '?' else '?'

        h_def   = tactical.get('home_avg_def_line', None)
        a_def   = tactical.get('away_avg_def_line', None)
        h_comp  = tactical.get('home_avg_compact', '?')
        a_comp  = tactical.get('away_avg_compact', '?')
        h_wid   = tactical.get('home_avg_width', '?')
        a_wid   = tactical.get('away_avg_width', '?')
        h_hp    = tactical.get('home_high_press_pct', '?')
        a_hp    = tactical.get('away_high_press_pct', '?')
        h_box   = tactical.get('home_box_occ', '?')
        a_box   = tactical.get('away_box_occ', '?')
        h_stab  = tactical.get('home_def_stability', '?')
        a_stab  = tactical.get('away_def_stability', '?')
        h_pres  = tactical.get('home_avg_pressure', '?')
        a_pres  = tactical.get('away_avg_pressure', '?')
        h_terr  = tactical.get('territory_home_pct', '?')
        a_terr  = tactical.get('territory_away_pct', '?')
        h_poss  = tactical.get('possession_home_pct', '?')
        a_poss  = tactical.get('possession_away_pct', '?')
        h_bias  = tactical.get('home_avg_y_bias', 0)
        a_bias  = tactical.get('away_avg_y_bias', 0)
        h_trans = tactical.get('home_transition_s', '?')
        a_trans = tactical.get('away_transition_s', '?')
        press_d = tactical.get('avg_pressing_dist', '?')

        def _bias_str(b):
            try:
                b = float(b)
                if b > 1.0:   return f"+{b:.1f}m (sesgo derecha)"
                if b < -1.0:  return f"{b:.1f}m (sesgo izquierda)"
                return "centrado"
            except: return '?'

        def_diff_note = ""
        try:
            diff = float(h_def) - float(a_def)
            if diff > 3:
                def_diff_note = f"  → {home_team} defiende {diff:.1f}m MÁS ALTO (más audaz)"
            elif diff < -3:
                def_diff_note = f"  → {away_team} defiende {abs(diff):.1f}m MÁS ALTO (más audaz)"
            else:
                def_diff_note = f"  → líneas defensivas similares (Δ={abs(diff):.1f}m)"
        except Exception:
            pass

        def _box_str(v):
            try:
                f_ = float(v)
                if f_ > 0.8:  return f"{v} jug/frame  ★ amenaza constante"
                if f_ > 0.4:  return f"{v} jug/frame  presencia moderada"
                return f"{v} jug/frame  poca presencia en área"
            except: return str(v)

        def _stab_str(v):
            try:
                f_ = float(v)
                if f_ < 3:  return f"{v}m  ★ bloque muy organizado"
                if f_ < 6:  return f"{v}m  organización aceptable"
                return f"{v}m  línea inestable / desorganizada"
            except: return str(v)

        def _press_str(v):
            try:
                f_ = float(v)
                if f_ < 10:  return f"{v}m  PRESSING ALTO"
                if f_ < 18:  return f"{v}m  pressing medio"
                return f"{v}m  bloque bajo / poca presión"
            except: return str(v)

        def _trans_str(v):
            try:
                f_ = float(v)
                if f_ < 8:   return f"{v}s  ★ transición rápida"
                if f_ < 15:  return f"{v}s  transición moderada"
                return f"{v}s  transición lenta / reactiva"
            except: return str(v)

        col_w = 26
        lines += [
            "── MÉTRICAS POSICIONALES — alta fiabilidad (calculadas frame a frame) ──",
            f"  {'Métrica':<28}  {home_team:<{col_w}}  {away_team}",
            f"  {'─'*28}  {'─'*col_w}  {'─'*col_w}",
            f"  {'Posesión estimada':<28}  {str(h_poss)+'%':<{col_w}}  {a_poss}%",
            f"  {'Control territorial':<28}  {str(h_terr)+'%':<{col_w}}  {a_terr}%",
            f"  {'Línea defensiva':<28}  {str(h_def)+'m desde portería':<{col_w}}  {a_def}m desde portería",
            f"  {'':28}  {def_diff_note}",
            f"  {'Compacidad (long.)':<28}  {str(h_comp)+'m':<{col_w}}  {a_comp}m",
            f"  {'Anchura lateral':<28}  {str(h_wid)+'m':<{col_w}}  {a_wid}m",
            f"  {'Press alto (% frames)':<28}  {str(h_hp)+'%':<{col_w}}  {a_hp}%",
            f"  {'Jugadores campo rival':<28}  {str(h_pres)+'/frame':<{col_w}}  {a_pres}/frame",
            f"  {'Ocupación área rival':<28}  {_box_str(h_box):<{col_w}}  {_box_str(a_box)}",
            f"  {'Estabilidad def. (σ)':<28}  {_stab_str(h_stab):<{col_w}}  {_stab_str(a_stab)}",
            f"  {'Asimetría lateral':<28}  {_bias_str(h_bias):<{col_w}}  {_bias_str(a_bias)}",
            f"  {'Transición centroide':<28}  {_trans_str(h_trans):<{col_w}}  {_trans_str(a_trans)}",
        ]
        if press_d != '?':
            lines.append(f"  {'Pressing global (3 más próximos)':<28}  {_press_str(press_d)}")
        lines.append("")

    # ── 4. Mapa zonal (3 franjas × 3 tercios) ────────────────
    if zone_stats:
        lines.append("── CONTROL TERRITORIAL POR ZONA (densidad = % tiempo con jugadores en zona) ──")
        lines.append(f"  {'':14}  {'TERCIO DEF (propio)':22}  {'TERCIO MEDIO':22}  {'TERCIO ATK (rival)':22}")
        flank_labels = ['Flanco der. ', 'Centro     ', 'Flanco izq. ']
        tercio_cols  = [([0, 1], 'DEF'), ([2], 'MED'), ([3, 4], 'ATK')]
        h_totals, a_totals = [0.0]*3, [0.0]*3
        for r in range(3):
            parts = []
            for ti, (cols, _) in enumerate(tercio_cols):
                h = sum(zone_stats.get(f'r{r}c{c}', {}).get('home_density', 0.0) for c in cols)
                a = sum(zone_stats.get(f'r{r}c{c}', {}).get('away_density', 0.0) for c in cols)
                h_totals[ti] += h
                a_totals[ti] += a
                dom = 'H' if h >= a else 'V'
                parts.append(f"H={h:.1f}%/V={a:.1f}% [{dom}]")
            lines.append(f"  {flank_labels[r]}  {'  |  '.join(f'{p:<22}' for p in parts)}")
        lines.append("")
        lines.append("  Totales por tercio:")
        for ti, (_, label) in enumerate(tercio_cols):
            dom = home_team if h_totals[ti] >= a_totals[ti] else away_team
            lines.append(f"    {label}: {home_team}={h_totals[ti]:.1f}%  {away_team}={a_totals[ti]:.1f}%  → domina {dom}")
        lines.append("")

    # ── 5. Físico (ORIENTATIVO — solo ratios) ────────────────
    if len(hp_df) or len(ap_df):
        n_h = len(hp_df)
        n_a = len(ap_df)
        frag_h = "⚠ FRAGMENTADO" if n_h > 25 else "OK"
        frag_a = "⚠ FRAGMENTADO" if n_a > 25 else "OK"
        lines += [
            "── FÍSICO — solo orientativo (promedios por TRACK, NO por jugador completo) ──",
            f"  Tracks detectados: {home_team}={n_h} ({frag_h})  |  {away_team}={n_a} ({frag_a})",
            "  Con tracking fragmentado un jugador puede tener 4-6 tracks con contadores separados.",
            "  USA estos valores SOLO para comparar ratios entre equipos, NUNCA como absolutos.",
            "",
        ]
        for team, td in [(home_team, hp_df), (away_team, ap_df)]:
            if len(td):
                _n10   = 10  # jugadores de campo por equipo (sin GK)
                vmax   = td['max_speed_ms'].max()
                hsr    = td['hsr_dist_m'].sum()   / _n10
                sprnt  = td['sprint_count'].sum()  / _n10
                d_def  = td['dist_def_m'].sum()   / _n10
                d_att  = td['dist_att_m'].sum()   / _n10
                ratio  = (d_att / d_def) if d_def > 0 else 0
                lines.append(
                    f"  {team}: vel_max={vmax:.1f}m/s  "
                    f"HSR/jug={hsr:.0f}m  sprints/jug={sprnt:.1f}  "
                    f"dist_def/jug={d_def:.0f}m  dist_ata/jug={d_att:.0f}m  "
                    f"ratio_ata/def={ratio:.2f}"
                )
        lines.append("")

    stats_block = "\n".join(lines)

    halves_note = (
        f"NOTA TEMPORAL: los datos cubren {halves_tag}. "
        + ("Las métricas son representativas del partido completo." if "ambos" in halves_tag
           else "Ten en cuenta que el análisis puede no reflejar el partido completo.")
    )

    return f"""Eres un analista táctico de fútbol con experiencia en fútbol amateur y semiprofesional. \
Los datos provienen de un sistema de seguimiento automático por visión artificial (cámara VEO panorámica). \
Tu análisis va directamente al entrenador — debe ser concreto, técnico y accionable.

{stats_block}

════════════════════════════════════════════════════════════════
GUÍA DE FIABILIDAD — LEE ANTES DE ANALIZAR
════════════════════════════════════════════════════════════════
{halves_note}

✅ ALTA FIABILIDAD — úsalos con confianza total:
   • Línea defensiva, compacidad, anchura, asimetría lateral
   • Press alto %, estabilidad defensiva (σ), ocupación área rival
   • Control territorial, posesión estimada
   • Mapa zonal 3×3 — es el dato posicional más rico del sistema
   • Transición del centroide, pressing global (distancia al balón)
   • Formación global (4-3-3, 4-4-2… fiable; roles individuales NO)

⚠️ ORIENTATIVO — úsalos solo para comparar RATIOS entre equipos:
   • Velocidad máxima: 100% fiable (pico global real)
   • HSR/jug, sprints/jug: estimación dividiendo totales entre 11 jugadores. \
Útil para comparar actividad entre equipos, NO para afirmar distancias absolutas.
   • ratio_ata/def >1 = sesgo ofensivo | <0.8 = equipo muy defensivo

❌ NO DISPONIBLE:
   • Estadísticas individuales (sin identificación de dorsales)
   • Posesión absoluta exacta (tracking del balón incompleto)

════════════════════════════════════════════════════════════════
BENCHMARKS — fútbol amateur/semipro
════════════════════════════════════════════════════════════════
Línea defensiva (m desde portería propia):
  <25m replegado | 25-38m bloque medio-bajo | 38-50m medio-alto | >50m muy adelantado
Compacidad longitudinal:
  <18m muy compacto | 18-28m normal | >28m líneas separadas / disperso
Press alto %:
  <15% bloque bajo | 15-30% pressing selectivo | >30% pressing alto sostenido
Estabilidad def. σ:
  <3m muy organizado | 3-6m aceptable | >6m línea sin referencia / inestable
Ocupación área rival (jug/frame):
  <0.3 sin amenaza | 0.3-0.7 moderada | >0.7 amenaza constante
Transición centroide:
  <7s explosiva | 7-14s normal | >14s reactiva / lenta
Pressing global (3 más próximos):
  <10m agresivo | 10-18m medio | >18m bloque pasivo
HSR/jugador estimado:
  <200m actividad baja | 200-400m normal | >400m equipo muy intenso

════════════════════════════════════════════════════════════════
INSTRUCCIONES DE ANÁLISIS
════════════════════════════════════════════════════════════════
1. Cita SIEMPRE el valor numérico exacto + interpretación táctica. Sin números no es análisis.
2. COMPARA los dos equipos en cada punto — el valor relativo es más importante que el absoluto.
3. El mapa zonal 3×3 muestra qué pasillos (flanco izq/centro/flanco der) × tercios (def/med/atk) \
   controla cada equipo. Es la base para prescribir cómo atacar y cómo presionar al rival.
4. CRUZA métricas: línea alta + σ alto = espacio a la espalda. Press alto % + poco box_occ = \
   presiona sin profundidad. Sesgo lateral + zona flanco dominada = sobrecargan ese pasillo.
5. PRESSING AL RIVAL: prescribe triggers concretos — cuándo presionar (en qué zona del campo \
   pierde el rival el balón según el mapa zonal), en qué momento del partido, con qué sistema \
   (bloque alto, medio, bajo, pressing en salida de balón, pressing tras pérdida).
6. ENTRADA AL PARTIDO: indica cómo atacar al rival desde el inicio según sus debilidades \
   (si defiende alto → atacar por la espalda; si σ alto → atacar rápido en transición; \
   si sesgo lateral claro → sobrecargar el flanco libre; si compacto pero estrecho → por fuera).
7. VULNERABILIDAD: combinación de 2-3 métricas, no una sola.
8. RECOMENDACIONES: específicas y medibles. Ejemplo válido: "Replegar la línea a ~35m cuando \
   se pierde el balón en campo propio para cerrar el espacio que el rival explota en transición \
   rápida (transición centroide {away_team}: X s)". Inválido: "mejorar la defensa".
9. Si un dato es '?' dilo explícitamente. No inventes nombres de jugadores.
10. Tono: analista UEFA hablando a un entrenador de fútbol base. Directo, técnico, 5 minutos de lectura.

Responde en español con EXACTAMENTE este formato (no añadas ni elimines secciones):

## Resumen ejecutivo
(3-4 frases: quién dominó territorialmente y en qué zonas, el patrón táctico diferencial, \
y si hay diferencias significativas entre tiempos si se analizaron ambos)

## {home_team} — {h_form}
**Control del partido:** (posesión + territorio + qué tercios y pasillos dominaron vs cedieron)
**Bloque defensivo:** (línea def + compacidad + estabilidad — qué sistema defensivo revela)
**Pressing y transiciones:** (intensidad pressing + % press alto + velocidad de transición)
**Capacidad ofensiva:** (ocupación área + tercio ataque + sesgo lateral + peligrosidad real)
**Vulnerabilidad principal:** (2-3 métricas combinadas que definen el punto más explotable)

## {away_team} — {a_form}
**Control del partido:** ...
**Bloque defensivo:** ...
**Pressing y transiciones:** ...
**Capacidad ofensiva:** ...
**Vulnerabilidad principal:** ...

## Comparativa directa
- **Línea defensiva:** (quién defiende más alto, diferencia exacta en metros, espacio generado)
- **Pressing:** (quién presiona más intensamente, en qué zona del campo, con qué distancia al balón)
- **Bloque:** (compacidad y anchura comparada — ¿quién gestiona mejor el espacio entre líneas?)
- **Zona clave del mapa:** (la celda 3×3 con mayor desequilibrio — qué implica tácticamente)
- **Intensidad física:** (comparativa de ratios HSR y sprints si los datos lo permiten)

## Plan de partido — cómo atacar al rival
**Si juegas contra {away_team}:**
- **Pressing:** cuándo y dónde presionar (trigger concreto basado en sus zonas de pérdida y transición)
- **Cómo entrar al partido:** qué sistema/estructura usar los primeros minutos según sus debilidades
- **Zonas a explotar:** qué pasillo atacar preferentemente y por qué (datos del mapa zonal)
- **En transición:** cómo aprovechar su velocidad de cambio de bloque

**Si juegas contra {home_team}:**
- **Pressing:** cuándo y dónde presionar
- **Cómo entrar al partido:** estructura inicial recomendada
- **Zonas a explotar:** pasillo prioritario según mapa zonal
- **En transición:** cómo aprovechar sus tiempos de reorganización

## Recomendaciones para el entrenamiento
- **{home_team}:** (2 ejercicios o situaciones tácticas concretas a trabajar esta semana, \
basadas en las métricas — cita el valor que motiva cada ejercicio)
- **{away_team}:** (ídem)

## Conclusión
(2 frases: el factor diferencial del partido y la tendencia más importante para preparar el próximo)"""


def generate_ai_analysis(physical_df: pd.DataFrame,
                         tactical: dict,
                         zone_stats: dict,
                         cfg: dict,
                         fps: float,
                         api_key: str,
                         lineup_data: dict = None) -> str:
    """Llama a la API de Claude y devuelve el análisis táctico en Markdown."""
    try:
        import anthropic
    except ImportError:
        return "**Error:** instala el paquete `anthropic` (`pip install anthropic`)"

    prompt = build_ai_prompt(physical_df, tactical, zone_stats, cfg, fps, lineup_data)
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ═══════════════════════════════════════════════════════════
#  12. PROCESADO DE SEGMENTO (process_segment)
# ═══════════════════════════════════════════════════════════

def process_segment(vid_pan: str, vid_std: str,
                    H_a, H_b,
                    protos_day: dict, protos_night: dict,
                    config: dict,
                    model_A, model_B, model_ball,
                    progress_dict: dict,
                    model_sam=None):
    """
    Procesa un segmento de vídeo en un hilo secundario.
    Actualiza progress_dict con: pct, frame_idx, tracked, ball,
    frame_records, ball_records, heatmaps, heatmap_ball, done, error.
    """
    try:
        _process_segment_inner(
            vid_pan, vid_std, H_a, H_b,
            protos_day, protos_night, config,
            model_A, model_B, model_ball, progress_dict, model_sam
        )
    except Exception as e:
        import traceback
        progress_dict['error'] = traceback.format_exc()
        progress_dict['done']  = True


def _process_segment_inner(vid_pan, vid_std, H_a, H_b,
                            protos_day, protos_night, config,
                            model_A, model_B, model_ball,
                            progress_dict, model_sam):
    start_f         = config.get('start_frame', 0)
    n_frames        = config.get('n_frames',    900)
    skip_frames     = max(1, int(config.get('skip_frames', 1)))
    match_intervals = config.get('match_intervals', None)  # [(f1s,f1e),(f2s,f2e)] o None

    def _in_play(fidx):
        """True si el frame pertenece a un intervalo de juego real (excluye descanso)."""
        if match_intervals is None:
            return True
        return any(s <= fidx <= e for s, e in match_intervals)
    conf_thresh   = config.get('conf_thresh',   CONF_THRESH)
    conf_ball     = config.get('conf_ball',     CONF_BALL)
    bright_thresh = config.get('bright_thresh', BRIGHT_THRESH)
    field_length  = config.get('field_length',  L_M)
    field_width   = config.get('field_width',   A_M)
    _default_dev  = 'cuda' if torch.cuda.is_available() else 'cpu'
    device        = config.get('device', _default_dev)
    half_prec     = config.get('half_precision', device == 'cuda')
    imgsz         = config.get('imgsz', 640)

    info   = get_video_info(vid_pan)
    fps    = info['fps']
    vid_w  = info['width']
    half_h = info['height'] // 2
    progress_dict['fps'] = fps

    init_undistort_maps(half_h, vid_w)

    limite_x = config.get('limite_x', compute_limite_x(H_a, H_b, vid_w, half_h))

    tracker_type = config.get('tracker_type', 'bytetrack')
    if tracker_type == 'bytetrack':
        tracker = ByteFieldTracker(
            fps=fps,
            high_thresh=config.get('bt_high_thresh', BT_HIGH_THRESH),
            match_thresh_m=config.get('bt_match_thresh_m', BT_MATCH_THRESH_M),
            max_lost_s=config.get('bt_max_lost_s', BT_MAX_LOST_S),
            gallery_ttl_s=config.get('bt_gallery_ttl_s', BT_GALLERY_TTL_S),
            reid_hue_thresh=config.get('bt_reid_hue_thresh', BT_REID_HUE_THRESH),
        )
    elif tracker_type == 'strongkalman':
        tracker = StrongKalmanTrackerV3(
            fps=fps,
            max_dist_m=config.get('track_max_dist_m',       TRACK_MAX_DIST_M),
            window_s=config.get('track_window_s',           TRACK_WINDOW_S),
            class_weight=config.get('track_class_w',        TRACK_CLASS_W),
            lost_s=config.get('v3_lost_s',                  V3_LOST_S),
            gallery_ttl_s=config.get('track_gallery_ttl_s', V3_GALLERY_TTL_S),
            reid_hue_thresh=config.get('track_reid_hue_thresh', V3_REID_HUE_THRESH),
        )
    else:
        tracker = KalmanFieldTracker(fps=fps)
    HM_W        = int(field_length * ESCALA)
    HM_H        = int(field_width  * ESCALA)
    heatmaps    = {cls: np.zeros((HM_H, HM_W), dtype=np.float32)
                   for cls in CLASSES_ORDER}
    heatmap_ball = np.zeros((HM_H, HM_W), dtype=np.float32)
    frame_records = []
    ball_records  = []

    # Kalman para el balón
    ball_kf = cv2.KalmanFilter(4, 2)
    ball_kf.transitionMatrix     = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
    ball_kf.measurementMatrix    = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
    ball_kf.processNoiseCov      = np.diag([0.1, 0.1, 1.0, 1.0]).astype(np.float32)
    ball_kf.measurementNoiseCov  = np.eye(2, dtype=np.float32) * 0.5
    ball_kf.errorCovPost         = np.eye(4, dtype=np.float32)
    ball_initialized = False

    def _detect_ball_local(frame, _half_h):
        nonlocal ball_initialized
        if model_ball is None:
            return None
        best_det = None
        for half, H, cam_side in [
            (preprocess_half(frame[:_half_h, :], 'left'),  H_a, 'left'),
            (preprocess_half(frame[_half_h:,  :], 'right'), H_b, 'right'),
        ]:
            try:
                res = model_ball.predict(half, conf=conf_ball, verbose=False, device=device, half=half_prec, imgsz=imgsz)
            except Exception:
                continue
            if not res or res[0].boxes is None:
                continue
            for box in res[0].boxes:
                bc = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                try:
                    xm, ym = pixel_to_field(cx, cy, H)
                    if not (-5 <= xm <= field_length + 5 and -5 <= ym <= field_width + 5):
                        continue
                except Exception:
                    continue
                if best_det is None or bc > best_det[2]:
                    best_det = (xm, ym, bc)

        if best_det is None:
            return None
        if not ball_initialized:
            ball_kf.statePost = np.array(
                [[best_det[0]], [best_det[1]], [0.0], [0.0]], dtype=np.float32)
            ball_initialized = True
        else:
            ball_kf.predict()
        meas = np.array([[best_det[0]], [best_det[1]]], dtype=np.float32)
        ball_kf.correct(meas)
        s = ball_kf.statePost
        return (float(s[0, 0]), float(s[1, 0]), best_det[2])

    # Config de partido
    home_attacks_left = config.get('home_attacks_left', True)
    halftime_frame    = config.get('halftime_frame')
    gk_home_color_lab = config.get('gk_home_color_lab')
    gk_away_color_lab = config.get('gk_away_color_lab')

    def _top3_labs_from_proto(proto: dict | None) -> list | None:
        """Extrae lista de colores LAB individuales del top3 de un prototipo PKL."""
        if proto is None:
            return None
        top3 = proto.get('top3') or proto.get('bgr_top3') or []
        if not top3:
            return None
        labs = []
        for c in top3:
            bgr_img = np.asarray(c, dtype=np.uint8).reshape(1, 1, 3)
            labs.append(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32))
        return labs

    def _resolve_gk_color(color_lab, protos_src: dict | None, key: str):
        """Usa color manual si existe (envuelto en lista); si no, deriva del PKL top3."""
        if color_lab is not None:
            return [np.asarray(color_lab, dtype=np.float32)]
        if protos_src is None:
            return None
        return _top3_labs_from_proto(protos_src.get(key))

    # Detectar si algún PKL es v3 (tiene hue_sig)
    def _proto_is_v3(proto):
        return proto is not None and any('hue_sig' in p for p in proto.values())
    _use_v3 = _proto_is_v3(protos_day) or _proto_is_v3(protos_night)

    def _track_half_local(model, half_img, H, protos, _conf):
        results = model.predict(half_img, conf=_conf, verbose=False, device=device, half=half_prec, imgsz=imgsz)
        dets = []
        if results and results[0].boxes is not None:
            boxes_all = results[0].boxes
            boxes_np  = boxes_all.xyxy.cpu().numpy()
            confs_np  = boxes_all.conf.cpu().numpy()

            bboxes_list = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes_np]
            sam_masks   = extract_sam_masks_batch(half_img, bboxes_list, model_sam, device)
            gh          = get_grass_hue(half_img) or 55.0

            for idx_b, (box, conf_val) in enumerate(zip(boxes_np, confs_np)):
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                crop = half_img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                if crop.size == 0:
                    continue
                mask  = sam_masks[idx_b] if idx_b < len(sam_masks) else None

                # SAM mask aplicado al crop para BGR y hue_sig — fondo negro
                # filtrado por mask_black en get_player_color (S<5 & V<20)
                if mask is not None:
                    msk_r = cv2.resize(mask.astype(np.uint8),
                                       (crop.shape[1], crop.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                    crop_m = crop.copy(); crop_m[msk_r == 0] = 0
                else:
                    crop_m = crop
                torso = get_torso_crop(crop_m)
                if torso.size == 0:
                    torso = get_torso_crop(crop)

                if _use_v3 and any('hue_sig' in p for p in protos.values()):
                    cls_name, _, _, _ = classify_player_v3(crop, protos,
                                                            grass_hue=gh, mask=mask)
                    hue_sig = get_hue_signature_masked(crop, mask, gh).tolist()
                else:
                    cls_name, _ = classify_player(torso, protos, gh)
                    hue_sig = None

                _, color_mean, _ = get_player_color(torso, gh)
                cx_px = (x1 + x2) / 2.0
                cy_px = float(y2)
                try:
                    xm, ym = pixel_to_field(cx_px, cy_px, H)
                except Exception:
                    xm, ym = None, None
                dets.append({
                    'bbox_px':    (int(x1), int(y1), int(x2), int(y2)),
                    'conf':       float(conf_val),
                    'cls':        cls_name,
                    'pos_m':      (xm, ym) if xm is not None else None,
                    'color_desc': color_mean.tolist(),
                    'hue_sig':    hue_sig,
                })
        return dets

    cap = cv2.VideoCapture(vid_pan)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    progress_dict.update({
        'pct': 0, 'frame_idx': start_f, 'tracked': [],
        'ball': None, 'frame_records': frame_records,
        'ball_records': ball_records, 'heatmaps': heatmaps,
        'heatmap_ball': heatmap_ball, 'done': False, 'error': None,
    })

    last_unified    = []
    # _gk_slot: UN ÚNICO candidato por slot ('gk_home'/'gk_away')
    # {'gk_home': {'gid': X, 'count': N}, 'gk_away': {...}}
    # Evita la oscilación cuando dos candidatos se alternan: solo el candidato
    # del slot puede seguir acumulando racha; cualquier otro es demoteado.
    _gk_slot        = {}
    _gk_confirmed   = {}  # gid -> clase GK confirmada ('gk_home' | 'gk_away')
    _gk_gone_frames = {}
    _gk_far_frames  = {}
    GK_CONFIRM_F    = 5   # frames consecutivos requeridos para confirmar GK (evita confirmaciones espurias)
    GK_EXPIRE_F     = 270   # >= V3_LOST_S*FPS (8s×30=240) + margen
    GK_FAR_EXPIRE_F = 30    # frames fuera del área penal antes de desconfirmar
    for n_proc in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = start_f + n_proc

        # Descanso: avanzar el tracker con detecciones vacías para expirar tracks del
        # tiempo anterior, pero no grabar nada en frame_records ni procesar detecciones.
        if not _in_play(frame_idx):
            last_unified = []
            tracker.update(frame_idx, [])
            # Actualizar progreso visual sin grabar datos
            if n_proc % max(1, n_frames // 200) == 0:
                pct = int(n_proc / n_frames * 100)
                progress_dict.update({'pct': pct, 'frame_idx': frame_idx})
            continue

        if n_proc % skip_frames == 0:
            cam_a = preprocess_half(frame[:half_h, :], 'left')
            cam_b = preprocess_half(frame[half_h:,  :], 'right')
            protos = select_protos(frame, protos_day, protos_night, bright_thresh)
            dets_a = _track_half_local(model_A, cam_a, H_a, protos, conf_thresh)
            dets_b = _track_half_local(model_B, cam_b, H_b, protos, conf_thresh)
            last_unified = _global_nms(
                _full_field_dedup(
                    [{**d, 'cam': 'A'} for d in dets_a],
                    [{**d, 'cam': 'B'} for d in dets_b],
                ), field_length, field_width)

        tracked = tracker.update(frame_idx, last_unified)

        try:
            hal = (not home_attacks_left
                   if halftime_frame is not None and frame_idx >= halftime_frame
                   else home_attacks_left)

            def _gk_goal_x(cls):
                if cls == 'gk_home':
                    return field_length if hal else 0.0
                else:
                    return 0.0 if hal else field_length

            def _near_goal(d, cls):
                pos = d.get('pos_m')
                if pos is None:
                    return False  # sin posición: no restaurar como GK
                gx = _gk_goal_x(cls)
                # Hard: el portero no puede estar en el campo rival
                mid = field_length / 2.0
                if (gx > mid and pos[0] < mid) or (gx < mid and pos[0] > mid):
                    return False
                return abs(pos[0] - gx) <= 16.5  # área penal real (era 20m)

            if _use_v3:
                # PRE-PASS: restaurar sticky label ANTES de apply_gk_by_position
                # Solo restaura 1 gid por slot (el más reciente en _gk_confirmed).
                _pre_restored_home = False
                _pre_restored_away = False

                # 1) Restaurar jugadores YA CONFIRMADOS
                for d in tracked:
                    gid = d.get('gid')
                    if gid is None or gid not in _gk_confirmed:
                        continue
                    label = _gk_confirmed[gid]
                    if label == 'gk_home':
                        if _near_goal(d, label) and not _pre_restored_home:
                            if d.get('cls') not in ('gk_home', 'gk_away'):
                                d['cls'] = label
                            _pre_restored_home = True
                        elif _pre_restored_home:
                            _gk_confirmed.pop(gid, None)
                            for _s in list(_gk_slot):
                                if _gk_slot[_s]['gid'] == gid:
                                    _gk_slot.pop(_s, None)
                    elif label == 'gk_away':
                        if _near_goal(d, label) and not _pre_restored_away:
                            if d.get('cls') not in ('gk_home', 'gk_away'):
                                d['cls'] = label
                            _pre_restored_away = True
                        elif _pre_restored_away:
                            _gk_confirmed.pop(gid, None)
                            for _s in list(_gk_slot):
                                if _gk_slot[_s]['gid'] == gid:
                                    _gk_slot.pop(_s, None)

                # 2) Restaurar candidatos PROVISIONALES (en racha pero no confirmados).
                # Crítico: sin esto, apply_gk_by_position puede elegir otro candidato
                # cada frame y romper la racha, causando la oscilación.
                for slot, info in list(_gk_slot.items()):
                    if (slot == 'gk_home' and _pre_restored_home) or \
                       (slot == 'gk_away' and _pre_restored_away):
                        continue
                    cand_gid = info['gid']
                    found = False
                    for d in tracked:
                        if d.get('gid') == cand_gid and _near_goal(d, slot):
                            if d.get('cls') not in ('gk_home', 'gk_away'):
                                d['cls'] = slot
                            if slot == 'gk_home':
                                _pre_restored_home = True
                            else:
                                _pre_restored_away = True
                            found = True
                            break
                    if not found:
                        _gk_slot.pop(slot, None)

                tracked = apply_gk_by_position(
                    tracked,
                    home_attacks_left=hal,
                    gk_home_color_lab=_resolve_gk_color(gk_home_color_lab, protos, 'gk_home'),
                    gk_away_color_lab=_resolve_gk_color(gk_away_color_lab, protos, 'gk_away'),
                    field_l=field_length,
                    field_a=field_width,
                )
                # Pass 2: classify_gk_score for unknowns still unresolved
                _gk_h_proto = protos.get('gk_home')
                _gk_a_proto = protos.get('gk_away')
                if (_gk_h_proto is not None or _gk_a_proto is not None) and \
                   (_gk_h_proto is None or 'top3' in _gk_h_proto or 'bgr_top3' in _gk_h_proto or 'hue_sig' in _gk_h_proto) and \
                   (_gk_a_proto is None or 'top3' in _gk_a_proto or 'bgr_top3' in _gk_a_proto or 'hue_sig' in _gk_a_proto):
                    _p2_has_home = any(d.get('cls') == 'gk_home' for d in tracked)
                    _p2_has_away = any(d.get('cls') == 'gk_away' for d in tracked)
                    for d in tracked:
                        if d.get('cls') == 'unknown':
                            gk_cls = classify_gk_score(
                                d, _gk_h_proto, _gk_a_proto,
                                home_attacks_left=hal,
                                field_l=field_length,
                                field_a=field_width,
                                min_color_gate=0.45,
                                min_combined=0.60,
                            )
                            if gk_cls == 'gk_home' and not _p2_has_home:
                                d['cls'] = gk_cls
                                _p2_has_home = True
                            elif gk_cls == 'gk_away' and not _p2_has_away:
                                d['cls'] = gk_cls
                                _p2_has_away = True

            # POST-PASS: filtro temporal GK — un único candidato por slot.
            # Liberar slots provisionales cuyo GID ya no está en este frame,
            # para que no bloqueen un nuevo candidato durante GK_EXPIRE_F frames.
            _current_gids = {d.get('gid') for d in tracked if d.get('gid') is not None}
            for _slot in list(_gk_slot.keys()):
                _si = _gk_slot[_slot]
                if _si['gid'] not in _current_gids and _si['gid'] not in _gk_confirmed:
                    _gk_slot.pop(_slot, None)
            active_gids = set()
            for d in tracked:
                gid = d.get('gid')
                if gid is None:
                    continue
                active_gids.add(gid)
                cls = d.get('cls')
                if cls in ('gk_home', 'gk_away'):
                    slot = cls
                    info = _gk_slot.get(slot)
                    if info is None:
                        # Nuevo candidato — si ya estaba confirmado, arrancar con
                        # count=GK_CONFIRM_F para no demoterlo a unknown
                        initial = GK_CONFIRM_F if gid in _gk_confirmed else 1
                        _gk_slot[slot] = {'gid': gid, 'count': initial}
                        info = _gk_slot[slot]
                    elif info['gid'] == gid:
                        # El mismo candidato avanza su racha
                        info['count'] += 1
                    else:
                        # Otro candidato ya ocupa el slot → demotear al intruso
                        d['cls'] = 'unknown'
                        continue
                    if info['count'] >= GK_CONFIRM_F or gid in _gk_confirmed:
                        # Confirmar: evictar cualquier confirmado anterior del mismo slot
                        for old_g in [g for g, c in list(_gk_confirmed.items())
                                      if c == slot and g != gid]:
                            _gk_confirmed.pop(old_g, None)
                        _gk_confirmed[gid] = slot
                    else:
                        d['cls'] = 'unknown'  # provisional: no mostrar como GK aún
                else:
                    # Jugador no etiquetado como GK en este frame
                    # Si era candidato provisional, limpiar su slot
                    for slot, info in list(_gk_slot.items()):
                        if info['gid'] == gid and gid not in _gk_confirmed:
                            _gk_slot.pop(slot, None)
                            break
                    if gid in _gk_confirmed:
                        confirmed_cls = _gk_confirmed[gid]
                        if _near_goal(d, confirmed_cls):
                            d['cls'] = confirmed_cls
                            _gk_far_frames.pop(gid, None)
                        else:
                            _gk_far_frames[gid] = _gk_far_frames.get(gid, 0) + 1
                            if _gk_far_frames[gid] >= GK_FAR_EXPIRE_F:
                                _gk_confirmed.pop(gid, None)
                                for slot in list(_gk_slot):
                                    if _gk_slot[slot]['gid'] == gid:
                                        _gk_slot.pop(slot, None)
                                _gk_far_frames.pop(gid, None)

            _streaking_gids = {info['gid'] for info in _gk_slot.values()}
            for g in (_streaking_gids | set(_gk_confirmed)) - active_gids:
                _gk_gone_frames[g] = _gk_gone_frames.get(g, 0) + 1
                if _gk_gone_frames[g] >= GK_EXPIRE_F:
                    _gk_confirmed.pop(g, None)
                    for slot in list(_gk_slot):
                        if _gk_slot[slot]['gid'] == g:
                            _gk_slot.pop(slot, None)
                    _gk_gone_frames.pop(g, None)
                    _gk_far_frames.pop(g, None)
            for g in active_gids:
                _gk_gone_frames.pop(g, None)

        except Exception as _gk_exc:
            progress_dict['gk_errors'] = progress_dict.get('gk_errors', 0) + 1
            if progress_dict['gk_errors'] <= 5:
                import traceback as _tb
                progress_dict.setdefault('gk_error_last', _tb.format_exc()[-400:])

        ball_det = _detect_ball_local(frame, half_h)

        for d in tracked:
            if d['pos_m'] is None:
                continue
            xm, ym = d['pos_m']
            if not (0 <= xm <= field_length and 0 <= ym <= field_width):
                continue
            cls = d['cls']
            px  = int(np.clip(xm * ESCALA, 0, HM_W - 1))
            py  = int(np.clip(ym * ESCALA, 0, HM_H - 1))
            if cls in heatmaps:
                heatmaps[cls][py, px] += 1.0
            frame_records.append({
                'frame':    frame_idx,
                'gid':      d['gid'],
                'cls':      cls,
                'x_m':      round(xm, 2),
                'y_m':      round(ym, 2),
                'conf':     round(d['conf'], 3),
                'speed_ms': round(d.get('speed_ms', 0.0), 2),
                'cam':      d.get('cam', '?'),
            })

        if ball_det is not None:
            bxm, bym, bconf = ball_det
            if 0 <= bxm <= field_length and 0 <= bym <= field_width:
                bpx = int(np.clip(bxm * ESCALA, 0, HM_W - 1))
                bpy = int(np.clip(bym * ESCALA, 0, HM_H - 1))
                heatmap_ball[bpy, bpx] += 1.0
                ball_records.append({
                    'frame': frame_idx,
                    'x_m':   round(bxm, 2),
                    'y_m':   round(bym, 2),
                    'conf':  round(bconf, 3),
                })

        pct = (n_proc + 1) / n_frames
        progress_dict.update({
            'pct':       pct,
            'frame_idx': frame_idx,
            'tracked':   tracked,
            'ball':      ball_det,
        })

    cap.release()
    progress_dict['done'] = True
