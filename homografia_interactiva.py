"""
homografia_interactiva.py  v4
==============================
FLUJO:
  0. Pregunta las medidas del campo (o usa las del .veo por defecto).
  1. Muestra diagrama del campo numerado y pregunta qué marcas son visibles.
  2. Abre el frame CILINDRICO SIN corrección (sin remap) con UN PUNTO
     draggable por cada marca visible seleccionada.
  3. El usuario arrastra cada punto hasta la marca real del césped.
  4. El overlay del campo se actualiza en tiempo real con findHomography.
  5. S → guarda H en data/homography/H_veo_<half>.npy

CONTROLES (en la ventana)
  Clic + arrastrar   — mover punto de control
  1…9 / 0 / a-z      — seleccionar punto para nudge
  Flechas            — nudge 1 px   |  Shift+Flechas → 10 px
  S                  — guardar homografía
  R                  — reiniciar puntos a posición inicial
  A / Z              — + / - transparencia overlay
  F                  — toggle overlay campo ON/OFF
  G                  — toggle rejilla de metros ON/OFF
  Q / ESC            — salir

USO
  python homografia_interactiva.py                     # TOP, frame 3000
  python homografia_interactiva.py --half bottom
  python homografia_interactiva.py --frame 5000 --half top
"""

import argparse
import os
import subprocess
import sys
import cv2
import numpy as np

# ─── MEDIDAS DEL CAMPO (.veo) ─────────────────────────────────────────────────
FIELD_LENGTH = 105.0
FIELD_WIDTH  =  70.94

# Líneas FIFA
_AP  = 16.5;   _AW = 40.32   # área grande
_A6  =  5.5;   _BW = 18.32   # área chica
_PEN = 11.0                   # penalti
_RCC =  9.15                  # radio círculo central

VIDEO_PATH = "data/videos/veo_panoramico.mp4"

# Anchura del frame VEO apilado (2 cámaras × 1024 = 2048 de alto)
VEO_STACKED_W = 2048
VEO_HALF_H    = 1024   # altura de cada mitad (imagen cilíndrica cruda)

# Esquinas VEO (coordenadas normalizadas en imagen apilada 2048×2048)
VEO_NORM = {
    "cl": (0.0288, 0.3160),   # top  → campo (0, W)
    "fl": (0.5942, 0.2149),   # top  → campo (0, 0)
    "cr": (0.8121, 0.7539),   # bot  → campo (L, W)
    "fr": (0.3613, 0.7151),   # bot  → campo (L, 0)
}

# ─── CATÁLOGO DE MARCAS ───────────────────────────────────────────────────────

def build_catalog():
    """
    Lista completa de marcas identificables en el campo.
    Cada entrada: (n, etiqueta_corta, etiqueta_larga, (xm, ym))
    """
    L, W = FIELD_LENGTH, FIELD_WIDTH
    CX, CY = L/2, W/2
    AY1, AY2 = (W-_AW)/2, (W-_AW)/2 + _AW
    BY1, BY2 = (W-_BW)/2, (W-_BW)/2 + _BW

    return [
        # n   etiq-corta              etiq-larga                               (xm,   ym)
        ( 1, "ESQ_IZQ_LEJ",  f"Esquina IZQ lejana    (0, 0)",                 (0,       0  )),
        ( 2, "ESQ_IZQ_CER",  f"Esquina IZQ cercana   (0, {W})",               (0,       W  )),
        ( 3, "ESQ_DER_LEJ",  f"Esquina DER lejana    ({L}, 0)",               (L,       0  )),
        ( 4, "ESQ_DER_CER",  f"Esquina DER cercana   ({L}, {W})",             (L,       W  )),
        ( 5, "LC_LEJ",       f"Línea central LEJANA  ({CX:.1f}, 0)",          (CX,      0  )),
        ( 6, "LC_CER",       f"Línea central CERCANA ({CX:.1f}, {W})",        (CX,      W  )),
        ( 7, "CENTRO",       f"Centro del campo      ({CX:.1f}, {CY:.2f})",   (CX,      CY )),
        ( 8, "CC_ARR",       f"Círculo central ARRIBA({CX:.1f},{CY-_RCC:.2f})",(CX,  CY-_RCC)),
        ( 9, "CC_ABA",       f"Círculo central ABAJO ({CX:.1f},{CY+_RCC:.2f})",(CX,  CY+_RCC)),
        (10, "CC_IZQ",       f"Círculo central IZQ   ({CX-_RCC:.2f},{CY:.2f})",(CX-_RCC,CY)),
        (11, "CC_DER",       f"Círculo central DER   ({CX+_RCC:.2f},{CY:.2f})",(CX+_RCC,CY)),
        (12, "ADG_IZQ_ESQ_LEJ", f"Área gde IZQ esq lejana   ({_AP},{AY1:.2f})",  (_AP,    AY1)),
        (13, "ADG_IZQ_ESQ_CER", f"Área gde IZQ esq cercana  ({_AP},{AY2:.2f})",  (_AP,    AY2)),
        (14, "ADG_IZQ_LG_LEJ",  f"Área gde IZQ línea gol lejana  (0,{AY1:.2f})",(0,      AY1)),
        (15, "ADG_IZQ_LG_CER",  f"Área gde IZQ línea gol cercana (0,{AY2:.2f})",(0,      AY2)),
        (16, "ADG_DER_ESQ_LEJ", f"Área gde DER esq lejana   ({L-_AP:.1f},{AY1:.2f})", (L-_AP, AY1)),
        (17, "ADG_DER_ESQ_CER", f"Área gde DER esq cercana  ({L-_AP:.1f},{AY2:.2f})", (L-_AP, AY2)),
        (18, "ADG_DER_LG_LEJ",  f"Área gde DER línea gol lejana  ({L},{AY1:.2f})",    (L,     AY1)),
        (19, "ADG_DER_LG_CER",  f"Área gde DER línea gol cercana ({L},{AY2:.2f})",    (L,     AY2)),
        (20, "APQ_IZQ_ESQ_LEJ", f"Área chica IZQ esq lejana   ({_A6},{BY1:.2f})",     (_A6,   BY1)),
        (21, "APQ_IZQ_ESQ_CER", f"Área chica IZQ esq cercana  ({_A6},{BY2:.2f})",     (_A6,   BY2)),
        (22, "APQ_DER_ESQ_LEJ", f"Área chica DER esq lejana   ({L-_A6:.1f},{BY1:.2f})",(L-_A6,BY1)),
        (23, "APQ_DER_ESQ_CER", f"Área chica DER esq cercana  ({L-_A6:.1f},{BY2:.2f})",(L-_A6,BY2)),
        (24, "PEN_IZQ",         f"Penalti IZQ ({_PEN:.1f}, {CY:.2f})",            (_PEN,    CY)),
        (25, "PEN_DER",         f"Penalti DER ({L-_PEN:.1f}, {CY:.2f})",          (L-_PEN,  CY)),
    ]

# Marcas recomendadas por cámara (1-based)
RECOMMENDED = {
    "top":    [1, 2, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 20, 21, 24],
    "bottom": [3, 4, 5, 6, 7, 8, 9, 11, 16, 17, 18, 19, 22, 23, 25],
}


# ─── DIAGRAMA DEL CAMPO ───────────────────────────────────────────────────────

DIAG_SCALE = 9    # px/metro para el diagrama
DIAG_MARGIN = 30  # px de margen

def draw_field_diagram(catalog, half, path):
    """
    Guarda un PNG del campo completo con todos los puntos del catálogo numerados.
    Los RECOMENDADOS para esta cámara se destacan en verde, los otros en gris.
    """
    L, W = FIELD_LENGTH, FIELD_WIDTH
    S, M = DIAG_SCALE, DIAG_MARGIN
    rec = RECOMMENDED[half]

    img_w = int(L * S) + 2 * M
    img_h = int(W * S) + 2 * M + 70   # espacio para leyenda

    img = np.zeros((img_h, img_w, 3), np.uint8)

    # Fondo
    cv2.rectangle(img, (M, M), (M + int(L*S), M + int(W*S)), (0, 80, 0), -1)
    cv2.rectangle(img, (M, M), (M + int(L*S), M + int(W*S)), (255,255,255), 2)

    def fp(xm, ym):
        return (M + int(xm * S), M + int(ym * S))

    # Líneas básicas
    cx = M + int(L/2 * S)
    cv2.line(img, (cx, M), (cx, M + int(W*S)), (255,255,255), 1)

    AY1, AY2 = (W-_AW)/2, (W-_AW)/2 + _AW
    BY1, BY2 = (W-_BW)/2, (W-_BW)/2 + _BW
    CY = W/2

    # Áreas IZQ
    cv2.rectangle(img, fp(0, AY1), fp(_AP, AY2), (255,255,255), 1)
    cv2.rectangle(img, fp(0, BY1), fp(_A6, BY2), (255,255,255), 1)
    # Áreas DER
    cv2.rectangle(img, fp(L-_AP, AY1), fp(L, AY2), (255,255,255), 1)
    cv2.rectangle(img, fp(L-_A6, BY1), fp(L, BY2), (255,255,255), 1)
    # Círculo central
    cv2.circle(img, fp(L/2, W/2), int(_RCC * S), (255,255,255), 1)
    cv2.circle(img, fp(L/2, W/2), 3, (255,255,255), -1)
    # Arcos penalti
    cv2.ellipse(img, fp(_PEN, CY), (int(_RCC*S),int(_RCC*S)), 0, -53, 53, (255,255,255), 1)
    cv2.ellipse(img, fp(L-_PEN, CY), (int(_RCC*S),int(_RCC*S)), 0, 127, 233, (255,255,255), 1)

    # Zona destacada de esta cámara
    zcol = (0, 160, 80) if half == "top" else (160, 80, 0)
    x0_z = 0 if half == "top" else L/2
    x1_z = L/2 if half == "top" else L
    cv2.rectangle(img, fp(x0_z, 0), fp(x1_z, W), zcol, 3)
    label_zone = f"ZONA {half.upper()}"
    cv2.putText(img, label_zone, (fp(x0_z+1, 0)[0]+5, M+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, zcol, 1)

    # Puntos del catálogo
    for n, short, _, (xm, ym) in catalog:
        px, py = fp(xm, ym)
        is_rec = (n in rec)
        col = (0, 255, 100) if is_rec else (130, 130, 130)
        r = 7 if is_rec else 5
        cv2.circle(img, (px, py), r, col, -1)
        cv2.circle(img, (px, py), r+2, (255,255,255), 1)
        cv2.putText(img, str(n), (px+4, py-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

    # Leyenda
    yl = img_h - 58
    cv2.circle(img, (M, yl), 7, (0, 255, 100), -1)
    cv2.putText(img, "RECOMENDADO para esta camara", (M+12, yl+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
    cv2.circle(img, (M, yl+22), 5, (130, 130, 130), -1)
    cv2.putText(img, "disponible (menos visible desde aqui)", (M+12, yl+26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)
    cv2.putText(img, f"Camara {half.upper()} — elige los numeros que YES ves en el video",
                (M, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 1)

    cv2.imwrite(path, img)
    return img


# ─── POSICIÓN DE CADA MARCA EN LA IMAGEN FULL-FIELD PNG ──────────────────────

FIELD_PNG_SCALE  = 8     # px/metro para el overlay PNG
FIELD_PNG_MARGIN = 15    # px de margen

def world_to_field_png(xm, ym):
    """Coordenadas metros → píxel en el PNG del campo completo."""
    return (
        FIELD_PNG_MARGIN + int(xm * FIELD_PNG_SCALE),
        FIELD_PNG_MARGIN + int(ym * FIELD_PNG_SCALE),
    )

def draw_full_field_png():
    """
    Genera el PNG BGRA del campo completo (vista cenital) para el overlay.
    Incluye todas las líneas FIFA estándar.
    """
    L, W = FIELD_LENGTH, FIELD_WIDTH
    S, M = FIELD_PNG_SCALE, FIELD_PNG_MARGIN
    img_w = int(L * S) + 2*M
    img_h = int(W * S) + 2*M

    img = np.zeros((img_h, img_w, 4), np.uint8)
    img[:,:,0]=34; img[:,:,1]=120; img[:,:,2]=34; img[:,:,3]=230   # verde opaco

    WHITE = (255,255,255,255)
    LW = max(1, S//5)

    def p(xm, ym):
        return (M + int(xm*S), M + int(ym*S))

    AY1, AY2 = (W-_AW)/2, (W-_AW)/2+_AW
    BY1, BY2 = (W-_BW)/2, (W-_BW)/2+_BW
    CX, CY = L/2, W/2

    def seg(a, b):  cv2.line(img, p(*a), p(*b), WHITE, LW)
    def rect(tl,br): cv2.rectangle(img, p(*tl), p(*br), WHITE, LW)
    def arc(c,r,a0,a1):
        cv2.ellipse(img, p(*c), (int(r*S),int(r*S)), 0, a0, a1, WHITE, LW)

    # Borde
    rect((0,0),(L,W))
    # Línea central
    seg((CX,0),(CX,W))
    # Círculo central + punto
    arc((CX,CY), _RCC, 0, 360)
    cv2.circle(img, p(CX,CY), max(3,LW), WHITE, -1)
    # Áreas IZQ
    rect((0,AY1),(_AP,AY2))
    rect((0,BY1),(_A6,BY2))
    arc((_PEN,CY), _RCC, -53, 53)
    cv2.circle(img, p(_PEN,CY), max(3,LW), WHITE, -1)
    # Áreas DER
    rect((L-_AP,AY1),(L,AY2))
    rect((L-_A6,BY1),(L,BY2))
    arc((L-_PEN,CY), _RCC, 127, 233)
    cv2.circle(img, p(L-_PEN,CY), max(3,LW), WHITE, -1)
    # Arcos de esquina
    for xc, yc, a0, a1 in [(0,0,0,90),(0,W,-90,0),(L,0,90,180),(L,W,180,270)]:
        arc((xc,yc), 1.0, a0, a1)

    return img


# ─── LECTURA DEL VIDEO (imagen cilíndrica cruda, sin corrección) ──────────────

def get_frame_half(frame_idx, half):
    """Devuelve la mitad superior o inferior del frame VEO, SIN ningún remap."""
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video no encontrado: {os.path.abspath(VIDEO_PATH)}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total-1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"No se pudo leer frame {frame_idx}")
    mid = frame.shape[0] // 2
    return frame[:mid] if half == "top" else frame[mid:]


def infer_half_from_image_path(image_path, fallback="top"):
    name = os.path.basename(image_path).lower()
    if "cama" in name or "_top" in name or "top" in name:
        return "top"
    if "camb" in name or "_bottom" in name or "bottom" in name:
        return "bottom"
    return fallback


def output_homography_path(image_path, half):
    if image_path:
        abs_img = os.path.abspath(image_path)
        folder = os.path.dirname(abs_img)
        base = os.path.splitext(os.path.basename(abs_img))[0]
        return os.path.join(folder, f"{base}_homography.npy")
    return f"data/homography/H_veo_{half}.npy"


# ─── POSICIONES INICIALES DE LOS PUNTOS ──────────────────────────────────────

def _veo_cyl_px(key):
    """
    Devuelve (px, py) en píxeles de la imagen cilíndrica cruda (2048 × 1024 por mitad).
    No se aplica ningún remap: son coordenadas directas en el frame sin corregir.
    """
    sx, sy_n = VEO_NORM[key]
    px  = sx * VEO_STACKED_W
    py_stacked = sy_n * VEO_STACKED_W          # en imagen apilada 2048×2048
    py  = py_stacked if key in ("cl", "fl") else py_stacked - VEO_HALF_H
    return (int(px), int(py))


def get_initial_drag_points(selected, half, frame_w, frame_h, h_path_override=None):
    """
    Devuelve lista de [px, py] en píxeles cilíndricos crudos para cada marca.

    Estrategia:
      1. Si existe H guardada → back-proyectar directamente.
      2. Si no → H burda usando las 4 esquinas VEO (ya en px cilíndricos)
         + estimaciones para la línea central, luego proyectar todas las marcas.
    """
    world_pts = [(xm, ym) for _, _, _, (xm, ym) in selected]
    L, W = FIELD_LENGTH, FIELD_WIDTH

    # Opción 1: H existente
    h_path = h_path_override if h_path_override else f"data/homography/H_veo_{half}.npy"
    if os.path.exists(h_path):
        H = np.load(h_path)
        H_inv = np.linalg.inv(H)
        wpts = np.array(world_pts, dtype=np.float32).reshape(-1,1,2)
        ipts = cv2.perspectiveTransform(wpts, H_inv).reshape(-1,2)
        print("  Posiciones iniciales cargadas desde H existente.")
        return [[int(p[0]), int(p[1])] for p in ipts]

    # Opción 2: H burda usando esquinas VEO en espacio cilíndrico crudo
    # Las coordenadas VEO son directamente píxeles en el frame sin remap
    if half == "top":
        anchors_img   = [_veo_cyl_px("fl"), _veo_cyl_px("cl"),
                         (int(frame_w * 0.80), int(frame_h * 0.15)),
                         (int(frame_w * 0.78), int(frame_h * 0.82))]
        anchors_world = [(0, 0), (0, W), (L/2, 0), (L/2, W)]
    else:
        anchors_img   = [(int(frame_w * 0.20), int(frame_h * 0.15)),
                         (int(frame_w * 0.22), int(frame_h * 0.82)),
                         _veo_cyl_px("fr"), _veo_cyl_px("cr")]
        anchors_world = [(L/2, 0), (L/2, W), (L, 0), (L, W)]

    src = np.array(anchors_world, dtype=np.float32)
    dst = np.array(anchors_img,   dtype=np.float32)
    H_rough, _ = cv2.findHomography(src, dst)

    if H_rough is None:
        print("  AVISO: H burda fallida. Puntos iniciales en el centro.")
        return [[frame_w // 2, frame_h // 2]] * len(selected)

    wpts = np.array(world_pts, dtype=np.float32).reshape(-1,1,2)
    ipts = cv2.perspectiveTransform(wpts, H_rough).reshape(-1,2)
    print("  Posiciones iniciales estimadas desde esquinas VEO (imagen cilíndrica).")
    return [
        [int(np.clip(p[0], -300, frame_w + 300)),
         int(np.clip(p[1], -300, frame_h + 300))]
        for p in ipts
    ]


# ─── SELECCIÓN INTERACTIVA POR TERMINAL ──────────────────────────────────────

def ask_visible_points(half):
    """
    Muestra el catálogo completo, abre el diagrama del campo y pregunta
    al usuario cuáles marcas son visibles en el frame.
    Devuelve la sublista del catálogo seleccionada (mínimo 4 puntos).
    """
    catalog = build_catalog()
    rec = RECOMMENDED[half]

    # Guardar y abrir diagrama
    os.makedirs("data/frames", exist_ok=True)
    diag_path = os.path.abspath(f"data/frames/diagrama_campo_{half}.png")
    draw_field_diagram(catalog, half, diag_path)
    print(f"\n  [DIAGRAMA] {diag_path}")
    try:
        subprocess.Popen(["explorer", diag_path])
    except Exception:
        pass

    # Mostrar catálogo en terminal
    print(f"\n{'='*70}")
    print(f"  Cámara {half.upper()} — ¿Qué marcas ves en el vídeo?")
    print(f"{'='*70}")
    print(f"  {'N':>3}  {'Etiqueta':<18} Descripción")
    print(f"  {'-'*65}")
    for n, short, desc, (xm, ym) in catalog:
        tag = " *" if n in rec else "  "
        print(f"  {n:>3}{tag} {short:<18} {desc}")
    print(f"\n  * = recomendado para cámara {half.upper()}")
    print(f"  Introduce los números separados por coma (mínimo 4).")
    print(f"  Ejemplo: {','.join(str(x) for x in rec[:8])}\n")

    catalog_dict = {n: (short, desc, coords) for n, short, desc, coords in catalog}

    while True:
        try:
            raw = input("  Tu selección → ").strip()
            if not raw:
                # default: recomendados
                chosen = rec[:]
                print(f"  (usando recomendados: {chosen})")
            else:
                chosen = [int(x.strip()) for x in raw.split(",") if x.strip()]

            # Validar
            invalid = [x for x in chosen if x not in catalog_dict]
            if invalid:
                print(f"  [!] Números fuera de rango: {invalid}. Vuelve a intentarlo.")
                continue
            if len(chosen) < 4:
                print(f"  [!] Necesitas al menos 4 puntos (tienes {len(chosen)}). Añade más.")
                continue

            # Construir sublista
            selected = [(n, catalog_dict[n][0], catalog_dict[n][1], catalog_dict[n][2])
                        for n in chosen]
            break

        except ValueError:
            print("  [!] Formato inválido. Ejemplo: 1,2,5,6,7")

    print(f"\n  Puntos seleccionados ({len(selected)}):")
    for n, short, _, (xm, ym) in selected:
        print(f"    {n:>3}. {short:<20} ({xm:6.1f}, {ym:6.2f}) m")

    return selected


# ─── OVERLAY: WARP DEL CAMPO SOBRE EL FRAME ──────────────────────────────────

def compute_overlay(field_png, drag_pts, selected, fh, fw):
    """
    Calcula un warpPerspective del PNG completo del campo sobre el frame,
    usando los N puntos de control actuales y sus posiciones en el PNG.
    Necesita mínimo 4 puntos no colineales.
    """
    if len(drag_pts) < 4:
        return None

    # Posiciones de cada marca seleccionada en el PNG del campo
    src_png = np.array(
        [world_to_field_png(xm, ym) for _, _, _, (xm, ym) in selected],
        dtype=np.float32
    )
    dst_vid = np.array(drag_pts, dtype=np.float32)

    H, mask = cv2.findHomography(src_png, dst_vid, cv2.RANSAC, 8.0)
    if H is None:
        return None

    warped = cv2.warpPerspective(field_png, H, (fw, fh),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0,0,0,0))
    return warped


def blend_overlay(frame_bgr, warped_bgra, alpha):
    if warped_bgra is None:
        return frame_bgr
    mask  = warped_bgra[:,:,3:4].astype(np.float32) / 255.0 * alpha
    field = warped_bgra[:,:,:3].astype(np.float32)
    base  = frame_bgr.astype(np.float32)
    return (base*(1.0-mask) + field*mask).astype(np.uint8)


# ─── REJILLA DE METROS ────────────────────────────────────────────────────────

def draw_grid(canvas, drag_pts, selected):
    if len(drag_pts) < 4:
        return
    L, W = FIELD_LENGTH, FIELD_WIDTH
    world = np.array([(xm,ym) for _,_,_,(xm,ym) in selected], dtype=np.float32)
    video = np.array(drag_pts, dtype=np.float32)
    H, _ = cv2.findHomography(world, video)
    if H is None:
        return

    def proj(xm, ym):
        pt = cv2.perspectiveTransform(np.array([[[xm,ym]]], np.float32), H)[0][0]
        return (int(pt[0]), int(pt[1]))

    for xm in np.arange(0, L+0.1, 10.0):
        p1, p2 = proj(xm, 0), proj(xm, W)
        cv2.line(canvas, p1, p2, (190,190,190), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{xm:.0f}", (p1[0]+2, p1[1]+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (190,190,190), 1)
    for ym in np.arange(0, W+0.1, 10.0):
        cv2.line(canvas, proj(0,ym), proj(L,ym), (190,190,190), 1, cv2.LINE_AA)


# ─── DIBUJADO DE UI ──────────────────────────────────────────────────────────

# Paleta de colores para los puntos (BGR), máx 25
_COLORS = [
    (0,220,255),(0,255,120),(50,150,255),(200,80,255),(0,200,180),
    (255,180,0),(80,255,200),(255,100,80),(180,255,0),(0,160,255),
    (255,200,0),(180,80,255),(0,255,255),(120,80,255),(255,80,180),
    (0,180,100),(200,200,0),(80,180,255),(255,140,0),(100,255,80),
    (0,100,255),(180,255,180),(255,60,60),(60,255,255),(180,0,255),
]
HIT_R = 16
PAD   = 400   # px de borde negro alrededor del frame para situar puntos fuera


def draw_ui(canvas, drag_pts, selected, selected_idx, pad=0):
    """Dibuja puntos de control y líneas sobre canvas (pad=offset borde negro)."""
    pts_t = [(p[0]+pad, p[1]+pad) for p in drag_pts]
    n = len(pts_t)

    for i in range(n):
        for j in range(i+1, n):
            xi, yi = selected[i][3]; xj, yj = selected[j][3]
            if ((xi-xj)**2 + (yi-yj)**2)**0.5 < 20:
                cv2.line(canvas, pts_t[i], pts_t[j], (160,160,160), 1, cv2.LINE_AA)

    for i, pt in enumerate(pts_t):
        col = _COLORS[i % len(_COLORS)]
        r = 15 if (i == selected_idx) else 9
        cv2.circle(canvas, pt, r, (0,0,0), 2, cv2.LINE_AA)
        cv2.circle(canvas, pt, r, col,     1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 5, col,    -1, cv2.LINE_AA)
        n_cat = selected[i][0]; short = selected[i][1]
        lbl = f"{n_cat}:{short}"
        tx, ty = pt[0]+10, pt[1]-8
        cv2.putText(canvas, lbl, (tx+1,ty+1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 2)
        cv2.putText(canvas, lbl, (tx,  ty  ), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)


def draw_hud(display, n_pts, selected_idx, selected, alpha,
             show_field, show_grid, dirty, zoom):
    """HUD informativo dibujado siempre sobre la imagen de display final."""
    info = [
        (f"Alpha:{alpha:.2f}  Campo:{'ON' if show_field else 'OFF'}"
         f"  Grid:{'ON' if show_grid else 'OFF'}"
         f"  Pts:{n_pts}  Zoom:{zoom:.2f}x"),
        "LClick=mover | Ctrl+LClick=teleportar_sel | RClick+drag=pan | Rueda=zoom | Espacio=centrar_sel | 1-9/a-z=sel | flechas=nudge | S/R/A/Z/F/G/Q",
        "*** CAMBIOS SIN GUARDAR ***" if dirty else "",
    ]
    for li, txt in enumerate(info):
        if not txt: continue
        yy = 18 + li * 17
        col = (0,220,255) if li < 2 else (0,60,255)
        cv2.putText(display, txt, (9,yy+1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 2)
        cv2.putText(display, txt, (8,yy  ), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    if 0 <= selected_idx < n_pts:
        n_cat, short, desc, (xm, ym) = selected[selected_idx]
        # cv2.putText no renderiza Unicode: usar solo ASCII
        desc_ascii = desc.encode('ascii', errors='replace').decode('ascii').replace('?', '')
        lbl = f"SEL [{selected_idx+1}/{n_pts}] {n_cat}:{short} -> ({xm:.1f},{ym:.2f}) m | {desc_ascii}"
        col = _COLORS[selected_idx % len(_COLORS)]
        h = display.shape[0]
        cv2.putText(display, lbl, (9, h-10+1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 2)
        cv2.putText(display, lbl, (8, h-10  ), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)


# ─── GUARDAR HOMOGRAFÍA ───────────────────────────────────────────────────────

def save_homography(drag_pts, selected, half, output_path=None):
    """H: píxel perspectiva → metros campo."""
    world = np.array([(xm,ym) for _,_,_,(xm,ym) in selected], dtype=np.float32)
    video = np.array(drag_pts, dtype=np.float32)
    H, mask = cv2.findHomography(video, world, cv2.RANSAC, 5.0)
    if H is None:
        print("  [!] findHomography falló. Necesitas ≥4 puntos bien distribuidos.")
        return False
    inliers = int(mask.sum()) if mask is not None else len(drag_pts)
    path = output_path if output_path else f"data/homography/H_veo_{half}.npy"
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs("data/homography", exist_ok=True)
    np.save(path, H)
    # Errores de reproyección
    vp = video.reshape(-1,1,2)
    proj = cv2.perspectiveTransform(vp, H).reshape(-1,2)
    errs = [np.linalg.norm(proj[i] - world[i]) for i in range(len(drag_pts))]
    print(f"\n  OK Guardado: {os.path.abspath(path)}")
    print(f"    Puntos: {len(drag_pts)} | Inliers RANSAC: {inliers}")
    print(f"    Error medio: {np.mean(errs):.3f} m | Máx: {np.max(errs):.3f} m")
    return True


# ─── ESTADO + CALLBACK DE RATÓN ──────────────────────────────────────────────

state = {
    "drag_idx": -1, "selected": 0,
    "alpha": 0.45, "show_field": True, "show_grid": False,
    "points": None, "init_pts": None, "dirty": False,
    "dscale": 1.0,
    "zoom": 1.0,
    "pan_x": float(PAD), "pan_y": float(PAD),
    "pan_dragging": False, "pan_last": (0, 0),
}

def _display_to_orig(mx, my):
    """Display pixel → original image coords (puede ser negativo si está en el borde negro)."""
    dscale = state["dscale"]; zoom = state["zoom"]
    pcx = state["pan_x"] + mx / (dscale * zoom)
    pcy = state["pan_y"] + my / (dscale * zoom)
    return int(pcx - PAD), int(pcy - PAD)


def mouse_cb(event, x, y, flags, param):
    s = state
    pts = s["points"]
    xi, yi = _display_to_orig(x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            # Ctrl+click: teleporta el punto seleccionado a la posición del cursor
            pts[s["selected"]] = [xi, yi]
            s["dirty"] = True
        else:
            for i, p in enumerate(pts):
                if abs(xi - p[0]) < HIT_R and abs(yi - p[1]) < HIT_R:
                    s["drag_idx"] = i
                    s["selected"] = i
                    break
    elif event == cv2.EVENT_MOUSEMOVE:
        if s["drag_idx"] >= 0:
            pts[s["drag_idx"]] = [xi, yi]
            s["dirty"] = True
        if s["pan_dragging"]:
            lx, ly = s["pan_last"]
            factor = s["dscale"] * s["zoom"]
            s["pan_x"] -= (x - lx) / factor
            s["pan_y"] -= (y - ly) / factor
            s["pan_last"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        s["drag_idx"] = -1
    elif event == cv2.EVENT_RBUTTONDOWN:
        s["pan_dragging"] = True
        s["pan_last"] = (x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        s["pan_dragging"] = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        dscale = s["dscale"]; zoom = s["zoom"]
        pcx = s["pan_x"] + x / (dscale * zoom)
        pcy = s["pan_y"] + y / (dscale * zoom)
        factor = 1.25 if flags > 0 else 1.0 / 1.25
        s["zoom"] = float(np.clip(zoom * factor, 0.15, 16.0))
        new_zoom = s["zoom"]
        s["pan_x"] = pcx - x / (dscale * new_zoom)
        s["pan_y"] = pcy - y / (dscale * new_zoom)


# ─── NUDGE CON FLECHAS ────────────────────────────────────────────────────────

def handle_key(key_full, key_byte, n_pts):
    st  = state
    pts = st["points"]
    idx = st["selected"]

    # Selección por número: 1-9 → índices 0-8, 0 → 9, a-z → 10-35
    if ord('1') <= key_byte <= ord('9'):
        new = key_byte - ord('1')
        if new < n_pts: st["selected"] = new
        return
    if key_byte == ord('0'):
        if 9 < n_pts: st["selected"] = 9
        return
    # letras a-z para puntos 10+
    if ord('a') <= key_byte <= ord('z'):
        new = 10 + (key_byte - ord('a'))
        if new < n_pts: st["selected"] = new
        return

    # Espacio: centra la vista en el punto seleccionado
    if key_byte == ord(' '):
        p = pts[idx]
        dscale = st["dscale"]; zoom = st["zoom"]
        dw_vis = 1280 * dscale   # aprox ancho display
        dh_vis = 700  * dscale
        st["pan_x"] = (p[0] + PAD) - (dw_vis / 2) / (dscale * zoom)
        st["pan_y"] = (p[1] + PAD) - (dh_vis / 2) / (dscale * zoom)
        return

    # Flechas (Windows: valores >256, Linux: 81-84)
    step = 10 if (key_full & 0xFFFF0000 == 0x10000) else 1
    p = pts[idx]
    moved = False
    if   key_full in (2490368, 82): p[1] -= step; moved = True   # ↑
    elif key_full in (2621440, 84): p[1] += step; moved = True   # ↓
    elif key_full in (2424832, 81): p[0] -= step; moved = True   # ←
    elif key_full in (2555904, 83): p[0] += step; moved = True   # →
    if moved:
        st["dirty"] = True


# ─── BUCLE PRINCIPAL ─────────────────────────────────────────────────────────

def run(
    half,
    frame_idx,
    alpha_init,
    selected,
    image_path=None,
    homography_output_path=None
):

    # ---------------------------------------------------------
    # CARGA FRAME
    # ---------------------------------------------------------

    if image_path is not None:

        print(f"\nCargando imagen externa...")

        ext = os.path.splitext(image_path)[1].lower()

        # imagen normal
        if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:

            frame = cv2.imread(image_path)

            if frame is None:
                raise ValueError(
                    f"No se pudo cargar la imagen:\n{image_path}"
                )

        # numpy
        elif ext == ".npy":

            frame = np.load(image_path)

            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        else:
            raise ValueError(
                f"Formato no soportado: {ext}"
            )

    else:

        print(f"\nCargando frame {frame_idx} VEO...")
        frame = get_frame_half(frame_idx, half)

    # ---------------------------------------------------------

    fh, fw = frame.shape[:2]

    print(f"Resolución: {fw}x{fh}")
    print(f"  Frame cilíndrico (sin corrección): {fw}×{fh}")

    # PNG completo del campo para overlay
    field_png = draw_full_field_png()

    # Posiciones iniciales de los puntos de control
    init_pts = get_initial_drag_points(
        selected, half, fw, fh,
        h_path_override=homography_output_path
    )
    state["points"]   = [list(p) for p in init_pts]
    state["init_pts"] = [list(p) for p in init_pts]
    state["alpha"]    = alpha_init
    state["dirty"]    = False

    # Escalar la ventana para que quepa en pantallas pequeñas (máx 1280×700)
    MAX_W, MAX_H = 1280, 700
    dscale = min(1.0, MAX_W / fw, MAX_H / fh)
    dw, dh = int(fw * dscale), int(fh * dscale)
    state["dscale"] = dscale
    state["zoom"]   = 1.0
    state["pan_x"]  = float(PAD)
    state["pan_y"]  = float(PAD)
    print(f"  Escala display: {dscale:.3f}  →  ventana {dw}×{dh}")

    WIN = "Homografia Interactiva VEO"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, dw, dh)
    cv2.moveWindow(WIN, 20, 20)   # esquina superior izquierda, con margen
    cv2.setMouseCallback(WIN, mouse_cb)

    print(f"\n  {len(selected)} puntos de control listos.")
    print("  Arrastra cada punto hasta su marca real en el césped.")
    print("  Empieza por las ESQUINAS, luego afina los interiores.")
    print("  S=guardar  R=reset  A/Z=alfa  F=campo  G=rejilla  Q=salir\n")

    pc_h, pc_w = fh + 2*PAD, fw + 2*PAD   # tamaño del canvas con borde negro

    def _viewport(padded):
        """Extrae la región visible del canvas padded según zoom/pan y escala a (dw,dh)."""
        z = state["zoom"]
        vis_w = int(round(dw / (dscale * z)))
        vis_h = int(round(dh / (dscale * z)))
        x0, y0 = int(state["pan_x"]), int(state["pan_y"])
        x1, y1 = x0 + vis_w, y0 + vis_h
        crop = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(pc_w, x1), min(pc_h, y1)
        if sx1 > sx0 and sy1 > sy0:
            dx0, dy0 = sx0 - x0, sy0 - y0
            dx1, dy1 = dx0 + sx1 - sx0, dy0 + sy1 - sy0
            crop[dy0:dy1, dx0:dx1] = padded[sy0:sy1, sx0:sx1]
        return cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_LINEAR)

    while True:
        # Overlay y rejilla sobre canvas original (coords imagen sin pad)
        canvas = frame.copy()
        if state["show_field"]:
            warped = compute_overlay(field_png, state["points"], selected, fh, fw)
            canvas = blend_overlay(canvas, warped, state["alpha"])
        if state["show_grid"]:
            draw_grid(canvas, state["points"], selected)

        # Incrustar canvas en padded (borde negro)
        padded = np.zeros((pc_h, pc_w, 3), dtype=np.uint8)
        padded[PAD:PAD+fh, PAD:PAD+fw] = canvas

        # Dibujar puntos en padded (offset +PAD)
        draw_ui(padded, state["points"], selected, state["selected"], pad=PAD)

        # Extraer viewport y añadir HUD encima
        display = _viewport(padded)
        draw_hud(display, len(selected), state["selected"], selected,
                 state["alpha"], state["show_field"], state["show_grid"],
                 state["dirty"], state["zoom"])

        cv2.imshow(WIN, display)
        key_full = cv2.waitKey(16)
        key      = key_full & 0xFF

        if key in (27, ord('q')):
            break
        elif key == ord('s'):
            ok = save_homography(
                state["points"], selected, half,
                output_path=homography_output_path
            )
            if ok: state["dirty"] = False
        elif key == ord('r'):
            state["points"] = [list(p) for p in state["init_pts"]]
            state["dirty"]  = False
            print("  Puntos reiniciados.")
        elif key == ord('a'):
            state["alpha"] = min(1.0, state["alpha"] + 0.05)
        elif key == ord('z'):
            state["alpha"] = max(0.0, state["alpha"] - 0.05)
        elif key == ord('f'):
            state["show_field"] = not state["show_field"]
        elif key == ord('g'):
            state["show_grid"]  = not state["show_grid"]
        else:
            handle_key(key_full, key, len(selected))

    cv2.destroyAllWindows()
    return state["points"]


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def ask_field_dimensions():
    """
    Pregunta las medidas del campo al usuario.
    Actualiza los globals FIELD_LENGTH y FIELD_WIDTH.
    Permite pulsar Enter para usar los valores del archivo .veo.
    """
    global FIELD_LENGTH, FIELD_WIDTH

    print(f"\n{'='*60}")
    print(f"  MEDIDAS DEL CAMPO")
    print(f"{'='*60}")
    print(f"  Valores del archivo .veo:  {FIELD_LENGTH} × {FIELD_WIDTH} m")
    print(f"  Pulsa Enter en ambos para usarlos.")
    print()

    while True:
        try:
            raw_l = input(f"  Longitud (m) [{FIELD_LENGTH}]: ").strip()
            raw_w = input(f"  Anchura  (m) [{FIELD_WIDTH}]:  ").strip()
            length = float(raw_l) if raw_l else FIELD_LENGTH
            width  = float(raw_w) if raw_w else FIELD_WIDTH
            if length <= 0 or width <= 0:
                print("  Las medidas deben ser positivas.")
                continue
            if not (85 <= length <= 125):
                print(f"  Longitud {length}m fuera del rango habitual (85-125m). Continuar? [s/N] ", end="")
                if input().strip().lower() != "s":
                    continue
            if not (50 <= width <= 100):
                print(f"  Anchura {width}m fuera del rango habitual (50-100m). Continuar? [s/N] ", end="")
                if input().strip().lower() != "s":
                    continue
            FIELD_LENGTH = length
            FIELD_WIDTH  = width
            print(f"\n  Campo: {FIELD_LENGTH} × {FIELD_WIDTH} m")
            break
        except ValueError:
            print("  Formato inválido. Escribe un número (ej: 105.0)")


def main():
    global FIELD_LENGTH, FIELD_WIDTH

    p = argparse.ArgumentParser(
        description="Calibración interactiva de homografía")

    # imagen opcional
    p.add_argument(
        "--imagen",
        type=str,
        default=None,
        help="Ruta a frame externo (.png/.jpg/.jpeg)"
    )

    # compatibilidad antigua
    p.add_argument(
        "imagen_legacy",
        nargs="?",
        default=None,
        help="(legacy) ruta imagen"
    )

    p.add_argument("--half", choices=["top", "bottom"], default=None)

    p.add_argument("--frame", type=int, default=3000)

    p.add_argument("--alpha", type=float, default=0.45)

    p.add_argument("--longitud", type=float, default=None)
    p.add_argument("--anchura", type=float, default=None)

    args = p.parse_args()

    # compatibilidad:
    image_path = args.imagen if args.imagen else args.imagen_legacy

    # inferir cámara automáticamente
    if args.half is not None:
        half = args.half
    elif image_path:
        half = infer_half_from_image_path(image_path, fallback="top")
    else:
        half = "top"

    # dimensiones campo
    if (args.longitud is None) ^ (args.anchura is None):
        p.error("Debes indicar --longitud y --anchura juntos.")

    if args.longitud is not None:
        FIELD_LENGTH = args.longitud
        FIELD_WIDTH = args.anchura

    print("\nHOMOGRAFIA INTERACTIVA")

    if image_path:
        print(f"Modo imagen externa")
        print(f"Imagen : {image_path}")
    else:
        print(f"Modo vídeo VEO")
        print(f"Frame  : {args.frame}")

    print(f"Cámara : {half.upper()}")

    # medidas
    if args.longitud is not None:
        print(f"Campo: {FIELD_LENGTH} × {FIELD_WIDTH} m")
    else:
        ask_field_dimensions()

    # selección marcas
    selected = ask_visible_points(half)

    # output H
    h_out = output_homography_path(image_path, half)

    # ejecutar
    final_pts = run(
        half=half,
        frame_idx=args.frame,
        alpha_init=args.alpha,
        selected=selected,
        image_path=image_path,
        homography_output_path=h_out
    )

    print("\nPuntos finales:")
    for i, (pt, (n, short, _, (xm, ym))) in enumerate(zip(final_pts, selected)):
        print(
            f"{i+1:>2}. "
            f"{n:>2}:{short:<20} "
            f"pixel ({pt[0]:5d},{pt[1]:4d}) "
            f"→ ({xm:.1f},{ym:.2f}) m"
        )

    if state["dirty"]:
        print("\n[!] Hay cambios sin guardar.")

if __name__ == "__main__":
    main()
