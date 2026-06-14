"""
marcar_tiempos.py
=================
Herramienta visual para marcar los intervalos de juego (inicio/fin de cada
tiempo) sobre el vídeo estándar VEO (1920×1080).

Uso:
    python marcar_tiempos.py ruta/al/video_standard.mp4

Controles:
    ←  /  →        Retroceder / avanzar 1 frame
    A  /  D        Retroceder / avanzar 30 frames (~1 s)
    Q  /  E        Retroceder / avanzar 300 frames (~10 s)
    Trackbar       Saltar a cualquier frame directamente

    1   →  Marcar INICIO del 1er tiempo   (t1_inicio)
    2   →  Marcar FIN    del 1er tiempo   (t1_fin)
    3   →  Marcar INICIO del 2º tiempo    (t2_inicio)
    4   →  Marcar FIN    del 2º tiempo    (t2_fin)

    S   →  Guardar resultado en tiempos_partido.json
    ESC →  Salir

Al guardar imprime también los valores exactos listos para copiar en veo_app.py.
"""

import sys
import json
import cv2
import numpy as np

WINDOW = "marcar_tiempos  |  1=inicio1T  2=fin1T  3=inicio2T  4=fin2T  S=guardar  ESC=salir"
OUTPUT_JSON = "tiempos_partido.json"

# Colores BGR
COL_1T  = (255, 180,  50)   # azul-claro  → 1er tiempo
COL_2T  = (50,  100, 220)   # rojo-coral  → 2º tiempo
COL_CUR = (50,  230,  50)   # verde       → frame actual
COL_TXT = (255, 255, 255)


def fmt_time(frame: int, fps: float) -> str:
    total_s = frame / fps
    m = int(total_s // 60)
    s = total_s % 60
    return f"{m:02d}:{s:05.2f}  (frame {frame})"


def draw_overlay(frame_img, marks: dict, cur_frame: int, fps: float, total: int):
    h, w = frame_img.shape[:2]
    overlay = frame_img.copy()

    # Barra de progreso en la parte inferior
    bar_y  = h - 18
    bar_h  = 12
    bar_x0 = 10
    bar_x1 = w - 10
    cv2.rectangle(overlay, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (40, 40, 40), -1)

    def bar_x(f):
        return int(bar_x0 + (f / max(total - 1, 1)) * (bar_x1 - bar_x0))

    # Pintar rangos marcados
    for (k_s, k_e), col in [
        (('t1_inicio', 't1_fin'), COL_1T),
        (('t2_inicio', 't2_fin'), COL_2T),
    ]:
        if marks.get(k_s) is not None and marks.get(k_e) is not None:
            xs = bar_x(marks[k_s])
            xe = bar_x(marks[k_e])
            cv2.rectangle(overlay, (xs, bar_y), (xe, bar_y + bar_h), col, -1)

    # Marcas individuales
    label_map = {
        't1_inicio': ('1T inicio', COL_1T),
        't1_fin':    ('1T fin',   COL_1T),
        't2_inicio': ('2T inicio', COL_2T),
        't2_fin':    ('2T fin',   COL_2T),
    }
    for key, (label, col) in label_map.items():
        f = marks.get(key)
        if f is not None:
            xm = bar_x(f)
            cv2.line(overlay, (xm, bar_y - 4), (xm, bar_y + bar_h + 4), col, 2)

    # Posición actual
    xc = bar_x(cur_frame)
    cv2.line(overlay, (xc, bar_y - 6), (xc, bar_y + bar_h + 6), COL_CUR, 2)

    # Panel de texto superior
    panel_h = 140
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (15, 15, 15), -1)

    def put(text, row, col=COL_TXT, scale=0.52, bold=False):
        thick = 2 if bold else 1
        cv2.putText(overlay, text, (12, 22 + row * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

    put(f"Frame actual:  {fmt_time(cur_frame, fps)}", 0, COL_CUR, bold=True)

    for i, (key, (label, col)) in enumerate(label_map.items()):
        f = marks.get(key)
        val = fmt_time(f, fps) if f is not None else "—  (no marcado)"
        put(f"[{i+1}] {label:<12}  {val}", i + 1, col)

    # Instrucciones rapidas
    put("← → : ±1f   A D : ±1s   Q E : ±10s", 5, (160, 160, 160), scale=0.42)

    cv2.addWeighted(overlay, 0.88, frame_img, 0.12, 0, frame_img)
    return frame_img


def main():
    if len(sys.argv) < 2:
        print("Uso: python marcar_tiempos.py <video_standard.mp4>")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: no se puede abrir '{video_path}'")
        sys.exit(1)

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 29.97
    cur    = 0
    marks  = {'t1_inicio': None, 't1_fin': None,
               't2_inicio': None, 't2_fin': None}
    dirty  = True   # necesita leer frame nuevo

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    def on_trackbar(val):
        nonlocal cur, dirty
        if val != cur:
            cur   = val
            dirty = True

    cv2.createTrackbar("Frame", WINDOW, 0, total - 1, on_trackbar)

    frame_img = None

    while True:
        if dirty:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            ret, raw = cap.read()
            if not ret:
                cur = max(0, cur - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
                ret, raw = cap.read()
            if ret:
                frame_img = raw.copy()
            dirty = False
            cv2.setTrackbarPos("Frame", WINDOW, cur)

        if frame_img is not None:
            disp = frame_img.copy()
            draw_overlay(disp, marks, cur, fps, total)
            cv2.imshow(WINDOW, disp)

        key = cv2.waitKey(30) & 0xFF

        if key == 27:        # ESC
            break
        elif key == 83 or key == ord('s'):   # S → guardar
            _save(marks, fps, video_path)
        elif key == ord('1'):
            marks['t1_inicio'] = cur
            print(f"  [1T inicio] frame {cur}  →  {fmt_time(cur, fps)}")
        elif key == ord('2'):
            marks['t1_fin'] = cur
            print(f"  [1T fin]    frame {cur}  →  {fmt_time(cur, fps)}")
        elif key == ord('3'):
            marks['t2_inicio'] = cur
            print(f"  [2T inicio] frame {cur}  →  {fmt_time(cur, fps)}")
        elif key == ord('4'):
            marks['t2_fin'] = cur
            print(f"  [2T fin]    frame {cur}  →  {fmt_time(cur, fps)}")
        # Navegación
        elif key == 81 or key == ord('a'):   # ← o A : -1s
            cur = max(0, cur - int(fps))
            dirty = True
        elif key == 83 or key == ord('d'):   # → o D : +1s
            cur = min(total - 1, cur + int(fps))
            dirty = True
        elif key == 2:                        # ← flecha
            cur = max(0, cur - 1)
            dirty = True
        elif key == 3:                        # → flecha
            cur = min(total - 1, cur + 1)
            dirty = True
        elif key == ord('q'):                 # Q : -10s
            cur = max(0, cur - int(fps * 10))
            dirty = True
        elif key == ord('e'):                 # E : +10s
            cur = min(total - 1, cur + int(fps * 10))
            dirty = True

    cap.release()
    cv2.destroyAllWindows()

    # Preguntar si guardar al salir
    if any(v is not None for v in marks.values()):
        _save(marks, fps, video_path)


def _save(marks: dict, fps: float, video_path: str):
    def to_min(f):
        return round(f / fps / 60, 3) if f is not None else None

    result = {
        'video':     video_path,
        'fps':       round(fps, 4),
        't1_inicio_frame': marks['t1_inicio'],
        't1_fin_frame':    marks['t1_fin'],
        't2_inicio_frame': marks['t2_inicio'],
        't2_fin_frame':    marks['t2_fin'],
        't1_inicio_min': to_min(marks['t1_inicio']),
        't1_fin_min':    to_min(marks['t1_fin']),
        't2_inicio_min': to_min(marks['t2_inicio']),
        't2_fin_min':    to_min(marks['t2_fin']),
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Guardado en: {OUTPUT_JSON}")
    print(f"{'='*55}")
    print(f"  Valores para veo_app.py (en minutos):")
    print(f"    Inicio 1T : {result['t1_inicio_min']}")
    print(f"    Fin    1T : {result['t1_fin_min']}")
    print(f"    Inicio 2T : {result['t2_inicio_min']}")
    print(f"    Fin    2T : {result['t2_fin_min']}")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
