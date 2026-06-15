"""
generar_prototipos_v3.py  (v4.0)
Nueva arquitectura híbrida en 4 fases:

  FASE 1 — Clustering principal (k=3)
    · 2 clusters grandes → player_home / player_away
    · Cluster pequeño   → skip  (GKs + árbitros mezclados)
    · Prototipo desde los top-N más centrales del cluster
    · max_dist = percentil 90 distancias intra-cluster  (umbral de confianza)

  FASE 2 — Pool de outliers → clases especiales
    · Outliers: det con dist > max_dist contra home Y away
    · Si --H_a / --H_b: split automático por posición
        x < 16.5 m              → pool GK izquierda
        x > field_length-16.5 m → pool GK derecha
        zona central             → pool árbitro
    · Sin homografías: un único pool para todo
    · Galería interactiva por pool:
        S = aceptar como prototipo (etiqueta la clase)
        D = dividir en k sub-clusters (etiqueta cada uno)
        R = ver más muestras (siguiente página)
        X = saltar este pool

  Todos los prototipos incluyen: hue_sig + BGR top3/mean/median + max_dist.

Uso:
  python generar_prototipos_v3.py \\
    --video  data/videos/tor-cgi_pan.mp4 \\
    --model  runs/detect/modelo_players_v24_panoramic2/weights/best.pt \\
    --output prototypes_v3_tor_cgi.pkl \\
    --H_a    data/videos/camA-torroella_homography.npy \\
    --H_b    data/videos/camB-torroella_homography.npy \\
    --field_length 100 --field_width 60 \\
    --intervals "17:11-1:06:00" --conf 0.45 --k 3
"""

import argparse, sys, pickle, cv2, numpy as np
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_core import (
    preprocess_half, init_undistort_maps, HALF_H,
    get_grass_hue, get_torso_crop, get_player_color,
    get_hue_signature, get_hue_signature_masked,
    extract_sam_masks_batch, _HUE_BINS, hue_sig_distance,
    classify_player_v3, CLASSIFY_V3_MAX_SCORE, CLASSIFY_V3_MARGIN,
    pixel_to_field,
)
from ultralytics import YOLO

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    print("[AVISO] sklearn no instalado: pip install scikit-learn"); sys.exit(1)

CLASES_VALIDAS = ['player_home', 'player_away', 'gk_home', 'gk_away', 'referee', 'skip']
CROP_W, CROP_H = 52, 88
COLS_GALLERY   = 14
PAGE_SIZE      = 20
PA_DEPTH_M     = 16.5   # profundidad del área penal


# ═══════════════════════════════════════════════════════════
#  1. SAMPLING
# ═══════════════════════════════════════════════════════════

def sample_video(video_path, model, sample_sec, step_s, conf,
                 sam_model=None, intervals=None, default_home_left=True, verbose=True):
    cap       = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    cap.release()

    # intervals: list of (start_s, end_s, home_left|None)
    def _get_home_left(t):
        if intervals:
            for t_start, t_end, hl in intervals:
                if t_start <= t < t_end:
                    return hl if hl is not None else default_home_left
        return default_home_left

    if intervals:
        timestamps = []
        for t_start, t_end, _ in intervals:
            t_end = min(t_end, total_sec - 1)
            timestamps.extend(np.arange(t_start, t_end, step_s).tolist())
        timestamps = np.array(sorted(timestamps))
    else:
        timestamps = np.arange(0, min(sample_sec, total_sec - 1), step_s)

    if verbose:
        print(f'  Video: {total_sec:.0f}s @ {fps:.1f}fps  '
              f'→  {len(timestamps)} frames × 2 cámaras')

    rows = []
    cap  = cv2.VideoCapture(video_path)

    for fi, t in enumerate(timestamps):
        if verbose and fi % 20 == 0:
            print(f'    [{fi}/{len(timestamps)}] t={t:.0f}s  dets={len(rows)}')

        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_raw = cap.read()
        if not ret:
            continue

        h_fr             = frame_raw.shape[0]
        frame_brightness = float(np.mean(cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)))
        home_left_local  = _get_home_left(t)
        for half_raw, cam_side in [
            (frame_raw[:h_fr // 2], 'left'),
            (frame_raw[h_fr // 2:], 'right'),
        ]:
            cam_img = preprocess_half(half_raw, cam_side)
            gh      = get_grass_hue(cam_img) or 55.0
            results = model(cam_img, verbose=False, conf=conf)
            if not results or results[0].boxes is None:
                continue

            boxes_np    = results[0].boxes.xyxy.cpu().numpy()
            confs_np    = results[0].boxes.conf.cpu().numpy()
            bboxes_list = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes_np]
            sam_masks   = extract_sam_masks_batch(cam_img, bboxes_list, sam_model)

            for idx_b, (box_t, c_t) in enumerate(zip(boxes_np, confs_np)):
                x1, y1, x2, y2 = map(int, box_t)
                crop = cam_img[y1:y2, x1:x2]
                if crop.size == 0 or (y2 - y1) < 30:
                    continue

                mask  = sam_masks[idx_b] if idx_b < len(sam_masks) else None
                torso = get_torso_crop(crop)
                if torso.size == 0:
                    continue

                sig = (get_hue_signature_masked(crop, mask, gh)
                       if mask is not None
                       else get_hue_signature(torso, gh))
                med, mean_, top3 = get_player_color(torso, gh)

                rows.append({
                    'crop':       cv2.resize(crop,  (CROP_W, CROP_H)),
                    'torso':      cv2.resize(torso, (64, 64)),
                    'hue_sig':    sig.tolist(),
                    'median':     med.tolist(),
                    'mean':       mean_.tolist(),
                    'top3':       [c.tolist() for c in top3],
                    'gh':         gh,
                    't':          t,
                    'frame_idx':  frame_idx,
                    'cam':        cam_side,
                    'bbox':       (x1, y1, x2, y2),
                    'used_sam':   mask is not None,
                    'pos_m':           None,
                    'brightness':      frame_brightness,
                    'home_left_local': home_left_local,
                })

    cap.release()
    if verbose:
        n_sam = sum(1 for r in rows if r.get('used_sam'))
        print(f'  Total: {len(rows)} detecciones  '
              f'(SAM2: {n_sam} / {len(rows)}  torso crop: {len(rows)-n_sam})')
    return rows


def compute_pos_m(rows, H_a, H_b):
    """Calcula pos_m para cada row usando las homografías."""
    n_ok = 0
    for r in rows:
        if r.get('pos_m') is not None:
            continue
        H = H_a if r['cam'] == 'left' else H_b
        if H is None:
            continue
        x1, y1, x2, y2 = r['bbox']
        cx = (x1 + x2) / 2.0
        cy = float(y2)
        try:
            xm, ym   = pixel_to_field(cx, cy, H)
            r['pos_m'] = (float(xm), float(ym))
            n_ok += 1
        except Exception:
            r['pos_m'] = None
    print(f'  pos_m calculado para {n_ok}/{len(rows)} detecciones')


# ═══════════════════════════════════════════════════════════
#  2. GALLERY RENDERING
# ═══════════════════════════════════════════════════════════

def _hue_bar(sig, width=120, height=20):
    bar  = np.zeros((height, width, 3), dtype=np.uint8)
    n    = _HUE_BINS
    bins = np.array(sig[:n], dtype=np.float32)
    wf   = float(sig[n])
    df   = float(sig[n + 1])
    bw   = max(1, int(width * 0.80) // n)
    for i, v in enumerate(bins):
        hue_deg = int(i * 180 / n)
        rgb     = cv2.cvtColor(np.array([[[hue_deg, 220, 200]]], dtype=np.uint8),
                               cv2.COLOR_HSV2BGR)[0][0]
        h_fill  = max(1, int(v * height))
        x0      = i * bw
        bar[height - h_fill:, x0:x0 + bw] = rgb
    x_wd = int(width * 0.82)
    wbw  = int(width * 0.08)
    h_w  = max(1, int(wf * height))
    h_d  = max(1, int(df * height))
    bar[height - h_w:, x_wd:x_wd + wbw]           = (220, 220, 220)
    bar[height - h_d:, x_wd + wbw:x_wd + 2 * wbw] = (30,  30,  30)
    return bar


def _render_rows_block(rows_list, title, n_cols=COLS_GALLERY,
                       header_color=(40, 40, 60)):
    """Bloque visual: header + fila de crops + hue bar."""
    HEADER_H = 38
    BAR_H    = 20

    crops = [r['crop'] for r in rows_list[:n_cols]]
    while len(crops) < n_cols:
        crops.append(np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8))
    crops_img = np.hstack(crops)
    row_w     = crops_img.shape[1]

    if rows_list:
        sigs     = np.array([r['hue_sig'] for r in rows_list], dtype=np.float32)
        centroid = sigs.mean(axis=0)
    else:
        centroid = np.zeros(_HUE_BINS + 2)

    dom = int(np.argmax(centroid[:_HUE_BINS]) * 180 / _HUE_BINS)
    wf  = centroid[_HUE_BINS]
    df  = centroid[_HUE_BINS + 1]

    header = np.full((HEADER_H, row_w, 3), header_color, dtype=np.uint8)
    cv2.putText(header, title,
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 80), 1)
    cv2.putText(header, f'hue_dom={dom}°  W={wf:.2f}  D={df:.2f}',
                (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 255), 1)

    hue_bar = _hue_bar(centroid.tolist(), width=row_w, height=BAR_H)
    sep     = np.full((4, row_w, 3), (80, 80, 80), dtype=np.uint8)
    return np.vstack([header, crops_img, hue_bar, sep])


def _render_cluster_image(rows, labels, ki, n_show=COLS_GALLERY):
    idxs     = np.where(labels == ki)[0]
    n_det    = len(idxs)
    sigs     = np.array([rows[i]['hue_sig'] for i in idxs], dtype=np.float32)
    centroid = sigs.mean(axis=0)
    dists    = np.linalg.norm(sigs - centroid, axis=1)
    best_idx = idxs[np.argsort(dists)][:n_show]
    top_rows = [rows[i] for i in best_idx]
    dom      = int(np.argmax(centroid[:_HUE_BINS]) * 180 / _HUE_BINS)
    wf       = centroid[_HUE_BINS]; df = centroid[_HUE_BINS + 1]
    title    = f'CLUSTER {ki}  ({n_det} dets)  hue_dom={dom}°  W={wf:.2f}  D={df:.2f}'
    return _render_rows_block(top_rows, title)


def show_cluster_gallery(rows, labels, k):
    cluster_imgs = [_render_cluster_image(rows, labels, ki) for ki in range(k)]
    gallery      = np.vstack(cluster_imgs)
    max_h        = 900
    if gallery.shape[0] > max_h:
        scale   = max_h / gallery.shape[0]
        gallery = cv2.resize(gallery, (int(gallery.shape[1] * scale), max_h))
    cv2.imshow('Clusters — cierra con cualquier tecla', gallery)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════
#  3. PURITY INSPECTION  (igual que v3.1)
# ═══════════════════════════════════════════════════════════

def show_subcluster_gallery(rows, cluster_idxs, sub_labels, ki, n_show=10):
    HEADER_H = 26; BAR_H = 20; SEP_W = 8
    sub_blocks = []
    for sub_id in range(2):
        sub_idxs = cluster_idxs[sub_labels == sub_id]
        n_sub    = len(sub_idxs)
        if n_sub == 0:
            sub_blocks.append(np.zeros((HEADER_H + CROP_H + BAR_H,
                                        CROP_W * n_show, 3), dtype=np.uint8))
            continue
        sigs_sub = np.array([rows[i]['hue_sig'] for i in sub_idxs], dtype=np.float32)
        centroid = sigs_sub.mean(axis=0)
        dists    = np.linalg.norm(sigs_sub - centroid, axis=1)
        best     = sub_idxs[np.argsort(dists)][:n_show]
        crops_img = np.hstack([rows[i]['crop'] for i in best] +
                               [np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)]
                               * max(0, n_show - len(best)))
        row_w    = crops_img.shape[1]
        hue_bar  = _hue_bar(centroid.tolist(), width=row_w, height=BAR_H)
        label_c  = (120, 220, 255) if sub_id == 0 else (255, 180, 100)
        header   = np.full((HEADER_H, row_w, 3), (30, 30, 50), dtype=np.uint8)
        dom      = int(np.argmax(centroid[:_HUE_BINS]) * 180 / _HUE_BINS)
        cv2.putText(header,
                    f'Sub-{"AB"[sub_id]}  ({n_sub} dets)  hue_dom={dom}  '
                    f'W={centroid[_HUE_BINS]:.2f}  D={centroid[_HUE_BINS+1]:.2f}',
                    (6, HEADER_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, label_c, 1)
        sub_blocks.append(np.vstack([header, crops_img, hue_bar]))

    max_h  = max(b.shape[0] for b in sub_blocks)
    padded = []
    for b in sub_blocks:
        if b.shape[0] < max_h:
            b = np.vstack([b, np.zeros((max_h - b.shape[0], b.shape[1], 3), dtype=np.uint8)])
        padded.append(b)

    sep     = np.full((max_h, SEP_W, 3), (90, 90, 90), dtype=np.uint8)
    gallery = np.hstack([padded[0], sep, padded[1]])
    title   = np.full((32, gallery.shape[1], 3), (15, 15, 15), dtype=np.uint8)
    cv2.putText(title, f'CLUSTER {ki} \u2014 inspeccion de pureza (Sub-A vs Sub-B)',
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 80), 1)
    cv2.imshow(f'Pureza cluster {ki} \u2014 cierra con cualquier tecla',
               np.vstack([title, gallery]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inspect_cluster_purity(rows, labels, k, n_show=10):
    new_labels = labels.copy()
    next_id    = k

    print(f'\n{"="*60}')
    print(f'  FASE 2b: INSPECCION DE PUREZA')
    print(f'  Sub-KMeans k=2 por cluster — {n_show} muestras por sub-grupo.')
    print(f'  dist_Hellinger > 0.12 entre sub-centroides = posible mezcla.')
    print(f'{"="*60}')

    X = np.array([r['hue_sig'] for r in rows], dtype=np.float32)

    for ki in range(k):
        orig_idxs = np.where(labels == ki)[0]
        if len(orig_idxs) < 6:
            print(f'\n  Cluster {ki}: {len(orig_idxs)} muestras — omitido')
            continue

        sigs     = np.array([rows[i]['hue_sig'] for i in orig_idxs], dtype=np.float32)
        centroid = sigs.mean(axis=0)
        dom      = int(np.argmax(centroid[:_HUE_BINS]) * 180 / _HUE_BINS)
        inertia  = float(np.mean(np.linalg.norm(sigs - centroid, axis=1)))

        sub_km = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=200)
        sub_lb = sub_km.fit_predict(np.sqrt(sigs))
        n0, n1 = int((sub_lb == 0).sum()), int((sub_lb == 1).sum())
        sep    = hue_sig_distance(sub_km.cluster_centers_[0] ** 2,
                                  sub_km.cluster_centers_[1] ** 2)

        hint = '  <- POSIBLE MEZCLA' if sep > 0.12 else '  <- aspecto uniforme'
        print(f'\n  Cluster {ki}  ({len(orig_idxs)} dets)  '
              f'hue_dom={dom}°  inercia={inertia:.3f}')
        print(f'    Sub-A:{n0}  Sub-B:{n1}  '
              f'dist_Hellinger(A,B)={sep:.3f}{hint}')

        show_subcluster_gallery(rows, orig_idxs, sub_lb, ki, n_show)

        resp = input(f'  Dividir cluster {ki} en Sub-A y Sub-B? [y/n]: ').strip().lower()
        if resp == 'y':
            for j, orig_idx in enumerate(orig_idxs):
                if sub_lb[j] == 1:
                    new_labels[orig_idx] = next_id
            print(f'    Sub-A → ID={ki} ({n0} dets)  |  Sub-B → ID={next_id} ({n1} dets)')
            next_id += 1
        else:
            print(f'    Cluster {ki} sin cambios')

    new_k = next_id
    if new_k > k:
        print(f'\n  {k} clusters → {new_k} tras {new_k-k} division(es)')
    return new_labels, new_k


# ═══════════════════════════════════════════════════════════
#  4. PROTOTYPE BUILDING
# ═══════════════════════════════════════════════════════════

def build_proto_for_class(cls_rows):
    if not cls_rows:
        return None
    meds  = np.array([r['median']  for r in cls_rows], dtype=np.float32)
    means = np.array([r['mean']    for r in cls_rows], dtype=np.float32)
    sigs  = np.array([r['hue_sig'] for r in cls_rows], dtype=np.float32)
    top3s = [r['top3'] for r in cls_rows]

    top3_ok = [t for t in top3s if len(t) >= 3]
    proto_top3 = (
        [np.median(np.array(top3_ok, dtype=np.float32)[:, i, :], axis=0)
         .astype(np.uint8).tolist() for i in range(3)]
        if top3_ok else [[128, 128, 128]] * 3
    )
    return {
        'median':    np.median(meds,  axis=0).astype(np.uint8).tolist(),
        'mean':      np.median(means, axis=0).astype(np.uint8).tolist(),
        'top3':      proto_top3,
        'hue_sig':   sigs.mean(axis=0).tolist(),
        'n_samples': len(cls_rows),
    }


def build_proto_top_n(pool_rows, n_top=20):
    """
    Prototipo desde los top-N más centrales del pool.
    max_dist = percentil 90 de las distancias intra-pool → umbral de confianza.
    """
    if not pool_rows:
        return None
    sigs     = np.array([r['hue_sig'] for r in pool_rows], dtype=np.float32)
    centroid = sigs.mean(axis=0)
    dists    = np.linalg.norm(sigs - centroid, axis=1)
    n_use    = min(n_top, len(pool_rows))
    top_rows = [pool_rows[i] for i in np.argsort(dists)[:n_use]]
    max_dist = float(np.percentile(dists, 90))
    proto    = build_proto_for_class(top_rows)
    if proto:
        proto['max_dist'] = round(max_dist, 4)
        proto['n_total']  = len(pool_rows)
    return proto


# ═══════════════════════════════════════════════════════════
#  5. INTERACTIVE POOL
# ═══════════════════════════════════════════════════════════

def interactive_pool(pool_rows, pool_name, page_size=PAGE_SIZE):
    """
    Galería paginada de un pool de detecciones.
    Devuelve dict {clase: [rows]} con 0..N entradas.
      S = aceptar pool completo → pide clase
      D = sub-clustering interactivo → etiqueta cada sub
      R = siguiente página
      X = saltar
    """
    if not pool_rows:
        print(f'  Pool "{pool_name}": vacío — omitido')
        return {}

    n    = len(pool_rows)
    sigs = np.array([r['hue_sig'] for r in pool_rows], dtype=np.float32)
    centroid = sigs.mean(axis=0)
    dists    = np.linalg.norm(sigs - centroid, axis=1)
    sorted_rows = [pool_rows[i] for i in np.argsort(dists)]

    page = 0
    while True:
        start     = page * page_size
        end       = min(start + page_size, n)
        page_rows = sorted_rows[start:end]

        title = (f'Pool: {pool_name}  ({n} dets)  '
                 f'[{start+1}–{end}]  '
                 f'S=aceptar  D=dividir  R=mas  X=saltar')
        img   = _render_rows_block(page_rows, title,
                                   n_cols=page_size, header_color=(20, 50, 30))
        cv2.imshow(f'Pool: {pool_name}', img)
        cv2.waitKey(1)

        print(f'\n  Pool "{pool_name}"  ({n} dets)  [{start+1}–{end}]')
        print('  S=aceptar  D=dividir en sub-clusters  R=ver más  X=saltar')
        resp = input('  → ').strip().lower()
        cv2.destroyAllWindows()

        if resp == 's':
            while True:
                cls = input(f'  Clase para este pool {CLASES_VALIDAS}: ').strip()
                if cls in CLASES_VALIDAS and cls != 'skip':
                    return {cls: sorted_rows}
                if cls == 'skip':
                    return {}
                print(f'  [!] Opciones: {CLASES_VALIDAS}')

        elif resp == 'd':
            resp_k = input('  k sub-clusters? [2]: ').strip()
            k_sub  = int(resp_k) if resp_k.isdigit() and int(resp_k) >= 2 else 2

            sub_km   = KMeans(n_clusters=k_sub, random_state=42,
                              n_init=20, max_iter=300)
            sub_lbls = sub_km.fit_predict(np.sqrt(np.clip(sigs, 0, None)))
            show_cluster_gallery(pool_rows, sub_lbls, k_sub)

            result = {}
            for si in range(k_sub):
                sub_idxs = np.where(sub_lbls == si)[0]
                n_si     = len(sub_idxs)
                sub_sig  = sigs[sub_lbls == si].mean(axis=0)
                dom      = int(np.argmax(sub_sig[:_HUE_BINS]) * 180 / _HUE_BINS)
                wf       = sub_sig[_HUE_BINS]; df = sub_sig[_HUE_BINS + 1]

                img_si = _render_cluster_image(pool_rows, sub_lbls, si)
                cv2.imshow(f'Sub-{si}  (cierra con tecla)', img_si)
                cv2.waitKey(1)

                print(f'\n  Sub-{si} ({n_si} dets) '
                      f'hue_dom={dom}° W={wf:.2f} D={df:.2f}')
                while True:
                    cls = input('  → clase (skip=ignorar): ').strip()
                    if cls in CLASES_VALIDAS:
                        if cls != 'skip':
                            result[cls] = [pool_rows[j] for j in sub_idxs]
                        break
                    print(f'  [!] Opciones: {CLASES_VALIDAS}')
                cv2.destroyAllWindows()

            return result

        elif resp == 'r':
            page = 0 if end >= n else page + 1

        elif resp == 'x':
            return {}

    return {}


# ═══════════════════════════════════════════════════════════
#  6. AUTO-K SUGGESTION
# ═══════════════════════════════════════════════════════════

def suggest_k(X_sqrt, k_min=2, k_max=10, sample_size=800):
    n = len(X_sqrt)
    X_sub = X_sqrt[np.random.default_rng(42).choice(n, min(sample_size, n), replace=False)]
    scores = {}
    print(f'  Silhouette scores (muestra {len(X_sub)} dets):')
    for k in range(k_min, min(k_max + 1, len(X_sub))):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=200)
        lbl = km.fit_predict(X_sub)
        if len(np.unique(lbl)) < 2:
            continue
        s      = silhouette_score(X_sub, lbl, sample_size=min(300, len(X_sub)))
        scores[k] = s
        marker = '  ← mejor' if s == max(scores.values()) else ''
        print(f'    k={k:2d}: {s:.4f}{marker}')
    best_k = max(scores, key=scores.get) if scores else k_min
    return best_k, scores


# ═══════════════════════════════════════════════════════════
#  7. VALIDATION
# ═══════════════════════════════════════════════════════════

def validate_prototypes(rows, labeled_map, proto_dict):
    """
    Clasifica todas las rows etiquetadas y muestra distribución + errores.
    labeled_map: {row_idx: clase_verdadera}  (puede estar vacío)
    """
    print(f'\n{"="*60}')
    print(f'  VALIDACIÓN POST-BUILD')
    print(f'  Clasificando {len(rows)} detecciones con el PKL...')
    print(f'{"="*60}')

    # Distribución de clasificación sobre todas las rows
    dist: dict = Counter()
    for row in rows:
        sig    = np.array(row['hue_sig'], dtype=np.float32)
        scores = {cls: hue_sig_distance(sig, np.asarray(p['hue_sig']))
                  for cls, p in proto_dict.items() if 'hue_sig' in p}
        if not scores:
            dist['unknown'] += 1
            continue
        sorted_s        = sorted(scores.items(), key=lambda x: x[1])
        best_cls, best  = sorted_s[0]
        margin          = (sorted_s[1][1] - best) if len(sorted_s) > 1 else 0.0
        max_d           = proto_dict[best_cls].get('max_dist', CLASSIFY_V3_MAX_SCORE)
        if max_d and best > max_d:
            dist['unknown'] += 1
        elif margin < CLASSIFY_V3_MARGIN:
            dist['unknown'] += 1
        else:
            dist[best_cls] += 1

    print('\n  Distribución de clasificación:')
    total = sum(dist.values())
    for cls, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        bar = '█' * int(pct / 2)
        print(f'    {cls:15s}: {cnt:5d}  ({pct:5.1f}%)  {bar}')

    # Matriz de confusión si hay etiquetas
    if not labeled_map:
        return

    confusion: dict = defaultdict(lambda: defaultdict(int))
    correct = total_lbl = 0
    error_rows = []
    for i, row in enumerate(rows):
        true_cls = labeled_map.get(i)
        if not true_cls or true_cls == 'skip':
            continue
        sig    = np.array(row['hue_sig'], dtype=np.float32)
        scores = {cls: hue_sig_distance(sig, np.asarray(p['hue_sig']))
                  for cls, p in proto_dict.items() if 'hue_sig' in p}
        if not scores:
            pred = 'unknown'
        else:
            sorted_s    = sorted(scores.items(), key=lambda x: x[1])
            best_cls, b = sorted_s[0]
            margin      = (sorted_s[1][1] - b) if len(sorted_s) > 1 else 0.0
            max_d       = proto_dict[best_cls].get('max_dist', CLASSIFY_V3_MAX_SCORE)
            uncertain   = (max_d and b > max_d) or margin < CLASSIFY_V3_MARGIN
            pred        = 'unknown' if uncertain else best_cls
        confusion[true_cls][pred] += 1
        total_lbl += 1
        if pred == true_cls:
            correct += 1
        else:
            error_rows.append({'row': row, 'true': true_cls, 'pred': pred})

    if total_lbl == 0:
        return
    print(f'\n  Accuracy: {correct}/{total_lbl} = {correct/total_lbl*100:.1f}%')

    if not error_rows:
        print('  Sin errores')
        return

    resp = input(f'\n  Mostrar galería de {len(error_rows)} errores? [y/n]: ').strip().lower()
    if resp != 'y':
        return

    by_type: dict = defaultdict(list)
    for e in error_rows:
        by_type[(e['true'], e['pred'])].append(e['row'])

    for (true_c, pred_c), err_list in sorted(by_type.items(), key=lambda x: -len(x[1])):
        crops = [r['crop'] for r in err_list[:COLS_GALLERY]]
        while len(crops) < COLS_GALLERY:
            crops.append(np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8))
        hdr = np.full((40, CROP_W * COLS_GALLERY, 3), (60, 20, 20), dtype=np.uint8)
        cv2.putText(hdr,
                    f'ERROR: {true_c} → {pred_c}  ({len(err_list)} casos)',
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 180, 255), 1)
        cv2.imshow(f'Errores {true_c}→{pred_c}',
                   np.vstack([hdr, np.hstack(crops)]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════
#  8. HELPERS
# ═══════════════════════════════════════════════════════════

def _parse_time(s):
    s = s.strip()
    parts = s.split(':')
    if len(parts) == 1:   return float(parts[0])
    elif len(parts) == 2: return int(parts[0]) * 60 + float(parts[1])
    else:                 return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def _parse_intervals(raw, default_home_left=None):
    """
    Parsea una cadena de intervalos con formato START-END[:L|:R][,...]
    :L → home ataca hacia izquierda (home_left=True)
    :R → home ataca hacia derecha   (home_left=False)
    Sin sufijo → usa default_home_left
    Retorna lista de (start_s, end_s, home_left|None)
    """
    if not raw:
        return None
    intervals = []
    for part in raw.split(','):
        part = part.strip()
        home_left = default_home_left
        up = part.upper()
        if up.endswith(':L'):
            home_left = True
            part = part[:-2]
        elif up.endswith(':R'):
            home_left = False
            part = part[:-2]
        if '-' not in part:
            raise ValueError(f'Intervalo inválido: "{part}" (usa START-END[:L|:R])')
        idx   = part.rfind('-')
        start = _parse_time(part[:idx])
        end   = _parse_time(part[idx + 1:])
        if start >= end:
            raise ValueError(f'Intervalo inválido: start={start} >= end={end}')
        intervals.append((start, end, home_left))
    return intervals


# ═══════════════════════════════════════════════════════════
#  9. DETECCIÓN DE CAMBIO DE ILUMINACIÓN
# ═══════════════════════════════════════════════════════════

ILLUM_MIN_ROWS     = 50    # mínimo de rows por grupo para considerar split válido
ILLUM_STD_MIN      = 18.0  # desviación mínima de brillo para analizar
ILLUM_GAP_MIN      = 28.0  # diferencia mínima entre medias de los dos grupos
ILLUM_BRIGHT_THR   = 75.0  # umbral día/noche (igual que pipeline_core.BRIGHT_THRESH)


def detect_illumination_split(rows, bright_thresh=ILLUM_BRIGHT_THR):
    """
    Analiza los valores de brillo de los rows para detectar cambio de iluminación.

    Retorna (split, threshold):
      split=True  → hay dos condiciones lumínicas bien separadas
      threshold   → valor de brillo que separa los grupos
      split=False → un único PKL es suficiente
    """
    brightnesses = np.array([r.get('brightness', 128.0) for r in rows], dtype=np.float32)

    std = float(brightnesses.std())
    if std < ILLUM_STD_MIN:
        return False, bright_thresh

    group_day   = brightnesses[brightnesses >  bright_thresh]
    group_night = brightnesses[brightnesses <= bright_thresh]

    n_day   = len(group_day)
    n_night = len(group_night)

    if n_day < ILLUM_MIN_ROWS or n_night < ILLUM_MIN_ROWS:
        return False, bright_thresh

    gap = float(group_day.mean() - group_night.mean())
    if gap < ILLUM_GAP_MIN:
        return False, bright_thresh

    return True, bright_thresh


def plot_illumination_analysis(rows, illum_thresh, has_split,
                               out_png='illumination_analysis.png'):
    """
    Guarda la gráfica de diagnóstico de iluminación como PNG y la abre
    con el visor del sistema (funciona aunque matplotlib use backend Agg).
    """
    import matplotlib
    matplotlib.use('Agg')          # forzar Agg para guardar sin display
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    brightnesses = np.array([r.get('brightness', 128.0) for r in rows], dtype=np.float32)
    times_min    = np.array([r.get('t', 0.0)            for r in rows], dtype=np.float32) / 60.0
    is_day = brightnesses > illum_thresh

    fig = plt.figure(figsize=(13, 4))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.04)
    ax_t = fig.add_subplot(gs[0])
    ax_h = fig.add_subplot(gs[1], sharey=ax_t)

    ax_t.scatter(times_min[is_day],  brightnesses[is_day],  c='#F4A430', s=5, alpha=0.55,
                 linewidths=0, label=f'Día (>{illum_thresh:.0f}): {is_day.sum()} frames')
    ax_t.scatter(times_min[~is_day], brightnesses[~is_day], c='#3A6BD4', s=5, alpha=0.55,
                 linewidths=0, label=f'Noche (≤{illum_thresh:.0f}): {(~is_day).sum()} frames')
    ax_t.axhline(illum_thresh, color='crimson', lw=1.5, ls='--', label=f'umbral = {illum_thresh:.0f}')
    ax_t.set_xlabel('Tiempo (min)', fontsize=10)
    ax_t.set_ylabel('Brillo medio del fotograma', fontsize=10)
    ax_t.set_ylim(0, 255)

    if has_split:
        ax_t.set_title('Cambio de iluminación detectado  ->  2 PKLs (día / noche)',
                       fontsize=11, color='darkred', fontweight='bold')
    else:
        ax_t.set_title('Iluminación uniforme  →  un único PKL',
                       fontsize=11, color='darkgreen')
    ax_t.legend(fontsize=8, markerscale=2)

    bins = np.linspace(0, 255, 45)
    ax_h.hist(brightnesses[is_day],  bins=bins, orientation='horizontal',
              color='#F4A430', alpha=0.85, label='Día')
    ax_h.hist(brightnesses[~is_day], bins=bins, orientation='horizontal',
              color='#3A6BD4', alpha=0.85, label='Noche')
    ax_h.axhline(illum_thresh, color='crimson', lw=1.5, ls='--')
    ax_h.set_xlabel('# frames', fontsize=9)
    ax_h.tick_params(labelleft=False)
    ax_h.legend(fontsize=8)

    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Grafica guardada: {out_png}')

    # Abrir con el visor del sistema (Windows/Mac/Linux)
    import os, subprocess, sys as _sys
    try:
        if _sys.platform == 'win32':
            os.startfile(os.path.abspath(out_png))
        elif _sys.platform == 'darwin':
            subprocess.Popen(['open', out_png])
        else:
            subprocess.Popen(['xdg-open', out_png])
    except Exception as e:
        print(f'  (Ábrela manualmente: {os.path.abspath(out_png)}  —  {e})')


# ═══════════════════════════════════════════════════════════
#  10. MAIN
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Genera PKL v3 híbrido (clustering + pools posicionales)')
    ap.add_argument('--video',        required=True)
    ap.add_argument('--model',        required=True)
    ap.add_argument('--output',       default='prototypes_v3.pkl')
    ap.add_argument('--cache',        default='rows_cache.pkl')
    ap.add_argument('--no_cache',     action='store_true')
    ap.add_argument('--sample_sec',   type=int,   default=180)
    ap.add_argument('--step',         type=float, default=2.5)
    ap.add_argument('--sam',          default='sam2_t.pt')
    ap.add_argument('--k',            type=int,   default=None)
    ap.add_argument('--n_top',        type=int,   default=20,
                    help='Muestras top-N más centrales para prototipo principal')
    ap.add_argument('--conf',         type=float, default=0.45)
    ap.add_argument('--intervals',    default=None)
    ap.add_argument('--H_a',          default=None, help='Homografía Cam A (.npy)')
    ap.add_argument('--H_b',          default=None, help='Homografía Cam B (.npy)')
    ap.add_argument('--field_length', type=float, default=105.0)
    ap.add_argument('--field_width',  type=float, default=68.0)
    ap.add_argument('--home_left',    action='store_true', default=True,
                    help='El equipo local ataca hacia x<0 (izquierda). '
                         'Puede sobreescribirse por intervalo con sufijo :L/:R en --intervals')
    args = ap.parse_args()

    intervals = _parse_intervals(args.intervals, default_home_left=args.home_left)

    print(f'\n{"="*60}')
    print(f'  generar_prototipos_v3.py  (v4.0)')
    print(f'  video:  {args.video}')
    print(f'  output: {args.output}')
    if intervals:
        total_iv = sum(e - s for s, e, _ in intervals)
        istr_parts = []
        for s, e, hl in intervals:
            side = ':L' if hl else (':R' if hl is False else '')
            istr_parts.append(f'{s/60:.1f}–{e/60:.1f}min{side}')
        istr = '  |  '.join(istr_parts)
        print(f'  intervalos: {istr}  ({total_iv:.0f}s útiles)')
    else:
        print(f'  sample_sec: {args.sample_sec}s  step: {args.step}s')
    print(f'{"="*60}\n')

    # Pre-scan rápido (~5s, sin YOLO) para mostrar la gráfica de iluminación antes de muestrear
    def _quick_brightness_scan(video_path, intervals, default_hl, step_s=5.0, n_max=120):
        cap     = cv2.VideoCapture(video_path)
        fps_v   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_s = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps_v

        if intervals:
            ts = []
            for t_s, t_e, _ in intervals:
                ts.extend(np.arange(t_s, min(t_e, total_s - 1), step_s).tolist())
            ts = np.array(sorted(ts))
        else:
            ts = np.arange(0, total_s - 1, step_s)

        if len(ts) > n_max:
            idx = np.round(np.linspace(0, len(ts) - 1, n_max)).astype(int)
            ts  = ts[idx]

        scan_rows = []
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps_v))
            ret, frame = cap.read()
            if not ret:
                continue
            b  = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            hl = default_hl
            if intervals:
                for t_s, t_e, iv_hl in intervals:
                    if t_s <= t < t_e:
                        hl = iv_hl if iv_hl is not None else default_hl
                        break
            scan_rows.append({'t': t, 'brightness': b, 'home_left_local': hl})
        cap.release()
        return scan_rows

    print('Pre-scan de iluminación (sin YOLO)...')
    scan_rows = _quick_brightness_scan(
        args.video, intervals, args.home_left,
        step_s=max(args.step * 4, 5.0)
    )
    has_illum_split_pre, illum_thresh_pre = detect_illumination_split(scan_rows)
    if has_illum_split_pre:
        n_d = sum(1 for r in scan_rows if r['brightness'] > illum_thresh_pre)
        n_n = len(scan_rows) - n_d
        print(f'  Cambio de iluminación detectado -> DOS PKLs (_day / _night)')
        print(f'  Dia: {n_d} muestras  |  Noche: {n_n} muestras')
    else:
        print('  Iluminación uniforme -> un único PKL')
    plot_illumination_analysis(scan_rows, illum_thresh_pre, has_illum_split_pre)

    # ── FASE 1: Sample / cargar caché ─────────────────────────────────────────
    cache_path = Path(args.cache)
    rows       = None

    if not args.no_cache and cache_path.exists():
        print(f'Cache: {cache_path}  ({cache_path.stat().st_size // 1024} KB)')
        resp = input('  Cargar rows del cache? [y/n]: ').strip().lower()
        if resp == 'y':
            with open(cache_path, 'rb') as f:
                rows = pickle.load(f)
            n_sam = sum(1 for r in rows if r.get('used_sam'))
            print(f'  {len(rows)} rows  (SAM2: {n_sam}/{len(rows)})')

            # Rellenar home_left_local en cachés viejos que no tienen el campo
            if intervals and any('home_left_local' not in r for r in rows):
                n_fixed = 0
                for r in rows:
                    if 'home_left_local' not in r:
                        t  = r.get('t', 0.0)
                        hl = args.home_left
                        for t_s, t_e, iv_hl in intervals:
                            if t_s <= t < t_e:
                                hl = iv_hl if iv_hl is not None else args.home_left
                                break
                        r['home_left_local'] = hl
                        n_fixed += 1
                print(f'  home_left_local rellenado en {n_fixed} rows desde intervalos')

    if rows is None:
        print('Cargando YOLO...')
        model = YOLO(args.model)
        init_undistort_maps()

        sam_model = None
        sam_path  = (args.sam or '').strip()
        if sam_path:
            try:
                from ultralytics import SAM
                sam_model = SAM(sam_path)
                print(f'SAM2 listo ({sam_path})')
            except Exception as e:
                print(f'  SAM2 no disponible ({e})')
        else:
            print('  SAM2 deshabilitado')

        print('Muestreando video...')
        rows = sample_video(args.video, model, args.sample_sec, args.step, args.conf,
                            sam_model=sam_model, intervals=intervals,
                            default_home_left=args.home_left)

        if len(rows) < 10:
            print(f'[!] Solo {len(rows)} dets. Aumenta --sample_sec o baja --conf')
            sys.exit(1)

        with open(cache_path, 'wb') as f:
            pickle.dump(rows, f)
        print(f'  Cache guardado: {cache_path}  ({len(rows)} rows)')

    # ── FASE 1b: pos_m con homografías ────────────────────────────────────────
    H_a = H_b = None
    if args.H_a and Path(args.H_a).exists():
        H_a = np.load(args.H_a)
        print(f'  H_a cargada: {args.H_a}')
    if args.H_b and Path(args.H_b).exists():
        H_b = np.load(args.H_b)
        print(f'  H_b cargada: {args.H_b}')

    has_homography = H_a is not None and H_b is not None
    if has_homography:
        print('\nCalculando posiciones en campo...')
        compute_pos_m(rows, H_a, H_b)
    else:
        print('\n[!] Sin homografias — split posicional deshabilitado '
              '(usa --H_a / --H_b para habilitarlo).')

    has_illum_split, illum_thresh = detect_illumination_split(rows)
    if has_illum_split:
        n_day   = sum(1 for r in rows if r.get('brightness', 128) > illum_thresh)
        n_night = len(rows) - n_day
        print(f'\nIluminación: cambio confirmado (umbral={illum_thresh:.0f}):')
        print(f'  Dia   (brightness > {illum_thresh:.0f}): {n_day} rows')
        print(f'  Noche (brightness <= {illum_thresh:.0f}): {n_night} rows')
        print('  -> DOS PKLs (_day / _night).')
    else:
        print('\nIluminación uniforme confirmada -> un único PKL.')

    def _run_workflow(subset_rows, label, out_path):
        print(f'\n{"="*60}')
        print(f'  WORKFLOW: {label}  ({len(subset_rows)} rows)')
        print(f'{"="*60}')

        if len(subset_rows) < 10:
            print(f'  [!] Solo {len(subset_rows)} rows — PKL omitido para {label}.')
            return

        X    = np.array([r['hue_sig'] for r in subset_rows], dtype=np.float32)
        X_sq = np.sqrt(np.clip(X, 0, None))

        print('\nSelección de k...')
        best_k, _ = suggest_k(X_sq)

        if args.k is not None:
            default_k = args.k
            print(f'\n  --k={args.k} especificado en CLI.')
        else:
            default_k = best_k

        while True:
            resp_k = input(f'\n  k a usar? [{default_k}]: ').strip()
            if resp_k == '':
                K = default_k; break
            try:
                K = int(resp_k)
                if K >= 2: break
                print('  [!] k >= 2')
            except ValueError:
                print('  [!] Escribe un entero o Enter')

        print(f'\n  KMeans k={K}...')
        km     = KMeans(n_clusters=K, random_state=42, n_init=30, max_iter=500)
        labels = km.fit_predict(X_sq)

        sizes  = sorted([(ki, int((labels == ki).sum())) for ki in range(K)],
                        key=lambda x: -x[1])
        print('\n  Clusters por tamaño (↓):')
        for ki, n in sizes:
            sig = X[labels == ki].mean(axis=0)
            dom = int(np.argmax(sig[:_HUE_BINS]) * 180 / _HUE_BINS)
            print(f'    [{ki}] {n:4d} dets  hue_dom={dom}°  '
                  f'W={sig[_HUE_BINS]:.2f}  D={sig[_HUE_BINS+1]:.2f}')

        show_cluster_gallery(subset_rows, labels, K)

        resp_p = input('\nInspeccionar pureza de clusters? [y/n]: ').strip().lower()
        if resp_p == 'y':
            labels, K = inspect_cluster_purity(subset_rows, labels, K)
            sizes      = sorted([(ki, int((labels == ki).sum())) for ki in range(K)],
                                 key=lambda x: -x[1])
            if K > (args.k or best_k):
                show_cluster_gallery(subset_rows, labels, K)

        # FASE 3: Etiquetado clusters principales ─────────────────────────────
        print(f'\n{"="*60}')
        print('  FASE 3: ETIQUETADO CLUSTERS PRINCIPALES')
        print('  Los 2 más grandes suelen ser player_home / player_away.')
        print('  El resto → skip (se recuperan como outliers en Fase 4).')
        print(f'{"="*60}')

        assignments = {}
        for ki, n in sizes:
            img = _render_cluster_image(subset_rows, labels, ki)
            cv2.imshow(f'Cluster {ki}', img)
            cv2.waitKey(1)

            sig = X[labels == ki].mean(axis=0)
            dom = int(np.argmax(sig[:_HUE_BINS]) * 180 / _HUE_BINS)
            wf  = sig[_HUE_BINS]; df = sig[_HUE_BINS + 1]
            print(f'\n  Cluster {ki} ({n} dets) hue_dom={dom}° W={wf:.2f} D={df:.2f}')
            print(f'  Opciones: player_home, player_away, referee, skip')
            while True:
                cls = input('  → clase: ').strip()
                if cls in ('player_home', 'player_away', 'referee', 'skip'):
                    assignments[ki] = cls
                    break
                print('  [!] Opciones: player_home, player_away, referee, skip')
            cv2.destroyAllWindows()

        proto_dict  = {}
        labeled_map = {}

        for ki, cls in assignments.items():
            if cls == 'skip':
                continue
            cluster_rows = [subset_rows[i] for i in range(len(subset_rows)) if labels[i] == ki]
            for i in range(len(subset_rows)):
                if labels[i] == ki:
                    labeled_map[i] = cls
            proto = build_proto_top_n(cluster_rows, n_top=args.n_top)
            if proto:
                proto_dict[cls] = proto
                print(f'  {cls}: {proto["n_total"]} dets  '
                      f'top-{args.n_top} usados  max_dist={proto["max_dist"]:.3f}')

        if not any(c in proto_dict for c in ('player_home', 'player_away')):
            print('\n[!] Sin player_home ni player_away — comprueba el labeling.')

        # FASE 4: Pool de outliers → GKs + árbitro ────────────────────────────
        print(f'\n{"="*60}')
        print('  FASE 4: CLASES ESPECIALES  (GK izq / GK der / Árbitro)')
        print(f'{"="*60}')

        outlier_rows = []
        for r in subset_rows:
            sig        = np.array(r['hue_sig'], dtype=np.float32)
            is_outlier = True
            for cls in ('player_home', 'player_away'):
                if cls not in proto_dict:
                    continue
                d     = hue_sig_distance(sig, np.array(proto_dict[cls]['hue_sig']))
                max_d = proto_dict[cls].get('max_dist', CLASSIFY_V3_MAX_SCORE)
                if d < max_d:
                    is_outlier = False
                    break
            if is_outlier:
                outlier_rows.append(r)

        print(f'\n  Outliers (fuera de home/away): {len(outlier_rows)} / {len(subset_rows)} dets')

        if len(outlier_rows) < 4:
            print('  ✓ Muy pocos outliers — GKs y árbitro ya cubiertos o no detectados.')
            pools = []
        elif has_homography:
            field_l  = args.field_length
            field_a  = args.field_width
            pa_y_min = field_a * 0.10
            pa_y_max = field_a * 0.90

            # Asignación por row: usa home_left_local si está disponible (Opción B),
            # fallback a args.home_left para rows de caché sin el campo.
            pool_gk_home = []
            pool_gk_away = []
            for r in outlier_rows:
                pm = r.get('pos_m')
                if not pm or not (pa_y_min < pm[1] < pa_y_max):
                    continue
                hl = r.get('home_left_local', args.home_left)
                in_left  = pm[0] < PA_DEPTH_M
                in_right = pm[0] > field_l - PA_DEPTH_M
                if in_left:
                    (pool_gk_home if hl else pool_gk_away).append(r)
                elif in_right:
                    (pool_gk_away if hl else pool_gk_home).append(r)

            excl     = {id(r) for r in pool_gk_home + pool_gk_away}
            pool_ref = [r for r in outlier_rows if id(r) not in excl]

            print(f'\n  Pool gk_home: {len(pool_gk_home)} dets')
            print(f'  Pool gk_away: {len(pool_gk_away)} dets')
            print(f'  Pool zona central (arbitro): {len(pool_ref)} dets')

            pools = [
                (pool_gk_home, 'GK local     (sugerido: gk_home)'),
                (pool_gk_away, 'GK visitante (sugerido: gk_away)'),
                (pool_ref,     'Zona central (árbitro / otros)'),
            ]
        else:
            pools = [(outlier_rows, 'Outliers  (GK + Árbitro — sin split posicional)')]

        for pool_rows_i, pool_name in pools:
            result = interactive_pool(pool_rows_i, pool_name)
            for cls, cls_rows in result.items():
                n_use = min(args.n_top, len(cls_rows))
                proto = build_proto_top_n(cls_rows, n_top=n_use)
                if proto:
                    proto_dict[cls] = proto
                    row_id_to_idx = {id(r): i for i, r in enumerate(subset_rows)}
                    for r in cls_rows:
                        idx = row_id_to_idx.get(id(r), -1)
                        if idx >= 0:
                            labeled_map[idx] = cls
                    print(f'  {cls}: {proto["n_total"]} dets  top-{n_use} usados')

        # Resumen ─────────────────────────────────────────────────────────────
        print(f'\n{"="*60}')
        print(f'  RESUMEN PROTOTIPOS — {label}')
        print(f'{"="*60}')
        for cls, proto in proto_dict.items():
            dom = int(np.argmax(np.array(proto['hue_sig'])[:_HUE_BINS]) * 180 / _HUE_BINS)
            max_d = proto.get('max_dist')
            max_d_str = f'{max_d:.3f}' if max_d is not None else 'N/A'
            print(f'  {cls:15s}: {proto["n_samples"]:4d} muestras  '
                  f'hue_dom={dom}°  '
                  f'max_dist={max_d_str}')

        missing = {'player_home', 'player_away'} - set(proto_dict.keys())
        if missing:
            print(f'\n  ⚠️  Clases sin prototipo: {missing}')

        validate_prototypes(subset_rows, labeled_map, proto_dict)

        with open(out_path, 'wb') as f:
            pickle.dump(proto_dict, f)
        print(f'\nPKL guardado: {out_path}')
        print(f'   Clases: {list(proto_dict.keys())}')

    # ── Ejecutar workflow(s) ──────────────────────────────────────────────────
    if has_illum_split:
        stem = Path(args.output).stem
        ext  = Path(args.output).suffix or '.pkl'
        parent = Path(args.output).parent
        out_day   = str(parent / f'{stem}_day{ext}')
        out_night = str(parent / f'{stem}_night{ext}')

        rows_day   = [r for r in rows if r.get('brightness', 128) >  illum_thresh]
        rows_night = [r for r in rows if r.get('brightness', 128) <= illum_thresh]

        _run_workflow(rows_day,   'DÍA',   out_day)
        _run_workflow(rows_night, 'NOCHE', out_night)

        print(f'\nPKLs generados:')
        print(f'   Día:   {out_day}')
        print(f'   Noche: {out_night}')
        print(f'\n  Úsalos en pipeline con:')
        print(f'    --protos_day {out_day} --protos_night {out_night}')
    else:
        _run_workflow(rows, 'ÚNICO', args.output)

    print(f'\n  Prueba rápida:')
    print(f'    python generar_prototipos_v3.py --video {args.video} '
          f'--model {args.model} --output {args.output} --cache {args.cache}')


if __name__ == '__main__':
    main()
