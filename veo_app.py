"""
veo_app.py — VEO Tactical Pipeline — Streamlit App
Run: streamlit run veo_app.py
"""
import streamlit as st
import cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import threading, time, subprocess, pickle, io, os
from pathlib import Path
from ultralytics import YOLO
import torch

import pipeline_core as core

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
_HALF_PREC = _DEVICE == 'cuda'


def _hex_to_lab(hex_color: str) -> list:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]
    return [float(lab[0]), float(lab[1]), float(lab[2])]

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VEO Tactical",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
#  CSS — Football Analytics Dark Theme
# ══════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Variables ── */
:root {
    --bg:         #060d1a;
    --surface:    #0c1830;
    --surface-2:  #112040;
    --border:     #1a3060;
    --border-2:   #243a6a;
    --accent:     #00e676;
    --accent-dim: #00b85a;
    --blue:       #2979ff;
    --blue-dim:   #1a56cc;
    --text:       #e8edf8;
    --text-2:     #8a9bbf;
    --text-3:     #4a5a80;
    --success:    #00c853;
    --warn:       #ff9800;
    --error:      #f44336;
}

/* ── Base ── */
.stApp { background-color: var(--bg); color: var(--text); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #050d18 100%);
    border-right: 1px solid var(--border);
}

/* ── Typography ── */
h1, h2, h3 { color: var(--text) !important; letter-spacing: -0.3px; }
.stMarkdown p { color: var(--text-2); }

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 200px; height: 100%;
    background: linear-gradient(135deg, transparent 0%, rgba(0,230,118,0.04) 100%);
}
.page-header h2 { margin: 0 0 4px 0; font-size: 22px; font-weight: 700; }
.page-header p  { margin: 0; font-size: 14px; color: var(--text-2); }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
}
.card-accent {
    background: linear-gradient(135deg, #091a32 0%, #0d2245 100%);
    border: 1px solid #1a4080;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.card-green {
    background: linear-gradient(135deg, #071a10 0%, #0a2218 100%);
    border: 1px solid #1a5030;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.card-warn {
    background: linear-gradient(135deg, #1a1000 0%, #221500 100%);
    border: 1px solid #4a3000;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}

/* ── KPI cards ── */
.kpi-grid { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
.kpi-card {
    flex: 1; min-width: 120px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--blue));
    border-radius: 0 0 14px 14px;
}
.kpi-icon  { font-size: 22px; margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 800; color: var(--text); line-height: 1; }
.kpi-label { font-size: 12px; color: var(--text-2); margin-top: 4px; letter-spacing: 0.3px; }

/* ── Section title ── */
.section-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 0 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Step indicator (sidebar) ── */
.step-item {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 12px; border-radius: 10px;
    margin: 4px 0; cursor: default;
}
.step-item.done   { opacity: 0.7; }
.step-item.active { background: rgba(41,121,255,0.12); border: 1px solid rgba(41,121,255,0.25); }
.step-item.pending{ opacity: 0.4; }
.step-num {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; flex-shrink: 0;
}
.step-num.done    { background: var(--success); color: #000; }
.step-num.active  { background: var(--blue); color: #fff;
                    box-shadow: 0 0 14px rgba(41,121,255,0.45); }
.step-num.pending { background: #1a2540; color: var(--text-3); border: 1px solid #243060; }
.step-text { font-size: 13px; font-weight: 600; }
.step-text.done    { color: var(--success); }
.step-text.active  { color: var(--text); }
.step-text.pending { color: var(--text-3); }

/* ── Status dots ── */
.dot-ok  { display:inline-block; width:8px; height:8px; border-radius:50%;
           background:var(--success); margin-right:6px; flex-shrink:0; }
.dot-err { display:inline-block; width:8px; height:8px; border-radius:50%;
           background:var(--error); margin-right:6px; flex-shrink:0; }
.dot-off { display:inline-block; width:8px; height:8px; border-radius:50%;
           background:#2a3a55; margin-right:6px; flex-shrink:0; }

/* ── Badges ── */
.badge-home    { background:rgba(220,20,60,0.15); border:1px solid #dc143c; color:#ff6b88;
                 border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-away    { background:rgba(30,144,255,0.15); border:1px solid #1e90ff; color:#60b8ff;
                 border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-ref     { background:rgba(0,255,127,0.1); border:1px solid #00ff7f; color:#00ff7f;
                 border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-gk      { background:rgba(255,0,255,0.1); border:1px solid #ff00ff; color:#ff88ff;
                 border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-unknown { background:rgba(128,128,128,0.1); border:1px solid #555; color:#888;
                 border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }

/* ── Metric pill ── */
.metric-pill {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border-2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px; color: var(--text-2); margin: 3px;
}

/* ── GPU badge ── */
.gpu-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: linear-gradient(135deg, #071a10 0%, #0a2218 100%);
    border: 1px solid #1a5030;
    border-radius: 10px; padding: 8px 14px;
    font-size: 13px; font-weight: 600; color: var(--success);
}
.cpu-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: linear-gradient(135deg, #1a1000 0%, #221500 100%);
    border: 1px solid #4a3000;
    border-radius: 10px; padding: 8px 14px;
    font-size: 13px; font-weight: 600; color: var(--warn);
}

/* ── Timeline ── */
.timeline-wrap {
    background: #0c1830; border-radius: 8px; overflow: hidden;
    height: 22px; margin: 10px 0; position: relative;
    border: 1px solid var(--border);
}
.timeline-half1 { position:absolute; height:100%;
                  background:linear-gradient(90deg,#1a4aaa,var(--blue)); opacity:0.85; }
.timeline-half2 { position:absolute; height:100%;
                  background:linear-gradient(90deg,#8a1a4a,#c0395e); opacity:0.85; }

/* ── Progress bar ── */
.progress-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;
}
.progress-label { font-size: 13px; color: var(--text-2); margin-bottom: 8px; }
.progress-bar-bg {
    background: #1a2540; border-radius: 6px; height: 10px; overflow: hidden;
}
.progress-bar-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, var(--blue), var(--accent));
    transition: width 0.4s ease;
}

/* ── Input fields ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-2) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(41,121,255,0.2) !important;
}
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--blue), #4f8ef7) !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(41,121,255,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(41,121,255,0.4) !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] { background: transparent !important; gap: 6px; }
[data-baseweb="tab"] {
    background: var(--surface) !important;
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid var(--border) !important;
    border-bottom: none !important;
    color: var(--text-3) !important;
    font-weight: 500 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border-color: var(--border-2) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    padding: 14px 18px !important;
    font-weight: 600 !important;
}

/* ── Table ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
thead tr th { background: var(--surface-2) !important; color: var(--text-2) !important; }

/* ── Metric ── */
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 16px !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: var(--text-2) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 3px; }

/* ── Hero input (step 1) ── */
.hero-input .stTextInput > div > div > input {
    font-size: 16px !important;
    padding: 12px 16px !important;
    height: 52px !important;
}

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 60px 20px; color: var(--text-3);
}
.empty-state .icon { font-size: 52px; margin-bottom: 16px; opacity: 0.5; }
.empty-state h3 { color: var(--text-2) !important; font-size: 18px; margin-bottom: 8px; }
.empty-state p { font-size: 14px; color: var(--text-3); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════

st.session_state.setdefault('step', 1)
st.session_state.setdefault('vid_std', None)
st.session_state.setdefault('vid_pan', None)
st.session_state.setdefault('scrubber_frame', 0)
st.session_state.setdefault('mark_t1s', None)
st.session_state.setdefault('mark_t1e', None)
st.session_state.setdefault('mark_t2s', None)
st.session_state.setdefault('mark_t2e', None)
st.session_state.setdefault('H_a', None)
st.session_state.setdefault('H_b', None)
st.session_state.setdefault('protos_day', None)
st.session_state.setdefault('protos_night', None)
st.session_state.setdefault('model_A', None)
st.session_state.setdefault('model_B', None)
st.session_state.setdefault('model_ball', None)
st.session_state.setdefault('model_sam',  None)
st.session_state.setdefault('config', {
    'start_frame': 0,
    'n_frames': 9000,
    'skip_frames': 3,
    'conf_thresh': 0.35,
    'conf_ball': core.CONF_BALL,
    'limite_x': 49.0,
    'home_attacks_left': True,
    'halftime_frame': None,
    'match_intervals': None,
    'tracker_type':       'strongkalman',
    'bt_high_thresh':     0.38,
    'bt_match_thresh_m':  5.0,
    'bt_max_lost_s':      3.0,
    'bt_gallery_ttl_s':   20.0,
    'bt_reid_hue_thresh':  0.33,
    'track_max_dist_m':   core.TRACK_MAX_DIST_M,
    'track_window_s':     7.0,
    'track_class_w':      8.0,
    'v3_lost_s':          8.0,
    'track_gallery_ttl_s':  120.0,
    'track_reid_hue_thresh': 0.33,
    'device':             _DEVICE,
    'half_precision':     _HALF_PREC,
    'imgsz':              1280,
    'home_dot_hex':       '#e63946',
    'away_dot_hex':       '#1d70b8',
    'field_length':       core.L_M,
    'field_width':        core.A_M,
})
st.session_state.setdefault('progress', {})
st.session_state.setdefault('proc_thread', None)
st.session_state.setdefault('match_name', '')
st.session_state.setdefault('match_meta', {
    'competition': '', 'season': '',
    'home_team': '', 'away_team': '',
    'date': '', 'venue': '',
})

# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════

with st.sidebar:
    # Logo / brand
    st.markdown("""
    <div style='padding: 20px 8px 16px; text-align: center;'>
        <div style='font-size: 38px; line-height: 1;'>⚽</div>
        <div style='font-size: 17px; font-weight: 800; color: #e8edf8;
                    letter-spacing: -0.3px; margin-top: 8px;'>VEO Tactical</div>
        <div style='font-size: 10px; color: #2a3a5a; letter-spacing: 2px;
                    text-transform: uppercase; margin-top: 3px;'>Football Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Step navigation
    cur = st.session_state.step
    steps = [(1, "📥", "Descarga"),
             (2, "⚙️", "Calibración"),
             (3, "▶️", "Procesar"),
             (4, "📊", "Resultados")]

    for i, icon, name in steps:
        if i < cur:   state, sym = "done",    "✓"
        elif i == cur: state, sym = "active",  str(i)
        else:          state, sym = "pending", str(i)

        st.markdown(
            f'<div class="step-item {state}">'
            f'  <div class="step-num {state}">{sym}</div>'
            f'  <div class="step-text {state}">{icon} {name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if i < cur:
            if st.button(f"← Volver a {name}", key=f"nav_{i}",
                         width="stretch"):
                st.session_state.step = i
                st.rerun()

    st.divider()

    # Status panel
    st.markdown('<div style="font-size:11px;font-weight:700;color:#2a3a5a;'
                'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">'
                'Estado del sistema</div>', unsafe_allow_html=True)

    def _dot(ok): return '<span class="dot-ok"></span>' if ok else '<span class="dot-off"></span>'
    def _c(ok):   return '#5aff8a' if ok else '#2a3a55'

    pan_ok = bool(st.session_state.vid_pan and os.path.exists(st.session_state.vid_pan or ''))
    Ha_ok  = st.session_state.H_a is not None
    Hb_ok  = st.session_state.H_b is not None
    pkl_ok = st.session_state.protos_day is not None
    mod_ok = st.session_state.model_A is not None

    rows_html = (
        f'<div style="font-size:13px;line-height:2.2;">'
        f'{_dot(pan_ok)}<span style="color:{_c(pan_ok)};">Vídeo panorámico</span><br>'
        f'{_dot(Ha_ok and Hb_ok)}<span style="color:{_c(Ha_ok and Hb_ok)};">Homografías (A+B)</span><br>'
        f'{_dot(pkl_ok)}<span style="color:{_c(pkl_ok)};">PKL colores</span><br>'
        f'{_dot(mod_ok)}<span style="color:{_c(mod_ok)};">Modelos YOLO</span><br>'
        f'</div>'
    )
    st.markdown(rows_html, unsafe_allow_html=True)

    # GPU badge
    if _DEVICE == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        st.markdown(f'<div class="gpu-badge" style="margin-top:10px;font-size:12px;">'
                    f'🟢 {gpu_name[:22]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="cpu-badge" style="margin-top:10px;font-size:12px;">'
                    '🟡 CPU · sin GPU</div>', unsafe_allow_html=True)

    # Match meta quick view
    meta = st.session_state.match_meta
    if meta.get('home_team') or meta.get('competition'):
        st.divider()
        st.markdown(
            f'<div style="font-size:12px;color:#4a5a80;line-height:1.9;">'
            f'<b style="color:#6a7fa0;">{meta.get("home_team","?")} vs {meta.get("away_team","?")}</b><br>'
            f'{meta.get("competition","")} {meta.get("season","")}<br>'
            f'{meta.get("date","")} · {meta.get("venue","")}</div>',
            unsafe_allow_html=True,
        )

    # PKL badges
    if st.session_state.protos_day:
        st.divider()
        cls_list = list(st.session_state.protos_day.keys())
        badge_map_s = {
            'player_home': 'badge-home', 'player_away': 'badge-away',
            'referee': 'badge-ref', 'gk_home': 'badge-gk', 'gk_away': 'badge-gk',
        }
        html_b = '<div style="line-height:2.2;display:flex;flex-wrap:wrap;gap:4px;">'
        for c in cls_list:
            b = badge_map_s.get(c, 'badge-unknown')
            html_b += f'<span class="{b}" style="font-size:10px;">{c}</span>'
        html_b += '</div>'
        st.markdown(html_b, unsafe_allow_html=True)

    st.divider()

    # SAM2 loader
    st.markdown('<div style="font-size:11px;font-weight:700;color:#2a3a5a;'
                'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">'
                'SAM2 (segmentación)</div>', unsafe_allow_html=True)
    sam_path = st.text_input("Ruta modelo SAM2", value="sam2_t.pt",
                              key="sam_path_input", label_visibility="collapsed")
    sam_ok = st.session_state.model_sam is not None
    if not sam_ok and os.path.exists(sam_path):
        if st.button("Cargar SAM2", width="stretch"):
            with st.spinner("Cargando SAM2…"):
                try:
                    from ultralytics import SAM
                    st.session_state.model_sam = SAM(sam_path)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    elif sam_ok:
        st.markdown('<span class="dot-ok"></span>'
                    '<span style="color:#5aff8a;font-size:12px;">SAM2 listo</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="dot-off"></span>'
                    '<span style="color:#2a3a55;font-size:12px;">No cargado</span>',
                    unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;color:#1a2540;text-align:center;margin-top:20px;">'
                'v3 · StrongKalman + SAM2</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def _video_thumbnail(path, frame=100):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, f = cap.read(); cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ret else None


def _status_row(label, ok, detail=""):
    dot = '<span class="dot-ok"></span>' if ok else '<span class="dot-off"></span>'
    col = "#5aff8a" if ok else "#2a3a55"
    return (f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;">'
            f'{dot}<span style="color:{col};font-weight:600;font-size:13px;">{label}</span>'
            f'<span style="color:#2a3a55;font-size:12px;margin-left:auto;">{detail}</span></div>')


def _proto_swatches(protos):
    if not protos: return ""
    is_v3 = any('hue_sig' in p for p in protos.values())
    badge_map = {
        'player_home': ('#dc143c', 'Local'),
        'player_away': ('#1e90ff', 'Visitante'),
        'referee':     ('#00ff7f', 'Árbitro'),
        'gk_home':     ('#ff00ff', 'Port. Local'),
        'gk_away':     ('#00ced1', 'Port. Visit.'),
    }
    v3_tag = ('<span style="background:#071a10;border:1px solid #22c55e;color:#22c55e;'
              'font-size:10px;padding:1px 6px;border-radius:4px;margin-left:6px;">v3 hue</span>'
              if is_v3 else '')
    html = f'<div style="margin-bottom:6px;">{v3_tag}</div>'
    html += '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:6px;">'
    for cls, proto in protos.items():
        color, label = badge_map.get(cls, ('#888888', cls))
        n   = proto.get('n_samples', proto.get('n', '?'))
        med = proto.get('median')
        swatch = f'rgb({med[2]},{med[1]},{med[0]})' if med is not None else color
        html += (f'<div style="text-align:center;min-width:72px;">'
                 f'<div style="width:64px;height:28px;border-radius:8px;'
                 f'background:{swatch};border:2px solid {color};margin:0 auto 5px;'
                 f'box-shadow:0 0 8px {color}44;"></div>'
                 f'<div style="font-size:11px;color:{color};font-weight:700;">{label}</div>'
                 f'<div style="font-size:10px;color:#4a5a80;">{n} muestras</div></div>')
    html += '</div>'
    return html


def _section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


_AI_SECTION_META = {
    'resumen':              {'icon': '⚽', 'label': 'Resumen del partido',    'color': '#00e676'},
    'métricas destacadas':  {'icon': '📊', 'label': 'Métricas destacadas',    'color': '#FFD600'},
    'metricas destacadas':  {'icon': '📊', 'label': 'Métricas destacadas',    'color': '#FFD600'},
    'recomendaciones':      {'icon': '🎯', 'label': 'Recomendaciones tácticas','color': '#FF6F00'},
    'conclusión':           {'icon': '🏁', 'label': 'Conclusión',             'color': '#B0BEC5'},
    'conclusion':           {'icon': '🏁', 'label': 'Conclusión',             'color': '#B0BEC5'},
}


def _ai_render_report(text: str, home_team: str, away_team: str,
                      home_color: str = '#1565C0', away_color: str = '#B71C1C'):
    """Render the AI markdown report as visual section cards."""
    import re

    # Split on ## headings
    parts = re.split(r'\n(?=## )', text.strip())
    if not parts[0].startswith('## '):
        parts = parts[1:]  # discard any text before first heading

    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#00e676;'
        'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">'
        '✅ Informe táctico — Análisis IA</div>',
        unsafe_allow_html=True,
    )

    for part in parts:
        lines = part.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].lstrip('#').strip()
        body    = '\n'.join(lines[1:]).strip()

        # Determine styling
        key = heading.lower().split(' — ')[0].strip()
        if home_team.lower() in key:
            meta = {'icon': '🔵', 'label': heading, 'color': home_color}
        elif away_team.lower() in key:
            meta = {'icon': '🔴', 'label': heading, 'color': away_color}
        else:
            meta = _AI_SECTION_META.get(key, {'icon': '📋', 'label': heading, 'color': '#78909C'})

        border_color = meta['color']
        icon         = meta['icon']
        label        = meta['label']

        st.markdown(
            f'<div style="border-left:4px solid {border_color};padding:14px 18px;'
            f'background:#1a1a2e;border-radius:6px;margin-bottom:12px;">'
            f'<div style="font-size:13px;font-weight:700;color:{border_color};'
            f'margin-bottom:8px;">{icon} {label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(body)
        st.markdown('</div>', unsafe_allow_html=True)


def _fmt_min(frame, fps):
    if frame is None:
        return "—"
    s = frame / fps
    return f"{int(s//60):02d}:{s%60:05.2f}  (frame {frame})"


@st.cache_data(show_spinner=False)
def _get_video_frame(vid_path: str, frame_idx: int) -> bytes:
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, img = cap.read()
    cap.release()
    if not ret:
        return None
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


def _video_scrubber(vid_path: str, fps: float, total_frames: int):
    """Scrubber de vídeo inline para marcar tiempos del partido."""
    st.markdown(
        '<div style="background:#0c1830;border:1px solid #1a3060;border-radius:8px;'
        'padding:16px;margin-bottom:12px;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:13px;color:#8a9bbf;margin-bottom:10px;">'
        '🎬 Navega por el vídeo y marca el inicio/fin de cada tiempo</div>',
        unsafe_allow_html=True,
    )

    cur = st.session_state.scrubber_frame
    cur = max(0, min(cur, total_frames - 1))

    # ── Navegación por botones ──
    jump = int(fps)
    jump10 = int(fps * 10)
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 2, 1, 1, 1])
    if c1.button("⏮ -10s", key="scr_m10"):
        cur = max(0, cur - jump10)
    if c2.button("◀ -1s",  key="scr_m1"):
        cur = max(0, cur - jump)
    if c3.button("◀ -1f",  key="scr_mf"):
        cur = max(0, cur - 1)
    c4.markdown(
        f'<div style="text-align:center;font-size:13px;color:#00e676;padding-top:6px;">'
        f'{_fmt_min(cur, fps)}</div>',
        unsafe_allow_html=True,
    )
    if c5.button("+1f ▶",  key="scr_pf"):
        cur = min(total_frames - 1, cur + 1)
    if c6.button("+1s ▶",  key="scr_p1"):
        cur = min(total_frames - 1, cur + jump)
    if c7.button("+10s ▶", key="scr_p10"):
        cur = min(total_frames - 1, cur + jump10)

    # ── Slider ──
    cur = st.slider("Frame", 0, total_frames - 1, cur, key="scr_slider",
                    label_visibility="collapsed")
    st.session_state.scrubber_frame = cur

    # ── Frame actual ──
    img_bytes = _get_video_frame(vid_path, cur)
    if img_bytes:
        st.image(img_bytes, width="stretch")

    # ── Botones de marca ──
    st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns([2, 2, 2, 2, 1])
    if m1.button("📌 Inicio 1T", key="mk_t1s", width="stretch"):
        st.session_state.mark_t1s = cur
        st.toast(f"Inicio 1T marcado: {_fmt_min(cur, fps)}")
    if m2.button("🏁 Fin 1T",    key="mk_t1e", width="stretch"):
        st.session_state.mark_t1e = cur
        st.toast(f"Fin 1T marcado: {_fmt_min(cur, fps)}")
    if m3.button("📌 Inicio 2T", key="mk_t2s", width="stretch"):
        st.session_state.mark_t2s = cur
        st.toast(f"Inicio 2T marcado: {_fmt_min(cur, fps)}")
    if m4.button("🏁 Fin 2T",    key="mk_t2e", width="stretch"):
        st.session_state.mark_t2e = cur
        st.toast(f"Fin 2T marcado: {_fmt_min(cur, fps)}")
    if m5.button("🗑 Reset", key="mk_reset", width="stretch"):
        st.session_state.mark_t1s = None
        st.session_state.mark_t1e = None
        st.session_state.mark_t2s = None
        st.session_state.mark_t2e = None

    # ── Resumen de marcas ──
    marks = [
        ("🔵 Inicio 1T", st.session_state.mark_t1s),
        ("🔵 Fin 1T",    st.session_state.mark_t1e),
        ("🔴 Inicio 2T", st.session_state.mark_t2s),
        ("🔴 Fin 2T",    st.session_state.mark_t2e),
    ]
    rows = "".join(
        f'<span style="margin-right:20px;color:{"#2979ff" if "1T" in l else "#c0395e"};">'
        f'{l}: <b style="color:#e8edf8;">{_fmt_min(v, fps)}</b></span>'
        for l, v in marks
    )
    st.markdown(
        f'<div style="font-size:12px;margin-top:8px;padding:8px;'
        f'background:#060d1a;border-radius:6px;">{rows}</div>',
        unsafe_allow_html=True,
    )

    # ── Guardar JSON ──
    if st.button("💾 Guardar tiempos_partido.json", key="scr_save"):
        def _to_min(f): return round(f / fps / 60, 3) if f is not None else None
        import json
        data = {
            'fps': round(fps, 4),
            't1_inicio_frame': st.session_state.mark_t1s,
            't1_fin_frame':    st.session_state.mark_t1e,
            't2_inicio_frame': st.session_state.mark_t2s,
            't2_fin_frame':    st.session_state.mark_t2e,
            't1_inicio_min':   _to_min(st.session_state.mark_t1s),
            't1_fin_min':      _to_min(st.session_state.mark_t1e),
            't2_inicio_min':   _to_min(st.session_state.mark_t2s),
            't2_fin_min':      _to_min(st.session_state.mark_t2e),
        }
        with open('tiempos_partido.json', 'w') as f:
            json.dump(data, f, indent=2)
        st.success("Guardado en tiempos_partido.json")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  STEP 1 — DOWNLOAD
# ══════════════════════════════════════════════════════════

def step_download():
    st.markdown(
        '<div class="page-header">'
        '<h2>📥 Descarga del partido</h2>'
        '<p>Introduce la URL de VEO o carga los archivos ya descargados en tu equipo.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # ── Auto download ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        _section("🔗 Descarga automática desde VEO")

        with st.container():
            st.markdown('<div class="hero-input">', unsafe_allow_html=True)
            url = st.text_input(
                "URL del partido VEO",
                placeholder="https://app.veo.co/matches/…",
                key="veo_url",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        col_name, col_dir = st.columns(2)
        match_name = col_name.text_input("Nombre del partido",
                                          placeholder="banyoles-mollet-20260228",
                                          key="match_name_input")
        out_dir = col_dir.text_input("Directorio de salida",
                                      value="data/videos", key="out_dir_input")

        if st.button("⬇️  Descargar ambos vídeos", type="primary",
                     disabled=not bool(url), width="stretch"):
            path_std = os.path.join(out_dir, f"{match_name}_std.mp4")
            path_pan = os.path.join(out_dir, f"{match_name}_pan.mp4")
            os.makedirs(out_dir, exist_ok=True)
            prog = st.progress(0)
            log  = st.empty()
            success = True
            for i, (fmt, out_path, label) in enumerate([
                ("standard-1080p", path_std, "Estándar 1080p"),
                ("panorama-2048p", path_pan, "Panorámico 2048p"),
            ]):
                prog.progress(i * 0.5, text=f"Descargando {label}…")
                log.caption(f"`yt-dlp -f {fmt} -o {out_path} <url>`")
                result = subprocess.run(["yt-dlp", "-f", fmt, "-o", out_path, url],
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    st.error(f"Error en {label}:\n```\n{result.stderr[-500:]}\n```")
                    success = False; break
            if success:
                prog.progress(1.0, text="✅ Descarga completa")
                st.session_state.vid_std = path_std
                st.session_state.vid_pan = path_pan
                st.session_state.match_name = match_name
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Manual load ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        _section("📂 Cargar archivos existentes")
        col_a, col_b = st.columns(2)
        manual_std = col_a.text_input("Vídeo estándar (.mp4)", key="manual_std",
                                       placeholder="data/videos/partido_std.mp4")
        manual_pan = col_b.text_input("Vídeo panorámico (.mp4)", key="manual_pan",
                                       placeholder="data/videos/partido_pan.mp4")
        if manual_std and os.path.exists(manual_std):
            st.session_state.vid_std = manual_std
        if manual_pan and os.path.exists(manual_pan):
            st.session_state.vid_pan = manual_pan
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # ── Status ──
        std_ok = bool(st.session_state.vid_std and os.path.exists(st.session_state.vid_std or ''))
        pan_ok = bool(st.session_state.vid_pan and os.path.exists(st.session_state.vid_pan or ''))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        _section("Estado de carga")
        st.markdown(
            _status_row("Vídeo estándar",   std_ok, Path(st.session_state.vid_std).name if std_ok else "—") +
            _status_row("Vídeo panorámico", pan_ok, Path(st.session_state.vid_pan).name if pan_ok else "—"),
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if std_ok and pan_ok:
            for path, label, color in [
                (st.session_state.vid_std, "Estándar",   "#2979ff"),
                (st.session_state.vid_pan, "Panorámico", "#00e676"),
            ]:
                thumb = _video_thumbnail(path, 300)
                info  = core.get_video_info(path)
                if thumb is not None:
                    st.image(thumb, width="stretch",
                             caption=f"{label} · {info['width']}×{info['height']} · {info['duration_s']/60:.1f} min")

    if st.session_state.vid_std and st.session_state.vid_pan:
        st.divider()
        col_next, _ = st.columns([2, 5])
        if col_next.button("Siguiente → Calibración  ⚙️", type="primary",
                           width="stretch"):
            st.session_state.step = 2
            st.rerun()


# ══════════════════════════════════════════════════════════
#  STEP 2 — SETUP / CALIBRATION
# ══════════════════════════════════════════════════════════

def step_setup():
    st.markdown(
        '<div class="page-header">'
        '<h2>⚙️ Calibración</h2>'
        '<p>Carga las homografías de campo, los prototipos de color y configura el partido.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_calib, tab_partido = st.tabs(["📐 Calibración & Colores", "🏆 Partido & Tracker"])

    # ════════════════════════════════════════════════════════
    #  TAB 1 — Calibración & Colores
    # ════════════════════════════════════════════════════════
    with tab_calib:
        col_ha, col_hb, col_stitch = st.columns([2, 2, 2], gap="medium")

        def _load_homography(label, key_name, default_path, col):
            with col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                _section(label)
                H_state = st.session_state.get(key_name)

                if H_state is None and os.path.exists(default_path):
                    st.session_state[key_name] = np.load(default_path)
                    H_state = st.session_state[key_name]
                    st.markdown(f'<div class="card-green"><span class="dot-ok"></span>'
                                f'<small style="color:#5aff8a;">Auto: <code style="color:#4adf7a;">{default_path}</code></small></div>',
                                unsafe_allow_html=True)

                uploaded = st.file_uploader(f"Subir {key_name} (.npy)", type=['npy'],
                                             key=f'upload_{key_name}')
                if uploaded:
                    st.session_state[key_name] = np.load(io.BytesIO(uploaded.read()))
                    H_state = st.session_state[key_name]

                if H_state is not None:
                    det = np.linalg.det(H_state)
                    st.markdown(
                        f'<div style="display:flex;gap:20px;margin-top:10px;">'
                        f'<div><div style="font-size:10px;color:#4a5a80;text-transform:uppercase;'
                        f'letter-spacing:0.5px;">det(H)</div>'
                        f'<div style="font-size:16px;font-weight:700;color:#60b8ff;">{det:.4f}</div></div>'
                        f'<div><div style="font-size:10px;color:#4a5a80;text-transform:uppercase;'
                        f'letter-spacing:0.5px;">estado</div>'
                        f'<div style="font-size:13px;color:#5aff8a;font-weight:600;">✓ Cargada</div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<div style="color:#ff7755;font-size:13px;margin-top:8px;">⚠ No cargada</div>',
                                unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        _load_homography("Cam A — mitad superior", "H_a",
                         "data/calib_alpha1/calib_camA_alpha1_homography.npy", col_ha)
        _load_homography("Cam B — mitad inferior", "H_b",
                         "data/calib_alpha1/calib_camB_alpha1_homography.npy", col_hb)

        with col_stitch:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            _section("Costura automática")
            H_a, H_b = st.session_state.H_a, st.session_state.H_b
            if H_a is not None and H_b is not None and st.session_state.vid_pan:
                info_lx = core.get_video_info(st.session_state.vid_pan)
                lx = core.compute_limite_x(H_a, H_b, info_lx['width'], info_lx['height'] // 2)
                lx = float(np.clip(lx, 20.0, 80.0))
                st.markdown(
                    f'<div style="text-align:center;padding:14px 0;">'
                    f'<div style="font-size:38px;font-weight:800;color:#2979ff;">{lx:.1f}</div>'
                    f'<div style="font-size:11px;color:#4a5a80;margin-top:2px;">metros desde línea izquierda</div>'
                    f'<div style="font-size:11px;color:#5aff8a;margin-top:6px;">✓ Calculado automáticamente</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.config['limite_x'] = round(lx, 1)
            else:
                st.markdown('<div style="color:#2a3a55;font-size:13px;padding:10px 0;">'
                            'Carga ambas homografías y el vídeo primero.</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # ── PKL + Preview ──────────────────────────────────
        col_pkl, col_prev = st.columns([3, 3], gap="medium")

        with col_pkl:
            _section("🎨 Prototipos de color (PKL)")
            col_day, col_night = st.columns(2)

            def _load_pkl(label, key_name, defaults, col):
                with col:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    _section(label)
                    proto = st.session_state.get(key_name)

                    if proto is None:
                        for d in defaults:
                            if os.path.exists(d):
                                with open(d, 'rb') as pf:
                                    st.session_state[key_name] = pickle.load(pf)
                                proto = st.session_state[key_name]
                                st.markdown(f'<small style="color:#5aff8a;">Auto: <code>{d}</code></small>',
                                            unsafe_allow_html=True)
                                break

                    uploaded = st.file_uploader(f"Subir {label}", type=['pkl'],
                                                key=f'pkl_{key_name}')
                    if uploaded:
                        st.session_state[key_name] = pickle.load(uploaded)
                        proto = st.session_state[key_name]

                    if proto:
                        st.markdown(f'<div style="color:#5aff8a;font-size:13px;margin:6px 0;">'
                                    f'✓ {len(proto)} clases</div>', unsafe_allow_html=True)
                        st.markdown(_proto_swatches(proto), unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="color:#ff7755;font-size:13px;margin-top:8px;">'
                                    '⚠ No cargado</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            _load_pkl("PKL Día", "protos_day",
                      ["prototypes_v3_day.pkl", "prototypes_v3.pkl", "prototypes_banyoles.pkl"], col_day)
            _load_pkl("PKL Noche", "protos_night",
                      ["prototypes_v3_night.pkl", "prototypes_banyoles_night.pkl"], col_night)

        with col_prev:
            _section("🔍 Preview de campo")
            frame_sec = st.slider("Frame de referencia (segundos)", 10, 300, 60, 10,
                                   key="preview_sec")
            if st.button("Generar preview", key="btn_preview"):
                if st.session_state.vid_pan:
                    cap_v = cv2.VideoCapture(st.session_state.vid_pan)
                    fps_v = cap_v.get(cv2.CAP_PROP_FPS) or 30.0
                    cap_v.set(cv2.CAP_PROP_POS_FRAMES, int(frame_sec * fps_v))
                    ret_v, frm_v = cap_v.read(); cap_v.release()
                    if ret_v:
                        core.init_undistort_maps()
                        lx = st.session_state.config.get('limite_x', 49.0)
                        ca = core.preprocess_half(frm_v[:core.HALF_H, :], 'left')
                        cb = core.preprocess_half(frm_v[core.HALF_H:, :], 'right')
                        ca_r, cb_r = st.columns(2)
                        ca_r.image(cv2.cvtColor(ca, cv2.COLOR_BGR2RGB),
                                   caption=f"Cam A · costura {lx:.1f}m →",
                                   width="stretch")
                        cb_r.image(cv2.cvtColor(cb, cv2.COLOR_BGR2RGB),
                                   caption=f"← {lx:.1f}m · Cam B",
                                   width="stretch")
                else:
                    st.warning("Carga el vídeo panorámico primero.")

        # ── Exportar frames para calibrador ───────────────
        st.divider()
        with st.expander("📷 Exportar frames para el calibrador de homografías"):
            st.markdown('<div style="font-size:13px;color:#8a9bbf;margin-bottom:12px;">'
                        'Exporta los frames a la resolución exacta del pipeline para calibrar '
                        'sin error de escala.</div>', unsafe_allow_html=True)
            fl = float(st.session_state.config.get('field_length', core.L_M))
            fw = float(st.session_state.config.get('field_width',  core.A_M))
            st.info(f"Campo: {fl:.1f} × {fw:.1f} m — cámbialo en la pestaña 🏆 Partido")

            col_fs, col_bt = st.columns([3, 1])
            with col_fs:
                frame_sec_c = st.slider("Frame de referencia (s)", 10, 600, 120, 10,
                                        key="calib_dl_sec")
            with col_bt:
                st.markdown("<div style='margin-top:28px;'>", unsafe_allow_html=True)
                gen_calib = st.button("📷 Generar", type="primary",
                                      width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)

            if gen_calib:
                if not st.session_state.vid_pan:
                    st.warning("Carga el vídeo panorámico primero.")
                else:
                    cap_c = cv2.VideoCapture(st.session_state.vid_pan)
                    fps_c = cap_c.get(cv2.CAP_PROP_FPS) or 30.0
                    cap_c.set(cv2.CAP_PROP_POS_FRAMES, int(frame_sec_c * fps_c))
                    ret_c, frm_c = cap_c.read(); cap_c.release()
                    if ret_c:
                        core.init_undistort_maps()
                        ca_c = core.preprocess_half(frm_c[:core.HALF_H, :], 'left')
                        cb_c = core.preprocess_half(frm_c[core.HALF_H:, :], 'right')
                        h_px, w_px = ca_c.shape[:2]
                        match_name = st.session_state.get('match_name') or 'match'
                        _, buf_a = cv2.imencode('.png', ca_c)
                        _, buf_b = cv2.imencode('.png', cb_c)
                        col_da, col_db = st.columns(2)
                        with col_da:
                            st.image(cv2.cvtColor(ca_c, cv2.COLOR_BGR2RGB),
                                     caption=f"Cam A  ({w_px}×{h_px}px)",
                                     width="stretch")
                            st.download_button("⬇️ camA.png", data=buf_a.tobytes(),
                                               file_name=f"camA-{match_name}.png",
                                               mime="image/png", width="stretch")
                        with col_db:
                            st.image(cv2.cvtColor(cb_c, cv2.COLOR_BGR2RGB),
                                     caption=f"Cam B  ({w_px}×{h_px}px)",
                                     width="stretch")
                            st.download_button("⬇️ camB.png", data=buf_b.tobytes(),
                                               file_name=f"camB-{match_name}.png",
                                               mime="image/png", width="stretch")
                        st.success(f"✅ Imágenes a {w_px}×{h_px}px")
                        st.code(
                            f"python veo_calibrador.py camA-{match_name}.png "
                            f"--longitud {fl:.1f} --anchura {fw:.1f}\n"
                            f"python veo_calibrador.py camB-{match_name}.png "
                            f"--longitud {fl:.1f} --anchura {fw:.1f}",
                            language="bash",
                        )

    # ════════════════════════════════════════════════════════
    #  TAB 2 — Partido & Tracker
    # ════════════════════════════════════════════════════════
    with tab_partido:
        col_m1, col_m2 = st.columns(2, gap="large")

        with col_m1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            _section("🏆 Identificación del partido")
            meta = st.session_state.match_meta
            meta['home_team']   = st.text_input("Equipo local",    value=meta.get('home_team',''),   key='meta_home')
            meta['away_team']   = st.text_input("Equipo visitante",value=meta.get('away_team',''),   key='meta_away')
            meta['competition'] = st.text_input("Competición",     value=meta.get('competition',''), key='meta_comp')
            meta['season']      = st.text_input("Temporada",       value=meta.get('season',''),      key='meta_season')
            meta['date']        = st.text_input("Fecha (AAAA-MM-DD)", value=meta.get('date',''),     key='meta_date')
            meta['venue']       = st.text_input("Estadio / Campo", value=meta.get('venue',''),       key='meta_venue')
            st.session_state.match_meta = meta
            st.markdown('</div>', unsafe_allow_html=True)

        with col_m2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            _section("📐 Dimensiones del campo")
            st.markdown('<div style="color:#4a5a80;font-size:12px;margin-bottom:10px;">'
                        'FIFA: largo 90–120m, ancho 45–90m. Competición: 100–110m × 64–75m.</div>',
                        unsafe_allow_html=True)
            fl = st.number_input("Largo (línea de banda, metros)", 90.0, 120.0,
                                  value=float(st.session_state.config.get('field_length', core.L_M)),
                                  step=0.5, format="%.1f", key='field_length_meta')
            fw = st.number_input("Ancho (línea de fondo, metros)", 45.0, 90.0,
                                  value=float(st.session_state.config.get('field_width', core.A_M)),
                                  step=0.5, format="%.1f", key='field_width_meta')
            st.session_state.config['field_length'] = fl
            st.session_state.config['field_width']  = fw
            st.markdown(
                f'<div class="card-green" style="margin-top:10px;">'
                f'<div style="font-size:11px;color:#00e676;font-weight:700;margin-bottom:4px;">Campo configurado</div>'
                f'<div style="color:#8a9bbf;font-size:13px;">Largo: <b style="color:#e8edf8;">{fl:.1f} m</b> · '
                f'Ancho: <b style="color:#e8edf8;">{fw:.1f} m</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Tracker ──
            st.markdown('<div class="card">', unsafe_allow_html=True)
            _section("🎯 Algoritmo de tracking")
            _tracker_opts   = ['strongkalman', 'bytetrack', 'kalman']
            _tracker_labels = {
                'strongkalman': 'StrongKalman V3 — recomendado',
                'bytetrack':    'ByteTrack + Re-ID Gallery',
                'kalman':       'KalmanField (legacy)',
            }
            _cur_tracker = st.session_state.config.get('tracker_type', 'strongkalman')
            tracker_type = st.selectbox(
                "Algoritmo",
                options=_tracker_opts,
                index=_tracker_opts.index(_cur_tracker) if _cur_tracker in _tracker_opts else 0,
                format_func=lambda x: _tracker_labels.get(x, x),
                key='tracker_type_sel',
            )
            st.session_state.config['tracker_type'] = tracker_type

            if tracker_type == 'bytetrack':
                st.session_state.config['bt_high_thresh'] = st.slider(
                    "Umbral confianza alta", 0.30, 0.70,
                    value=float(st.session_state.config.get('bt_high_thresh', 0.38)), step=0.01,
                    help="Detecciones con conf ≥ este valor actualizan tracks activos directamente.",
                    key='bt_ht_slider')
                st.session_state.config['bt_match_thresh_m'] = st.slider(
                    "Distancia máx. asociación (m)", 2.0, 12.0,
                    value=float(st.session_state.config.get('bt_match_thresh_m', 5.0)), step=0.5,
                    key='bt_mt_slider')
                st.session_state.config['bt_max_lost_s'] = st.slider(
                    "Tiempo Lost antes de gallery (s)", 0.5, 10.0,
                    value=float(st.session_state.config.get('bt_max_lost_s', 3.0)), step=0.5,
                    key='bt_ml_slider')
                st.session_state.config['bt_gallery_ttl_s'] = st.slider(
                    "TTL gallery Re-ID (s)", 5.0, 60.0,
                    value=float(st.session_state.config.get('bt_gallery_ttl_s', 20.0)), step=5.0,
                    key='bt_gttl_slider')
                st.session_state.config['bt_reid_hue_thresh'] = st.slider(
                    "Umbral Hellinger Re-ID", 0.15, 0.50,
                    value=float(st.session_state.config.get('bt_reid_hue_thresh', 0.33)), step=0.01,
                    key='bt_rht_slider')
            elif tracker_type == 'strongkalman':
                st.session_state.config['track_max_dist_m'] = st.slider(
                    "Distancia máx. asociación (m)", 2.0, 12.0,
                    value=float(st.session_state.config.get('track_max_dist_m', 8.0)), step=0.5,
                    help="Radio máximo en metros para asociar una detección a un track existente.",
                    key='sk_dist_slider')
                st.session_state.config['track_window_s'] = st.slider(
                    "Ventana supervivencia (s)", 1.0, 15.0,
                    value=float(st.session_state.config.get('track_window_s', 7.0)), step=0.5,
                    help="Segundos sin detección antes de eliminar definitivamente un ID.",
                    key='sk_win_slider')
                st.session_state.config['track_class_w'] = st.slider(
                    "Penalización cambio de clase (m)", 0.0, 15.0,
                    value=float(st.session_state.config.get('track_class_w', 3.0)), step=0.5,
                    help="Coste extra en metros si la clase detectada no coincide con el track.",
                    key='sk_cls_slider')
                st.session_state.config['v3_lost_s'] = st.slider(
                    "Buffer lost (s)", 1.0, 10.0,
                    value=float(st.session_state.config.get('v3_lost_s', 7.0)), step=0.5,
                    help="Segundos que un track permanece en 'lost' antes de eliminarse. Súbelo si hay oclusiones.",
                    key='sk_lost_slider')
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Estado global y siguiente ──
    st.divider()
    setup_ok = (st.session_state.H_a is not None and
                st.session_state.H_b is not None and
                st.session_state.protos_day is not None)

    col_st, col_btn = st.columns([4, 2])
    with col_st:
        items = [
            ("Homografía Cam A", st.session_state.H_a is not None),
            ("Homografía Cam B", st.session_state.H_b is not None),
            ("PKL Día",          st.session_state.protos_day is not None),
            ("PKL Noche (opt.)", st.session_state.protos_night is not None),
        ]
        html = '<div style="display:flex;flex-wrap:wrap;gap:18px;align-items:center;">'
        for name, ok in items:
            dot = '<span class="dot-ok"></span>' if ok else '<span class="dot-off"></span>'
            c   = "#5aff8a" if ok else "#2a3a55"
            html += f'<span>{dot}<span style="color:{c};font-size:13px;">{name}</span></span>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    with col_btn:
        if setup_ok:
            if st.button("Siguiente → Procesar  ▶️", type="primary",
                         width="stretch"):
                st.session_state.step = 3
                st.rerun()
        else:
            missing = [n for n, ok in [("H_A", st.session_state.H_a is not None),
                                        ("H_B", st.session_state.H_b is not None),
                                        ("PKL", st.session_state.protos_day is not None)] if not ok]
            st.warning(f"Faltan: {', '.join(missing)}")


# ══════════════════════════════════════════════════════════
#  STEP 3 — PROCESS
# ══════════════════════════════════════════════════════════

def step_process():
    progress      = st.session_state.progress
    thread        = st.session_state.proc_thread
    is_processing = thread is not None and thread.is_alive()
    is_done       = progress.get('done', False)

    # ── CONFIG ──────────────────────────────────────────────
    if not is_processing and not is_done:
        st.markdown(
            '<div class="page-header">'
            '<h2>▶️ Procesado</h2>'
            '<p>Configura el segmento y lanza el análisis táctico.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        info_pan     = core.get_video_info(st.session_state.vid_pan)
        total_frames = info_pan['frames']
        fps          = info_pan['fps'] or 30.0
        total_min    = total_frames / fps / 60

        col_cfg, col_launch = st.columns([3, 2], gap="large")

        with col_cfg:
            # ── Segmento ──
            with st.expander("🎬 Segmento a procesar", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    start_min = st.number_input("Inicio (min)", 0.0, total_min,
                                                 value=st.session_state.config['start_frame'] / fps / 60,
                                                 step=1.0, format="%.1f")
                    start_f = int(start_min * 60 * fps)
                    st.caption(f"Frame {start_f:,}")
                with col2:
                    dur_min = st.number_input("Duración (min)", 0.5, total_min,
                                               value=st.session_state.config['n_frames'] / fps / 60,
                                               step=1.0, format="%.1f")
                    n_frames = int(dur_min * 60 * fps)
                    st.caption(f"{n_frames:,} frames")

                col3, col4 = st.columns(2)
                with col3:
                    skip = st.selectbox("Detección cada N frames", [1, 2, 3, 5],
                                         index=2,
                                         help="1 = máxima precisión y lentitud · 5 = máxima velocidad")
                    st.caption(f"≈ {fps/skip:.0f} detecciones/s")
                with col4:
                    conf = st.slider("Confianza YOLO", 0.30, 0.80,
                                      value=st.session_state.config['conf_thresh'], step=0.05)
                    st.caption(f"Umbral: {conf:.2f}")

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    field_length = st.number_input("Largo del campo (m)", 80.0, 120.0,
                                                    value=float(st.session_state.config.get('field_length', 105.0)),
                                                    step=0.5, format="%.1f")
                with col_f2:
                    field_width = st.number_input("Ancho del campo (m)", 50.0, 100.0,
                                                   value=float(st.session_state.config.get('field_width', 68.0)),
                                                   step=0.5, format="%.1f")

            # ── Iluminación ──
            has_dual_pkl = (st.session_state.protos_day is not None and
                            st.session_state.protos_night is not None)
            with st.expander("💡 Iluminación (PKL dual)", expanded=has_dual_pkl):
                bright_thresh = st.slider(
                    "Umbral día/noche (brillo medio del frame)",
                    min_value=30.0, max_value=160.0,
                    value=float(st.session_state.config.get('bright_thresh', 75.0)),
                    step=1.0,
                    help="Frames más brillantes que este umbral usan el PKL de día. "
                         "Activo solo si hay dos PKLs cargados.",
                    disabled=not has_dual_pkl,
                )
                if not has_dual_pkl:
                    st.caption("ℹ️ Carga PKL Día y PKL Noche para activar el switch automático.")
                else:
                    st.caption(f"Histéresis: día→noche < {bright_thresh-5:.0f}  |  "
                               f"noche→día > {bright_thresh+5:.0f}  |  suavizado: 7 frames")

                if has_dual_pkl and st.session_state.vid_pan:
                    if st.button("📊 Analizar curva de brillo", key='btn_calib'):
                        with st.spinner("Leyendo vídeo…"):
                            _cap = cv2.VideoCapture(st.session_state.vid_pan)
                            _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            _fps_c = _cap.get(cv2.CAP_PROP_FPS) or 30.0
                            _bright_vals, _frame_ids = [], []
                            fi = int(st.session_state.config.get('start_frame', 0))
                            while fi < _total:
                                _cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                                ret_c, frm_c = _cap.read()
                                if not ret_c: break
                                _bright_vals.append(float(np.mean(cv2.cvtColor(frm_c, cv2.COLOR_BGR2GRAY))))
                                _frame_ids.append(fi)
                                fi += 100
                            _cap.release()

                        import matplotlib
                        matplotlib.use('Agg')
                        fig_cal, ax_cal = plt.subplots(figsize=(12, 3), facecolor='#060d1a')
                        ax_cal.set_facecolor('#0c1830')
                        t_arr = [f / _fps_c for f in _frame_ids]
                        ax_cal.plot(t_arr, _bright_vals, color='#60a5fa', lw=1.2, alpha=0.8)
                        ax_cal.fill_between(t_arr, _bright_vals, alpha=0.12, color='#60a5fa')
                        ax_cal.axhline(bright_thresh, color='#fde047', lw=1.5,
                                       linestyle='--', label=f'umbral={bright_thresh:.0f}')
                        _med = float(np.median(_bright_vals))
                        ax_cal.axhline(_med, color='#a855f7', lw=1.0,
                                       linestyle='-.', alpha=0.7, label=f'mediana={_med:.0f}')
                        ax_cal.set_xlabel('Tiempo (s)', color='#8a9bbf', fontsize=9)
                        ax_cal.set_ylabel('Brillo', color='#8a9bbf', fontsize=9)
                        ax_cal.tick_params(colors='#8a9bbf', labelsize=8)
                        ax_cal.legend(fontsize=8, facecolor='#0c1830', labelcolor='white',
                                      loc='upper right')
                        ax_cal.set_title(
                            f'Curva de brillo — mediana={_med:.0f}  '
                            f'min={min(_bright_vals):.0f}  max={max(_bright_vals):.0f}',
                            color='white', fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig_cal)
                        plt.close(fig_cal)
                        _p25 = float(np.percentile(_bright_vals, 25))
                        _p75 = float(np.percentile(_bright_vals, 75))
                        st.info(f"💡 Umbral sugerido: entre {_p25:.0f} (P25) y {_p75:.0f} (P75)")

            # ── Costura ──
            lx = st.session_state.config.get('limite_x', 49.0)
            if st.session_state.H_a is not None and st.session_state.H_b is not None:
                info_lx = core.get_video_info(st.session_state.vid_pan)
                lx_auto = core.compute_limite_x(st.session_state.H_a, st.session_state.H_b,
                                                info_lx['width'], info_lx['height'] // 2)
                lx = float(np.clip(lx_auto, 20.0, 80.0))

            with st.expander("🧩 Costura A/B", expanded=False):
                limite_x = st.slider("Línea de costura (metros desde izquierda)", 20.0, 80.0,
                                      value=lx, step=0.5,
                                      help="Calculado automáticamente desde las homografías")

            # ── Tiempos del partido ──
            with st.expander("⏱️ Tiempos del partido", expanded=True):
                st.markdown('<div style="color:#4a5a80;font-size:13px;margin-bottom:12px;">'
                            'Define los intervalos de juego para excluir calentamiento y descanso.</div>',
                            unsafe_allow_html=True)

                # ── Scrubber de vídeo estándar ──
                vid_std_path = st.session_state.get('vid_std')
                if vid_std_path and os.path.exists(vid_std_path):
                    with st.expander("🎬 Marcar tiempos visualmente (vídeo estándar)", expanded=False):
                        _video_scrubber(vid_std_path, fps, total_frames)
                else:
                    st.caption("💡 Carga el vídeo estándar en el paso 1 para poder marcar los tiempos visualmente.")

                use_intervals = st.checkbox("Definir intervalos de juego", value=True)

                match_intervals = None
                halftime_frame  = None

                if use_intervals:
                    # Valores por defecto: marcas del scrubber si existen
                    def _f_to_min(f, default):
                        return round(f / fps / 60, 2) if f is not None else default

                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.markdown('<div style="color:#2979ff;font-weight:600;font-size:13px;'
                                    'margin-bottom:8px;">🔵 Primer tiempo</div>', unsafe_allow_html=True)
                        t1s = st.number_input("Inicio 1T (min)", 0.0, total_min,
                                              value=_f_to_min(st.session_state.mark_t1s, 2.0),
                                              step=0.5, key='t1s')
                        t1e = st.number_input("Fin 1T (min)",    0.0, total_min,
                                              value=_f_to_min(st.session_state.mark_t1e, 47.0),
                                              step=0.5, key='t1e')
                    with col_t2:
                        st.markdown('<div style="color:#c0395e;font-weight:600;font-size:13px;'
                                    'margin-bottom:8px;">🔴 Segundo tiempo</div>', unsafe_allow_html=True)
                        t2s = st.number_input("Inicio 2T (min)", 0.0, total_min,
                                              value=_f_to_min(st.session_state.mark_t2s, 50.0),
                                              step=0.5, key='t2s')
                        t2e = st.number_input("Fin 2T (min)",    0.0, total_min,
                                              value=_f_to_min(st.session_state.mark_t2e, 95.0),
                                              step=0.5, key='t2e')

                    f1s = int(t1s * 60 * fps); f1e = int(t1e * 60 * fps)
                    f2s = int(t2s * 60 * fps); f2e = int(t2e * 60 * fps)
                    match_intervals = [(f1s, f1e), (f2s, f2e)]
                    halftime_frame  = f2s
                    total_play_min  = ((f1e - f1s) + (f2e - f2s)) / fps / 60

                    w1s, w1e = f1s / total_frames, f1e / total_frames
                    w2s, w2e = f2s / total_frames, f2e / total_frames
                    st.markdown(
                        f'<div style="font-size:11px;color:#4a5a80;margin-top:10px;margin-bottom:4px;">'
                        f'Timeline del vídeo</div>'
                        f'<div class="timeline-wrap">'
                        f'  <div class="timeline-half1" style="left:{w1s*100:.1f}%;width:{(w1e-w1s)*100:.1f}%;"></div>'
                        f'  <div class="timeline-half2" style="left:{w2s*100:.1f}%;width:{(w2e-w2s)*100:.1f}%;"></div>'
                        f'</div>'
                        f'<div style="display:flex;gap:16px;font-size:12px;color:#4a5a80;margin-top:4px;">'
                        f'  <span>🔵 1T: {t1s:.0f}–{t1e:.0f} min</span>'
                        f'  <span>🔴 2T: {t2s:.0f}–{t2e:.0f} min</span>'
                        f'  <span style="margin-left:auto;color:#00e676;">⏱ {total_play_min:.1f} min de juego</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Selector de segmento ──
                    _SEG_OPTS = ["🔵 Solo 1T", "🔴 Solo 2T", "⚽ Partido completo", "✏️ Personalizado"]
                    _seg_sel = st.radio(
                        "Segmento a analizar",
                        _SEG_OPTS,
                        index=st.session_state.get('_seg_sel_idx', 2),
                        horizontal=True,
                        key='seg_sel',
                    )
                    st.session_state['_seg_sel_idx'] = _SEG_OPTS.index(_seg_sel)
                    if _seg_sel == "🔵 Solo 1T":
                        start_f  = f1s
                        n_frames = f1e - f1s
                    elif _seg_sel == "🔴 Solo 2T":
                        start_f  = f2s
                        n_frames = f2e - f2s
                    elif _seg_sel == "⚽ Partido completo":
                        start_f  = f1s
                        n_frames = f2e - f1s  # incluye descanso; frames vacíos son inocuos
                    # "✏️ Personalizado": usar start_f y n_frames de los inputs de arriba
                    dur_min = n_frames / fps / 60

            # ── Porteros + colores ──
            with st.expander("🧤 Porteros & Colores del minimap", expanded=True):
                home_attacks_left = st.toggle(
                    "Local defiende la portería IZQUIERDA en el 1T",
                    value=st.session_state.config.get('home_attacks_left', True),
                )
                ht_1 = "izquierda ←" if home_attacks_left else "→ derecha"
                ht_2 = "→ derecha"   if home_attacks_left else "izquierda ←"
                st.markdown(
                    f'<div style="font-size:13px;padding:8px 0 12px;">'
                    f'<span class="badge-home">Local</span> '
                    f'<span style="color:#8a9bbf;">defiende 1T: <b style="color:#e8edf8;">{ht_1}</b> · '
                    f'2T: <b style="color:#e8edf8;">{ht_2}</b></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div style="font-size:12px;color:#4a5a80;margin-bottom:8px;">'
                            'Color de camiseta para el minimap 2D</div>', unsafe_allow_html=True)
                col_tc1, col_tc2 = st.columns(2)
                with col_tc1:
                    home_dot_hex = st.color_picker(
                        "Local (jugadores)",
                        value=st.session_state.config.get('home_dot_hex', '#e63946'),
                        key='home_dot_pick')
                with col_tc2:
                    away_dot_hex = st.color_picker(
                        "Visitante (jugadores)",
                        value=st.session_state.config.get('away_dot_hex', '#1d70b8'),
                        key='away_dot_pick')

                gk_home_hex  = st.session_state.config.get('gk_home_hex', '#ffff00')
                gk_away_hex  = st.session_state.config.get('gk_away_hex', '#ff6600')
                use_gk_color = False

            # ── Salida ──
            with st.expander("💾 Salida de video", expanded=False):
                export_video = st.checkbox("Generar vídeo con overlay táctico", value=True,
                                           help="Más lento. Se guarda en data/videos/.")
                if export_video:
                    match_n = st.session_state.match_name or "match"
                    st.caption(f"→ `data/videos/{match_n}_f{start_f}_{start_f+n_frames}_tracking.mp4`")

        with col_launch:
            # ── Cargar modelos si hace falta ──
            if st.session_state.model_A is None:
                model_path = "runs/detect/modelo_players_v24_panoramic2/weights/best.pt"
                ball_path  = "runs/detect/modelo_ball_v33/weights/best.pt"
                with st.spinner("Cargando modelos YOLO…"):
                    st.session_state.model_A    = YOLO(model_path)
                    st.session_state.model_B    = YOLO(model_path)
                    st.session_state.model_ball = YOLO(ball_path) if os.path.exists(ball_path) else None
                st.success("✅ Modelos cargados")

            # ── Launch card ──
            st.markdown('<div class="card-accent">', unsafe_allow_html=True)
            _section("🚀 Lanzar análisis")

            if _DEVICE == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                st.markdown(f'<div class="gpu-badge">'
                            f'🟢 GPU: {gpu_name[:24]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="cpu-badge">'
                            '🟡 CPU — procesado lento</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            detect_ball = st.checkbox(
                "🎾 Detectar balón",
                value=False,
                help="Desactívalo para ~2× más velocidad. Añade 2 llamadas YOLO extra por frame.",
            )

            # ETA estimada (pre-lanzamiento)
            # fps_est = frames de vídeo procesados por segundo de reloj (no multiplicador realtime)
            _sam_loaded = st.session_state.model_sam is not None
            fps_est = 12.0 if _DEVICE == 'cuda' else 1.5   # ~12 fps proc. sin SAM en GPU
            if _sam_loaded:  fps_est *= 0.06   # SAM2 ~17× más lento → ~0.7 fps proc.
            if detect_ball:  fps_est *= 0.70
            # skip=N → procesamos 1/N frames al mismo ritmo → N× más rápido
            eta_s = n_frames / (fps_est * max(1, skip))
            st.markdown(
                f'<div style="background:#0c1830;border:1px solid #1a3060;border-radius:10px;'
                f'padding:14px;margin:12px 0;">'
                f'<div style="font-size:11px;color:#4a5a80;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:8px;">Estimación</div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                f'<span style="color:#8a9bbf;font-size:13px;">Frames</span>'
                f'<span style="color:#e8edf8;font-weight:700;font-size:13px;">{n_frames:,}</span></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                f'<span style="color:#8a9bbf;font-size:13px;">Duración</span>'
                f'<span style="color:#e8edf8;font-weight:700;font-size:13px;">{dur_min:.1f} min</span></div>'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="color:#8a9bbf;font-size:13px;">ETA aprox.</span>'
                f'<span style="color:#00e676;font-weight:700;font-size:13px;">'
                f'{"~{:.0f} min".format(eta_s/60) if eta_s > 90 else "~{:.0f} s".format(eta_s)}'
                f'</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            cfg_txt = (f"frames {start_f:,}–{start_f+n_frames:,} · "
                       f"conf={conf:.2f} · skip={skip} · "
                       f"costura={lx:.1f}m")
            st.caption(cfg_txt)

            if st.button("▶️  Iniciar análisis", type="primary",
                         width="stretch"):
                out_video = None
                if export_video:
                    match_n = st.session_state.match_name or "match"
                    out_video = f"data/videos/{match_n}_f{start_f}-{start_f+n_frames}_tracking.mp4"
                    os.makedirs("data/videos", exist_ok=True)

                gk_home_lab = _hex_to_lab(gk_home_hex) if use_gk_color else None
                gk_away_lab = _hex_to_lab(gk_away_hex) if use_gk_color else None
                st.session_state.config = {
                    'start_frame':        start_f,
                    'n_frames':           n_frames,
                    'skip_frames':        skip,
                    'conf_thresh':        conf,
                    'conf_ball':          core.CONF_BALL,
                    'limite_x':           limite_x,
                    'output_video_path':  out_video,
                    'minimap_size':       (400, 230),
                    'home_attacks_left':  home_attacks_left,
                    'halftime_frame':     halftime_frame,
                    'match_intervals':    match_intervals,
                    'gk_home_color_lab':  gk_home_lab,
                    'gk_away_color_lab':  gk_away_lab,
                    'home_dot_hex':       home_dot_hex,
                    'away_dot_hex':       away_dot_hex,
                    'gk_home_hex':        gk_home_hex,
                    'gk_away_hex':        gk_away_hex,
                    'use_gk_color':       use_gk_color,
                    'field_length':       field_length,
                    'field_width':        field_width,
                    'bright_thresh':      bright_thresh,
                    'tracker_type':       st.session_state.config.get('tracker_type', 'strongkalman'),
                    'bt_high_thresh':     st.session_state.config.get('bt_high_thresh', 0.38),
                    'bt_match_thresh_m':  st.session_state.config.get('bt_match_thresh_m', 5.0),
                    'bt_max_lost_s':      st.session_state.config.get('bt_max_lost_s', 3.0),
                    'bt_gallery_ttl_s':   st.session_state.config.get('bt_gallery_ttl_s', 20.0),
                    'bt_reid_hue_thresh': st.session_state.config.get('bt_reid_hue_thresh', 0.33),
                    'track_max_dist_m':   st.session_state.config.get('track_max_dist_m', 8.0),
                    'track_window_s':     st.session_state.config.get('track_window_s', 7.0),
                    'track_class_w':      st.session_state.config.get('track_class_w', 3.0),
                    'v3_lost_s':          st.session_state.config.get('v3_lost_s', 7.0),
                    'device':             _DEVICE,
                    'half_precision':     _HALF_PREC,
                    'imgsz':              1280,
                }
                import time as _time
                progress_dict = {'t_start': _time.time()}
                st.session_state.progress = progress_dict
                t = threading.Thread(
                    target=core.process_segment,
                    args=(
                        st.session_state.vid_pan,
                        st.session_state.vid_std,
                        st.session_state.H_a,
                        st.session_state.H_b,
                        st.session_state.protos_day,
                        st.session_state.protos_night,
                        st.session_state.config,
                        st.session_state.model_A,
                        st.session_state.model_B,
                        st.session_state.model_ball if detect_ball else None,
                        progress_dict,
                        st.session_state.model_sam,
                    ),
                    daemon=True,
                )
                t.start()
                st.session_state.proc_thread = t
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    # ── LIVE VIEW ────────────────────────────────────────────────────────────
    elif is_processing or is_done:
        pct       = progress.get('pct', 0)
        frame_idx = progress.get('frame_idx', 0)
        tracked   = progress.get('tracked', [])
        ball      = progress.get('ball', None)
        error     = progress.get('error', None)

        if error:
            st.markdown(
                f'<div class="card-warn">⚠️ <b>Error en procesado</b><br>'
                f'<code style="font-size:12px;">{error}</code></div>',
                unsafe_allow_html=True,
            )
            return

        # ── Header ──
        if is_processing:
            st.markdown(
                '<div class="page-header">'
                '<h2>⚙️ Analizando…</h2>'
                '<p>El análisis está en curso. La vista se actualiza en tiempo real.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="page-header" style="border-color:#1a5030;">'
                '<h2 style="color:#00e676 !important;">✅ Análisis completado</h2>'
                '<p>Revisa los resultados o descarga los datos.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Progress bar ──
        if is_processing:
            import time as _time
            pct_int = int(pct * 100)
            n_cfg   = st.session_state.config.get('n_frames', 1)
            t_start = progress.get('t_start')
            if t_start and pct > 0.005:
                elapsed   = _time.time() - t_start
                eta_s     = elapsed / pct * (1.0 - pct)
                proc_fps  = (pct * n_cfg) / max(elapsed, 1)
                if eta_s >= 3600:
                    eta_str = f"~{eta_s/3600:.1f} h"
                elif eta_s >= 90:
                    eta_str = f"~{eta_s/60:.0f} min"
                else:
                    eta_str = f"~{eta_s:.0f} s"
                elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60):02d}s"
                speed_str   = f"{proc_fps:.1f} fps proc."
            else:
                eta_str = "calculando…"
                elapsed_str = "—"
                speed_str   = "—"
            st.markdown(
                f'<div class="progress-wrap">'
                f'<div class="progress-label" style="display:flex;justify-content:space-between;">'
                f'<span>Frame {frame_idx:,} — {pct_int}% completado</span>'
                f'<span style="color:#00e676;font-weight:700;">ETA {eta_str} &nbsp;·&nbsp; '
                f'<span style="color:#8a9bbf;font-weight:400;">{speed_str} &nbsp;·&nbsp; transcurrido {elapsed_str}</span></span>'
                f'</div>'
                f'<div class="progress-bar-bg">'
                f'  <div class="progress-bar-fill" style="width:{pct_int}%;"></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── KPI row ──
        cls_count = {}
        for d in tracked:
            cls_count[d.get('cls', '?')] = cls_count.get(d.get('cls', '?'), 0) + 1

        kpi_html = '<div class="kpi-grid">'
        kpis = [
            ("📍", f"{frame_idx:,}",  "Frame"),
            ("👥", str(len(tracked)), "IDs activos"),
            ("🔴", str(cls_count.get('player_home', 0)), "Locales"),
            ("🔵", str(cls_count.get('player_away', 0)), "Visitantes"),
            ("🧤", str(cls_count.get('gk_home', 0) + cls_count.get('gk_away', 0)), "Porteros"),
            ("⚽", f"{ball[0]:.0f},{ball[1]:.0f}m" if ball else "—", "Balón"),
        ]
        for icon, val, label in kpis:
            kpi_html += (f'<div class="kpi-card">'
                         f'<div class="kpi-icon">{icon}</div>'
                         f'<div class="kpi-value">{val}</div>'
                         f'<div class="kpi-label">{label}</div>'
                         f'</div>')
        kpi_html += '</div>'
        st.markdown(kpi_html, unsafe_allow_html=True)

        # ── Badges ──
        badge_map_live = {
            'player_home': ('badge-home', 'Local'),
            'player_away': ('badge-away', 'Visitante'),
            'referee':     ('badge-ref',  'Árbitro'),
            'gk_home':     ('badge-gk',   'GK Local'),
            'gk_away':     ('badge-gk',   'GK Visit.'),
            'unknown':     ('badge-unknown', '?'),
        }
        html_badges = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">'
        for cls, cnt in sorted(cls_count.items(), key=lambda x: -x[1]):
            bc, lbl = badge_map_live.get(cls, ('badge-unknown', cls))
            html_badges += f'<span class="{bc}">{lbl}: {cnt}</span>'
        html_badges += '</div>'
        st.markdown(html_badges, unsafe_allow_html=True)

        # ── Dual view ──
        col_vid, col_map = st.columns([3, 2], gap="medium")

        with col_vid:
            if st.session_state.vid_std:
                cap_live = cv2.VideoCapture(st.session_state.vid_std)
                cap_live.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_live, frm_live = cap_live.read(); cap_live.release()
                if ret_live:
                    _fL = st.session_state.config.get('field_length', 98.0)
                    _fA = st.session_state.config.get('field_width',  61.0)
                    _cfg = st.session_state.config
                    minimap = core.render_minimap_frame(
                        tracked, ball, L=_fL, A=_fA,
                        protos=st.session_state.protos_day,
                        home_hex=_cfg.get('home_dot_hex','#e63946'),
                        away_hex=_cfg.get('away_dot_hex','#1d70b8'),
                        gk_home_hex=_cfg.get('gk_home_hex','#ffff00'),
                        gk_away_hex=_cfg.get('gk_away_hex','#ff6600'),
                    )
                    annotated = core.overlay_minimap(frm_live, minimap)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             caption=f"Vídeo estándar + minimap · frame {frame_idx:,}",
                             width="stretch")

        with col_map:
            _fL = st.session_state.config.get('field_length', 98.0)
            _fA = st.session_state.config.get('field_width',  61.0)
            _cfg = st.session_state.config
            minimap_lg = core.render_minimap_frame(
                tracked, ball, size_px=(560, 320), L=_fL, A=_fA,
                protos=st.session_state.protos_day,
                home_hex=_cfg.get('home_dot_hex','#e63946'),
                away_hex=_cfg.get('away_dot_hex','#1d70b8'),
                gk_home_hex=_cfg.get('gk_home_hex','#ffff00'),
                gk_away_hex=_cfg.get('gk_away_hex','#ff6600'),
            )
            st.image(cv2.cvtColor(minimap_lg, cv2.COLOR_BGR2RGB),
                     caption="Mapa táctico en tiempo real",
                     width="stretch")

            frame_records = progress.get('frame_records', [])
            if frame_records:
                df_live = pd.DataFrame(frame_records[-500:])
                if 'cls' in df_live.columns:
                    counts = df_live.groupby('cls').size().reset_index(name='dets')
                    st.dataframe(counts, hide_index=True, width="stretch",
                                 column_config={'cls': 'Clase', 'dets': 'Detecciones'})

        if is_processing:
            time.sleep(0.8)
            st.rerun()
        else:
            st.divider()
            col_ok, _ = st.columns([2, 5])
            col_ok.success("✅ Análisis completado")
            if col_ok.button("Ver resultados  📊", type="primary",
                             width="stretch"):
                st.session_state.step = 4
                st.rerun()


# ══════════════════════════════════════════════════════════
#  STEP 4 — RESULTS
# ══════════════════════════════════════════════════════════

def _pitch_fig(fl, fw):
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#060d1a')
    ax.set_facecolor('#060d1a')
    core.draw_pitch_mpl(ax, fl, fw)
    return fig, ax


def _save_fig(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#060d1a', bbox_inches='tight', dpi=dpi)
    buf.seek(0); plt.close(fig)
    return buf


def step_results():
    progress = st.session_state.progress

    if not progress.get('done'):
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">📊</div>'
            '<h3>Sin resultados todavía</h3>'
            '<p>Procesa un segmento de vídeo en el paso 3 para ver aquí el análisis táctico.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    frame_records = progress.get('frame_records', [])
    ball_records  = progress.get('ball_records',  [])
    heatmaps      = progress.get('heatmaps',      {})
    cfg           = st.session_state.config
    fps           = float(progress.get('fps', 30.0))
    _FL           = float(cfg.get('field_length', core.L_M))
    _FW           = float(cfg.get('field_width',  core.A_M))

    # ── Segment filter ──────────────────────────────────────
    if frame_records:
        _all_f  = [r['frame'] for r in frame_records]
        _f_min, _f_max = min(_all_f), max(_all_f)
        _t_min  = round(_f_min / fps / 60.0, 2)
        _t_max  = round(_f_max / fps / 60.0, 2)
        with st.expander("✂️ Filtrar segmento temporal", expanded=False):
            _seg_s, _seg_e = st.slider(
                "Rango de análisis (min)",
                min_value=_t_min, max_value=_t_max,
                value=(_t_min, _t_max), step=0.5, format="%.1f",
                key="seg_range",
            )
            st.caption(
                f"Segmento seleccionado: **{_seg_s:.1f} – {_seg_e:.1f} min** "
                f"({_seg_e - _seg_s:.1f} min de {_t_max - _t_min:.1f} min totales)"
            )
        if _seg_s > _t_min or _seg_e < _t_max:
            _fs = int(_seg_s * 60 * fps)
            _fe = int(_seg_e * 60 * fps)
            frame_records = [r for r in frame_records if _fs <= r['frame'] <= _fe]
            ball_records  = [r for r in ball_records  if _fs <= r['frame'] <= _fe]
            _HW = int(_FL * core.ESCALA)
            _HH = int(_FW * core.ESCALA)
            _SEG_CLS = ['player_home', 'player_away', 'gk_home', 'gk_away', 'referee', 'unknown']
            heatmaps = {c: np.zeros((_HH, _HW), dtype=np.float32) for c in _SEG_CLS}
            for _r in frame_records:
                if _r['cls'] in heatmaps:
                    heatmaps[_r['cls']][
                        int(np.clip(_r['y_m'] * core.ESCALA, 0, _HH - 1)),
                        int(np.clip(_r['x_m'] * core.ESCALA, 0, _HW - 1)),
                    ] += 1.0
    # ────────────────────────────────────────────────────────

    df = pd.DataFrame(frame_records) if frame_records else pd.DataFrame()

    home_team = st.session_state.match_meta.get('home_team') or 'Local'
    away_team = st.session_state.match_meta.get('away_team') or 'Visitante'
    cfg_with_teams = {**cfg, 'home_team': home_team, 'away_team': away_team}

    st.markdown(
        '<div class="page-header">'
        '<h2>📊 Resultados del análisis</h2>'
        '<p>Trayectorias, heatmaps, estadísticas físicas y tácticas con análisis por IA.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── KPI summary ──
    if len(df):
        n_frames_proc = cfg.get('n_frames', 0)
        dur_min       = n_frames_proc / fps / 60
        n_home        = df[df['cls'] == 'player_home']['gid'].nunique()
        n_away        = df[df['cls'] == 'player_away']['gid'].nunique()
        n_gk          = df[df['cls'].isin(['gk_home','gk_away'])]['gid'].nunique()

        kpi_html = '<div class="kpi-grid">'
        for icon, val, label in [
            ("⏱",  f"{dur_min:.1f} min",    "Duración analizada"),
            ("🔴",  str(n_home),             home_team),
            ("🔵",  str(n_away),             away_team),
            ("🧤",  str(n_gk),               "Porteros"),
            ("⚽",  str(len(ball_records)),  "Det. balón"),
        ]:
            kpi_html += (f'<div class="kpi-card">'
                         f'<div class="kpi-icon">{icon}</div>'
                         f'<div class="kpi-value">{val}</div>'
                         f'<div class="kpi-label">{label}</div>'
                         f'</div>')
        kpi_html += '</div>'
        st.markdown(kpi_html, unsafe_allow_html=True)

    st.divider()

    tab_map, tab_heat, tab_phys, tab_tact, tab_ai, tab_export = st.tabs([
        "🗺️ Mapa táctico",
        "🔥 Heatmaps",
        "📏 Físico",
        "🧠 Táctico",
        "🤖 Análisis IA",
        "💾 Exportar",
    ])

    # ════════════════════════════════════════════════════════
    #  Mapa táctico
    # ════════════════════════════════════════════════════════
    with tab_map:
        if len(df):
            import matplotlib.patches as _mpatch
            from matplotlib.lines import Line2D

            hal    = cfg.get('halftime_frame', None)
            hal_al = cfg.get('home_attacks_left', True)

            home_form, home_cdf = core.extract_formation_density(
                df, team='home', field_length=_FL, field_width=_FW,
                home_attacks_left=hal_al, halftime_frame=hal)
            away_form, away_cdf = core.extract_formation_density(
                df, team='away', field_length=_FL, field_width=_FW,
                home_attacks_left=hal_al, halftime_frame=hal)

            home_color = cfg.get('home_dot_hex', '#dc143c')
            away_color = cfg.get('away_dot_hex', '#1e90ff')

            def _cdf_plot_x(cdf, is_home):
                # x_norm encodes team-relative (0=own goal). To recover absolute field x:
                # when is_home==hal_al the team's own goal is at low-x → absolute = x_norm*FL
                # otherwise (own goal at high-x) → absolute = FL - x_norm*FL
                direct = (is_home == hal_al)
                return cdf['x_norm'].values * _FL if direct else (_FL - cdf['x_norm'].values * _FL)

            def _gk_field_pos(is_home, gk_cls):
                gk_sub = df[df['cls'] == gk_cls]
                if gk_sub.empty:
                    return None, None
                gk_xn = core._x_norm_series(
                    gk_sub['x_m'], gk_sub['frame'], is_home, hal_al, hal, _FL)
                direct = (is_home == hal_al)
                gk_px = float(np.mean(gk_xn)) if direct else (_FL - float(np.mean(gk_xn)))
                return gk_px, float(gk_sub['y_m'].mean())

            # ── Cabecera formaciones ──
            col_hf, col_af = st.columns(2)
            col_hf.markdown(
                f'<div style="text-align:center;background:#1a1a2e;border-radius:8px;'
                f'padding:10px;border:1px solid {home_color};">'
                f'<span style="color:{home_color};font-size:1.3em;font-weight:bold;">'
                f'{home_team}</span><br>'
                f'<span style="color:white;font-size:2em;font-weight:bold;">{home_form}</span>'
                f'</div>', unsafe_allow_html=True)
            col_af.markdown(
                f'<div style="text-align:center;background:#1a1a2e;border-radius:8px;'
                f'padding:10px;border:1px solid {away_color};">'
                f'<span style="color:{away_color};font-size:1.3em;font-weight:bold;">'
                f'{away_team}</span><br>'
                f'<span style="color:white;font-size:2em;font-weight:bold;">{away_form}</span>'
                f'</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

            # ════════════════════════════════════════════════════
            # Plot 1 — Posición media + alineación (sin etiquetas)
            # ════════════════════════════════════════════════════
            fig1, ax1 = _pitch_fig(_FL, _FW)

            home_arrow_x = (_FL * 0.55, _FL * 0.85) if hal_al else (_FL * 0.45, _FL * 0.15)
            away_arrow_x = (_FL * 0.45, _FL * 0.15) if hal_al else (_FL * 0.55, _FL * 0.85)
            for arr_x, arr_y, color, tname, va in [
                (home_arrow_x, _FW * 0.08, home_color, home_team, 'top'),
                (away_arrow_x, _FW * 0.92, away_color, away_team, 'bottom'),
            ]:
                ax1.annotate('', xy=(arr_x[1], arr_y), xytext=(arr_x[0], arr_y),
                             arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
                dy = -_FW * 0.04 if va == 'top' else _FW * 0.04
                ax1.text((arr_x[0] + arr_x[1]) / 2, arr_y + dy,
                         f'Ataque {tname}', color=color, fontsize=7,
                         ha='center', va=va, fontweight='bold')

            for is_home, color, gk_cls, cdf in [
                (True,  home_color, 'gk_home', home_cdf),
                (False, away_color, 'gk_away', away_cdf),
            ]:
                if not cdf.empty:
                    plot_x = _cdf_plot_x(cdf, is_home)
                    # líneas de formación
                    for line_id in sorted(cdf['line'].unique()):
                        mask = (cdf['line'] == line_id).values
                        lx = plot_x[mask]
                        ly = cdf.loc[cdf['line'] == line_id, 'y_m'].values
                        if len(lx) >= 2:
                            s = np.argsort(ly)
                            ax1.plot(lx[s], ly[s], color=color, alpha=0.3,
                                     lw=2, zorder=3, solid_capstyle='round')
                    # puntos de jugadores de campo (sin etiqueta de rol)
                    ax1.scatter(plot_x, cdf['y_m'].values, c=color, s=140, marker='o',
                                alpha=0.95, zorder=5, edgecolors='white', linewidths=1.5)
                # GK
                gk_px, gk_py = _gk_field_pos(is_home, gk_cls)
                if gk_px is not None:
                    ax1.scatter(gk_px, gk_py, c=color, s=220, marker='D',
                                alpha=0.95, zorder=5, edgecolors='white', linewidths=1.5)
                    ax1.text(gk_px, gk_py, 'GK', color='white', fontsize=5.5,
                             ha='center', va='center', fontweight='bold', zorder=6)

            legend_els = [
                Line2D([0],[0], marker='o', color='w', markerfacecolor=home_color,
                       markersize=9, label=f'{home_team}  {home_form}'),
                Line2D([0],[0], marker='D', color='w', markerfacecolor=home_color,
                       markersize=8, label=f'GK {home_team}'),
                Line2D([0],[0], marker='o', color='w', markerfacecolor=away_color,
                       markersize=9, label=f'{away_team}  {away_form}'),
                Line2D([0],[0], marker='D', color='w', markerfacecolor=away_color,
                       markersize=8, label=f'GK {away_team}'),
            ]
            ax1.legend(handles=legend_els, facecolor='#0c1830', labelcolor='white',
                       fontsize=8.5, loc='lower right', framealpha=0.9, edgecolor='#2a3a5a')
            ax1.set_title(
                f"Posición media — {home_team} {home_form}  vs  {away_team} {away_form}",
                color='white', fontsize=11, fontweight='bold', pad=8)
            st.image(_save_fig(fig1), width="stretch")

            # ════════════════════════════════════════════════════
            # Plots 2 + 3 en columnas
            # ════════════════════════════════════════════════════
            col_def, col_blk = st.columns(2)

            # ── Plot 2: Línea defensiva ──────────────────────────
            with col_def:
                fig2, ax2 = _pitch_fig(_FL, _FW)
                for is_home, color, gk_cls, cdf, tname in [
                    (True,  home_color, 'gk_home', home_cdf, home_team),
                    (False, away_color, 'gk_away', away_cdf, away_team),
                ]:
                    if cdf.empty:
                        continue
                    xn_m   = cdf['x_norm'].values * _FL
                    plot_x = xn_m if (is_home == hal_al) else (_FL - xn_m)
                    # línea 0 = más próxima a la portería propia = línea defensiva
                    def_mask = (cdf['line'] == 0).values
                    def_x = plot_x[def_mask]
                    def_y = cdf.loc[cdf['line'] == 0, 'y_m'].values
                    if len(def_x) == 0:
                        continue
                    def_line_x = float(np.mean(def_x))
                    # depth_from_goal siempre en metros desde la portería propia del equipo
                    depth_from_goal = float(np.mean(xn_m[def_mask]))
                    # banda sombreada alrededor de la línea
                    ax2.axvspan(def_line_x - 1.5, def_line_x + 1.5,
                                color=color, alpha=0.08, zorder=1)
                    # línea vertical discontinua
                    ax2.axvline(def_line_x, color=color, lw=2, ls='--', alpha=0.85, zorder=4)
                    # puntos de los defensas
                    ax2.scatter(def_x, def_y, c=color, s=130, marker='o',
                                alpha=0.95, zorder=5, edgecolors='white', linewidths=1.5)
                    # etiqueta: profundidad desde portería propia + posición absoluta en campo
                    label_y = _FW * 0.07 if is_home else _FW * 0.93
                    va_lbl  = 'bottom' if is_home else 'top'
                    ax2.text(def_line_x, label_y,
                             f'{tname}\n{depth_from_goal:.1f}m (portería propia)',
                             color=color, fontsize=7.5, ha='center', va=va_lbl,
                             fontweight='bold', zorder=6,
                             bbox=dict(boxstyle='round,pad=0.3', fc='#0c1830', alpha=0.7, ec='none'))
                ax2.set_title('Línea defensiva media', color='white',
                              fontsize=10, fontweight='bold', pad=6)
                st.image(_save_fig(fig2), width="stretch")

            # ── Plot 3: Bloque de equipo ─────────────────────────
            with col_blk:
                fig3, ax3 = _pitch_fig(_FL, _FW)
                for is_home, color, gk_cls, cdf, tname in [
                    (True,  home_color, 'gk_home', home_cdf, home_team),
                    (False, away_color, 'gk_away', away_cdf, away_team),
                ]:
                    if cdf.empty:
                        continue
                    plot_x = _cdf_plot_x(cdf, is_home)
                    all_y  = cdf['y_m'].values
                    xmin, xmax = float(np.min(plot_x)), float(np.max(plot_x))
                    ymin, ymax = float(np.min(all_y)),  float(np.max(all_y))
                    blk_w = xmax - xmin
                    blk_h = ymax - ymin
                    # padding para que el rectángulo cubra el espacio real del equipo
                    _pad = 4.0
                    # rectángulo del bloque
                    rect = _mpatch.FancyBboxPatch(
                        (xmin - _pad, ymin - _pad), blk_w + 2 * _pad, blk_h + 2 * _pad,
                        boxstyle='round,pad=0.5',
                        lw=2, edgecolor=color, facecolor=color, alpha=0.35, zorder=2)
                    ax3.add_patch(rect)
                    # centroide del equipo
                    cx, cy = float(np.mean(plot_x)), float(np.mean(all_y))
                    ax3.scatter(cx, cy, c=color, s=200, marker='*',
                                alpha=1.0, zorder=5, edgecolors='white', linewidths=1)
                    # dimensiones del bloque
                    ax3.text(cx, cy + _FW * 0.07,
                             f'{tname}  {blk_w:.0f}×{blk_h:.0f}m',
                             color=color, fontsize=7, ha='center', va='bottom',
                             fontweight='bold', zorder=6,
                             bbox=dict(boxstyle='round,pad=0.3', fc='#0c1830', alpha=0.7, ec='none'))
                ax3.set_title('Bloque de equipo (prof. × anchura)', color='white',
                              fontsize=10, fontweight='bold', pad=6)
                st.image(_save_fig(fig3), width="stretch")

        else:
            st.markdown('<div class="empty-state"><div class="icon">🗺️</div>'
                        '<h3>Sin datos de posiciones</h3></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  Heatmaps
    # ════════════════════════════════════════════════════════
    with tab_heat:
        if heatmaps:
            smooth_hm = core.compute_heatmaps_smooth(heatmaps)
            col_h1, col_h2 = st.columns(2)
            for col, cls, title, cmap in [
                (col_h1, 'player_home', f'🔴 {home_team}', 'Reds'),
                (col_h2, 'player_away', f'🔵 {away_team}', 'Blues'),
            ]:
                hm = smooth_hm.get(cls, np.zeros((1, 1)))
                if hm.max() == 0:
                    col.info(f"Sin datos para {title}"); continue
                fig_h, ax_h = plt.subplots(figsize=(9, 6), facecolor='#060d1a')
                ax_h.set_facecolor('#060d1a')
                core.draw_pitch_mpl(ax_h, _FL, _FW)
                ax_h.imshow(hm / hm.max(), extent=[0, _FL, _FW, 0],
                            origin='upper', cmap=cmap, alpha=0.72, vmin=0, vmax=1, zorder=4)
                ax_h.set_title(title, color='white', fontsize=11, fontweight='bold')
                col.image(_save_fig(fig_h, dpi=120), width="stretch")

            hm_ball = progress.get('heatmap_ball')
            if hm_ball is not None and hm_ball.max() > 0:
                from scipy.ndimage import gaussian_filter
                hm_b_s = gaussian_filter(hm_ball, sigma=6)
                fig_b, ax_b = plt.subplots(figsize=(14, 7), facecolor='#060d1a')
                ax_b.set_facecolor('#060d1a')
                core.draw_pitch_mpl(ax_b, _FL, _FW)
                ax_b.imshow(hm_b_s / hm_b_s.max(), extent=[0, _FL, _FW, 0],
                            origin='upper', cmap='YlOrRd', alpha=0.68, vmin=0, vmax=1, zorder=4)
                ax_b.set_title("Densidad del balon", color='white', fontsize=11, fontweight='bold')
                st.image(_save_fig(fig_b, dpi=120), width="stretch")
        else:
            st.markdown('<div class="empty-state"><div class="icon">🔥</div>'
                        '<h3>Sin datos de heatmaps</h3></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  Físico
    # ════════════════════════════════════════════════════════
    with tab_phys:
        if not len(df):
            st.markdown('<div class="empty-state"><div class="icon">📏</div>'
                        '<h3>Sin datos</h3></div>', unsafe_allow_html=True)
        else:
            phys = core.compute_physical_stats(frame_records, fps, _FL, cfg=cfg_with_teams)

            # Convertir velocidades a km/h ANTES de cualquier slice para evitar
            # ambigüedad entre vistas y copias
            phys = phys.copy()
            phys['max_speed_ms'] = (phys['max_speed_ms'] * 3.6).round(1)
            phys['avg_speed_ms'] = (phys['avg_speed_ms'] * 3.6).round(1)

            # ── KPI por equipo ──
            hp = phys[phys['cls'] == 'player_home']
            ap = phys[phys['cls'] == 'player_away']

            col_ph, col_pa = st.columns(2, gap="large")
            for col, td, team, color in [
                (col_ph, hp, home_team, '#ff6b88'),
                (col_pa, ap, away_team, '#60b8ff'),
            ]:
                with col:
                    st.markdown(f'<div class="card">', unsafe_allow_html=True)
                    _section(f"{'🔴' if 'Local' in team or team==home_team else '🔵'} {team}")
                    if len(td):
                        khtml = '<div class="kpi-grid" style="gap:8px;">'
                        _n = 10  # jugadores de campo por equipo (sin GK)
                        for icon, val, lbl in [
                            ("🏃", f"{td['dist_m'].sum()/_n:.0f}m",           "Dist. media"),
                            ("⚡", f"{td['sprint_count'].sum()/_n:.1f}",       "Sprints/jug."),
                            ("💨", f"{td['max_speed_ms'].max():.1f}km/h",     "Vel. máx."),
                            ("🔥", f"{td['sprint_dist_m'].sum()/_n:.0f}m",    "Dist. sprint"),
                        ]:
                            khtml += (f'<div class="kpi-card" style="min-width:90px;">'
                                      f'<div class="kpi-icon">{icon}</div>'
                                      f'<div class="kpi-value" style="font-size:18px;color:{color};">{val}</div>'
                                      f'<div class="kpi-label">{lbl}</div></div>')
                        khtml += '</div>'
                        st.markdown(khtml, unsafe_allow_html=True)

                        # Distribución de distancia por tercio
                        thirds_vals = [td['dist_def_m'].sum()/_n, td['dist_mid_m'].sum()/_n, td['dist_att_m'].sum()/_n]
                        fig_t, ax_t = plt.subplots(figsize=(5, 2.5), facecolor='#060d1a')
                        ax_t.set_facecolor('#0c1830')
                        bars_t = ax_t.bar(['Def.', 'Med.', 'Ata.'], thirds_vals,
                                          color=[color + '88', color + 'aa', color],
                                          edgecolor='none', width=0.6)
                        ax_t.bar_label(bars_t, fmt='%.0fm', padding=3, color='white', fontsize=9)
                        ax_t.tick_params(colors='#8a9bbf', labelsize=9)
                        ax_t.set_ylabel('Dist. media (m)', color='#4a5a80', fontsize=8)
                        for sp in ax_t.spines.values(): sp.set_visible(False)
                        ax_t.set_title('Distancia por tercio del campo', color='#8a9bbf', fontsize=9)
                        plt.tight_layout()
                        buf_t = io.BytesIO()
                        fig_t.savefig(buf_t, format='png', facecolor='#060d1a', bbox_inches='tight', dpi=110)
                        buf_t.seek(0); plt.close(fig_t)
                        st.image(buf_t, width="stretch")
                    st.markdown('</div>', unsafe_allow_html=True)

            # ── Tabla completa ──
            st.divider()
            _section("📋 Detalle por jugador")
            _cls_colors = {
                'player_home': 'color: #ff6b88', 'player_away': 'color: #60b8ff',
                'gk_home': 'color: #ff88ff',     'gk_away': 'color: #00ced1',
                'referee': 'color: #00ff7f',
            }
            col_names = {
                'gid': 'ID', 'cls': 'Clase',
                'dist_m': 'Dist. total', 'dist_def_m': 'Def.',
                'dist_mid_m': 'Med.', 'dist_att_m': 'Ata.',
                'sprint_count': 'Sprints', 'sprint_dist_m': 'Dist. sprint',
                'sprint_time_s': 'T. sprint', 'hsr_dist_m': 'Dist. HSR',
                'max_speed_ms': 'Vel. máx. (km/h)', 'avg_speed_ms': 'Vel. med. (km/h)',
                'time_s': 'Tiempo',
            }
            # Solo mostrar columnas conocidas (excluir sprint_peak_* y otras internas)
            _display_cols = [c for c in col_names if c in phys.columns]
            phys_display = phys[_display_cols].rename(columns=col_names)
            styled_phys = (phys_display.style
                           .map(lambda v: _cls_colors.get(v, 'color:#888'), subset=[col_names.get('cls', 'cls')])
                           .background_gradient(subset=[col_names.get('dist_m', 'dist_m')], cmap='YlOrRd')
                           .background_gradient(subset=[col_names.get('sprint_count', 'sprint_count')], cmap='Blues')
                           .format({
                               col_names.get('dist_m',        'dist_m'):        '{:.0f}m',
                               col_names.get('dist_def_m',    'dist_def_m'):    '{:.0f}m',
                               col_names.get('dist_mid_m',    'dist_mid_m'):    '{:.0f}m',
                               col_names.get('dist_att_m',    'dist_att_m'):    '{:.0f}m',
                               col_names.get('sprint_dist_m', 'sprint_dist_m'): '{:.0f}m',
                               col_names.get('hsr_dist_m',    'hsr_dist_m'):    '{:.0f}m',
                               col_names.get('max_speed_ms',  'max_speed_ms'):  '{:.1f}',
                               col_names.get('avg_speed_ms',  'avg_speed_ms'):  '{:.1f}',
                               col_names.get('time_s',        'time_s'):        '{:.1f}s',
                               col_names.get('sprint_time_s', 'sprint_time_s'): '{:.1f}s',
                           }, na_rep='-'))
            st.dataframe(styled_phys, width="stretch", height=420)

            # ── Gráfico: sprints por jugador top-20 ──
            top20 = phys.nlargest(20, 'sprint_count')
            fig_sp, ax_sp = plt.subplots(figsize=(12, 5), facecolor='#060d1a')
            ax_sp.set_facecolor('#0c1830')
            colors_sp = [cfg.get('home_dot_hex','#dc143c')
                         if c in ('player_home','gk_home') else
                         cfg.get('away_dot_hex','#1e90ff')
                         for c in top20['cls']]
            ax_sp.bar(range(len(top20)), top20['sprint_count'], color=colors_sp, width=0.7)
            ax_sp.set_xticks(range(len(top20)))
            ax_sp.set_xticklabels([f"#{g}" for g in top20['gid']], rotation=45, ha='right',
                                   color='#8a9bbf', fontsize=8)
            ax_sp.set_ylabel('N.º sprints', color='#4a5a80')
            ax_sp.set_title('Sprints por jugador (top 20)', color='white', fontweight='bold')
            ax_sp.tick_params(colors='#8a9bbf')
            for sp in ax_sp.spines.values(): sp.set_color('#1a3060')
            home_patch = mpatches.Patch(color=cfg.get('home_dot_hex','#dc143c'), label=home_team)
            away_patch = mpatches.Patch(color=cfg.get('away_dot_hex','#1e90ff'), label=away_team)
            ax_sp.legend(handles=[home_patch, away_patch], facecolor='#0c1830',
                         labelcolor='white', fontsize=9)
            plt.tight_layout()
            buf_sp = io.BytesIO()
            fig_sp.savefig(buf_sp, format='png', facecolor='#060d1a', bbox_inches='tight', dpi=120)
            buf_sp.seek(0); plt.close(fig_sp)
            st.image(buf_sp, width="stretch")

            # ── Galería de sprints: crop del jugador en su frame de vel. máxima ──
            _section("🎬 Galería de sprints — imagen en vel. máxima")
            st.caption("Top jugadores por velocidad máxima. Imagen extraída del frame panorámico.")
            _vid_pan = st.session_state.get('vid_pan')
            _H_a     = st.session_state.get('H_a')
            _H_b     = st.session_state.get('H_b')

            @st.cache_data(show_spinner=False)
            def _sprint_crop(vid_path, frame_idx, x_m, y_m, H_a_flat, H_b_flat, cam):
                H_a_arr = np.array(H_a_flat, dtype=np.float64).reshape(3, 3)
                H_b_arr = np.array(H_b_flat, dtype=np.float64).reshape(3, 3)

                cap = cv2.VideoCapture(vid_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return None

                half_h = frame.shape[0] // 2

                def _try_cam(c):
                    raw_half = frame[:half_h] if c == 'A' else frame[half_h:]
                    side = 'left' if c == 'A' else 'right'
                    try:
                        proc_half = core.preprocess_half(raw_half, side)
                    except Exception:
                        proc_half = raw_half
                    H = H_a_arr if c == 'A' else H_b_arr
                    try:
                        H_inv = np.linalg.inv(H)
                        pts = cv2.perspectiveTransform(
                            np.array([[[float(x_m), float(y_m)]]], dtype=np.float32), H_inv)
                        px, py = int(pts[0, 0, 0]), int(pts[0, 0, 1])
                    except Exception:
                        return None
                    h_img, w_img = proc_half.shape[:2]
                    if not (10 <= px < w_img - 10 and 20 <= py < h_img - 5):
                        return None
                    cw, ch_up, ch_dn = 45, 120, 15
                    x0 = max(0, px - cw);    x1 = min(w_img, px + cw)
                    y0 = max(0, py - ch_up); y1 = min(h_img, py + ch_dn)
                    crop = proc_half[y0:y1, x0:x1]
                    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
                        return None
                    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                # Intentar en orden: cam preferida primero, luego la otra
                if cam == 'A':
                    order = ['A', 'B']
                elif cam == 'B':
                    order = ['B', 'A']
                else:  # 'AB', '?' — probar ambas
                    order = ['A', 'B']
                for c in order:
                    result = _try_cam(c)
                    if result is not None:
                        return result
                return None

            _has_peak_cols = 'sprint_peak_frame' in phys.columns
            if _vid_pan and _H_a is not None and _H_b is not None and _has_peak_cols:
                _top_sprinters = phys.nlargest(min(16, len(phys)), 'max_speed_ms')
                try:
                    _H_a_flat = np.array(_H_a, dtype=np.float64).flatten().tolist()
                    _H_b_flat = np.array(_H_b, dtype=np.float64).flatten().tolist()
                except Exception:
                    _H_a_flat = _H_b_flat = None
                if _H_a_flat:
                    _cols_sprint = st.columns(8)
                    _shown = 0
                    for _, row in _top_sprinters.iterrows():
                        if _shown >= 16:
                            break
                        try:
                            crop_img = _sprint_crop(
                                _vid_pan,
                                int(row['sprint_peak_frame']),
                                float(row['sprint_peak_x']),
                                float(row['sprint_peak_y']),
                                _H_a_flat, _H_b_flat,
                                str(row['sprint_peak_cam']),
                            )
                        except Exception:
                            crop_img = None
                        _col = _cols_sprint[_shown % 8]
                        with _col:
                            if crop_img is not None:
                                st.image(crop_img, width="stretch")
                            else:
                                st.markdown('<div style="height:80px;background:#0c1830;'
                                            'border-radius:4px;"></div>', unsafe_allow_html=True)
                            _team_col = cfg.get('home_dot_hex','#dc143c') if row['cls'] in ('player_home','gk_home') else cfg.get('away_dot_hex','#1e90ff')
                            st.markdown(
                                f'<div style="text-align:center;font-size:10px;color:{_team_col};">'
                                f'#{int(row["gid"])} · {row["max_speed_ms"]:.1f}km/h</div>',
                                unsafe_allow_html=True)
                        _shown += 1
            elif not _has_peak_cols:
                st.info("Relanza el análisis para ver los crops (datos de sprint actualizados).")
            else:
                st.info("Carga el vídeo panorámico y las homografías para ver los crops.")

    # ════════════════════════════════════════════════════════
    #  Táctico
    # ════════════════════════════════════════════════════════
    with tab_tact:
        if not len(df):
            st.markdown('<div class="empty-state"><div class="icon">🧠</div>'
                        '<h3>Sin datos</h3></div>', unsafe_allow_html=True)
        else:
            tact  = core.compute_tactical_stats(frame_records, ball_records, fps, _FL, _FW, cfg=cfg_with_teams)
            zones = core.compute_zone_stats(frame_records, _FL, _FW, cfg=cfg_with_teams)
            pf    = tact.get('per_frame', pd.DataFrame())

            # ── KPI tácticos ──
            kpi_tact = [
                ("home_avg_compact",  f"{tact.get('home_avg_compact','?')}m",
                 f"Compacidad {home_team}",
                 "Dispersión media del equipo en metros. <15m = bloque cerrado."),
                ("away_avg_compact",  f"{tact.get('away_avg_compact','?')}m",
                 f"Compacidad {away_team}", ""),
                ("home_avg_def_line", f"{tact.get('home_avg_def_line','?')}m",
                 f"Línea def. {home_team}",
                 f"Metros del 4º defensor desde su portería propia. >{round(_FL/2,0):.0f}m = línea alta (mitad del campo)."),
                ("away_avg_def_line", f"{tact.get('away_avg_def_line','?')}m",
                 f"Línea def. {away_team}",
                 f"Metros del 4º defensor desde su portería propia. >{round(_FL/2,0):.0f}m = línea alta (mitad del campo)."),
                ("home_avg_width",    f"{tact.get('home_avg_width','?')}m",
                 f"Amplitud {home_team}",
                 "Rango lateral del equipo. >45m = equipo muy abierto."),
                ("away_avg_width",    f"{tact.get('away_avg_width','?')}m",
                 f"Amplitud {away_team}", ""),
                ("possession_home_pct", f"{tact.get('possession_home_pct','?')}%",
                 f"Posesión {home_team}",
                 "% de frames con jugador más cercano al balón."),
                ("possession_away_pct", f"{tact.get('possession_away_pct','?')}%",
                 f"Posesión {away_team}", ""),
                ("territory_home_pct", f"{tact.get('territory_home_pct','?')}%",
                 f"Territorio {home_team}",
                 "% de momentos con más jugadores en campo rival."),
                ("territory_away_pct", f"{tact.get('territory_away_pct','?')}%",
                 f"Territorio {away_team}", ""),
            ]
            if 'avg_pressing_dist' in tact:
                kpi_tact.append(("avg_pressing_dist",
                                 f"{tact['avg_pressing_dist']}m",
                                 "Pressing medio (3 más cercanos al balón)",
                                 "<10m = press muy alto."))

            cols_tact = st.columns(4)
            for i, (_, val, label, tip) in enumerate(kpi_tact):
                with cols_tact[i % 4]:
                    help_txt = tip if tip else None
                    st.metric(label, val, help=help_txt)

            st.divider()

            # t_min compartido por todos los gráficos temporales
            if len(pf):
                pf['t_min'] = (pf['frame'] - pf['frame'].min()) / fps / 60

            # ── Centroide a lo largo del tiempo ──
            if len(pf) and 'home_cx_norm' in pf.columns:
                _section("📈 Avance del centroide (perspectiva de ataque de cada equipo)")

                fig_c, ax_c = plt.subplots(figsize=(14, 3.5), facecolor='#060d1a')
                ax_c.set_facecolor('#0c1830')
                ax_c.plot(pf['t_min'], pf['home_cx_norm'].rolling(30, min_periods=1).mean(),
                          color=cfg.get('home_dot_hex','#dc143c'), lw=1.5, label=home_team)
                ax_c.plot(pf['t_min'], pf['away_cx_norm'].rolling(30, min_periods=1).mean(),
                          color=cfg.get('away_dot_hex','#1e90ff'), lw=1.5, label=away_team)
                ax_c.axhline(_FL / 2, color='white', lw=0.8, linestyle='--', alpha=0.4,
                             label='Mitad campo')
                ax_c.fill_between(pf['t_min'],
                                  pf['home_cx_norm'].rolling(30, min_periods=1).mean(),
                                  _FL / 2,
                                  where=pf['home_cx_norm'].rolling(30, min_periods=1).mean() > _FL / 2,
                                  alpha=0.08, color=cfg.get('home_dot_hex','#dc143c'))
                ax_c.set_xlabel('Tiempo (min)', color='#4a5a80')
                ax_c.set_ylabel('Posición (m) → portería rival', color='#4a5a80')
                ax_c.tick_params(colors='#8a9bbf')
                ax_c.legend(facecolor='#0c1830', labelcolor='white', fontsize=9)
                ax_c.set_title(
                    'Centroide de equipo — 0=portería propia · FL=portería rival (suavizado 30f)',
                    color='white', fontsize=10)
                for sp in ax_c.spines.values(): sp.set_color('#1a3060')
                plt.tight_layout()
                buf_c = io.BytesIO()
                fig_c.savefig(buf_c, format='png', facecolor='#060d1a', bbox_inches='tight', dpi=120)
                buf_c.seek(0); plt.close(fig_c)
                st.image(buf_c, width="stretch")

            # ── Compacidad a lo largo del tiempo ──
            if len(pf) and 'home_compact' in pf.columns:
                _section("📐 Compacidad del equipo a lo largo del tiempo")
                fig_k, ax_k = plt.subplots(figsize=(14, 3), facecolor='#060d1a')
                ax_k.set_facecolor('#0c1830')
                ax_k.plot(pf['t_min'], pf['home_compact'].rolling(30, min_periods=1).mean(),
                          color=cfg.get('home_dot_hex','#dc143c'), lw=1.5, label=home_team)
                ax_k.plot(pf['t_min'], pf['away_compact'].rolling(30, min_periods=1).mean(),
                          color=cfg.get('away_dot_hex','#1e90ff'), lw=1.5, label=away_team)
                ax_k.set_xlabel('Tiempo (min)', color='#4a5a80')
                ax_k.set_ylabel('Compacidad (m)', color='#4a5a80')
                ax_k.tick_params(colors='#8a9bbf')
                ax_k.legend(facecolor='#0c1830', labelcolor='white', fontsize=9)
                for sp in ax_k.spines.values(): sp.set_color('#1a3060')
                plt.tight_layout()
                buf_k = io.BytesIO()
                fig_k.savefig(buf_k, format='png', facecolor='#060d1a', bbox_inches='tight', dpi=110)
                buf_k.seek(0); plt.close(fig_k)
                st.image(buf_k, width="stretch")

            # ── Presión por equipo a lo largo del tiempo ──
            if len(pf) and 'home_in_att_half' in pf.columns:
                _section("⚡ Presión ofensiva — jugadores en campo rival")
                fig_p, ax_p = plt.subplots(figsize=(14, 3), facecolor='#060d1a')
                ax_p.set_facecolor('#0c1830')
                _roll = 60
                h_press = pf['home_in_att_half'].rolling(_roll, min_periods=1).mean()
                a_press = pf['away_in_att_half'].rolling(_roll, min_periods=1).mean()
                ax_p.fill_between(pf['t_min'], h_press, alpha=0.35,
                                  color=cfg.get('home_dot_hex','#dc143c'), label=home_team)
                ax_p.fill_between(pf['t_min'], a_press, alpha=0.35,
                                  color=cfg.get('away_dot_hex','#1e90ff'), label=away_team)
                ax_p.plot(pf['t_min'], h_press, color=cfg.get('home_dot_hex','#dc143c'), lw=1.5)
                ax_p.plot(pf['t_min'], a_press, color=cfg.get('away_dot_hex','#1e90ff'), lw=1.5)
                ax_p.set_xlabel('Tiempo (min)', color='#4a5a80')
                ax_p.set_ylabel('Jugadores en campo rival', color='#4a5a80')
                ax_p.tick_params(colors='#8a9bbf')
                ax_p.legend(facecolor='#0c1830', labelcolor='white', fontsize=9)
                ax_p.set_title('Presión ofensiva (suavizado 60 frames)', color='white', fontsize=10)
                for sp in ax_p.spines.values(): sp.set_color('#1a3060')
                plt.tight_layout()
                buf_p = io.BytesIO()
                fig_p.savefig(buf_p, format='png', facecolor='#060d1a', bbox_inches='tight', dpi=110)
                buf_p.seek(0); plt.close(fig_p)
                st.image(buf_p, width="stretch")

            # ── Mapa de posición por zona (bloque de cada equipo) ──
            if zones:
                _section("🟥🟦 Posición por zona — bloque de cada equipo")
                st.caption("Intensidad = % de las propias detecciones acumuladas en cada zona (muestra dónde se sitúa el bloque)")

                home_col = cfg.get('home_dot_hex', '#dc143c')
                away_col = cfg.get('away_dot_hex', '#1e90ff')

                _hal = cfg.get('home_attacks_left', True)
                home_attacks_right = _hal
                away_attacks_right = not _hal

                # Cuadrícula 5×3 (misma que compute_zone_stats)
                n_cols_z, n_rows_z = 5, 3
                col_w_z = _FL / n_cols_z
                row_h_z = _FW / n_rows_z
                zone_coords = {
                    f'r{r}c{c}': (c * col_w_z, (c+1) * col_w_z,
                                  r * row_h_z, (r+1) * row_h_z)
                    for r in range(n_rows_z) for c in range(n_cols_z)
                }

                def _draw_zone_pitch(title, team_col, density_key, attacks_right):
                    densities = [zones.get(k, {}).get(density_key, 0.0)
                                 for k in zone_coords]
                    max_d = max(densities) if densities else 1.0

                    fig_z, ax_z = plt.subplots(figsize=(9, 5.6), facecolor='#060d1a')
                    ax_z.set_facecolor('#060d1a')
                    # draw_pitch_mpl fija ylim invertido: y=0 arriba, y=_FW abajo
                    core.draw_pitch_mpl(ax_z, _FL, _FW)

                    for zname, (x0, x1, y0, y1) in zone_coords.items():
                        density = zones.get(zname, {}).get(density_key, 0.0)
                        alpha   = min(density / max(max_d, 0.01), 1.0) * 0.75
                        rect    = plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                                facecolor=team_col, alpha=alpha, zorder=3)
                        ax_z.add_patch(rect)
                        cx, cy = (x0+x1)/2, (y0+y1)/2
                        if density >= 1.0:
                            ax_z.text(cx, cy, f"{density:.1f}%",
                                      ha='center', va='center', fontsize=8, fontweight='bold',
                                      color='white', zorder=6)

                    # Etiquetas de portería — y=_FW/2 está en el centro vertical del campo
                    mid_y = _FW / 2
                    goal_def_x = 0.0  if attacks_right else _FL
                    goal_atk_x = _FL  if attacks_right else 0.0
                    ax_z.text(goal_def_x, mid_y, "[ DEF ]",
                              ha='center', va='center', fontsize=7.5, fontweight='bold',
                              color='#bbbbbb', zorder=7,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', alpha=0.85, edgecolor='#555', linewidth=0.8))
                    ax_z.text(goal_atk_x, mid_y, "[ ATK ]",
                              ha='center', va='center', fontsize=7.5, fontweight='bold',
                              color=team_col, zorder=7,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', alpha=0.85, edgecolor=team_col, linewidth=1.2))

                    # Flecha dirección ataque — en el sistema invertido y<0 queda ARRIBA del campo
                    arrow_y = -6.5
                    if attacks_right:
                        ax_z.annotate("", xy=(_FL * 0.78, arrow_y), xytext=(_FL * 0.22, arrow_y),
                                      arrowprops=dict(arrowstyle='->', color=team_col, lw=2.2))
                        ax_z.text(_FL / 2, arrow_y, "ATACA  →", ha='center', va='center',
                                  fontsize=8.5, color=team_col, fontweight='bold')
                    else:
                        ax_z.annotate("", xy=(_FL * 0.22, arrow_y), xytext=(_FL * 0.78, arrow_y),
                                      arrowprops=dict(arrowstyle='->', color=team_col, lw=2.2))
                        ax_z.text(_FL / 2, arrow_y, "←  ATACA", ha='center', va='center',
                                  fontsize=8.5, color=team_col, fontweight='bold')

                    # Extender ylim hacia negativo para mostrar la flecha (sistema invertido)
                    ax_z.set_xlim(-5, _FL + 5)
                    ax_z.set_ylim(_FW + 2, -10)
                    ax_z.set_title(title, color='white', fontsize=10, fontweight='bold', pad=4)
                    plt.tight_layout()
                    buf = io.BytesIO()
                    fig_z.savefig(buf, format='png', facecolor='#060d1a', bbox_inches='tight', dpi=130)
                    buf.seek(0); plt.close(fig_z)
                    return buf

                col_zh, col_za = st.columns(2, gap="medium")
                with col_zh:
                    buf_h = _draw_zone_pitch(
                        f"{home_team} — bloque posicional",
                        home_col, 'home_density', home_attacks_right,
                    )
                    st.image(buf_h, width="stretch")
                with col_za:
                    buf_a = _draw_zone_pitch(
                        f"{away_team} — bloque posicional",
                        away_col, 'away_density', away_attacks_right,
                    )
                    st.image(buf_a, width="stretch")

                # Asimetría lateral
                if 'home_avg_y_bias' in tact:
                    def _bias_str(bias):
                        if bias > 0.5:  return f"+{bias:.1f}m → derecha"
                        if bias < -0.5: return f"{bias:.1f}m → izquierda"
                        return "centrado"
                    st.caption(
                        f"Asimetría lateral — "
                        f"{home_team}: {_bias_str(tact['home_avg_y_bias'])}  |  "
                        f"{away_team}: {_bias_str(tact['away_avg_y_bias'])}"
                    )

            # ── Dominancia territorial compuesta [-1, +1] ──────────────
            if len(frame_records):
                _section("⚖️ Dominancia territorial — índice compuesto [-1, +1]")
                st.caption(
                    "Territorio (40%) + 3er tercio (35%) + Área (15%) + Balón (10%) · "
                    "Positivo = local domina · Negativo = visitante domina"
                )

                _AREA_D   = 16.5
                _THIRD_W  = _FL / 3.0
                _W_T, _W_A3, _W_B, _W_BL = 0.40, 0.35, 0.15, 0.10
                _ROLL_D   = 60
                _HOME_CLS_D = {'player_home', 'gk_home'}
                _AWAY_CLS_D = {'player_away', 'gk_away'}
                _hal_d    = cfg.get('home_attacks_left', True)
                _htf_d    = cfg.get('halftime_frame', int(1e9))
                _ball_lut = {br['frame']: br for br in ball_records}

                _df_fr = pd.DataFrame(frame_records)
                _dom_rows = []
                for _fr, _grp in _df_fr.groupby('frame'):
                    _home_high = core._home_attacks_high_x(int(_fr), _hal_d, _htf_d)
                    _hxs = _grp.loc[_grp['cls'].isin(_HOME_CLS_D), 'x_m'].values
                    _axs = _grp.loc[_grp['cls'].isin(_AWAY_CLS_D), 'x_m'].values
                    if not len(_hxs) and not len(_axs):
                        continue
                    _nh, _na = max(len(_hxs), 1), max(len(_axs), 1)
                    _cxh = float(_hxs.mean()) if len(_hxs) else _FL / 2
                    _cxa = float(_axs.mean()) if len(_axs) else _FL / 2

                    if _home_high:
                        _terr = float(np.clip(
                            (_cxh / _FL - 0.5) * 2 - (0.5 - _cxa / _FL) * 2, -1, 1))
                        _atk3 = float(np.clip(
                            (_hxs > 2 * _THIRD_W).sum() / _nh
                            - (_axs < _THIRD_W).sum() / _na, -1, 1))
                        _box  = float(np.clip(
                            (_hxs > _FL - _AREA_D).sum() / _nh
                            - (_axs < _AREA_D).sum() / _na, -1, 1))
                        _nhb  = int((_hxs > _FL - _AREA_D).sum())
                        _nab  = int((_axs < _AREA_D).sum())
                    else:
                        _terr = float(np.clip(
                            (0.5 - _cxh / _FL) * 2 - (_cxa / _FL - 0.5) * 2, -1, 1))
                        _atk3 = float(np.clip(
                            (_hxs < _THIRD_W).sum() / _nh
                            - (_axs > 2 * _THIRD_W).sum() / _na, -1, 1))
                        _box  = float(np.clip(
                            (_hxs < _AREA_D).sum() / _nh
                            - (_axs > _FL - _AREA_D).sum() / _na, -1, 1))
                        _nhb  = int((_hxs < _AREA_D).sum())
                        _nab  = int((_axs > _FL - _AREA_D).sum())

                    _has_ball = int(_fr) in _ball_lut
                    if _has_ball:
                        _bx = _ball_lut[int(_fr)]['x_m']
                        _bv = float(np.clip(
                            (_bx / _FL - 0.5) * 2 if _home_high
                            else (0.5 - _bx / _FL) * 2, -1, 1))
                    else:
                        _bv = 0.0

                    _wb = _W_BL if _has_ball else 0.0
                    _dom = _W_T * _terr + _W_A3 * _atk3 + _W_B * _box + _wb * _bv
                    if _wb == 0.0:
                        _dom /= (_W_T + _W_A3 + _W_B)
                    _dom_rows.append({
                        'frame': int(_fr),
                        't_min': (int(_fr) - int(_df_fr['frame'].min())) / fps / 60,
                        'dom':   round(float(_dom), 4),
                        'terr':  round(_terr, 3),
                        'atk3':  round(_atk3, 3),
                        'box':   round(_box,  3),
                        'ball':  round(_bv,   3),
                        'n_h_box': _nhb,
                        'n_a_box': _nab,
                    })

                if _dom_rows:
                    _dfdom = pd.DataFrame(_dom_rows)
                    _dfdom['dom_s'] = (_dfdom['dom']
                                       .rolling(_ROLL_D, min_periods=1, center=True).mean())

                    _dom_pct_h = float((_dfdom['dom'] > 0).mean() * 100)
                    _dom_pct_a = 100.0 - _dom_pct_h
                    _mean_idx  = float(_dfdom['dom'].mean())
                    _winner    = home_team if _mean_idx > 0 else away_team

                    _hc = cfg.get('home_dot_hex', '#dc143c')
                    _ac = cfg.get('away_dot_hex', '#1e90ff')
                    _DK = '#060d1a'
                    _AX = '#0a0f1a'

                    _fig_dom = plt.figure(figsize=(16, 11))
                    _fig_dom.patch.set_facecolor(_DK)
                    _gs = _fig_dom.add_gridspec(3, 1,
                                                height_ratios=[3, 1.2, 1.2], hspace=0.12)

                    # Panel 1: barras de dominancia
                    _ax0 = _fig_dom.add_subplot(_gs[0])
                    _ax0.set_facecolor(_AX)
                    for _sp in _ax0.spines.values(): _sp.set_color('#222')
                    _t   = _dfdom['t_min'].values
                    _d   = _dfdom['dom'].values
                    _s   = _dfdom['dom_s'].values
                    _dt  = float(np.diff(_t).mean()) if len(_t) > 1 else (1.0 / fps / 60)
                    _cols = np.where(_d >= 0, _hc, _ac)
                    _ax0.bar(_t, _d, width=_dt * 0.85, color=_cols,
                             alpha=0.55, align='center', zorder=2)
                    _ax0.plot(_t, _s, color='white', lw=2.0, zorder=4, alpha=0.9,
                              label='Tendencia suavizada')
                    _ax0.fill_between(_t, _s, 0, where=(_s >= 0),
                                      alpha=0.18, color=_hc, zorder=3)
                    _ax0.fill_between(_t, _s, 0, where=(_s <  0),
                                      alpha=0.18, color=_ac, zorder=3)
                    _ax0.axhline(0, color='white', lw=1.2, alpha=0.5, zorder=3)
                    _ax0.set_ylim(-1.05, 1.05)
                    _ax0.set_xlim(_t[0] - _dt, _t[-1] + _dt)
                    _ax0.set_ylabel('Índice de dominancia', color='#aaa', fontsize=10)
                    _ax0.tick_params(colors='#aaa', labelbottom=False)
                    _ax0.grid(axis='y', alpha=0.1, color='#555')
                    _ax0.text(0.01, 0.97, f'▲ {home_team} domina',
                              transform=_ax0.transAxes, color=_hc,
                              fontsize=9, fontweight='bold', va='top')
                    _ax0.text(0.01, 0.03, f'▼ {away_team} domina',
                              transform=_ax0.transAxes, color=_ac,
                              fontsize=9, fontweight='bold', va='bottom')
                    _ax0.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
                                edgecolor='#333', loc='upper right', framealpha=0.8)

                    # Panel 2: jugadores en área
                    _ax1 = _fig_dom.add_subplot(_gs[1], sharex=_ax0)
                    _ax1.set_facecolor(_AX)
                    for _sp in _ax1.spines.values(): _sp.set_color('#222')
                    _nh_r = _dfdom['n_h_box'].rolling(_ROLL_D, min_periods=1, center=True).mean()
                    _na_r = _dfdom['n_a_box'].rolling(_ROLL_D, min_periods=1, center=True).mean()
                    _ax1.fill_between(_t, _nh_r, alpha=0.5, color=_hc,
                                      label=f'{home_team} en área rival')
                    _ax1.fill_between(_t, _na_r, alpha=0.5, color=_ac,
                                      label=f'{away_team} en área rival')
                    _ax1.plot(_t, _nh_r, color=_hc, lw=1.2)
                    _ax1.plot(_t, _na_r, color=_ac, lw=1.2)
                    _ax1.set_ylabel('Jugadores\nen área', color='#aaa', fontsize=8)
                    _ax1.tick_params(colors='#aaa', labelbottom=False, labelsize=8)
                    _ax1.grid(alpha=0.1, color='#555')
                    _ax1.legend(fontsize=7.5, facecolor='#1a1a2e', labelcolor='white',
                                edgecolor='#333', loc='upper right', framealpha=0.8, ncol=2)
                    _ax1.set_ylim(bottom=0)

                    # Panel 3: desglose de componentes
                    _ax2 = _fig_dom.add_subplot(_gs[2], sharex=_ax0)
                    _ax2.set_facecolor(_AX)
                    for _sp in _ax2.spines.values(): _sp.set_color('#222')
                    _rl = lambda col: _dfdom[col].rolling(_ROLL_D, min_periods=1, center=True).mean()
                    _ax2.plot(_t, _rl('terr'), color='#22c55e', lw=1.2, alpha=0.85, label='Territorio')
                    _ax2.plot(_t, _rl('atk3'), color='#f59e0b', lw=1.2, alpha=0.85, label='3er tercio')
                    _ax2.plot(_t, _rl('box'),  color='#ef4444', lw=1.2, alpha=0.85, label='Área')
                    if _dfdom['ball'].abs().sum() > 0:
                        _ax2.plot(_t, _rl('ball'), color='#a855f7', lw=1.2, alpha=0.85,
                                  label='Balón')
                    _ax2.axhline(0, color='white', lw=0.8, alpha=0.4)
                    _ax2.set_ylim(-1.05, 1.05)
                    _ax2.set_ylabel('Componentes', color='#aaa', fontsize=8)
                    _ax2.set_xlabel('Tiempo (min)', color='#aaa', fontsize=9)
                    _ax2.tick_params(colors='#aaa', labelsize=8)
                    _ax2.grid(alpha=0.1, color='#555')
                    _ax2.legend(fontsize=7.5, facecolor='#1a1a2e', labelcolor='white',
                                edgecolor='#333', loc='upper right', framealpha=0.8, ncol=4)

                    _fig_dom.suptitle(
                        f'Dominancia territorial  —  '
                        f'{home_team} {_dom_pct_h:.0f}%  vs  {away_team} {_dom_pct_a:.0f}%  '
                        f'│  Índice medio: {_mean_idx:+.2f}  ({_winner} domina)',
                        color='white', fontsize=12, fontweight='bold'
                    )
                    plt.tight_layout()
                    _buf_dom = io.BytesIO()
                    _fig_dom.savefig(_buf_dom, format='png', facecolor=_DK,
                                     bbox_inches='tight', dpi=130)
                    _buf_dom.seek(0); plt.close(_fig_dom)
                    st.image(_buf_dom, width="stretch")

                    _cd1, _cd2, _cd3 = st.columns(3)
                    with _cd1:
                        st.metric(f"Dominio {home_team}", f"{_dom_pct_h:.1f}%")
                    with _cd2:
                        st.metric(f"Dominio {away_team}", f"{_dom_pct_a:.1f}%")
                    with _cd3:
                        st.metric("Índice medio", f"{_mean_idx:+.3f}",
                                  help="Positivo = local domina · Negativo = visitante domina")

    # ════════════════════════════════════════════════════════
    #  Análisis IA
    # ════════════════════════════════════════════════════════
    with tab_ai:
        st.markdown(
            '<div class="card-accent">'
            '<div style="font-size:15px;font-weight:700;color:#e8edf8;margin-bottom:6px;">'
            '🤖 Análisis táctico con IA</div>'
            '<div style="font-size:13px;color:#8a9bbf;">'
            'El sistema calcula automáticamente las métricas físicas y tácticas del partido '
            'y se las envía a Claude para que genere un informe táctico completo con '
            'recomendaciones sobre cómo atacar a cada equipo.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            st.warning("⚠️ Variable `ANTHROPIC_API_KEY` no encontrada. Añádela al archivo `.env` en la raíz del proyecto.")

        # Preview del prompt que se enviará
        with st.expander("👁️ Ver datos que se enviarán al modelo", expanded=False):
            if len(df):
                phys_prev   = core.compute_physical_stats(frame_records, fps, _FL, cfg=cfg_with_teams)
                tact_prev   = core.compute_tactical_stats(frame_records, ball_records, fps, _FL, _FW, cfg=cfg_with_teams)
                zones_prev  = core.compute_zone_stats(frame_records, _FL, _FW, cfg=cfg_with_teams)
                lineup_prev = core.compute_lineup_and_formation(frame_records, fps, _FL, cfg=cfg_with_teams)
                prompt_prev = core.build_ai_prompt(phys_prev, tact_prev, zones_prev,
                                                    cfg_with_teams, fps, lineup_prev)
                st.code(prompt_prev, language="text")
            else:
                st.info("Procesa un segmento primero.")

        col_btn_ai, col_model = st.columns([2, 3])
        with col_model:
            ai_model = st.selectbox(
                "Modelo",
                ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
                index=1,
                help="Opus = mejor análisis, más lento. Haiku = más rápido.",
            )

        ai_result_key = 'ai_analysis_result'
        st.session_state.setdefault(ai_result_key, None)

        with col_btn_ai:
            if st.button("🚀 Generar análisis IA", type="primary",
                         width="stretch",
                         disabled=not bool(api_key and len(df))):
                with st.spinner("Analizando el partido con IA…"):
                    try:
                        phys_ai   = core.compute_physical_stats(frame_records, fps, _FL, cfg=cfg_with_teams)
                        tact_ai   = core.compute_tactical_stats(frame_records, ball_records, fps, _FL, _FW, cfg=cfg_with_teams)
                        zones_ai  = core.compute_zone_stats(frame_records, _FL, _FW, cfg=cfg_with_teams)
                        lineup_ai = core.compute_lineup_and_formation(frame_records, fps, _FL, cfg=cfg_with_teams)

                        import anthropic as _ant
                        _client = _ant.Anthropic(api_key=api_key)
                        _prompt = core.build_ai_prompt(phys_ai, tact_ai, zones_ai,
                                                       cfg_with_teams, fps, lineup_ai)
                        _msg = _client.messages.create(
                            model=ai_model,
                            max_tokens=6000,
                            messages=[{"role": "user", "content": _prompt}],
                        )
                        st.session_state[ai_result_key] = _msg.content[0].text
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.get(ai_result_key):
            st.divider()
            _ai_render_report(
                st.session_state[ai_result_key],
                home_team, away_team,
            )
            st.download_button(
                "⬇️  Descargar informe (.md)",
                data=st.session_state[ai_result_key].encode('utf-8'),
                file_name=f"analisis_tactico_{home_team}_vs_{away_team}.md",
                mime="text/markdown",
            )

    # ════════════════════════════════════════════════════════
    #  Exportar
    # ════════════════════════════════════════════════════════
    with tab_export:
        _section("💾 Descargas")
        col_e1, col_e2, col_e3 = st.columns(3, gap="large")

        with col_e1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            _section("📄 Trayectorias")
            if len(df):
                match = st.session_state.match_name or "match"
                st.download_button(
                    "⬇️  CSV trayectorias completo",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name=f"tracking_{match}.csv",
                    mime="text/csv", width="stretch",
                )
                if ball_records:
                    st.download_button(
                        "⬇️  CSV posiciones del balón",
                        data=pd.DataFrame(ball_records).to_csv(index=False).encode('utf-8'),
                        file_name=f"ball_{match}.csv",
                        mime="text/csv", width="stretch",
                    )
            else:
                st.info("Sin datos.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_e2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            _section("📏 Estadísticas")
            if len(df):
                match = st.session_state.match_name or "match"
                phys_exp  = core.compute_physical_stats(frame_records, fps, _FL, cfg=cfg_with_teams)
                st.download_button(
                    "⬇️  CSV estadísticas físicas",
                    data=phys_exp.to_csv(index=False).encode('utf-8'),
                    file_name=f"stats_fisico_{match}.csv",
                    mime="text/csv", width="stretch",
                )
                zones_exp = core.compute_zone_stats(frame_records, _FL, _FW, cfg=cfg_with_teams)
                if zones_exp:
                    zones_rows = [{'zona': k, **v} for k, v in zones_exp.items()]
                    st.download_button(
                        "⬇️  CSV dominancia zonal",
                        data=pd.DataFrame(zones_rows).to_csv(index=False).encode('utf-8'),
                        file_name=f"stats_zonas_{match}.csv",
                        mime="text/csv", width="stretch",
                    )
            else:
                st.info("Sin datos.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_e3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            _section("🎬 Vídeo con overlay")
            out_vid = cfg.get('output_video_path')
            if out_vid and os.path.exists(out_vid):
                sz_mb = os.path.getsize(out_vid) / 1e6
                st.markdown(
                    f'<div class="card-green">'
                    f'<div style="font-size:12px;color:#00e676;font-weight:700;">✅ Generado</div>'
                    f'<code style="font-size:11px;color:#5aff8a;">{Path(out_vid).name}</code><br>'
                    f'<span style="font-size:12px;color:#4a5a80;">{sz_mb:.1f} MB</span>'
                    f'</div>', unsafe_allow_html=True,
                )
                with open(out_vid, 'rb') as vf:
                    st.download_button("⬇️  Descargar vídeo", data=vf.read(),
                                       file_name=Path(out_vid).name, mime="video/mp4",
                                       width="stretch")
            elif out_vid:
                st.info(f"Vídeo en proceso…")
            else:
                st.caption("No se generó vídeo.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        if st.button("🔄  Nuevo análisis", width="content"):
            st.session_state.progress    = {}
            st.session_state.proc_thread = None
            st.session_state[ai_result_key] = None
            st.session_state.step        = 3
            st.rerun()


# ══════════════════════════════════════════════════════════
#  ROUTING
# ══════════════════════════════════════════════════════════

s = st.session_state.step
if   s == 1: step_download()
elif s == 2: step_setup()
elif s == 3: step_process()
elif s == 4: step_results()
