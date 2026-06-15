import os
import shutil
from roboflow import Roboflow


rf = Roboflow(api_key="fjEDPLJZxMQXJMWH3af0")
project = rf.workspace("javiers-workspace-osouu").project("panoramic-cam")
version = project.version(1)
dataset = version.download("yolo26")
                
                

# ─── CONFIG ────────────────────────────────────────────────────────────────
BASE_PATH = dataset.location
SPLITS = ["train", "valid"]

BALL_PATH    = BASE_PATH + "_ball"
PLAYERS_PATH = BASE_PATH + "_players"


# ─── CREAR ESTRUCTURA ──────────────────────────────────────────────────────
def create_structure(base_out):
    for split in SPLITS:
        os.makedirs(os.path.join(base_out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_out, split, "labels"), exist_ok=True)


# ─── SEGMENTACIÓN → BBOX ───────────────────────────────────────────────────
def segment_to_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    return x, y, w, h


# ─── PROCESAR LABELS ───────────────────────────────────────────────────────
def process_labels(src_label_path, target_class, name=""):

    if not os.path.exists(src_label_path):
        return []

    with open(src_label_path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:
            cls = int(parts[0])
        except:
            continue

        if cls != target_class:
            continue

        # ─── CASO 1: BBOX ─────────────────────────
        if len(parts) == 5:
            try:
                x, y, w, h = map(float, parts[1:])
            except:
                continue

        # ─── CASO 2: SEGMENTACIÓN ─────────────────
        else:
            try:
                coords = list(map(float, parts[1:]))

                if len(coords) % 2 != 0:
                    continue

                x, y, w, h = segment_to_bbox(coords)

            except:
                continue

        # ─── VALIDACIÓN GENERAL ───────────────────
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue

        # ─── FILTROS ESPECÍFICOS ──────────────────
        if name == "ball":
            # Eliminar cajas demasiado grandes (error de anotación)
            if w > 0.15 or h > 0.15:
                continue

            # Eliminar cajas demasiado pequeñas (ruido)
            if w < 0.003 or h < 0.003:
                continue

        # Convertir SIEMPRE a clase 0
        new_lines.append(f"0 {x} {y} {w} {h}\n")

    return new_lines


# ─── PROCESAR DATASET ──────────────────────────────────────────────────────
def process_dataset():

    print("Preparando datasets...")

    create_structure(BALL_PATH)
    create_structure(PLAYERS_PATH)

    for split in SPLITS:
        print(f"\nProcesando {split}...")

        img_src = os.path.join(BASE_PATH, split, "images")
        lbl_src = os.path.join(BASE_PATH, split, "labels")

        for file in os.listdir(img_src):

            img_path = os.path.join(img_src, file)
            lbl_path = os.path.join(
                lbl_src,
                file.replace(".jpg", ".txt").replace(".png", ".txt")
            )

            # Procesar
            ball_labels   = process_labels(lbl_path, 0, name="ball")
            player_labels = process_labels(lbl_path, 1, name="player")

            # ─── BALÓN ─────────────────────────────
            if len(ball_labels) > 0:
                img_dst = os.path.join(BALL_PATH, split, "images", file)
                lbl_dst = os.path.join(
                    BALL_PATH, split, "labels",
                    os.path.basename(lbl_path)
                )

                shutil.copy(img_path, img_dst)

                with open(lbl_dst, "w") as f:
                    f.writelines(ball_labels)

            # ─── JUGADORES ─────────────────────────
            if len(player_labels) > 0:
                img_dst = os.path.join(PLAYERS_PATH, split, "images", file)
                lbl_dst = os.path.join(
                    PLAYERS_PATH, split, "labels",
                    os.path.basename(lbl_path)
                )

                shutil.copy(img_path, img_dst)

                with open(lbl_dst, "w") as f:
                    f.writelines(player_labels)

    # ─── BORRAR CACHE ─────────────────────────────────
    print("\nEliminando cache...")

    for path in [BALL_PATH, PLAYERS_PATH]:
        for split in SPLITS:
            cache = os.path.join(path, split, "labels.cache")
            if os.path.exists(cache):
                os.remove(cache)

    print("\nDATASETS GENERADOS")


# ─── CREAR YAML ──────────────────────────────────────
def create_yaml(path, name):
    yaml_content = f"""
path: {os.path.abspath(path)}
train: train/images
val: valid/images

nc: 1
names: ['{name}']
"""
    with open(os.path.join(path, "data.yaml"), "w") as f:
        f.write(yaml_content.strip())


# ─── MAIN ────────────────────────────────────────────
if __name__ == "__main__":

    process_dataset()

    create_yaml(BALL_PATH, "ball")
    create_yaml(PLAYERS_PATH, "player")

    print("\nTODO LISTO")