import torch
from ultralytics import YOLO
import os

# ─── RUTAS A LOS DATASETS ───────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
BALL_YAML    = os.path.join(BASE, "data", "Analisis-Tactico-Futbol-16_ball",    "data.yaml")
PLAYERS_YAML = os.path.join(BASE, "data", "Analisis-Tactico-Futbol-16_players", "data.yaml")


def train_models():

    # ─── HARDWARE ────────────────────────────────────────────────────────────
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponible: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.1f} GB")

    # ═══════════════════════════════════════════════════════════════════════
    # MODELO 1: BALON
    # ═══════════════════════════════════════════════════════════════════════
    print("\nEntrenando modelo de BALON...\n")

    model_ball = YOLO("yolo26s.pt")

    model_ball.train(
        data=BALL_YAML,
        epochs=200,
        imgsz=1280,
        batch=8,
        device=0,
        name="modelo_ball_v3",

        # ── rendimiento ────────────────────────────────────────────────────
        workers=4,
        amp=True,
        cache="ram",

        # ── optimizador ────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        momentum=0.937,

        cos_lr=True,
        warmup_epochs=3,

        # ── early stopping ─────────────────────────────────────────────────
        patience=40,

        # ── augmentaciones para objeto pequeño ─────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.5,

        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,

        close_mosaic=20,

        # ── guardado ──────────────────────────────────────────────────────
        save=True,
        save_period=20,
        plots=True,
        verbose=True,
    )

    print("\nModelo de balon entrenado")

    # ── limpieza GPU ────────────────────────────────────────────────────────
    print("Liberando memoria GPU...")
    del model_ball
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # MODELO 2: JUGADORES
    # ═══════════════════════════════════════════════════════════════════════
    print("\nEntrenando modelo de JUGADORES...\n")

    model_players = YOLO("yolo26m.pt")

    model_players.train(
        data=PLAYERS_YAML,
        epochs=250,
        imgsz=1280,
        batch=6,
        device=0,
        name="modelo_players_v3",

        # ── rendimiento ────────────────────────────────────────────────────
        workers=4,
        amp=True,
        cache="ram",

        # ── optimizador ────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        momentum=0.937,

        cos_lr=True,
        warmup_epochs=3,

        patience=40,

        # ── augmentaciones ─────────────────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=0.0,        # campo siempre horizontal
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.5,

        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,

        close_mosaic=20,

        # ── guardado ──────────────────────────────────────────────────────
        save=True,
        save_period=20,
        plots=True,
        verbose=True,
    )

    print("\nModelo de jugadores entrenado")

    # ─── RESUMEN ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 55)
    print("Modelos guardados en:")
    print("  runs/detect/modelo_ball_v3/weights/best.pt")
    print("  runs/detect/modelo_players_v3/weights/best.pt")
    print("=" * 55)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_models()
