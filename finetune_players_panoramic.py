import argparse
import torch
from ultralytics import YOLO
import os

# ─── RUTAS ───────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
BALL_YAML    = os.path.join(BASE, "panoramic-cam-1_ball",    "data.yaml")
PLAYERS_YAML = os.path.join(BASE, "panoramic-cam-1_players", "data.yaml")

WEIGHTS_PLAYERS = os.path.join(BASE, "runs", "detect", "modelo_players_v23", "weights", "best.pt")
WEIGHTS_BALL    = os.path.join(BASE, "runs", "detect", "modelo_ball_v33",    "weights", "best.pt")

RESUME_PLAYERS = os.path.join(BASE, "runs", "detect", "modelo_players_v24_panoramic", "weights", "last.pt")
RESUME_BALL    = os.path.join(BASE, "runs", "detect", "modelo_ball_v34_panoramic",    "weights", "last.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Reanudar desde last.pt")
    parser.add_argument("--only", choices=["players", "ball"], default=None,
                        help="Entrenar solo un modelo")
    args = parser.parse_args()
    # ─── HARDWARE ────────────────────────────────────────────────────────────
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")

    if not args.resume:
        for p in [BALL_YAML, PLAYERS_YAML, WEIGHTS_PLAYERS, WEIGHTS_BALL]:
            assert os.path.exists(p), f"No encontrado: {p}"

    # ═══════════════════════════════════════════════════════════════════════
    #  MODELO 1: JUGADORES
    # ═══════════════════════════════════════════════════════════════════════
    if args.only != "ball":
        if args.resume:
            assert os.path.exists(RESUME_PLAYERS), f"No hay checkpoint: {RESUME_PLAYERS}"
            print(f"\nReanudando JUGADORES desde {RESUME_PLAYERS}\n")
            model_players = YOLO(RESUME_PLAYERS)
            model_players.train(resume=True)
        else:
            print(f"\nFine-tuning JUGADORES desde {WEIGHTS_PLAYERS}\n")
            model_players = YOLO(WEIGHTS_PLAYERS)
            model_players.train(
                data=PLAYERS_YAML,
                epochs=80,
                imgsz=1280,
                batch=4,
                device=0,
                name="modelo_players_v24_panoramic",

                optimizer="AdamW",
                lr0=1e-4,
                lrf=0.01,
                weight_decay=0.0005,
                momentum=0.937,

                cos_lr=True,
                warmup_epochs=2,
                patience=20,

                # freeze=10,  # descomentar si el dataset es pequeño (<500 imágenes)

                workers=0,         # 0 evita deadlocks en Windows
                amp=True,
                cache="disk",      # disk es más estable que ram en Windows

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
                mixup=0.1,
                copy_paste=0.0,
                close_mosaic=15,

                save=True,
                save_period=20,
                plots=True,
                verbose=True,
            )

        metrics = model_players.val()
        print(f"[JUGADORES] mAP@50: {metrics.box.map50:.4f}  |  mAP@50-95: {metrics.box.map:.4f}")
        del model_players
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════
    #  MODELO 2: BALÓN
    # ═══════════════════════════════════════════════════════════════════════
    if args.only != "players":
        if args.resume:
            assert os.path.exists(RESUME_BALL), f"No hay checkpoint: {RESUME_BALL}"
            print(f"\nReanudando BALÓN desde {RESUME_BALL}\n")
            model_ball = YOLO(RESUME_BALL)
            model_ball.train(resume=True)
        else:
            print(f"\nFine-tuning BALÓN desde {WEIGHTS_BALL}\n")
            model_ball = YOLO(WEIGHTS_BALL)
            model_ball.train(
                data=BALL_YAML,
                epochs=80,
                imgsz=1280,
                batch=4,
                device=0,
                name="modelo_ball_v34_panoramic",

                optimizer="AdamW",
                lr0=1e-4,
                lrf=0.01,
                weight_decay=0.0005,
                momentum=0.937,

                cos_lr=True,
                warmup_epochs=2,
                patience=20,

                workers=0,         # 0 evita deadlocks en Windows
                amp=True,
                cache="disk",

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
                mixup=0.0,        # sin mixup para balón (objeto pequeño)
                copy_paste=0.1,   # ayuda a aumentar instancias de balón
                close_mosaic=15,

                save=True,
                save_period=20,
                plots=True,
                verbose=True,
            )

        metrics_ball = model_ball.val()
        print(f"[BALÓN]     mAP@50: {metrics_ball.box.map50:.4f}  |  mAP@50-95: {metrics_ball.box.map:.4f}")
        del model_ball
        torch.cuda.empty_cache()

    print("\n" + "═" * 55)
    print("FINE-TUNING COMPLETADO")
    print("═" * 55)
    print("Modelos guardados en:")
    print("  runs/detect/modelo_players_v24_panoramic/weights/best.pt")
    print("  runs/detect/modelo_ball_v34_panoramic/weights/best.pt")
    print("═" * 55)


if __name__ == '__main__':
    main()
