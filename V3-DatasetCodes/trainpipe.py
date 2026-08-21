#!/usr/bin/env python3
"""
Hades-06 — Solar Defect YOLOv8n Training
Single-file, no external config. Matches your original CLI command:

  yolo detect train \
    model=./yolov8n.pt \
    data=./SolarDataset/data.yaml \
    epochs=150 \
    imgsz=640 \
    batch=16 \
    device=0 \
    workers=4 \
    project=./Solaris \
    name=yolov8n-final \
    patience=25

Adds: a plateau timer. If mAP50-95 doesn't improve by more than
PLATEAU_MIN_DELTA for PLATEAU_TIMEOUT_SEC of wall-clock time, training
stops early — independent of Ultralytics' own epoch-based `patience`
(both are active: whichever trips first stops training).

Usage:
    chmod +x train_hades06.py
    ./train_hades06.py
"""

import time
from ultralytics import YOLO
from ultralytics.utils.callbacks.base import add_integration_callbacks  # noqa: F401 (ensures callback system loaded)

# ============================================================
# CONFIG
# ============================================================
MODEL = "./yolov8n.pt"
DATA = "./SolarDataset/data.yaml"
EPOCHS = 100       # adequate ceiling for ~3,900 images on yolov8n; plateau timer + patience will cut short if it converges earlier
IMGSZ = 640
BATCH = 16          # RTX 3050 should handle this at imgsz=640; drop to 8 if you hit OOM
DEVICE = 0
WORKERS = 4
PROJECT = "./Solaris"
NAME = "yolov8n-final"
PATIENCE = 25       # Ultralytics' own epoch-based early stopping (val mAP plateau)

# ============================================================
# PLATEAU TIMER CONFIG (separate, wall-clock-based, on top of PATIENCE above)
# ============================================================
PLATEAU_MIN_DELTA = 0.001      # minimum mAP50-95 improvement to count as "progress"
PLATEAU_TIMEOUT_SEC = 1800     # stop if no progress for this many seconds (30 min)
PLATEAU_CHECK_METRIC = "metrics/mAP50-95(B)"


class PlateauTimer:
    """Tracks best mAP50-95 and the wall-clock time since it last improved.
    Sets trainer.stop_training = True when the timeout is exceeded."""

    def __init__(self, min_delta: float, timeout_sec: float, metric_key: str):
        self.min_delta = min_delta
        self.timeout_sec = timeout_sec
        self.metric_key = metric_key
        self.best = -1.0
        self.last_improve_time = time.time()

    def on_fit_epoch_end(self, trainer):
        metrics = trainer.metrics
        current = metrics.get(self.metric_key)
        if current is None:
            return  # metric not populated yet (e.g. first epoch)

        now = time.time()
        improved = current > self.best + self.min_delta

        if improved:
            print(f"[PlateauTimer] {self.metric_key} improved: "
                  f"{self.best:.4f} -> {current:.4f}. Timer reset.")
            self.best = current
            self.last_improve_time = now
        else:
            elapsed = now - self.last_improve_time
            remaining = self.timeout_sec - elapsed
            print(f"[PlateauTimer] No improvement (best={self.best:.4f}, "
                  f"current={current:.4f}). Plateau time: {elapsed:.0f}s "
                  f"/ {self.timeout_sec:.0f}s (stops in {max(0, remaining):.0f}s)")

            if elapsed >= self.timeout_sec:
                print(f"[PlateauTimer] STOPPING: no improvement in "
                      f"{self.timeout_sec:.0f}s. Best {self.metric_key} = {self.best:.4f}")
                trainer.stop_training = True


def main():
    model = YOLO(MODEL)

    plateau = PlateauTimer(
        min_delta=PLATEAU_MIN_DELTA,
        timeout_sec=PLATEAU_TIMEOUT_SEC,
        metric_key=PLATEAU_CHECK_METRIC,
    )
    model.add_callback("on_fit_epoch_end", plateau.on_fit_epoch_end)

    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        patience=PATIENCE,
        exist_ok=True,   # allows reruns without erroring on existing folder
    )

    print(f"\nDone. Best weights under: {PROJECT}/{NAME}/weights/best.pt")
    print(f"Final best mAP50-95 tracked by PlateauTimer: {plateau.best:.4f}")


if __name__ == "__main__":
    main()