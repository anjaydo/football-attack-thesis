from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'Ultralytics is missing. Install with: python -m pip install -e ".[vision]"'
        ) from error
    return YOLO


def train_ball_model(
    data_yaml: Path,
    output_directory: Path,
    model: str = "yolo26s.pt",
    epochs: int = 100,
    image_size: int = 1280,
    batch_size: int = 4,
    device: str = "cpu",
    workers: int = 0,
    run_name: str = "soccernet-ball",
) -> dict[str, object]:
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset YAML does not exist: {data_yaml}")
    if epochs <= 0 or image_size <= 0 or batch_size == 0 or workers < 0:
        raise ValueError("epochs, image_size and batch_size must be positive; workers >= 0")
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    yolo = _load_yolo()(model)
    yolo.train(
        data=str(data_yaml.resolve()),
        project=str(output_directory),
        name=run_name,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=workers,
        classes=[0],
        single_cls=True,
        patience=max(10, min(30, epochs // 4)),
        close_mosaic=max(0, min(10, epochs // 10)),
        plots=True,
    )
    run_directory = output_directory / run_name
    return {
        "data_yaml": str(data_yaml.resolve()),
        "base_model": model,
        "run_directory": str(run_directory),
        "best_weights": str(run_directory / "weights" / "best.pt"),
        "last_weights": str(run_directory / "weights" / "last.pt"),
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "device": device,
    }
