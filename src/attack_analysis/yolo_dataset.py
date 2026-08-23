from __future__ import annotations

import json
import os
import shutil
import statistics
from pathlib import Path
from typing import Iterable


def _link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        return
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _export_split(
    sequence_directories: Iterable[Path],
    output_directory: Path,
    split: str,
    link_mode: str,
) -> dict[str, object]:
    image_output = output_directory / "images" / split
    label_output = output_directory / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    image_count = 0
    labeled_image_count = 0
    box_widths: list[float] = []
    box_heights: list[float] = []
    sequences: list[str] = []

    for sequence_directory in sequence_directories:
        labels_path = sequence_directory / "Labels-GameState.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"missing SoccerNet labels: {labels_path}")
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        category_names = {
            int(item["id"]): str(item["name"]).lower()
            for item in payload.get("categories", [])
        }
        ball_category_ids = {
            category_id for category_id, name in category_names.items() if name == "ball"
        }
        if not ball_category_ids:
            raise ValueError(f"ball category is missing in {labels_path}")
        annotations_by_image: dict[str, list[dict[str, object]]] = {}
        for annotation in payload.get("annotations", []):
            if int(annotation.get("category_id", -1)) not in ball_category_ids:
                continue
            if not isinstance(annotation.get("bbox_image"), dict):
                continue
            annotations_by_image.setdefault(str(annotation["image_id"]), []).append(annotation)

        sequence_name = sequence_directory.name
        sequences.append(sequence_name)
        for image in payload.get("images", []):
            image_id = str(image.get("image_id", image.get("id")))
            source_name = Path(str(image["file_name"])).name
            source = sequence_directory / "img1" / source_name
            if not source.exists():
                source = sequence_directory / str(image["file_name"])
            if not source.exists():
                raise FileNotFoundError(f"missing SoccerNet image: {source}")
            stem = f"{sequence_name}_{Path(source_name).stem}"
            destination = image_output / f"{stem}{source.suffix.lower()}"
            _link_or_copy(source, destination, link_mode)

            width = float(image["width"])
            height = float(image["height"])
            lines: list[str] = []
            for annotation in annotations_by_image.get(image_id, []):
                box = annotation["bbox_image"]
                box_width = float(box["w"])
                box_height = float(box["h"])
                center_x = float(box.get("x_center", float(box["x"]) + box_width / 2.0))
                center_y = float(box.get("y_center", float(box["y"]) + box_height / 2.0))
                normalized = (
                    center_x / width,
                    center_y / height,
                    box_width / width,
                    box_height / height,
                )
                if not all(0.0 <= value <= 1.0 for value in normalized):
                    raise ValueError(f"ball box is outside image bounds: {sequence_name}/{source_name}")
                lines.append("0 " + " ".join(f"{value:.8f}" for value in normalized))
                box_widths.append(box_width)
                box_heights.append(box_height)
            if lines:
                labeled_image_count += 1
            (label_output / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="ascii"
            )
            image_count += 1

    def stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "min": round(min(values), 3),
            "median": round(statistics.median(values), 3),
            "max": round(max(values), 3),
        }

    return {
        "sequences": sequences,
        "images": image_count,
        "images_with_ball": labeled_image_count,
        "ball_boxes": len(box_widths),
        "ball_width_px": stats(box_widths),
        "ball_height_px": stats(box_heights),
    }


def export_soccernet_ball_dataset(
    output_directory: Path,
    train_sequences: Iterable[Path],
    val_sequences: Iterable[Path],
    link_mode: str = "hardlink",
) -> dict[str, object]:
    if link_mode not in {"hardlink", "copy"}:
        raise ValueError("link_mode must be hardlink or copy")
    train = tuple(path.resolve() for path in train_sequences)
    val = tuple(path.resolve() for path in val_sequences)
    if not train or not val:
        raise ValueError("at least one train and one validation sequence are required")
    overlap = set(train).intersection(val)
    if overlap:
        raise ValueError("train and validation sequences must not overlap")
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    train_report = _export_split(train, output_directory, "train", link_mode)
    val_report = _export_split(val, output_directory, "val", link_mode)
    data_yaml = output_directory / "data.yaml"
    data_yaml.write_text(
        f'path: "{output_directory.as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: ball\n",
        encoding="utf-8",
    )
    report = {
        "dataset": "SoccerNet-GSR ball detection",
        "output_directory": str(output_directory),
        "data_yaml": str(data_yaml),
        "link_mode": link_mode,
        "train": train_report,
        "val": val_report,
    }
    (output_directory / "export-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
