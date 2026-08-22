from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .models import AttackSequence, Event
from .scoring import evaluate_attack
from .segmentation import SegmentationConfig, segment_attacks
from .video import inspect_video


COMMANDS = {"evaluate", "analyze-events", "video-info"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate football attacks from structured events."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate one pre-segmented attack JSON file"
    )
    evaluate_parser.add_argument("input", type=Path)

    events_parser = subparsers.add_parser(
        "analyze-events", help="Segment a match event stream and evaluate every attack"
    )
    events_parser.add_argument("input", type=Path)
    events_parser.add_argument("--max-gap", type=float, default=15.0)
    events_parser.add_argument("--min-events", type=int, default=1)

    video_parser = subparsers.add_parser(
        "video-info", help="Read video metadata through the optional OpenCV backend"
    )
    video_parser.add_argument("input", type=Path)
    return parser


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _evaluate_file(path: Path) -> dict[str, object]:
    return evaluate_attack(AttackSequence.from_dict(_load_json(path))).to_dict()


def _analyze_event_stream(path: Path, max_gap: float, min_events: int) -> dict[str, object]:
    payload = _load_json(path)
    raw_events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(raw_events, list):
        raise ValueError("event stream must be a JSON list or an object containing 'events'")

    events = tuple(Event.from_dict(item) for item in raw_events)
    attacks = segment_attacks(
        events,
        SegmentationConfig(max_gap_sec=max_gap, min_events=min_events),
    )
    return {
        "source": str(path),
        "attack_count": len(attacks),
        "attacks": [
            {
                "attack": attack.to_dict(),
                "evaluation": evaluate_attack(attack).to_dict(),
            }
            for attack in attacks
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Keep the original `attack-analysis file.json` command working.
    if arguments and arguments[0] not in COMMANDS and not arguments[0].startswith("-"):
        arguments.insert(0, "evaluate")

    args = build_parser().parse_args(arguments)
    if args.command == "evaluate":
        result = _evaluate_file(args.input)
    elif args.command == "analyze-events":
        result = _analyze_event_stream(args.input, args.max_gap, args.min_events)
    else:
        result = inspect_video(args.input).to_dict()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

