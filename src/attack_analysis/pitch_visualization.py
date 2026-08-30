from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib.lines import Line2D
from mplsoccer import Pitch

from .models import AttackSequence, Event, PITCH_LENGTH, PITCH_WIDTH


ACTION_COLORS = {
    "possession_start": "#9ca3af",
    "drive": "#facc15",
    "carry": "#facc15",
    "pass": "#38bdf8",
    "cross": "#c084fc",
    "throw_in": "#2dd4bf",
    "shot": "#fb4b4e",
    "header": "#f97316",
    "tackle": "#a3e635",
    "block": "#94a3b8",
    "turnover": "#f59e0b",
}

ACTION_LABELS = {
    "possession_start": "Bắt đầu kiểm soát",
    "drive": "Dẫn bóng",
    "carry": "Dẫn bóng",
    "pass": "Chuyền",
    "cross": "Tạt bóng",
    "throw_in": "Ném biên",
    "shot": "Sút",
    "header": "Đánh đầu",
    "tackle": "Tranh bóng",
    "block": "Cản phá",
    "turnover": "Mất bóng",
}


def _positioned_events(sequence: AttackSequence) -> tuple[Event, ...]:
    return tuple(event for event in sequence.events if event.x is not None and event.y is not None)


def _event_label(index: int, event: Event) -> str:
    label = ACTION_LABELS.get(event.event_type, event.event_type.replace("_", " ").title())
    details: list[str] = []
    if event.jersey is not None:
        details.append(f"#{event.jersey}")
    if event.xg is not None:
        details.append(f"xG {event.xg:.3f}")
    suffix = f" · {' · '.join(details)}" if details else ""
    return f"{index}. {label}{suffix}"


def _draw_sequence_arrows(pitch: Pitch, ax: Any, events: Sequence[Event]) -> None:
    for start, end in zip(events, events[1:]):
        pitch.arrows(
            float(start.x),
            float(start.y),
            float(end.x),
            float(end.y),
            ax=ax,
            color="#f8fafc",
            alpha=0.62,
            width=1.5,
            headwidth=5,
            headlength=5,
            zorder=2,
        )


def plot_attack_pitch(sequence: AttackSequence) -> Any:
    """Plot an ordered attack sequence on a metric 105 x 68 metre pitch."""

    events = _positioned_events(sequence)
    if not events:
        raise ValueError("attack has no pitch coordinates to plot")

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH,
        pitch_width=PITCH_WIDTH,
        pitch_color="#123a2d",
        line_color="#e2eee9",
        linewidth=1.5,
        stripe=True,
        stripe_color="#174735",
        corner_arcs=True,
    )
    figure, ax = pitch.draw(figsize=(12.5, 7.2))
    figure.patch.set_facecolor("#0e3026")

    # Make the final third readable without changing the underlying coordinates.
    ax.axvspan(70.0, PITCH_LENGTH, color="#facc15", alpha=0.07, zorder=0)
    ax.text(
        87.5,
        2.0,
        "1/3 cuối sân",
        color="#fde68a",
        fontsize=9,
        ha="center",
        va="bottom",
        alpha=0.9,
        zorder=1,
    )

    _draw_sequence_arrows(pitch, ax, events)
    for index, event in enumerate(events, start=1):
        color = ACTION_COLORS.get(event.event_type, "#e2e8f0")
        marker = "*" if event.event_type in {"shot", "header"} else "o"
        size = 430 if marker == "*" else 285
        pitch.scatter(
            float(event.x),
            float(event.y),
            ax=ax,
            s=size,
            marker=marker,
            c=color,
            edgecolors="#ffffff",
            linewidth=1.8,
            zorder=4,
        )
        ax.text(
            float(event.x),
            float(event.y),
            str(index),
            color="#071a14",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        event_y = float(event.y)
        if event_y < 7.0:
            above = True
        elif event_y > PITCH_WIDTH - 7.0:
            above = False
        else:
            # Alternate nearby labels so consecutive actions remain readable.
            above = index % 2 == 1
        label_y = event_y + 4.2 if above else event_y - 4.2
        ax.text(
            float(event.x),
            label_y,
            _event_label(index, event),
            color="#ffffff",
            fontsize=8.5,
            fontweight="semibold",
            ha="center",
            va="bottom" if above else "top",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "#071a14",
                "edgecolor": color,
                "alpha": 0.88,
            },
            zorder=6,
        )

    present_types = tuple(dict.fromkeys(event.event_type for event in events))
    legend = ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="*" if action in {"shot", "header"} else "o",
                color="none",
                markerfacecolor=ACTION_COLORS.get(action, "#e2e8f0"),
                markeredgecolor="#ffffff",
                markersize=11,
                label=ACTION_LABELS.get(action, action.replace("_", " ").title()),
            )
            for action in present_types
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=min(5, len(present_types)),
        frameon=False,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color("#ffffff")

    ax.annotate(
        "HƯỚNG TẤN CÔNG  →",
        xy=(0.98, 1.025),
        xycoords="axes fraction",
        color="#ffffff",
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    figure.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.12)
    return figure
