from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from matplotlib.lines import Line2D
from mplsoccer import Pitch

from .footpass import actions_in_attack_view
from .models import AttackSequence, Event, ObservedActionV2, PITCH_LENGTH, PITCH_WIDTH


ACTION_COLORS = {
    "possession_start": "#9ca3af",
    "drive": "#facc15",
    "carry": "#facc15",
    "pass": "#38bdf8",
    "cross": "#c084fc",
    "throw_in": "#2dd4bf",
    "shot": "#fb4b4e",
    "shot_attempt": "#fb4b4e",
    "header": "#f97316",
    "headed_attempt": "#f97316",
    "tackle": "#a3e635",
    "block": "#94a3b8",
    "forced_goal_line_out": "#e879f9",
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
    "shot_attempt": "Dứt điểm",
    "header": "Đánh đầu",
    "headed_attempt": "Đánh đầu về cầu môn",
    "tackle": "Tranh bóng",
    "block": "Cản phá",
    "forced_goal_line_out": "Ép bóng ra ngoài cuối sân",
    "turnover": "Mất bóng",
}


@dataclass(frozen=True)
class PitchEventView:
    x: float
    y: float
    action_type: str
    annotation: str
    end_x: float | None = None
    end_y: float | None = None
    opponent: bool = False


def _action_label(action_type: str) -> str:
    return ACTION_LABELS.get(action_type, action_type.replace("_", " ").title())


def _format_bool(value: bool | None) -> str | None:
    if value is None:
        return None
    return "có" if value else "không"


def _join_parts(parts: Sequence[str]) -> str:
    return " · ".join(part for part in parts if part)


def _event_annotation(index: int, event: Event) -> str:
    lines = [f"{index}. {_action_label(event.event_type)}"]
    lines.append(f"t = {event.timestamp:.2f}s")
    if event.frame is not None:
        lines.append(f"frame {event.frame}")
    lines.append(f"Đội {event.team}")
    player_parts = []
    if event.jersey is not None:
        player_parts.append(f"#{event.jersey}")
    if event.player_id:
        player_parts.append(f"id {event.player_id}")
    if event.role:
        player_parts.append(event.role)
    if player_parts:
        lines.append(_join_parts(player_parts))
    if event.x is not None and event.y is not None:
        lines.append(f"vị trí ({event.x:.1f}, {event.y:.1f})")
    if event.end_x is not None and event.end_y is not None:
        lines.append(f"đích ({event.end_x:.1f}, {event.end_y:.1f})")
    if event.outcome:
        lines.append(f"kết quả: {event.outcome}")
    if event.xg is not None:
        xg_line = f"xG {event.xg:.3f}"
        if event.xg_source:
            xg_line += f" ({event.xg_source})"
        lines.append(xg_line)
    if event.xg_inputs:
        inputs = ", ".join(
            f"{key}={value:.3g}" if value is not None else f"{key}=—"
            for key, value in event.xg_inputs.items()
        )
        lines.append(f"xG inputs: {inputs}")
    if event.xt_before is not None or event.xt_after is not None:
        before = "—" if event.xt_before is None else f"{event.xt_before:.3f}"
        after = "—" if event.xt_after is None else f"{event.xt_after:.3f}"
        lines.append(f"xT {before} → {after}")
    lines.append(f"tin cậy {event.confidence:.2f}")
    if event.source:
        lines.append(f"nguồn: {event.source}")
    if event.evidence_id:
        lines.append(f"evidence {event.evidence_id}")
    return "\n".join(lines)


def _observed_annotation(index: int, action: ObservedActionV2) -> str:
    lines = [f"{index}. {_action_label(action.action_type)}"]
    lines.append(f"t = {action.timestamp_sec:.2f}s")
    lines.append(f"frame {action.frame} · hiệp {action.period}")
    team_parts = []
    if action.actor_team_id:
        team_parts.append(f"đội {action.actor_team_id}")
    if action.possession_team_id and action.possession_team_id != action.actor_team_id:
        team_parts.append(f"kiểm soát {action.possession_team_id}")
    if action.beneficiary_team_id and action.beneficiary_team_id not in {
        action.actor_team_id,
        action.possession_team_id,
    }:
        team_parts.append(f"hưởng lợi {action.beneficiary_team_id}")
    if team_parts:
        lines.append(_join_parts(team_parts))
    player_parts = []
    if action.jersey is not None:
        player_parts.append(f"#{action.jersey}")
    if action.actor_id:
        player_parts.append(f"id {action.actor_id}")
    if action.role:
        player_parts.append(action.role)
    if player_parts:
        lines.append(_join_parts(player_parts))
    if action.x is not None and action.y is not None:
        lines.append(f"vị trí ({action.x:.1f}, {action.y:.1f})")
    if action.body_part:
        lines.append(f"bộ phận: {action.body_part}")
    if action.outcome:
        lines.append(f"kết quả: {action.outcome}")
    outcome_sources = [
        qualifier.removeprefix("outcome_source:")
        for qualifier in action.qualifiers
        if qualifier.startswith("outcome_source:")
    ]
    if outcome_sources:
        lines.append(f"nguồn outcome: {outcome_sources[-1]}")
    if action.qualifiers:
        lines.append("qualifier: " + ", ".join(action.qualifiers))
    bbox = _format_bool(action.bbox_visible)
    if bbox is not None:
        lines.append(f"bbox: {bbox}")
    if action.replay:
        lines.append("replay: có")
    probabilities = []
    if action.action_probability is not None:
        probabilities.append(f"P(action) {action.action_probability:.2f}")
    if action.actor_probability is not None:
        probabilities.append(f"P(actor) {action.actor_probability:.2f}")
    if action.timestamp_probability is not None:
        probabilities.append(f"P(time) {action.timestamp_probability:.2f}")
    if probabilities:
        lines.append(_join_parts(probabilities))
    if action.raw_left_to_right is not None:
        lines.append("hướng thô: " + ("L→R" if action.raw_left_to_right else "R→L"))
    lines.append(f"tier: {action.source_tier.value}")
    if action.association_id:
        lines.append(f"association {action.association_id}")
    if action.evidence_ids:
        lines.append("evidence: " + ", ".join(action.evidence_ids))
    if action.action_id:
        lines.append(f"id {action.action_id}")
    return "\n".join(lines)


def _view_from_event(index: int, event: Event) -> PitchEventView | None:
    if event.x is None or event.y is None:
        return None
    return PitchEventView(
        x=float(event.x),
        y=float(event.y),
        action_type=event.event_type,
        annotation=_event_annotation(index, event),
        end_x=event.end_x,
        end_y=event.end_y,
    )


def _view_from_action(index: int, action: ObservedActionV2) -> PitchEventView | None:
    if action.x is None or action.y is None:
        return None
    opponent = bool(
        action.possession_team_id
        and action.actor_team_id
        and action.actor_team_id != action.possession_team_id
    )
    return PitchEventView(
        x=float(action.x),
        y=float(action.y),
        action_type=action.action_type,
        annotation=_observed_annotation(index, action),
        opponent=opponent,
    )


def pitch_event_views(
    items: Sequence[Event | ObservedActionV2],
    *,
    attack_relative: bool = True,
) -> tuple[PitchEventView, ...]:
    prepared: list[Event | ObservedActionV2]
    actions = [item for item in items if isinstance(item, ObservedActionV2)]
    if attack_relative and actions:
        remapped = actions_in_attack_view(tuple(actions))
        iterator = iter(remapped)
        prepared = [next(iterator) if isinstance(item, ObservedActionV2) else item for item in items]
    else:
        prepared = list(items)
    views: list[PitchEventView] = []
    for item in prepared:
        index = len(views) + 1
        if isinstance(item, ObservedActionV2):
            view = _view_from_action(index, item)
        elif isinstance(item, Event):
            view = _view_from_event(index, item)
        else:
            raise TypeError(f"unsupported pitch event type: {type(item)!r}")
        if view is not None:
            views.append(view)
    return tuple(views)


def _label_anchor(index: int, x: float, y: float) -> tuple[float, float]:
    ring = 1 + (index - 1) // 8
    angle = ((index - 1) % 8) * (math.pi / 4) + (0.18 * ((index - 1) // 8))
    dx = 8.8 * ring * math.cos(angle)
    dy = 6.6 * ring * math.sin(angle)
    if abs(dx) < 2.5:
        dx = 2.5 if dx >= 0 else -2.5
    return (
        min(max(x + dx, 3.0), PITCH_LENGTH - 3.0),
        min(max(y + dy, 4.0), PITCH_WIDTH - 4.0),
    )


def _draw_sequence_arrows(pitch: Pitch, ax: Any, views: Sequence[PitchEventView]) -> None:
    for start, end in zip(views, views[1:]):
        pitch.arrows(
            start.x,
            start.y,
            end.x,
            end.y,
            ax=ax,
            color="#f8fafc",
            alpha=0.45,
            width=1.4,
            headwidth=5,
            headlength=5,
            zorder=2,
        )
    for view in views:
        if view.end_x is None or view.end_y is None:
            continue
        pitch.arrows(
            view.x,
            view.y,
            float(view.end_x),
            float(view.end_y),
            ax=ax,
            color="#7dd3fc",
            alpha=0.85,
            width=1.8,
            headwidth=6,
            headlength=6,
            zorder=3,
        )


def plot_pitch_events(
    items: Sequence[Event | ObservedActionV2],
    *,
    attack_relative: bool = True,
) -> Any:
    """Plot event points on a metric 105 x 68 m mplsoccer pitch with full labels."""

    views = pitch_event_views(items, attack_relative=attack_relative)
    if not views:
        raise ValueError("attack has no pitch coordinates to plot")

    height = 8.0 + 0.38 * max(0, len(views) - 3)
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
    figure, ax = pitch.draw(figsize=(13.2, height))
    figure.patch.set_facecolor("#0e3026")

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

    _draw_sequence_arrows(pitch, ax, views)
    for index, view in enumerate(views, start=1):
        color = ACTION_COLORS.get(view.action_type, "#e2e8f0")
        marker = "*" if view.action_type in {"shot", "shot_attempt", "header", "headed_attempt"} else "o"
        size = 430 if marker == "*" else 285
        pitch.scatter(
            view.x,
            view.y,
            ax=ax,
            s=size,
            marker=marker,
            c=color,
            edgecolors="#f97316" if view.opponent else "#ffffff",
            linewidth=2.2 if view.opponent else 1.8,
            zorder=4,
        )
        ax.text(
            view.x,
            view.y,
            str(index),
            color="#071a14",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        label_x, label_y = _label_anchor(index, view.x, view.y)
        ax.annotate(
            view.annotation,
            xy=(view.x, view.y),
            xytext=(label_x, label_y),
            color="#ffffff",
            fontsize=7.2,
            ha="center",
            va="center",
            linespacing=1.25,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 1.0,
                "alpha": 0.85,
            },
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "#071a14",
                "edgecolor": color,
                "alpha": 0.92,
            },
            zorder=6,
        )

    present_types = tuple(dict.fromkeys(view.action_type for view in views))
    legend = ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="*" if action in {"shot", "shot_attempt", "header", "headed_attempt"} else "o",
                color="none",
                markerfacecolor=ACTION_COLORS.get(action, "#e2e8f0"),
                markeredgecolor="#ffffff",
                markersize=11,
                label=_action_label(action),
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


def plot_attack_pitch(
    sequence: AttackSequence | Sequence[Event | ObservedActionV2],
    *,
    attack_relative: bool = True,
) -> Any:
    """Plot events on a metric 105 x 68 metre mplsoccer pitch."""

    items = sequence.events if isinstance(sequence, AttackSequence) else sequence
    return plot_pitch_events(items, attack_relative=attack_relative)
