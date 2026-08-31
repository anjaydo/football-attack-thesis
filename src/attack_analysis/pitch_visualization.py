from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from matplotlib.lines import Line2D
from mplsoccer import Pitch

from .footpass import actions_in_attack_view
from .models import AttackSequence, Event, ObservedActionV2, PITCH_LENGTH, PITCH_WIDTH

COCO_SKELETON = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
)


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

SHOT_ACTION_TYPES = frozenset({"shot", "shot_attempt", "header", "headed_attempt"})

# FIFA markings on a 105 × 68 m pitch (same geometry mplsoccer uses for custom).
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_WIDTH_M = 18.32
GOAL_WIDTH_M = 7.32
CENTER_CIRCLE_RADIUS_M = 9.15
PENALTY_SPOT_M = 11.0
CORNER_ARC_RADIUS_M = 1.0
PITCH_PAD_M = 2.0
PIXELS_PER_METRE = 8.0

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
        marker = "*" if view.action_type in SHOT_ACTION_TYPES else "o"
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
                marker="*" if action in SHOT_ACTION_TYPES else "o",
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


def _line_segment(x1: float, y1: float, x2: float, y2: float, group: str) -> list[dict[str, float | str | int]]:
    return [
        {"x": x1, "y": y1, "t": 0, "group": group},
        {"x": x2, "y": y2, "t": 1, "group": group},
    ]


def _arc_points(
    cx: float,
    cy: float,
    radius: float,
    start_rad: float,
    end_rad: float,
    group: str,
    *,
    steps: int = 48,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for index in range(steps + 1):
        angle = start_rad + (end_rad - start_rad) * index / steps
        rows.append(
            {
                "x": cx + radius * math.cos(angle),
                "y": cy + radius * math.sin(angle),
                "t": index,
                "group": group,
            }
        )
    return rows


def _penalty_box_lines(depth: float, width: float, left: bool) -> list[dict[str, float | str | int]]:
    y0 = (PITCH_WIDTH - width) / 2.0
    y1 = y0 + width
    if left:
        x0, x1 = 0.0, depth
        tag = f"left-{depth}"
    else:
        x0, x1 = PITCH_LENGTH - depth, PITCH_LENGTH
        tag = f"right-{depth}"
    return [
        *_line_segment(x1, y0, x1, y1, f"{tag}-front"),
        *_line_segment(x0, y0, x1, y0, f"{tag}-bottom"),
        *_line_segment(x0, y1, x1, y1, f"{tag}-top"),
    ]


def _metric_pitch_geometry() -> dict[str, list[dict[str, float | str | int]]]:
    """Pitch outlines in metres so events and markings share one coordinate system."""

    length, width = PITCH_LENGTH, PITCH_WIDTH
    mid_x, mid_y = length / 2.0, width / 2.0
    stripes: list[dict[str, float | str | int]] = []
    stripe_count = 14
    stripe_width = length / stripe_count
    for index in range(0, stripe_count, 2):
        stripes.append(
            {
                "x": index * stripe_width,
                "x2": (index + 1) * stripe_width,
                "y": 0.0,
                "y2": width,
            }
        )
    lines = [
        *_line_segment(0.0, 0.0, length, 0.0, "sideline-bottom"),
        *_line_segment(length, 0.0, length, width, "goal-line-right"),
        *_line_segment(length, width, 0.0, width, "sideline-top"),
        *_line_segment(0.0, width, 0.0, 0.0, "goal-line-left"),
        *_line_segment(mid_x, 0.0, mid_x, width, "halfway"),
        *_penalty_box_lines(PENALTY_AREA_DEPTH_M, PENALTY_AREA_WIDTH_M, True),
        *_penalty_box_lines(PENALTY_AREA_DEPTH_M, PENALTY_AREA_WIDTH_M, False),
        *_penalty_box_lines(GOAL_AREA_DEPTH_M, GOAL_AREA_WIDTH_M, True),
        *_penalty_box_lines(GOAL_AREA_DEPTH_M, GOAL_AREA_WIDTH_M, False),
    ]
    goal_y0 = (width - GOAL_WIDTH_M) / 2.0
    goal_y1 = goal_y0 + GOAL_WIDTH_M
    lines.extend(_line_segment(0.0, goal_y0, 0.0, goal_y1, "goal-mouth-left"))
    lines.extend(_line_segment(length, goal_y0, length, goal_y1, "goal-mouth-right"))
    alpha = math.acos(min(1.0, (PENALTY_AREA_DEPTH_M - PENALTY_SPOT_M) / CENTER_CIRCLE_RADIUS_M))
    arcs = [
        *_arc_points(mid_x, mid_y, CENTER_CIRCLE_RADIUS_M, 0.0, 2.0 * math.pi, "center-circle"),
        *_arc_points(PENALTY_SPOT_M, mid_y, CENTER_CIRCLE_RADIUS_M, -alpha, alpha, "penalty-arc-left"),
        *_arc_points(
            length - PENALTY_SPOT_M,
            mid_y,
            CENTER_CIRCLE_RADIUS_M,
            math.pi - alpha,
            math.pi + alpha,
            "penalty-arc-right",
        ),
        *_arc_points(0.0, 0.0, CORNER_ARC_RADIUS_M, 0.0, math.pi / 2.0, "corner-sw"),
        *_arc_points(0.0, width, CORNER_ARC_RADIUS_M, -math.pi / 2.0, 0.0, "corner-nw"),
        *_arc_points(length, 0.0, CORNER_ARC_RADIUS_M, math.pi / 2.0, math.pi, "corner-se"),
        *_arc_points(length, width, CORNER_ARC_RADIUS_M, math.pi, 3.0 * math.pi / 2.0, "corner-ne"),
    ]
    spots = [
        {"x": mid_x, "y": mid_y, "group": "center-spot"},
        {"x": PENALTY_SPOT_M, "y": mid_y, "group": "penalty-left"},
        {"x": length - PENALTY_SPOT_M, "y": mid_y, "group": "penalty-right"},
    ]
    return {"stripes": stripes, "lines": lines, "arcs": arcs, "spots": spots}


def _metre_scales(alt: Any) -> tuple[Any, Any, float, float]:
    x0, x1 = -PITCH_PAD_M, PITCH_LENGTH + PITCH_PAD_M
    y0, y1 = -PITCH_PAD_M, PITCH_WIDTH + PITCH_PAD_M
    width = (x1 - x0) * PIXELS_PER_METRE
    height = (y1 - y0) * PIXELS_PER_METRE
    x_scale = alt.Scale(domain=[x0, x1], nice=False, zero=False)
    # Same orientation as mplsoccer/matplotlib: y = 0 on the near touchline.
    y_scale = alt.Scale(domain=[y0, y1], nice=False, zero=False, reverse=True)
    return x_scale, y_scale, width, height


def _pitch_layers(alt: Any, pd: Any, x_scale: Any, y_scale: Any) -> list[Any]:
    geometry = _metric_pitch_geometry()
    grass = (
        alt.Chart(
            pd.DataFrame(
                {"x": [0.0], "x2": [PITCH_LENGTH], "y": [0.0], "y2": [PITCH_WIDTH]}
            )
        )
        .mark_rect(color="#123a2d")
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=None),
            x2=alt.X2("x2:Q"),
            y=alt.Y("y:Q", scale=y_scale, axis=None),
            y2=alt.Y2("y2:Q"),
        )
    )
    stripes = (
        alt.Chart(pd.DataFrame(geometry["stripes"]))
        .mark_rect(color="#174735", opacity=0.85)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=None),
            x2=alt.X2("x2:Q"),
            y=alt.Y("y:Q", scale=y_scale, axis=None),
            y2=alt.Y2("y2:Q"),
        )
    )
    final_third = (
        alt.Chart(
            pd.DataFrame(
                {"x": [70.0], "x2": [PITCH_LENGTH], "y": [0.0], "y2": [PITCH_WIDTH]}
            )
        )
        .mark_rect(color="#facc15", opacity=0.10)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=None),
            x2=alt.X2("x2:Q"),
            y=alt.Y("y:Q", scale=y_scale, axis=None),
            y2=alt.Y2("y2:Q"),
        )
    )
    lines = (
        alt.Chart(pd.DataFrame(geometry["lines"]))
        .mark_line(color="#e2eee9", strokeWidth=1.6)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=None),
            y=alt.Y("y:Q", scale=y_scale, axis=None),
            order="t:Q",
            detail="group:N",
        )
    )
    arcs = (
        alt.Chart(pd.DataFrame(geometry["arcs"]))
        .mark_line(color="#e2eee9", strokeWidth=1.6)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=None),
            y=alt.Y("y:Q", scale=y_scale, axis=None),
            order="t:Q",
            detail="group:N",
        )
    )
    spots = (
        alt.Chart(pd.DataFrame(geometry["spots"]))
        .mark_point(filled=True, color="#e2eee9", size=28, opacity=1)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=None),
            y=alt.Y("y:Q", scale=y_scale, axis=None),
        )
    )
    caption = (
        alt.Chart(pd.DataFrame({"x": [87.5], "y": [2.2], "label": ["1/3 cuối sân"]}))
        .mark_text(color="#fde68a", fontSize=11, baseline="bottom")
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=None),
            y=alt.Y("y:Q", scale=y_scale, axis=None),
            text="label:N",
        )
    )
    return [grass, stripes, final_third, lines, arcs, spots, caption]


def interactive_pitch_events(
    items: Sequence[Event | ObservedActionV2],
    *,
    attack_relative: bool = True,
) -> Any:
    """Interactive 105×68 m pitch: events sit in metre coords, details on hover."""

    try:
        import altair as alt
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Interactive pitch plots require pandas and altair (Streamlit demo extra)."
        ) from error

    views = pitch_event_views(items, attack_relative=attack_relative)
    if not views:
        raise ValueError("attack has no pitch coordinates to plot")

    x_scale, y_scale, width, height = _metre_scales(alt)
    event_rows = []
    for index, view in enumerate(views, start=1):
        lines = [line for line in view.annotation.split("\n") if line]
        event_rows.append(
            {
                "x": view.x,
                "y": view.y,
                "index": str(index),
                "action_label": _action_label(view.action_type),
                "shape": "triangle-up" if view.action_type in SHOT_ACTION_TYPES else "circle",
                "size": 280 if view.action_type in SHOT_ACTION_TYPES else 170,
                "stroke": "#f97316" if view.opponent else "#ffffff",
                "title": lines[0] if lines else _action_label(view.action_type),
                "detail": " · ".join(lines[1:]),
                "hover": view.annotation,
            }
        )
    events = pd.DataFrame(event_rows)
    present_labels = tuple(dict.fromkeys(row["action_label"] for row in event_rows))
    present_types = tuple(dict.fromkeys(view.action_type for view in views))
    color_range = [ACTION_COLORS.get(action, "#e2e8f0") for action in present_types]

    link_rows: list[dict[str, float | str]] = []
    for start, end in zip(views, views[1:]):
        link_rows.append(
            {
                "x": start.x,
                "y": start.y,
                "x2": end.x,
                "y2": end.y,
                "kind": "sequence",
            }
        )
    for view in views:
        if view.end_x is None or view.end_y is None:
            continue
        link_rows.append(
            {
                "x": view.x,
                "y": view.y,
                "x2": float(view.end_x),
                "y2": float(view.end_y),
                "kind": "target",
            }
        )

    layers: list[Any] = _pitch_layers(alt, pd, x_scale, y_scale)
    if link_rows:
        layers.append(
            alt.Chart(pd.DataFrame(link_rows))
            .mark_rule(opacity=0.55)
            .encode(
                x=alt.X("x:Q", scale=x_scale, axis=None),
                y=alt.Y("y:Q", scale=y_scale, axis=None),
                x2=alt.X2("x2:Q"),
                y2=alt.Y2("y2:Q"),
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(
                        domain=["sequence", "target"],
                        range=["#f8fafc", "#7dd3fc"],
                    ),
                    legend=None,
                ),
                strokeWidth=alt.condition(
                    alt.datum.kind == "target",
                    alt.value(2.2),
                    alt.value(1.4),
                ),
            )
        )
    layers.append(
        alt.Chart(events)
        .mark_point(filled=True, opacity=1)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=None),
            y=alt.Y("y:Q", scale=y_scale, axis=None),
            color=alt.Color(
                "action_label:N",
                scale=alt.Scale(domain=list(present_labels), range=color_range),
                legend=alt.Legend(
                    title="Loại",
                    orient="bottom",
                    columns=min(5, len(present_labels)),
                ),
            ),
            shape=alt.Shape("shape:N", legend=None),
            size=alt.Size("size:Q", legend=None),
            stroke=alt.Stroke(
                "stroke:N",
                scale=alt.Scale(domain=["#f97316", "#ffffff"], range=["#f97316", "#ffffff"]),
                legend=None,
            ),
            strokeWidth=alt.value(1.8),
            tooltip=[
                alt.Tooltip("title:N", title="Hành động"),
                alt.Tooltip("detail:N", title="Chi tiết"),
            ],
        )
    )
    layers.append(
        alt.Chart(events)
        .mark_text(fontSize=10, fontWeight="bold", color="#071a14")
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=None),
            y=alt.Y("y:Q", scale=y_scale, axis=None),
            text="index:N",
        )
    )
    return (
        alt.layer(*layers)
        .resolve_scale(x="shared", y="shared")
        .properties(
            width=width,
            height=height,
            title="Hướng tấn công  →",
        )
        .configure(background="#0e3026", autosize={"type": "none"})
        .configure_view(
            strokeWidth=0,
            fill="#0e3026",
            continuousWidth=width,
            continuousHeight=height,
        )
        .configure_title(color="#ffffff", fontSize=13, anchor="end", font="sans-serif")
        .configure_legend(
            labelColor="#e2e8f0",
            titleColor="#e2e8f0",
            symbolStrokeColor="#ffffff",
        )
    )


def plot_freeze_frame(
    players: Sequence[Mapping[str, Any]],
    *,
    x_shooter: float | None = None,
    y_shooter: float | None = None,
    x_goalkeeper: float | None = None,
    y_goalkeeper: float | None = None,
    events: Sequence[Mapping[str, Any]] = (),
    yolo_shooter: Mapping[str, Any] | None = None,
    shooter_keypoints: Sequence[Mapping[str, Any]] = (),
    shooter_bbox: Mapping[str, Any] | None = None,
) -> Any:
    """Plot a possession-relative freeze-frame (shooter, GK, allies, foes)."""

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
    figure, ax = pitch.draw(figsize=(11.5, 7.4))
    figure.patch.set_facecolor("#0e3026")
    for item in players:
        x = item.get("x")
        y = item.get("y")
        if x is None or y is None:
            continue
        side = str(item.get("side") or "ally")
        is_gk = bool(item.get("is_goalkeeper"))
        is_shooter = bool(item.get("is_shooter"))
        color = "#fb4b4e" if side == "foe" else "#38bdf8"
        if is_gk:
            color = "#facc15"
        marker = "*" if is_shooter else "s" if is_gk else "o"
        size = 420 if is_shooter else 260 if is_gk else 90
        pitch.scatter(
            float(x),
            float(y),
            ax=ax,
            s=size,
            marker=marker,
            c=color,
            edgecolors="#ffffff",
            linewidth=1.4,
            zorder=4,
        )
    event_views = [item for item in events if item.get("x") is not None and item.get("y") is not None]
    for start, end in zip(event_views, event_views[1:]):
        ax.plot(
            [float(start["x"]), float(end["x"])],
            [float(start["y"]), float(end["y"])],
            color="#f8fafc",
            alpha=0.45,
            linewidth=1.3,
            zorder=3,
        )
    for index, item in enumerate(event_views, start=1):
        color = ACTION_COLORS.get(str(item.get("action_type")), "#e2e8f0")
        ax.scatter(
            float(item["x"]),
            float(item["y"]),
            s=36,
            c=color,
            edgecolors="#ffffff",
            linewidths=0.8,
            zorder=5,
        )
        ax.text(
            float(item["x"]),
            float(item["y"]) + 1.6,
            str(index),
            color="#ffffff",
            fontsize=7,
            ha="center",
            va="bottom",
            zorder=6,
        )
    if yolo_shooter and yolo_shooter.get("x") is not None and yolo_shooter.get("y") is not None:
        ax.scatter(
            float(yolo_shooter["x"]),
            float(yolo_shooter["y"]),
            s=180,
            facecolors="none",
            edgecolors="#22d3ee",
            linewidths=2.0,
            marker="o",
            zorder=7,
            label="YOLO shooter",
        )
    _draw_pose_on_pitch(ax, shooter_keypoints)
    if x_shooter is not None and y_shooter is not None and x_goalkeeper is not None and y_goalkeeper is not None:
        ax.plot(
            [float(x_shooter), float(x_goalkeeper)],
            [float(y_shooter), float(y_goalkeeper)],
            color="#fde68a",
            linestyle="--",
            linewidth=1.4,
            zorder=3,
        )
    _draw_pose_inset(figure, shooter_keypoints, shooter_bbox)
    legend = ax.legend(
        handles=[
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#38bdf8", markeredgecolor="#ffffff", markersize=12, label="Người sút"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#facc15", markeredgecolor="#ffffff", markersize=10, label="Thủ môn đối phương"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#38bdf8", markeredgecolor="#ffffff", markersize=8, label="Đồng đội"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#fb4b4e", markeredgecolor="#ffffff", markersize=8, label="Đối phương"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#22d3ee", markersize=9, label="YOLO track"),
            Line2D([0], [0], color="#7dd3fc", linewidth=1.6, label="Pose trên sân"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=6,
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


def _draw_pose_on_pitch(ax: Any, keypoints: Sequence[Mapping[str, Any]]) -> None:
    located = {
        int(item["index"]): item
        for item in keypoints
        if "index" in item and item.get("pitch_x") is not None and item.get("pitch_y") is not None
    }
    if len(located) < 2:
        return
    for start, end in COCO_SKELETON:
        if start not in located or end not in located:
            continue
        ax.plot(
            [float(located[start]["pitch_x"]), float(located[end]["pitch_x"])],
            [float(located[start]["pitch_y"]), float(located[end]["pitch_y"])],
            color="#7dd3fc",
            linewidth=1.5,
            alpha=0.9,
            zorder=8,
        )
    ax.scatter(
        [float(item["pitch_x"]) for item in located.values()],
        [float(item["pitch_y"]) for item in located.values()],
        s=18,
        c="#fde68a",
        edgecolors="#0e3026",
        linewidths=0.4,
        zorder=9,
    )


def _draw_pose_inset(
    figure: Any,
    keypoints: Sequence[Mapping[str, Any]],
    bbox: Mapping[str, Any] | None,
) -> None:
    if not keypoints:
        return
    by_index = {int(item["index"]): item for item in keypoints if "index" in item}
    xs = [float(item["x"]) for item in keypoints]
    ys = [float(item["y"]) for item in keypoints]
    if bbox:
        xs.extend([float(bbox["x1"]), float(bbox["x2"])])
        ys.extend([float(bbox["y1"]), float(bbox["y2"])])
    pad = 8.0
    inset = figure.add_axes((0.04, 0.16, 0.16, 0.28))
    inset.set_facecolor("#071a14")
    for start, end in COCO_SKELETON:
        if start not in by_index or end not in by_index:
            continue
        inset.plot(
            [float(by_index[start]["x"]), float(by_index[end]["x"])],
            [float(by_index[start]["y"]), float(by_index[end]["y"])],
            color="#7dd3fc",
            linewidth=1.4,
        )
    inset.scatter(
        [float(item["x"]) for item in keypoints],
        [float(item["y"]) for item in keypoints],
        s=12,
        c="#fde68a",
        zorder=3,
    )
    inset.set_xlim(min(xs) - pad, max(xs) + pad)
    inset.set_ylim(max(ys) + pad, min(ys) - pad)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("#38bdf8")
    inset.set_title("YOLO pose", color="#e2e8f0", fontsize=7, pad=2)


def plot_attack_pitch(
    sequence: AttackSequence | Sequence[Event | ObservedActionV2],
    *,
    attack_relative: bool = True,
) -> Any:
    """Plot events on a metric 105 x 68 metre mplsoccer pitch."""

    items = sequence.events if isinstance(sequence, AttackSequence) else sequence
    return plot_pitch_events(items, attack_relative=attack_relative)
