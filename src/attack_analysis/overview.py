from __future__ import annotations

from collections import Counter
from typing import Mapping

from .benchmarks import FrozenBenchmarkStore
from .models import AttackSequenceV2, TerminalType
from .xg_v2 import ConfirmedXGRecordV2


def attack_v2_research_overview(
    attack: AttackSequenceV2,
    *,
    xg_record: ConfirmedXGRecordV2 | None = None,
    benchmarks: FrozenBenchmarkStore | None = None,
    query: str | None = None,
) -> dict[str, object]:
    """Research-only summary for a confirmed Attack-v2 sequence.

    This is not an effectiveness verdict. Categorical labels stay withheld.
    Numeric comparisons use only a frozen snapshot; live web results are not
    accepted as benchmark evidence.
    """

    if not isinstance(attack, AttackSequenceV2):
        raise TypeError("overview requires AttackSequenceV2")
    actions = attack.possession.actions
    action_counts = dict(Counter(item.action_type for item in actions))
    xg_value = None if xg_record is None else xg_record.xg_probability
    xg_status = "scored" if xg_record is not None else "unknown"
    store = benchmarks or FrozenBenchmarkStore()
    comparisons = ()
    if query:
        comparisons = tuple(record.to_dict() for record in store.search(query))
    elif attack.terminal.body_part:
        comparisons = tuple(
            record.to_dict()
            for record in store.get_role_benchmark("central_forward", "xg")
        )
    return {
        "schema_version": "attack-v2-research-overview/1.0",
        "release_status": "research_only",
        "attack_id": attack.attack_id,
        "match_id": attack.match_id,
        "period": attack.period,
        "team_id": attack.team_id,
        "terminal_type": attack.terminal.terminal_type.value,
        "terminal_allowlisted": attack.terminal.terminal_type
        in {TerminalType.SHOT_ATTEMPT, TerminalType.HEADED_ATTEMPT, TerminalType.FORCED_GOAL_LINE_OUT},
        "duration_sec": attack.possession.end_sec - attack.possession.start_sec,
        "action_count": len(actions),
        "action_counts": action_counts,
        "xg": xg_value,
        "xg_status": xg_status,
        "xg_model_id": None if xg_record is None else xg_record.model_id,
        "comparisons": list(comparisons),
        "effectiveness_verdict": None,
        "abstained_effectiveness": True,
        "limitations": [
            "Research-only overview; not a validated effectiveness verdict.",
            "xG is withheld unless a confirmed shot/headed terminal was scored by xG-v2.",
            "Historical numbers come only from frozen snapshots with season, sample size, source URL and date.",
            "Live web search cannot support numeric claims.",
        ],
        "evidence_ids": list(attack.evidence_ids),
    }


def overview_from_payload(
    attack_payload: Mapping[str, object],
    *,
    xg_payload: Mapping[str, object] | None = None,
    benchmarks: FrozenBenchmarkStore | None = None,
    query: str | None = None,
) -> dict[str, object]:
    attack = AttackSequenceV2.from_dict(dict(attack_payload))
    xg_record = ConfirmedXGRecordV2.from_dict(dict(xg_payload)) if xg_payload else None
    return attack_v2_research_overview(
        attack,
        xg_record=xg_record,
        benchmarks=benchmarks,
        query=query,
    )


__all__ = ["attack_v2_research_overview", "overview_from_payload"]
