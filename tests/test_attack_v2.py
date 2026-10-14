import unittest

from attack_analysis.attack_v2 import (
    FootpassAttackV2Config,
    decode_footpass_attack_v2,
    legacy_attack_to_possession_v2,
    qualify_attack_v2,
)
from attack_analysis.models import (
    ATTACK_TERMINAL_ALLOWLIST,
    AttackSequence,
    AttackSequenceV2,
    Event,
    FieldProvenanceV2,
    ObservedActionV2,
    PossessionSequenceV2,
    PossessionValueV2,
    SequenceStatus,
    SourceTier,
    TerminalEventV2,
    TerminalType,
)


SOURCE = SourceTier.PREDICTED_ACTIONS_ORACLE_TACTICAL


def _action(
    second: float,
    action_type: str,
    team: str | None,
    *,
    frame: int | None = None,
    replay: bool = False,
    beneficiary: str | None = None,
    qualifiers: tuple[str, ...] = (),
    body_part: str | None = None,
    outcome: str | None = None,
) -> ObservedActionV2:
    return ObservedActionV2(
        match_id="game_18_H1",
        period=1,
        frame=round(second * 25) if frame is None else frame,
        timestamp_sec=second,
        action_type=action_type,
        source_tier=SOURCE,
        actor_team_id=team,
        beneficiary_team_id=beneficiary,
        replay=replay,
        body_part=body_part,
        outcome=outcome,
        qualifiers=qualifiers,
        action_probability=0.8,
        actor_probability=0.7,
        timestamp_probability=0.9,
        evidence_ids=(f"evidence-{second}-{action_type}-{team}",),
    )


class AttackV2ModelTests(unittest.TestCase):
    def test_action_id_is_stable_across_enrichment_and_round_trip(self) -> None:
        raw = _action(1.0, "Pass", "A")
        enriched = ObservedActionV2(
            match_id=raw.match_id,
            period=raw.period,
            frame=raw.frame,
            timestamp_sec=raw.timestamp_sec,
            action_type=raw.action_type,
            source_tier=raw.source_tier,
            actor_team_id=raw.actor_team_id,
            possession_team_id="A",
            x=50.0,
            y=30.0,
            action_probability=0.2,
            actor_probability=0.3,
            timestamp_probability=0.4,
        )

        self.assertEqual(raw.action_id, enriched.action_id)
        self.assertEqual(ObservedActionV2.from_dict(raw.to_dict()), raw)

    def test_fused_actions_require_field_level_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "fused actions require field provenance"):
            ObservedActionV2(
                match_id="m",
                period=1,
                frame=1,
                timestamp_sec=0.04,
                action_type="pass",
                source_tier=SourceTier.FUSED,
                actor_team_id="A",
            )

        provenance = tuple(
            FieldProvenanceV2(
                field_name=name,
                source_tier=(
                    SourceTier.PREDICTED_ACTIONS_ORACLE_TACTICAL
                    if name == "actor_team_id"
                    else SourceTier.VIDEO_ONLY
                ),
            )
            for name in ("frame", "timestamp_sec", "action_type", "replay", "actor_team_id")
        )
        action = ObservedActionV2(
            match_id="m",
            period=1,
            frame=1,
            timestamp_sec=0.04,
            action_type="pass",
            source_tier=SourceTier.FUSED,
            actor_team_id="A",
            field_provenance=provenance,
            association_id="association:track-to-action:1",
            evidence_ids=("association:track-to-action:1",),
        )
        self.assertEqual(action.source_tier, SourceTier.FUSED)

    def test_strict_attack_invariants_for_all_terminal_types(self) -> None:
        action = _action(1.0, "pass", "A")
        for terminal_type in TerminalType:
            with self.subTest(terminal_type=terminal_type):
                beneficiary = "A" if terminal_type is TerminalType.FORCED_GOAL_LINE_OUT else None
                last_touch = "B" if terminal_type is TerminalType.FORCED_GOAL_LINE_OUT else None
                cause = "defender_deflection_out" if terminal_type is TerminalType.FORCED_GOAL_LINE_OUT else None
                terminal = TerminalEventV2(
                    match_id=action.match_id,
                    period=action.period,
                    frame=action.frame,
                    timestamp_sec=action.timestamp_sec,
                    terminal_type=terminal_type,
                    possession_team_id="A",
                    source_tier=SOURCE,
                    beneficiary_team_id=beneficiary,
                    last_touch_team_id=last_touch,
                    cause=cause,
                    evidence_ids=(action.action_id,),
                )
                possession = PossessionSequenceV2(
                    match_id=action.match_id,
                    period=action.period,
                    possession_team_id="A",
                    start_frame=action.frame,
                    end_frame=action.frame,
                    start_sec=action.timestamp_sec,
                    end_sec=action.timestamp_sec,
                    actions=(ObservedActionV2.from_dict({**action.to_dict(), "possession_team_id": "A"}),),
                    status=(
                        SequenceStatus.UNCERTAIN
                        if terminal_type is TerminalType.UNKNOWN
                        else SequenceStatus.COMPLETE
                    ),
                    source_tier=SOURCE,
                    terminal=terminal,
                )
                if terminal_type in ATTACK_TERMINAL_ALLOWLIST:
                    self.assertIsInstance(AttackSequenceV2(possession), AttackSequenceV2)
                else:
                    with self.assertRaises(ValueError):
                        AttackSequenceV2(possession)

    def test_forced_out_contract_requires_opponent_last_touch_and_cause(self) -> None:
        common = {
            "match_id": "game_18_H1",
            "period": 1,
            "frame": 50,
            "timestamp_sec": 2.0,
            "terminal_type": TerminalType.FORCED_GOAL_LINE_OUT,
            "possession_team_id": "A",
            "source_tier": SOURCE,
            "beneficiary_team_id": "A",
        }
        with self.assertRaisesRegex(ValueError, "opponent .* last touch"):
            TerminalEventV2(**common, last_touch_team_id="A", cause="defender_block_out")
        with self.assertRaisesRegex(ValueError, "observed save, block, or deflection"):
            TerminalEventV2(**common, last_touch_team_id="B")

        terminal = TerminalEventV2(
            **common,
            last_touch_team_id="B",
            cause="keeper_save_out",
        )
        self.assertTrue(terminal.qualifies_as_attack)

    def test_only_possessions_with_an_observed_end_can_be_attacks(self) -> None:
        decoded = decode_footpass_attack_v2(
            [_action(0, "pass", "A"), _action(20, "shot", "A")],
            FootpassAttackV2Config(max_observation_gap_sec=4),
        )
        open_possession, shot_possession = decoded.possessions
        self.assertEqual(open_possession.status, SequenceStatus.RIGHT_CENSORED)
        self.assertIsNone(qualify_attack_v2(open_possession))
        with self.assertRaisesRegex(ValueError, "observed end"):
            AttackSequenceV2(open_possession)

        self.assertEqual(shot_possession.status, SequenceStatus.LEFT_CENSORED)
        self.assertIsNotNone(qualify_attack_v2(shot_possession))
        self.assertEqual([item.possession for item in decoded.attacks], [shot_possession])

    def test_possession_value_has_no_attack_flag_and_round_trips(self) -> None:
        value = PossessionValueV2(
            possession_id="posv2_example",
            source_tier=SourceTier.VIDEO_ONLY,
            peak_progression_m=42.0,
            box_entry=True,
            xt=0.12,
            epv=0.08,
            p_score_10s=0.2,
            p_concede_10s=0.03,
        )
        self.assertFalse(hasattr(value, "is_attack"))
        self.assertEqual(PossessionValueV2.from_dict(value.to_dict()), value)


class FootpassAttackV2DecoderTests(unittest.TestCase):
    def test_failed_defensive_touches_do_not_split_possession(self) -> None:
        result = decode_footpass_attack_v2(
            [
                _action(1, "drive", "A"),
                _action(2, "tackle", "B"),
                _action(3, "block", "B"),
                _action(4, "header", "B"),
                _action(5, "pass", "A"),
                _action(6, "shot", "A"),
            ]
        )

        self.assertEqual(len(result.possessions), 1)
        self.assertEqual(len(result.attacks), 1)
        possession = result.possessions[0]
        self.assertEqual(possession.possession_team_id, "A")
        self.assertEqual({action.actor_team_id for action in possession.actions}, {"A", "B"})
        self.assertTrue(all(action.possession_team_id == "A" for action in possession.actions))
        self.assertEqual(possession.terminal.terminal_type, TerminalType.SHOT_ATTEMPT)

    def test_generic_header_is_not_an_attack_terminal(self) -> None:
        generic = decode_footpass_attack_v2(
            [_action(1, "pass", "A"), _action(2, "header", "A")]
        )
        explicit = decode_footpass_attack_v2(
            [
                _action(1, "pass", "A"),
                _action(2, "header", "A", qualifiers=("headed_attempt",)),
            ]
        )

        self.assertEqual(len(generic.attacks), 0)
        self.assertEqual(generic.possessions[0].status, SequenceStatus.RIGHT_CENSORED)
        self.assertEqual(len(explicit.attacks), 1)
        self.assertEqual(explicit.attacks[0].terminal.terminal_type, TerminalType.HEADED_ATTEMPT)

    def test_terminal_without_established_control_is_left_censored_attack(self) -> None:
        lone_shot = decode_footpass_attack_v2([_action(1, "shot", "A")])
        contested_then_shot = decode_footpass_attack_v2(
            [_action(1, "tackle", "B"), _action(2, "shot", "A")]
        )

        self.assertEqual(lone_shot.possessions[0].status, SequenceStatus.LEFT_CENSORED)
        self.assertEqual(len(lone_shot.attacks), 1)
        self.assertEqual(
            contested_then_shot.possessions[0].status,
            SequenceStatus.LEFT_CENSORED,
        )
        self.assertEqual(len(contested_then_shot.attacks), 1)

    def test_unknown_replay_status_cannot_form_complete_attack(self) -> None:
        result = decode_footpass_attack_v2(
            [
                _action(1, "pass", "A", qualifiers=("replay_status_unknown",)),
                _action(2, "shot", "A", qualifiers=("replay_status_unknown",)),
            ]
        )
        self.assertEqual(len(result.attacks), 0)
        self.assertEqual(result.possessions[0].status, SequenceStatus.UNCERTAIN)

    def test_scene_cut_censors_both_sides(self) -> None:
        result = decode_footpass_attack_v2(
            [
                _action(1, "pass", "A"),
                _action(2, "scene_cut", None),
                _action(3, "pass", "A"),
                _action(4, "shot", "A"),
            ]
        )
        self.assertEqual(
            [item.status for item in result.possessions],
            [SequenceStatus.RIGHT_CENSORED, SequenceStatus.LEFT_CENSORED],
        )
        self.assertEqual([item.possession for item in result.attacks], [result.possessions[1]])

    def test_shot_and_defensive_block_in_same_cluster_form_one_attack(self) -> None:
        result = decode_footpass_attack_v2(
            [
                _action(1, "pass", "A"),
                _action(2.0, "shot", "A"),
                _action(2.2, "block", "B"),
            ]
        )

        self.assertEqual(len(result.attacks), 1)
        self.assertEqual(len(result.attacks[0].possession.actions), 3)
        self.assertEqual(result.attacks[0].terminal.terminal_type, TerminalType.SHOT_ATTEMPT)

    def test_controlled_rebound_stays_in_the_same_attack(self) -> None:
        result = decode_footpass_attack_v2(
            [
                _action(1.0, "pass", "A"),
                _action(2.0, "shot", "A"),
                _action(2.1, "block", "B"),
                _action(2.2, "drive", "A"),
                _action(2.4, "shot", "A"),
            ]
        )

        self.assertEqual(len(result.attacks), 1)
        attack = result.attacks[0]
        self.assertEqual(attack.terminal.action_id, _action(2.4, "shot", "A").action_id)
        action_ids = [item.action_id for item in attack.possession.actions]
        self.assertEqual(
            action_ids,
            [
                _action(1.0, "pass", "A").action_id,
                _action(2.0, "shot", "A").action_id,
                _action(2.1, "block", "B").action_id,
                _action(2.2, "drive", "A").action_id,
                _action(2.4, "shot", "A").action_id,
            ],
        )

    def test_same_team_shot_outside_rebound_window_opens_a_new_attack(self) -> None:
        result = decode_footpass_attack_v2(
            [
                _action(1.0, "pass", "A"),
                _action(2.0, "shot", "A"),
                _action(6.0, "drive", "A"),
                _action(7.0, "shot", "A"),
            ]
        )

        self.assertEqual(len(result.attacks), 2)
        self.assertEqual(result.attacks[0].terminal.action_id, _action(2.0, "shot", "A").action_id)
        self.assertEqual(result.attacks[1].terminal.action_id, _action(7.0, "shot", "A").action_id)

    def test_throw_in_closes_previous_possession_and_starts_restart(self) -> None:
        result = decode_footpass_attack_v2(
            [
                _action(1, "pass", "A"),
                _action(2, "throw_in", "B"),
                _action(3, "pass", "B"),
                _action(4, "shot", "B"),
            ]
        )

        self.assertEqual(len(result.possessions), 2)
        self.assertEqual(result.possessions[0].terminal.terminal_type, TerminalType.UNFORCED_OUT)
        self.assertEqual(len(result.attacks), 1)
        restart_attack = result.attacks[0]
        self.assertEqual(restart_attack.team_id, "B")
        self.assertEqual(restart_attack.possession.actions[0].action_type, "throw_in")

    def test_throw_in_alone_is_never_a_singleton_attack(self) -> None:
        result = decode_footpass_attack_v2([_action(1, "throw_in", "A")])
        self.assertEqual(len(result.attacks), 0)
        self.assertEqual(len(result.possessions), 1)
        self.assertEqual(result.possessions[0].status, SequenceStatus.RIGHT_CENSORED)

    def test_large_gap_censors_both_sides_instead_of_creating_attacks(self) -> None:
        result = decode_footpass_attack_v2(
            [
                _action(0, "drive", "A"),
                _action(12, "pass", "A"),
                _action(13, "shot", "A"),
            ],
            FootpassAttackV2Config(max_observation_gap_sec=4),
        )

        self.assertEqual([item.status for item in result.possessions], [
            SequenceStatus.RIGHT_CENSORED,
            SequenceStatus.LEFT_CENSORED,
        ])
        self.assertEqual([item.possession for item in result.attacks], [result.possessions[1]])

    def test_replay_actions_are_excluded_from_live_decoding(self) -> None:
        replay_shot = _action(1, "shot", "A", replay=True)
        result = decode_footpass_attack_v2(
            [replay_shot, _action(2, "pass", "A")]
        )

        self.assertEqual(result.excluded_replay_action_ids, (replay_shot.action_id,))
        self.assertEqual(len(result.attacks), 0)
        self.assertTrue(all(not action.replay for item in result.possessions for action in item.actions))

    def test_conflicting_same_frame_predictions_do_not_flip_team(self) -> None:
        result = decode_footpass_attack_v2(
            [
                _action(1, "drive", "A"),
                _action(2, "pass", "A", frame=50),
                _action(2, "pass", "B", frame=50),
                _action(3, "shot", "A"),
            ]
        )

        self.assertEqual(len(result.possessions), 1)
        self.assertEqual(len(result.attacks), 1)
        self.assertEqual(result.attacks[0].team_id, "A")

    def test_forced_out_requires_explicit_qualifier_and_attacking_restart(self) -> None:
        ordinary = decode_footpass_attack_v2(
            [_action(1, "cross", "A"), _action(2, "block", "B")]
        )
        forced = decode_footpass_attack_v2(
            [
                _action(1, "cross", "A"),
                _action(
                    2,
                    "block",
                    "B",
                    beneficiary="A",
                    qualifiers=("forced_goal_line_out",),
                ),
            ]
        )

        self.assertEqual(len(ordinary.attacks), 0)
        self.assertEqual(len(forced.attacks), 1)
        self.assertEqual(forced.attacks[0].terminal.terminal_type, TerminalType.FORCED_GOAL_LINE_OUT)

    def test_forced_out_requires_verified_cause(self) -> None:
        result = decode_footpass_attack_v2(
            [
                _action(1, "cross", "A"),
                _action(
                    2,
                    "header",
                    "B",
                    beneficiary="A",
                    qualifiers=("forced_goal_line_out",),
                ),
            ]
        )
        self.assertEqual(len(result.attacks), 0)

    def test_attacker_last_touch_cannot_be_forced_out(self) -> None:
        result = decode_footpass_attack_v2(
            [
                _action(1, "cross", "A"),
                _action(
                    2,
                    "forced_goal_line_out",
                    "A",
                    beneficiary="A",
                    qualifiers=("forced_goal_line_out",),
                ),
            ]
        )
        self.assertEqual(len(result.attacks), 0)

    def test_decoder_is_deterministic_for_reordered_input(self) -> None:
        actions = [
            _action(1, "pass", "A"),
            _action(2, "tackle", "B"),
            _action(3, "shot", "A"),
        ]
        first = decode_footpass_attack_v2(actions)
        second = decode_footpass_attack_v2(reversed(actions))
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_sequence_and_attack_ids_are_stable(self) -> None:
        actions = [_action(1, "pass", "A"), _action(2, "shot", "A")]
        first = decode_footpass_attack_v2(actions)
        round_tripped = AttackSequenceV2.from_dict(first.attacks[0].to_dict())
        second = decode_footpass_attack_v2(actions)

        self.assertEqual(round_tripped, first.attacks[0])
        self.assertEqual(first.possessions[0].possession_id, second.possessions[0].possession_id)
        self.assertEqual(first.attacks[0].attack_id, second.attacks[0].attack_id)


class AttackV2LegacyConversionTests(unittest.TestCase):
    def test_legacy_shot_is_only_an_uncertain_possession_candidate(self) -> None:
        legacy = AttackSequence(
            attack_id="legacy-1",
            match_id="game_18_H1",
            team="A",
            start_sec=1,
            end_sec=2,
            events=(
                Event(1, "pass", "A", frame=25),
                Event(2, "shot", "A", frame=50),
            ),
        )
        legacy.validate()

        possession = legacy_attack_to_possession_v2(legacy)

        self.assertEqual(possession.status, SequenceStatus.UNCERTAIN)
        self.assertEqual(possession.source_tier, SourceTier.LEGACY_UNVALIDATED)
        self.assertIsNone(possession.terminal)
        self.assertIsNone(qualify_attack_v2(possession))

    def test_legacy_converter_requires_stable_match_context(self) -> None:
        legacy = AttackSequence(
            attack_id="legacy-1",
            team="A",
            start_sec=1,
            end_sec=1,
            events=(Event(1, "pass", "A"),),
        )
        with self.assertRaisesRegex(ValueError, "match_id is required"):
            legacy_attack_to_possession_v2(legacy)


if __name__ == "__main__":
    unittest.main()
