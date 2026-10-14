# Attack-v2: ontology, contracts and validation workflow

## Release status

Attack-v2 is the only current attack ontology. The deterministic decoder and all
evaluation infrastructure are implemented and unit-tested, but the release is
still `research_only`: continuous human labels have not been completed and no
trained temporal model has passed the frozen human-development gates.

Legacy `Event`/`AttackSequence` APIs remain readable for old artifacts. Their
segments are emitted as `PossessionCandidateV1` with
`validation_status=legacy_unvalidated`; they cannot enter Attack-v2 metrics,
xG-v2 or an effectiveness verdict.

The legacy calibrated-window path follows the same rule. It persists bounded
raw evidence as `selected-possession-candidate-v1.json`, never as an attack
analysis. Missing left/right boundaries are censored and a terminal outside the
requested window is never borrowed. A V1 `forced_goal_line_out` is retained for
audit but fails terminal validation because V1 cannot independently encode the
beneficiary, defensive last touch and cause. The legacy derivation writes a
separate `possession-candidate-v1-manifest.json`; it never overwrites the
validated upstream vision manifest and cannot be accepted by the v2 cache.

## Locked ontology

`PossessionSequenceV2` and `AttackSequenceV2` are separate objects. An attack
must reference one possession whose end is observed (`complete` or
`left_censored`) and end with exactly one of:

- `shot_attempt`;
- `headed_attempt`, which requires evidence of intent toward goal;
- `forced_goal_line_out`, which requires a defensive save/block/deflection and
  a restart awarded to the attacking team.

Turnover, ordinary out, foul, offside and period end close possession but do not
create an attack. A generic header, block or tackle is contested evidence and
does not switch owner. A throw-in backfills an approximate ordinary out for the
previous possession and starts the next possession. Gaps, cuts and truncated
windows produce censoring, never a football terminal. A rebound controlled
after a shot starts a new sequence.

A left-censored possession only lacks its build-up (restart after a stoppage,
set piece, penalty, short header sequence after a cross, first action after a
cut); its terminal is observed, so it still qualifies. Right-censored and
uncertain possessions never do. Requiring `complete` dropped most goals: on 12
scoreboard-checked train matches only 17/44 scoring attempts were in the list,
against 44/44 with the observed-end rule.

The current video trajectory rule emits `shot_candidate`, not `shot`; it has no
xG and cannot qualify an attack.

## Public contracts

The v2 contracts live in `src/attack_analysis/models.py`:

- `ObservedActionV2` has stable content ID, match/period/frame/time, independent
  action/actor/time probabilities, actor/possession/beneficiary teams,
  direction, normalized coordinates, role/jersey/bbox/replay, outcome and
  field-level provenance;
- `TerminalEventV2` contains terminal type, body part, outcome, last touch,
  beneficiary, cause and evidence;
- `PossessionSequenceV2` contains both teams' actions, state transitions,
  terminal and `complete | left_censored | right_censored | uncertain` status;
- `AttackSequenceV2` enforces an observed end (`ATTACK_QUALIFYING_STATUSES`)
  and the terminal allowlist;
- `PossessionValueV2` carries measurable threat components but has no
  `is_attack` flag.

Stable sequence IDs hash match, period, stable team, start frame, terminal
frame/type and rubric version. V1 conversion always yields an uncertain
`legacy_unvalidated` possession without a terminal.

## Source tiers and provenance

Exactly four validated source tiers are accepted:

- `oracle_actions_oracle_tactical`;
- `predicted_actions_oracle_tactical`;
- `video_only`;
- `fused` only after a real association stage supplies provenance for every
  populated field.

FOOTPASS direction 0/1 is not team identity. The adapter uses the stable HDF5
player-roster namespace across halves and keeps direction as a period property.
Oracle bbox/replay flags are preserved. Raw FOOTPASS `header` remains generic.

Every persisted v2 run/cache manifest fingerprints input video, datasets,
source tree, config, schema/rubric, postprocess and models. Missing or mismatched
fields make the artifact stale; legacy manifests never pass cache validation.

## Human gold

The frozen match split is encoded in `gold_annotations.py`:

- train: games 6, 8, 16, 21, 33, 48;
- development: games 18, 24, 47;
- final test: games 0, 36, 50.

The Streamlit annotation tab reads the 12-match manifest and presents continuous
video without model-generated candidates. Test labels use an isolated writer.
Gold manifests fingerprint every available video with SHA-256; size and mtime
are not identities. A newly built manifest reports `test_labels_sealed=false`.
It may report true only when `--sealed-directory` passes full seal verification.

Period declarations use inclusive frame bounds, an explicit FPS and start/end
times. The coverage validator requires every frame to have exactly one human
label: `possession`, `attack`, `uncertain`, or `dead_ball`. It reports gaps and
overlaps and never fills them automatically. `seal-gold-test` additionally
requires both periods of exactly game 0, 36 and 50, complete continuous coverage,
and records full SHA-256 hashes for the label bytes/content and period-declaration
bytes/content. Further appends fail. Normal readers reject test labels.
Intra-rater comparison uses temporal one-to-one matching so legitimate boundary
differences do not destroy pairing.

```powershell
attack-analysis gold-status
attack-analysis build-gold-manifest "G:\My Drive\footpass-videos" data\attack-v2-gold-manifest.json
attack-analysis validate-gold-annotations data\labels\attack-v2-development.jsonl `
  --period-intervals data\labels\development-periods.json
attack-analysis gold-intra-rater round-1.jsonl round-2.jsonl
attack-analysis seal-gold-test data\labels\sealed-test `
  --period-intervals data\labels\frozen-test-periods.json
attack-analysis verify-gold-test-seal data\labels\sealed-test
attack-analysis build-gold-manifest "G:\My Drive\footpass-videos" `
  data\attack-v2-gold-manifest-sealed.json `
  --sealed-directory data\labels\sealed-test
```

Start on development games 18/24/47. Do not open frozen test to tune. Full order:
[gold-annotation-runbook.md](gold-annotation-runbook.md). `gold-status` never reads
test labels. P11 stays `open` until labels exist **and** xG is human-domain validated.


The period declaration file contains `schema_version`, `rubric_version`,
`frame_interval_convention: "inclusive"`, and a `period_intervals` array. Each
entry declares `match_id`, `period`, `start_frame`, `end_frame`, `start_sec`,
`end_sec`, and `fps`.

## Model infrastructure

- `action_spotting_v2.py`: full-record match/class-balanced sampling, hard
  negatives, 30-second windows with 50% overlap, center-weighted merge,
  constrained decode, development-only threshold freeze, 12-config/3-seed and
  RTX 4050 limits;
- `temporal_model.py`: 5 Hz features, six-state 30-second dilated TCN contract,
  nine multitask heads, class-balanced focal plus structured NLL specification,
  and deterministic whole-period semi-Markov decoding;
- `tracking_evaluation.py`: AP50-compatible detection metrics, IDF1, HOTA,
  team/jersey coverage, ball recall and a bbox GS-HOTA-style diagnostic. The
  latter is explicitly not the official SoccerNet evaluator;
- `xg_v2.py` and `xg_training_v2.py`: explicit unknown/missingness semantics,
  grouped inner model selection, disjoint calibration, untouched outer test,
  Brier/ECE/slope/intercept and paired match-bootstrap comparison.

These modules describe and test the training/evaluation path. They do not imply
that a learned TCN/action/vision model has already reached acceptance gates.

## Evaluation commands

```powershell
# Strict FOOTPASS half; V1 is nested only as a compatibility ablation
attack-analysis footpass-ingest ..\FOOTPASS\playbyplay_GT\playbyplay_val.json game_18_H1 `
  --source oracle --tactical-h5 ..\FOOTPASS\data\val_tactical_data.h5 `
  --output runs\game_18_H1.attack-v2.json

# Full current-geometry oracle-assisted audit with video/dataset/tree hashes
attack-analysis audit-footpass-v2 `
  ..\FOOTPASS\playbyplay_GT\playbyplay_val.json `
  ..\FOOTPASS\data\val_tactical_data.h5 `
  "G:\My Drive\footpass-videos" `
  runs\verification-v2\footpass-oracle-audit.json

# Human sequence metrics: Hungarian one-to-one, per-match slices, tIoU
# .25/.5/.75 and a paired 2,000-resample comparison with the frozen baseline.
attack-analysis evaluate-attack-v2 development-gold.jsonl predictions.json `
  --frozen-baseline frozen-baseline.attack-v2.json `
  --output runs\development-sequence-metrics.json

# xG-v2 dataset and grouped nested-CV training
attack-analysis build-xg-v2-dataset ..\xg-from-videos\data\open-data\data data\xg-shots-v2.csv
attack-analysis train-xg-v2 data\xg-shots-v2.csv models\xg-v2.joblib
```

## Release rule

`freeze-attack-v2-release` binds the config, model, schema, rubric,
development data **and strict baseline prediction file** into release schema
`attack-v2-frozen-release/v2`. The final evaluator accepts only the sealed
`attack-v2-frozen-test.jsonl`, requires that manifest plus a one-shot ledger,
and rejects a baseline whose content hash differs from the frozen release.
Missing paired-baseline evidence is a failed acceptance gate, not a zero-valued
improvement.

`validated` is allowed only when human-development/frozen-test metrics,
invariants, reproducibility and runtime gates pass. Until then, the UI exposes
possessions and threat components as research-only, withholds xG where terminal
or feature evidence is insufficient, and never lowers the ontology threshold.
