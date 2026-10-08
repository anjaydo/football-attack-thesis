# Action spotting model card

## Selected development baseline

The predicted path uses FOOTPASS TAAD + DST at confidence threshold 0.15. Raw
TAAD remains available only as an ablation. Tactical coordinates are still
oracle-enriched, so this is `predicted-actions/oracle-tracking`, not a complete
video-only result.

## Validation protocol

- Dataset: six halves from FOOTPASS validation (`game_18`, `game_24`, `game_47`).
- Matching: match, team, jersey and action class; tolerance ±12 frames.
- Primary threshold: 0.15, matching the FOOTPASS evaluation protocol.
- Threshold experiment: select a global threshold on `game_18`, then report it
  on held-out `game_24` and `game_47`.
- Challenge matches are never used for development.

## Results

| Model | Threshold | Precision | Recall | Micro-F1 | Macro-F1 | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TAAD | 0.15 | 0.304 | 0.630 | 0.410 | 0.243 | 12,576 |
| DST | 0.15 | 0.735 | 0.703 | 0.719 | 0.493 | 5,806 |
| DST precision ablation | 0.60 | 0.783 | 0.685 | 0.731 | 0.497 | 5,318 |

### H100 TAAD checkpoint (raw, ablation)

`best_model.pt` from the H100 run (20 epochs, best val CE 0.02704 at epoch 19,
SHA-256 `4ac120d8…0076`) was inferred locally on the RTX 4050 (batch 4, about
26 min per half, videos `D:\videos`) and scored with FOOTPASS `evaluation.py`
(τ=0.15, δ=12):

| Source | Precision | Recall | Micro-F1 | Predictions |
| --- | ---: | ---: | ---: | ---: |
| `TAAD_v1` (original checkpoint) | 0.304 | 0.630 | 0.410 | 12,576 |
| `TAAD_h100` | 0.244 | 0.599 | 0.346 | 14,906 |

Lower val CE did not translate into better raw spotting: precision fell on
every class except pass. Raw TAAD is only DST input, so this does not decide
the DST outcome. `TAAD_h100` stays an ablation in `models/prediction-sources.json`;
`DST_v1` remains primary until a DST file beats 0.719
([prediction-sources.md](prediction-sources.md)).

### H100 DST full pipeline (ablation)

The H100 job finished the DST stage (15 epochs, `best_model.pt` SHA-256
`f0acbab0…defa`) and wrote val JSON for the six halves. Local
`evaluation.py` (τ=0.15, δ=12) matches the server log:

| Source | Precision | Recall | Micro-F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DST_v1` | 0.735 | 0.703 | 0.719 | 4267 | 1539 | 1803 |
| `DST_h100` | 0.706 | 0.671 | 0.688 | 4076 | 1700 | 1994 |

Shot class on `DST_h100`: precision 0.564, recall 0.463 (31 TP, 24 FP, 36 FN, 67 GT).
`DST_h100` is registered as an ablation. It is not primary. The delivered JSON
was first saved on top of `playbyplay_DST_val.json`, so the pinned `DST_v1`
bytes (SHA-256 `34ef3f16…5741`) are no longer at that path. The H100 file is
`playbyplay_DST_h100_val.json` (SHA-256 `1beba6ae…68ab`).

The threshold selected on the `game_18` pilot was 0.60. On held-out
`game_24`/`game_47`, it improves micro-F1 from 0.726 to 0.736, but suppresses
too many shot and possession boundaries for the downstream objective. Therefore
0.15 remains the primary setting and 0.60 is reported only as an ablation.

## Downstream propagation

| Path | Attacks | Shot attacks | Start boundary F1 | End boundary F1 |
| --- | ---: | ---: | ---: | ---: |
| Oracle-derived segmentation | 1,360 | 67 | — | — |
| Raw TAAD | 5,849 | 304 | 0.230 | 0.251 |
| DST 0.15 | 1,215 | 66 | 0.701 | 0.703 |
| DST 0.60 | 1,114 | 46 | 0.656 | 0.665 |

The table above is the historical V1 segmentation. Attack-v2 decode on the same
six halves (`runs/compare_prediction_sources.py` for the earlier rows,
`runs/verify_dst_h100_xg.py` for `DST_h100`; oracle decode = 755
possessions, 46 attacks, 45 ending in a shot or header):

| Source | Possessions | Attacks | Possession start / end F1 | Attack start / end F1 |
| --- | ---: | ---: | ---: | ---: |
| `DST_v1` (primary) | 696 | 42 | 0.647 / 0.652 | 0.386 / 0.591 |
| `DST_h100` | 700 | 38 | 0.609 / 0.608 | 0.333 / 0.452 |
| `TAAD_v1` | 1,862 | 194 | 0.296 / 0.302 | 0.050 / 0.167 |
| `TAAD_h100` | 2,357 | 295 | 0.264 / 0.273 | 0.041 / 0.111 |

Raw TAAD over-segments possessions about 3x; the H100 checkpoint does so more
than the original.

The low `DST_v1` attack-start F1 is a terminal-detection problem, not a start
logic problem (`runs/diagnose_attack_start.py`). Matching oracle attacks to DST
attacks by terminal time (±1 s) pairs 26 of 46; 20 oracle terminals have no DST
counterpart. Among the 26 pairs, 16 starts fall within ±1 s, 4 start later and
6 earlier, with a median offset of −0.12 s. The decoder start rule is therefore
left unchanged; tuning it on these val halves would also leak into evaluation.

Boundary matching uses a ±1 second tolerance against boundaries derived from
oracle actions. A manually annotated pilot remains required for the thesis's
final boundary-F1 claim.

## Remaining weaknesses

- Header, tackle and block remain weak and class-imbalanced.
- Shot F1 is 0.602 at the primary threshold; false and missed shots must remain
  visible in the error analysis.
- These predictions do not validate broadcast-video tracking or calibration.
- Final model selection must be rerun after annotation rules and the test split
  are frozen.
