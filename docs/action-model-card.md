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
