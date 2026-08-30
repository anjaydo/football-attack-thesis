# Research pipeline: FOOTPASS to grounded verdict

> **Historical V1 runbook.** Counts such as 1,360/1,215 are heuristic
> possession candidates, not attacks. Geometry xG and V1 categorical verdicts
> are disabled in current output. Use [attack-v2.md](attack-v2.md) for the
> current ontology, continuous annotation workflow and evaluator.

This document describes the implemented research path. FOOTPASS videos remain on
the external drive and are not copied into the repository.

## Install

Use the existing FOOTPASS Python 3.11 environment or install all research extras:

```powershell
python -m pip install -e ".[research,vision]"
$env:FOOTPASS_VIDEOS_DIR = "D:\videos"
```

OpenAI calls are optional. The deterministic grounded baseline works without an
API key. To enable the Responses API backend, set `OPENAI_API_KEY` and select API
mode in the Streamlit app. The adapter uses strict custom function tools and a
JSON Schema response contract following the official
[Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).

## 1. Build oracle and predicted Attack-v2 sequences

Oracle events and tactical coordinates:

```powershell
attack-analysis footpass-ingest `
  ..\FOOTPASS\playbyplay_GT\playbyplay_val.json game_18_H1 `
  --source oracle `
  --tactical-h5 ..\FOOTPASS\data\val_tactical_data.h5 `
  --output runs\game_18_H1.oracle.json
```

DST-denoised predictions enriched with the same tactical coordinate contract:

```powershell
attack-analysis footpass-ingest `
  ..\FOOTPASS\playbyplay_PRED\playbyplay_DST_val.json game_18_H1 `
  --source predicted `
  --model-name DST --min-confidence 0.15 `
  --tactical-h5 ..\FOOTPASS\data\val_tactical_data.h5 `
  --output runs\game_18_H1.predicted.json
```

The current output contains `ObservedActionV2`, possessions and strict attacks.
The old segments are nested only under `legacy_compatibility` as
`PossessionCandidateV1`. xG is withheld until a v2 terminal and xG-v2 feature
contract are independently validated.

Use raw TAAD only as an ablation. On the six validation halves, the official
strict evaluator at ±12 frames improves from TAAD micro-F1 0.410 to DST 0.719.
DST produces 1,215 **legacy candidates** and 66 candidates containing a shot,
versus oracle 1,360/67 and raw TAAD 5,849/304. These counts are diagnostic only.
At a ±1 second tolerance, historical V1 start/end boundary
F1 rises from 0.230/0.251 for TAAD to 0.701/0.703 for DST. Reproduce the action
comparison with:

```powershell
attack-analysis evaluate-footpass-actions `
  ..\FOOTPASS\playbyplay_GT\playbyplay_val.json `
  ..\FOOTPASS\playbyplay_PRED\playbyplay_DST_val.json `
  --model-name DST --min-confidence 0.15 --delta 12
```

The demo also exposes a precision-oriented threshold of 0.60. It was selected
on `game_18` only and evaluated on held-out `game_24`/`game_47`; it raises
held-out micro-F1 from 0.726 to 0.736, but reduces shot/attack recall, so 0.15
remains the primary downstream setting.

## 2. Create the annotation set

Generate attacks for multiple train matches, combine their `attacks` arrays, then:

```powershell
attack-analysis build-annotations runs\train-attacks.json data\annotation-manifest.json `
  --sample-size 300 --seed 42
attack-analysis serve-demo
```

Use the Annotation tab to record label, score, confidence and reason as JSONL.
Create the second-round 20% sample only after the first labels have been frozen.

## 3. Train research baselines

The xG CSV must contain `match_id`, binary `goal`, and the observable columns
listed in `attack_analysis.ml.XG_FEATURE_COLUMNS`:

```powershell
attack-analysis train-xg data\xg-shots.csv models\xg.joblib
attack-analysis train-effectiveness data\effectiveness.jsonl models\effectiveness.joblib
```

Both commands split by match and persist the selected model, split manifest and
validation/test metrics. xG selects by validation Brier score; effectiveness
selects by validation macro-F1.

The `attack_analysis.research_evaluation` module supplies class-wise action
spotting F1 at ±12 frames, attack boundary F1, median/P90 top-view error, and
cluster-bootstrap confidence intervals that resample whole matches.

## 4. Frozen role benchmark

```powershell
attack-analysis build-footpass-benchmarks `
  ..\FOOTPASS\data\train_tactical_data.h5 data\benchmarks.jsonl `
  --season 2023-24 --snapshot-date 2026-08-23
```

Named-player enrichment requires a separate `data/identities.jsonl`. Each line
must contain `match_id`, `team`, `jersey`, `player_name`, `role` and a source ID.
The runtime refuses to infer a player from jersey number without this exact mapping.

## 5. Demo and evidence contract

```powershell
attack-analysis serve-demo --port 8501
```

For an exploratory external clip, `attack-analysis analyze-clip VIDEO RUN_DIR`
is an alias of the complete CV pipeline. Without calibrated tracking and a ball
quality gate, its evidence contract abstains instead of emitting xG claims.

The FOOTPASS tab decodes Attack-v2, plots possession-relative pitch coordinates,
runs a YOLO window for spatial context, then **Đưa ra kết luận** builds an
`AttackEvidenceBundle`: FOOTPASS terminal + YOLO `observations.jsonl` + local
`models/xg-v2.joblib` (research_only). The conclusion is `attack-verdict-v2`
(chance quality, tactical read, frozen benchmarks, optional Brave Search context).
It does **not** emit a categorical effectiveness label.

```powershell
# Frozen role benchmarks (optional numeric comparison)
attack-analysis build-footpass-benchmarks ..\FOOTPASS\data\val_tactical_data.h5 data\benchmarks.jsonl

# MCP stdio tools: get_attack_evidence, frozen stats, Brave context, verify_claims
attack-analysis serve-mcp --benchmarks data\benchmarks.jsonl
```

Set `BRAVE_SEARCH_API_KEY` for non-numeric web snippets. Set `XG_V2_MODEL_PATH`
if the joblib artifact is not at `models/xg-v2.joblib`.

The three project skills under `.cursor/skills/` are versioned prompt modules.
The API client loads them, exposes only the whitelist in `EvidenceToolRegistry`,
uses a strict JSON schema, and rejects a verdict that cites unavailable evidence
or skips `verify_claims`.

## Current scientific boundaries

- FOOTPASS tactical enrichment is an oracle coordinate path even when actions are
  DST/TAAD predictions; report it as predicted-actions/oracle-tracking, not fully video-only.
- Geometry xG is a transparent fallback, not the final trained xG model.
- Uploaded arbitrary video must abstain until tracking, calibration and event gates pass.
- Challenge matches are not used for model selection or threshold tuning.
