---
name: football-claim-verifier
description: Verifies football-analysis claims against structured evidence and frozen benchmark IDs. Use before presenting an attack verdict, player identity, metric, percentile, or seasonal comparison.
---

# Football Claim Verifier

## Required workflow

1. Collect each proposed claim as `{text, evidence_ids}`.
2. Call `verify_claims` with the selected attack ID and all claims.
3. Remove or rewrite claims with missing or unsupported evidence IDs.
4. Numeric claims must match supplied structured values; do not recompute or round in a misleading way.
5. Player names require an identity source ID. Benchmark comparisons require benchmark IDs.
6. When evidence conflicts, report the conflict as a limitation and reduce confidence.
7. Return the final verdict only after `verify_claims` reports `valid=true`.

An unsupported but plausible statement is still unsupported and must not be shown as fact.
