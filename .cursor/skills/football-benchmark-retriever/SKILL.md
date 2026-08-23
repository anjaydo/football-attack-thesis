---
name: football-benchmark-retriever
description: Retrieves frozen, sourced football player and tactical-role benchmarks. Use when comparing an attack or player with role, team, season, or per-90 reference distributions.
---

# Football Benchmark Retriever

## Required workflow

1. Prefer `get_role_benchmark` for the thesis core comparison.
2. Resolve a named player only with `resolve_player_identity(match_id, team, jersey)`.
3. If the identity registry has no exact match, state that identity is unresolved. Never infer a name from jersey number alone.
4. Use `lookup_player_snapshot` only after identity resolution.
5. Use `search_frozen_stats` for numeric comparisons. A valid record must include season, sample size, source URL and snapshot date.
6. Web search is contextual only. Do not cite a live web result as a frozen numeric benchmark.
7. Compare like with like: same metric definition, per-90 normalization, role and applicable season.

Return benchmark IDs with every comparison so the claim verifier can trace the result.
