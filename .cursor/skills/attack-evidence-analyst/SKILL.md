---
name: attack-evidence-analyst
description: Evaluates football attacking sequences from validated EvidenceBundle data. Use when producing effective, promising, ineffective, or insufficient-evidence verdicts for an attack.
---

# Attack Evidence Analyst

## Required workflow

1. Read the supplied `EvidenceBundle`; do not inspect or infer facts from raw video.
2. If `downstream_ready=false`, return `insufficient_evidence` and list the failed quality gates.
3. Consider progression, penetration, shots, xG, action sequence, turnover and uncertainty together.
4. Treat xG as a probability estimate, not ground truth and not the sole definition of effectiveness.
5. Assign one label: `effective`, `promising`, `ineffective`, or `insufficient_evidence`.
6. Assign score 1–5 and confidence 0–1. Reduce confidence when uncertainty is high or evidence is sparse.
7. Every claim must cite one or more exact `evidence_id` values from the bundle.
8. Call `verify_claims` before returning the final verdict.

## Rubric

- `effective`: created a high-quality chance, scored, or clearly broke the defensive structure while retaining meaningful control.
- `promising`: progressed or created an advantage but did not produce sufficiently strong danger.
- `ineffective`: failed to progress, lost possession early, or created only weak danger.
- `insufficient_evidence`: tracking, calibration, event or source quality is not sufficient for a defensible conclusion.

Never calculate new xG values, invent missing events, or convert contextual web results into numeric evidence.
