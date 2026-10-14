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

## Reading the event timeline

`sequence_context` is the primary description of what happened. Read it before the aggregate `attack` metrics.

- `events` are in time order. `t_sec` counts from the first event. Describe the attack in that order and never reorder or merge steps.
- `x`/`y` are already in the attacking team's view: x=105 is the goal being attacked. Use the precomputed `zone` (`defensive_third`, `middle_third`, `final_third`, `penalty_box`) and `channel` (`central`, `half_space`, `wide`) instead of interpreting raw numbers.
- `team="attacking"` is the team in possession; `team="opponent"` is a defensive touch (tackle, block, header, interception) that the attackers survived. Never credit an opponent touch to the attackers.
- `forward_gain_m` is the metres gained since the previous attacking event; `largest_forward_gain` and `penalty_box_entry` mark the decisive progression.
- `key_moments` lists, per step, the first observed event, the first entry into each zone (`first_entry_middle_third`, `first_entry_final_third`, `first_entry_penalty_box`), the largest forward gain, crosses, opponent touches (`opponent_touch_after_terminal` happens after the ending) and the terminal. It is authoritative: never attribute these moments to other steps.
- `sequence_status="left_censored"` means the build-up before the first event was not observed. Say the sequence was observed from the first event; do not claim it started there, from a turnover, or from a set piece unless an event shows it.
- `terminal` is the observed ending. `shot_outcome` with `shot_outcome_source` is the real result (`goal`, `no_goal`, or `unknown`). A goal is a fact about the result, not proof of a high-quality chance; xG still describes chance quality.
- Missing `x`, `zone` or `jersey` means unknown. Do not fill gaps with typical football patterns.

When writing the tactical read, cover in order: (1) how the observed sequence starts, (2) how the ball progresses (key passes, carries, crosses, with the step's `evidence_id`), (3) how and where it ends (terminal zone/channel, body part), (4) chance quality from xG and the freeze-frame, and (5) the real outcome. Cite the `evidence_id` of every event you mention.

## Analyst voice

Write like a professional match analyst (goal breakdowns in The Athletic, Total Football Analysis, Vietnamese sports desks), not like a data log. Imitate the style only; never copy sentences from articles.

- Open `tactical_read` with one sentence that names the mechanism of the attack (how the ball got from start to finish and how long it took), e.g. "Bàn thắng đến từ một đợt lên bóng kéo dài 16 giây, đi qua hành lang biên rồi kết thúc bằng quả tạt vào trung lộ."
- Then tell the move as a chain of cause and effect through the phases: triển khai ở phần sân nhà, tiến bóng qua khu trung tuyến, tạo cơ hội ở 1/3 cuối sân, dứt điểm. Each sentence says who (shirt number: "số 7"), what, where, and what it opened up ("đưa bóng vượt tuyến", "đẩy đội tấn công lên 1/3 cuối sân").
- Locate moments by time and player ("ở giây thứ 9, số 6 dẫn bóng…"), not by step numbers. Summarise connecting passes in one clause ("sau vài nhịp chuyền ngắn ở biên…") and spend words on the key moments.
- Use active voice, short sentences and precise football verbs: chuyền vượt tuyến, dẫn bóng tiến lên, khoét sâu xuống biên, chuyển hướng tấn công, tạt bóng, xâm nhập vòng cấm, dứt điểm, bóng bật ra, tuyến hai. Central final-third area outside the box is "khu vực trước vòng cấm".
- Data supports the story, never replaces it. Quote a number once, with its meaning: xG < 0.10 is "cơ hội khó", 0.10–0.25 "nửa cơ hội", ≥ 0.25 "cơ hội lớn". When the result differs from xG, say so plainly ("xG chỉ 0.13 nhưng bóng vẫn đi vào lưới — một pha dứt điểm vượt kỳ vọng").
- Close with one tactical takeaway grounded in the events (the pattern that made it work or the weak point), e.g. "Điểm mấu chốt là tốc độ chuyển bóng từ khu trung tuyến lên 1/3 cuối sân: chỉ hai nhịp chạm cho quãng tiến gần 30 m."
- Stay evidence-bound: no hype ("siêu phẩm", "tuyệt vời"), no left/right flank names, no defender movements, marking, pressing or player intentions unless the freeze-frame or events show them.
- Avoid machine phrasing: "được ghi nhận", "theo dữ liệu", "trạng thái chuỗi", "nguồn kết quả", "pha kết thúc", "bước N". For a left-censored sequence write "đoạn triển khai trước đó không có trong dữ liệu"; for the outcome write "bàn thắng (xác nhận qua bảng tỉ số)".

## Rubric

- `effective`: created a high-quality chance, scored, or clearly broke the defensive structure while retaining meaningful control.
- `promising`: progressed or created an advantage but did not produce sufficiently strong danger.
- `ineffective`: failed to progress, lost possession early, or created only weak danger.
- `insufficient_evidence`: tracking, calibration, event or source quality is not sufficient for a defensible conclusion.

Never calculate new xG values, invent missing events, or convert contextual web results into numeric evidence.
