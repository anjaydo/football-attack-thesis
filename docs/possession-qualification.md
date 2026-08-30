# Phân biệt possession và đợt tấn công

> **Deprecated V1 ablation.** Rubric score trong tài liệu này không còn quyết
> định `is_attack`. Attack-v2 chỉ chấp nhận possession hoàn chỉnh kết thúc bằng
> `shot_attempt`, `headed_attempt` hoặc `forced_goal_line_out`; xem
> [attack-v2.md](attack-v2.md). V1 luôn được render là
> `PossessionCandidateV1/legacy_unvalidated` và không có effectiveness verdict.

Pipeline dùng ba tầng độc lập:

```text
event stream
→ possession segmentation
→ attack qualification
→ effectiveness evaluation (chỉ áp dụng khi is_attack=true)
```

Team switch, restart, shot, action gap và giới hạn 45 giây chỉ tạo **ranh giới
possession**. Chúng không đủ để gọi mọi segment là một đợt tấn công.

## Rubric baseline v1

`possession-attack-qualification-v1` tạo score 0–100 từ các tín hiệu:

- chance creation: shot và xG;
- zone penetration: final-third và penalty-box touches;
- peak progression: vị trí xa nhất so với điểm bắt đầu, không chỉ điểm cuối;
- attacking action: cross/header;
- directness: peak progression trên thời lượng;
- positive possession value khi xT có sẵn.

Phân loại hiện tại:

- `< 30`: `normal_possession`;
- `30–59.9`: `attack`;
- `>= 60`: `dangerous_attack`;
- dưới 50% event có tọa độ và không có shot: `insufficient_evidence`.

Đây là rule baseline có thể audit, **không phải xác suất ML và không phải ground
truth**. UI hiển thị score components, reasons, limitations và rubric version.
Effectiveness model/LLM phải abstain nếu `attack_qualified=false`.

## Ví dụ được báo cáo

`game_18_H2:sequence-0168` có 13 event trong 23,24 giây. Chuỗi bắt đầu ở
`x=6,1 m`, đạt `x=94,4 m`, peak progression `88,3 m`, có tám action ở 1/3 cuối,
một action chạm vùng cấm và kết thúc bằng cross. Rubric xếp chuỗi này là
`dangerous_attack` với 65/100, nhưng ghi limitation rằng chuỗi không tạo shot/xG.

## Đường nghiên cứu tiếp theo

1. Lấy mẫu possession theo match, cân bằng normal/attack, shot/no-shot, start zone
   và duration.
2. Gán nhãn `normal_possession/attack/dangerous_attack/uncertain` trước.
3. Chỉ gán `ineffective/promising/effective` cho các mẫu được xác nhận là attack.
4. Dùng rubric v1 làm baseline; train Logistic Regression và
   HistGradientBoosting trên cùng match split.
5. Chọn threshold bằng Macro-F1/balanced accuracy và báo cáo confusion matrix,
   calibration cùng bootstrap CI theo match.
6. Đo riêng error propagation giữa oracle events và predicted events.

Thiết kế này phù hợp với cách xT xem possession là khoảng cùng đội kiểm soát bóng
và chỉ định giá các action di chuyển bóng giữ quyền kiểm soát. Tham khảo:
<https://github.com/ML-KULeuven/socceraction/blob/master/docs/documentation/valuing_actions/xT.rst>.
