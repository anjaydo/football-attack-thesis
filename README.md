# Football Attack-v2 Thesis

Hệ thống possession-first cho video/FOOTPASS: tái dựng chuỗi kiểm soát bóng,
xác nhận terminal rồi mới tạo attack, sau đó mới cho phép xG/value evaluation.

> **Trạng thái hiện tại: `research_only`.** Contract, decoder, provenance,
> annotation workflow và evaluator Attack-v2 đã có; chưa có human ground truth
> hoàn chỉnh hoặc learned model vượt gate, nên không có claim “validated”.

## Luồng Attack-v2

FOOTPASS chỉ là action tức thời. Nó không được dùng trực tiếp như attack:

```text
video/actions/tracks -> temporal clustering -> possession-state decoder
  -> terminal-first assembly -> strict attack validation -> xG/value components
```

Một `AttackSequenceV2` chỉ tồn tại khi possession hoàn chỉnh kết thúc bằng
`shot_attempt`, `headed_attempt` hoặc `forced_goal_line_out`. Turnover, cross,
high progression, gap, throw-in hay generic header không tự tạo attack. Gap/cut
chỉ tạo censoring. Video forward-ball heuristic chỉ phát `shot_candidate`, không
phát shot/xG.

Đã triển khai:

- schema versioned cho observed action, possession, terminal, attack và value;
- stable IDs, field-level provenance và bốn source tier tách biệt;
- deterministic terminal-first decoder và hạ tầng TCN/semi-Markov 5 Hz;
- continuous human annotation cho 12 trận, validator gap/overlap, SHA-256 frozen
  test seal chỉ cho đúng game 0/36/50 và intra-rater report;
- Hungarian sequence metrics ở tIoU 0.25/0.5/0.75, bootstrap theo match và gates;
- action-spotting overlap/sampler/hard-negative/freeze infrastructure;
- tracking AP/IDF1/HOTA/team/ball/GS-HOTA-style evaluator;
- xG-v2 missingness semantics, nested grouped CV, disjoint calibration và gates;
- strict cache/content-tree/dataset/video/model/config/schema hash validation;
- schema quan sát cầu thủ/bóng theo từng frame;
- đọc metadata video qua OpenCV tùy chọn;
- xuất video có skeleton, player/ball boxes, track ID và tốc độ;
- lọc người ngoài pitch ROI, tùy chọn lọc bóng, và phân nhóm màu áo `team_a/team_b`;
- xuất JSONL chứa keypoints, boxes, speed và pitch coordinates;
- kiểm tra chất lượng JSONL: ball/player/tracking/keypoint coverage;
- đánh giá bbox theo ground truth SoccerNet: precision, recall, F1, IoU và center error;
- export tập YOLO một lớp từ SoccerNet, fine-tune model bóng riêng và benchmark
  confidence bằng quality gate có ground truth;
- phát hiện scene cut và reset lịch sử speed/team;
- suy luận baseline possession/carry/pass/turnover từ calibrated tracks;
- FOOTPASS adapter giữ bbox/replay, stable roster team và generic header semantics;
- frozen role benchmark store, grounded LLM tools và claim verification;
- Streamlit demo chỉ render strict attack hoặc possession/threat research-only;
- chọn cửa sổ thời gian trên FOOTPASS, fit homography động theo frame với holdout
  error gate, chạy YOLO đúng frame toàn trận và lưu event/xG artifacts;
- dữ liệu ví dụ và unit tests.

Legacy `Event/AttackSequence`, rule score và segmenter vẫn đọc được để tái lập
baseline, nhưng output luôn mang nhãn `PossessionCandidateV1/legacy_unvalidated`
và không được dùng cho headline metric hay categorical effectiveness.

## Chạy nhanh

Yêu cầu Python 3.10+.

```powershell
python -m pip install -e .
attack-analysis examples/attack_effective.json
attack-analysis examples/attack_ineffective.json
attack-analysis analyze-events examples/match_events.json
python -m unittest discover -s tests -v
```

Tải và phân tích sequence validation đầu tiên của SoccerNet GameState mà không
tải toàn bộ archive 11.17 GB:

```powershell
attack-analysis soccernet-download .\data\soccernet-gs --split valid --sequence first
attack-analysis soccernet-analyze .\data\soccernet-gs\gamestate-2024\valid\SNGS-033
```

Đây là sequence GSR khoảng 30 giây, không phải trận đấu 90 phút. Video full-match
của SoccerNet yêu cầu NDA/password nghiên cứu do chủ dataset cấp.

Để đọc metadata video thật:

```powershell
python -m pip install -e ".[video]"
attack-analysis video-info path/to/video.mp4
```

Để tạo video skeleton/tracking bằng YOLO26:

```powershell
python -m pip install -e ".[vision]"
attack-analysis annotate-video input.mp4 output.mp4 `
  --observations output.observations.jsonl `
  --roi examples/roi.example.json `
  --max-frames 300
attack-analysis quality-report output.observations.jsonl
attack-analysis infer-events output.observations.jsonl --output events.json
attack-analysis analyze-events events.json
```

Không dùng ball detector COCO để kết luận xG. Với SoccerNet, phải chạy benchmark
và chỉ chấp nhận model khi `accepted=true`:

```powershell
attack-analysis benchmark-ball output.observations.jsonl `
  data\soccernet-gs\gamestate-2024\valid\SNGS-033\Labels-GameState.json `
  --start-frame 575 `
  --output output.ball-benchmark.json
```

Gate mặc định yêu cầu ball F1 `>= 0.80`, precision và recall đều `>= 0.75` tại
IoU `0.5`. Baseline hiện tại chỉ đạt F1 `0.348` và bị từ chối đúng thiết kế.

Chạy kiểm tra môi trường và pipeline một lệnh:

```powershell
attack-analysis doctor
attack-analysis run-pipeline input.mp4 runs/demo `
  --roi examples/roi.example.json `
  --max-frames 300
```

Thêm `--calibration calibration.json` để pipeline tiếp tục từ vision sang event
candidates và Attack-v2 decoding. Nếu chưa có terminal evidence, output là
possession/threat và xG/effectiveness bị giữ lại.

Tab FOOTPASS hiển thị riêng strict Attack-v2 và toàn bộ possession, kèm source
tier/provenance. `FOOTPASS-assisted calibration` là oracle-assisted, không phải
video-only fusion. xG-v1 không còn được tự động gắn vào candidate; unknown giữ là
unknown.

Mặc định pipeline dùng `yolo26n.pt` + BoT-SORT để detect/track cầu thủ,
`yolo26n-pose.pt` để ghép skeleton theo IoU, và một tracker ByteTrack độc lập cho
lớp `sports ball`. Model weights được tải ở lần chạy đầu tiên. Dùng `--device 0`
khi có NVIDIA GPU.

Không có calibration, speed được báo bằng `px/s`. Với clip camera cố định và
file calibration bốn điểm, có thể xuất `km/h`:

```powershell
attack-analysis annotate-video input.mp4 output.mp4 `
  --calibration examples/calibration.example.json `
  --observations output.observations.jsonl
```

Lệnh cũ `attack-analysis file.json` vẫn được hỗ trợ và tương đương với
`attack-analysis evaluate file.json`.

Tài liệu:

- `demo.md`: kịch bản demo 8–10 phút, số liệu kỳ vọng và phương án dự phòng;
- `docs/external-calibration-data.md`: nguồn dữ liệu calibration/tracking bổ sung,
  license, provenance và thứ tự tích hợp;
- `docs/possession-qualification.md`: rubric tách possession thông thường khỏi
  attack trước khi đánh giá effectiveness;
- `docs/cli-guide.md`: toàn bộ CLI và thứ tự chạy thử từ đầu đến cuối;
- `docs/thesis-kickoff.md`: định hướng nghiên cứu và kế hoạch thực nghiệm;
- `docs/data-contracts.md`: hợp đồng dữ liệu giữa video, tracking, event và scoring.
- `docs/vision-pipeline.md`: kiến trúc YOLO, speed và giới hạn của video broadcast.
- `docs/implementation-status.md`: ma trận chức năng, mức kiểm thử và phần còn thiếu.
- `docs/attack-v2.md`: ontology, contracts, source tiers, gold split, evaluator và runbook v2.
- `docs/yolo-field-preview-analysis.md`: phép thử YOLO + field geometry trên SoccerNet.
- `docs/ball-accuracy-plan.md`: benchmark, dataset, lệnh train GPU và quality gate cho bóng.
- `docs/research-pipeline.md`: luồng FOOTPASS oracle/predicted, annotation, ML, LLM tools và demo.
