# Football Attack Effectiveness Thesis

Prototype khởi đầu cho luận văn: từ video bóng đá, tái dựng trạng thái trận đấu,
tính các đặc trưng định lượng và trả lời một đợt tấn công có hiệu quả hay không.

## Phạm vi MVP

MVP hiện tại bắt đầu từ dữ liệu sự kiện có cấu trúc (tọa độ, đường chuyền, cú sút,
xG/xT nếu có). Đây là điểm nối giữa khối computer vision và khối đánh giá:

```text
Video -> YOLO pose + detection -> tracking -> calibration -> events + pitch coordinates
      -> metrics -> effectiveness model -> grounded explanation
```

Prototype đã có:

- schema cho một đợt tấn công;
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
- suy luận baseline possession, carry, pass và turnover từ calibrated tracks;
- tự động tách event stream thành các đợt tấn công;
- các chỉ số progression, final-third/box entry, shot, xG và xT;
- rule-based baseline cho điểm 0-100 và kết luận;
- explanation deterministic chỉ dựa trên dữ liệu đã tính;
- FOOTPASS oracle/predicted adapters, possession segmentation và attack feature contract;
- stratified annotation manifest cho nhãn ba mức, score và confidence;
- leakage-safe xG/effectiveness trainers với split theo match;
- frozen role benchmark store, grounded LLM tools và claim verification;
- Streamlit demo cho FOOTPASS, annotation và exploratory upload;
- dữ liệu ví dụ và unit tests.

Rule-based score chỉ là baseline kỹ thuật. Trong luận văn, nó sẽ được thay/thêm
bằng mô hình học có giám sát từ nhãn của chuyên gia bóng đá.

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

Thêm `--calibration calibration.json` để pipeline tiếp tục từ vision sang
event inference, attack segmentation và effectiveness scoring.

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

- `docs/cli-guide.md`: toàn bộ CLI và thứ tự chạy thử từ đầu đến cuối;
- `docs/thesis-kickoff.md`: định hướng nghiên cứu và kế hoạch thực nghiệm;
- `docs/data-contracts.md`: hợp đồng dữ liệu giữa video, tracking, event và scoring.
- `docs/vision-pipeline.md`: kiến trúc YOLO, speed và giới hạn của video broadcast.
- `docs/implementation-status.md`: ma trận chức năng, mức kiểm thử và phần còn thiếu.
- `docs/yolo-field-preview-analysis.md`: phép thử YOLO + field geometry trên SoccerNet.
- `docs/ball-accuracy-plan.md`: benchmark, dataset, lệnh train GPU và quality gate cho bóng.
- `docs/research-pipeline.md`: luồng FOOTPASS oracle/predicted, annotation, ML, LLM tools và demo.
